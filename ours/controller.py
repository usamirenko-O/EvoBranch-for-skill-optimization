"""Run one search round: collect evidence, update history, propose, evolve, and select."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from ours import history
from ours.adapters.base import EvaluatorAdapter, OptimizerAdapter
from ours.beam import BeamManager, BeamSelection
from ours.evaluation import PoolScorer, ScoringReport
from ours.operator import SkillOperator
from ours.partition import EvidencePartitioner
from ours.pool import CandidatePool, RawProposalStore
from ours.types import (
    CandidateState,
    Evidence,
    EvidenceSubset,
    Origin,
    RawProposal,
    ROUND_STAGE_ORDER,
    Stage,
)
from ours.workspace import RunWorkspace, assert_fresh_dir


@dataclass(frozen=True)
class TraceEvent:
    stage: Stage
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class RoundTrace:
    """Ordered record of what the controller did, in the order it did it."""

    events: list[TraceEvent] = field(default_factory=list)

    def record(self, stage: Stage, **detail: Any) -> None:
        self.events.append(TraceEvent(stage=stage, detail=detail))

    @property
    def stages(self) -> tuple[Stage, ...]:
        return tuple(e.stage for e in self.events)

    @property
    def distinct_stage_order(self) -> tuple[Stage, ...]:
        seen: list[Stage] = []
        for stage in self.stages:
            if stage not in seen:
                seen.append(stage)
        return tuple(seen)

    def index_of(self, stage: Stage) -> int:
        return self.stages.index(stage)

    def last_index_of(self, stage: Stage) -> int:
        return len(self.stages) - 1 - self.stages[::-1].index(stage)


@dataclass
class RoundResult:
    round_id: int
    pool: CandidatePool
    evidence: tuple[Evidence, ...]
    subsets: tuple[EvidenceSubset, ...]
    raw_proposals: tuple[RawProposal, ...]
    scoring: ScoringReport
    selection: BeamSelection
    trace: RoundTrace
    next_beam: tuple[CandidateState, ...]
    wall_time_s: float = 0.0
    #: What the framework's OWN components (partitioner / operator / tie-break)
    #: spent this round through the run's RunResources, per role. Already
    #: metered into the run's optimization accumulator; kept here for audit.
    component_cost: dict = field(default_factory=dict)
    #: The native optimizer's epoch-level update of this round's survivors
    #: (adapter.finish_round), or None (last round / no such mechanism).
    native_epoch: dict | None = None


class SearchController:
    """Runs one OURS round over a beam of parents."""

    def __init__(
        self,
        *,
        adapter: OptimizerAdapter,
        partitioner: EvidencePartitioner,
        operator: SkillOperator,
        evaluator: EvaluatorAdapter,
        beam_manager: BeamManager,
        workspace: RunWorkspace,
        k_subsets: int = 3,
        seed: int | None = 42,
        resources=None,
        tracker=None,
        attempt_aware_proposal: bool = True,
    ) -> None:
        self.adapter = adapter
        self.partitioner = partitioner
        self.operator = operator
        self.evaluator = evaluator
        self.beam_manager = beam_manager
        self.workspace = workspace
        self.k_subsets = k_subsets
        self.seed = seed


        self.tracker = tracker


        self.attempt_aware_proposal = bool(attempt_aware_proposal)
        #: The run's RunResources (may be None offline). Components were bound
        #: to it by the design; the controller only reads its usage snapshots
        #: so each stage's component spend is visible in the trace.
        self.resources = resources
        self.scorer = PoolScorer(evaluator)

    def _usage(self) -> dict:
        return self.resources.usage_snapshot() if self.resources is not None else {}

    def _spent_since(self, before: dict) -> dict:
        if self.resources is None:
            return {}
        return self.resources.usage_delta(before, self.resources.usage_snapshot())

    # ── ids ──────────────────────────────────────────────────────────────

    @staticmethod
    def _cid(round_id: int, origin: Origin, index: int) -> str:
        tag = {Origin.PARENT: "par", Origin.OPERATOR: "op"}[origin]
        return f"r{round_id:03d}-{tag}-{index:03d}"

    @staticmethod
    def _pid(round_id: int, index: int) -> str:
        """Id of a RAW PROPOSAL. Deliberately not in the candidate id space:
        a proposal id is not a candidate id, and the pool checks that no id
        crosses over."""
        return f"r{round_id:03d}-raw-{index:03d}"

    # ── the round ────────────────────────────────────────────────────────

    def run_round(
        self,
        parents: Sequence[CandidateState],
        tasks: Sequence[Any],
        *,
        round_id: int,
        final_round: bool = False,
    ) -> RoundResult:
        if not parents:
            raise ValueError("run_round needs at least one parent")
        t0 = time.time()
        ws = self.workspace.round_workspace(round_id)
        trace = RoundTrace()
        self.adapter.begin_round(round_id)

        beam_parents = [p.carried_forward(round_id) for p in parents]
        parent_by_id = {p.candidate_id: p for p in beam_parents}
        component_cost: dict[str, dict] = {}

        # ① collect evidence, per parent ─────────────────────────────────
        evidence: list[Evidence] = []
        for parent in beam_parents:
            ev = self.adapter.collect_evidence(
                parent,
                tasks,
                out_dir=ws.evidence_dir(os.path.join("evidence", parent.candidate_id)),
                seed=self.seed,
            )
            evidence.append(ev)
            trace.record(
                Stage.COLLECT_EVIDENCE,
                parent_id=parent.candidate_id,
                evidence_id=ev.evidence_id,
                n_tasks=len(ev.task_ids),
            )

        # The per-task feature view of every parent's evidence, computed once
        # and reused by the tracker and the partitioner. The native text of each
        # record is also stashed for the operator (underscore key: excluded from
        # the artifact's evidence summary).
        features_by_parent: dict[str, dict] = {}
        for ev in evidence:
            features = self.adapter.evidence_features(ev)
            ev.metadata["_native_text_by_task"] = {
                tid: view.get("native_text") for tid, view in features.items()
                if view.get("native_text") is not None
            }
            features_by_parent[ev.parent_id] = features


        if self.tracker is not None:
            usage_before = self._usage()
            for ev in evidence:
                parent = parent_by_id[ev.parent_id]
                prior_issues = history.issues(parent.metadata.get("history"))
                issues, tracker_report = self.tracker.update(
                    branch_id=parent.candidate_id,
                    prior_issues=prior_issues,
                    evidence_features=features_by_parent[ev.parent_id],
                    round_id=round_id,
                )
                history.set_issues(history.get(parent.metadata), issues)
                # inject the issue view into this parent's partitioner features
                task_features = history.task_issue_features(parent.metadata.get("history"))
                for tid, feat in task_features.items():
                    if tid in features_by_parent[ev.parent_id]:
                        features_by_parent[ev.parent_id][tid].update(feat)
                trace.record(
                    Stage.COLLECT_EVIDENCE,
                    parent_id=parent.candidate_id,
                    evidence_id=ev.evidence_id,
                    tracker=self.tracker.name,
                    issue_tracker={k: tracker_report.get(k) for k in (
                        "fallback", "llm_calls", "n_issues", "n_new",
                        "n_recurring", "n_resolved")},
                    issue_ids=history.issue_ids(parent.metadata.get("history")),
                )
            component_cost["issue_tracker"] = self._spent_since(usage_before)

        # ② partition into E1..EK ────────────────────────────────────────
        subsets: list[EvidenceSubset] = []
        usage_before = self._usage()
        for ev in evidence:
            features = features_by_parent[ev.parent_id]
            parts = self.partitioner.partition(
                ev, features, k=self.k_subsets,
                materialise=lambda subset_id, task_ids, partitioner, _ev=ev: (
                    self.adapter.slice_evidence(
                        _ev, subset_id, task_ids, partitioner=partitioner,
                    )
                ),
            )
            subsets.extend(parts)
            trace.record(
                Stage.PARTITION,
                evidence_id=ev.evidence_id,
                parent_id=ev.parent_id,
                partitioner=self.partitioner.name,
                is_placeholder=self.partitioner.is_placeholder,
                subset_ids=[s.subset_id for s in parts],
                subset_task_ids=[list(s.task_ids) for s in parts],
                report=self.partitioner.report(),
            )
        component_cost["partition"] = self._spent_since(usage_before)

        # ③ native proposal per (parent, subset) -> RAW MATERIAL ────────
        #    Every raw proposal is kept, and none of them is a candidate: they
        #    go to the operator's material store, not to the pool.
        material = RawProposalStore(round_id=round_id)
        pool = CandidatePool(round_id=round_id, proposal_material=material)
        for index, subset in enumerate(subsets):
            parent = parent_by_id[subset.parent_id]
            proposal_id = self._pid(round_id, index)
            out_dir = ws.candidate_root(proposal_id)


            history_context = (
                history.render_for_proposer(parent.metadata.get("history"), subset.task_ids)
                if (self.tracker is not None and self.attempt_aware_proposal) else ""
            )
            result = self.adapter.propose(
                parent, subset, out_dir=out_dir, round_id=round_id,
                history_context=history_context,
            )
            propose_detail = dict(
                proposal_id=proposal_id,
                parent_id=parent.candidate_id,
                subset_id=subset.subset_id,
                status=result.status,
                produced_proposal=result.produced_proposal,
                role="operator_material",
                is_competitive_candidate=False,
            )


            if self.tracker is not None:
                propose_detail["history_context_chars"] = len(history_context)
                propose_detail["attempt_aware_proposal"] = self.attempt_aware_proposal
            trace.record(Stage.PROPOSE, **propose_detail)
            if not result.produced_proposal:
                # The native mechanism declined to produce material for this
                # subset. Not a rejection -- there is simply nothing to add.
                continue
            material.add(RawProposal(
                proposal_id=proposal_id,
                optimizer_name=self.adapter.name,
                round_id=round_id,
                skill=result.skill,
                parent_id=parent.candidate_id,
                evidence_subset_id=subset.subset_id,
                provenance=dict(result.provenance),
                workspace_dir=result.workspace_dir or out_dir,
                cost=result.cost,
                metadata={"partitioner": subset.partitioner},
            ))


            if self.tracker is not None:
                target = [i["issue_id"] for i in history.relevant_issues(
                    parent.metadata.get("history"), subset.task_ids)]
                history.record_attempt(
                    history.get(parent.metadata), round=round_id,
                    kind=history.KIND_RAW_PROPOSAL, target_issue_ids=target,
                    summary=f"native proposal from subset {subset.subset_id}",
                    source_id=proposal_id,
                )

        # ④ operator: ALL parents + ALL raw proposals + ALL evidence ─────
        #    Opening the stage freezes the material store and hands back
        #    everything in it, so the operator provably sees the whole set.
        all_proposals = pool.open_operator_stage()

        def _allocate_operator_dir(tag: str) -> str:
            return assert_fresh_dir(
                os.path.join(ws.step_dir, "operator", tag), f"operator {tag}",
            )

        usage_before = self._usage()
        operator_out = self.operator.expand(
            parents=tuple(beam_parents),
            proposals=all_proposals,
            evidence=tuple(evidence),
            round_id=round_id,
            allocate_dir=_allocate_operator_dir,
        )
        component_cost["operator"] = self._spent_since(usage_before)
        # The operator's outputs are the round's ONLY new challengers.
        for index, op in enumerate(operator_out):
            candidate_id = self._cid(round_id, Origin.OPERATOR, index)
            source_parent_ids = tuple(op.source_parent_ids)
            source_proposal_ids = tuple(op.source_proposal_ids)
            anchor = (
                parent_by_id.get(source_parent_ids[0], beam_parents[0])
                if source_parent_ids else beam_parents[0]
            )


            metadata: dict = {}
            if self.tracker is not None:
                source_histories = [
                    parent_by_id[pid].metadata.get("history")
                    for pid in source_parent_ids if pid in parent_by_id
                ] or [anchor.metadata.get("history")]
                inherited = history.inherit(source_histories)
                op_type = op.provenance.get("operator_type") or self.operator.name
                history.record_attempt(
                    inherited, round=round_id, kind=op_type,
                    target_issue_ids=[str(t) for t in (op.provenance.get("target_issue_ids") or [])],
                    summary=str(op.provenance.get("rationale") or ""),
                    source_id=candidate_id,
                )
                if op_type == history.KIND_REGENERATION:
                    inherited["restarts"] = int(inherited.get("restarts", 0)) + 1
                metadata = {"history": inherited}


            op_skill, native_meta = self.adapter.adopt_native_state(op.skill, anchor)
            metadata.update(native_meta)
            pool.add_operator_candidate(CandidateState(
                candidate_id=candidate_id,
                optimizer_name=self.adapter.name,
                round_id=round_id,
                skill=op_skill,
                origin=Origin.OPERATOR,
                parent_id=anchor.candidate_id,
                operator_provenance={
                    "operator": self.operator.name,
                    "is_placeholder": self.operator.is_placeholder,
                    "source_proposal_ids": list(source_proposal_ids),
                    "source_parent_ids": list(source_parent_ids),
                    **dict(op.provenance),
                },
                workspace_dir=ws.candidate_root(candidate_id),
                cost=op.cost,
                lineage=anchor.lineage_including_self,
                metadata=metadata,
            ))
        trace.record(
            Stage.OPERATOR,
            operator=self.operator.name,
            is_placeholder=self.operator.is_placeholder,
            n_parents_seen=len(beam_parents),
            parent_ids_seen=[p.candidate_id for p in beam_parents],
            n_proposals_seen=len(all_proposals),
            proposal_ids_seen=[q.proposal_id for q in all_proposals],
            n_evidence_seen=len(evidence),
            evidence_ids_seen=[e.evidence_id for e in evidence],
            n_added=len(operator_out),
            operator_candidate_ids=[
                c.candidate_id for c in pool.operator_candidates()
            ],
            component_usage=component_cost["operator"],
        )

        # ⑤ the COMPETITIVE pool: C_t = Beam_t ∪ O_t ─────────────────────
        for parent in beam_parents:
            parent.workspace_dir = ws.candidate_root(parent.candidate_id)
            pool.add_parent(parent)
        pool.seal()   # also re-checks that no raw proposal leaked in
        competitive = pool.competitive()
        trace.record(
            Stage.POOL_COMPLETE,
            n_members=len(competitive),
            n_parents=len(pool.parents()),
            n_operator=len(pool.operator_candidates()),
            member_ids=[c.candidate_id for c in competitive],
            member_origins=[c.origin.value for c in competitive],
            n_raw_proposals_excluded=len(all_proposals),
            raw_proposal_ids_excluded=[q.proposal_id for q in all_proposals],
            raw_proposals_in_pool=[
                q.proposal_id for q in all_proposals
                if q.proposal_id in {c.candidate_id for c in competitive}
            ],
            competitive_pool_formula="parents + operator_outputs",
        )

        # ⑥ validation scores: reuse parents', measure the new ones ──────
        def _eval_dir(candidate: CandidateState) -> str:
            base = candidate.workspace_dir or ws.candidate_root(candidate.candidate_id)
            return assert_fresh_dir(
                os.path.join(base, "eval"), f"candidate {candidate.candidate_id} eval",
            )

        scoring = self.scorer.score_pool(
            pool, allocate_dir=_eval_dir, round_id=round_id,
        )
        trace.record(
            Stage.SCORE,
            evaluator=self.evaluator.name,
            validation_context=scoring.validation_context,
            n_members=scoring.n_members,
            n_evaluated=scoring.n_evaluated,
            n_reused=scoring.n_reused,
            n_cache_hits=scoring.n_cache_hits,
            n_invalidated=scoring.n_invalidated,
            evaluated_ids=list(scoring.evaluated_ids),
            reused_ids=list(scoring.reused_ids),
            scored_ids=list(scoring.scored_ids),
        )

        # ⑦ ONE global Top-M over C_t -> Beam_{t+1} ──────────────────────
        usage_before = self._usage()
        selection = self.beam_manager.select(pool)
        component_cost["select"] = self._spent_since(usage_before)


        native_epoch = None
        if not final_round:
            native_epoch = self.adapter.finish_round(
                selection.survivors, parents_by_id=parent_by_id, round_id=round_id,
                out_dir=os.path.join(ws.step_dir, "native_epoch"),
            )
        next_beam = tuple(
            c.carried_forward(round_id + 1) for c in selection.survivors
        )
        trace.record(
            Stage.SELECT,
            m=self.beam_manager.m,
            selection="global_top_m",
            pool_size=selection.pool_size,
            selection_input_ids=list(selection.ranked_ids),
            survivor_ids=list(selection.survivor_ids),
            survivor_origins=list(selection.survivor_origins),
            tie_at_cutoff=selection.tie_at_cutoff,
            tie_policy=selection.tie_policy,
            next_beam_scores={
                c.candidate_id: float(c.score.value) for c in next_beam
            },
        )

        self.adapter.end_round(round_id)
        return RoundResult(
            round_id=round_id,
            pool=pool,
            evidence=tuple(evidence),
            subsets=tuple(subsets),
            raw_proposals=all_proposals,
            scoring=scoring,
            selection=selection,
            trace=trace,
            next_beam=next_beam,
            wall_time_s=round(time.time() - t0, 3),
            component_cost=component_cost,
            native_epoch=native_epoch,
        )


def authoritative_stage_order() -> tuple[Stage, ...]:
    """The one true round order, for tests and documentation."""
    return ROUND_STAGE_ORDER
