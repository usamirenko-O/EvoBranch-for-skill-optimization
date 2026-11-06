"""Shared resource setup, initial skills, and result handling for OURS runners."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from integration.benchmarks.manifest import SplitManifest
from integration.experiment.cost import PhaseCost
from ours import history as _history
from ours.types import CandidateState, Origin
from ours.workspace import assert_fresh_dir


class LeakageError(RuntimeError):
    """A task from a split reached a phase that must not see it."""


def assert_no_deployment_ids(manifest: SplitManifest, optimization_ids) -> None:
    deployment = set(manifest.task_ids("deployment"))
    leaked = sorted(set(map(str, optimization_ids)) & deployment)
    if leaked:
        raise LeakageError(
            f"{len(leaked)} deployment task id(s) reached the optimization "
            f"phase: {leaked[:5]}"
        )


def harness_options(base_config, adapter) -> dict:
    """The run's harness options, with ALFWorld's data root resolved exactly as
    the native launcher resolves it (`integration.experiment.execute._context`)."""
    options = dict(base_config.harness_options)
    if adapter.name == "alfworld":
        from integration.benchmarks.alfworld import alfworld_data_root

        options.setdefault("alfworld_data", alfworld_data_root())
    return options


def _public(metadata: dict) -> dict:
    """Metadata minus underscore-prefixed working keys (e.g. the native text
    cache the controller hands the operator), which are not artifact material."""
    return {k: v for k, v in (metadata or {}).items() if not str(k).startswith("_")}


def _routing_plan(pool) -> list:
    """The round's deterministic routing plan (history-aware operator family),
    read off the operator report an offspring carries. Empty for the base /
    placeholder operators, which have none."""
    for c in pool.operator_candidates():
        report = c.operator_provenance.get("operator_report") or {}
        plan = report.get("routing_plan")
        if plan is not None:
            return plan
    return []


def round_summary(r) -> dict:
    """One OURS round, as data. Every claim the Method makes is readable here."""
    pool = r.pool
    competitive = pool.competitive()
    return {
        "round_id": r.round_id,
        "stage_order": [s.value for s in r.trace.distinct_stage_order],
        "evidence": [
            {"evidence_id": e.evidence_id, "parent_id": e.parent_id,
             "task_ids": list(e.task_ids), **_public(e.metadata)}
            for e in r.evidence
        ],
        "subsets": [
            {"subset_id": s.subset_id, "parent_id": s.parent_id,
             "partitioner": s.partitioner, "task_ids": list(s.task_ids), **_public(s.metadata)}
            for s in r.subsets
        ],
        "partitions": [
            {"evidence_id": e.detail.get("evidence_id"), "parent_id": e.detail.get("parent_id"),
             "partitioner": e.detail.get("partitioner"),
             "subset_ids": e.detail.get("subset_ids"),
             "subset_task_ids": e.detail.get("subset_task_ids"),
             "report": e.detail.get("report")}
            for e in r.trace.events if e.stage.value == "partition"
        ],
        "proposal_calls": [
            {"proposal_id": e.detail.get("proposal_id"),
             "parent_id": e.detail.get("parent_id"),
             "subset_id": e.detail.get("subset_id"),
             "status": e.detail.get("status"),
             "produced_proposal": e.detail.get("produced_proposal"),
             # history-aware upgrade: the attempt-aware context this native
             # proposal received (absent when the history layer is off).
             "history_context_chars": e.detail.get("history_context_chars")}
            for e in r.trace.events if e.stage.value == "propose"
        ],
        "raw_proposals": [
            {"proposal_id": q.proposal_id, "parent_id": q.parent_id,
             "evidence_subset_id": q.evidence_subset_id, "skill_hash": q.skill_hash,
             "role": "operator_material", "is_competitive_candidate": False,
             "has_score_field": hasattr(q, "score"),
             "native_step": q.provenance.get("native_step"),
             "changed_parent_body": q.provenance.get("changed_parent_body")}
            for q in r.raw_proposals
        ],
        "operator_candidates": [
            {"candidate_id": c.candidate_id, "parent_id": c.parent_id,
             "skill_hash": c.skill_hash,
             "source_proposal_ids": c.operator_provenance.get("source_proposal_ids"),
             "source_parent_ids": c.operator_provenance.get("source_parent_ids"),
             "operator": c.operator_provenance.get("operator"),
             "is_placeholder": c.operator_provenance.get("is_placeholder"),
             # history-aware upgrade: which evolution operator produced this
             # candidate and the issues it targeted (absent for placeholders).
             "operator_type": c.operator_provenance.get("operator_type"),
             "target_issue_ids": c.operator_provenance.get("target_issue_ids")}
            for c in pool.operator_candidates()
        ],
        "routing_plan": _routing_plan(pool),
        "competitive_pool": {
            "formula": "parents + operator_candidates",
            "member_ids": [c.candidate_id for c in competitive],
            "member_origins": [c.origin.value for c in competitive],
            "n_raw_proposals_excluded": len(r.raw_proposals),
        },
        "scoring": {
            "validation_context": r.scoring.validation_context,
            "n_members": r.scoring.n_members,
            "n_evaluated": r.scoring.n_evaluated,
            "n_reused": r.scoring.n_reused,
            "n_cache_hits": r.scoring.n_cache_hits,
            "n_invalidated": r.scoring.n_invalidated,
            "evaluated_ids": list(r.scoring.evaluated_ids),
            "reused_ids": list(r.scoring.reused_ids),
        },
        "selection": {
            "policy": "global_top_m",
            "m": r.selection.m,
            "ranked_ids": list(r.selection.ranked_ids),
            "survivor_ids": list(r.selection.survivor_ids),
            "survivor_origins": list(r.selection.survivor_origins),
            "scores": {c.candidate_id: float(c.score.value) for c in r.selection.ranked},
            "tie_at_cutoff": r.selection.tie_at_cutoff,
            "tie_policy": r.selection.tie_policy,
        },
        "next_beam": [c.candidate_id for c in r.next_beam],
        "component_cost": dict(getattr(r, "component_cost", {}) or {}),


        "native_epoch": getattr(r, "native_epoch", None),
        "wall_time_s": r.wall_time_s,
    }


# ───────────────────────────── run resources ─────────────────────────────────


def run_resources(config, accumulator):
    """The :class:`~ours.resources.RunResources` every OURS driver binds to its
    design components: the run's optimizer-side alias and call contract (the
    same the native optimizer's optimizer-side roles use), the framework's
    embedding alias, and the run's OPTIMIZATION accumulator."""
    from ours.experiment.config import OURS_EMBEDDING_MODEL_KEY
    from ours.resources import RunResources

    base = config.base
    return RunResources(
        optimizer_model=base.optimizer_model,
        proxy_base_url=base.proxy_base_url,
        api_key=base.api_key,
        temperature=base.temperature,
        max_tokens=base.optimizer_max_tokens,
        timeout_seconds=base.timeout_seconds,
        max_retries=base.max_retries,
        embedding_model=(base.extra.get(OURS_EMBEDDING_MODEL_KEY)
                         or base.extra.get("skillgen_embedding_model") or None),
        accumulator=accumulator,
        seed=config.seed,
    )


def final_beam_summary(result) -> list:
    return [
        {"candidate_id": c.candidate_id, "origin": c.origin.value,
         "validation_score": float(c.score.value) if c.score else None,
         "first_scored_in_round": c.first_scored_in_round,
         "lineage": list(c.lineage),
         # history-aware upgrade: each Beam_T survivor's persistent branch
         # history (issues, attempts, restarts). Empty when the layer is off.
         "history": _history.peek(c.metadata)}
        for c in result.final_beam
    ]


# ───────────────────────────── Beam_0 ────────────────────────────────────────


def native_initial_beam(optimizer: str, benchmark: str) -> list[CandidateState]:
    """Create an unscored beam containing the optimizer's native starting skill."""
    from integration.experiment.runners.skills import NATIVE_START_CONDITION, initial_skill_text

    return [CandidateState(
        candidate_id="r000-par-000",
        optimizer_name=optimizer,
        round_id=0,
        skill=initial_skill_text(optimizer, benchmark),
        origin=Origin.PARENT,
        metadata={
            "native_origin": "S_0",
            "start_condition": NATIVE_START_CONDITION[optimizer],
        },
    )]


# ─────────────────────── Beam_T -> deployment ───────────────────────────────


def finalize_search(design, controller, result, *, run_dir: Path):
    """Run the design's deployment rule's `finalize` on Beam_T, INSIDE the
    optimization phase: whatever it generates is produced by the run's bound
    operator and scored on the same fixed VAL through the run's scorer, so it
    is charged to optimization. Records the decision on ``result``."""
    rule = design.build_deployment_rule()
    round_id = result.n_rounds  # the round after the last search round
    ws = controller.workspace.round_workspace(round_id)

    def _allocate(tag: str) -> str:
        return assert_fresh_dir(os.path.join(ws.step_dir, "final", tag), f"final {tag}")

    selection = rule.finalize(result.final_beam, controller=controller, round_id=round_id,
                              allocate_dir=_allocate)
    result.final_selection = selection
    result.metadata["final_selection"] = {"rule": rule.name, "rule_is_placeholder": bool(rule.is_placeholder),
                                          **selection.to_dict()}
    (run_dir / "final_selection.json").write_text(
        json.dumps(result.metadata["final_selection"], indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return selection


def sum_phase_costs(costs) -> PhaseCost:
    """Exact sum of several phase costs. A token count stays None if ANY part
    lacks it -- a total is never estimated."""
    costs = list(costs)
    if len(costs) == 1:
        return costs[0]

    def _sum(attr):
        values = [getattr(c, attr) for c in costs]
        return None if any(v is None for v in values) else sum(values)

    sources = {c.token_source for c in costs if c.token_source != "unavailable"}
    source = "unavailable" if not sources else (sources.pop() if len(sources) == 1 else "mixed")
    per_role: dict = {}
    for c in costs:
        for role, bucket in c.per_role.items():
            acc = per_role.setdefault(role, {k: 0 for k in bucket})
            for k, v in bucket.items():
                acc[k] = acc.get(k, 0) + (v or 0)
    return PhaseCost.build(
        llm_calls=sum(c.llm_calls for c in costs),
        input_tokens=_sum("input_tokens"),
        output_tokens=_sum("output_tokens"),
        cached_tokens=_sum("cached_tokens"),
        token_source=source,
        per_role=per_role,
    )


@dataclass
class FinalDeployment:
    rule_name: str
    rule_is_placeholder: bool
    members: list = field(default_factory=list)      # [(CandidateState, DeploymentResult)]
    aggregate: object = None                          # AggregateResult
    cost: PhaseCost | None = None
    wall_time_s: float = 0.0

    @property
    def primary(self):
        return self.members[0][1]

    def performance(self) -> dict:
        """Fields for the record's `performance` section."""
        first = self.primary.result
        if len(self.members) == 1:
            return {
                "metric_name": first.metric_name,
                "metric_value": first.metric_value,
                "n_tasks": first.n_tasks,
                "scorer_ref": first.scorer_ref,
                "secondary_metrics": dict(first.secondary_metrics),
            }
        return {
            "metric_name": first.metric_name,
            "metric_value": self.aggregate.metric_value,
            "n_tasks": self.aggregate.n_tasks,
            "scorer_ref": first.scorer_ref,
            "secondary_metrics": {
                "deployment_aggregation": self.aggregate.method,
                "per_member": {c.candidate_id: r.result.metric_value for c, r in self.members},
            },
        }

    def to_dict(self) -> dict:
        return {
            "rule": self.rule_name,
            "rule_is_placeholder": self.rule_is_placeholder,
            "deployed_candidate_ids": [c.candidate_id for c, _ in self.members],
            "aggregate": {
                "metric_value": self.aggregate.metric_value,
                "n_tasks": self.aggregate.n_tasks,
                "method": self.aggregate.method,
                "details": self.aggregate.details,
            },
            "members": [
                {"candidate_id": c.candidate_id,
                 "validation_score": float(c.score.value) if c.score else None,
                 "metric_value": r.result.metric_value, "n_tasks": r.result.n_tasks,
                 "llm_calls": r.cost.llm_calls}
                for c, r in self.members
            ],
        }


def deploy_final_beam(*, design, final_beam, adapter, manifest, run_dir: Path,
                      solver_config, harness_options: dict, deployment_workers: int) -> FinalDeployment:
    """Run the design's deployment rule on Beam_T, for any optimizer.

    The ONLY place deployment ids are read. Each selected member is executed
    on the DEPLOYMENT split through the shared Direct harness and scored by the
    benchmark's official scorer (`integration.experiment.deployment`); the rule
    then aggregates, and the deployment cost is the exact sum over members.
    """
    import time

    from integration.experiment.deployment import run_deployment
    from integration.harness.registry import build_harness, max_parallel_tasks
    from ours.deployment_rule import MemberDeployment

    rule = design.build_deployment_rule()
    selected = rule.select(tuple(final_beam))
    if not selected:
        raise RuntimeError(f"deployment rule {rule.name!r} selected nothing")
    single = len(selected) == 1
    t0 = time.time()
    members = []
    for candidate in selected:
        suffix = "" if single else f"__{candidate.candidate_id}"
        harness = build_harness(adapter.name, work_dir=run_dir / f"deployment_harness{suffix}",
                                options=harness_options)
        try:
            result = run_deployment(
                adapter=adapter,
                harness=harness,
                task_ids=manifest.task_ids("deployment"),
                skill_text=candidate.skill,
                solver_config=solver_config,
                max_workers=min(int(deployment_workers), max_parallel_tasks(adapter.name)),
                transcripts_path=run_dir / f"deployment_transcripts{suffix}.jsonl",
            )
        finally:
            harness.close()
        members.append((candidate, result))

    aggregate = rule.aggregate([
        MemberDeployment(
            candidate_id=c.candidate_id,
            metric_name=r.result.metric_name,
            metric_value=r.result.metric_value,
            n_tasks=r.result.n_tasks,
            validation_score=float(c.score.value) if c.score else None,
            secondary_metrics=dict(r.result.secondary_metrics),
        )
        for c, r in members
    ])
    return FinalDeployment(
        rule_name=rule.name,
        rule_is_placeholder=bool(rule.is_placeholder),
        members=members,
        aggregate=aggregate,
        cost=sum_phase_costs(r.cost for _, r in members),
        wall_time_s=time.time() - t0,
    )


def design_placeholders(config) -> dict:
    """The placeholder declarations every OURS record carries: one entry per
    component slot and per hyperparameter whose scientific choice is still a
    placeholder FOR THIS RUN'S OPTIMIZER (a frozen main-experiment default,
    e.g. SSO's T, is not a placeholder; see `OursRunConfig.hyperparameter_status`)."""
    report = config.design_report()
    hyper = report.pop("hyperparameters")
    text = {
        "partitioner": "clustering / evidence partition: algorithm, representation, "
                       "similarity, K selection",
        "operator": "Operator: prompt, model, merge/dedup/diversity/conflict policy, "
                    "number of candidates",
        "tie_break": "order of equal-score candidates at the Top-M cutoff",
        "deployment_rule": "Beam_T -> deployed skill(s) and reported metric",
        "rounds": "T (outer rounds)",
        "beam_m": "M (beam width)",
        "k_subsets": "K (evidence subsets per Parent)",
    }
    out = {
        slot: f"{entry['name']} -- PLACEHOLDER; OPEN: {text[slot]}"
        for slot, entry in report.items() if entry["is_placeholder"]
    }
    out.update({
        name: f"{entry['value']} -- PLACEHOLDER; OPEN: {text[name]}"
        for name, entry in hyper.items() if entry["is_placeholder"]
    })
    return out


__all__ = [
    "FinalDeployment", "LeakageError", "assert_no_deployment_ids", "deploy_final_beam",
    "design_placeholders", "final_beam_summary", "finalize_search", "harness_options",
    "native_initial_beam", "round_summary", "run_resources", "sum_phase_costs",
]
