"""Adapt SkillGen induction, verification, and refinement to OURS."""
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from typing import Any, Sequence

from ours.adapters.base import OptimizerAdapter, ProposalResult
from ours.adapters.skillgen_native import (
    DIAGNOSTIC_BUCKETS,
    ContrastiveEvidence,
    NoSkillBaseline,
    SkillGenNativeConfig,
    SkillGenNativeError,
    candidate_from_skill,
    collect_contrastive_evidence,
    collect_no_skill_baseline,
    generate_from_no_skill_trajectories,
    induce,
    refine_once,
    slice_contrastive_evidence,
)
from ours.types import CandidateState, CostAccounting, Evidence, EvidenceSubset

#: Which train tasks the OURS partitioner is allowed to cluster over.
#:
#: SCIENTIFIC TODO -- this is an evidence-representation choice, not a settled
#: one. ``diagnostic_cases`` (the default) offers exactly the units SkillGen's
#: own refinement consumes: the repair / regression / still-failing cases. A
#: ``both_pass`` task produces no native case, so a subset made only of them
#: carries no signal at all. ``all_paired`` is kept as the alternative for when
#: the clustering method wants unchanged tasks as negative space.
EVIDENCE_SCOPES = ("diagnostic_cases", "all_paired")


class SkillGenAdapterError(RuntimeError):
    """The SkillGen adapter was misconfigured or used out of order."""


@dataclass
class NoSkillTrajectories:
    """Evidence of a no-skill Parent: its TRAIN trajectories, i.e. the baseline."""

    parent_id: str
    baseline: NoSkillBaseline
    is_subset: bool = False


def is_no_skill(parent: CandidateState) -> bool:
    return not (parent.skill or "").strip()


class SkillGenAdapter(OptimizerAdapter):
    """Native SkillGen evidence collection and one-step refinement."""

    name = "skillgen"

    def __init__(
        self,
        *,
        cfg: SkillGenNativeConfig,
        instances: Sequence[Any] = (),
        baseline: NoSkillBaseline | None = None,
        analysis=None,
        evidence_scope: str = "diagnostic_cases",
        resynthesise_subset_guidance: bool = True,
    ) -> None:
        """``instances`` are the TRAIN TaskInstances. ``baseline`` may be
        supplied (it is then not re-collected); otherwise it is collected the
        first time a Parent needs it -- in a normal run, as Beam_0's evidence.
        ``analysis`` may inject the RUN-LEVEL analysis (tests); otherwise it
        is induced once, from the full no-skill TRAIN baseline, the first time
        a skilled Parent needs it."""
        if evidence_scope not in EVIDENCE_SCOPES:
            raise SkillGenAdapterError(
                f"evidence_scope must be one of {EVIDENCE_SCOPES}, got {evidence_scope!r}"
            )
        if baseline is None and not instances:
            raise SkillGenAdapterError("provide the TRAIN instances or a baseline")
        self.cfg = cfg
        self.instances = tuple(instances) if instances else tuple(baseline.instances)
        self.baseline = baseline
        #: The run-level SkillAnalysis every skilled Parent refines with.
        self.run_analysis = analysis
        #: analysis_id -> SkillAnalysis, for every analysis this run produced
        #: (the run-level one and round 0's per-subset ones), for reporting.
        self.analyses: dict = {}
        if analysis is not None and getattr(analysis, "analysis_id", None):
            self.analyses[analysis.analysis_id] = analysis
        self.evidence_scope = evidence_scope
        self.resynthesise_subset_guidance = resynthesise_subset_guidance
        self._round_id: int | None = None
        # Observability for tests and run artifacts; never read by the flow.
        self.rounds_begun: list[int] = []
        self.evidence_calls: list[str] = []
        self.propose_calls: list[tuple[str, str]] = []
        self.history_contexts: list[str] = []

    # ── round lifecycle ──────────────────────────────────────────────────

    def begin_round(self, round_id: int) -> None:
        self._round_id = round_id
        self.rounds_begun.append(round_id)

    def end_round(self, round_id: int) -> None:
        self._round_id = None

    # ── parent <-> native candidate ──────────────────────────────────────

    def _contextual_abstract(self, parent: CandidateState) -> str:
        """The parent's candidate-level contextual abstract, if its lineage
        carries one (native `generate_skill` / `refine_skill` rewrite it and it
        rides with the skill through provenance); else the run-level analysis's
        task abstract -- exactly native `refine_skill`'s own fallback."""
        for source in (parent.metadata, parent.operator_provenance,
                       parent.proposal_provenance):
            value = (source or {}).get("contextual_abstract")
            if value:
                return str(value)
        analysis = self.run_analysis
        return getattr(analysis, "contextual_abstract", "") or ""

    def ensure_run_analysis(self, out_dir: str | None = None):
        """The run-level analysis: native induction over the FULL no-skill
        TRAIN baseline, done once per run and shared by every skilled Parent
        (plan B). Requires the baseline, which round 0 always collects."""
        if self.run_analysis is None:
            if self.baseline is None:
                raise SkillGenAdapterError(
                    "the run-level analysis needs the no-skill baseline; round 0 collects it"
                )
            if out_dir is None:
                out_dir = os.path.join(os.path.dirname(self.baseline.out_dir.rstrip("/")),
                                       "run_level_induction")
            self.run_analysis = induce(self.baseline, self.cfg, out_dir=out_dir)
            self.analyses[self.run_analysis.analysis_id] = self.run_analysis
        return self.run_analysis

    def analysis_for(self, parent: CandidateState, *, required: bool = True):
        """The analysis a skilled Parent refines with: the run-level one, for
        every skilled Parent regardless of lineage (plan B)."""
        if not required and self.run_analysis is None:
            return None
        return self.ensure_run_analysis()

    def native_candidate(self, parent: CandidateState):
        analysis = self.analysis_for(parent, required=False)
        return candidate_from_skill(
            candidate_id=parent.candidate_id,
            body=parent.skill,
            contextual_abstract=self._contextual_abstract(parent),
            analysis_id=getattr(analysis, "analysis_id", ""),
        )

    def ensure_baseline(self, out_dir: str) -> NoSkillBaseline:
        """The fixed no-skill TRAIN reference, collected at most once per run."""
        if self.baseline is None:
            self.baseline = collect_no_skill_baseline(self.instances, self.cfg, out_dir=out_dir)
        return self.baseline

    # ── ① evidence, per parent ───────────────────────────────────────────

    def collect_evidence(
        self,
        parent: CandidateState,
        tasks: Sequence[Any],
        *,
        out_dir: str,
        seed: int | None = None,
    ) -> Evidence:
        """This parent's OWN contrastive evidence on the train tasks.

        Per parent, never shared: the repair/regression picture of one skill
        says nothing about another's. What IS shared is the fixed no-skill
        baseline it is measured against -- one reference condition for the
        whole run, which is what makes the pictures comparable.
        """
        task_ids = [_task_id(t) for t in tasks]
        if is_no_skill(parent):
            return self._no_skill_evidence(parent, task_ids, out_dir)
        baseline = self.ensure_baseline(os.path.join(out_dir, "no_skill_baseline"))
        native = collect_contrastive_evidence(
            parent_candidate=self.native_candidate(parent),
            parent_id=parent.candidate_id,
            baseline=baseline,
            cfg=self.cfg,
            task_ids=task_ids,
            out_dir=out_dir,
            round_id=self._round_id if self._round_id is not None else parent.round_id,
        )
        self.evidence_calls.append(parent.candidate_id)
        eff = native.effectiveness
        return Evidence(
            evidence_id=f"{parent.candidate_id}:ev",
            parent_id=parent.candidate_id,
            optimizer_name=self.name,
            task_ids=self._partitionable_ids(native),
            payload=native,
            workspace_dir=out_dir,
            cost=CostAccounting(),
            metadata={
                "evidence_scope": self.evidence_scope,
                "n_train_tasks": len(native.task_ids),
                "bucket_counts": native.bucket_counts(),
                "baseline_reference": "fixed no-skill train baseline (not a split)",
                "repair_count": eff.repair_count if eff else None,
                "regression_count": eff.regression_count if eff else None,
                "net_gain": eff.net_gain if eff else None,
                "native_gate_passed": eff.passed if eff else None,
                "native_gate_is_not_used_by_ours": True,
            },
        )

    def _no_skill_evidence(self, parent, task_ids, out_dir) -> Evidence:
        """A no-skill Parent's evidence: its TRAIN trajectories.

        These ARE SkillGen's Stage-1 no-skill rollouts, and therefore also the
        fixed reference -- so they are collected once, and a no-skill Parent
        that is still in the beam later reuses them rather than producing a
        second, different "baseline".
        """
        baseline = self.ensure_baseline(out_dir)
        view = baseline.restricted(task_ids)
        self.evidence_calls.append(parent.candidate_id)
        return Evidence(
            evidence_id=f"{parent.candidate_id}:ev",
            parent_id=parent.candidate_id,
            optimizer_name=self.name,
            task_ids=tuple(i.instance_id for i in view.instances),
            payload=NoSkillTrajectories(parent_id=parent.candidate_id, baseline=view),
            workspace_dir=out_dir,
            cost=CostAccounting(),
            metadata={
                "evidence_kind": "no_skill_trajectories (SkillGen Stage 1)",
                "native_next_step": "induction + generate_skill",
                "n_train_tasks": len(view.instances),
                "n_failures": len(view.target_failure_ids),
                "n_successes": len(view.success_guard_ids),
                "baseline_reference": "fixed no-skill train baseline (not a split)",
            },
        )

    def _partitionable_ids(self, native: ContrastiveEvidence) -> tuple[str, ...]:
        """Which of this parent's train tasks the partitioner may cluster."""
        if self.evidence_scope == "all_paired":
            return tuple(
                tid for tid, rec in sorted(native.outcomes.items())
                if rec["bucket"] != "unpaired"
            )
        return tuple(
            tid for tid, rec in sorted(native.outcomes.items())
            if rec["bucket"] in DIAGNOSTIC_BUCKETS
        )

    def evidence_features(self, evidence: Evidence) -> dict[str, dict[str, Any]]:
        """Per-task, optimizer-agnostic view for the partitioner.

        Deliberately generic so one clustering method can be written once for
        every optimizer. The SkillGen-specific richness (the case analyst's
        narrative and micro-recommendation) is offered under stable generic
        keys rather than as native objects, so a clusterer never has to import
        SkillGen.

        SCIENTIFIC TODO -- which of these fields a real clusterer should use,
        and in what representation (embedding? symbolic failure mode? bucket
        alone?), is undecided.
        """
        if isinstance(evidence.payload, NoSkillTrajectories):
            baseline = evidence.payload.baseline
            failed = set(baseline.target_failure_ids)
            return {
                tid: {
                    "solved": tid not in failed,
                    "baseline_solved": tid not in failed,
                    "bucket": "baseline_failure" if tid in failed else "baseline_success",
                    "task_type": "",
                    "error_signal": str(getattr(baseline.trajectories.get(tid), "error_summary", "") or ""),
                    "skill_influence": "",
                    "recommendation": "",
                    # The no-skill trajectory as SkillGen's induction reads it.
                    "native_text": _trajectory_text(baseline.trajectories.get(tid),
                                                    baseline.instance_map.get(tid)),
                }
                for tid in evidence.task_ids
            }
        native: ContrastiveEvidence = evidence.payload
        analyses = {str(a.instance_id): a for a in native.feedback.case_analyses}
        eff = native.effectiveness
        cases = {str(c.get("instance_id")): c for c in ((eff.cases if eff else None) or [])}
        out: dict[str, dict[str, Any]] = {}
        for tid in evidence.task_ids:
            rec = native.outcomes.get(tid, {})
            analysis = analyses.get(tid)
            out[tid] = {
                "solved": bool(rec.get("skill_solved")),
                "baseline_solved": bool(rec.get("baseline_solved")),
                "bucket": rec.get("bucket", ""),
                # SkillGen's shared-direct instances carry no native task type
                # (SkillOpt's rollout results do). Reporting the bucket here
                # instead would give a cross-optimizer clusterer two names for
                # one field, so it stays empty and honest.
                "task_type": "",
                "error_signal": (analysis.analysis if analysis else ""),
                "skill_influence": (analysis.skill_influence if analysis else ""),
                "recommendation": (analysis.micro_recommendation if analysis else ""),
                # The native diagnostic case as SkillGen's verification /
                # refinement reads it: the case record the case analyst was
                # given, plus the analyst's own fields.
                "native_text": _case_text(cases.get(tid, rec), analysis),
            }
        return out

    # ── ② subsets ────────────────────────────────────────────────────────

    def slice_evidence(
        self,
        evidence: Evidence,
        subset_id: str,
        task_ids: Sequence[str],
        *,
        partitioner: str = "",
    ) -> EvidenceSubset:
        """Restrict this parent's native evidence to one evidence subset."""
        wanted = [str(t) for t in task_ids]
        if not wanted:
            raise SkillGenAdapterError(f"subset {subset_id} selected no evidence")
        if isinstance(evidence.payload, NoSkillTrajectories):
            view = evidence.payload.baseline.restricted(wanted)
            return EvidenceSubset(
                subset_id=subset_id,
                evidence_id=evidence.evidence_id,
                parent_id=evidence.parent_id,
                task_ids=tuple(i.instance_id for i in view.instances),
                payload=NoSkillTrajectories(parent_id=evidence.parent_id, baseline=view,
                                            is_subset=True),
                partitioner=partitioner,
                metadata={
                    "n_tasks": len(view.instances),
                    "n_failures": len(view.target_failure_ids),
                    "n_successes": len(view.success_guard_ids),
                },
            )
        native: ContrastiveEvidence = evidence.payload
        sliced = slice_contrastive_evidence(
            native, wanted, self.cfg,
            resynthesise_guidance=self.resynthesise_subset_guidance,
        )
        eff = sliced.effectiveness
        return EvidenceSubset(
            subset_id=subset_id,
            evidence_id=evidence.evidence_id,
            parent_id=evidence.parent_id,
            task_ids=tuple(sliced.task_ids),
            payload=sliced,
            partitioner=partitioner,
            metadata={
                "n_tasks": len(sliced.task_ids),
                "bucket_counts": sliced.bucket_counts(),
                "n_case_analyses": len(sliced.feedback.case_analyses),
                "repair_count": eff.repair_count if eff else None,
                "regression_count": eff.regression_count if eff else None,
                "guidance_resynthesised": self.resynthesise_subset_guidance,
            },
        )

    # ── ③ ONE native refinement per (parent, subset) -> RAW MATERIAL ─────

    def propose(
        self,
        parent: CandidateState,
        subset: EvidenceSubset,
        *,
        out_dir: str,
        round_id: int,
        history_context: str = "",
    ) -> ProposalResult:
        """Generate an independent SkillGen proposal from a copy of the parent.
        
        Refinement receives branch history through revision guidance; no-skill generation does not use this context."""
        self.propose_calls.append((parent.candidate_id, subset.subset_id))
        self.history_contexts.append(history_context)
        if isinstance(subset.payload, NoSkillTrajectories):
            return self._propose_from_no_skill(parent, subset, out_dir, history_context)
        native_subset: ContrastiveEvidence = subset.payload
        if history_context:
            native_subset = _with_history_context(native_subset, history_context)
        parent_candidate = self.native_candidate(parent)
        analysis = self.analysis_for(parent)
        try:
            refined = refine_once(
                parent_candidate=parent_candidate,
                analysis=analysis,
                evidence_subset=native_subset,
                cfg=self.cfg,
                out_dir=os.path.join(out_dir, "refine"),
            )
        except SkillGenNativeError as exc:
            return ProposalResult(
                None, status=f"skip_native_error:{exc}", workspace_dir=out_dir,
            )
        body = (refined.body or "").strip()
        if not body:
            # The native refiner returned nothing usable. Not a rejection --
            # there is simply no material from this subset.
            return ProposalResult(None, status="skip_empty_body", workspace_dir=out_dir)
        if body == (parent.skill or "").strip():
            # Legitimate native outcome: the refiner chose to preserve the
            # skill. It is still raw material and still goes to the operator;
            # the status records that nothing changed.
            status = "ok_unchanged"
        else:
            status = "ok"
        eff = native_subset.effectiveness
        return ProposalResult(
            refined.body,
            status=status,
            provenance={
                "native_mechanism": "agents.generation.refine_skill",
                "native_step": "refine_once",
                "analysis_id": getattr(analysis, "analysis_id", None),
                "contextual_abstract": refined.contextual_abstract,
                "parent_candidate_id": parent.candidate_id,
                "subset_task_ids": list(subset.task_ids),
                "subset_bucket_counts": native_subset.bucket_counts(),
                "subset_repair_count": eff.repair_count if eff else None,
                "subset_regression_count": eff.regression_count if eff else None,
                "subset_diagnostic": eff.diagnostic_summary if eff else "",
                "changed_parent_body": status == "ok",
                "history_context_chars": len(history_context or ""),
                "history_context_via": "native_revision_guidance",
            },
            workspace_dir=out_dir,
        )

    def _propose_from_no_skill(self, parent, subset, out_dir, history_context="") -> ProposalResult:
        """Generate a skill from one subset of no-skill trajectories.
        
        History context is recorded but is not passed to native generation."""
        view: NoSkillBaseline = subset.payload.baseline
        if not view.target_failure_ids:
            # Natively: "No failures found. Pipeline stops early - nothing to learn."
            return ProposalResult(None, status="skip_no_failure_evidence", workspace_dir=out_dir)
        try:
            candidate, analysis = generate_from_no_skill_trajectories(
                view, self.cfg, out_dir=os.path.join(out_dir, "generate"),
            )
        except SkillGenNativeError as exc:
            return ProposalResult(None, status=f"skip_native_error:{exc}", workspace_dir=out_dir)
        body = (candidate.body or "").strip()
        if not body:
            return ProposalResult(None, status="skip_empty_body", workspace_dir=out_dir)
        self.analyses[analysis.analysis_id] = analysis
        return ProposalResult(
            candidate.body,
            status="ok",
            provenance={
                "native_mechanism": "agents.induction.run_induction + agents.generation.generate_skill",
                "native_step": "induce_generate",
                "subset_analysis_id": analysis.analysis_id,
                "contextual_abstract": candidate.contextual_abstract,
                "parent_candidate_id": parent.candidate_id,
                "subset_task_ids": list(subset.task_ids),
                "n_failures": len(view.target_failure_ids),
                "n_successes": len(view.success_guard_ids),
                "changed_parent_body": True,
                "history_context_chars": len(history_context or ""),
                "history_context_via": "none: no-skill first step",
            },
            workspace_dir=out_dir,
        )


#: Header separating OURS attempt-aware history from the native guidance.
_HISTORY_CONTEXT_HEADER = "OURS branch history (attempt-aware context):"


def _with_history_context(evidence: ContrastiveEvidence, history_context: str) -> ContrastiveEvidence:
    """Copy evidence and append branch history to its revision guidance."""
    feedback = evidence.feedback
    guidance = (feedback.revision_guidance or "").rstrip()
    joined = f"{guidance}\n\n{_HISTORY_CONTEXT_HEADER}\n{history_context}" if guidance \
        else f"{_HISTORY_CONTEXT_HEADER}\n{history_context}"
    augmented = dataclasses.replace(feedback, revision_guidance=joined)
    return dataclasses.replace(evidence, feedback=augmented)


def _trajectory_text(trajectory, instance=None) -> str:
    """A no-skill trajectory, serialised as SkillGen's own induction reads it.

    Native induction summarises each trajectory from the task input, the
    rendered trace (`clustering.render_trace`, SkillGen's own renderer), the
    outcome and the error summary; those same fields, in that form, are the
    text. Nothing is re-interpreted.
    """
    import json

    from ours.adapters.skillgen_native import render_trajectory_trace

    if trajectory is None:
        return ""
    record = {
        "instance_id": getattr(trajectory, "instance_id", None),
        "task_input": str(getattr(instance, "input", "") or "") if instance is not None else "",
        "trace": render_trajectory_trace(trajectory),
        "final_output": str(getattr(trajectory, "final_output", "") or ""),
        "success": getattr(trajectory, "success", None),
        "score": getattr(trajectory, "score", None),
        "error_summary": getattr(trajectory, "error_summary", None) or "",
    }
    return json.dumps(record, ensure_ascii=False, sort_keys=True, default=str)


def _case_text(case: dict, analysis) -> str:
    """A diagnostic case as SkillGen's verification reads it: the native case
    record (`EffectivenessResult.cases[i]`, what the case analyst is given)
    plus the analyst's own fields that the refiner then consumes."""
    import json

    record = dict(case)
    if analysis is not None:
        record["case_analysis"] = {
            key: getattr(analysis, key, None)
            for key in ("bucket", "analysis", "skill_influence", "micro_recommendation")
        }
    return json.dumps(record, ensure_ascii=False, sort_keys=True, default=str)


def _task_id(task: Any) -> str:
    if isinstance(task, str):
        return task
    if isinstance(task, dict):
        return str(task.get("id") or task.get("task_id") or task.get("instance_id"))
    for attr in ("task_id", "instance_id", "id"):
        if hasattr(task, attr):
            return str(getattr(task, attr))
    raise SkillGenAdapterError(f"cannot read a task id from {task!r}")


__all__ = ["EVIDENCE_SCOPES", "NoSkillTrajectories", "SkillGenAdapter",
           "SkillGenAdapterError", "is_no_skill"]
