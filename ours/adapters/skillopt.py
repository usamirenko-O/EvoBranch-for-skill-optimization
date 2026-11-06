"""Adapt SkillOpt evidence collection and native proposal generation to OURS."""
from __future__ import annotations

import os
from typing import Any, Sequence

from integration.skillopt_phase2.native_adapter import (
    EvalContext,
    Evidence as NativeEvidence,
    StepBudgetPlan,
    collect_evidence as native_collect_evidence,
    evaluate_native,
    propose_native,
)
from ours.adapters.base import EvaluatorAdapter, OptimizerAdapter, ProposalResult
from ours.types import (
    CandidateState,
    CostAccounting,
    Evidence,
    EvidenceSubset,
    Score,
)
from ours.workspace import assert_fresh_dir
from skillopt.utils import compute_score


class SkillOptAdapterError(RuntimeError):
    """The adapter was used outside a round, or misconfigured."""


class SkillOptAdapter(OptimizerAdapter):
    """Native SkillOpt evidence collection and proposal, behind the protocol."""

    name = "skillopt"

    def __init__(
        self,
        *,
        env_adapter,
        scheduler,
        cfg: dict | None = None,
        update_mode: str = "patch",
        lr_control_mode: str = "fixed",
        merge_batch_size: int = 8,
        analyst_workers: int = 1,
        step_buffer_context: str = "",
        meta_skill_context: str = "",
        rewrite_reasoning_effort: str | None = "high",
        rewrite_max_completion_tokens: int = 64000,
        verbose: bool = False,
        round_batches=None,
        env_out_root: str | None = None,
        one_step_scheduler_factory=None,
        epoch_protocol=None,
    ) -> None:
        """Configure native batch collection and proposal budgets.
        
        A scheduler factory creates a fresh one-step scheduler for each round; otherwise the shared scheduler advances across rounds."""
        if one_step_scheduler_factory is not None and (step_buffer_context or meta_skill_context):
            raise SkillOptAdapterError(
                "stateless one-step mode carries no cross-step optimizer state: "
                "step_buffer_context and meta_skill_context must be empty"
            )
        self.env_adapter = env_adapter
        self.scheduler = scheduler
        self.cfg = dict(cfg or {})
        self.update_mode = update_mode
        self.lr_control_mode = lr_control_mode
        self.merge_batch_size = merge_batch_size
        self.analyst_workers = analyst_workers
        self.step_buffer_context = step_buffer_context
        self.meta_skill_context = meta_skill_context
        self.rewrite_reasoning_effort = rewrite_reasoning_effort
        self.rewrite_max_completion_tokens = rewrite_max_completion_tokens
        self.verbose = verbose
        self.round_batches = round_batches
        self.env_out_root = env_out_root
        self.one_step_scheduler_factory = one_step_scheduler_factory


        self.epoch_protocol = epoch_protocol
        #: round_id -> {"mode", "value"} of the budget resolved for that round.
        self.budget_by_round: dict[int, dict] = {}
        self._budget_plan: StepBudgetPlan | None = None
        self._round_id: int | None = None
        self.scheduler_ticks_by_round: dict[int, int] = {}

    # ── round lifecycle ──────────────────────────────────────────────────

    def begin_round(self, round_id: int) -> None:
        """Resolve ONE budget for the whole round, shared by every proposal.

        Shared-scheduler mode consumes one tick of ``self.scheduler``.
        Stateless one-step mode consumes the one tick of a scheduler built for
        this round alone and then discarded.
        """
        scheduler = (self.one_step_scheduler_factory()
                     if self.one_step_scheduler_factory is not None else self.scheduler)
        self._budget_plan = StepBudgetPlan.for_step(
            scheduler,
            update_mode=self.update_mode,
            lr_control_mode=self.lr_control_mode,
        )
        self.budget_by_round[round_id] = {
            "mode": self._budget_plan.budget.mode,
            "value": self._budget_plan.budget.value,
        }
        self._round_id = round_id
        self.scheduler_ticks_by_round[round_id] = (
            self.scheduler_ticks_by_round.get(round_id, 0) + 1
        )

    def end_round(self, round_id: int) -> None:
        self._budget_plan = None
        self._round_id = None

    # ── evidence ─────────────────────────────────────────────────────────

    def collect_evidence(
        self,
        parent: CandidateState,
        tasks: Sequence[Any],
        *,
        out_dir: str,
        seed: int | None = None,
    ) -> Evidence:
        batch_meta: dict = {}
        if self.round_batches is None:
            rollout_input = list(tasks)
        else:
            rollout_input, seed, batch_meta = self._native_round_input(tasks)
        native = native_collect_evidence(
            self.env_adapter, parent.skill, rollout_input,
            out_dir=out_dir, seed=seed,
        )
        return Evidence(
            evidence_id=f"{parent.candidate_id}:ev",
            parent_id=parent.candidate_id,
            optimizer_name=self.name,
            task_ids=tuple(str(r["id"]) for r in native.results),
            payload=native,
            workspace_dir=out_dir,
            cost=CostAccounting(),
            metadata={"hard": native.hard, "soft": native.soft,
                      "n_items": native.n_items, **batch_meta},
        )

    def _native_round_input(self, tasks):
        """This round's native TRAIN batch, built into SkillOpt's own env.

        Every Parent of a round rolls out on the SAME batch (same BatchSpec,
        same seed), exactly as a native step rolls its one incumbent out on
        it. The batch's ids must lie inside ``tasks`` -- the TRAIN pool the
        controller was given -- so a wrong batch can never pull a validation
        or deployment item into evidence.
        """
        if self._round_id is None:
            raise SkillOptAdapterError("collect_evidence() called outside a round")
        batch = self.round_batches(self._round_id)
        pool = {_item_id(t) for t in tasks}
        ids = [_item_id(i) for i in (batch.payload or [])]
        outside = sorted({i for i in ids if i not in pool})
        if outside:
            raise SkillOptAdapterError(
                f"round {self._round_id} native batch names ids outside the TRAIN pool: {outside[:5]}"
            )
        env = self.env_adapter.build_env_from_batch(batch, out_root=self.env_out_root)
        return env, batch.seed, {
            "native_batch_seed": batch.seed,
            "native_batch_task_ids": ids,
        }

    def evidence_features(self, evidence: Evidence) -> dict[str, dict[str, Any]]:
        """Generic per-task view a clusterer can read for any optimizer.

        ``native_text`` is exactly what SkillOpt's own reflection shows its
        analyst for that task: ``skillopt.gradient.reflect
        .fmt_minibatch_trajectories`` applied to that one rollout result (the
        header -- task, type, outcome, fail reason -- plus the rendered
        ``conversation.json`` and any preview / prompt files it includes). If
        native reflection would render nothing for the item (no conversation
        file), the rollout result record itself is serialised as JSON so the
        item stays clusterable; nothing is re-interpreted either way.
        """
        import json

        from skillopt.gradient.reflect import fmt_minibatch_trajectories

        native: NativeEvidence = evidence.payload
        out: dict[str, dict[str, Any]] = {}
        for r in native.results:
            text = fmt_minibatch_trajectories([r], native.prediction_dir)
            if not text.strip():
                text = json.dumps(r, ensure_ascii=False, sort_keys=True, default=str)
            out[str(r["id"])] = {
                "solved": bool(r.get("hard")),
                "hard": float(r.get("hard", 0.0)),
                "soft": float(r.get("soft", 0.0)),
                "task_type": str(r.get("task_type", "")),
                "error_signal": str(r.get("fail_reason", "")),
                "native_text": text,
            }
        return out

    def slice_evidence(
        self,
        evidence: Evidence,
        subset_id: str,
        task_ids: Sequence[str],
        *,
        partitioner: str = "",
    ) -> EvidenceSubset:
        native: NativeEvidence = evidence.payload
        wanted = {str(t) for t in task_ids}
        results = [r for r in native.results if str(r["id"]) in wanted]
        if not results:
            raise SkillOptAdapterError(
                f"subset {subset_id} selected no evidence (task ids {sorted(wanted)})"
            )
        hard, soft = compute_score(results)
        # `prediction_dir` is read-only during reflection, so every subset --
        # and every candidate -- shares the parent's one evidence directory.
        # That is required, not merely safe: all K must see the same evidence.
        sliced = NativeEvidence(
            results=results,
            prediction_dir=native.prediction_dir,
            rollout_dir=native.rollout_dir,
            seed=native.seed,
            hard=hard,
            soft=soft,
            n_items=len(results),
            parent_hash=native.parent_hash,
        )
        return EvidenceSubset(
            subset_id=subset_id,
            evidence_id=evidence.evidence_id,
            parent_id=evidence.parent_id,
            task_ids=tuple(str(r["id"]) for r in results),
            payload=sliced,
            partitioner=partitioner,
            metadata={"hard": hard, "soft": soft, "n_items": len(results)},
        )

    # ── proposal ─────────────────────────────────────────────────────────

    def propose(
        self,
        parent: CandidateState,
        subset: EvidenceSubset,
        *,
        out_dir: str,
        round_id: int,
        history_context: str = "",
    ) -> ProposalResult:
        """Generate one raw SkillOpt proposal, passing branch history through step_buffer_context."""
        if self._budget_plan is None:
            raise SkillOptAdapterError(
                "propose() called outside a round; begin_round() must resolve "
                "the LR-scheduler tick first (once per round, not per candidate)"
            )
        dirs = {
            "base": out_dir,
            "patches": assert_fresh_dir(
                os.path.join(out_dir, "patches"), "proposal patches",
            ),
            "artifacts": assert_fresh_dir(
                os.path.join(out_dir, "artifacts"), "proposal artifacts",
            ),
        }
        step_buffer_context = history_context or self.step_buffer_context
        meta_skill_context = self.meta_skill_context
        if self.epoch_protocol is not None:
            meta_skill_context = self.epoch_protocol.meta_context(parent)
        proposal = propose_native(
            self.env_adapter,
            parent.skill,
            subset.payload,
            cfg=self.cfg,
            dirs=dirs,
            budget_plan=self._budget_plan,
            update_mode=self.update_mode,
            lr_control_mode=self.lr_control_mode,
            merge_batch_size=self.merge_batch_size,
            analyst_workers=self.analyst_workers,
            step_buffer_context=step_buffer_context,
            meta_skill_context=meta_skill_context,
            rewrite_reasoning_effort=self.rewrite_reasoning_effort,
            rewrite_max_completion_tokens=self.rewrite_max_completion_tokens,
            verbose=self.verbose,
        )
        produced = proposal.status == "ok"
        return ProposalResult(
            skill=proposal.skill if produced else None,
            status=proposal.status,
            provenance={
                "native_status": proposal.status,
                "skill_hash": proposal.skill_hash,
                "edit_budget": proposal.edit_budget,
                "lr_control_mode": proposal.lr_control_mode,
                "n_edits_merged": proposal.n_edits_merged,
                "n_edits_ranked": proposal.n_edits_ranked,
                "support_counts": list(proposal.support_counts),
                "update_mode": self.update_mode,
                "subset_task_ids": list(subset.task_ids),
                "history_context_chars": len(history_context or ""),
                "history_context_via": "native_reflection_context",
                # only under the epoch protocol, so a protocol-off run's
                # provenance is byte-identical to before
                **({"meta_skill_context_chars": len(meta_skill_context or "")}
                   if self.epoch_protocol is not None else {}),
            },
            workspace_dir=out_dir,
        )


    def adopt_native_state(self, skill: str, anchor: CandidateState) -> tuple[str, dict]:
        if self.epoch_protocol is None:
            return skill, {}
        return self.epoch_protocol.adopt(skill, anchor)

    def finish_round(self, survivors, *, parents_by_id: dict, round_id: int, out_dir: str):
        if self.epoch_protocol is None:
            return None
        return self.epoch_protocol.finish_round(
            survivors, parents_by_id=parents_by_id, round_id=round_id, out_dir=out_dir)


def _item_id(item) -> str:
    if isinstance(item, dict):
        return str(item.get("id"))
    return str(item)


class SkillOptEvaluatorAdapter(EvaluatorAdapter):
    """Scores any skill with SkillOpt's native selection-set rollout + scorer.

    Returns a number. Never calls ``evaluate_gate``: whether a candidate
    survives is decided by the beam manager over the whole pool.

    The split/size/seed/metric it is constructed with ARE the run's fixed
    validation condition; they are folded into
    :attr:`validation_context_id` so a stored score can be reused across
    rounds only while they are unchanged.
    """

    name = "skillopt_selection_set"

    def __init__(
        self,
        *,
        env_adapter,
        build_eval_env,
        split: str = "valid_seen",
        env_num: int = 0,
        seed: int = 42,
        gate_metric: str = "hard",
        gate_mixed_weight: float = 0.5,
        use_semantic_density: bool = False,
        semantic_density_weight: float = 0.05,
        leading_words: list[str] | None = None,
    ) -> None:
        self.env_adapter = env_adapter
        self.ctx = EvalContext(
            build_eval_env=build_eval_env,
            split=split,
            env_num=env_num,
            seed=seed,
            gate_metric=gate_metric,
            gate_mixed_weight=gate_mixed_weight,
            use_semantic_density=use_semantic_density,
            semantic_density_weight=semantic_density_weight,
            leading_words=leading_words,
        )

    @property
    def validation_context_id(self) -> str:
        """Everything that would make two of these numbers incomparable.

        The validation split is fixed for the whole run, so this string is
        constant and a candidate's score stays reusable across rounds. If any
        of it is changed mid-run the stored scores stop matching and are
        re-measured rather than compared across conditions.
        """
        ctx = self.ctx
        return ":".join(str(x) for x in (
            self.name,
            ctx.split,
            ctx.env_num,
            ctx.seed,
            ctx.gate_metric,
            ctx.gate_mixed_weight,
            ctx.use_semantic_density,
            ctx.semantic_density_weight,
        ))

    def score(self, candidate: CandidateState, *, out_dir: str) -> Score:
        result = evaluate_native(
            self.env_adapter, candidate.skill, self.ctx, out_dir=out_dir,
        )
        return Score(
            value=float(result.gate_score),
            components={
                "hard": result.hard,
                "soft": result.soft,
                "gate_metric": self.ctx.gate_metric,
                "skill_hash": result.skill_hash,
            },
            evaluator=self.name,
            cache_hit=result.cache_hit,
            n_items=result.n_items,
        )
