"""Generate one candidate skill from a parent and native execution evidence."""
from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from skillopt.gradient.aggregate import merge_patches
from skillopt.optimizer.clip import rank_and_select
from skillopt.optimizer.lr_autonomous import decide_autonomous_learning_rate
from skillopt.optimizer.rewrite import rewrite_skill_from_suggestions
from skillopt.optimizer.skill import apply_patch_with_report
from skillopt.optimizer.update_modes import (
    get_payload_items,
    is_full_rewrite_minibatch_mode,
    normalize_update_mode,
    payload_label,
)
from skillopt.utils import skill_hash


# ── Types ────────────────────────────────────────────────────────────────

@dataclass
class EvidenceBatch:
    """One accumulation batch's worth of task-level evidence.

    ``results`` are the per-task rollout dicts (see
    :class:`skillopt.types.RolloutResult`); ``prediction_dir`` is the
    directory holding ``<task_id>/conversation.json`` for those same tasks.
    Both are what stage ② REFLECT consumes.
    """

    results: list[dict]
    prediction_dir: str
    patches_dir: str
    reflect_dir: str
    seed: int | None = None
    n_items: int = 0
    hard: float = 0.0
    soft: float = 0.0

    def isolated(self) -> EvidenceBatch:
        """Deep copy of everything a proposal call could mutate."""
        return EvidenceBatch(
            results=copy.deepcopy(self.results),
            prediction_dir=self.prediction_dir,
            patches_dir=self.patches_dir,
            reflect_dir=self.reflect_dir,
            seed=self.seed,
            n_items=self.n_items,
            hard=self.hard,
            soft=self.soft,
        )


@dataclass
class ReflectOutput:
    """Result of stage ② for one batch."""

    raw_patches: list = field(default_factory=list)
    failure_patches: list[dict] = field(default_factory=list)
    success_patches: list[dict] = field(default_factory=list)


@dataclass
class StepBudget:
    """A single resolved ④ SELECT budget for one step.

    ``mode`` is ``"fixed"`` (``value`` is the edit budget, one scheduler tick
    already consumed), ``"autonomous"`` (``value`` is None; the budget is an
    optimizer LLM call made per candidate, as native does) or ``"none"``
    (full-rewrite-minibatch mode, which has no budget at all).
    """

    mode: str
    value: int | None = None


@dataclass
class CandidateProposal:
    """Output of one native proposal step.

    ``status`` is ``"ok"``, ``"skip_no_patches"`` or ``"skip_no_rewrite"`` —
    the same three outcomes the native step loop distinguishes.
    """

    status: str
    skill: str
    skill_hash: str = ""
    merged_patch: dict | None = None
    ranked_patch: dict | None = None
    merged_items: list = field(default_factory=list)
    ranked_items: list = field(default_factory=list)
    apply_report: list = field(default_factory=list)
    rewrite_result: dict | None = None
    raw_patches: list = field(default_factory=list)
    failure_patches: list = field(default_factory=list)
    success_patches: list = field(default_factory=list)
    edit_budget: int | None = None
    lr_decision: dict | None = None
    lr_control_mode: str = ""
    n_edits_merged: int = 0
    n_edits_ranked: int = 0
    support_counts: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)


# ── Patch normalisation (moved verbatim from trainer.py) ─────────────────

def normalise_patches(
    raw_patches: list[dict | None],
    update_mode: str = "patch",
) -> tuple[list[dict], list[dict]]:
    """Extract inner 'patch' sub-dict, split into failure/success lists.

    Each element is expected to conform to :class:`~skillopt.types.RawPatch`.

    NOTE: this annotates the incoming patch dicts in place (``source_type`` /
    ``support_count``) and returns aliases of their inner ``patch`` sub-dicts,
    exactly as it always has.  Callers that reuse one ``raw_patches`` list
    across several proposals must copy first — see
    :meth:`EvidenceBatch.isolated` and ``propose_candidate(isolate=True)``.
    """
    mode = normalize_update_mode(update_mode)
    failure: list[dict] = []
    success: list[dict] = []
    for p in raw_patches:
        if not isinstance(p, dict):
            continue
        inner = p.get("patch", p)
        if not isinstance(inner, dict):
            continue
        items = get_payload_items(inner, mode)
        if not items:
            continue
        support = max(int(p.get("batch_size", 0) or 0), 1)
        for item in items:
            if isinstance(item, dict):
                item.setdefault("source_type", p.get("source_type", "failure"))
                item.setdefault("support_count", support)
        if p.get("source_type", "failure") == "success":
            success.append(inner)
        else:
            failure.append(inner)
    return failure, success


# ── ④ SELECT budget ──────────────────────────────────────────────────────

def resolve_step_budget(
    scheduler,
    *,
    update_mode: str,
    lr_control_mode: str,
) -> StepBudget:
    """Consume exactly one scheduler tick and return this step's budget.

    Must be called at most once per training step, and only for a step that
    actually reaches ④ SELECT — that is the native contract.  In
    full-rewrite-minibatch or autonomous mode no tick is consumed, matching
    native, which never reached ``scheduler.step()`` on those paths.
    """
    if is_full_rewrite_minibatch_mode(update_mode):
        return StepBudget(mode="none", value=None)
    if lr_control_mode == "autonomous":
        return StepBudget(mode="autonomous", value=None)
    return StepBudget(mode="fixed", value=scheduler.step())


# ── ② REFLECT ────────────────────────────────────────────────────────────

def reflect_batch(
    adapter,
    parent_skill: str,
    batch: EvidenceBatch,
    *,
    update_mode: str,
    step_buffer_context: str = "",
    meta_skill_context: str = "",
) -> ReflectOutput:
    """Reflect on one accumulation batch to generate candidate patches."""
    raw_patches = adapter.reflect(
        batch.results, parent_skill, batch.reflect_dir,
        prediction_dir=batch.prediction_dir, patches_dir=batch.patches_dir,
        random_seed=batch.seed,
        step_buffer_context=step_buffer_context,
        meta_skill_context=meta_skill_context,
    )
    failure_patches, success_patches = normalise_patches(
        raw_patches,
        update_mode=update_mode,
    )
    return ReflectOutput(
        raw_patches=list(raw_patches),
        failure_patches=failure_patches,
        success_patches=success_patches,
    )


# ── ②③④⑤: parent skill + evidence → ONE candidate skill ─────────────────

def propose_candidate(
    adapter,
    parent_skill: str,
    batches: list[EvidenceBatch],
    *,
    cfg: dict,
    artifact_dir: str,
    update_mode: str,
    lr_control_mode: str,
    budget_resolver: Callable[[], StepBudget],
    merge_batch_size: int,
    analyst_workers: int,
    rollout_hard: float = 0.0,
    rollout_soft: float = 0.0,
    rollout_n: int = 0,
    step_buffer_context: str = "",
    meta_skill_context: str = "",
    rewrite_reasoning_effort: str | None = "high",
    rewrite_max_completion_tokens: int = 64000,
    precomputed: list[ReflectOutput] | None = None,
    lr_history_path: str | None = None,
    lr_history_extra: dict | None = None,
    isolate: bool = False,
    verbose: bool = True,
) -> CandidateProposal:
    """Generate one candidate from native reflection output.
    
    Use precomputed patches when provided; isolate=True copies mutable inputs before applying edits."""
    update_mode = normalize_update_mode(update_mode)
    timings: dict = {}

    if isolate:
        batches = [b.isolated() for b in batches]
        precomputed = copy.deepcopy(precomputed) if precomputed is not None else None

    # ② REFLECT ──────────────────────────────────────────────────────────
    all_raw_patches: list = []
    all_failure_patches: list[dict] = []
    all_success_patches: list[dict] = []

    t_phase = time.time()
    if precomputed is None:
        reflect_outputs = [
            reflect_batch(
                adapter, parent_skill, batch,
                update_mode=update_mode,
                step_buffer_context=step_buffer_context,
                meta_skill_context=meta_skill_context,
            )
            for batch in batches
        ]
    else:
        reflect_outputs = list(precomputed)
    for out in reflect_outputs:
        all_failure_patches.extend(out.failure_patches)
        all_success_patches.extend(out.success_patches)
        all_raw_patches.extend(out.raw_patches)
    if precomputed is None:
        timings["reflect_s"] = round(time.time() - t_phase, 1)

    # ── No patches? Skip ─────────────────────────────────────────────────
    # Native returns here BEFORE consuming a scheduler tick.
    if not all_failure_patches and not all_success_patches:
        return CandidateProposal(
            status="skip_no_patches",
            skill=parent_skill,
            raw_patches=all_raw_patches,
            failure_patches=all_failure_patches,
            success_patches=all_success_patches,
            lr_control_mode=lr_control_mode,
            timings=timings,
        )

    # ③ AGGREGATE ────────────────────────────────────────────────────────
    t_phase = time.time()
    merged_patch = merge_patches(
        parent_skill, all_failure_patches, all_success_patches,
        batch_size=merge_batch_size, verbose=verbose,
        workers=analyst_workers,
        update_mode=update_mode,
        meta_skill_context=meta_skill_context,
    )
    with open(os.path.join(artifact_dir, "merged_patch.json"), "w") as f:
        json.dump(merged_patch, f, ensure_ascii=False, indent=2)

    merged_items = get_payload_items(merged_patch, update_mode)
    n_edits_merged = len(merged_items)
    timings["aggregate_s"] = round(time.time() - t_phase, 1)
    if verbose:
        print(f"    [3/6 done] merged {n_edits_merged} {payload_label(update_mode)}")

    # ④ SELECT ───────────────────────────────────────────────────────────
    t_phase = time.time()
    lr_decision = None
    effective_lr_control_mode = lr_control_mode
    if is_full_rewrite_minibatch_mode(update_mode):
        edit_budget = None
        ranked_patch = merged_patch
        ranked_items = merged_items
        n_edits_ranked = len(ranked_items)
        effective_lr_control_mode = "none"
        with open(os.path.join(artifact_dir, "ranked_edits.json"), "w") as f:
            json.dump(ranked_patch, f, ensure_ascii=False, indent=2)
    else:
        budget = budget_resolver()
        if budget.mode == "autonomous":
            lr_decision = decide_autonomous_learning_rate(
                skill_content=parent_skill,
                merged_patch=merged_patch,
                update_mode=update_mode,
                rollout_hard=rollout_hard,
                rollout_soft=rollout_soft,
                rollout_n=rollout_n,
                step_buffer_context=step_buffer_context,
                meta_skill_context=meta_skill_context,
            )
            edit_budget = int(lr_decision["learning_rate"])
            with open(os.path.join(artifact_dir, "lr_decision.json"), "w") as f:
                json.dump(lr_decision, f, ensure_ascii=False, indent=2)
            if lr_history_path:
                with open(lr_history_path, "a") as f:
                    f.write(json.dumps({
                        **(lr_history_extra or {}),
                        **lr_decision,
                    }, ensure_ascii=False) + "\n")
        else:
            edit_budget = budget.value
        ranked_patch = rank_and_select(
            parent_skill, merged_patch,
            max_edits=edit_budget,
            update_mode=update_mode,
            meta_skill_context=meta_skill_context,
        )
        with open(os.path.join(artifact_dir, "ranked_edits.json"), "w") as f:
            json.dump(ranked_patch, f, ensure_ascii=False, indent=2)

        ranked_items = get_payload_items(ranked_patch, update_mode)
        n_edits_ranked = len(ranked_items)
    timings["select_s"] = round(time.time() - t_phase, 1)

    support_counts = [
        item.get("support_count", 0) for item in ranked_items if isinstance(item, dict)
    ]
    if verbose:
        if is_full_rewrite_minibatch_mode(update_mode):
            print(
                f"    [4/6 SELECT] skipped LR/select; "
                f"using {n_edits_ranked} merged {payload_label(update_mode)}"
            )
        else:
            print(
                f"    [4/6 SELECT] "
                f"{n_edits_merged} -> {n_edits_ranked} {payload_label(update_mode)} "
                f"(budget={edit_budget}, lr_control={effective_lr_control_mode})"
            )

    # ⑤ UPDATE ───────────────────────────────────────────────────────────
    t_phase = time.time()
    rewrite_result = None
    if update_mode == "rewrite_from_suggestions":
        rewrite_result = rewrite_skill_from_suggestions(
            parent_skill,
            ranked_patch,
            step_buffer_context=step_buffer_context,
            env=cfg.get("env"),
            reasoning_effort=rewrite_reasoning_effort,
            max_completion_tokens=rewrite_max_completion_tokens,
        )
        if rewrite_result and rewrite_result.get("new_skill"):
            candidate_skill = rewrite_result["new_skill"]
            apply_report = []
            with open(os.path.join(artifact_dir, "rewrite_result.json"), "w") as f:
                json.dump(rewrite_result, f, ensure_ascii=False, indent=2)
        else:
            candidate_skill = parent_skill
            apply_report = []
    elif is_full_rewrite_minibatch_mode(update_mode):
        skill_candidates = get_payload_items(ranked_patch, update_mode)
        selected_candidate = next(
            (
                item for item in skill_candidates
                if isinstance(item, dict) and str(item.get("new_skill", "")).strip()
            ),
            None,
        )
        if selected_candidate:
            candidate_skill = str(selected_candidate["new_skill"]).rstrip() + "\n"
            apply_report = []
            rewrite_result = {
                "reasoning": ranked_patch.get("reasoning", ""),
                "change_summary": selected_candidate.get("change_summary", []),
                "title": selected_candidate.get("title", ""),
                "source_type": selected_candidate.get("source_type", ""),
            }
            with open(os.path.join(artifact_dir, "full_rewrite_result.json"), "w") as f:
                json.dump(
                    {
                        "selected_candidate": selected_candidate,
                        "merged_patch": ranked_patch,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        else:
            candidate_skill = parent_skill
            apply_report = []
    else:
        candidate_skill, apply_report = apply_patch_with_report(parent_skill, ranked_patch)
    with open(os.path.join(artifact_dir, "candidate_skill.md"), "w") as f:
        f.write(candidate_skill)
    if apply_report:
        with open(os.path.join(artifact_dir, "edit_apply_report.json"), "w") as f:
            json.dump(apply_report, f, indent=2, ensure_ascii=False)

    timings["update_s"] = round(time.time() - t_phase, 1)

    proposal = CandidateProposal(
        status="ok",
        skill=candidate_skill,
        skill_hash=skill_hash(candidate_skill),
        merged_patch=merged_patch,
        ranked_patch=ranked_patch,
        merged_items=merged_items,
        ranked_items=ranked_items,
        apply_report=apply_report,
        rewrite_result=rewrite_result,
        raw_patches=all_raw_patches,
        failure_patches=all_failure_patches,
        success_patches=all_success_patches,
        edit_budget=edit_budget,
        lr_decision=lr_decision,
        lr_control_mode=effective_lr_control_mode,
        n_edits_merged=n_edits_merged,
        n_edits_ranked=n_edits_ranked,
        support_counts=support_counts,
        timings=timings,
    )

    if (
        update_mode == "rewrite_from_suggestions"
        and rewrite_result is None
    ) or (
        is_full_rewrite_minibatch_mode(update_mode)
        and rewrite_result is None
    ):
        proposal.status = "skip_no_rewrite"
        proposal.skill = parent_skill
        proposal.skill_hash = skill_hash(parent_skill)

    return proposal
