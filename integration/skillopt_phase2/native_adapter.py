"""Wrap native SkillOpt proposal generation and evaluation."""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from typing import Any, Callable

from skillopt.engine.propose import (
    CandidateProposal,
    EvidenceBatch,
    StepBudget,
    propose_candidate,
    resolve_step_budget,
)
from skillopt.evaluation.gate import select_gate_score
from skillopt.utils import compute_score, skill_hash


# Use the shared workspace utilities for isolated candidate directories.


from ours.workspace import (  
    CandidateIsolationError,
    CandidateWorkspace,
    assert_fresh_dir as _assert_fresh,
)

__all__ = [
    "CandidateIsolationError",
    "CandidateWorkspace",
    "Evidence",
    "EvalContext",
    "CandidateScore",
    "StepBudgetPlan",
    "collect_evidence",
    "propose_native",
    "evaluate_native",
]


# ── COLLECT_EVIDENCE ─────────────────────────────────────────────────────

@dataclass
class Evidence:
    """Task-level evidence for one parent skill, shared by all K candidates.

    ``results`` and ``prediction_dir`` are exactly what stage ② REFLECT reads.
    ``prediction_dir`` is read-only during reflection, so sharing it across
    candidates is not merely safe but required -- all K must see the same
    evidence.
    """

    results: list[dict]
    prediction_dir: str
    rollout_dir: str
    seed: int | None
    hard: float
    soft: float
    n_items: int
    parent_hash: str = ""

    def batches(self, patches_dir: str, reflect_dir: str) -> list[EvidenceBatch]:
        return [
            EvidenceBatch(
                results=self.results,
                prediction_dir=self.prediction_dir,
                patches_dir=patches_dir,
                reflect_dir=reflect_dir,
                seed=self.seed,
                n_items=self.n_items,
                hard=self.hard,
                soft=self.soft,
            )
        ]


def collect_evidence(
    adapter,
    parent_skill: str,
    task_subset: list[dict],
    *,
    out_dir: str,
    seed: int | None = None,
) -> Evidence:
    """COLLECT_EVIDENCE — native ① ROLLOUT over a task subset.

    ``out_dir`` must be unique per (parent, call): ``run_batch`` resumes from
    ``<out_dir>/results.jsonl`` by task id, ignoring which skill produced it.
    """
    _assert_fresh(out_dir, "evidence rollout")
    results = adapter.rollout(
        task_subset, parent_skill, out_dir,
        use_eval_feedback=True,
    )
    hard, soft = compute_score(results)
    return Evidence(
        results=results,
        prediction_dir=os.path.join(out_dir, "predictions"),
        rollout_dir=out_dir,
        seed=seed,
        hard=hard,
        soft=soft,
        n_items=len(results),
        parent_hash=skill_hash(parent_skill),
    )


# ── Budget: one scheduler tick per step, shared by K candidates ──────────

@dataclass
class StepBudgetPlan:
    """Resolve the ④ SELECT budget once per step; hand it to every candidate.

    Native consumes exactly one ``scheduler.step()`` per step that reaches
    SELECT.  Calling it per candidate would advance the LR schedule K times
    per step and silently change the recipe.
    """

    budget: StepBudget

    @classmethod
    def for_step(
        cls,
        scheduler,
        *,
        update_mode: str,
        lr_control_mode: str,
    ) -> StepBudgetPlan:
        return cls(budget=resolve_step_budget(
            scheduler,
            update_mode=update_mode,
            lr_control_mode=lr_control_mode,
        ))

    def resolver(self) -> Callable[[], StepBudget]:
        """A resolver that returns the already-consumed tick, never a new one."""
        return lambda: self.budget


# ── PROPOSE_NATIVE ───────────────────────────────────────────────────────

def propose_native(
    adapter,
    parent_skill: str,
    evidence: Evidence,
    *,
    cfg: dict,
    dirs: dict[str, str],
    budget_plan: StepBudgetPlan,
    update_mode: str,
    lr_control_mode: str,
    merge_batch_size: int,
    analyst_workers: int,
    step_buffer_context: str = "",
    meta_skill_context: str = "",
    rewrite_reasoning_effort: str | None = "high",
    rewrite_max_completion_tokens: int = 64000,
    verbose: bool = False,
) -> CandidateProposal:
    """PROPOSE_NATIVE — one native ②③④⑤ pass → ONE candidate skill.

    Calls the same :func:`skillopt.engine.propose.propose_candidate` the
    native trainer calls.  ``dirs`` must come from
    :meth:`CandidateWorkspace.candidate_dirs`, which guarantees this
    candidate's ``patches``/``artifacts`` are its own.
    """
    batches = evidence.batches(
        patches_dir=dirs["patches"],
        reflect_dir=dirs["base"],
    )
    return propose_candidate(
        adapter,
        parent_skill,
        batches,
        cfg=cfg,
        artifact_dir=dirs["artifacts"],
        update_mode=update_mode,
        lr_control_mode=lr_control_mode,
        budget_resolver=budget_plan.resolver(),
        merge_batch_size=merge_batch_size,
        analyst_workers=analyst_workers,
        rollout_hard=evidence.hard,
        rollout_soft=evidence.soft,
        rollout_n=evidence.n_items,
        step_buffer_context=step_buffer_context,
        meta_skill_context=meta_skill_context,
        rewrite_reasoning_effort=rewrite_reasoning_effort,
        rewrite_max_completion_tokens=rewrite_max_completion_tokens,
        precomputed=None,          # re-run REFLECT: this is a fresh proposal
        lr_history_path=None,      # per-candidate; never appended to the run-level log
        isolate=True,              # deep-copy everything a stage could annotate
        verbose=verbose,
    )


# ── EVALUATE_NATIVE ──────────────────────────────────────────────────────

@dataclass
class EvalContext:
    """Everything EVALUATE_NATIVE needs, minus the candidate itself.

    ``sel_cache`` is intentionally shared across candidates: it is keyed by
    ``skill_hash``, so two identical candidates cost one evaluation. That is
    native behaviour (``trainer.py``'s own ``sel_cache``) and it means the
    number of *distinct* candidates, not K, drives evaluation cost.
    """

    build_eval_env: Callable[..., Any]
    split: str = "valid_seen"
    env_num: int = 0
    seed: int = 42
    gate_metric: str = "hard"
    gate_mixed_weight: float = 0.5
    use_semantic_density: bool = False
    semantic_density_weight: float = 0.05
    leading_words: list[str] | None = None
    sel_cache: dict[str, tuple[float, float]] = field(default_factory=dict)


@dataclass
class CandidateScore:
    hard: float
    soft: float
    gate_score: float
    skill_hash: str
    n_items: int
    cache_hit: bool


def evaluate_native(
    adapter,
    candidate_skill: str,
    ctx: EvalContext,
    *,
    out_dir: str,
) -> CandidateScore:
    """EVALUATE_NATIVE — native ⑥ scoring of one candidate on the selection set.

    Deliberately does **not** call ``evaluate_gate``: with K candidates the
    accept/reject decision needs an argmax over K first, and that decision
    layer is not this function's job.  It returns the same metric-space score
    the native gate would compare.
    """
    cand_hash = skill_hash(candidate_skill)
    if cand_hash in ctx.sel_cache:
        hard, soft = ctx.sel_cache[cand_hash]
        cache_hit = True
        n_items = 0
    else:
        _assert_fresh(out_dir, "selection eval")
        sel_env, n_items = ctx.build_eval_env(
            split=ctx.split, env_num=ctx.env_num, seed=ctx.seed,
        )
        results = adapter.rollout(sel_env, candidate_skill, out_dir)
        hard, soft = compute_score(results)
        ctx.sel_cache[cand_hash] = (hard, soft)
        cache_hit = False
    gate_score = select_gate_score(
        hard, soft, ctx.gate_metric, ctx.gate_mixed_weight,
        skill_content=candidate_skill,
        use_semantic_density=ctx.use_semantic_density,
        semantic_density_weight=ctx.semantic_density_weight,
        leading_words=ctx.leading_words,
    )
    return CandidateScore(
        hard=hard,
        soft=soft,
        gate_score=gate_score,
        skill_hash=cand_hash,
        n_items=n_items,
        cache_hit=cache_hit,
    )
