"""Wrap native SkillGen generation, verification, and independent refinement steps."""
from __future__ import annotations

import copy
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Sequence

from integration.common.paths import SKILLGEN_ROOT, ensure_on_sys_path

ensure_on_sys_path(SKILLGEN_ROOT)

import llm as skillgen_llm  # noqa: E402  SkillGen-internal, flat module layout
from agents.generation import generate_skill, refine_skill  # noqa: E402
from agents.induction import run_induction, save_analysis  # noqa: E402
from agents.verification import (  # noqa: E402
    _synthesise_revision_guidance as native_synthesise_revision_guidance,
    run_verification,
)
from models import (  # noqa: E402
    CandidateSkill,
    EffectivenessResult,
    SkillAnalysis,
    TaskInstance,
    TaskType,
    Trajectory,
    VerificationFeedback,
)
from trajectory import AgentConfig, collect_trajectories  # noqa: E402
from clustering import render_trace as _native_render_trace  # noqa: E402


def render_trajectory_trace(trajectory: Trajectory) -> str:
    """SkillGen's own trace rendering (`clustering.render_trace`), the text its
    induction summariser reads for a trajectory. Exposed here so no module
    outside this file imports SkillGen."""
    return _native_render_trace(trajectory)


# ───────────── SkillGen-internal plumbing, kept behind this module ──────────
#
# Everything below exists so that no module outside this file has to import a
# SkillGen module. SkillGen has a FLAT module layout (`llm`, `models`,
# `trajectory`, `benchmarks.*`), so an import of it from anywhere else would be
# both invisible in a dependency graph and fragile against name collisions.


def reset_token_stats() -> None:
    """Zero SkillGen's own optimizer-side token tracker, at a phase boundary."""
    skillgen_llm.reset_token_stats()


def token_stats() -> list[dict]:
    """SkillGen's own per-(model, stage, call_type) usage rows.

    Covers ONLY SkillGen's optimizer-side calls (induction, generation,
    refinement, the case analyst, the revision synthesiser, embeddings).
    Solver-side work goes through the shared harness's ``SolverSession``, so
    the two never double count.
    """
    return skillgen_llm.get_token_stats()


def task_type(name: str) -> TaskType:
    return TaskType(name)


def build_task_instances(rows: Sequence[dict]) -> list[TaskInstance]:
    """Turn the shared split-materialisation rows into native TaskInstances."""
    return [
        TaskInstance(
            instance_id=r["instance_id"], input=r["input"],
            ground_truth=r["ground_truth"], metadata=r["metadata"],
        )
        for r in rows
    ]


@contextmanager
def shared_direct_context(*, adapter, harness, session):
    """Route SkillGen's task execution through the SHARED Direct harness.

    SkillGen's own ``benchmarks/shared_direct_adapter.py`` is a pre-existing
    hook, not something added here: tagging an instance
    ``metadata["benchmark"] = "shared_direct"`` makes both execution and
    grading go to the shared harness and the benchmark's official scorer, and
    there is deliberately no fallback path. Installing the context is what
    makes a SkillGen+OURS rollout and a SkillOpt rollout the same execution.
    """
    from benchmarks.shared_direct_adapter import (  # SkillGen-internal
        SharedDirectContext, clear_context, install_context,
    )

    install_context(SharedDirectContext(adapter=adapter, harness=harness,
                                        session=session))
    try:
        yield
    finally:
        clear_context()


class SkillGenNativeError(RuntimeError):
    """A native SkillGen step could not be run under OURS semantics."""


#: Buckets of SkillGen's own contrastive accounting. ``both_pass`` is the one
#: outcome that produces no native diagnostic case, because the skill changed
#: nothing there.
DIAGNOSTIC_BUCKETS = ("repair", "regression", "still_failing")
ALL_BUCKETS = DIAGNOSTIC_BUCKETS + ("both_pass",)


@dataclass
class SkillGenNativeConfig:
    """Native knobs, all of them pre-existing SkillGen settings.

    Model roles are proxy aliases exactly as the native runner passes them
    (``integration/experiment/runners/skillgen.py``): the two solver-side roles
    get the run's solver alias, every optimizer-side role gets the run's
    optimizer alias.
    """

    solver_model: str
    optimizer_model: str
    embedding_model: str
    task_type: TaskType = TaskType.BINARY
    temperature: float = 0.0
    max_tokens: int = 4096
    max_tokens_generation: int = 16384
    max_workers: int = 4
    # induction / clustering -- native defaults, surfaced so a smoke run can
    # shrink them. The OURS evidence partition is a SEPARATE mechanism (see
    # `ours/partition.py`); these govern SkillGen's own induction clustering.
    clustering_method: str = "kmeans"
    #: Native `clustering.n_failure_clusters` / `n_success_clusters`. None
    #: means SkillGen picks k itself with Calinski-Harabasz
    #: (`clustering._auto_n_clusters`), which is the native default.
    n_failure_clusters: int | None = None
    n_success_clusters: int | None = None
    min_clusters: int = 2
    max_failure_clusters: int = 8
    max_success_clusters: int = 8
    min_cluster_size: int = 1
    max_contrastive_pairs: int = 16
    # verification / case analysis
    case_analyst_workers: int = 4
    case_analyst_max_tokens: int = 2048
    revision_synthesiser_max_tokens: int = 4096
    # prompt budgets for generation / refinement
    max_failure_clusters_in_prompt: int = 6
    max_success_clusters_in_prompt: int = 6
    max_contrastive_pairs_in_prompt: int = 8

    def agent_config(self) -> AgentConfig:
        """The solver-side agent config, identical in shape to the native one."""
        return AgentConfig(
            model=self.solver_model,
            judge_model=self.solver_model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["task_type"] = self.task_type.value
        return d


# ───────────────────────────── no-skill baseline ────────────────────────────


@dataclass
class NoSkillBaseline:
    """The fixed no-skill reference condition on the TRAIN tasks.

    NOT a split. It is measured once, with no skill in the prompt, on exactly
    the manifest's train ids, and then reused for every parent in every round.
    Everything SkillGen calls evidence is defined relative to it.
    """

    trajectories: dict[str, Trajectory] = field(default_factory=dict)
    instances: tuple[TaskInstance, ...] = ()
    target_failure_ids: tuple[str, ...] = ()
    success_guard_ids: tuple[str, ...] = ()
    out_dir: str = ""

    @property
    def instance_map(self) -> dict[str, TaskInstance]:
        return {i.instance_id: i for i in self.instances}

    def instances_for(self, task_ids: Sequence[str]) -> list[TaskInstance]:
        m = self.instance_map
        return [m[str(t)] for t in task_ids if str(t) in m]

    def target_failures(self) -> list[TaskInstance]:
        return self.instances_for(self.target_failure_ids)

    def success_guard(self) -> list[TaskInstance]:
        return self.instances_for(self.success_guard_ids)

    def baseline_solved(self, task_id: str) -> bool:
        return str(task_id) in set(self.success_guard_ids)

    def restricted(self, task_ids: Sequence[str]) -> "NoSkillBaseline":
        """The same fixed reference, restricted to a subset of its tasks.

        Nothing is re-run: trajectories, instances and the failure/success
        partition are filtered, never re-measured.
        """
        wanted = {str(t) for t in task_ids}
        return NoSkillBaseline(
            trajectories={k: v for k, v in self.trajectories.items() if k in wanted},
            instances=tuple(i for i in self.instances if i.instance_id in wanted),
            target_failure_ids=tuple(t for t in self.target_failure_ids if t in wanted),
            success_guard_ids=tuple(t for t in self.success_guard_ids if t in wanted),
            out_dir=self.out_dir,
        )

    def summary(self) -> dict:
        n = len(self.instances)
        return {
            "role": "fixed no-skill reference condition on TRAIN (not a split)",
            "n_train_instances": n,
            "n_baseline_failures": len(self.target_failure_ids),
            "n_baseline_successes": len(self.success_guard_ids),
            "baseline_accuracy": (len(self.success_guard_ids) / n) if n else None,
        }


def collect_no_skill_baseline(
    instances: Sequence[TaskInstance],
    cfg: SkillGenNativeConfig,
    *,
    out_dir: str,
) -> NoSkillBaseline:
    """Run the native baseline agent with NO skill over the train instances.

    This is ``pipeline.run_pipeline``'s stage 1, unchanged, called once for the
    whole run instead of once per pipeline invocation.
    """
    instances = list(instances)
    if not instances:
        raise SkillGenNativeError("the no-skill baseline needs at least one train task")
    os.makedirs(out_dir, exist_ok=True)
    with skillgen_llm.stage_scope("ours_baseline"):
        trajs = collect_trajectories(
            instances,
            cfg.task_type,
            config=cfg.agent_config(),
            max_workers=cfg.max_workers,
            progress_desc="OURS no-skill baseline",
        )
    by_id = {t.instance_id: t for t in trajs}
    target = tuple(i.instance_id for i in instances
                   if not (by_id.get(i.instance_id) and by_id[i.instance_id].success))
    guard = tuple(i.instance_id for i in instances
                  if by_id.get(i.instance_id) and by_id[i.instance_id].success)
    baseline = NoSkillBaseline(
        trajectories=by_id,
        instances=tuple(instances),
        target_failure_ids=target,
        success_guard_ids=guard,
        out_dir=out_dir,
    )
    with open(os.path.join(out_dir, "no_skill_baseline.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                **baseline.summary(),
                "target_failure_ids": list(target),
                "success_guard_ids": list(guard),
            },
            f, indent=2, ensure_ascii=False,
        )
    return baseline


# ──────────────────────────── induction + seeding ───────────────────────────


def induce(
    baseline: NoSkillBaseline,
    cfg: SkillGenNativeConfig,
    *,
    out_dir: str,
) -> SkillAnalysis:
    """Native multi-aspect induction over the no-skill baseline trajectories.

    Run-level, not per-parent: SkillGen's induction summarises the *baseline*,
    which is a fixed reference, so it does not change when the beam does.
    """
    os.makedirs(out_dir, exist_ok=True)
    failures = [baseline.trajectories[i] for i in baseline.target_failure_ids
                if i in baseline.trajectories]
    successes = [baseline.trajectories[i] for i in baseline.success_guard_ids
                 if i in baseline.trajectories]
    if not failures:
        raise SkillGenNativeError(
            "the no-skill baseline solved every train task; SkillGen has no "
            "failure evidence to induce a skill from"
        )
    with skillgen_llm.stage_scope("ours_induction"):
        analysis = run_induction(
            failures, successes, list(baseline.instances), cfg.task_type,
            contextual_model=cfg.optimizer_model,
            summary_model=cfg.optimizer_model,
            pattern_model=cfg.optimizer_model,
            contrastive_model=cfg.optimizer_model,
            embedding_model=cfg.embedding_model,
            clustering_method=cfg.clustering_method,
            n_failure_clusters=cfg.n_failure_clusters,
            n_success_clusters=cfg.n_success_clusters,
            max_failure_clusters=cfg.max_failure_clusters,
            max_success_clusters=cfg.max_success_clusters,
            min_clusters=cfg.min_clusters,
            min_cluster_size=cfg.min_cluster_size,
            max_contrastive_pairs=cfg.max_contrastive_pairs,
            max_workers=cfg.max_workers,
            artifact_dir=out_dir,
        )
    save_analysis(analysis, out_dir)
    return analysis


def generate_seed_candidate(
    analysis: SkillAnalysis,
    cfg: SkillGenNativeConfig,
    *,
    out_dir: str,
) -> CandidateSkill:
    """Native ``generate_skill`` -- how SkillGen turns induction into a skill.

    Under OURS this is the native step applied to a NO-SKILL Parent (SkillGen's
    native start): see :func:`generate_from_no_skill_trajectories`.
    """
    os.makedirs(out_dir, exist_ok=True)
    with skillgen_llm.stage_scope("ours_generation"):
        return generate_skill(
            analysis,
            plan_model=cfg.optimizer_model,
            execute_model=cfg.optimizer_model,
            max_failure_clusters=cfg.max_failure_clusters_in_prompt,
            max_success_clusters=cfg.max_success_clusters_in_prompt,
            max_contrastive_pairs=cfg.max_contrastive_pairs_in_prompt,
            max_tokens_generation=cfg.max_tokens_generation,
            candidate_dir=out_dir,
            generate_scripts=False,
            use_web_search=False,
        )


def generate_from_no_skill_trajectories(
    baseline_subset: NoSkillBaseline,
    cfg: SkillGenNativeConfig,
    *,
    out_dir: str,
):
    """ONE native SkillGen step from the no-skill state, on one evidence subset.

    SkillGen's native first step, unchanged: induction over the no-skill
    trajectories (failures and successes) -> ``generate_skill``. Under OURS it
    runs once per evidence subset, each on only that subset's trajectories, so
    the K generations are siblings of the same no-skill Parent.

    Returns ``(candidate, analysis)``. Raises :class:`SkillGenNativeError` when
    the subset holds no failure: natively SkillGen then has nothing to learn
    and produces no skill ("No failures found. Pipeline stops early").
    """
    analysis = induce(baseline_subset, cfg, out_dir=os.path.join(out_dir, "induction"))
    candidate = generate_seed_candidate(analysis, cfg, out_dir=os.path.join(out_dir, "generation"))
    return assert_body_only(candidate), analysis


def candidate_from_skill(
    *,
    candidate_id: str,
    body: str,
    contextual_abstract: str,
    analysis_id: str | None = None,
) -> CandidateSkill:
    """Rebuild a native ``CandidateSkill`` from a candidate's skill text.

    OURS moves skills as text, because text is what the beam ranks and what
    deployment runs. The body-only generation path is used throughout (no
    scripts, no reference docs), so the text IS the candidate -- see
    :func:`assert_body_only`.
    """
    return CandidateSkill(
        candidate_id=candidate_id,
        analysis_id=analysis_id or "",
        body=body,
        contextual_abstract=contextual_abstract,
    )


def assert_body_only(candidate: CandidateSkill) -> CandidateSkill:
    """Refuse a candidate whose meaning is not fully carried by its body.

    SkillGen's resource-bundle path (``generate_skill_with_resources``) emits
    scripts and reference docs alongside the body. Under OURS a candidate
    travels as text through the pool, the beam and deployment, so a bundle
    candidate would lose half of itself silently. Supporting bundles is an
    open design question; until it is answered this fails loudly.
    """
    if getattr(candidate, "scripts", None) or getattr(candidate, "reference_docs", None):
        raise SkillGenNativeError(
            f"candidate {candidate.candidate_id} carries scripts/reference_docs; "
            "OURS moves skills as text only. Use the body-only generation path "
            "(generate_scripts=False) -- bundle support is an open design question."
        )
    return candidate


# ────────────────────────── contrastive evidence ────────────────────────────


@dataclass
class ContrastiveEvidence:
    """SkillGen-native contrastive evidence for ONE parent on the train tasks.

    ``feedback`` is the native :class:`VerificationFeedback` verbatim: the
    paired :class:`EffectivenessResult` (repair / regression / still-failing
    against the fixed no-skill baseline), the per-case analyst narratives, and
    the synthesised revision guidance. It is exactly what native
    ``refine_skill`` consumes, which is why nothing here needs translating.

    ``outcomes`` is the per-task view the framework's partitioner reads
    (baseline solved? skill solved? which bucket?), derived from the native
    result rather than recomputed from a second rollout.
    """

    parent_id: str
    feedback: VerificationFeedback
    #: The parent skill text this evidence was measured under. Carried so a
    #: subset's revision guidance can be re-synthesised against the same body
    #: the cases were produced from.
    parent_body: str = ""
    outcomes: dict[str, dict] = field(default_factory=dict)
    task_ids: tuple[str, ...] = ()
    out_dir: str = ""
    is_subset: bool = False

    @property
    def effectiveness(self) -> EffectivenessResult | None:
        return self.feedback.effectiveness

    def bucket_counts(self) -> dict:
        counts = {b: 0 for b in ALL_BUCKETS}
        for rec in self.outcomes.values():
            counts[rec["bucket"]] = counts.get(rec["bucket"], 0) + 1
        return counts


def _derive_outcomes(
    eff: EffectivenessResult,
    baseline: NoSkillBaseline,
    instances: Sequence[TaskInstance],
) -> dict[str, dict]:
    """Per-task (baseline, skill) outcomes, read off the native result.

    Complete by construction: the target bucket is exactly the baseline
    failures and the guard bucket exactly the baseline successes, and the
    native result names every id that flipped in either direction, so the rest
    of the guard bucket is ``both_pass``.
    """
    repaired = set(eff.repaired_ids)
    regressed = set(eff.regression_ids)
    still_failing = set(eff.failed_ids_after_skill)
    guard = set(baseline.success_guard_ids)

    out: dict[str, dict] = {}
    for inst in instances:
        tid = inst.instance_id
        baseline_ok = tid in guard
        if tid in repaired:
            bucket, skill_ok = "repair", True
        elif tid in regressed:
            bucket, skill_ok = "regression", False
        elif tid in still_failing:
            bucket, skill_ok = "still_failing", False
        elif baseline_ok:
            bucket, skill_ok = "both_pass", True
        else:
            # Baseline-failed and named by none of the three id lists: the
            # native pass produced no paired outcome for it (a dropped run).
            # Recorded as unpaired rather than guessed at.
            bucket, skill_ok = "unpaired", False
        out[tid] = {
            "task_id": tid,
            "baseline_solved": baseline_ok,
            "skill_solved": skill_ok,
            "bucket": bucket,
            "in_target_bucket": not baseline_ok,
        }
    return out


def collect_contrastive_evidence(
    *,
    parent_candidate: CandidateSkill,
    parent_id: str,
    baseline: NoSkillBaseline,
    cfg: SkillGenNativeConfig,
    task_ids: Sequence[str],
    out_dir: str,
    round_id: int = 0,
) -> ContrastiveEvidence:
    """Native ``run_verification`` for ONE parent against the fixed baseline.

    Contrast is the native identity: the same train tasks are run once with no
    skill (already done -- the cached fixed reference) and once with this
    parent's skill, and the difference is the evidence. This is the same
    function ``pipeline.py`` calls, with the same arguments, differing only in
    that the baseline cache is the run-level fixed reference rather than a
    per-pipeline one.
    """
    os.makedirs(out_dir, exist_ok=True)
    wanted = [str(t) for t in task_ids]
    instances = baseline.instances_for(wanted)
    if not instances:
        raise SkillGenNativeError(f"parent {parent_id}: no train instances for {wanted[:5]}")

    guard_ids = set(baseline.success_guard_ids)
    targets = [i for i in instances if i.instance_id not in guard_ids]
    guards = [i for i in instances if i.instance_id in guard_ids]

    # The fixed reference is passed by copy: `verify_effectiveness` tops the
    # cache up in place, and the run-level reference must not silently acquire
    # entries from a parent-specific call.
    cache = {k: v for k, v in baseline.trajectories.items()}

    with skillgen_llm.stage_scope(f"ours_evidence_r{round_id}"):
        feedback, _ = run_verification(
            parent_candidate, targets, guards, cfg.task_type,
            round_idx=round_id,
            baseline_cache=cache,
            baseline_agent_model=cfg.solver_model,
            baseline_judge_model=cfg.solver_model,
            effectiveness_agent_model=cfg.solver_model,
            effectiveness_judge_model=cfg.solver_model,
            agent_max_tokens=cfg.max_tokens,
            case_analyst_model=cfg.optimizer_model,
            case_analyst_max_workers=cfg.case_analyst_workers,
            case_analyst_max_tokens=cfg.case_analyst_max_tokens,
            revision_synthesiser_model=cfg.optimizer_model,
            revision_synthesiser_max_tokens=cfg.revision_synthesiser_max_tokens,
            artifact_dir=out_dir,
            artifact_prefix="ours_evidence",
            progress_label=f"OURS evidence {parent_id}",
            router_model=None,
        )

    eff = feedback.effectiveness
    if eff is None:
        raise SkillGenNativeError(
            f"parent {parent_id}: native verification returned no effectiveness result"
        )
    outcomes = _derive_outcomes(eff, baseline, instances)
    return ContrastiveEvidence(
        parent_id=parent_id,
        feedback=feedback,
        parent_body=parent_candidate.body or "",
        outcomes=outcomes,
        task_ids=tuple(i.instance_id for i in instances),
        out_dir=out_dir,
    )


def _subset_effectiveness(
    eff: EffectivenessResult,
    outcomes: dict[str, dict],
    subset_ids: Sequence[str],
) -> EffectivenessResult:
    """Recount the native contrastive statistics over one evidence subset.

    The arithmetic mirrors ``effectiveness.verify_effectiveness``. What it
    deliberately does NOT mirror is the gate: ``passed`` is always False and
    the summary says so, because the native net-gain gate is a 1-vs-baseline
    accept/reject decision and OURS never consults it -- selection is one
    global Top-M over the competitive pool.
    """
    ids = [str(t) for t in subset_ids]
    recs = [outcomes[t] for t in ids if t in outcomes]
    paired = [r for r in recs if r["bucket"] != "unpaired"]

    repaired = [r["task_id"] for r in paired if r["bucket"] == "repair"]
    regressed = [r["task_id"] for r in paired if r["bucket"] == "regression"]
    still_failing = [r["task_id"] for r in paired if r["bucket"] == "still_failing"]
    n_target = sum(1 for r in paired if r["in_target_bucket"])
    n_guard = len(paired) - n_target
    baseline_ok = sum(1 for r in paired if r["baseline_solved"])
    skill_ok = sum(1 for r in paired if r["skill_solved"])
    n = len(paired)
    baseline_fail = n - baseline_ok
    net_gain = len(repaired) - len(regressed)

    summary = (
        "mode=ours_evidence_subset (NOT the native net-gain gate; OURS selects "
        "by global Top-M on the fixed validation set), "
        f"subset_n={n}, n_target={n_target}, n_success_guard={n_guard}, "
        f"baseline_acc={(baseline_ok / n if n else 0.0):.1%}, "
        f"skill_acc={(skill_ok / n if n else 0.0):.1%}, "
        f"repair={len(repaired)}/{baseline_fail}, "
        f"regression={len(regressed)}/{baseline_ok}, "
        f"still_failing={len(still_failing)}, net_gain={net_gain}"
    )
    subset_ids_set = set(ids)
    return EffectivenessResult(
        passed=False,
        n_target=n_target,
        n_boundary=n_guard,
        paired_n=n,
        baseline_acc=(baseline_ok / n) if n else 0.0,
        skill_acc=(skill_ok / n) if n else 0.0,
        repair_count=len(repaired),
        regression_count=len(regressed),
        repair_rate=(len(repaired) / baseline_fail) if baseline_fail else 0.0,
        regression_rate=(len(regressed) / baseline_ok) if baseline_ok else 0.0,
        net_gain=net_gain,
        target_repair_count=len(repaired),
        target_fail_count=len(still_failing),
        success_guard_regression_count=len(regressed),
        success_guard_pass_count=sum(
            1 for r in paired if r["baseline_solved"] and r["skill_solved"]
        ),
        repaired_ids=repaired,
        regression_ids=regressed,
        failed_ids_after_skill=still_failing,
        cases=[c for c in (eff.cases or []) if str(c.get("instance_id")) in subset_ids_set],
        diagnostic_summary=summary,
    )


def slice_contrastive_evidence(
    evidence: ContrastiveEvidence,
    task_ids: Sequence[str],
    cfg: SkillGenNativeConfig,
    *,
    parent_body: str | None = None,
    resynthesise_guidance: bool = True,
) -> ContrastiveEvidence:
    """Restrict one parent's native evidence to a subset of its train tasks."""
    ids = [str(t) for t in task_ids]
    eff = evidence.effectiveness
    if eff is None:
        raise SkillGenNativeError(f"evidence for {evidence.parent_id} has no effectiveness")
    body = evidence.parent_body if parent_body is None else parent_body

    subset_eff = _subset_effectiveness(eff, evidence.outcomes, ids)
    subset_ids = set(ids)
    analyses = [a for a in evidence.feedback.case_analyses if str(a.instance_id) in subset_ids]

    subset_feedback = VerificationFeedback(
        effectiveness=subset_eff,
        round_idx=evidence.feedback.round_idx,
        case_analyses=analyses,
        revision_guidance=evidence.feedback.revision_guidance,
    )
    if resynthesise_guidance:
        if analyses:
            with skillgen_llm.stage_scope("ours_evidence_subset_guidance"):
                subset_feedback.revision_guidance = native_synthesise_revision_guidance(
                    skill_body=body,
                    diagnostic_summary=subset_eff.diagnostic_summary,
                    case_analyses=analyses,
                    model=cfg.optimizer_model,
                    max_tokens=cfg.revision_synthesiser_max_tokens,
                )
        else:
            # No diagnostic case in this subset: nothing to synthesise from,
            # and inheriting the parent-level guidance would be exactly the
            # cross-subset leak this branch exists to avoid.
            subset_feedback.revision_guidance = json.dumps(
                {
                    "overall_direction": (
                        "This evidence subset contains no repair, regression or "
                        "still-failing case - the skill changed nothing on these "
                        "tasks. Preserve the current skill body."
                    ),
                    "keep": [], "remove_or_soften": [], "add_or_sharpen": [],
                    "length_budget_note": "(no case signal in this subset)",
                },
                indent=2, ensure_ascii=False,
            )

    return ContrastiveEvidence(
        parent_id=evidence.parent_id,
        feedback=subset_feedback,
        parent_body=body,
        outcomes={t: evidence.outcomes[t] for t in ids if t in evidence.outcomes},
        task_ids=tuple(ids),
        out_dir=evidence.out_dir,
        is_subset=True,
    )


# ─────────────────────────────── refine once ────────────────────────────────


def refine_once(
    *,
    parent_candidate: CandidateSkill,
    analysis: SkillAnalysis,
    evidence_subset: ContrastiveEvidence,
    cfg: SkillGenNativeConfig,
    out_dir: str,
) -> CandidateSkill:
    """ONE native SkillGen refinement step, from ONE parent, on ONE subset.

    ``agents.generation.refine_skill`` mutates the candidate it is handed and
    returns the same object. A deep copy per call is therefore load-bearing,
    not hygiene: without it the K calls for one parent would each start from
    the previous call's output and produce the sequential chain
    ``S -> Q1 -> Q2 -> Q3`` instead of K siblings of S.

    The result is RAW MATERIAL for the OURS operator. It is not compared to
    the parent here, not gated, not scored, and not a competitive candidate.
    """
    os.makedirs(out_dir, exist_ok=True)
    working = copy.deepcopy(parent_candidate)
    if working.body != parent_candidate.body:  # pragma: no cover - defensive
        raise SkillGenNativeError("deep copy did not preserve the parent body")
    with skillgen_llm.stage_scope("ours_refine_once"):
        refined = refine_skill(
            working, analysis, evidence_subset.feedback,
            model=cfg.optimizer_model,
            max_failure_clusters=cfg.max_failure_clusters_in_prompt,
            max_success_clusters=cfg.max_success_clusters_in_prompt,
            max_tokens_generation=cfg.max_tokens_generation,
            candidate_dir=out_dir,
            generate_scripts=False,
            use_web_search=False,
        )
    if refined is parent_candidate:  # pragma: no cover - defensive
        raise SkillGenNativeError(
            "refine_once mutated the parent candidate in place; siblings would "
            "degenerate into a sequential refinement chain"
        )
    return assert_body_only(refined)


__all__ = [
    "ALL_BUCKETS",
    "render_trajectory_trace",
    "build_task_instances",
    "reset_token_stats",
    "shared_direct_context",
    "task_type",
    "token_stats",
    "DIAGNOSTIC_BUCKETS",
    "ContrastiveEvidence",
    "NoSkillBaseline",
    "SkillGenNativeConfig",
    "SkillGenNativeError",
    "assert_body_only",
    "candidate_from_skill",
    "collect_contrastive_evidence",
    "collect_no_skill_baseline",
    "generate_from_no_skill_trajectories",
    "generate_seed_candidate",
    "induce",
    "refine_once",
    "slice_contrastive_evidence",
]
