"""Run SSO with OURS search and shared benchmark evaluation."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from integration.benchmarks.manifest import SplitManifest
from integration.benchmarks.registry import get_adapter
from integration.common.resolve_models import resolve_model_aliases
from integration.experiment.config import code_freeze_reference, optimizer_side_env
from integration.experiment.cost import PhaseCost, PhaseCostAccumulator
from integration.experiment.execute import PreflightError, preflight, resolve_manifest
from integration.experiment.runners.skills import NATIVE_START_CONDITION
from integration.experiment.runners.sso import (
    resolve_sso_knobs,
    resolve_training_schedule,
    schedule_from,
)
from integration.harness.base import SolverSession
from integration.harness.registry import HARNESS_KIND, build_harness
from ours.adapters.shared_benchmark import SharedBenchmarkEvaluator
from ours.adapters.sso import (
    SSOAdapter,
    build_native_components,
    optimizer_side_usage,
    tasks_from_benchmark,
)
from ours.experiment.common import (
    assert_no_deployment_ids,
    deploy_final_beam,
    design_placeholders,
    final_beam_summary,
    finalize_search,
    harness_options,
    native_initial_beam,
    round_summary,
    run_resources,
)
from ours.experiment.config import OursRunConfig
from ours.experiment.record import build_ours_record, write_ours_record
from ours.search import SearchResult, run_search
from ours.workspace import RunWorkspace


@dataclass
class SSOOptimizationOutcome:
    search: SearchResult
    optimization_cost: PhaseCost
    native_state: dict
    wall_time_s: float
    run_dir: Path
    manifest: SplitManifest


@dataclass
class OursOutcome:
    record: dict
    artifact_path: Path
    manifest: SplitManifest
    search: SearchResult
    final_skill_path: Path
    run_dir: Path
    deployment: object = None
    extras: dict = field(default_factory=dict)


# ─────────────────────────────── settings ───────────────────────────────────


def training_schedule(config: OursRunConfig):
    """SSO's training schedule for this run, through the SAME resolver native
    SSO uses (`runners/sso.py::resolve_training_schedule`). |X_t| and the seed
    are the `--extra` overrides or the main-experiment values for the
    benchmark; T is this run's OURS rounds (itself the main-experiment
    `sso_steps` unless given explicitly, see `OursRunConfig.build`)."""
    return resolve_training_schedule(
        config.benchmark, config.base.extra,
        steps=(config.rounds, config.rounds_source or "OursDesign.rounds"),
    )


def native_settings(config: OursRunConfig) -> dict:
    """SSO's own knobs, through the SAME resolvers native SSO uses
    (`runners/sso.py::resolve_training_schedule` / `resolve_sso_knobs`): the
    `--extra sso_*` override, else the authoritative main-experiment config.
    `sso_steps` has no meaning here: the number of outer rounds is `--rounds`."""
    extra = config.base.extra
    training = training_schedule(config)
    knobs = resolve_sso_knobs(extra)
    return {
        "train_per_step": training.train_per_step,
        "history_per_step": knobs.history_per_step,
        "num_probes": knobs.num_probes,
        "update_budget": knobs.update_budget,
        "schedule_seed": training.seed,
        "optimizer_max_tokens": int(extra.get("sso_optimizer_max_tokens",
                                              config.base.optimizer_max_tokens)),
        "optimizer_temperature": float(extra.get("sso_optimizer_temperature",
                                                 config.base.temperature)),
    }


def native_schedule(config: OursRunConfig, manifest: SplitManifest) -> list:
    """The native X_t / H_t schedule, one entry per OURS round: native SSO's
    own `schedule_from` / `build_schedule`, so round t's X_t equals native
    step t's X_t for the same benchmark, |X_t| and seed."""
    s = native_settings(config)
    return schedule_from(
        training_schedule(config), manifest.task_ids("train"), manifest.task_ids("val"),
        history_per_step=s["history_per_step"],
    )


def training_schedule_plan(config: OursRunConfig, manifest: SplitManifest) -> dict:
    """The resolved training schedule and every round's X_t, for a dry-run."""
    return {**training_schedule(config).record(native_schedule(config, manifest), steps_name="rounds"),
            "sso_knobs": resolve_sso_knobs(config.base.extra).record()}


# ─────────────────────────────── preflight ──────────────────────────────────


def ours_preflight(config: OursRunConfig, manifest: SplitManifest, *,
                   for_deployment: bool = True) -> list:
    """Every blocking problem, before a single token is spent.

    The native SSO blockers are reused verbatim through the shared
    `preflight` (manifest validation, benchmark resources, SSO's own checks);
    design-level blockers come from `OursDesign.preflight`, identically for
    every optimizer.
    """
    errors = list(preflight(config.base, manifest))
    design_errors = config.design.preflight()
    if not for_deployment:
        rule_errors = set(config.design.build_deployment_rule().check(config.beam_m))
        design_errors = [e for e in design_errors if e not in rule_errors]
    errors.extend(design_errors)
    extra = config.base.extra
    if "sso_steps" in extra:
        errors.append(
            "--extra sso_steps is the NATIVE step count; under OURS the number of "
            "outer rounds is --rounds. Remove sso_steps so the two cannot disagree."
        )
    try:
        native_settings(config)
        if manifest.task_ids("train"):
            native_schedule(config, manifest)
    except (TypeError, ValueError) as exc:
        errors.append(f"native SSO schedule/settings invalid: {exc}")
    return errors


# ─────────────────────────────── optimization ───────────────────────────────


def _prepare(config: OursRunConfig, manifest: SplitManifest | None):
    adapter = get_adapter(config.benchmark)
    manifest = manifest or resolve_manifest(config.base)
    run_dir = config.run_dir()
    return adapter, manifest, run_dir


def run_optimization(config: OursRunConfig, *, manifest: SplitManifest | None = None,
                     _preflight: bool = True) -> SSOOptimizationOutcome:
    """Beam_0 -> T OURS rounds -> Beam_T. Never reads a deployment task."""
    adapter, manifest, run_dir = _prepare(config, manifest)
    if _preflight:
        errors = ours_preflight(config, manifest, for_deployment=False)
        if errors:
            raise PreflightError("refusing to start; fix these first:\n  " + "\n  ".join(errors))

    run_dir.mkdir(parents=True, exist_ok=True)
    manifest.write(run_dir / "split_manifest.json")
    (run_dir / "ours_run_config.json").write_text(
        json.dumps(config.to_dict(), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    options = harness_options(config.base, adapter)
    for key, value in optimizer_side_env(config.base).items():
        os.environ[key] = value
    if adapter.name == "alfworld":
        os.environ["ALFWORLD_DATA"] = options["alfworld_data"]

    settings = native_settings(config)
    schedule = native_schedule(config, manifest)
    train_ids = list(manifest.task_ids("train"))
    val_ids = set(manifest.task_ids("val"))
    for i, step in enumerate(schedule):
        used = set(step["training_task_ids"]) | set(step["history_task_ids"])
        assert_no_deployment_ids(manifest, used)
        if used & val_ids or not used <= set(train_ids):
            raise PreflightError(f"native schedule round {i} names non-TRAIN ids")
    assert_no_deployment_ids(manifest, train_ids)
    (run_dir / "sso_schedule.json").write_text(
        json.dumps([{"round_id": i, "training_task_ids": s["training_task_ids"],
                     "history_task_ids": s["history_task_ids"]}
                    for i, s in enumerate(schedule)], indent=2),
        encoding="utf-8",
    )

    t0 = time.time()
    solver_config = config.solver_config()
    accumulator = PhaseCostAccumulator("optimization")
    session = SolverSession(config=solver_config, accumulator=accumulator,
                            role="sso_solver_side")
    harness = build_harness(adapter.name, work_dir=run_dir / "optimization_harness",
                            options=options)
    try:
        components = build_native_components(
            adapter=adapter,
            harness=harness,
            session=session,
            proxy_base_url=config.base.proxy_base_url,
            optimizer_model=config.base.optimizer_model,
            api_key=config.base.api_key,
            max_tokens=settings["optimizer_max_tokens"],
            temperature=settings["optimizer_temperature"],
            timeout=config.base.timeout_seconds,
            num_probes=settings["num_probes"],
            update_budget=settings["update_budget"],
        )
        tasks_by_id = {
            t.task_id: t
            for t in tasks_from_benchmark(adapter.tasks_for_ids(train_ids, role="train"))
        }
        optimizer_adapter = SSOAdapter(
            components=components,
            tasks_by_id=tasks_by_id,
            schedule=schedule,
            history_per_step=settings["history_per_step"],
            history_seed=settings["schedule_seed"],
        )
        evaluator = SharedBenchmarkEvaluator(
            adapter=adapter,
            manifest=manifest,
            solver_config=solver_config,
            accumulator=accumulator,
            harness_options=options,
            max_workers=config.validation_workers,
        )
        resources = run_resources(config, accumulator)
        controller = config.design.build_controller(
            adapter=optimizer_adapter, evaluator=evaluator,
            workspace=RunWorkspace(root=str(run_dir / "search")), seed=config.seed,
            resources=resources,
        )
        # Beam_0 = SSO's native start (S_0 = no skill), one member.
        beam_0 = native_initial_beam("sso", adapter.name)
        rounds_dir = run_dir / "rounds"
        rounds_dir.mkdir(parents=True, exist_ok=True)

        def _dump(round_result) -> None:
            (rounds_dir / f"round_{round_result.round_id:03d}.json").write_text(
                json.dumps(round_summary(round_result), indent=2,
                           ensure_ascii=False, default=str),
                encoding="utf-8",
            )

        result = run_search(controller, beam_0, train_ids,
                            n_rounds=config.rounds, on_round=_dump)
        # Beam_T -> deployment decision (final merge + Top-1 on VAL), still optimization spend
        finalize_search(config.design, controller, result, run_dir=run_dir)
    finally:
        harness.close()

    usage = optimizer_side_usage(components)
    accumulator.add(
        llm_calls=int(usage.get("calls", 0)),
        input_tokens=usage.get("prompt_tokens"),
        output_tokens=usage.get("completion_tokens"),
        cached_tokens=usage.get("cached_tokens"),
        token_source="provider_usage",
        role="sso_optimizer_side",
    )
    wall = time.time() - t0
    native_state = {
        "settings": settings,
        "schedule": [{"round_id": i, "training_task_ids": s["training_task_ids"],
                      "native_history_task_ids": s["history_task_ids"]}
                     for i, s in enumerate(schedule)],
        "schedule_source": "integration/experiment/runners/sso.py::build_schedule",
        "training_schedule": training_schedule(config).record(schedule, steps_name="rounds"),
        "initial_skill": {"condition": NATIVE_START_CONDITION["sso"],
                          "chars": len(beam_0[0].skill)},
        "train_task_ids": train_ids,
        "validation_task_ids": list(manifest.task_ids("val")),
        "n_validation_evaluations": evaluator.n_scored,
        "framework_resources": resources.describe(),
        "framework_component_cost_by_round": {r.round_id: r.component_cost for r in result.rounds},
        "optimizer_side_usage": usage,
        "native_gate": {
            "used_by_ours": False,
            "mechanism": "SSOOptimizer.step is called with validation_batch=(); "
                         "validate_candidate executes and judges nothing and its "
                         "vacuous verdict is discarded",
        },
        "step_traces_per_round": _proposal_counts(optimizer_adapter),
    }
    (run_dir / "ours_optimization_summary.json").write_text(
        json.dumps({
            "search": {**config.search_settings(),
                       "search_trace": [round_summary(r) for r in result.rounds],
                       "carry_forward": result.carry_forward_report(),
                       "final_beam": final_beam_summary(result)},
            "native": native_state,
            "optimization_cost": accumulator.snapshot().to_dict(),
        }, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return SSOOptimizationOutcome(
        search=result, optimization_cost=accumulator.snapshot(),
        native_state=native_state, wall_time_s=wall, run_dir=run_dir,
        manifest=manifest,
    )


def _proposal_counts(adapter: SSOAdapter) -> dict:
    return {
        "n_native_steps": len(adapter.step_traces),
        "n_parent_anchor_replays": sum(t["n_parent_anchors_replayed"] for t in adapter.step_traces),
        "n_native_gate_comparisons": sum(t["native_gate"]["executed_comparisons"]
                                         for t in adapter.step_traces),
    }


# ─────────────────────────────────── run ────────────────────────────────────


def run(config: OursRunConfig, *, manifest: SplitManifest | None = None) -> OursOutcome:
    """Run one SSO+OURS cell for real: optimization, deployment, artifact."""
    adapter, manifest, run_dir = _prepare(config, manifest)
    errors = ours_preflight(config, manifest, for_deployment=True)
    if errors:
        raise PreflightError("refusing to start; fix these first:\n  " + "\n  ".join(errors))

    t0 = time.time()
    opt = run_optimization(config, manifest=manifest, _preflight=False)
    result = opt.search

    # ── DEPLOYMENT: the design's rule on Beam_T; the only read of deployment ids
    options = harness_options(config.base, adapter)
    deployment = deploy_final_beam(
        design=config.design, final_beam=result.final_selection.selected, adapter=adapter,
        manifest=manifest, run_dir=run_dir, solver_config=config.solver_config(),
        harness_options=options, deployment_workers=config.base.deployment_workers,
    )
    final_skill_path = run_dir / "final_skill.md"
    final_skill_path.write_text(deployment.members[0][0].skill, encoding="utf-8")
    wall = time.time() - t0

    resolved = resolve_model_aliases(
        config.base.proxy_base_url,
        [config.base.solver_model, config.base.optimizer_model],
    )
    record = build_ours_record(
        optimizer=config.optimizer,
        method=config.method,
        benchmark=adapter.name,
        harness=config.base.harness,
        solver_model=config.base.solver_model,
        solver_model_resolved=resolved.get(config.base.solver_model),
        optimizer_model=config.base.optimizer_model,
        optimizer_model_resolved=resolved.get(config.base.optimizer_model),
        split_manifest_id=manifest.manifest_id,
        split_manifest_version=manifest.version,
        split_manifest_hash=manifest.content_hash,
        split_counts=manifest.counts(),
        **_performance_kwargs(deployment),
        optimization_cost=opt.optimization_cost,
        deployment_cost=deployment.cost,
        run_id=config.run_id,
        code_freeze=code_freeze_reference(),
        launch_command=config.base.extra.get("launch_command", ""),
        ours=_ours_section(config, result, opt.native_state, deployment),
    )
    record["experiment"]["harness_kind"] = HARNESS_KIND
    record["experiment"]["harness_options"] = options
    record["provenance"]["wall_time_s"] = round(wall, 2)
    record["provenance"]["optimization_wall_time_s"] = round(opt.wall_time_s, 2)
    record["provenance"]["deployment_wall_time_s"] = round(deployment.wall_time_s, 2)
    record["provenance"]["run_dir"] = str(run_dir)
    record["provenance"]["final_skill_path"] = str(final_skill_path)
    record["provenance"]["split_manifest_path"] = str(run_dir / "split_manifest.json")
    (run_dir / "deployment_result.json").write_text(
        json.dumps({**deployment.to_dict(),
                    "member_results": {c.candidate_id: r.to_dict() for c, r in deployment.members}},
                   indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    artifact_path = write_ours_record(config.artifact_path(), record)
    return OursOutcome(
        record=record, artifact_path=artifact_path, manifest=manifest, search=result,
        final_skill_path=final_skill_path, run_dir=run_dir, deployment=deployment,
        extras=opt.native_state,
    )


def _performance_kwargs(deployment) -> dict:
    perf = deployment.performance()
    return {k: perf[k] for k in ("metric_name", "metric_value", "n_tasks", "scorer_ref",
                                 "secondary_metrics")}


def _ours_section(config: OursRunConfig, result: SearchResult, native_state: dict,
                  deployment) -> dict:
    return {
        **config.search_settings(),
        "not_applicable_to_sso": ["evidence_scope", "resynthesise_subset_guidance"],
        "sso_knobs": resolve_sso_knobs(config.base.extra).record(),
        "search_trace": [round_summary(r) for r in result.rounds],
        "carry_forward": result.carry_forward_report(),
        "final_beam": final_beam_summary(result),
        "final_selection": result.metadata.get("final_selection"),
        "deployment": deployment.to_dict(),
        "native": native_state,
        "placeholders": {
            **design_placeholders(config),
            "subset_history": (
                "H for an evidence subset = a random sso_history_per_step of that "
                "subset's executed pairs (all if fewer): the author's native H_t "
                "rule applied to the batch the subset step runs on."
            ),
        },
    }


__all__ = [
    "OursOutcome", "SSOOptimizationOutcome",
    "native_schedule", "native_settings", "ours_preflight", "run", "run_optimization",
    "training_schedule", "training_schedule_plan",
]
