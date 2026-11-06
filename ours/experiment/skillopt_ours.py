"""Run SkillOpt with OURS search and shared benchmark evaluation."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from integration.benchmarks.manifest import SplitManifest
from integration.benchmarks.registry import get_adapter
from integration.common.resolve_models import resolve_model_aliases
from integration.experiment.config import code_freeze_reference
from integration.experiment.cost import PhaseCost, PhaseCostAccumulator
from integration.experiment.execute import PreflightError, preflight, resolve_manifest
from integration.experiment.runners.base import RunContext
from integration.experiment.runners.skills import NATIVE_START_CONDITION
from integration.harness.registry import HARNESS_KIND
from ours.experiment.common import (
    LeakageError,
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

SPLIT_DIRNAME = "skillopt_split"
NATIVE_OUT_DIRNAME = "skillopt_out"


@dataclass
class SkillOptOptimizationOutcome:
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


def _truthy(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _context(config: OursRunConfig, adapter, manifest: SplitManifest) -> RunContext:
    return RunContext(
        config=config.base,
        adapter=adapter,
        manifest=manifest,
        run_dir=config.run_dir(),
        solver_config=config.solver_config(),
        harness_options=harness_options(config.base, adapter),
    )


def native_config(config: OursRunConfig, adapter, manifest: SplitManifest) -> tuple[Path, list, dict, dict]:
    """``(config_path, cfg_options, env_updates, flat_cfg)`` for this run -- the
    native runner's launch inputs and what SkillOpt's own loader makes of them.
    Pure: nothing is written except SkillOpt's config override files, which the
    native runner writes too."""
    from ours.adapters.skillopt_native import load_native_cfg, native_launch_inputs

    run_dir = config.run_dir()
    ctx = _context(config, adapter, manifest)
    path, options, env = native_launch_inputs(
        ctx, split_dir=run_dir / SPLIT_DIRNAME, out_root=run_dir / NATIVE_OUT_DIRNAME,
    )
    return path, options, env, load_native_cfg(path, options, run_dir / NATIVE_OUT_DIRNAME)


# ─────────────────────────────── preflight ──────────────────────────────────


def ours_preflight(config: OursRunConfig, manifest: SplitManifest, *,
                   for_deployment: bool = True) -> list:
    """Every blocking problem, before a single token is spent.

    The native SkillOpt blockers are reused verbatim through the shared
    `preflight` (manifest validation, benchmark resources, SkillOpt's own
    checks); design-level blockers come from `OursDesign.preflight`,
    identically for every optimizer.
    """
    from ours.adapters.skillopt_native import unsupported_native_settings

    errors = list(preflight(config.base, manifest))
    if not manifest.task_ids("val"):
        errors.append(
            "manifest validation split is empty; OURS selects on validation and "
            "has nothing to rank with"
        )
    design_errors = config.design.preflight()
    if not for_deployment:
        rule_errors = set(config.design.build_deployment_rule().check(config.beam_m))
        design_errors = [e for e in design_errors if e not in rule_errors]
    errors.extend(design_errors)
    if "skillopt_num_epochs" in config.base.extra:
        errors.append(
            "--extra skillopt_num_epochs sets the NATIVE step count, which has no "
            "meaning for SkillOpt+OURS: the number of outer rounds is --rounds and "
            "every round is one stateless native step. Remove skillopt_num_epochs."
        )
    if "skillopt_batch_size" in config.base.extra:
        errors.append(
            "--extra skillopt_batch_size sets the NATIVE per-step rollout batch; under "
            "SkillOpt+OURS every round's evidence is the whole TRAIN split (Method "
            "decision). Remove skillopt_batch_size."
        )
    if errors:
        return errors
    try:
        _, _, _, flat = native_config(config, get_adapter(config.benchmark), manifest)
        errors.extend(unsupported_native_settings(flat))
    except (TypeError, ValueError, KeyError, FileNotFoundError) as exc:
        errors.append(f"native SkillOpt config invalid: {exc}")
    return errors


# ─────────────────────────────── optimization ───────────────────────────────


def _write_train_only_split(manifest: SplitManifest, run_dir: Path) -> Path:
    """SkillOpt's split directory with TRAIN rows only.

    Rows come from the SAME builder the native runner uses
    (`materialize_skillopt_split`); ``val/`` and ``test/`` are then emptied, so
    SkillOpt's loader cannot hand a validation or deployment item to anything.
    """
    from integration.experiment.runners.split_materialization import materialize_skillopt_split

    split_dir = materialize_skillopt_split(manifest, run_dir / SPLIT_DIRNAME)
    for name in ("val", "test"):
        (split_dir / name / "items.json").write_text("[]", encoding="utf-8")
    rows = json.loads((split_dir / "train" / "items.json").read_text(encoding="utf-8"))
    ids = [str(r.get("id")) for r in rows]
    assert_no_deployment_ids(manifest, ids)
    if sorted(ids) != sorted(manifest.task_ids("train")):
        raise LeakageError("SkillOpt train split does not equal the manifest TRAIN ids")
    return split_dir


def run_optimization(config: OursRunConfig, *, manifest: SplitManifest | None = None,
                     _preflight: bool = True) -> SkillOptOptimizationOutcome:
    """Beam_0 -> T OURS rounds -> Beam_T. Never reads a deployment task."""
    from ours.adapters.shared_benchmark import SharedBenchmarkEvaluator
    from ours.adapters.skillopt import SkillOptAdapter
    from ours.adapters.skillopt_native import (
        ONE_STEP_TOTAL_STEPS,
        apply_native_environment,
        configure_native_runtime,
        inflight_rollouts_at_boundary,
        cross_round_state,
        redacted_cfg,
        tracker_summary,
    )

    adapter = get_adapter(config.benchmark)
    manifest = manifest or resolve_manifest(config.base)
    run_dir = config.run_dir()
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
    if adapter.name == "alfworld":
        os.environ["ALFWORLD_DATA"] = options["alfworld_data"]

    _write_train_only_split(manifest, run_dir)
    config_path, cfg_options, env_updates, flat = native_config(config, adapter, manifest)
    applied_env = apply_native_environment(env_updates)
    (run_dir / "skillopt_launch.json").write_text(
        json.dumps({
            "config": str(config_path),
            "cfg_options": cfg_options,
            "env_redacted": {k: ("***" if "API_KEY" in k else v) for k, v in env_updates.items()},
            "source": "ours/adapters/skillopt_native.py::native_launch_inputs "
                      "(== integration/experiment/runners/skillopt.py launch inputs)",
        }, indent=2),
        encoding="utf-8",
    )

    t0 = time.time()
    solver_config = config.solver_config()
    accumulator = PhaseCostAccumulator("optimization")
    runtime = configure_native_runtime(flat)  # resets SkillOpt's tracker
    (run_dir / "skillopt_effective_config.json").write_text(
        json.dumps(redacted_cfg(runtime.cfg), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    beam_0 = native_initial_beam("skillopt", adapter.name)
    native_init = Path(runtime.skill_init_path).read_text(encoding="utf-8")
    if beam_0[0].skill != native_init:
        raise PreflightError(
            "Beam_0 differs from the native env.skill_init file "
            f"({runtime.skill_init_path}); runners/skills.py and the native config disagree"
        )

    train_pool = list(runtime.dataloader.train_items)
    train_ids = [str(i.get("id")) for i in train_pool]
    assert_no_deployment_ids(manifest, train_ids)
    if set(train_ids) & set(manifest.task_ids("val")):
        raise LeakageError("SkillOpt's TRAIN pool contains validation ids")


    epoch_protocol = None
    if _truthy(config.base.extra.get("skillopt_epoch_protocol", False)):
        from ours.adapters.skillopt_epoch import NativeEpochProtocol

        epoch_protocol = NativeEpochProtocol.from_runtime(runtime, train_pool_ids=frozenset(train_ids))

    optimizer_adapter = SkillOptAdapter(
        env_adapter=runtime.env_adapter,
        scheduler=None,
        one_step_scheduler_factory=runtime.scheduler_factory,
        cfg=runtime.cfg,
        update_mode=runtime.update_mode,
        lr_control_mode=runtime.lr_control_mode,
        merge_batch_size=runtime.merge_batch_size,
        analyst_workers=runtime.analyst_workers,
        rewrite_reasoning_effort=runtime.rewrite_reasoning_effort,
        rewrite_max_completion_tokens=runtime.rewrite_max_completion_tokens,
        round_batches=runtime.batch_for_round,
        env_out_root=runtime.out_root,
        epoch_protocol=epoch_protocol,
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
    rounds_dir = run_dir / "rounds"
    rounds_dir.mkdir(parents=True, exist_ok=True)

    def _dump(round_result) -> None:
        (rounds_dir / f"round_{round_result.round_id:03d}.json").write_text(
            json.dumps(round_summary(round_result), indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    result = run_search(controller, beam_0, train_pool, n_rounds=config.rounds, on_round=_dump)
    # Beam_T -> deployment decision (final merge + Top-1 on VAL), still optimization spend
    finalize_search(config.design, controller, result, run_dir=run_dir)

    # SkillOpt's own tracker covers every call SkillOpt made itself -- the
    # native TRAIN rollouts (target role) and reflection / aggregation /
    # rewriting (optimizer role). Validation went through the shared harness's
    # SolverSession and is already counted; the two never overlap.
    tracker = tracker_summary()
    inflight = inflight_rollouts_at_boundary(str(runtime.cfg.get("env", "")))
    (run_dir / "skillopt_token_summary.json").write_text(json.dumps(tracker, indent=2), encoding="utf-8")
    for stage, bucket in sorted(tracker.items()):
        if stage == "_total" or not bucket.get("calls"):
            continue
        accumulator.adopt_native_totals(
            llm_calls=int(bucket.get("calls", 0) or 0),
            input_tokens=bucket.get("prompt_tokens"),
            output_tokens=bucket.get("completion_tokens"),
            cached_tokens=bucket.get("cached_tokens"),
            role=f"skillopt_native_tracker:{stage}",
        )
    optimization_cost = accumulator.snapshot()
    wall = time.time() - t0

    native_state = {
        "launch": {"config": str(config_path), "cfg_options": cfg_options,
                   "source": "native runner launch inputs, loaded by SkillOpt's own load_config"},
        "backend_state_applied": applied_env,
        "recipe": {
            "train_size": runtime.train_size, "batch_size": runtime.batch_size,
            "steps_per_epoch": runtime.steps_per_epoch,
            "update_mode": runtime.update_mode, "lr_control_mode": runtime.lr_control_mode,
            "merge_batch_size": runtime.merge_batch_size, "analyst_workers": runtime.analyst_workers,
            "seed": runtime.seed,
        },
        "round_batches": list(runtime.batch_log),
        "round_batch_source": "batch t of SkillOpt's dataloader.plan_train_epoch (TRAIN data "
                              "sampling only; advances no scheduler, carries no optimizer state)",
        "proposal_budget": {
            "rule": "every OURS round resolves the native one-step edit budget on a fresh "
                    "scheduler (total_steps=1); T is OURS search depth, not budget decay",
            "lr_scheduler": runtime.cfg.get("lr_scheduler"),
            "edit_budget": runtime.cfg.get("edit_budget"),
            "min_edit_budget": runtime.cfg.get("min_edit_budget"),
            "one_step_scheduler_total_steps": ONE_STEP_TOTAL_STEPS,
            "one_step_budget": runtime.one_step_budget(),
            "budget_by_round": dict(optimizer_adapter.budget_by_round),
            "identical_across_rounds": len({json.dumps(b, sort_keys=True) for b
                                            in optimizer_adapter.budget_by_round.values()}) <= 1,
        },
        "scheduler_ticks_by_round": dict(optimizer_adapter.scheduler_ticks_by_round),
        "initial_skill": {"condition": NATIVE_START_CONDITION["skillopt"],
                          "path": runtime.skill_init_path, "chars": len(native_init)},
        "cross_round_state": cross_round_state(
            runtime.cfg, protocol=epoch_protocol.describe() if epoch_protocol is not None else None),
        "epoch_protocol": (
            {**epoch_protocol.describe(), "reports_by_round": list(epoch_protocol.reports)}
            if epoch_protocol is not None else
            {"enabled": False, "note": "Stateless native one-step proposer."}),
        "native_gate": {"used_by_ours": False,
                        "mechanism": "skillopt.evaluation.gate.evaluate_gate is never called"},
        "train_task_ids": train_ids,
        "validation_task_ids": list(manifest.task_ids("val")),
        "n_validation_evaluations": evaluator.n_scored,
        "framework_resources": resources.describe(),
        "framework_component_cost_by_round": {r.round_id: r.component_cost for r in result.rounds},
        "skillopt_token_summary": tracker,
        "inflight_rollouts_at_phase_boundary": inflight,
    }
    (run_dir / "ours_optimization_summary.json").write_text(
        json.dumps({
            "search": {**config.search_settings(),
                       "search_trace": [round_summary(r) for r in result.rounds],
                       "carry_forward": result.carry_forward_report(),
                       "final_beam": final_beam_summary(result)},
            "native": native_state,
            "optimization_cost": optimization_cost.to_dict(),
        }, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return SkillOptOptimizationOutcome(
        search=result, optimization_cost=optimization_cost, native_state=native_state,
        wall_time_s=wall, run_dir=run_dir, manifest=manifest,
    )


# ─────────────────────────────────── run ────────────────────────────────────


def run(config: OursRunConfig, *, manifest: SplitManifest | None = None) -> OursOutcome:
    """Run one SkillOpt+OURS cell for real: optimization, deployment, artifact."""
    adapter = get_adapter(config.benchmark)
    manifest = manifest or resolve_manifest(config.base)
    run_dir = config.run_dir()
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
    perf = deployment.performance()
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
        metric_name=perf["metric_name"],
        metric_value=perf["metric_value"],
        n_tasks=perf["n_tasks"],
        scorer_ref=perf["scorer_ref"],
        secondary_metrics=perf["secondary_metrics"],
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


def _ours_section(config: OursRunConfig, result: SearchResult, native_state: dict,
                  deployment) -> dict:
    return {
        **config.search_settings(),
        "not_applicable_to_skillopt": ["evidence_scope", "resynthesise_subset_guidance"],
        "search_trace": [round_summary(r) for r in result.rounds],
        "carry_forward": result.carry_forward_report(),
        "final_beam": final_beam_summary(result),
        "final_selection": result.metadata.get("final_selection"),
        "deployment": deployment.to_dict(),
        "native": native_state,
        "placeholders": dict(design_placeholders(config)),
    }


__all__ = [
    "OursOutcome", "SkillOptOptimizationOutcome", "native_config",
    "ours_preflight", "run", "run_optimization",
]
