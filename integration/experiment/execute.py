"""Run native optimization, evaluate the selected skill, and record results."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from integration.benchmarks.manifest import SplitManifest
from integration.benchmarks.registry import get_adapter
from integration.benchmarks.user_manifest import load_split_manifest, validate_manifest
from integration.common.resolve_models import resolve_model_aliases
from integration.experiment.config import ExperimentConfig, code_freeze_reference, optimizer_side_env
from integration.experiment.cost import PhaseCost
from integration.experiment.deployment import run_deployment
from integration.experiment.record import build_record, write_record
from integration.experiment.runners import RunContext, get_runner
from integration.harness.registry import HARNESS_KIND, build_harness, max_parallel_tasks


class PreflightError(RuntimeError):
    """Raised before any LLM call when a run cannot legitimately proceed."""


@dataclass
class ExecutionOutcome:
    record: dict
    artifact_path: Path
    manifest: SplitManifest


def resolve_manifest(config: ExperimentConfig) -> SplitManifest:
    """The manifest governing this run: the user's, or the checked-in default."""
    if config.split_manifest_path:
        return load_split_manifest(config.split_manifest_path)
    return get_adapter(config.benchmark).manifest()


def preflight(config: ExperimentConfig, manifest: SplitManifest) -> list:
    """Every blocking problem, collected before a single token is spent."""
    errors: list = []
    adapter = get_adapter(config.benchmark)

    if manifest.benchmark != adapter.name:
        errors.append(
            f"manifest is for {manifest.benchmark!r} but the run requested {adapter.name!r}"
        )
        return errors

    errors.extend(validate_manifest(manifest, benchmark=config.benchmark))

    status = adapter.resource_status()
    errors.extend(status.blocking)

    ctx = _context(config, manifest, adapter, dry=True)
    errors.extend(type(get_runner(config.optimizer)).preflight(ctx))
    return errors


def _context(config: ExperimentConfig, manifest: SplitManifest, adapter, *, dry: bool = False) -> RunContext:
    run_dir = config.run_dir()
    if not dry:
        run_dir.mkdir(parents=True, exist_ok=True)
    harness_options = dict(config.harness_options)
    if adapter.name == "alfworld":
        from integration.benchmarks.alfworld import alfworld_data_root

        harness_options.setdefault("alfworld_data", alfworld_data_root())
    return RunContext(
        config=config,
        adapter=adapter,
        manifest=manifest,
        run_dir=run_dir,
        solver_config=config.solver_config(),
        harness_options=harness_options,
    )


def execute(config: ExperimentConfig, *, manifest: SplitManifest | None = None) -> ExecutionOutcome:
    """Run one cell of the matrix for real."""
    adapter = get_adapter(config.benchmark)
    manifest = manifest or resolve_manifest(config)

    errors = preflight(config, manifest)
    if errors:
        raise PreflightError(
            "refusing to start; fix these first:\n  " + "\n  ".join(errors)
        )

    ctx = _context(config, manifest, adapter)
    run_dir = ctx.run_dir
    manifest.write(run_dir / "split_manifest.json")

    # Optimizer-side clients that read their endpoint from the environment
    # (SkillGen's llm.py) must point at the same proxy as the solver.
    for key, value in optimizer_side_env(config).items():
        os.environ[key] = value
    if adapter.name == "alfworld":
        os.environ["ALFWORLD_DATA"] = ctx.harness_options["alfworld_data"]

    t0 = time.time()
    runner = get_runner(config.optimizer)
    optimization = runner.optimize(ctx)

    skill_path = run_dir / "final_skill.md"
    skill_path.write_text(optimization.final_skill_text, encoding="utf-8")

    deployment_harness = build_harness(
        adapter.name, work_dir=ctx.workdir("deployment_harness"), options=ctx.harness_options
    )
    try:
        deployment = run_deployment(
            adapter=adapter,
            harness=deployment_harness,
            task_ids=manifest.task_ids("deployment"),
            skill_text=optimization.final_skill_text,
            solver_config=config.solver_config(),
            max_workers=min(int(config.deployment_workers), max_parallel_tasks(adapter.name)),
            transcripts_path=run_dir / "deployment_transcripts.jsonl",
        )
    finally:
        deployment_harness.close()
    wall = time.time() - t0

    resolved = resolve_model_aliases(
        config.proxy_base_url,
        [config.solver_model] + ([config.optimizer_model] if config.optimizer_model else []),
    )

    record = build_record(
        optimizer=config.optimizer,
        benchmark=adapter.name,
        harness=config.harness,
        solver_model=config.solver_model,
        solver_model_resolved=resolved.get(config.solver_model),
        split_manifest_id=manifest.manifest_id,
        split_manifest_version=manifest.version,
        split_manifest_hash=manifest.content_hash,
        split_role="deployment",
        metric_name=deployment.result.metric_name,
        metric_value=deployment.result.metric_value,
        n_tasks=deployment.result.n_tasks,
        scorer_ref=deployment.result.scorer_ref,
        optimization_cost=optimization.cost,
        deployment_cost=deployment.cost,
        run_id=config.run_id,
        code_freeze=code_freeze_reference(),
        launch_command=config.extra.get("launch_command", ""),
        secondary_metrics=deployment.result.secondary_metrics,
        notes=[
            "final metric computed by the benchmark's official scorer on the "
            "deployment split, via the shared Direct harness",
        ],
    )
    record["experiment"]["split_counts"] = manifest.counts()
    record["experiment"]["optimizer_model"] = config.optimizer_model
    record["experiment"]["optimizer_model_resolved"] = resolved.get(config.optimizer_model)
    record["experiment"]["harness_kind"] = HARNESS_KIND
    record["experiment"]["harness_options"] = ctx.harness_options
    record["optimizer_internal_gate"] = {
        **optimization.internal_gate,
        "skill_accepted": optimization.skill_accepted,
        "is_benchmark_metric": False,
    }
    record["provenance"]["wall_time_s"] = round(wall, 2)
    record["provenance"]["optimization_wall_time_s"] = round(optimization.wall_time_s, 2)
    record["provenance"]["deployment_wall_time_s"] = round(deployment.wall_time_s, 2)
    record["provenance"]["run_dir"] = str(run_dir)
    record["provenance"]["final_skill_path"] = str(skill_path)
    record["provenance"]["split_manifest_path"] = str(run_dir / "split_manifest.json")
    record["provenance"]["optimizer_native_summary"] = optimization.native_summary

    (run_dir / "deployment_result.json").write_text(
        json.dumps(deployment.to_dict(), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    artifact_path = write_record(config.artifact_path(), record)
    return ExecutionOutcome(record=record, artifact_path=artifact_path, manifest=manifest)


def empty_phase_cost() -> PhaseCost:
    from integration.experiment.cost import empty_cost

    return empty_cost()
