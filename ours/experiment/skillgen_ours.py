"""Run SkillGen with OURS search and shared benchmark evaluation."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from integration.benchmarks.manifest import SplitManifest
from integration.benchmarks.registry import get_adapter
from integration.common.paths import SKILLGEN_ROOT, ensure_on_sys_path
from integration.common.resolve_models import resolve_model_aliases
from integration.experiment.config import code_freeze_reference, optimizer_side_env
from integration.experiment.cost import PhaseCostAccumulator
from integration.experiment.execute import PreflightError, preflight, resolve_manifest
from integration.experiment.runners.base import RunContext
from integration.experiment.runners.skillgen import build_effective_config
from integration.experiment.runners.skills import NATIVE_START_CONDITION
from integration.experiment.runners.split_materialization import skillgen_instances
from integration.harness.base import SolverSession
from integration.harness.registry import HARNESS_KIND, build_harness
from ours.adapters.shared_benchmark import SharedBenchmarkEvaluator
from ours.adapters.skillgen import SkillGenAdapter
from ours.adapters.skillgen_native import (
    SkillGenNativeConfig,
    build_task_instances,
    reset_token_stats,
    shared_direct_context,
    task_type,
    token_stats,
)
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

ensure_on_sys_path(SKILLGEN_ROOT)


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


# ───────────────────────────────── preflight ────────────────────────────────


def ours_preflight(config: OursRunConfig, manifest: SplitManifest, *,
                   for_deployment: bool = True) -> list:
    """Every blocking problem, before a single token is spent.

    The native blockers are reused verbatim (`integration.experiment.execute
    .preflight`): the same manifest validation, the same benchmark resource
    check and the same SkillGen-specific checks a native SkillGen run faces.
    Design-level blockers come from `OursDesign.preflight`, identically for
    every optimizer.
    """
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
    return errors


# ────────────────────────────── native config ───────────────────────────────


def _opt_int(value):
    """None stays None -- it means "let SkillGen choose k itself"."""
    return None if value is None else int(value)


def build_native_config(ctx: RunContext, config: OursRunConfig) -> tuple[SkillGenNativeConfig, dict]:
    """Translate the run into SkillGen's own settings.

    `build_effective_config` is the NATIVE runner's own function: it reads
    SkillGen's checked-in `config.yaml` as a template (never writing to it),
    maps the two solver-side roles to the run's solver alias and the thirteen
    optimizer-side roles to the run's optimizer alias, and applies
    `--extra skillgen_config_overrides`. Reusing it is what guarantees a
    SkillGen+OURS run and a SkillGen-native run are configured identically.
    """
    cfg_dict = build_effective_config(ctx)
    models = cfg_dict.get("models", {})
    llm_cfg = cfg_dict.get("llm", {})
    cl = cfg_dict.get("clustering", {})
    va = cfg_dict.get("verification_analysis", {}) or {}
    ind = cfg_dict.get("induction", {}) or {}
    pipe = cfg_dict.get("pipeline", {})
    gen = cfg_dict.get("generation", {}) or {}

    if gen.get("generate_scripts"):
        raise PreflightError(
            "generation.generate_scripts is on. OURS moves skills as text, so a "
            "resource-bundle candidate would lose its scripts between the beam "
            "and deployment. Bundle support is an open design question."
        )

    native = SkillGenNativeConfig(
        solver_model=models.get("baseline_agent", ctx.config.solver_model),
        optimizer_model=models.get("induction", ctx.config.optimizer_model),
        embedding_model=(cfg_dict.get("embedding", {}) or {}).get("model", ""),
        task_type=task_type(ctx.config.extra.get("skillgen_task_type", "binary")),
        temperature=float(llm_cfg.get("temperature", ctx.config.temperature)),
        max_tokens=int(llm_cfg.get("max_tokens", ctx.config.max_tokens)),
        max_tokens_generation=int(llm_cfg.get("max_tokens_generation", 16384)),
        max_workers=int(pipe.get("max_workers", 4)),
        clustering_method=cl.get("method", "kmeans"),
        n_failure_clusters=_opt_int(cl.get("n_failure_clusters", cl.get("n_clusters"))),
        n_success_clusters=_opt_int(cl.get("n_success_clusters", cl.get("n_clusters"))),
        min_clusters=int(cl.get("min_clusters", 2)),
        max_failure_clusters=int(cl.get("max_failure_clusters", cl.get("max_clusters", 8))),
        max_success_clusters=int(cl.get("max_success_clusters", cl.get("max_clusters", 8))),
        min_cluster_size=int(cl.get("min_cluster_size", 1)),
        max_contrastive_pairs=int(ind.get("max_contrastive_pairs", 16)),
        case_analyst_workers=int(va.get("case_analyst_workers", 4)),
        case_analyst_max_tokens=int(va.get("case_analyst_max_tokens", 2048)),
        revision_synthesiser_max_tokens=int(va.get("revision_synthesiser_max_tokens", 4096)),
        max_failure_clusters_in_prompt=int(gen.get("max_failure_clusters_in_prompt", 6)),
        max_success_clusters_in_prompt=int(gen.get("max_success_clusters_in_prompt", 6)),
        max_contrastive_pairs_in_prompt=int(gen.get("max_contrastive_pairs_in_prompt", 8)),
    )
    return native, cfg_dict


# ─────────────────────────────────── run ────────────────────────────────────


def run(config: OursRunConfig, *, manifest: SplitManifest | None = None) -> OursOutcome:
    """Run one SkillGen+OURS cell for real."""
    import yaml

    adapter = get_adapter(config.benchmark)
    manifest = manifest or resolve_manifest(config.base)

    errors = ours_preflight(config, manifest)
    if errors:
        raise PreflightError("refusing to start; fix these first:\n  " + "\n  ".join(errors))

    run_dir = config.run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest.write(run_dir / "split_manifest.json")
    (run_dir / "ours_run_config.json").write_text(
        json.dumps(config.to_dict(), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    ctx = RunContext(
        config=config.base,
        adapter=adapter,
        manifest=manifest,
        run_dir=run_dir,
        solver_config=config.solver_config(),
        harness_options=harness_options(config.base, adapter),
    )
    for key, value in optimizer_side_env(config.base).items():
        os.environ[key] = value
    if adapter.name == "alfworld":
        os.environ["ALFWORLD_DATA"] = ctx.harness_options["alfworld_data"]

    native_cfg, effective = build_native_config(ctx, config)
    (run_dir / "skillgen_effective_config.yaml").write_text(
        yaml.safe_dump(effective, sort_keys=False), encoding="utf-8",
    )

    # ── TRAIN instances only. Validation is the evaluator's job and
    #    deployment is not addressable from here at all. ──────────────────
    rows = skillgen_instances(manifest, roles=("train",))
    assert_no_deployment_ids(manifest, [r["instance_id"] for r in rows])
    instances = build_task_instances(rows)
    (run_dir / "ours_train_instances.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    t0 = time.time()
    optimization = PhaseCostAccumulator("optimization")
    result, native_state = _optimize(
        config=config, ctx=ctx, adapter=adapter, manifest=manifest,
        native_cfg=native_cfg, instances=instances, accumulator=optimization,
        run_dir=run_dir,
    )
    optimization_cost = optimization.snapshot()
    optimization_wall = time.time() - t0

    # ── DEPLOYMENT: the design's rule on Beam_T; the only read of deployment ids
    deployment = deploy_final_beam(
        design=config.design, final_beam=result.final_selection.selected, adapter=adapter,
        manifest=manifest, run_dir=run_dir, solver_config=config.solver_config(),
        harness_options=ctx.harness_options,
        deployment_workers=config.base.deployment_workers,
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
        optimization_cost=optimization_cost,
        deployment_cost=deployment.cost,
        run_id=config.run_id,
        code_freeze=code_freeze_reference(),
        launch_command=config.base.extra.get("launch_command", ""),
        secondary_metrics=perf["secondary_metrics"],
        ours=_ours_section(config, result, native_state, deployment),
    )
    record["experiment"]["harness_kind"] = HARNESS_KIND
    record["experiment"]["harness_options"] = ctx.harness_options
    record["provenance"]["wall_time_s"] = round(wall, 2)
    record["provenance"]["optimization_wall_time_s"] = round(optimization_wall, 2)
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
        record=record,
        artifact_path=artifact_path,
        manifest=manifest,
        search=result,
        final_skill_path=final_skill_path,
        run_dir=run_dir,
        deployment=deployment,
        extras=native_state,
    )


def _optimize(*, config, ctx, adapter, manifest, native_cfg, instances,
              accumulator, run_dir) -> tuple[SearchResult, dict]:
    """Beam_0 + T rounds, all of it inside the OPTIMIZATION phase."""
    train_ids = [i.instance_id for i in instances]
    session = SolverSession(
        config=ctx.solver_config, accumulator=accumulator,
        role="skillgen_solver_side",
    )
    harness = build_harness(
        adapter.name, work_dir=ctx.workdir("optimization_harness"),
        options=ctx.harness_options,
    )
    reset_token_stats()

    try:
        with shared_direct_context(adapter=adapter, harness=harness, session=session):
            # The fixed no-skill TRAIN reference is NOT collected here: it is
            # Beam_0's evidence, collected by the adapter in round 0.
            optimizer_adapter = SkillGenAdapter(
                cfg=native_cfg,
                instances=instances,
                evidence_scope=config.evidence_scope,
                resynthesise_subset_guidance=config.resynthesise_subset_guidance,
            )
            evaluator = SharedBenchmarkEvaluator(
                adapter=adapter,
                manifest=manifest,
                solver_config=ctx.solver_config,
                accumulator=accumulator,
                harness_options=ctx.harness_options,
                max_workers=config.validation_workers,
            )
            resources = run_resources(config, accumulator)
            controller = config.design.build_controller(
                adapter=optimizer_adapter, evaluator=evaluator,
                workspace=RunWorkspace(root=str(run_dir / "search")), seed=config.seed,
                resources=resources,
            )
            # Beam_0 = SkillGen's native start (no skill), one member.
            beam_0 = native_initial_beam("skillgen", adapter.name)
            rounds_dir = run_dir / "rounds"
            rounds_dir.mkdir(parents=True, exist_ok=True)

            def _dump(round_result) -> None:
                path = rounds_dir / f"round_{round_result.round_id:03d}.json"
                path.write_text(
                    json.dumps(round_summary(round_result), indent=2,
                               ensure_ascii=False, default=str),
                    encoding="utf-8",
                )

            result = run_search(
                controller, beam_0, train_ids,
                n_rounds=config.rounds, on_round=_dump,
            )
            # Beam_T -> deployment decision (final merge + Top-1 on VAL), still optimization spend
            finalize_search(config.design, controller, result, run_dir=run_dir)
    finally:
        harness.close()

    # SkillGen's own tracker covers only its optimizer-side calls (induction,
    # generation, refinement, case analyst, revision synthesiser, embeddings).
    # Solver-side calls went through SolverSession and are already counted, so
    # nothing here is double counted.
    stage_rows = token_stats()
    (run_dir / "skillgen_token_stats.json").write_text(
        json.dumps(stage_rows, indent=2), encoding="utf-8",
    )
    for row in stage_rows:
        accumulator.add(
            llm_calls=int(row.get("calls", 0) or 0),
            input_tokens=row.get("prompt_tokens"),
            output_tokens=row.get("completion_tokens"),
            cached_tokens=row.get("cached_tokens"),
            token_source="provider_usage",
            role=f"skillgen_optimizer_side:{row.get('stage', 'unknown')}",
        )

    baseline = optimizer_adapter.baseline
    native_state = {
        "initial_skill": {"condition": NATIVE_START_CONDITION["skillgen"],
                          "chars": len(beam_0[0].skill)},
        "no_skill_baseline": baseline.summary() if baseline is not None else None,
        "run_level_analysis": {
            "analysis_id": getattr(optimizer_adapter.run_analysis, "analysis_id", None),
            "rule": "Induce analysis once from the no-skill TRAIN baseline and reuse it for skilled-parent refinement.",
        },
        "analyses": [
            {"analysis_id": a_id,
             "n_failure_clusters": len(getattr(a, "failure_clusters", []) or []),
             "n_success_clusters": len(getattr(a, "success_clusters", []) or [])}
            for a_id, a in optimizer_adapter.analyses.items()
        ],
        "train_task_ids": list(train_ids),
        "validation_task_ids": list(manifest.task_ids("val")),
        "n_validation_evaluations": evaluator.n_scored,
        "framework_resources": resources.describe(),
        "framework_component_cost_by_round": {r.round_id: r.component_cost for r in result.rounds},
        "token_stats_path": str(run_dir / "skillgen_token_stats.json"),
    }
    return result, native_state


# ───────────────────────────── artifact sections ────────────────────────────


def _ours_section(config: OursRunConfig, result: SearchResult, native_state: dict,
                  deployment) -> dict:
    native_gate = []
    for r in result.rounds:
        for e in r.evidence:
            if "native_gate_passed" not in e.metadata:
                continue  # a no-skill Parent has no skill for the gate to judge
            native_gate.append({
                "round_id": r.round_id,
                "parent_id": e.parent_id,
                "net_gain": e.metadata.get("net_gain"),
                "passed": e.metadata.get("native_gate_passed"),
            })
    return {
        **config.search_settings(),
        "search_trace": [round_summary(r) for r in result.rounds],
        "carry_forward": result.carry_forward_report(),
        "final_beam": final_beam_summary(result),
        "final_selection": result.metadata.get("final_selection"),
        "deployment": deployment.to_dict(),
        "native": native_state,
        # Collected for reporting only. OURS never acts on it: the search
        # policy is the global Top-M above.
        "native_gate_observed": native_gate,
        "placeholders": {
            **design_placeholders(config),
            "evidence_scope": (
                f"{config.evidence_scope} -- which of a skilled Parent's train tasks "
                "the partitioner may cluster is OPEN."
            ),
            "subset_guidance": (
                "each subset's revision guidance is re-synthesised from that "
                "subset's own case analyses; whether to re-synthesise, inherit "
                "or drop it is OPEN."
            ),
        },
    }


__all__ = ["LeakageError", "OursOutcome", "build_native_config", "ours_preflight", "run"]
