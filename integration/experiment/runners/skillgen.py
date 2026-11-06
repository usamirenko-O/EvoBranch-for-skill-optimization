"""Run the native SkillGen pipeline on shared benchmark splits."""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import yaml

from integration.common.paths import SKILLGEN_ROOT, WORKSPACE_ROOT, ensure_on_sys_path
from integration.experiment.cost import PhaseCostAccumulator
from integration.experiment.runners.base import OptimizationResult, OptimizerRunner, RunContext
from integration.experiment.runners.skills import NATIVE_START_CONDITION
from integration.experiment.runners.split_materialization import skillgen_instances
from integration.harness.base import SolverSession
from integration.harness.registry import build_harness, max_parallel_tasks

ensure_on_sys_path(SKILLGEN_ROOT)

#: SkillGen model roles that execute the benchmark task (solver side).
_SOLVER_ROLES = ("baseline_agent", "verification_agent")
#: Everything else is optimizer-side machinery.
_OPTIMIZER_ROLES = (
    "default",
    "baseline_judge",
    "induction",
    "induction_contextual",
    "induction_summary",
    "induction_pattern",
    "induction_contrastive",
    "generation_plan",
    "generation_execute",
    "refinement",
    "verification_judge",
    "verification_case_analyst",
    "verification_revision_synthesiser",
)


def build_effective_config(ctx: RunContext) -> dict:
    """SkillGen's own config.yaml as a template, with explicit overrides."""
    template = yaml.safe_load((SKILLGEN_ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg = copy.deepcopy(template)
    run_cfg = ctx.config
    extra = run_cfg.extra

    models = cfg.setdefault("models", {})
    for role in _SOLVER_ROLES:
        models[role] = run_cfg.solver_model
    optimizer_alias = run_cfg.optimizer_model
    for role in _OPTIMIZER_ROLES:
        models[role] = optimizer_alias

    cfg.setdefault("llm", {})["max_tokens"] = int(run_cfg.max_tokens)
    cfg["llm"]["temperature"] = float(run_cfg.temperature)

    # SkillGen's induction stage embeds trajectory summaries before clustering.
    # Its config ships an OpenAI model name ("text-embedding-3-small") that is
    # meaningless to our proxy, and `SKILLGEN_EMBEDDING_BASE_URL` only redirects
    # *where* the call goes, not what it asks for -- so the proxy must expose an
    # alias by this name, or the run must override it here.
    embedding_alias = extra.get("skillgen_embedding_model")
    if embedding_alias:
        cfg.setdefault("embedding", {})["model"] = str(embedding_alias)

    cfg.setdefault("pipeline", {})["artifact_root"] = str(ctx.workdir("skillgen_artifacts"))
    cfg.setdefault("skill_output", {})["path"] = str(ctx.workdir("skillgen_skills"))

    for key, value in (extra.get("skillgen_config_overrides") or {}).items():
        section, _, field = key.partition(".")
        if field:
            cfg.setdefault(section, {})[field] = value
        else:
            cfg[section] = value

    # Applied last, so no override can raise it: SkillGen fans its baseline and
    # verification rollouts out over a thread pool, and the shared harness caps
    # how many tasks a benchmark may run at once (1 for ALFWorld, whose
    # TextWorld games are not thread-safe and would corrupt each other rather
    # than fail loudly).
    pipeline_cfg = cfg["pipeline"]
    pipeline_cfg["max_workers"] = min(
        int(pipeline_cfg.get("max_workers", 16)), max_parallel_tasks(ctx.adapter.name)
    )
    return cfg


class SkillGenRunner(OptimizerRunner):
    optimizer = "skillgen"

    @staticmethod
    def preflight(ctx: RunContext) -> list:
        errors = []
        embedding_model = ctx.config.extra.get("skillgen_embedding_model")
        if not embedding_model:
            template = yaml.safe_load((SKILLGEN_ROOT / "config.yaml").read_text(encoding="utf-8"))
            default_embedding = (template.get("embedding") or {}).get("model", "")
            errors.append(
                "SkillGen's induction stage calls an embeddings endpoint with model "
                f"{default_embedding!r}, which is an OpenAI model name, not a proxy alias. "
                "Pass --extra skillgen_embedding_model=<alias your proxy serves> "
                "(or =" + repr(default_embedding) + " if the proxy really does expose that name)."
            )
        if not (SKILLGEN_ROOT / "pipeline.py").is_file():
            errors.append("missing SkillGen/pipeline.py")
        if not (SKILLGEN_ROOT / "benchmarks" / "shared_direct_adapter.py").is_file():
            errors.append("missing SkillGen/benchmarks/shared_direct_adapter.py")
        if not ctx.train_ids:
            errors.append("manifest train split is empty; SkillGen has no failures to learn from")
        return errors

    def optimize(self, ctx: RunContext) -> OptimizationResult:
        import llm as skillgen_llm  # SkillGen-internal
        from benchmarks.shared_direct_adapter import (
            SharedDirectContext,
            clear_context,
            install_context,
        )
        from models import TaskInstance, TaskType
        from pipeline import run_pipeline

        cfg_dict = build_effective_config(ctx)
        effective_path = ctx.run_dir / "skillgen_effective_config.yaml"
        effective_path.write_text(yaml.safe_dump(cfg_dict, sort_keys=False), encoding="utf-8")

        rows = skillgen_instances(ctx.manifest, roles=("train",))
        if {r["instance_id"] for r in rows} != set(ctx.train_ids):
            raise ValueError("SkillGen's instance pool must be exactly the manifest TRAIN ids")
        pool_ids = {r["instance_id"] for r in rows}
        leaked = pool_ids & set(ctx.deployment_ids)
        if leaked:
            raise ValueError(
                f"deployment ids leaked into SkillGen's instance pool: {sorted(leaked)[:5]}"
            )
        instances = [
            TaskInstance(
                instance_id=r["instance_id"],
                input=r["input"],
                ground_truth=r["ground_truth"],
                metadata=r["metadata"],
            )
            for r in rows
        ]
        (ctx.run_dir / "skillgen_instances.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        accumulator = PhaseCostAccumulator("optimization")
        session = SolverSession(
            config=ctx.solver_config, accumulator=accumulator, role="skillgen_solver_side"
        )
        harness = build_harness(
            ctx.adapter.name, work_dir=ctx.workdir("skillgen_harness"), options=ctx.harness_options
        )
        install_context(
            SharedDirectContext(adapter=ctx.adapter, harness=harness, session=session)
        )

        skillgen_llm.reset_token_stats()
        t0 = time.time()
        try:
            skill = run_pipeline(
                instances,
                TaskType(ctx.config.extra.get("skillgen_task_type", "binary")),
                config_path=str(effective_path),
                dataset_id=f"{ctx.adapter.name}__{ctx.manifest.manifest_id}",
                task_name=ctx.adapter.name,
                dataset_metadata={
                    "split_manifest_id": ctx.manifest.manifest_id,
                    "split_manifest_hash": ctx.manifest.content_hash,
                },
            )
        finally:
            clear_context()
            harness.close()
        wall = time.time() - t0

        stage_rows = skillgen_llm.get_token_stats()
        (ctx.run_dir / "skillgen_token_stats.json").write_text(
            json.dumps(stage_rows, indent=2), encoding="utf-8"
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

        skill_body = (getattr(skill, "body", "") or "") if skill is not None else ""
        status = getattr(getattr(skill, "status", None), "value", None) if skill is not None else None
        accepted = bool(skill_body.strip()) and status == "active"

        skill_path = None
        if skill_body.strip():
            skill_path = ctx.run_dir / "skillgen_final_skill.md"
            skill_path.write_text(skill_body, encoding="utf-8")

        return OptimizationResult(
            # SkillGen's native contract: a skill its verification gate did not
            # accept is not deployed. `eval_skill.py::main` refuses to run a
            # skill-conditioned evaluation for a DEPRECATED or empty skill, so
            # deployment here falls back to the no-skill condition -- which is
            # exactly what SkillGen itself does, not something invented here.
            final_skill_text=skill_body if accepted else "",
            cost=accumulator.snapshot(),
            wall_time_s=wall,
            skill_accepted=accepted,
            final_skill_path=str(skill_path) if skill_path else None,
            internal_gate={
                "mechanism": "SkillGen verification round (net-gain thresholds in config)",
                "skill_status": status,
                "skill_produced": skill is not None,
                "note": "this is SkillGen's own internal verification gate, NOT the benchmark metric",
            },
            native_summary={
                "initial_skill": {
                    "condition": NATIVE_START_CONDITION["skillgen"],
                    "chars": 0,
                    "source": "SkillGen's pipeline discovers a skill from baseline "
                              "failures; it takes no starting skill",
                },
                "effective_config": str(effective_path),
                "n_pool_instances": len(instances),
                "pool_roles": ["train"],
                "pool_note": "SkillGen has no native train/val split; its pool is the manifest "
                "TRAIN ids and it samples its own verification subset from within it "
                "(pipeline.py::_build_verification_sample); VAL is unused by native SkillGen",
                "skill_id": getattr(skill, "skill_id", None),
                "token_stats_path": str(ctx.run_dir / "skillgen_token_stats.json"),
            },
        )
