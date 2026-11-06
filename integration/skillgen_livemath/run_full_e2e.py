"""Run native SkillGen optimization and held-out evaluation on LiveMath."""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
_SKILLGEN_ROOT = _WORKSPACE_ROOT / "SkillGen"
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))
if str(_SKILLGEN_ROOT) not in sys.path:
    sys.path.insert(0, str(_SKILLGEN_ROOT))

from integration.common.run_logger import CallRecord, append_record  # noqa: E402
from integration.skillgen_livemath.build_dataset import build_dataset  # noqa: E402


def _instances_from_dataset_dict(d: dict):
    """Same shape main.py::load_dataset builds, without a disk round-trip."""
    from models import TaskInstance, TaskType  # SkillGen-internal, path added above

    instances = [
        TaskInstance(
            instance_id=item["instance_id"], input=item["input"],
            ground_truth=item.get("ground_truth"), metadata=item.get("metadata", {}),
        )
        for item in d["instances"]
    ]
    return instances, TaskType(d["task_type"])


def _assert_no_leakage(pool_instance_ids: set[str], test_instance_ids: set[str]) -> None:
    leaked = pool_instance_ids & test_instance_ids
    if leaked:
        raise SystemExit(
            f"D_test leakage: {sorted(leaked)} appear in the instance pool passed to "
            "run_pipeline(). Refusing to run."
        )


_SOLVER_ROLES = ("baseline_agent", "verification_agent")
_OPTIMIZER_ROLES = (
    "baseline_judge",
    "induction", "induction_contextual", "induction_summary",
    "induction_pattern", "induction_contrastive",
    "generation_plan", "generation_execute",
    "refinement",
    "verification_judge", "verification_case_analyst", "verification_revision_synthesiser",
)


def _build_effective_config(
    *, out_dir: Path, solver_model_alias: str, optimizer_model_alias: str,
    embedding_model_alias: str,
    max_tokens: int, max_tokens_generation: int,
    max_refine_rounds: int | None, verification_sample_size: int | None,
    verification_min_sample: int | None,
) -> Path:
    """Write a run-local SkillGen config with explicit solver, optimizer, and embedding aliases."""
    with open(_SKILLGEN_ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)

    all_roles = set(cfg["models"].keys()) - {"default"}
    covered = set(_SOLVER_ROLES) | set(_OPTIMIZER_ROLES)
    if all_roles != covered:
        raise ValueError(
            f"SkillGen/config.yaml model roles changed upstream: "
            f"missing from role split={sorted(all_roles - covered)}, "
            f"stale in role split={sorted(covered - all_roles)}"
        )
    for role in _SOLVER_ROLES:
        cfg["models"][role] = solver_model_alias
    for role in _OPTIMIZER_ROLES:
        cfg["models"][role] = optimizer_model_alias
    cfg["models"]["default"] = optimizer_model_alias

    cfg["embedding"]["model"] = embedding_model_alias

    cfg["llm"]["max_tokens"] = max_tokens
    cfg["llm"]["max_tokens_generation"] = max_tokens_generation

    if max_refine_rounds is not None:
        cfg["pipeline"]["max_refine_rounds"] = max_refine_rounds
    if verification_sample_size is not None:
        cfg["verification"]["sample_size"] = verification_sample_size
    if verification_min_sample is not None:
        cfg["verification"]["min_sample"] = verification_min_sample

    # Isolate every artifact this run produces under our own output dir --
    # never SkillGen/'s own ./artifacts or ./skill_output.
    cfg["pipeline"]["artifact_root"] = str(out_dir / "artifacts")
    cfg["skill_output"] = {"path": str(out_dir / "skill_output")}

    effective_path = out_dir / "effective_config.yaml"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(effective_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return effective_path


def _sum_token_rows(rows: list[dict]) -> dict:
    """Sum reported token usage across model, stage, and call-type rows."""
    return {
        "calls": sum(r["calls"] for r in rows),
        "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
        "completion_tokens": sum(r["completion_tokens"] for r in rows),
        "cached_tokens": sum(r.get("cached_tokens", 0) for r in rows),
        "total_tokens": sum(r["total_tokens"] for r in rows),
        "by_stage": rows,
    }


def _write_test_trajectories(path: Path, trajectories: list, *, deployment_condition: str) -> int:
    """Persist per-task deployment trajectories; leave unavailable usage fields null."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for traj in trajectories:
            eval_meta = (traj.metadata or {}).get("livemath_eval", {})
            final_output = str(traj.final_output or "")
            row = {
                "instance_id": traj.instance_id,
                "final_output": final_output,
                "predicted_label": eval_meta.get("predicted_label"),
                "gt_label": eval_meta.get("gt_label"),
                "success": traj.success,
                "answer_tag_found": "<answer>" in final_output.lower(),
                "completion_tokens": None,  # see docstring: not recoverable at this granularity yet
                "finish_reason": None,
                "condition": deployment_condition,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(trajectories)


def _deployment_phase(
    *, reloaded_skill, test_instances, task_type, solver_model_alias: str,
    skillgen_llm, run_condition_fn, max_workers: int = 8, max_tokens: int = 4096,
) -> dict:
    """Deployment: mirrors SkillGen/eval_skill.py's own native rejected-skill
    contract exactly (see module docstring). `skillgen_llm` needs only
    reset_token_stats()/get_token_stats(); `run_condition_fn` needs only
    eval_skill.run_condition()'s signature -- both are injected so this is
    testable without a real pipeline run or network access.
    """
    from models import SkillStatus  # SkillGen-internal

    skill_rejected = (
        reloaded_skill.status == SkillStatus.DEPRECATED
        or not (reloaded_skill.body or "").strip()
    )

    skillgen_llm.reset_token_stats()
    t1 = time.time()


    with skillgen_llm.stage_scope("deployment"):
        if skill_rejected:
            deployment_condition = "no_skill_baseline_fallback"
            # Native fallback (SkillGen/eval_skill.py::main): run the skill=None
            # condition once on frozen D_test and report it as the final result
            # for this rejected skill, so net_gain==0 by construction -- never
            # substituted with the D_train trajectory baseline or another split.
            condition = run_condition_fn(
                test_instances, task_type, model=solver_model_alias, judge_model=solver_model_alias,
                skill=None, max_workers=max_workers, max_tokens=max_tokens,
            )
            deployment_net_gain = 0.0
        else:
            deployment_condition = "skill_conditioned"
            condition = run_condition_fn(
                test_instances, task_type, model=solver_model_alias, judge_model=solver_model_alias,
                skill=reloaded_skill, max_workers=max_workers, max_tokens=max_tokens,
            )
            # Not computed here: this driver does not run a paired D_test
            # baseline for an accepted skill, so net_gain against D_test is
            # left unset rather than fabricated.
            deployment_net_gain = None
    deployment_wall_s = time.time() - t1
    deployment_rows = skillgen_llm.get_token_stats()
    deployment_token_summary = _sum_token_rows(deployment_rows)

    return {
        "skill_rejected": skill_rejected,
        "deployment_condition": deployment_condition,
        "deployment_net_gain": deployment_net_gain,
        "condition": condition,
        "deployment_wall_s": deployment_wall_s,
        "deployment_token_summary": deployment_token_summary,
    }


def run_full_e2e(
    *, solver_model_alias: str, optimizer_model_alias: str, embedding_model_alias: str,
    out_dir: Path,
    max_refine_rounds: int | None,
    verification_sample_size: int | None, verification_min_sample: int | None,
    max_tokens: int, max_tokens_generation: int, smoke_test: bool,
    mini_split_n: int | None = None,
) -> dict:
    import llm as skillgen_llm  # SkillGen-internal
    from pipeline import run_pipeline  # SkillGen-internal
    from skill_store import load_skill  # SkillGen-internal
    from eval_skill import run_condition  # SkillGen-internal

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- D_train / D_val / D_test, all from the canonical frozen split ----
    train_ds = build_dataset("train")
    val_ds = build_dataset("val")
    test_ds = build_dataset("test")
    if mini_split_n is not None:
        # *** SMOKE TEST ONLY *** same canonical ids, just a documented small
        # prefix of each split -- never a re-sample, never touches test.
        train_ds = {**train_ds, "instances": train_ds["instances"][:mini_split_n]}
        val_ds = {**val_ds, "instances": val_ds["instances"][:mini_split_n]}
        test_ds = {**test_ds, "instances": test_ds["instances"][:mini_split_n]}
    train_instances, task_type = _instances_from_dataset_dict(train_ds)
    val_instances, _ = _instances_from_dataset_dict(val_ds)
    test_instances, _ = _instances_from_dataset_dict(test_ds)

    pool_instances = train_instances + val_instances
    pool_ids = {i.instance_id for i in pool_instances}
    test_ids = {i.instance_id for i in test_instances}
    _assert_no_leakage(pool_ids, test_ids)

    effective_config_path = _build_effective_config(
        out_dir=out_dir, solver_model_alias=solver_model_alias,
        optimizer_model_alias=optimizer_model_alias, embedding_model_alias=embedding_model_alias,
        max_tokens=max_tokens, max_tokens_generation=max_tokens_generation,
        max_refine_rounds=max_refine_rounds,
        verification_sample_size=verification_sample_size,
        verification_min_sample=verification_min_sample,
    )


    with open(effective_config_path, encoding="utf-8") as _f:
        _effective_cfg_for_summary = yaml.safe_load(_f)
    verification_protocol = {
        "max_refine_rounds": _effective_cfg_for_summary["pipeline"]["max_refine_rounds"],
        "verification_sample_size": _effective_cfg_for_summary["verification"]["sample_size"],
        "verification_min_sample": _effective_cfg_for_summary["verification"]["min_sample"],
        "verification_min_net_gain_abs": _effective_cfg_for_summary["verification"]["min_net_gain_abs"],
        "verification_min_net_gain_rel": _effective_cfg_for_summary["verification"]["min_net_gain_rel"],
    }

    # ---- Optimization phase: SkillGen's real, unmodified native pipeline ----
    skillgen_llm.reset_token_stats()
    t0 = time.time()
    skill = run_pipeline(
        pool_instances, task_type,
        config_path=str(effective_config_path),
        dataset_id="livemathematicianbench_train_val_pool",
        task_name="LiveMathematicianBench",
        dataset_metadata={"n_train": len(train_instances), "n_val": len(val_instances)},
    )
    optimization_wall_s = time.time() - t0
    optimization_rows = skillgen_llm.get_token_stats()
    optimization_token_summary = _sum_token_rows(optimization_rows)

    if skill is None:
        raise RuntimeError(
            "run_pipeline() returned None (no baseline failures found -- nothing to "
            "learn from, or no candidate produced). No skill to persist or evaluate."
        )
    skill_dir = Path(effective_config_path).parent / "skill_output"
    # run_pipeline() itself further nests under a timestamped subdir; find it.
    ts_dirs = sorted(p for p in skill_dir.iterdir() if p.is_dir()) if skill_dir.exists() else []
    persisted_skill_dir = ts_dirs[-1] if ts_dirs else skill_dir
    reloaded_skill = load_skill(str(persisted_skill_dir), skill_id=skill.skill_id)

    # ---- Deployment phase: native rejected-skill fallback, held-out D_test ----
    deploy = _deployment_phase(
        reloaded_skill=reloaded_skill, test_instances=test_instances, task_type=task_type,
        solver_model_alias=solver_model_alias, skillgen_llm=skillgen_llm,
        run_condition_fn=run_condition, max_tokens=max_tokens,
    )
    condition = deploy["condition"]
    deployment_wall_s = deploy["deployment_wall_s"]
    deployment_token_summary = deploy["deployment_token_summary"]

    test_trajectories_path = out_dir / "deployment" / "test_trajectories.jsonl"
    _write_test_trajectories(
        test_trajectories_path, condition.trajectories,
        deployment_condition=deploy["deployment_condition"],
    )
    no_answer_tag_rate = (
        sum(1 for t in condition.trajectories if "<answer>" not in str(t.final_output or "").lower())
        / len(condition.trajectories)
        if condition.trajectories else None
    )

    summary = {
        "run_id": out_dir.name,
        "smoke_test": smoke_test,
        "solver_model_alias": solver_model_alias,
        "optimizer_model_alias": optimizer_model_alias,
        "embedding_model_alias": embedding_model_alias,
        "effective_config_path": str(effective_config_path),
        "n_train": len(train_instances),
        "n_val": len(val_instances),
        "n_test": len(test_instances),
        "train_task_ids": [i.instance_id for i in train_instances],
        "val_task_ids": [i.instance_id for i in val_instances],
        "test_task_ids": [i.instance_id for i in test_instances],
        "skill_id": skill.skill_id,
        "skill_status": skill.status.value,
        "skill_rejected": deploy["skill_rejected"],
        "deployment_condition": deploy["deployment_condition"],
        "deployment_skipped_rejected_skill": deploy["skill_rejected"],
        "deployment_net_gain": deploy["deployment_net_gain"],
        "persisted_skill_dir": str(persisted_skill_dir),
        "test_accuracy": condition.success_rate,
        "test_n": condition.n_instances,
        "test_n_success": condition.n_success,
        "optimization_wall_time_s": round(optimization_wall_s, 1),
        "deployment_wall_time_s": round(deployment_wall_s, 1),
        "optimization_token_summary": optimization_token_summary,
        "deployment_token_summary": deployment_token_summary,
        "no_answer_tag_rate": no_answer_tag_rate,
        "test_trajectories_path": str(test_trajectories_path),


        "truncation_rate": None,
        "truncation_rate_unavailable_reason": (
            "SkillGen/llm.py::chat_multi_turn returns only the parsed message, "
            "not finish_reason/usage -- deferred, see build log"
        ),
        "verification_protocol": verification_protocol,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    call_log_path = out_dir / "call_log.jsonl"
    common = dict(run_id=out_dir.name, method="skillgen", benchmark="livemathematicianbench",
                  model=f"solver={solver_model_alias} optimizer={optimizer_model_alias}", harness="direct")
    append_record(call_log_path, CallRecord(
        **common, phase="optimize", task_id="__aggregate__",
        prompt_tokens=optimization_token_summary["prompt_tokens"],
        completion_tokens=optimization_token_summary["completion_tokens"],
        cached_tokens=None, llm_calls=optimization_token_summary["calls"],
        wall_time=optimization_wall_s, task_score=None, optimization_iteration=1,
    ))
    append_record(call_log_path, CallRecord(
        **common, phase="deploy", task_id="__aggregate__",
        prompt_tokens=deployment_token_summary["prompt_tokens"],
        completion_tokens=deployment_token_summary["completion_tokens"],
        cached_tokens=None, llm_calls=deployment_token_summary["calls"],
        wall_time=deployment_wall_s, task_score=condition.success_rate, optimization_iteration=None,
    ))

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nWritten to: {out_dir}")
    return summary


def run_eval_only(
    *, solver_model_alias: str, skill_dir: Path, skill_id: str, out_dir: Path, run_id: str,
    max_tokens: int, test_task_ids: str | None,
) -> dict:
    """Load a saved skill and run held-out deployment evaluation."""
    import llm as skillgen_llm  # SkillGen-internal
    from skill_store import load_skill  # SkillGen-internal
    from eval_skill import run_condition  # SkillGen-internal

    out_dir.mkdir(parents=True, exist_ok=True)

    test_ds = build_dataset("test")
    ids = {str(item["instance_id"]) for item in test_ds["instances"]}
    requested_ids = (
        [t.strip() for t in test_task_ids.split(",") if t.strip()] if test_task_ids else None
    )
    if requested_ids is not None:
        not_in_test = set(requested_ids) - ids
        if not_in_test:
            raise SystemExit(f"--test-task-ids not in canonical D_test: {sorted(not_in_test)}")
        test_ds = {
            **test_ds,
            "instances": [i for i in test_ds["instances"] if str(i["instance_id"]) in requested_ids],
        }
    test_instances, task_type = _instances_from_dataset_dict(test_ds)

    reloaded_skill = load_skill(str(skill_dir), skill_id=skill_id)

    deploy = _deployment_phase(
        reloaded_skill=reloaded_skill, test_instances=test_instances, task_type=task_type,
        solver_model_alias=solver_model_alias, skillgen_llm=skillgen_llm,
        run_condition_fn=run_condition, max_tokens=max_tokens,
    )
    condition = deploy["condition"]

    test_trajectories_path = out_dir / "deployment" / "test_trajectories.jsonl"
    _write_test_trajectories(
        test_trajectories_path, condition.trajectories,
        deployment_condition=deploy["deployment_condition"],
    )

    summary = {
        "run_id": run_id,
        "mode": "eval_only",
        "solver_model_alias": solver_model_alias,
        "skill_dir": str(skill_dir),
        "skill_id": skill_id,
        "skill_status": reloaded_skill.status.value,
        "skill_rejected": deploy["skill_rejected"],
        "deployment_condition": deploy["deployment_condition"],
        "test_task_ids": [i.instance_id for i in test_instances],
        "test_accuracy": condition.success_rate,
        "test_n": condition.n_instances,
        "test_n_success": condition.n_success,
        "deployment_wall_time_s": round(deploy["deployment_wall_s"], 1),
        "deployment_token_summary": deploy["deployment_token_summary"],
        "test_trajectories_path": str(test_trajectories_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nWritten to: {out_dir}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--solver-model-alias", required=True,
                    help="Proxy alias for solver roles; configure optimizer roles separately.")
    p.add_argument("--optimizer-model-alias", required=True,
                    help="Proxy alias for every optimizer-side role (induction/generation/refinement/"
                         "verification judge+case-analyst+revision-synthesiser). May be the same "
                         "physical model as --solver-model-alias or a different one -- both must be "
                         "stated explicitly, never merged by default.")
    p.add_argument("--embedding-model-alias", required=True,
                    help="Proxy embedding model alias, e.g. livemath-embedding. Required (no "
                         "default) so baseline and any future +ours run must state it explicitly "
                         "and identically.")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--max-tokens-generation", type=int, default=16384)
    p.add_argument("--max-refine-rounds", type=int, default=None)
    p.add_argument("--verification-sample-size", type=int, default=None)
    p.add_argument("--verification-min-sample", type=int, default=None)
    p.add_argument("--smoke-test", action="store_true",
                   help="Mark this run's summary.json as SMOKE_TEST_ONLY (does not by itself "
                        "shrink the dataset -- use --smoke-mini-split for that).")
    p.add_argument("--smoke-mini-split", type=int, default=None,
                   help="*** SMOKE TEST ONLY *** cap D_train/D_val/D_test to the first N ids "
                        "each from the canonical split (still the same canonical ids, just a "
                        "documented small prefix). Omit for a real run over the full split.")
    p.add_argument("--eval-only", action="store_true",
                   help="Load a saved skill and run held-out deployment evaluation without optimization.")
    p.add_argument("--skill-dir", type=Path, help="Required with --eval-only.")
    p.add_argument("--skill-id", type=str, help="Required with --eval-only.")
    p.add_argument("--test-task-ids", type=str, default=None,
                   help="--eval-only only: comma-separated D_test task ids. Default: all 125.")
    args = p.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("skillgen_livemath_%Y%m%dT%H%M%SZ")
    out_dir = args.out_dir or (Path(__file__).resolve().parent / "outputs" / run_id)

    if args.eval_only:
        if not args.skill_dir or not args.skill_id:
            raise SystemExit("--eval-only requires --skill-dir and --skill-id")
        run_eval_only(
            solver_model_alias=args.solver_model_alias, skill_dir=args.skill_dir,
            skill_id=args.skill_id, out_dir=out_dir, run_id=run_id,
            max_tokens=args.max_tokens, test_task_ids=args.test_task_ids,
        )
        return

    run_full_e2e(
        solver_model_alias=args.solver_model_alias, optimizer_model_alias=args.optimizer_model_alias,
        embedding_model_alias=args.embedding_model_alias,
        out_dir=out_dir,
        max_refine_rounds=args.max_refine_rounds,
        verification_sample_size=args.verification_sample_size,
        verification_min_sample=args.verification_min_sample,
        max_tokens=args.max_tokens, max_tokens_generation=args.max_tokens_generation,
        smoke_test=args.smoke_test or args.smoke_mini_split is not None,
        mini_split_n=args.smoke_mini_split,
    )


if __name__ == "__main__":
    main()
