"""Run native SSO optimization and held-out evaluation on LiveMath."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from integration.common.paths import SSO_ROOT, ensure_on_sys_path
from integration.common.run_logger import CallRecord, append_record
from integration.sso_livemath.livemath_executor import LiveMathExecutor
from integration.sso_livemath.livemath_loader import LiveMathTaskLoader, canonical_split_task_ids
from integration.sso_livemath.livemath_scorer import LiveMathScorer

ensure_on_sys_path(SSO_ROOT)

from sso.core.algorithm import SSOOptimizer, StepInputs  # noqa: E402
from sso.core.types import Skill  # noqa: E402
from sso.llm.client import ChatCompletionsClient  # noqa: E402
from sso.llm.clusterer import LLMBehaviorClusterer  # noqa: E402
from sso.llm.extractor import LLMBehaviorExtractor  # noqa: E402
from sso.llm.judge import LLMPairwiseJudge  # noqa: E402
from sso.llm.probe_generator import LLMProbeGenerator  # noqa: E402
from sso.llm.renderer import LLMRenderer  # noqa: E402


def build_real_optimizer(
    *, judge_endpoint: str, judge_model: str, num_probes: int, update_budget: int,
    judge_max_tokens: int, judge_temperature: float,
) -> tuple[SSOOptimizer, ChatCompletionsClient, LiveMathExecutor]:
    """Construct SSO components with a shared optimizer client and a separate solver executor."""
    client = ChatCompletionsClient(
        endpoint=judge_endpoint, model=judge_model,
        max_tokens=judge_max_tokens, temperature=judge_temperature,
    )
    executor = LiveMathExecutor()  # real call_solver_raw against the configured proxy
    optimizer = SSOOptimizer(
        executor=executor,
        generator=LLMProbeGenerator(client),
        judge=LLMPairwiseJudge(client),
        extractor=LLMBehaviorExtractor(client),
        clusterer=LLMBehaviorClusterer(client),
        renderer=LLMRenderer(client),
        num_probes=num_probes,
        update_budget=update_budget,
    )
    return optimizer, client, executor


def _assert_schedule_within_canonical_split(
    schedule_spec: list[dict], test_ids: list[str], ids: dict[str, list[str]],
) -> None:
    """Reject training or validation schedule IDs outside their assigned shared splits."""
    train_set, val_set, test_set = set(ids["train"]), set(ids["val"]), set(ids["test"])
    errors: list[str] = []

    schedule_train_ids: set[str] = set()
    schedule_val_ids: set[str] = set()
    for i, step in enumerate(schedule_spec):
        step_train = set(step["training_task_ids"]) | set(step["history_task_ids"])
        step_val = set(step["validation_task_ids"])
        schedule_train_ids |= step_train
        schedule_val_ids |= step_val
        not_in_train = step_train - train_set
        if not_in_train:
            errors.append(
                f"step {i}: training/history ids not in canonical D_train: {sorted(not_in_train)}"
            )
        not_in_val = step_val - val_set
        if not_in_val:
            errors.append(f"step {i}: validation ids not in canonical D_val: {sorted(not_in_val)}")

    not_in_test = set(test_ids) - test_set
    if not_in_test:
        errors.append(f"--test-task-ids not in canonical D_test: {sorted(not_in_test)}")

    if errors:
        raise SystemExit(
            "Schedule/test ids are not confined to the frozen canonical split:\n  "
            + "\n  ".join(errors)
        )


def _step_validation_summary(steps) -> list[dict]:
    """Extract each step's validation comparisons and acceptance result."""
    return [
        {
            "step_idx": i,
            "had_candidate": step.validation is not None,
            "wins": step.validation.wins if step.validation else None,
            "losses": step.validation.losses if step.validation else None,
            "ties": step.validation.ties if step.validation else None,
            "accepted": step.validation.accepted if step.validation else None,
        }
        for i, step in enumerate(steps)
    ]


def _merge_usage_dicts(a: dict, b: dict) -> dict:
    """Sum usage counters over the union of keys, treating missing values as zero."""
    return {k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b)}


def load_schedule_from_spec(spec: list[dict], loader: LiveMathTaskLoader) -> list[StepInputs]:
    """Explicit schedule spec (already-parsed JSON list) -> list[StepInputs].
    Format: [{"training_task_ids": [...], "history_task_ids": [...],
    "validation_task_ids": [...]}, ...]. Every id must resolve through
    `loader`; StepInputs itself enforces train/val disjointness and
    history-is-a-subset-of-training."""
    if not isinstance(spec, list) or not spec:
        raise ValueError("schedule must be a nonempty JSON list of steps")
    steps = []
    for i, step in enumerate(spec):
        for key in ("training_task_ids", "history_task_ids", "validation_task_ids"):
            if key not in step:
                raise ValueError(f"schedule step {i} missing required key {key!r}")
        steps.append(StepInputs(
            training_batch=tuple(loader.tasks(step["training_task_ids"])),
            history_task_ids=tuple(step["history_task_ids"]),
            validation_batch=tuple(loader.tasks(step["validation_task_ids"])),
        ))
    return steps


def build_smoke_schedule() -> list[dict]:
    """*** SMOKE-TEST ONLY -- NOT AN EXPERIMENTAL PROTOCOL ***
    A tiny, hardcoded, explicitly-labeled one-step schedule drawn from the
    canonical D_train/D_val, for validating that the real optimization path
    mechanically works end-to-end. Deliberately small (cheap) and
    deliberately NOT tunable via CLI flags, so it can never be mistaken for
    a considered X_t/H_t/V_t sampling policy -- a real run must author and
    pass its own --schedule file instead."""
    ids = canonical_split_task_ids()
    train_ids = ids["train"][:2]
    val_ids = ids["val"][:3]
    return [{
        "training_task_ids": train_ids,
        "history_task_ids": train_ids[:1],
        "validation_task_ids": val_ids,
    }]


def evaluate_on_test(
    executor: LiveMathExecutor, scorer: LiveMathScorer, skill: Skill, test_tasks,
) -> dict:
    """Held-out deployment evaluation: run the frozen selected_skill against
    every D_test task directly through the executor (no optimizer, no
    training/validation machinery), then score. Never called before
    optimization finishes; test_task_ids must come only from
    canonical_split_task_ids()['test'] or an equivalent explicitly-held-out
    id list the caller supplies -- this function does not itself guard
    against a caller passing overlapping ids, matching StepInputs'
    stance that dataset partitioning is the caller's responsibility."""
    executions = {}
    for task in test_tasks:
        executions[task.task_id] = executor.execute(skill, task)
    return scorer.score_all(executions)


def _run_eval_only(args: argparse.Namespace, out_dir: Path, run_id: str, loader, ids: dict) -> None:
    """Load a saved skill and evaluate it on the held-out test split."""
    test_ids = (
        [t.strip() for t in args.test_task_ids.split(",") if t.strip()]
        if args.test_task_ids else ids["test"]
    )
    not_in_test = set(test_ids) - set(ids["test"])
    if not_in_test:
        raise SystemExit(f"--test-task-ids not in canonical D_test: {sorted(not_in_test)}")

    executor = LiveMathExecutor()
    scorer = LiveMathScorer({item.task_id: item for item in loader.items()})
    skill = Skill(args.skill_file.read_text(encoding="utf-8"))

    executor.reset_usage()
    test_tasks = loader.tasks(test_ids)
    t0 = time.time()
    test_scores = evaluate_on_test(executor, scorer, skill, test_tasks)
    deployment_wall_s = time.time() - t0
    deployment_solver_usage = executor.usage_snapshot()

    summary = {
        "run_id": run_id,
        "mode": "eval_only",
        "skill_file": str(args.skill_file),
        "test_task_ids": test_ids,
        "test_accuracy": test_scores["accuracy"],
        "test_n": test_scores["n"],
        "deployment_wall_time_s": round(deployment_wall_s, 1),
        "deployment_token_summary": {
            "solver_side": deployment_solver_usage,
            "_total": dict(deployment_solver_usage),
        },
        "sso_provenance": "unofficial reproduction of arXiv:2607.28777",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "test_scores.json").write_text(json.dumps(test_scores, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nWritten to: {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--schedule", type=Path, help="Path to an explicit schedule JSON file (see load_schedule).")
    p.add_argument("--smoke-schedule", action="store_true",
                   help="*** SMOKE TEST ONLY *** use build_smoke_schedule() instead of --schedule.")
    p.add_argument("--test-task-ids", type=str, default=None,
                   help="Comma-separated D_test task ids. Default: canonical_split_task_ids()['test'] (all 125).")
    p.add_argument("--initial-skill", type=Path,
                   help="Required unless --eval-only.")
    p.add_argument("--num-probes", type=int, help="Required unless --eval-only.")
    p.add_argument("--update-budget", type=int, help="Required unless --eval-only.")
    p.add_argument("--judge-endpoint", type=str, default="http://localhost:4001/v1/chat/completions",
                   help="Optimizer-side (all five modules) endpoint. Default port 4001 -- SSO's own proxy, "
                        "never the SkillOpt experiment's proxy on 4000.")
    p.add_argument("--judge-model", type=str, help="Required unless --eval-only.")
    p.add_argument("--judge-max-tokens", type=int, help="Required unless --eval-only -- see "
                        "build_real_optimizer's docstring for why this can't default silently.")
    p.add_argument("--judge-temperature", type=float, help="Required unless --eval-only.")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--eval-only", action="store_true",
                   help="Skip optimization entirely: load --skill-file from disk and only run the "
                        "held-out D_test deployment evaluation.")
    p.add_argument("--skill-file", type=Path, help="Required with --eval-only: a previously persisted "
                        "selected_skill.md (or any skill content file) to reload and deploy.")
    args = p.parse_args()

    if args.eval_only:
        if not args.skill_file:
            raise SystemExit("--eval-only requires --skill-file <path>")
    else:
        if bool(args.schedule) == bool(args.smoke_schedule):
            raise SystemExit("Pass exactly one of --schedule <file> or --smoke-schedule.")
        missing = [
            name for name, val in (
                ("--initial-skill", args.initial_skill), ("--num-probes", args.num_probes),
                ("--update-budget", args.update_budget), ("--judge-model", args.judge_model),
                ("--judge-max-tokens", args.judge_max_tokens),
                ("--judge-temperature", args.judge_temperature),
            ) if val is None
        ]
        if missing:
            raise SystemExit(f"Missing required argument(s) for a full (non-eval-only) run: {missing}")

    run_id = args.run_id or datetime.now(timezone.utc).strftime("sso_livemath_%Y%m%dT%H%M%SZ")
    out_dir = args.out_dir or (Path(__file__).resolve().parent / "outputs" / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = LiveMathTaskLoader()
    ids = canonical_split_task_ids()

    if args.eval_only:
        _run_eval_only(args, out_dir, run_id, loader, ids)
        return

    schedule_generation_params = None
    if args.smoke_schedule:
        schedule_spec = build_smoke_schedule()
        schedule_source = "SMOKE_TEST (build_smoke_schedule(), not a frozen experimental protocol)"
    else:
        raw = json.loads(args.schedule.read_text(encoding="utf-8"))
        # Two accepted shapes: a plain list of steps (hand-authored, or any
        # older schedule file), or {"generation_params": {...}, "steps": [
        # ...]} -- generate_schedule.py's output. Never a silent third
        # interpretation.
        if isinstance(raw, dict) and "steps" in raw:
            schedule_spec = raw["steps"]
            schedule_generation_params = raw.get("generation_params")
        else:
            schedule_spec = raw
        schedule_source = str(args.schedule)
    steps = load_schedule_from_spec(schedule_spec, loader)

    test_ids = (
        [t.strip() for t in args.test_task_ids.split(",") if t.strip()]
        if args.test_task_ids else ids["test"]
    )
    # Leakage guard: no test id may appear in any step's training/validation ids.
    schedule_train_val_ids = {
        tid for step in schedule_spec
        for tid in (step["training_task_ids"] + step["validation_task_ids"])
    }
    leaked = schedule_train_val_ids & set(test_ids)
    if leaked:
        raise SystemExit(f"D_test leakage: {sorted(leaked)} appear in the training/validation schedule")
    # Membership guard: every schedule/test id must actually come from the
    # frozen canonical split, not merely avoid overlapping D_test (see
    # _assert_schedule_within_canonical_split's docstring).
    _assert_schedule_within_canonical_split(schedule_spec, test_ids, ids)

    optimizer, client, executor = build_real_optimizer(
        judge_endpoint=args.judge_endpoint, judge_model=args.judge_model,
        num_probes=args.num_probes, update_budget=args.update_budget,
        judge_max_tokens=args.judge_max_tokens, judge_temperature=args.judge_temperature,
    )
    scorer = LiveMathScorer({item.task_id: item for item in loader.items()})
    initial_skill = Skill(args.initial_skill.read_text(encoding="utf-8"))

    # ---- Optimization phase ----
    client.reset_usage()
    executor.reset_usage()
    t0 = time.time()
    try:
        result = optimizer.optimize(initial_skill, steps)
    except BaseException:


        partial_wall_s = time.time() - t0
        partial = {
            "run_id": run_id,
            "schedule_source": schedule_source,
            "status": "CRASHED_MID_OPTIMIZATION",
            "optimization_wall_time_s_before_crash": round(partial_wall_s, 1),
            "optimization_token_summary_before_crash": {
                "optimizer_side": client.usage_snapshot(),
                "solver_side": executor.usage_snapshot(),
            },
        }
        (out_dir / "summary_partial.json").write_text(
            json.dumps(partial, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        raise
    optimization_wall_s = time.time() - t0
    optimization_optimizer_usage = client.usage_snapshot()
    optimization_solver_usage = executor.usage_snapshot()

    selected_skill_path = out_dir / "selected_skill.md"
    selected_skill_path.write_text(result.selected_skill.content, encoding="utf-8")

    # ---- Deployment phase: held-out D_test, frozen selected_skill only ----
    # Reloaded from the file just written, not the in-memory `result.
    # selected_skill` reference -- proves the persisted artifact (what a
    # later --eval-only run would also load) is genuinely what gets
    # deployed, not merely an in-memory object that happens to be correct.
    deployed_skill = Skill(selected_skill_path.read_text(encoding="utf-8"))
    client.reset_usage()  # optimizer-side client is not used for deployment, but reset for cleanliness
    executor.reset_usage()
    test_tasks = loader.tasks(test_ids)
    t1 = time.time()
    test_scores = evaluate_on_test(executor, scorer, deployed_skill, test_tasks)
    deployment_wall_s = time.time() - t1
    deployment_solver_usage = executor.usage_snapshot()

    step_validation_summary = _step_validation_summary(result.steps)

    summary = {
        "run_id": run_id,
        "schedule_source": schedule_source,
        "n_steps": len(steps),
        "step_validation_summary": step_validation_summary,
        "judge_endpoint": args.judge_endpoint,
        "judge_model": args.judge_model,
        "num_probes": args.num_probes,
        "update_budget": args.update_budget,
        "test_task_ids": test_ids,
        "test_accuracy": test_scores["accuracy"],
        "test_n": test_scores["n"],
        "optimization_wall_time_s": round(optimization_wall_s, 1),
        "deployment_wall_time_s": round(deployment_wall_s, 1),
        "optimization_token_summary": {
            "optimizer_side": optimization_optimizer_usage,
            "solver_side": optimization_solver_usage,
            "_total": _merge_usage_dicts(optimization_optimizer_usage, optimization_solver_usage),
        },
        "deployment_token_summary": {
            "solver_side": deployment_solver_usage,
            "_total": dict(deployment_solver_usage),
        },


        "sso_provenance": "unofficial reproduction of arXiv:2607.28777",
        "reproduction_choices_this_run": {
            "schedule_source": schedule_source,
            "schedule_generation_params": schedule_generation_params,
            "judge_max_tokens": args.judge_max_tokens,
            "judge_temperature": args.judge_temperature,
            "gate_rule": "sso.core.validation.validate_candidate (unmodified this run)",
        },


        "solver_transport_retries_not_counted": True,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "test_scores.json").write_text(json.dumps(test_scores, indent=2, ensure_ascii=False), encoding="utf-8")

    call_log_path = out_dir / "call_log.jsonl"
    common_kwargs: dict[str, Any] = dict(
        run_id=run_id, method="sso", benchmark="livemathematicianbench",
        model=f"solver={executor._solver_config.model} optimizer={args.judge_model}",
        harness="direct",
    )
    append_record(call_log_path, CallRecord(
        **common_kwargs, phase="optimize", task_id="__aggregate__",
        prompt_tokens=summary["optimization_token_summary"]["_total"]["prompt_tokens"],
        completion_tokens=summary["optimization_token_summary"]["_total"]["completion_tokens"],
        cached_tokens=summary["optimization_token_summary"]["_total"]["cached_tokens"] or None,
        llm_calls=summary["optimization_token_summary"]["_total"]["calls"],
        wall_time=optimization_wall_s, task_score=None, optimization_iteration=len(steps),
    ))
    append_record(call_log_path, CallRecord(
        **common_kwargs, phase="deploy", task_id="__aggregate__",
        prompt_tokens=summary["deployment_token_summary"]["_total"]["prompt_tokens"],
        completion_tokens=summary["deployment_token_summary"]["_total"]["completion_tokens"],
        cached_tokens=summary["deployment_token_summary"]["_total"]["cached_tokens"] or None,
        llm_calls=summary["deployment_token_summary"]["_total"]["calls"],
        wall_time=deployment_wall_s, task_score=test_scores["accuracy"], optimization_iteration=None,
    ))

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nWritten to: {out_dir}")


if __name__ == "__main__":
    main()
