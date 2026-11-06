"""Run a SkillGen skill with the LiveMath direct-chat evaluator."""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "SkillGen"))

from integration.common.run_logger import CallRecord, append_record  # noqa: E402

DEFAULT_DATASET = Path(__file__).resolve().parent / "generated_dataset" / "test.json"
DEFAULT_LOG_PATH = Path(__file__).resolve().parent / "run_log.jsonl"


def run_baseline(
    *,
    dataset_path: Path = DEFAULT_DATASET,
    model: str = "livemath-solver",
    skill=None,
    max_workers: int = 8,
    log_path: Path = DEFAULT_LOG_PATH,
    run_id: str | None = None,
) -> list:
    """Run SkillGen's real `collect_trajectories` over the LiveMath dataset.

    Import of `models`/`trajectory`/`llm` is deferred inside this function
    (rather than at module scope) so that importing this driver module never
    requires SkillGen's full dependency stack (numpy/sklearn/...) unless a
    caller actually invokes `run_baseline`.
    """
    from main import load_dataset
    from trajectory import AgentConfig, collect_trajectories
    import llm as skillgen_llm

    run_id = run_id or str(uuid.uuid4())
    dataset = load_dataset(str(dataset_path))
    config = AgentConfig(model=model, judge_model=model, temperature=0.0)

    skillgen_llm.reset_token_stats()
    t0 = time.time()
    trajectories = collect_trajectories(
        dataset.instances,
        dataset.task_type,
        skill=skill,
        config=config,
        max_workers=max_workers,
        progress_desc="livemath-baseline",
    )
    wall_time_total = time.time() - t0

    usage_rows = {row["stage"]: row for row in skillgen_llm.get_token_stats()}
    # Per-instance token/call attribution isn't exposed by SkillGen's stage-scoped
    # tracker at this granularity without a stage per instance; distribute the
    # per-run totals evenly across instances as a documented approximation
    # (exact per-call accounting would need SkillOpt-style per-item tracking,
    # out of scope for this thin driver -- see PHASE1_REPORT.md limitations).
    n = len(trajectories) or 1
    total_prompt = sum(r["prompt_tokens"] for r in usage_rows.values())
    total_completion = sum(r["completion_tokens"] for r in usage_rows.values())
    total_calls = sum(r["calls"] for r in usage_rows.values())

    for traj in trajectories:
        append_record(
            log_path,
            CallRecord(
                run_id=run_id,
                method="skillgen",
                benchmark="livemathematicianbench",
                model=model,
                harness="direct",
                phase="baseline_eval",
                task_id=traj.instance_id,
                prompt_tokens=total_prompt // n,
                completion_tokens=total_completion // n,
                cached_tokens=None,
                llm_calls=max(1, total_calls // n),
                wall_time=(traj.latency if traj.latency is not None else wall_time_total / n),
                task_score=traj.score,
            ),
        )
    return trajectories


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", default="livemath-solver")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    args = parser.parse_args()

    trajectories = run_baseline(
        dataset_path=args.dataset, model=args.model, max_workers=args.max_workers, log_path=args.log_path,
    )
    n_success = sum(1 for t in trajectories if t.success)
    print(f"n={len(trajectories)} success={n_success} acc={n_success / len(trajectories) if trajectories else 0:.3f}")
    print(f"Run log: {args.log_path}")


if __name__ == "__main__":
    main()
