"""Evaluate the selected final skill on the held-out deployment split."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from integration.benchmarks.base import BenchmarkAdapter, BenchmarkResult
from integration.common.solver_client import SolverConfig
from integration.experiment.cost import PhaseCost, PhaseCostAccumulator
from integration.harness.base import DirectHarness, SolverSession, TaskRun


@dataclass
class DeploymentResult:
    result: BenchmarkResult
    cost: PhaseCost
    runs: list = field(default_factory=list)
    wall_time_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "metric_name": self.result.metric_name,
            "metric_value": self.result.metric_value,
            "n_tasks": self.result.n_tasks,
            "scorer_ref": self.result.scorer_ref,
            "secondary_metrics": self.result.secondary_metrics,
            "cost": self.cost.to_dict(),
            "wall_time_s": round(self.wall_time_s, 2),
            "per_task": self.result.per_task,
        }


def run_tasks(
    *,
    adapter: BenchmarkAdapter,
    harness: DirectHarness,
    task_ids,
    skill_text: str,
    session: SolverSession,
    role: str,
    max_workers: int = 1,
) -> list:
    """Execute `task_ids` under `skill_text`. Returns TaskRun objects in order."""
    tasks = adapter.tasks_for_ids(task_ids, role=role)
    if max_workers <= 1:
        return [harness.run_task(skill_text, task, session) for task in tasks]

    results: list = [None] * len(tasks)

    def _one(idx_task):
        idx, task = idx_task
        try:
            return idx, harness.run_task(skill_text, task, session)
        except Exception as exc:  # noqa: BLE001 - one failed task must not lose the others
            return idx, TaskRun(
                task_id=task.task_id,
                output=None,
                error=f"{type(exc).__name__}: {exc}",
            )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for idx, run in pool.map(_one, list(enumerate(tasks))):
            results[idx] = run
    return results


def _score_input(run: TaskRun, benchmark: str):
    """What the official scorer for `benchmark` consumes for one task.

    A task the harness could not complete still has to be scored -- as a
    failure, by the official scorer's own rules -- rather than dropped, which
    would silently shrink the denominator.
    """
    if run.output is not None:
        return run.output
    if benchmark == "alfworld":
        return {"won": False, "steps": 0}
    if benchmark == "spreadsheetbench":
        return {}
    return ""


def run_deployment(
    *,
    adapter: BenchmarkAdapter,
    harness: DirectHarness,
    task_ids,
    skill_text: str,
    solver_config: SolverConfig,
    max_workers: int = 1,
    transcripts_path: Path | str | None = None,
) -> DeploymentResult:
    """Run the final skill over the deployment split and score it officially."""
    import time

    accumulator = PhaseCostAccumulator("deployment")
    session = SolverSession(config=solver_config, accumulator=accumulator, role="solver")

    t0 = time.time()
    runs = run_tasks(
        adapter=adapter,
        harness=harness,
        task_ids=task_ids,
        skill_text=skill_text,
        session=session,
        role="deployment",
        max_workers=max_workers,
    )
    wall = time.time() - t0

    outputs = {run.task_id: _score_input(run, adapter.name) for run in runs}
    result = adapter.evaluate(outputs, role="deployment")

    if transcripts_path is not None:
        path = Path(transcripts_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for run in runs:
                f.write(
                    json.dumps(
                        {
                            "task_id": run.task_id,
                            "llm_calls": run.llm_calls,
                            "error": run.error,
                            "extra": run.extra,
                            "transcript": run.transcript,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    return DeploymentResult(
        result=result,
        cost=accumulator.snapshot(),
        runs=runs,
        wall_time_s=wall,
    )
