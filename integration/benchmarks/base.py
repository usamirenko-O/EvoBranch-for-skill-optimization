"""Common interfaces for benchmark tasks, splits, and evaluation."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

from integration.benchmarks.manifest import SplitManifest


@dataclass(frozen=True)
class BenchmarkTask:
    """One task handed to an optimizer.

    `payload` is benchmark-shaped on purpose -- a LiveMath task is a rendered
    multiple-choice prompt, a SpreadsheetBench task is an instruction plus
    workbook paths, an ALFWorld task is a game file. Flattening those into a
    single "prompt" string would be inventing a harness, which this phase
    does not do. Each adapter documents its own payload keys.
    """

    task_id: str
    native_id: str
    split_role: str
    benchmark: str
    payload: dict = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkResult:
    """The official aggregate result for one evaluated split."""

    metric_name: str
    metric_value: float | None
    n_tasks: int
    scorer_ref: str
    secondary_metrics: dict = field(default_factory=dict)
    per_task: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ResourceStatus:
    """What is and is not available locally for this benchmark, right now.

    Reported honestly by every adapter so the dry-run can say "this
    combination cannot execute yet, and here is exactly what is missing"
    instead of failing halfway through a paid run. `blocking` means a real
    run cannot proceed; `degraded` means it can, with a documented caveat.
    """

    benchmark: str
    ready: bool
    blocking: list = field(default_factory=list)
    degraded: list = field(default_factory=list)
    details: dict = field(default_factory=dict)


class BenchmarkAdapter(ABC):
    """One benchmark, one frozen manifest, one official scorer."""

    #: benchmark name as used everywhere else in the experiment matrix
    name: str = ""
    #: the benchmark's own name for its headline metric
    metric_name: str = ""
    #: "module:function" of the official scorer, or "environment-native:<what>"
    scorer_ref: str = ""
    #: the manifest this adapter is pinned to
    manifest_id: str = ""

    # ---- splits -------------------------------------------------------

    @abstractmethod
    def manifest(self) -> SplitManifest:
        """Return the frozen SplitManifest. Never samples, never reshuffles."""

    def task_ids(self, role: str) -> list[str]:
        return self.manifest().task_ids(role)

    @abstractmethod
    def tasks(self, role: str) -> list[BenchmarkTask]:
        """Materialize the tasks for one split role, in manifest order."""

    @abstractmethod
    def population(self) -> list[str]:
        """Every task id this benchmark can actually execute locally, sorted.

        This is what a user-supplied split manifest is validated against: an
        id outside this set is rejected before a run starts, rather than
        producing a partial evaluation that still looks like a number.
        """

    def tasks_for_ids(self, task_ids, role: str = "custom") -> list[BenchmarkTask]:
        """Materialize an arbitrary id list, in the order given.

        Used by the shared deployment harness and by optimizer adapters that
        need a subset (a minibatch, a probe set) of a split. It reads the
        benchmark population directly, so no caller can smuggle in a task the
        manifest never named -- `run_baseline` validates the manifest against
        `population()` first.
        """
        raise NotImplementedError

    # ---- official scoring ---------------------------------------------

    @abstractmethod
    def evaluate(self, task_outputs: Mapping[str, Any], *, role: str = "deployment") -> BenchmarkResult:
        """Score `{task_id: output}` with the benchmark's OWN scorer.

        The accepted shape of `output` is benchmark-specific and documented
        on each adapter. Implementations must not re-derive a verdict that
        the official scorer already produces.
        """

    # ---- honesty about what is on disk --------------------------------

    @abstractmethod
    def resource_status(self) -> ResourceStatus:
        """Report missing local datasets/packages instead of fabricating them."""

    # ---- shared helpers ------------------------------------------------

    def _aggregate(
        self,
        per_task: dict,
        *,
        score_key: str = "score",
        secondary: dict | None = None,
    ) -> BenchmarkResult:
        """Mean of the official per-task scores. No re-weighting, no clipping."""
        scores = [v[score_key] for v in per_task.values() if v.get(score_key) is not None]
        value = (sum(scores) / len(scores)) if scores else None
        return BenchmarkResult(
            metric_name=self.metric_name,
            metric_value=value,
            n_tasks=len(per_task),
            scorer_ref=self.scorer_ref,
            secondary_metrics=dict(secondary or {}),
            per_task=per_task,
        )

    def _check_outputs_against_split(self, task_outputs: Mapping[str, Any], role: str) -> None:
        """Fail loudly if a caller scores tasks that are not in this split.

        A silent partial evaluation (or an accidental train-on-test) is worse
        than an exception here: the resulting number would look valid.
        """
        expected = set(self.manifest().task_ids(role))
        got = set(task_outputs)
        unknown = got - expected
        if unknown:
            raise KeyError(
                f"{self.name}: {len(unknown)} output task_id(s) are not in the "
                f"{role!r} split of manifest {self.manifest_id!r} "
                f"(first 5: {sorted(unknown)[:5]})"
            )

    def _check_outputs_against_population(self, task_outputs: Mapping[str, Any]) -> None:
        """Fail loudly if a caller scores an id this benchmark does not have.

        `evaluate` takes an explicit id list rather than a split role, because
        deployment scores exactly the ids the run's manifest named. The guard
        therefore has to be the benchmark population, not one split.
        """
        unknown = set(task_outputs) - set(self.population())
        if unknown:
            raise KeyError(
                f"{self.name}: {len(unknown)} output task_id(s) are not in the "
                f"benchmark population (first 5: {sorted(unknown)[:5]})"
            )
