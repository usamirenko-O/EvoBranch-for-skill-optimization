"""Common interfaces and records for native optimizer runs."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from integration.benchmarks.base import BenchmarkAdapter
from integration.benchmarks.manifest import SplitManifest
from integration.common.solver_client import SolverConfig
from integration.experiment.cost import PhaseCost


@dataclass
class RunContext:
    """Everything a runner is allowed to know."""

    config: object  # ExperimentConfig (avoids a circular import)
    adapter: BenchmarkAdapter
    manifest: SplitManifest
    run_dir: Path
    solver_config: SolverConfig
    harness_options: dict = field(default_factory=dict)

    @property
    def train_ids(self) -> list:
        return self.manifest.task_ids("train")

    @property
    def val_ids(self) -> list:
        return self.manifest.task_ids("val")

    @property
    def deployment_ids(self) -> list:
        return self.manifest.task_ids("deployment")

    def workdir(self, *parts: str) -> Path:
        path = self.run_dir.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path


@dataclass
class OptimizationResult:
    """What an optimizer produced, and what it cost to produce it."""

    final_skill_text: str
    cost: PhaseCost
    wall_time_s: float
    #: True when the optimizer's own internal gate accepted a changed skill.
    #: This is the OPTIMIZER'S verdict, never the benchmark metric.
    skill_accepted: bool = False
    final_skill_path: str | None = None
    #: The optimizer's own internal validation/gate outcome, preserved
    #: verbatim and reported separately from the benchmark metric.
    internal_gate: dict = field(default_factory=dict)
    #: Anything else the optimizer natively reports.
    native_summary: dict = field(default_factory=dict)


class OptimizerRunner(ABC):
    optimizer: str = ""

    @abstractmethod
    def optimize(self, ctx: RunContext) -> OptimizationResult:
        """Run the optimizer's native algorithm on train/validation only."""

    @staticmethod
    def preflight(ctx: RunContext) -> list:
        """Blocking problems specific to this optimizer, checked before spend."""
        return []
