"""Resolve optimizer names to native runners."""
from __future__ import annotations

from integration.experiment.runners.base import (
    OptimizationResult,
    OptimizerRunner,
    RunContext,
)


def get_runner(optimizer: str) -> OptimizerRunner:
    if optimizer == "skillopt":
        from integration.experiment.runners.skillopt import SkillOptRunner

        return SkillOptRunner()
    if optimizer == "sso":
        from integration.experiment.runners.sso import SSORunner

        return SSORunner()
    if optimizer == "skillgen":
        from integration.experiment.runners.skillgen import SkillGenRunner

        return SkillGenRunner()
    raise KeyError(f"no runner registered for optimizer {optimizer!r}")


def runner_class(optimizer: str):
    """The runner class without instantiating it (for preflight/tests)."""
    return type(get_runner(optimizer))


__all__ = ["OptimizationResult", "OptimizerRunner", "RunContext", "get_runner", "runner_class"]
