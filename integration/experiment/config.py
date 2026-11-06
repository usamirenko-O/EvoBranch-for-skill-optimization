"""Resolve the benchmark, solver, and optimizer configuration for one run."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from dataclasses import asdict, dataclass, field
from pathlib import Path

from integration.benchmarks.base import BenchmarkAdapter
from integration.benchmarks.registry import (
    BENCHMARK_NAMES,
    HARNESSES,
    OPTIMIZERS,
    canonical_benchmark_name,
    get_adapter,
)
from integration.common.paths import WORKSPACE_ROOT
from integration.common.solver_client import (
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_SECONDS,
    SolverConfig,
)

DEFAULT_OUT_ROOT = WORKSPACE_ROOT / "integration" / "runs"


DEFAULT_OPTIMIZER_MODEL = "optimizer-llm"

#: Repos whose HEAD is recorded in every run's `code_freeze`.
_FREEZE_REPOS = ("SkillOpt", "SSO", "SkillGen")

#: Shared infrastructure identified by its source-content digest.
_FREEZE_TREES = (
    "integration/benchmarks",
    "integration/experiment",
    "integration/common",
)


@dataclass(frozen=True)
class ExperimentConfig:
    optimizer: str
    benchmark: str
    solver_model: str
    harness: str = "direct"
    split_role: str = "deployment"
    optimizer_model: str = DEFAULT_OPTIMIZER_MODEL
    proxy_base_url: str = DEFAULT_BASE_URL
    api_key: str = DEFAULT_API_KEY
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    optimizer_max_tokens: int = DEFAULT_MAX_TOKENS
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES
    out_root: Path = DEFAULT_OUT_ROOT
    run_id: str = ""
    #: Path to the split manifest governing this run. None == the benchmark's
    #: checked-in default manifest.
    split_manifest_path: str | None = None
    #: Per-benchmark harness knobs (max tool turns, env steps, workers, ...).
    harness_options: dict = field(default_factory=dict)
    #: How many deployment tasks to execute concurrently.
    deployment_workers: int = 1
    #: Optimizer-specific knobs, namespaced by optimizer (see each runner).
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(
                f"optimizer must be one of {OPTIMIZERS}, got {self.optimizer!r}. "
                "This launcher serves native baselines only; OURS is never routed "
                "through it."
            )
        if self.harness not in HARNESSES:
            raise ValueError(f"harness must be one of {HARNESSES}, got {self.harness!r}")
        object.__setattr__(self, "benchmark", canonical_benchmark_name(self.benchmark))
        if not self.solver_model:
            raise ValueError("solver_model (a proxy alias) is required")
        if not self.optimizer_model:
            # Never silently fall back to the solver alias: that would conflate
            # the two roles the experiment is trying to keep apart, and the
            # artifact would record the wrong thing. Running both roles on one
            # model is a legitimate configuration -- it just has to be said.
            raise ValueError(
                "optimizer_model is empty. Pass an alias explicitly (use the same "
                "alias as --solver-model if you really want one model for both roles)."
            )
        if not self.run_id:
            object.__setattr__(self, "run_id", self.default_run_id())
        object.__setattr__(self, "out_root", Path(self.out_root))

    # ---- derived ----

    def default_run_id(self) -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        return f"{self.optimizer}__{self.benchmark}__{self.harness}__{self.solver_model}__{stamp}"

    def adapter(self) -> BenchmarkAdapter:
        return get_adapter(self.benchmark)

    def solver_config(self) -> SolverConfig:
        """Build the frozen shared solver call contract for this run."""
        return SolverConfig(
            base_url=self.proxy_base_url,
            model=self.solver_model,
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
        )

    def run_dir(self) -> Path:
        return self.out_root / self.run_id

    def artifact_path(self) -> Path:
        return self.run_dir() / "experiment_record.json"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["out_root"] = str(self.out_root)
        # never persist a credential, even the local proxy's placeholder
        d["api_key"] = "***"
        return d


# --------------------------------------------------------------- freeze ----


def _git(args: list, cwd: Path) -> str:
    if not (cwd / ".git").exists():
        return ""
    try:
        return subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=False, timeout=15
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _tree_digest(root: Path) -> str:
    """sha256 over (relative path, file sha256) for every .py under `root`."""
    h = hashlib.sha256()
    if not root.is_dir():
        return ""
    for p in sorted(root.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        h.update(str(p.relative_to(root)).encode("utf-8"))
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def code_freeze_reference(workspace_root: Path = WORKSPACE_ROOT) -> dict:
    """Record optimizer Git revisions and working-tree modification counts when available."""
    repos = {}
    for name in _FREEZE_REPOS:
        root = workspace_root / name
        head = _git(["rev-parse", "HEAD"], root)
        dirty = _git(["status", "--porcelain"], root)
        repos[name] = {
            "head": head or None,
            "dirty_files": len([ln for ln in dirty.splitlines() if ln.strip()]) if head else None,
        }
    trees = {rel: _tree_digest(workspace_root / rel) for rel in _FREEZE_TREES}
    return {
        "optimizer_repos": repos,
        "infrastructure_tree_sha256": trees,
        "note": "Optimizer Git revisions are recorded when available; "
        "shared infrastructure is identified by content digests.",
    }


def optimizer_side_env(config: ExperimentConfig) -> dict:
    """Environment variables an optimizer's own client reads, so that its
    optimizer-side calls reach the same proxy as the solver.

    SkillGen's `llm.py` chooses its endpoint from `SKILLGEN_ROUTER_BASE_URL`
    (and `SKILLGEN_EMBEDDING_BASE_URL` for embeddings) -- both pre-existing
    hooks in that file, not added here. Setting them is what stops SkillGen
    from silently falling back to OpenRouter with its own default models.
    """
    return {
        "SKILLGEN_ROUTER_BASE_URL": config.proxy_base_url,
        "SKILLGEN_ROUTER_API_KEY": config.api_key,
        "SKILLGEN_EMBEDDING_BASE_URL": config.proxy_base_url,
        "SKILLGEN_EMBEDDING_API_KEY": config.api_key,
    }


def load_config(path: Path | str) -> ExperimentConfig:
    """Load an ExperimentConfig from a JSON file (the config-file entry point)."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    payload.pop("api_key", None)  # never taken from a checked-in file
    return ExperimentConfig(**payload)


__all__ = [
    "BENCHMARK_NAMES",
    "DEFAULT_OPTIMIZER_MODEL",
    "optimizer_side_env",
    "ExperimentConfig",
    "OPTIMIZERS",
    "code_freeze_reference",
    "load_config",
]
