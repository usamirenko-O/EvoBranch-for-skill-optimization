"""Resolve repository, optimizer, and benchmark paths."""
from __future__ import annotations

import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
SKILLOPT_ROOT = WORKSPACE_ROOT / "SkillOpt"
SSO_ROOT = WORKSPACE_ROOT / "SSO"
SKILLGEN_ROOT = WORKSPACE_ROOT / "SkillGen"
BENCHMARKS_ROOT = WORKSPACE_ROOT / "benchmarks"
LIVEMATH_DATA_ROOT = BENCHMARKS_ROOT / "livemathematicianbench" / "data"


def ensure_on_sys_path(*roots: Path) -> None:
    """Idempotently prepend each root to sys.path if not already present."""
    for root in roots:
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
