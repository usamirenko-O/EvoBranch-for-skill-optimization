"""Resolve benchmark names to execution harnesses."""
from __future__ import annotations

from pathlib import Path

from integration.benchmarks.paths import ALFWORLD_ROOT, OFFICEQA_DOCS_ROOT
from integration.benchmarks.registry import canonical_benchmark_name
from integration.harness.base import DirectHarness

HARNESS_KIND = "direct"

#: Maximum tasks a benchmark may execute concurrently. ALFWorld is 1 on
#: purpose: each task drives a TextWorld game whose Python objects are not
#: thread-safe, and running two in one process corrupts both episodes
#: rather than failing loudly. The rest are I/O-bound on the solver call.
MAX_PARALLEL_TASKS = {
    "livemathematicianbench": 8,
    "officeqa": 8,
    "spreadsheetbench": 4,
    "alfworld": 1,
    "livecodebench": 8,
    "pubmedqa": 8,
    "chemllmbench": 8,
}


def max_parallel_tasks(benchmark: str) -> int:
    return MAX_PARALLEL_TASKS[canonical_benchmark_name(benchmark)]


def build_harness(benchmark: str, *, work_dir: Path | str, options: dict | None = None) -> DirectHarness:
    """Construct the Direct harness for `benchmark`.

    `work_dir` is where a harness may write scratch artifacts (SpreadsheetBench
    output workbooks); it is always inside the run's own directory, never
    inside `benchmarks/`.
    """
    name = canonical_benchmark_name(benchmark)
    options = dict(options or {})
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    if name == "livemathematicianbench":
        from integration.harness.livemath import LiveMathHarness

        return LiveMathHarness()
    if name == "officeqa":
        from integration.harness.officeqa import OfficeQAHarness

        return OfficeQAHarness(
            docs_root=options.get("docs_root") or str(OFFICEQA_DOCS_ROOT),
            max_tool_turns=int(options.get("max_tool_turns", 12)),
        )
    if name == "spreadsheetbench":
        from integration.harness.spreadsheetbench import SpreadsheetBenchHarness

        return SpreadsheetBenchHarness(
            work_root=str(work_dir / "spreadsheet_outputs"),
            max_turns=int(options.get("max_turns", 3)),
            exec_timeout=int(options.get("exec_timeout", 120)),
        )
    if name == "alfworld":
        from integration.harness.alfworld import ALFWorldHarness

        return ALFWorldHarness(
            alfworld_data=options.get("alfworld_data") or str(ALFWORLD_ROOT),
            max_steps=int(options.get("max_steps", 50)),
            history_length=int(options.get("history_length", 3)),
        )
    if name in ("livecodebench", "pubmedqa", "chemllmbench"):
        from integration.harness.single_turn import SingleTurnHarness

        return SingleTurnHarness(name)
    raise KeyError(f"no Direct harness registered for benchmark {name!r}")
