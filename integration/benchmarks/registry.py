"""Resolve benchmark names to their adapters."""
from __future__ import annotations

from integration.benchmarks.alfworld import ALFWorldAdapter
from integration.benchmarks.base import BenchmarkAdapter
from integration.benchmarks.chemllmbench import ChemLLMBenchAdapter
from integration.benchmarks.livecodebench import LiveCodeBenchAdapter
from integration.benchmarks.livemath import LiveMathAdapter
from integration.benchmarks.officeqa import OfficeQAAdapter
from integration.benchmarks.pubmedqa import PubMedQAAdapter
from integration.benchmarks.spreadsheetbench import SpreadsheetBenchAdapter

#: benchmark name -> adapter class
BENCHMARK_ADAPTERS: dict = {
    LiveMathAdapter.name: LiveMathAdapter,
    OfficeQAAdapter.name: OfficeQAAdapter,
    SpreadsheetBenchAdapter.name: SpreadsheetBenchAdapter,
    ALFWorldAdapter.name: ALFWorldAdapter,

    LiveCodeBenchAdapter.name: LiveCodeBenchAdapter,
    PubMedQAAdapter.name: PubMedQAAdapter,
    ChemLLMBenchAdapter.name: ChemLLMBenchAdapter,
}

BENCHMARK_NAMES = tuple(sorted(BENCHMARK_ADAPTERS))

#: Convenience aliases accepted on the command line. The canonical name is
#: always what gets persisted into a result artifact.
BENCHMARK_ALIASES: dict = {
    "livemath": "livemathematicianbench",
    "livemathbench": "livemathematicianbench",
    "ssb": "spreadsheetbench",
    "livecode": "livecodebench",
    "lcb": "livecodebench",
    "pubmed": "pubmedqa",
    "chem": "chemllmbench",
    "chemllm": "chemllmbench",
    "chemyield": "chemllmbench",
}

#: The three native baselines this infrastructure serves. `ours` is
#: deliberately absent -- native baselines must never be routed through it.
OPTIMIZERS = ("skillopt", "sso", "skillgen")

HARNESSES = ("direct",)


def canonical_benchmark_name(name: str) -> str:
    key = name.strip().lower()
    key = BENCHMARK_ALIASES.get(key, key)
    if key not in BENCHMARK_ADAPTERS:
        raise KeyError(f"unknown benchmark {name!r}; known: {', '.join(BENCHMARK_NAMES)}")
    return key


def get_adapter(name: str, **kwargs) -> BenchmarkAdapter:
    return BENCHMARK_ADAPTERS[canonical_benchmark_name(name)](**kwargs)
