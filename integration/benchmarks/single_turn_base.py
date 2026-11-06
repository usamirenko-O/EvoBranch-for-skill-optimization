"""Shared task loading and evaluation for single-turn benchmarks."""
from __future__ import annotations

import random
from typing import Any, Callable, Mapping

from integration.benchmarks.base import BenchmarkAdapter, BenchmarkResult, BenchmarkTask, ResourceStatus
from integration.benchmarks.manifest import SPLIT_ROLES, SplitEntry, SplitManifest, load_manifest
from integration.benchmarks.paths import MANIFEST_DIR
from integration.common.paths import SKILLOPT_ROOT, ensure_on_sys_path

ensure_on_sys_path(SKILLOPT_ROOT)


SPLIT_SIZES = {"train": 35, "val": 35, "deployment": 107}
#: The shared split seed (LiveMath's upstream split_seed is also 42).
SPLIT_SEED = 42


# ------------------------------------------------------------ samplers ----


def uniform_split(population: list, sizes: dict = SPLIT_SIZES, seed: int = SPLIT_SEED) -> dict:
    """One seeded draw of sum(sizes) ids from the sorted population, cut into
    the three roles in order; each role's list is then sorted."""
    ids = sorted(str(t) for t in population)
    total = sum(sizes[r] for r in SPLIT_ROLES)
    if total > len(ids):
        raise ValueError(f"population has {len(ids)} ids, need {total}")
    chosen = random.Random(seed).sample(ids, total)
    out, cursor = {}, 0
    for role in SPLIT_ROLES:
        out[role] = sorted(chosen[cursor: cursor + sizes[role]])
        cursor += sizes[role]
    return out


def _largest_remainder(total: int, weights: list) -> list:
    """Apportion `total` over `weights` (Hamilton's method)."""
    denom = float(sum(weights))
    raw = [total * w / denom for w in weights]
    counts = [int(r) for r in raw]
    order = sorted(range(len(raw)), key=lambda i: (raw[i] - counts[i], weights[i]), reverse=True)
    for i in order[: total - sum(counts)]:
        counts[i] += 1
    return counts


def stratified_split(
    population_by_stratum: dict, sizes: dict = SPLIT_SIZES, seed: int = SPLIT_SEED
) -> tuple:
    """Per-role quotas follow the population's stratum proportions (largest
    remainder); within a stratum one seeded shuffle is cut train / val /
    deployment in order. Returns (splits, quota_table)."""
    strata = sorted(population_by_stratum)
    weights = [len(population_by_stratum[s]) for s in strata]
    quotas = {role: dict(zip(strata, _largest_remainder(sizes[role], weights))) for role in SPLIT_ROLES}
    splits = {role: [] for role in SPLIT_ROLES}
    for s in strata:
        ids = sorted(str(t) for t in population_by_stratum[s])
        need = sum(quotas[r][s] for r in SPLIT_ROLES)
        if need > len(ids):
            raise ValueError(f"stratum {s!r} has {len(ids)} ids, need {need}")
        rng = random.Random(f"{seed}:{s}")
        rng.shuffle(ids)
        cursor = 0
        for role in SPLIT_ROLES:
            splits[role].extend(ids[cursor: cursor + quotas[role][s]])
            cursor += quotas[role][s]
    return {role: sorted(splits[role]) for role in SPLIT_ROLES}, quotas


def entries_for(ids: list, extra_for: Callable[[str], dict]) -> list:
    return [SplitEntry(task_id=t, native_id=t, native_split="", extra=extra_for(t)) for t in ids]


# ------------------------------------------------------------- adapter ----


class SingleTurnBenchmarkAdapter(BenchmarkAdapter):
    """Skeleton: subclasses set the class attributes and the four hooks."""

    #: absolute data path handed to the SkillOpt env loader
    data_path: str = ""
    #: SkillOpt env functions (set by subclasses at import time)
    _load_items: Callable[[str], list] = None  # type: ignore[assignment]
    _build_user: Callable[[dict], str] = None  # type: ignore[assignment]
    _evaluate: Callable[[str, dict], dict] = None  # type: ignore[assignment]
    #: payload keys copied from the item next to the rendered prompt
    payload_meta_keys: tuple = ()

    def __init__(self) -> None:
        self._manifest: SplitManifest | None = None
        self._items: dict | None = None

    def manifest(self) -> SplitManifest:
        if self._manifest is None:
            self._manifest = load_manifest(MANIFEST_DIR / f"{self.manifest_id}.json")
        return self._manifest

    def items_by_id(self) -> dict:
        if self._items is None:
            self._items = {str(i["id"]): i for i in type(self)._load_items(self.data_path)}
        return self._items

    def item(self, task_id: str) -> dict:
        return self.items_by_id()[str(task_id)]

    def _task(self, task_id: str, role: str) -> BenchmarkTask:
        item = self.item(task_id)
        payload = {"prompt": type(self)._build_user(item)}
        for key in self.payload_meta_keys:
            payload[key] = item.get(key)
        return BenchmarkTask(task_id=str(task_id), native_id=str(task_id), split_role=role,
                             benchmark=self.name, payload=payload)

    def tasks(self, role: str) -> list:
        return [self._task(e.task_id, role) for e in self.manifest().entries(role)]

    def tasks_for_ids(self, task_ids, role: str = "custom") -> list:
        return [self._task(str(t), role) for t in task_ids]

    def population(self) -> list:
        return sorted(self.items_by_id())

    def evaluate(self, task_outputs: Mapping[str, Any], *, role: str = "deployment") -> BenchmarkResult:
        """`task_outputs` maps task_id -> the solver's raw response text."""
        self._check_outputs_against_population(task_outputs)
        per_task = {}
        for task_id, raw in task_outputs.items():
            verdict = type(self)._evaluate("" if raw is None else str(raw), self.item(task_id))
            per_task[str(task_id)] = {k: v for k, v in verdict.items() if k != "detail"}
        return self._aggregate(per_task)

    def resource_status(self) -> ResourceStatus:
        blocking = []
        try:
            n = len(self.items_by_id())
        except Exception as exc:  # noqa: BLE001
            blocking.append(f"{self.name}: cannot load population from {self.data_path}: {exc}")
            n = 0
        return ResourceStatus(benchmark=self.name, ready=not blocking, blocking=blocking,
                              details={"population": n, "data_path": self.data_path})
