"""Load LiveMath tasks, resolve shared splits, and score multiple-choice answers."""
from __future__ import annotations

import json
from typing import Any, Mapping

from integration.benchmarks.base import (
    BenchmarkAdapter,
    BenchmarkResult,
    BenchmarkTask,
    ResourceStatus,
)
from integration.benchmarks.manifest import (
    SPLIT_ROLES,
    SplitEntry,
    SplitManifest,
    load_manifest,
    source_inputs,
)
from integration.benchmarks.paths import LIVEMATH_ID_SPLIT, MANIFEST_DIR
from integration.benchmarks.livemath_choices import (
    DEFAULT_CHOICE_SHUFFLE_SEED,
    shuffle_item,
    shuffle_seed_for,
)
from integration.common.livemath_data import _to_item, render_prompt
from integration.common.livemath_scoring import score as official_score
from integration.common.paths import LIVEMATH_DATA_ROOT

MANIFEST_ID = "livemath_frozen_4month_177_v2_val35"


VAL_EXTENSION_N = 18
VAL_EXTENSION_SEED = 42
METRIC_NAME = "livemath_accuracy_exact_choice_label"
SCORER_REF = "integration.common.livemath_scoring:score"

_ROLE_TO_SKILLOPT_SPLIT = {"train": "train", "val": "val", "deployment": "test"}


def _skillopt_manifest_items(split: str) -> list[dict]:
    with (LIVEMATH_ID_SPLIT / split / "items.json").open(encoding="utf-8") as f:
        return json.load(f)


def build_manifest() -> SplitManifest:
    upstream = json.loads((LIVEMATH_ID_SPLIT / "split_manifest.json").read_text(encoding="utf-8"))
    splits: dict = {}
    for role in SPLIT_ROLES:
        so_split = _ROLE_TO_SKILLOPT_SPLIT[role]
        splits[role] = [
            SplitEntry(
                task_id=str(item["id"]),
                native_id=str(item["id"]),
                native_split="",  # monthly files, no native train/val/test
                extra={
                    "month": item.get("month", ""),
                    "no": item.get("no"),
                    "source_file": item.get("source_file", ""),
                    "skillopt_split": so_split,
                },
            )
            for item in _skillopt_manifest_items(so_split)
        ]

    import random

    dep_ids = sorted(e.task_id for e in splits["deployment"])
    moved = set(random.Random(VAL_EXTENSION_SEED).sample(dep_ids, VAL_EXTENSION_N))
    by_id = {e.task_id: e for e in splits["deployment"]}
    splits["val"] = list(splits["val"]) + [
        SplitEntry(task_id=t, native_id=t, native_split="",
                   extra={**by_id[t].extra, "moved_from": "upstream_test"})
        for t in sorted(moved)
    ]
    splits["deployment"] = [e for e in splits["deployment"] if e.task_id not in moved]

    return SplitManifest(
        benchmark="livemathematicianbench",
        manifest_id=MANIFEST_ID,
        version="1",
        source={
            "upstream_dataset": upstream.get("source_repo"),
            "upstream_url": upstream.get("source_url"),
            "upstream_revision": upstream.get("source_revision"),
            "upstream_files": upstream.get("source_files"),
            "materialized_from": str(LIVEMATH_DATA_ROOT),
            "frozen_partition": str(LIVEMATH_ID_SPLIT),
            "inputs": source_inputs(
                [
                    LIVEMATH_ID_SPLIT / "split_manifest.json",
                    LIVEMATH_ID_SPLIT / "train" / "items.json",
                    LIVEMATH_ID_SPLIT / "val" / "items.json",
                    LIVEMATH_ID_SPLIT / "test" / "items.json",
                ]
            ),
            "official_scorer": {
                "ref": SCORER_REF,
                "upstream": "SkillOpt envs/livemathematicianbench/evaluator.py "
                "(parse_choice_label), used by import",
                "note": "frozen Phase-1 decision; this phase does not change it.",
            },
        },
        role_map={
            "train": "SkillOpt livemathematicianbench_id_split/train (35)",
            "val": "SkillOpt livemathematicianbench_id_split/val (17) + 18 tasks moved from upstream test (seed 42) = 35",
            "deployment": "SkillOpt livemathematicianbench_id_split/test (125) minus the 18 moved to val = 107",
            "native_splits": "none -- LiveMath publishes monthly qa_*_final.json files",
        },
        sampling={
            "method": "frozen_upstream_manifest_val_extended_seeded",
            "val_extension": {"n": VAL_EXTENSION_N, "seed": VAL_EXTENSION_SEED, "from": "upstream test split"},
            # The seed is upstream SkillOpt's, recorded because the partition
            # itself was produced by a seeded ratio split, not by this builder.
            "seed": upstream.get("split_seed"),
            "upstream_split_mode": upstream.get("split_mode"),
            "upstream_split_ratio": upstream.get("split_ratio"),
            "choice_shuffle_seed": DEFAULT_CHOICE_SHUFFLE_SEED,
        },
        splits=splits,
        notes=[
            "Validation contains 17 upstream validation tasks and 18 tasks sampled from the upstream test split with seed 42; deployment contains the remaining 107 tasks.",
            "Task population: identical task IDs to what "
            "integration/skillopt_livemath/join_livemath_split.py already produces.",
            "Covers 4 of the 8 monthly files present locally (177 of 608 items).",
            "Choices are deterministically shuffled (seed in sampling.choice_shuffle_seed). "
            "The raw corpus lists the correct option first in all 608 items, so an "
            "unshuffled presentation would score 100% for anything that always answers A.",
        ],
    ).with_hash()


class LiveMathAdapter(BenchmarkAdapter):
    name = "livemathematicianbench"
    metric_name = METRIC_NAME
    scorer_ref = SCORER_REF
    manifest_id = MANIFEST_ID

    def __init__(self) -> None:
        self._manifest: SplitManifest | None = None
        self._items: dict | None = None
        self._seed: int | None = None

    def manifest(self) -> SplitManifest:
        if self._manifest is None:
            self._manifest = load_manifest(MANIFEST_DIR / f"{MANIFEST_ID}.json")
        return self._manifest

    def _raw_items(self) -> dict:
        if self._items is None:
            from skillopt.envs.livemathematicianbench.dataloader import load_items

            self._items = {str(i["id"]): i for i in load_items(str(LIVEMATH_DATA_ROOT))}
        return self._items

    def _item(self, task_id: str):
        """One item, with its choices deterministically shuffled.

        The raw corpus lists the correct option first in all 608 items, so
        rendering file order would make the benchmark degenerate (always "A").
        The shuffle is SkillOpt's own, seeded by the manifest's
        `choice_shuffle_seed`, so every optimizer sees the identical question.
        """
        raw = self._raw_items()[task_id]
        return _to_item(shuffle_item(raw, self._shuffle_seed()))

    def _shuffle_seed(self) -> int:
        if self._seed is None:
            try:
                self._seed = shuffle_seed_for(self.manifest())
            except FileNotFoundError:
                self._seed = DEFAULT_CHOICE_SHUFFLE_SEED
        return self._seed

    def tasks(self, role: str) -> list[BenchmarkTask]:
        """Payload keys: `prompt` (rendered by the existing shared
        `render_prompt`), `month`, `no`. Ground truth stays out, as it
        already does in SSO's LiveMath loader."""
        return [self._task(e.task_id, role) for e in self.manifest().entries(role)]

    def _task(self, task_id: str, role: str) -> BenchmarkTask:
        item = self._item(task_id)
        return BenchmarkTask(
            task_id=task_id,
            native_id=task_id,
            split_role=role,
            benchmark=self.name,
            payload={"prompt": render_prompt(item), "month": item.month, "no": item.no},
        )

    def tasks_for_ids(self, task_ids, role: str = "custom") -> list[BenchmarkTask]:
        return [self._task(str(t), role) for t in task_ids]

    def population(self) -> list[str]:
        return sorted(self._raw_items())

    def evaluate(self, task_outputs: Mapping[str, Any], *, role: str = "deployment") -> BenchmarkResult:
        """`task_outputs` maps task_id -> the solver's raw response text."""
        self._check_outputs_against_population(task_outputs)
        per_task = {}
        for task_id, raw in task_outputs.items():
            item = self._item(task_id)
            verdict = official_score("" if raw is None else str(raw), list(item.choices), item.correct_label)
            per_task[task_id] = verdict
        return self._aggregate(per_task)

    def resource_status(self) -> ResourceStatus:
        blocking = []
        if not LIVEMATH_DATA_ROOT.is_dir():
            blocking.append(f"missing LiveMath data root: {LIVEMATH_DATA_ROOT}")
        n_months = len(list(LIVEMATH_DATA_ROOT.glob("*/qa_*_final.json"))) if LIVEMATH_DATA_ROOT.is_dir() else 0
        return ResourceStatus(
            benchmark=self.name,
            ready=not blocking,
            blocking=blocking,
            degraded=[
                "The shared split contains 177 tasks from four monthly files."
            ],
            details={"n_monthly_files_present": n_months},
        )
