"""Load ALFWorld game splits and evaluate skills in the TextWorld environment."""
from __future__ import annotations

import json
import os
import pathlib
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
from integration.benchmarks.paths import (
    ALFWORLD_JSON_ROOT,
    ALFWORLD_LOGIC_ROOT,
    ALFWORLD_PATH_SPLIT,
    ALFWORLD_ROOT,
    MANIFEST_DIR,
)

MANIFEST_ID = "alfworld_default_191_v2"
METRIC_NAME = "alfworld_success_rate"
SCORER_REF = "environment-native:alfworld TextWorld episode 'won' flag"

_ROLE_TO_SKILLOPT_SPLIT = {"train": "train", "val": "val", "deployment": "test"}
_ROLE_TO_NATIVE_SPLIT = {"train": "train", "val": "valid_seen", "deployment": "valid_unseen"}


# ------------------------------------------------------------- id scheme --


def task_id_for_gamefile(gamefile: str) -> str:
    """`json_2.1.1/<split>/<task>/<trial>/game.tw-pddl` -> `<split>/<task>/<trial>`."""
    rel = gamefile
    if rel.startswith("json_2.1.1/"):
        rel = rel[len("json_2.1.1/") :]
    if rel.endswith("/game.tw-pddl"):
        rel = rel[: -len("/game.tw-pddl")]
    return rel


def gamefile_for_task_id(task_id: str) -> str:
    """Inverse of `task_id_for_gamefile`."""
    return f"json_2.1.1/{task_id}/game.tw-pddl"


def alfworld_data_root() -> str:
    """Where the ALFWorld corpus actually is.

    `$ALFWORLD_DATA` is NOT read unconditionally: `import alfworld.info` sets
    it to the package's own cache directory (`~/.cache/alfworld`) if it is
    unset, so any process that has touched the alfworld package inherits a
    path that usually holds no data at all. The env var wins only when it
    really points at a corpus; otherwise this workspace's own copy does.
    """
    candidates = [os.environ.get("ALFWORLD_DATA"), str(ALFWORLD_ROOT)]
    for candidate in candidates:
        if candidate and os.path.isdir(os.path.join(candidate, "json_2.1.1")):
            return candidate
    return os.environ.get("ALFWORLD_DATA") or str(ALFWORLD_ROOT)


def task_type_of(task_id: str) -> str:
    """ALFWorld encodes the task type in the task directory name."""
    parts = task_id.split("/")
    return parts[1].split("-")[0] if len(parts) > 1 else ""


def native_split_of(task_id: str) -> str:
    return task_id.split("/")[0]


def _skillopt_manifest_items(split: str) -> list[dict]:
    with (ALFWORLD_PATH_SPLIT / split / "items.json").open(encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------- manifest build --


def build_manifest() -> SplitManifest:
    upstream = json.loads((ALFWORLD_PATH_SPLIT / "split_manifest.json").read_text(encoding="utf-8"))

    splits: dict = {}
    native_mismatch: dict = {}
    for role in SPLIT_ROLES:
        so_split = _ROLE_TO_SKILLOPT_SPLIT[role]
        expected_native = _ROLE_TO_NATIVE_SPLIT[role]
        entries, mismatched = [], []
        for item in _skillopt_manifest_items(so_split):
            gamefile = str(item["gamefile"])
            task_id = task_id_for_gamefile(gamefile)
            native = native_split_of(task_id)
            if native != expected_native:
                mismatched.append(gamefile)
            entries.append(
                SplitEntry(
                    task_id=task_id,
                    native_id=gamefile,
                    native_split=native,
                    extra={"task_type": task_type_of(task_id)},
                )
            )
        splits[role] = entries
        native_mismatch[role] = mismatched

    return SplitManifest(
        benchmark="alfworld",
        manifest_id=MANIFEST_ID,
        version="2",
        source={
            "upstream_repo": upstream.get("source_repo"),
            "upstream_url": upstream.get("source_url"),
            "upstream_file": upstream.get("source_file"),
            "compiled_games": "https://github.com/alfworld/alfworld/releases/download/0.4.0/"
            "json_2.1.2_tw-pddl.zip",
            "materialized_from": str(ALFWORLD_JSON_ROOT),
            "default_partition_seeded_from": str(ALFWORLD_PATH_SPLIT),
            "inputs": source_inputs(
                [
                    ALFWORLD_PATH_SPLIT / "split_manifest.json",
                    ALFWORLD_PATH_SPLIT / "train" / "items.json",
                    ALFWORLD_PATH_SPLIT / "val" / "items.json",
                    ALFWORLD_PATH_SPLIT / "test" / "items.json",
                ]
            ),
            "official_scorer": {
                "ref": SCORER_REF,
                "upstream": "https://github.com/alfworld/alfworld",
                "note": "success is decided by the simulator, not by a scoring "
                "function; no scorer is implemented in this workspace.",
            },
        },
        role_map={
            "train": "native json_2.1.1/train",
            "val": "native json_2.1.1/valid_seen -- in-distribution",
            "deployment": "native json_2.1.1/valid_unseen -- out-of-distribution "
            "(the full 134-game standard eval set)",
            "native_splits": "train / valid_seen / valid_unseen (+ non-standard valid_train)",
        },
        sampling={
            "method": "default_manifest_seeded_from_frozen_partition",
            "seed": None,
            "native_split_mismatches": {role: len(v) for role, v in native_mismatch.items()},
        },
        splits=splits,
        notes=[
            "Default manifest only. The run-time split is whatever manifest the "
            "launcher was given; this one is the checked-in default/example.",
            "Deployment == the complete compiled valid_unseen set (134 games), "
            "ALFWorld's standard out-of-distribution evaluation set.",
            "Task ids are trial paths relative to json_2.1.1/, so a user can "
            "write a split by hand without consulting any other file.",
        ],
    ).with_hash()


# ------------------------------------------------------------------ adapter --


class ALFWorldAdapter(BenchmarkAdapter):
    name = "alfworld"
    metric_name = METRIC_NAME
    scorer_ref = SCORER_REF
    manifest_id = MANIFEST_ID

    def __init__(self) -> None:
        self._manifest: SplitManifest | None = None
        self._population: list | None = None

    def manifest(self) -> SplitManifest:
        if self._manifest is None:
            self._manifest = load_manifest(MANIFEST_DIR / f"{MANIFEST_ID}.json")
        return self._manifest

    def _task(self, task_id: str, role: str) -> BenchmarkTask:
        data_root = alfworld_data_root()
        gamefile = gamefile_for_task_id(task_id)
        gamefile_abs = os.path.join(data_root, gamefile)
        return BenchmarkTask(
            task_id=task_id,
            native_id=gamefile,
            split_role=role,
            benchmark=self.name,
            payload={
                "gamefile": gamefile,
                "gamefile_abs": gamefile_abs,
                "trial_dir": os.path.dirname(gamefile_abs),
                "task_type": task_type_of(task_id),
                "native_split": native_split_of(task_id),
            },
        )

    def tasks(self, role: str) -> list[BenchmarkTask]:
        """Payload keys: `gamefile` (relative to `$ALFWORLD_DATA`),
        `gamefile_abs`, `trial_dir`, `task_type`, `native_split`.

        No natural-language goal string is materialized here: ALFWorld's goal
        text is emitted by the environment on `reset()`, and reading it out of
        `traj_data.json` instead would be a second, unofficial task rendering."""
        return [self._task(e.task_id, role) for e in self.manifest().entries(role)]

    def tasks_for_ids(self, task_ids, role: str = "custom") -> list[BenchmarkTask]:
        return [self._task(str(t), role) for t in task_ids]

    def population(self) -> list[str]:
        """Return and cache trials with compiled game.tw-pddl files."""
        if self._population is None:
            root = pathlib.Path(alfworld_data_root()) / "json_2.1.1"
            if not root.is_dir():
                self._population = []
            else:
                self._population = sorted(
                    str(p.parent.relative_to(root)) for p in root.rglob("game.tw-pddl")
                )
        return list(self._population)

    def evaluate(self, task_outputs: Mapping[str, Any], *, role: str = "deployment") -> BenchmarkResult:
        """`task_outputs` maps task_id -> the episode outcome the ALFWorld
        environment produced, which MUST contain a boolean `won`.

        Raises `ValueError` on an outcome without `won`: success in ALFWorld
        is the simulator's verdict, and deriving it here from a transcript
        would be exactly the substitute scorer the adapter contract forbids.
        """
        self._check_outputs_against_population(task_outputs)
        per_task: dict = {}
        by_task_type: dict = {}
        for task_id, outcome in task_outputs.items():
            if not isinstance(outcome, Mapping) or "won" not in outcome:
                raise ValueError(
                    f"alfworld: outcome for {task_id!r} has no 'won' key. ALFWorld "
                    "success is the environment's verdict; this adapter will not "
                    "infer it from a transcript."
                )
            won = bool(outcome["won"])
            per_task[task_id] = {
                "score": 1.0 if won else 0.0,
                "won": won,
                "steps": outcome.get("steps"),
            }
        for task_id, verdict in per_task.items():
            by_task_type.setdefault(task_type_of(task_id), []).append(verdict["score"])
        native_splits = sorted({native_split_of(t) for t in per_task})
        return self._aggregate(
            per_task,
            secondary={
                "success_rate_by_task_type": {
                    tt: sum(v) / len(v) for tt, v in sorted(by_task_type.items())
                },
                "native_splits_evaluated": native_splits,
            },
        )

    def resource_status(self) -> ResourceStatus:
        blocking, degraded = [], []
        for pkg in ("alfworld", "gymnasium", "textworld"):
            try:
                __import__(pkg)
            except ImportError:
                blocking.append(f"python package {pkg!r} not installed")
        root = alfworld_data_root()
        env_root = os.environ.get("ALFWORLD_DATA")
        if env_root and env_root != root:
            degraded.append(
                f"$ALFWORLD_DATA={env_root} holds no json_2.1.1 corpus (the alfworld "
                f"package sets this variable to its own cache on import); using {root}"
            )
        population = set(self.population())
        if not population:
            blocking.append(f"no compiled game.tw-pddl files under {root}/json_2.1.1")
        if not ALFWORLD_LOGIC_ROOT.is_dir():
            blocking.append(f"missing PDDL logic dir: {ALFWORLD_LOGIC_ROOT}")
        try:
            manifest_ids = set(self.manifest().all_task_ids())
            missing = manifest_ids - population
            if missing:
                blocking.append(
                    f"{len(missing)}/{len(manifest_ids)} default-manifest trials have no "
                    "compiled game.tw-pddl"
                )
        except FileNotFoundError:
            pass
        return ResourceStatus(
            benchmark=self.name,
            ready=not blocking,
            blocking=blocking,
            degraded=degraded,
            details={"n_population": len(population), "alfworld_data_root": root},
        )
