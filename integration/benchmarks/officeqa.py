"""Load OfficeQA tasks and score answers using its benchmark evaluator."""
from __future__ import annotations

import csv
import importlib.util
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
    file_digest,
    load_manifest,
    source_inputs,
)
from integration.benchmarks.paths import (
    EVOSKILL_ROOT,
    MANIFEST_DIR,
    OFFICEQA_DOCS_ROOT,
    OFFICEQA_FULL_CSV,
    OFFICEQA_ID_SPLIT,
    OFFICEQA_PRO_CSV,
)

MANIFEST_ID = "officeqa_pro_hard_133_v1"
METRIC_NAME = "officeqa_score_answer_accuracy"

#: The official reference scorer, as vendored in this workspace.
OFFICIAL_REWARD_PY = EVOSKILL_ROOT / "src" / "evaluation" / "reward.py"
SCORER_REF = "officeqa_reference_reward:score_answer"

#: `score_answer`'s own default. Recorded in every result artifact rather
#: than left implicit -- the dataset card shows examples with 0.0, 0.01 and
#: 0.05, so which tolerance a number was produced under is not obvious.
DEFAULT_TOLERANCE = 0.0

_ROLE_TO_SKILLOPT_SPLIT = {"train": "train", "val": "val", "deployment": "test"}


# ---------------------------------------------------------------- scorer --


def load_official_scorer():
    """Load the OfficeQA reference `reward.py` by path; return the module.

    Raises `FileNotFoundError` if it is absent. Never falls back to another
    implementation.
    """
    if not OFFICIAL_REWARD_PY.is_file():
        raise FileNotFoundError(
            f"OfficeQA official scorer not found at {OFFICIAL_REWARD_PY}. "
            "Obtain reward.py from https://github.com/databricks/officeqa. "
            "Refusing to substitute SkillOpt's EM/F1 evaluator, which is a "
            "reimplementation, not the official scorer."
        )
    spec = importlib.util.spec_from_file_location("officeqa_reference_reward", OFFICIAL_REWARD_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# -------------------------------------------------------------- raw data --


def load_pro_rows() -> dict:
    """`{uid: row}` from the locally-present `officeqa_pro.csv`."""
    with OFFICEQA_PRO_CSV.open(encoding="utf-8", newline="") as f:
        return {row["uid"]: row for row in csv.DictReader(f)}


def _skillopt_manifest_items(split: str) -> list[dict]:
    with (OFFICEQA_ID_SPLIT / split / "items.json").open(encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------- manifest build --


def build_manifest() -> SplitManifest:
    """Derive the manifest deterministically from files already on disk."""
    upstream = json.loads((OFFICEQA_ID_SPLIT / "split_manifest.json").read_text(encoding="utf-8"))
    available = set(load_pro_rows())

    splits: dict = {}
    excluded: dict = {}
    for role in SPLIT_ROLES:
        so_split = _ROLE_TO_SKILLOPT_SPLIT[role]
        kept, dropped = [], []
        for item in _skillopt_manifest_items(so_split):
            uid = str(item["uid"])
            if uid in available:
                kept.append(
                    SplitEntry(
                        task_id=uid,
                        native_id=uid,
                        native_split="train",  # HF publishes one split, named "train"
                        extra={
                            "category": item.get("category", ""),
                            "source_files": item.get("source_files", ""),
                            "skillopt_split": so_split,
                        },
                    )
                )
            else:
                dropped.append(uid)
        splits[role] = kept
        excluded[role] = dropped

    manifest = SplitManifest(
        benchmark="officeqa",
        manifest_id=MANIFEST_ID,
        version="1",
        source={
            "upstream_dataset": upstream.get("source_repo"),
            "upstream_url": upstream.get("source_url"),
            "upstream_revision": upstream.get("source_revision"),
            "upstream_file_used_by_skillopt": upstream.get("source_file"),
            "materialized_from": str(OFFICEQA_PRO_CSV),
            "frozen_partition": str(OFFICEQA_ID_SPLIT),
            "inputs": source_inputs(
                [
                    OFFICEQA_ID_SPLIT / "split_manifest.json",
                    OFFICEQA_ID_SPLIT / "train" / "items.json",
                    OFFICEQA_ID_SPLIT / "val" / "items.json",
                    OFFICEQA_ID_SPLIT / "test" / "items.json",
                    OFFICEQA_PRO_CSV,
                ]
            ),
            "official_scorer": {
                "ref": SCORER_REF,
                "upstream": "https://github.com/databricks/officeqa (reward.py)",
                "vendored_copy": str(OFFICIAL_REWARD_PY),
                "sha256": file_digest(OFFICIAL_REWARD_PY)
                if OFFICIAL_REWARD_PY.is_file()
                else None,
            },
        },
        role_map={
            "train": "SkillOpt officeqa_id_split/train, restricted to locally-present uids",
            "val": "SkillOpt officeqa_id_split/val, restricted to locally-present uids",
            "deployment": "SkillOpt officeqa_id_split/test, restricted to locally-present uids",
            "native_splits": "none -- OfficeQA publishes a single HF split named 'train'",
        },
        sampling={
            "method": "frozen_upstream_manifest_filtered_by_local_availability",
            "seed": None,
            "filter": "uid present in officeqa_pro.csv (== SkillOpt manifest category=='hard')",
            "excluded_counts": {role: len(v) for role, v in excluded.items()},
            "excluded_task_ids": excluded,
        },
        splits=splits,
        notes=[
            "OfficeQA has no native train/val/test split; the partition is SkillOpt's "
            "frozen officeqa_id_split, reused verbatim for the rows that exist locally.",
            "113 of SkillOpt's 246 manifest rows (all category=='easy') are excluded "
            "because officeqa_full.csv is gated and not present in this workspace. "
            "Verified offline: the 133 surviving rows are exactly the 133 uids in "
            "officeqa_pro.csv, so no row was chosen by this builder.",
            "Default manifest only. The run-time split is whatever manifest the "
            "launcher was given; this one is the checked-in default/example over "
            "the full locally-available population.",
            "Scoring tolerance is score_answer's own default (0.0) and is recorded "
            "in every result as a secondary metric; OfficeQAAdapter(tolerance=...) "
            "changes it without touching the scorer.",
        ],
    ).with_hash()
    return manifest


# ------------------------------------------------------------------ adapter --


class OfficeQAAdapter(BenchmarkAdapter):
    name = "officeqa"
    metric_name = METRIC_NAME
    scorer_ref = SCORER_REF
    manifest_id = MANIFEST_ID

    def __init__(self, *, tolerance: float = DEFAULT_TOLERANCE) -> None:
        self.tolerance = float(tolerance)
        self._manifest: SplitManifest | None = None
        self._rows: dict | None = None

    # ---- splits ----

    def manifest(self) -> SplitManifest:
        if self._manifest is None:
            self._manifest = load_manifest(MANIFEST_DIR / f"{MANIFEST_ID}.json")
        return self._manifest

    def _row(self, uid: str) -> dict:
        if self._rows is None:
            self._rows = load_pro_rows()
        return self._rows[uid]

    def _task(self, task_id: str, role: str) -> BenchmarkTask:
        row = self._row(task_id)
        return BenchmarkTask(
            task_id=task_id,
            native_id=task_id,
            split_role=role,
            benchmark=self.name,
            payload={
                "question": row["question"],
                "source_files": row.get("source_files", ""),
                "source_docs": row.get("source_docs", ""),
                "difficulty": row.get("difficulty", ""),
                "docs_root": str(OFFICEQA_DOCS_ROOT),
            },
        )

    def tasks(self, role: str) -> list[BenchmarkTask]:
        """Payload keys: `question`, `source_files`, `source_docs`, `difficulty`.

        Ground truth is deliberately absent -- it lives in `gold_map()` and
        is only read by `evaluate`, matching the LiveMath adapter's stance
        that an optimizer's algorithm never needs the answer key.
        """
        return [self._task(e.task_id, role) for e in self.manifest().entries(role)]

    def tasks_for_ids(self, task_ids, role: str = "custom") -> list[BenchmarkTask]:
        return [self._task(str(t), role) for t in task_ids]

    def population(self) -> list[str]:
        return sorted(self._rows if self._rows is not None else load_pro_rows())

    def gold_map(self, role: str) -> dict:
        """`{task_id: ground_truth}` for the scorer/driver only."""
        return {e.task_id: self._row(e.task_id)["answer"] for e in self.manifest().entries(role)}

    def gold_for_ids(self, task_ids) -> dict:
        return {str(t): self._row(str(t))["answer"] for t in task_ids}

    # ---- official scoring ----

    def evaluate(
        self,
        task_outputs: Mapping[str, Any],
        *,
        role: str = "deployment",
        extract_final_answer: bool = True,
    ) -> BenchmarkResult:
        """`task_outputs` maps task_id -> the solver's answer string.

        `extract_final_answer` applies the official reward module's own
        `extract_final_answer` (which returns the text unchanged when no
        `<FINAL_ANSWER>` tag is present), so a caller that already extracted
        its own answer -- SkillOpt's OfficeQA prompt asks for `<answer>`
        tags, not `<FINAL_ANSWER>` -- is unaffected. No extraction logic of
        this module's own is applied either way.
        """
        self._check_outputs_against_population(task_outputs)
        reward = load_official_scorer()
        gold = self.gold_for_ids(task_outputs)

        per_task: dict = {}
        for task_id, raw in task_outputs.items():
            prediction = "" if raw is None else str(raw)
            if extract_final_answer and prediction.strip():
                prediction = reward.extract_final_answer(prediction)
            ground_truth = gold[task_id]
            if not prediction.strip():
                # The official fuzzy matcher raises on an empty prediction;
                # an empty answer is a miss, not a scoring error.
                per_task[task_id] = {
                    "score": 0.0,
                    "prediction": "",
                    "ground_truth": ground_truth,
                    "note": "empty prediction",
                }
                continue
            score = reward.score_answer(ground_truth, prediction, self.tolerance)
            per_task[task_id] = {
                "score": float(score),
                "prediction": prediction,
                "ground_truth": ground_truth,
            }
        return self._aggregate(per_task, secondary={"tolerance": self.tolerance})

    # ---- resources ----

    def resource_status(self) -> ResourceStatus:
        blocking, degraded = [], []
        if not OFFICEQA_PRO_CSV.is_file():
            blocking.append(f"missing questions CSV: {OFFICEQA_PRO_CSV}")
        if not OFFICIAL_REWARD_PY.is_file():
            blocking.append(f"missing official scorer reward.py: {OFFICIAL_REWARD_PY}")
        n_docs = len(list(OFFICEQA_DOCS_ROOT.glob("*.txt"))) if OFFICEQA_DOCS_ROOT.is_dir() else 0
        if n_docs == 0:
            blocking.append(f"missing Treasury bulletin corpus: {OFFICEQA_DOCS_ROOT}")
        if not OFFICEQA_FULL_CSV.is_file():
            degraded.append(
                "officeqa_full.csv is gated and absent, so the benchmark population "
                "is the 133 Pro questions. A manifest may name any of them; the 113 "
                "'easy' questions of the Full set are not obtainable here."
            )
        return ResourceStatus(
            benchmark=self.name,
            ready=not blocking,
            blocking=blocking,
            degraded=degraded,
            details={"n_corpus_txt_files": n_docs},
        )
