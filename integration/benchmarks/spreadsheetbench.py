"""Load SpreadsheetBench tasks and evaluate generated workbooks."""
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
from integration.benchmarks.paths import (
    MANIFEST_DIR,
    SPREADSHEETBENCH_DATASET_JSON,
    SPREADSHEETBENCH_ID_SPLIT,
    SPREADSHEETBENCH_ROOT,
    SPREADSHEETBENCH_SPREADSHEET_ROOT,
)

MANIFEST_ID = "spreadsheetbench_verified_400_v1"
METRIC_NAME = "spreadsheetbench_hard_accuracy"
SCORER_REF = "skillopt.envs.spreadsheetbench.evaluator:evaluate"
VARIANT = "verified_400"

_ROLE_TO_SKILLOPT_SPLIT = {"train": "train", "val": "val", "deployment": "test"}


# -------------------------------------------------------------- raw data --


def load_dataset_rows() -> dict:
    """`{id: row}` from the local `verified_400` dataset.json."""
    with SPREADSHEETBENCH_DATASET_JSON.open(encoding="utf-8") as f:
        return {str(row["id"]): row for row in json.load(f)}


def answer_position_eval(row: dict) -> str:
    """The cell range the official evaluator is given, sheet-qualified.

    Verbatim from SkillOpt's own rollout (`rollout.py`): when the dataset row
    carries an `answer_sheet` and the `answer_position` does not already name a
    sheet, the two are joined as `<sheet>!<range>`. Getting this wrong silently
    compares the wrong sheet, so it is computed once here and carried on the
    task payload for both the harness prompt and the scorer.
    """
    answer_position = row.get("answer_position", "")
    answer_sheet = row.get("answer_sheet", "")
    if answer_position and answer_sheet and "!" not in answer_position:
        return f"{answer_sheet}!{answer_position}"
    return answer_position


def _skillopt_manifest_items(split: str) -> list[dict]:
    with (SPREADSHEETBENCH_ID_SPLIT / split / "items.json").open(encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------- manifest build --


def build_manifest() -> SplitManifest:
    upstream = json.loads(
        (SPREADSHEETBENCH_ID_SPLIT / "split_manifest.json").read_text(encoding="utf-8")
    )
    rows = load_dataset_rows()

    splits: dict = {}
    missing: dict = {}
    for role in SPLIT_ROLES:
        so_split = _ROLE_TO_SKILLOPT_SPLIT[role]
        kept, absent = [], []
        for item in _skillopt_manifest_items(so_split):
            task_id = str(item["id"])
            if task_id not in rows:
                absent.append(task_id)
                continue
            kept.append(
                SplitEntry(
                    task_id=task_id,
                    native_id=task_id,
                    native_split="",  # flat dataset, no native splits
                    extra={
                        "spreadsheet_path": item.get("spreadsheet_path", ""),
                        "instruction_type": item.get("instruction_type", ""),
                        "skillopt_split": so_split,
                    },
                )
            )
        splits[role] = kept
        missing[role] = absent

    return SplitManifest(
        benchmark="spreadsheetbench",
        manifest_id=MANIFEST_ID,
        version="1",
        source={
            "upstream_dataset": upstream.get("source_repo"),
            "upstream_url": upstream.get("source_url"),
            "upstream_revision": upstream.get("source_revision"),
            "upstream_file": upstream.get("source_file"),
            "variant": VARIANT,
            "materialized_from": str(SPREADSHEETBENCH_DATASET_JSON),
            "frozen_partition": str(SPREADSHEETBENCH_ID_SPLIT),
            "inputs": source_inputs(
                [
                    SPREADSHEETBENCH_ID_SPLIT / "split_manifest.json",
                    SPREADSHEETBENCH_ID_SPLIT / "train" / "items.json",
                    SPREADSHEETBENCH_ID_SPLIT / "val" / "items.json",
                    SPREADSHEETBENCH_ID_SPLIT / "test" / "items.json",
                    SPREADSHEETBENCH_DATASET_JSON,
                ]
            ),
            "official_scorer": {
                "ref": SCORER_REF,
                "upstream": "https://github.com/RUCKBReasoning/SpreadsheetBench "
                "(evaluation/evaluation.py)",
                "note": "SkillOpt's module is a documented faithful port; it is the "
                "only cell-value evaluator in this workspace.",
            },
        },
        role_map={
            "train": "SkillOpt spreadsheetbench_id_split/train (80)",
            "val": "SkillOpt spreadsheetbench_id_split/val (40)",
            "deployment": "SkillOpt spreadsheetbench_id_split/test (280)",
            "native_splits": "none -- verified_400 is a flat 400-task dataset.json",
        },
        sampling={
            "method": "frozen_upstream_manifest_verbatim",
            "seed": None,
            "filter": None,
            "missing_counts": {role: len(v) for role, v in missing.items()},
        },
        splits=splits,
        notes=[
            "All 400 frozen manifest ids resolve against the local verified_400 "
            "dataset.json; the partition is used unchanged.",
            "Pinned to variant 'verified_400'. SkillGen's own SpreadsheetBench "
            "adapter defaults to 'full_912' -- a driver must force this variant.",
            "5 of the 400 task directories (13284, 32023, 32789, 56274, 58109) use "
            "the bare 'initial.xlsx'/'golden.xlsx' naming instead of "
            "'1_<id>_init.xlsx'/'1_<id>_golden.xlsx'. SkillOpt's own test-case "
            "discovery already handles both, and this adapter delegates to it "
            "rather than encoding a second layout rule.",
            "Default manifest only -- the run-time split is whatever manifest the "
            "launcher was given. This default covers the whole verified_400 population.",
            "Each verified_400 task carries exactly one test case, so the official "
            "'hard' metric (all cases must pass) reduces to per-task pass/fail on "
            "this variant; the upstream 912-task variant has three cases per task.",
        ],
    ).with_hash()


# ------------------------------------------------------------------ adapter --


def _find_test_cases(task_dir: str):
    """Delegate test-case discovery to SkillOpt's own implementation.

    Imported lazily: `skillopt.envs.spreadsheetbench.rollout` imports
    `openpyxl` at module scope, so importing it at this module's top level
    would make the whole benchmarks package unimportable in an environment
    without openpyxl -- including every structural test in this phase.
    """
    from skillopt.envs.spreadsheetbench.rollout import _find_test_cases as _impl

    return _impl(task_dir)


class SpreadsheetBenchAdapter(BenchmarkAdapter):
    name = "spreadsheetbench"
    metric_name = METRIC_NAME
    scorer_ref = SCORER_REF
    manifest_id = MANIFEST_ID

    def __init__(self) -> None:
        self._manifest: SplitManifest | None = None
        self._rows: dict | None = None

    def manifest(self) -> SplitManifest:
        if self._manifest is None:
            self._manifest = load_manifest(MANIFEST_DIR / f"{MANIFEST_ID}.json")
        return self._manifest

    def _row(self, task_id: str) -> dict:
        if self._rows is None:
            self._rows = load_dataset_rows()
        return self._rows[task_id]

    def tasks(self, role: str) -> list[BenchmarkTask]:
        """Payload keys: `instruction`, `instruction_type`, `answer_position`,
        `answer_sheet`, `data_position`, `task_dir` (absolute path to the
        task's spreadsheet folder). Gold workbooks live inside `task_dir` and
        are located by the official discovery rule at scoring time, not
        listed here."""
        return [self._task(e.task_id, role) for e in self.manifest().entries(role)]

    def _task(self, task_id: str, role: str) -> BenchmarkTask:
        row = self._row(task_id)
        return BenchmarkTask(
            task_id=task_id,
            native_id=task_id,
            split_role=role,
            benchmark=self.name,
            payload={
                "instruction": row["instruction"],
                "instruction_type": row.get("instruction_type", ""),
                "answer_position": row.get("answer_position", ""),
                "answer_position_eval": answer_position_eval(row),
                "answer_sheet": row.get("answer_sheet", ""),
                "data_position": row.get("data_position", ""),
                "task_dir": str(SPREADSHEETBENCH_ROOT / row["spreadsheet_path"]),
            },
        )

    def tasks_for_ids(self, task_ids, role: str = "custom") -> list[BenchmarkTask]:
        return [self._task(str(t), role) for t in task_ids]

    def population(self) -> list[str]:
        return sorted(self._rows if self._rows is not None else load_dataset_rows())

    def evaluate(self, task_outputs: Mapping[str, Any], *, role: str = "deployment") -> BenchmarkResult:
        """`task_outputs` maps task_id -> `{case_no: produced_xlsx_path}`.

        Per test case the official evaluator decides pass/fail. A task scores
        1.0 only if every one of its cases passes (the official "hard"
        metric); the per-case ratio is reported as the secondary "soft"
        number, exactly as SkillOpt's own rollout aggregates it.
        """
        self._check_outputs_against_population(task_outputs)
        from skillopt.envs.spreadsheetbench.evaluator import evaluate as official_evaluate

        per_task: dict = {}
        soft_scores: list = []
        for task_id, case_outputs in task_outputs.items():
            row = self._row(task_id)
            task_dir = str(SPREADSHEETBENCH_ROOT / row["spreadsheet_path"])
            cases = _find_test_cases(task_dir)
            case_results = []
            n_pass = 0
            for case_no, _init_path, golden_path in cases:
                pred_path = (case_outputs or {}).get(case_no) or (case_outputs or {}).get(str(case_no))
                if not pred_path:
                    case_results.append({"case": case_no, "ok": False, "reason": "no output produced"})
                    continue
                try:
                    verdict = official_evaluate(
                        str(pred_path),
                        golden_path,
                        row.get("instruction_type", ""),
                        answer_position_eval(row),
                    )
                except Exception as exc:  # noqa: BLE001
                    # SkillOpt's own rollout does exactly this (rollout.py's
                    # `eval-exception` branch): three of the 400 verified_400
                    # rows carry an answer_position the official range parser
                    # cannot read (e.g. "Sheet3'!A:G"). An unparseable range is
                    # a failed case, not a crashed run, and it must not be
                    # silently dropped from the denominator.
                    verdict = {"ok": False, "reason": f"eval-exception: {type(exc).__name__}: {exc}"}
                case_results.append({"case": case_no, "ok": bool(verdict["ok"]), "reason": verdict.get("reason", "")})
                if verdict["ok"]:
                    n_pass += 1
            n_cases = len(cases)
            hard = 1.0 if (n_cases > 0 and n_pass == n_cases) else 0.0
            soft = (n_pass / n_cases) if n_cases else 0.0
            soft_scores.append(soft)
            per_task[task_id] = {
                "score": hard,
                "soft": soft,
                "n_cases": n_cases,
                "n_pass": n_pass,
                "cases": case_results,
            }
        secondary = {
            "spreadsheetbench_soft_accuracy": (sum(soft_scores) / len(soft_scores))
            if soft_scores
            else None
        }
        return self._aggregate(per_task, secondary=secondary)

    def resource_status(self) -> ResourceStatus:
        blocking, degraded = [], []
        if not SPREADSHEETBENCH_DATASET_JSON.is_file():
            blocking.append(f"missing dataset.json: {SPREADSHEETBENCH_DATASET_JSON}")
        try:
            import openpyxl  # noqa: F401
        except ImportError:
            blocking.append(
                "python package 'openpyxl' not installed -- the official "
                "SpreadsheetBench evaluator cannot be imported without it"
            )
        n_dirs = (
            sum(1 for p in SPREADSHEETBENCH_SPREADSHEET_ROOT.iterdir() if p.is_dir())
            if SPREADSHEETBENCH_SPREADSHEET_ROOT.is_dir()
            else 0
        )
        if n_dirs == 0:
            blocking.append(f"missing spreadsheet corpus: {SPREADSHEETBENCH_SPREADSHEET_ROOT}")
        return ResourceStatus(
            benchmark=self.name,
            ready=not blocking,
            blocking=blocking,
            degraded=degraded,
            details={"n_task_dirs": n_dirs, "variant": VARIANT},
        )
