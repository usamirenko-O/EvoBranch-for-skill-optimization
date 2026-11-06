"""LiveCodeBench task dataloader (release v5 + v6 population)."""
from __future__ import annotations

import glob
import json
import os
from typing import Any

from skillopt.datasets.base import SplitDataLoader

DIFFICULTIES = ("easy", "medium", "hard")


def _iter_files(data_path: str) -> list[str]:
    if os.path.isfile(data_path):
        return [data_path]
    if os.path.isdir(data_path):
        return sorted(glob.glob(os.path.join(data_path, "test*.jsonl")))
    return []


def _json_maybe(value: Any, default: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value if value is not None else default


def normalize_record(raw: dict, *, source_file: str, source_offset: int) -> dict:
    metadata = _json_maybe(raw.get("metadata"), {}) or {}
    public_tests = _json_maybe(raw.get("public_test_cases"), []) or []
    return {
        "id": str(raw["question_id"]),
        "question_title": str(raw.get("question_title") or ""),
        "question_content": str(raw.get("question_content") or ""),
        "starter_code": str(raw.get("starter_code") or ""),
        "platform": str(raw.get("platform") or ""),
        "contest_id": str(raw.get("contest_id") or ""),
        "contest_date": str(raw.get("contest_date") or ""),
        "difficulty": str(raw.get("difficulty") or "").lower(),
        "public_test_cases": public_tests,
        "func_name": str(metadata.get("func_name") or ""),
        "source_file": os.path.abspath(source_file),
        "source_offset": int(source_offset),
    }


def load_items(data_path: str) -> list[dict]:
    files = _iter_files(data_path)
    if not files:
        raise ValueError(
            "LiveCodeBench requires data_path to be a test*.jsonl file or a directory "
            f"containing them; got {data_path!r}"
        )
    items: list[dict] = []
    seen: set[str] = set()
    for path in files:
        with open(path, "rb") as f:
            offset = 0
            for line in f:
                if line.strip():
                    raw = json.loads(line)
                    item = normalize_record(raw, source_file=path, source_offset=offset)
                    if item["id"] in seen:
                        raise ValueError(f"duplicate LiveCodeBench question_id {item['id']!r} in {path}")
                    seen.add(item["id"])
                    items.append(item)
                offset += len(line)
    if not items:
        raise ValueError(f"no LiveCodeBench items loaded from {data_path}")
    return items


def read_raw_line(source_file: str, source_offset: int) -> dict:
    """Re-read one raw problem (including its private tests) by offset."""
    with open(source_file, "rb") as f:
        f.seek(int(source_offset))
        return json.loads(f.readline())


class LiveCodeBenchDataLoader(SplitDataLoader):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._task_types: list[str] = list(DIFFICULTIES)

    def load_raw_items(self, data_path: str) -> list[dict]:
        return load_items(data_path)

    def get_task_types(self) -> list[str]:
        return list(self._task_types)
