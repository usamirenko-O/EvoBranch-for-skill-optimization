"""PubMedQA (PQA-L, the 1000 expert-labelled questions) dataloader."""
from __future__ import annotations

import json
import os

from skillopt.datasets.base import SplitDataLoader

LABELS = ("yes", "no", "maybe")


def _find_file(data_path: str) -> str:
    if os.path.isfile(data_path):
        return data_path
    candidate = os.path.join(data_path, "ori_pqal.json")
    if os.path.isfile(candidate):
        return candidate
    raise ValueError(f"PubMedQA requires ori_pqal.json; not found at {data_path!r}")


def normalize_record(pmid: str, raw: dict) -> dict:
    decision = str(raw.get("final_decision") or "").strip().lower()
    return {
        "id": str(pmid),
        "question": str(raw.get("QUESTION") or "").strip(),
        "contexts": [str(c) for c in (raw.get("CONTEXTS") or [])],
        "labels": [str(l) for l in (raw.get("LABELS") or [])],
        "final_decision": decision,
        "long_answer": str(raw.get("LONG_ANSWER") or ""),
        "year": str(raw.get("YEAR") or ""),
        "meshes": [str(m) for m in (raw.get("MESHES") or [])],
    }


def load_items(data_path: str) -> list[dict]:
    path = _find_file(data_path)
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"expected a {{pmid: record}} object in {path}")
    items = []
    for pmid in sorted(raw, key=lambda k: str(k)):
        item = normalize_record(pmid, raw[pmid])
        if item["question"] and item["final_decision"] in LABELS:
            items.append(item)
    if not items:
        raise ValueError(f"no PubMedQA items loaded from {path}")
    return items


class PubMedQADataLoader(SplitDataLoader):
    def load_raw_items(self, data_path: str) -> list[dict]:
        return load_items(data_path)

    def get_task_types(self) -> list[str]:
        return list(LABELS)
