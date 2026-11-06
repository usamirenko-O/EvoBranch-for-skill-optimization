"""Read and write per-call usage records."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class CallRecord:
    run_id: str
    method: str  # "skillopt" | "sso" | "skillgen"
    benchmark: str  # "livemathematicianbench"
    model: str
    harness: str  # "direct"
    phase: str  # "optimize" | "deploy" | "baseline_eval"
    task_id: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int | None
    llm_calls: int
    wall_time: float
    task_score: float | None
    optimization_iteration: int | None = None


def append_record(path: Path | str, record: CallRecord) -> None:
    """Append one CallRecord as a JSONL line. Creates parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def read_records(path: Path | str) -> list[dict]:
    """Read call records from a JSONL log."""
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
