"""Render shared task splits in each optimizer's input format."""
from __future__ import annotations

import json
from pathlib import Path

from integration.benchmarks.manifest import SplitManifest
from integration.benchmarks.paths import (
    SPREADSHEETBENCH_DATASET_JSON,
    SPREADSHEETBENCH_ROOT,
)

#: manifest role -> SkillOpt split directory name
ROLE_TO_SKILLOPT_DIR = {"train": "train", "val": "val", "deployment": "test"}


def _officeqa_rows(task_ids: list) -> list:
    from integration.benchmarks.officeqa import load_pro_rows

    rows = load_pro_rows()
    out = []
    for task_id in task_ids:
        row = rows[task_id]
        out.append(
            {
                "id": row["uid"],
                "uid": row["uid"],
                "question": row["question"],
                "ground_truth": row["answer"],
                "answer": row["answer"],
                "category": row.get("difficulty", "officeqa"),
                "source_files": row.get("source_files", ""),
                "source_docs": row.get("source_docs", ""),
            }
        )
    return out


def _spreadsheetbench_rows(task_ids: list) -> list:
    with SPREADSHEETBENCH_DATASET_JSON.open(encoding="utf-8") as f:
        by_id = {str(r["id"]): r for r in json.load(f)}
    return [dict(by_id[str(t)]) for t in task_ids]


def _alfworld_rows(task_ids: list) -> list:
    from integration.benchmarks.alfworld import gamefile_for_task_id, task_type_of

    return [
        {
            "id": task_id,
            "gamefile": gamefile_for_task_id(task_id),
            "task_type": task_type_of(task_id),
        }
        for task_id in task_ids
    ]


def _livemath_rows(task_ids: list, *, shuffle_seed: int) -> list:
    """Pre-shuffled rows, so SkillOpt reads the same question the shared
    harness will present at deployment.

    SkillOpt's own dataloader would otherwise shuffle again with its batch
    seed, and its optimization phase would see a different presentation than
    its deployment phase. The runner pairs this with
    `env.shuffle_choices=false`.
    """
    from integration.benchmarks.livemath_choices import shuffle_item
    from integration.common.paths import LIVEMATH_DATA_ROOT
    from skillopt.envs.livemathematicianbench.dataloader import load_items

    by_id = {str(item["id"]): item for item in load_items(str(LIVEMATH_DATA_ROOT))}
    return [shuffle_item(by_id[str(t)], shuffle_seed) for t in task_ids]


def _single_turn_rows(benchmark: str):
    """The cheap single-turn benchmarks: rows are the SkillOpt env's own
    normalized items (the adapter already loads them through that env), so
    SkillOpt's split_dir carries exactly what its dataloader expects."""

    def build(task_ids: list) -> list:
        from integration.benchmarks.registry import get_adapter

        adapter = get_adapter(benchmark)
        return [dict(adapter.item(str(t))) for t in task_ids]

    return build


_ROW_BUILDERS = {
    "officeqa": _officeqa_rows,
    "spreadsheetbench": _spreadsheetbench_rows,
    "alfworld": _alfworld_rows,
    "livemathematicianbench": _livemath_rows,
    "livecodebench": _single_turn_rows("livecodebench"),
    "pubmedqa": _single_turn_rows("pubmedqa"),
    "chemllmbench": _single_turn_rows("chemllmbench"),
}


def materialize_skillopt_split(manifest: SplitManifest, out_dir: Path) -> Path:
    """Write `out_dir/{train,val,test}/items.json` from the manifest."""
    builder = _ROW_BUILDERS[manifest.benchmark]
    kwargs = {}
    if manifest.benchmark == "livemathematicianbench":
        from integration.benchmarks.livemath_choices import shuffle_seed_for

        kwargs["shuffle_seed"] = shuffle_seed_for(manifest)
    out_dir = Path(out_dir)
    for role, dirname in ROLE_TO_SKILLOPT_DIR.items():
        split_dir = out_dir / dirname
        split_dir.mkdir(parents=True, exist_ok=True)
        rows = builder(manifest.task_ids(role), **kwargs)
        with (split_dir / "items.json").open("w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
    return out_dir


def skillopt_env_overrides(benchmark: str) -> dict:
    """Benchmark-specific `--cfg-options` SkillOpt needs to find its data."""
    if benchmark == "livemathematicianbench":
        # The split_dir rows this module writes are already shuffled with the
        # manifest's protocol seed. Letting SkillOpt's dataloader shuffle them
        # again (its default) would give its optimization phase a different
        # question than the shared deployment harness presents.
        return {"env.shuffle_choices": "false"}
    if benchmark == "spreadsheetbench":
        return {"env.data_root": str(SPREADSHEETBENCH_ROOT)}
    if benchmark == "officeqa":
        from integration.benchmarks.paths import OFFICEQA_DOCS_ROOT

        return {"env.data_dirs": str(OFFICEQA_DOCS_ROOT)}
    return {}


def skillgen_instances(manifest: SplitManifest, roles=("train", "val")) -> list:
    """Build SkillGen task rows routed through the shared execution harness and scorer."""
    from integration.benchmarks.registry import get_adapter

    adapter = get_adapter(manifest.benchmark)
    instances = []
    for role in roles:
        for task in adapter.tasks_for_ids(manifest.task_ids(role), role=role):
            instances.append(
                {
                    "instance_id": task.task_id,
                    "input": _skillgen_input_text(task),
                    "ground_truth": "",  # held by the official scorer, never by the agent
                    "metadata": {
                        "benchmark": "shared_direct",
                        "shared_benchmark": manifest.benchmark,
                        "task_id": task.task_id,
                        "split_role": role,
                    },
                }
            )
    return instances


def _skillgen_input_text(task) -> str:
    """A human-readable rendering of the task, for SkillGen's own induction
    prompts. The authoritative task payload is re-fetched from the benchmark
    adapter at execution time by `shared_direct_adapter`, so this string is
    never the thing that actually gets executed."""
    payload = task.payload
    if "prompt" in payload:
        return payload["prompt"]
    if "question" in payload:
        return payload["question"]
    if "instruction" in payload:
        return payload["instruction"]
    if "gamefile" in payload:
        return f"ALFWorld interactive task ({payload.get('task_type', '')}): {task.task_id}"
    return task.task_id
