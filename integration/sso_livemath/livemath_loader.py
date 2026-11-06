"""Convert LiveMath records into SSO tasks."""
from __future__ import annotations

from pathlib import Path

from integration.common.livemath_data import LiveMathItem, load_livemath_items, render_prompt
from integration.common.paths import LIVEMATH_DATA_ROOT, SSO_ROOT, ensure_on_sys_path

ensure_on_sys_path(SSO_ROOT)

from sso.core.types import Task  # noqa: E402


def canonical_split_task_ids() -> dict[str, list[str]]:
    """Return shared LiveMath train, validation, and test task IDs."""
    from integration.skillopt_livemath.join_livemath_split import join_split

    joined = join_split()
    return {split: [str(item["id"]) for item in items] for split, items in joined.items()}


class LiveMathTaskLoader:
    """Loads LiveMath items as SSO Tasks, keeping ground truth out of them."""

    def __init__(self, data_root: Path | str = LIVEMATH_DATA_ROOT) -> None:
        self._items: dict[str, LiveMathItem] = {
            item.task_id: item for item in load_livemath_items(data_root)
        }

    def items(self) -> list[LiveMathItem]:
        return list(self._items.values())

    def item(self, task_id: str) -> LiveMathItem:
        return self._items[task_id]

    def tasks(self, task_ids: list[str] | None = None) -> list[Task]:
        """Build Task objects for the given ids (default: every loaded item)."""
        ids = task_ids if task_ids is not None else list(self._items.keys())
        return [Task(task_id=task_id, content=render_prompt(self._items[task_id])) for task_id in ids]

    def correct_label_map(self) -> dict[str, str]:
        """The side-channel ground-truth map, owned by the loader/driver, never
        exposed on the Task objects themselves."""
        return {task_id: item.correct_label for task_id, item in self._items.items()}
