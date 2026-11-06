"""Score SSO executions with the shared LiveMath evaluator."""
from __future__ import annotations

from integration.common.livemath_data import LiveMathItem
from integration.common.livemath_scoring import score as _score

from sso.core.types import Execution  # noqa: E402


class LiveMathScorer:
    def __init__(self, items_by_id: dict[str, LiveMathItem]) -> None:
        self._items_by_id = items_by_id

    def score_execution(self, execution: Execution) -> dict:
        item = self._items_by_id[execution.ref.task_id]
        return _score(execution.content, list(item.choices), item.correct_label)

    def score_all(self, executions: dict[str, Execution]) -> dict:
        """Aggregate accuracy over a `{task_id: Execution}` mapping."""
        results = {task_id: self.score_execution(execution) for task_id, execution in executions.items()}
        n = len(results)
        accuracy = sum(r["score"] for r in results.values()) / n if n else 0.0
        return {"per_task": results, "n": n, "accuracy": accuracy}
