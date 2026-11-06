"""Bind executions to task references and validate comparison bookkeeping."""

from typing import Sequence

from sso.core.types import Execution, ExecutionRef, PairwiseDecision, Skill, Task
from sso.executors import Executor
from sso.llm.judge import PairwiseJudge


def task_ids(tasks: Sequence[Task]) -> set:
    ids = [task.task_id for task in tasks]
    if any(not isinstance(key, str) or not key for key in ids):
        raise ValueError("task ids must be nonempty strings")
    if len(set(ids)) != len(ids):
        raise ValueError("a batch must contain unique task ids")
    return set(ids)


def execute_as(executor: Executor, skill: Skill, task: Task, ref: ExecutionRef) -> Execution:
    """Bind caller-owned role metadata without altering observable content."""
    execution = executor.execute(skill, task)
    if (not isinstance(execution, Execution) or execution.ref.task_id != task.task_id
            or not isinstance(execution.content, str) or ref.task_id != task.task_id):
        raise ValueError("executor returned an invalid execution or wrong task id")
    return Execution(ref=ref, content=execution.content)


def compare(
    judge: PairwiseJudge, task: Task, anchor: Execution, other: Execution, index: int,
) -> PairwiseDecision:
    """Use the same judge unchanged; reject mismatched decision bookkeeping."""
    decision = judge.judge(task, anchor, other, index)
    if (not isinstance(decision, PairwiseDecision) or decision.task_id != task.task_id
            or type(decision.probe_index) is not int or decision.probe_index != index
            or type(decision.y) is not int or decision.y not in (-1, 0, 1)):
        raise ValueError("judge returned an invalid or mismatched decision")
    return decision
