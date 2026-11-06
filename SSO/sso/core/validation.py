"""candidate validation, eq. 16-18 / Algorithm 1 lines 36-42."""

from dataclasses import dataclass
from typing import Sequence, Tuple

from sso.core._execution import compare, execute_as, task_ids
from sso.core.types import Execution, ExecutionRef, Skill, Task
from sso.executors import Executor
from sso.llm.judge import PairwiseJudge


@dataclass(frozen=True)
class ValidationComparison:
    """Hold candidate and anchor executions for validation, separate from training evidence."""

    task_id: str
    current_execution: Execution
    candidate_execution: Execution
    y: int


@dataclass(frozen=True)
class ValidationResult:
    """inspection record; counts/rule are PAPER_SPECIFIED."""

    selected_skill: Skill
    comparisons: Tuple[ValidationComparison, ...]
    wins: int
    losses: int
    ties: int
    accepted: bool


def validate_candidate(
    current_skill: Skill, candidate_skill: Skill, validation_batch: Sequence[Task],
    *, executor: Executor, judge: PairwiseJudge,
) -> ValidationResult:
    """Execute both skills and accept iff W > L, with ties in neither count.
    
    fresh current/candidate executions and the same comparative
    judge, eq. 16-18. Caller supplies V_t; no sampling or extra threshold.
    sequential input order, current then candidate per
    task, duplicate-id rejection, and a local non-anchor slot 1 adapting the
    existing judge signature. Empty V_t gives W=L=0 and rejects by eq. 18.
    Dataset split disjointness is checked by the optimizer, not this function
    (which has no training data). Exceptions propagate without retries."""
    tasks = tuple(validation_batch)
    task_ids(tasks)
    comparisons = []
    for task in tasks:
        current = execute_as(executor, current_skill, task, ExecutionRef.anchor(task.task_id))
        candidate = execute_as(executor, candidate_skill, task, ExecutionRef.probe(task.task_id, 1))
        decision = compare(judge, task, current, candidate, 1)
        comparisons.append(ValidationComparison(task.task_id, current, candidate, decision.y))
    wins = sum(item.y == 1 for item in comparisons)
    losses = sum(item.y == -1 for item in comparisons)
    ties = sum(item.y == 0 for item in comparisons)
    accepted = wins > losses
    return ValidationResult(
        candidate_skill if accepted else current_skill,
        tuple(comparisons), wins, losses, ties, accepted,
    )
