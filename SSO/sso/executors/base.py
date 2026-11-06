"""f_exec(S, x) -> r, eq. 1.

No benchmark harness, model call, or execution policy is implemented here."""

from abc import ABC, abstractmethod

from sso.core.types import Execution, Skill, Task


class Executor(ABC):
    """Execute a supplied skill on a task using a fixed caller-owned harness.
    
    an ABC returning the existing text Execution type.
    The executor does not know the optimization role or probe index. Return
    an Execution with the correct task_id (ExecutionRef.anchor(task.task_id)
    is a suitable local placeholder). Orchestration binds the contextual
    reference after execution, preserving content exactly. An executor must
    not infer roles from skill text or maintain a probe-index counter."""

    @abstractmethod
    def execute(self, skill: Skill, task: Task) -> Execution:
        """PAPER_SPECIFIED inputs/output; failures propagate to the caller."""
        raise NotImplementedError
