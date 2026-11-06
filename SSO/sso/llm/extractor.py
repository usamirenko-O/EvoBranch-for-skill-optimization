"""Extract behavior records from task execution comparisons."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

from sso.core.types import Activation, BehaviorRecord, Execution, Task
from sso.llm.client import LLMClient
from sso.llm._structured import (
    LLMResponseError, array, ask_json_validated, check_execution, object_keys, text,
)


class BehaviorExtractor(ABC):
    """f_beh: identifies behavioral differences for one task from its anchor execution and all K probe executions."""

    @abstractmethod
    def extract(
        self,
        task: Task,
        anchor_execution: Execution,
        probe_executions: Sequence[Execution],
    ) -> List[BehaviorRecord]:
        """Produce B_i = {(beta_ij, alpha_ij)} for one task.

        Args:
            task: x_i.
            anchor_execution: r_i^o.
            probe_executions: {r_i^k}_{k=1}^K, all K probe executions for
                this task (PAPER_SPECIFIED -- shown together, not one at a
                time).

        Returns:
            The task-local behavior records B_i (PAPER_SPECIFIED fields;
            see `BehaviorRecord`). J_i = len(result) may be 0.
        """
        raise NotImplementedError


# PAPER_SPECIFIED: eq. 6-7, Algorithm 1 line 13: jointly extract observable
# differences and assign present/absent/unclear to EVERY shown execution.
# REPRODUCTION_CHOICE: exact prompt, execution aliases, and JSON schema.
_EXTRACTOR_PROMPT = """Identify observable behavioral differences among all
the supplied executions of the task, considering them together. Describe
each observable behavior separately. For each behavior, assign its state
in EVERY supplied execution: present if observed, absent if clearly absent,
unclear if the content does not establish whether it occurs. Do not turn
uncertainty into absence. Do not infer preferences or rank executions.
Return {"behaviors": [{"description": "...", "activations":
{"e0": "present", "e1": "absent"}}]}.
The example activation keys are illustrative: each activations object must
contain exactly ALL execution ids from the input, with values from
"present", "absent", "unclear". Return an empty behaviors array if no
observable differences can be identified.
"""


class LLMBehaviorExtractor(BehaviorExtractor):
    """Concrete f_beh, with no judge-outcome input or stored judge state.
    
    e0 is the anchor; e1.. follow the supplied probe
    sequence, independent of actual probe indices. Returned record ids are
    task_id:j (j starts at 1 in response order), local to this extraction.
    Callers must not pool multiple extractions of the same task as one batch."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def extract(
        self, task: Task, anchor_execution: Execution,
        probe_executions: Sequence[Execution],
    ) -> List[BehaviorRecord]:
        check_execution(task, anchor_execution, anchor=True)
        if not probe_executions:
            raise ValueError("at least one probe execution is required")
        seen = set()
        for execution in probe_executions:
            check_execution(task, execution, anchor=False)
            if execution.ref in seen:
                raise ValueError("duplicate probe execution reference")
            seen.add(execution.ref)
        executions = [anchor_execution, *probe_executions]
        by_alias = {f"e{i}": execution for i, execution in enumerate(executions)}

        def build(response):
            obj = object_keys(response, ("behaviors",))
            records = []
            for j, item in enumerate(array(obj["behaviors"]), 1):
                behavior = object_keys(item, ("description", "activations"))
                description = text(behavior["description"])
                states = object_keys(behavior["activations"], by_alias)
                activations = {}
                for alias, execution in by_alias.items():
                    state = states[alias]
                    if not isinstance(state, str) or state not in ("present", "absent", "unclear"):
                        raise LLMResponseError("invalid behavior activation")
                    activations[execution.ref] = Activation(state)
                records.append(BehaviorRecord(
                    task_id=task.task_id, record_id=f"{task.task_id}:{j}",
                    description=description, activations=activations,
                ))
            return records

        return ask_json_validated(self.client, _EXTRACTOR_PROMPT, {
            "task": task.content,
            "executions": [{"id": alias, "content": execution.content}
                           for alias, execution in by_alias.items()],
        }, build)
