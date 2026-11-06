"""Generate exploratory skill probes from tasks and execution history."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence, Tuple

from sso.core.types import Execution, Probe, Skill, Task
from sso.llm.client import LLMClient
from sso.llm._structured import (
    LLMResponseError, array, ask_json_validated, check_execution, complete_skill,
    object_keys, positive_int,
)


class ProbeGenerator(ABC):
    """f_gen: generates K exploratory probe skills from the current skill and a subset of its task-execution history."""

    @abstractmethod
    def generate(
        self,
        current_skill: Skill,
        history: Sequence[Tuple[Task, Execution]],
        num_probes: int,
    ) -> List[Probe]:
        """Produce K complete probe skills P_1..P_K."""
        raise NotImplementedError


# PAPER_SPECIFIED: eq. 3-4, Algorithm 1 line 6: jointly generate K complete,
# distinct exploratory skills from S_t and current-batch task/anchor pairs.
# REPRODUCTION_CHOICE: exact prompt, full-document tag, and ordered JSON list.
_PROBE_PROMPT = """Generate exactly num_probes exploratory skill documents
jointly, using the current skill and the supplied task/execution history.
Each probe must be distinct from the current skill and all other probes.
You may revise multiple parts. Each content must be the ENTIRE standalone
reusable skill, including any retained instructions: never a patch, diff,
edit request, or reference to omitted sections of the current skill.
An empty current skill is the no-skill starting condition.
These probes expose alternative behaviors; they are not accepted updates
and are not the candidate produced by rendering aggregated behavior evidence.
Return {"probes": [{"document_type": "complete_skill", "content": "..."}]}.
The probes array must have exactly num_probes elements, ordered by index 1..K.
"""


class LLMProbeGenerator(ProbeGenerator):
    """Concrete f_gen; prompt and schema are REPRODUCTION_CHOICE.
    
    PAPER_SPECIFIED count and distinctness are checked structurally/textually.
    positive K, unique history task ids, 1-based response
    order indexing, and outer-whitespace-insensitive duplicate checks.
    Completeness and semantic distinctness still rely on the LLM instruction."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def generate(
        self, current_skill: Skill, history: Sequence[Tuple[Task, Execution]],
        num_probes: int,
    ) -> List[Probe]:
        positive_int(num_probes, "num_probes")
        history_data = []
        task_ids = set()
        for task, execution in history:
            check_execution(task, execution, anchor=True)
            if task.task_id in task_ids:
                raise ValueError("history must contain unique tasks")
            task_ids.add(task.task_id)
            history_data.append({"task": task.content, "execution": execution.content})

        def build(response):
            obj = object_keys(response, ("probes",))
            documents = array(obj["probes"])
            if len(documents) != num_probes:
                raise LLMResponseError("expected exactly num_probes skill documents")
            contents = [complete_skill(document) for document in documents]
            seen = {current_skill.content.strip()}
            for content in contents:
                if content.strip() in seen:
                    raise LLMResponseError("probes must differ from current skill and peers")
                seen.add(content.strip())
            return [Probe(probe_index=k, skill=Skill(content=content))
                    for k, content in enumerate(contents, 1)]

        return ask_json_validated(self.client, _PROBE_PROMPT, {
            "current_skill": current_skill.content,
            "history": history_data, "num_probes": num_probes,
        }, build)
