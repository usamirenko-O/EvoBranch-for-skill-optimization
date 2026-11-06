"""Compare probe and anchor executions with randomized presentation order."""

from __future__ import annotations

from abc import ABC, abstractmethod
import random
from typing import Optional

from sso.core.types import Execution, PairwiseDecision, Task
from sso.llm.client import LLMClient
from sso.llm._structured import (
    LLMResponseError, ask_json_validated, check_execution, object_keys, positive_int,
)


class PairwiseJudge(ABC):
    """Compare probe and anchor executions for one task.
    
    Return +1 for a preferred probe, -1 for a preferred anchor, or 0 for a tie. Implementations randomize presentation order."""

    @abstractmethod
    def judge(
        self,
        task: Task,
        anchor_execution: Execution,
        probe_execution: Execution,
        probe_index: int,
    ) -> PairwiseDecision:
        """Compare `probe_execution` against `anchor_execution` for `task`."""
        raise NotImplementedError


# PAPER_SPECIFIED: compare observable executions for the task, eq. 5 and
# Algorithm 1 line 9; presentation randomization and identity remapping.
# REPRODUCTION_CHOICE: exact wording, anonymous A/B labels, and JSON schema.
_JUDGE_PROMPT = """You are comparing two candidate solutions to the supplied task,
labelled FIRST and SECOND. Judge which solution better fulfills the task using
their observable content. No reference answer, reward, or external evaluator is
supplied.

IMPORTANT: You are NOT answering the task yourself. Your decision is which of the
two solutions is better, not what the task's own answer is. Even if the task is a
multiple-choice question whose options are letters (A, B, C, D, ...), your decision
must be exactly one of FIRST, SECOND, or tie -- never a task option letter.

Choose FIRST or SECOND if one is preferable; otherwise choose tie. Do not favor a
solution because of its presentation position.
Return {"decision": "FIRST"} or {"decision": "SECOND"} or {"decision": "tie"}.
"""


class LLMPairwiseJudge(PairwiseJudge):
    """Concrete f_judge; preserves the reviewed ABC signature.
    
    randomize then remap to probe-vs-anchor y.
    an independent random.Random instance, an unbiased
    randrange(2) draw per call, optional injected RNG, anonymous A/B payload,
    and strict JSON parsing. No seed is imposed by default. Supplied content
    is preserved; anonymization removes metadata, not words in executions."""

    def __init__(self, client: "LLMClient", *, rng: "Optional[random.Random]" = None) -> None:
        self.client = client
        self.rng = rng if rng is not None else random.Random()

    def judge(
        self, task: Task, anchor_execution: Execution,
        probe_execution: Execution, probe_index: int,
    ) -> PairwiseDecision:
        positive_int(probe_index, "probe_index")
        check_execution(task, anchor_execution, anchor=True)
        check_execution(task, probe_execution, anchor=False)
        if probe_execution.ref.probe_index != probe_index:
            raise ValueError("probe_index does not match execution reference")
        probe_first = self.rng.randrange(2) == 0
        a, b = ((probe_execution, anchor_execution) if probe_first
                else (anchor_execution, probe_execution))


        def build(response):
            obj = object_keys(response, ("decision",))
            decision = obj["decision"]
            if not isinstance(decision, str) or decision not in ("FIRST", "SECOND", "tie"):
                raise LLMResponseError("decision must be exactly FIRST, SECOND, or tie")
            probe_label = "FIRST" if probe_first else "SECOND"
            y = 0 if decision == "tie" else (1 if decision == probe_label else -1)
            return PairwiseDecision(task_id=task.task_id, probe_index=probe_index, y=y)

        return ask_json_validated(self.client, _JUDGE_PROMPT, {
            "task": task.content, "FIRST": a.content, "SECOND": b.content,
        }, build)
