"""Render ranked behavior units into a complete candidate skill."""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import Sequence

from sso.core.types import BehaviorUnit, Intent, Skill
from sso.llm.client import LLMClient
from sso.llm._structured import ask_json_validated, complete_skill


class Renderer(ABC):
    """f_render: renders a new complete candidate skill from the current skill and the ranked, budget-limited behavior units U_t."""

    @abstractmethod
    def render(self, current_skill: Skill, behavior_units: Sequence[BehaviorUnit]) -> Skill:
        """Render S~_{t+1} from S_t and U_t."""
        raise NotImplementedError


# PAPER_SPECIFIED: eq. 15, Algorithm 1 lines 32-35; whole-skill synthesis
# from selected evidence, encourage/suppress constraints, no unsupported opposite.
# REPRODUCTION_CHOICE: exact wording and complete-skill JSON schema.
_RENDER_PROMPT = """Create a new complete candidate skill from the current
skill and the supplied ranked behavior units, whose evidence is aggregated
across tasks and exploratory probes. You may add or strengthen an encouraged
behavior and remove, weaken, or restrict a suppressed behavior. Never
introduce an unsupported opposite of a suppressed behavior. Use the supplied
directions and evidence; do not invent additional behavior evidence.
The candidate can combine supported behaviors from different probes and need
not match any individual probe. Probes are exploratory only; this request
renders a candidate from the current skill and aggregated evidence.
Return the ENTIRE standalone reusable skill, including retained instructions,
not a diff, patch, edit request, or reference to omitted sections.
Return {"document_type": "complete_skill", "content": "..."}.
"""


class LLMRenderer(Renderer):
    """Concrete f_render; no candidate acceptance or optimization policy.
    
    reject empty U_t as a caller error without a call;
    check non-neutral, finite, sign-consistent evidence; serialize existing
    fields in supplied rank order. Full-document schema and prose constraints
    do not prove semantic fidelity or require lexical inequality to S_t."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def render(self, current_skill: Skill, behavior_units: Sequence[BehaviorUnit]) -> Skill:
        if not behavior_units:
            raise ValueError("empty behavior units: caller must skip rendering")
        data = []
        seen = set()
        for unit in behavior_units:
            if (unit.intent not in (Intent.ENCOURAGE, Intent.SUPPRESS)
                    or not math.isfinite(unit.q) or not math.isfinite(unit.e)
                    or unit.q == 0 or unit.e < abs(unit.q)
                    or (unit.q > 0) != (unit.intent == Intent.ENCOURAGE)):
                raise ValueError("behavior unit has invalid evidence or intent")
            if unit.cluster_id in seen:
                raise ValueError("duplicate behavior unit cluster id")
            seen.add(unit.cluster_id)
            data.append({
                "cluster_id": unit.cluster_id, "description": unit.description,
                "intent": unit.intent.value, "q": unit.q, "e": unit.e,
                "contributing_task_count": unit.contributing_task_count,
            })
        content = ask_json_validated(self.client, _RENDER_PROMPT, {
            "current_skill": current_skill.content, "behavior_units": data,
        }, complete_skill)
        return Skill(content=content)
