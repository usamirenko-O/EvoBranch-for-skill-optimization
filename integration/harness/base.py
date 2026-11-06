"""Solver sessions and common task-execution interfaces."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from integration.common.solver_client import SolverConfig
from integration.experiment.cost import PhaseCostAccumulator


class SolverError(RuntimeError):
    """A solver call failed after the SDK's own retries."""


@dataclass
class SolverSession:
    """A metered connection to the frozen solver for one phase of one run."""

    config: SolverConfig
    accumulator: PhaseCostAccumulator
    role: str = "solver"
    _client: Any = field(default=None, repr=False)

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.config.base_url,
                api_key=self.config.api_key,
                timeout=self.config.timeout_seconds,
                max_retries=self.config.max_retries,
            )
        return self._client

    def chat(self, messages: list, *, tools: list | None = None, tool_choice: str | None = None):
        """One chat-completion call. Returns the raw SDK message object.

        Usage is recorded before the value is returned, so a caller cannot
        consume a response without its cost being counted.
        """
        client = self._ensure_client()
        kwargs: dict = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"
        resp = client.chat.completions.create(**kwargs)
        usage = getattr(resp, "usage", None)
        cached = None
        if usage is not None:
            details = getattr(usage, "prompt_tokens_details", None)
            if details is not None:
                cached = getattr(details, "cached_tokens", None)
        self.accumulator.add(
            llm_calls=1,
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0) if usage else None,
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0) if usage else None,
            cached_tokens=cached,
            token_source="provider_usage",
            role=self.role,
        )
        if not resp.choices:
            raise SolverError("solver returned no choices")
        return resp.choices[0].message

    def text(self, system: str, user: str) -> str:
        msg = self.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}]
        )
        return msg.content or ""


@dataclass
class TaskRun:
    """The result of executing one skill on one task."""

    task_id: str
    output: Any
    transcript: str = ""
    llm_calls: int = 0
    error: str = ""
    extra: dict = field(default_factory=dict)


class DirectHarness(ABC):
    """Execute one skill on one task through the frozen solver."""

    benchmark: str = ""

    @abstractmethod
    def run_task(self, skill_text: str, task, session: SolverSession) -> TaskRun:
        """Execute `task` under `skill_text`. Never scores, never picks a model."""

    def close(self) -> None:
        """Release any environment/process resources. Default: nothing."""
