"""Execute SSO skills on LiveMath through the shared solver client."""
from __future__ import annotations

from typing import Callable

from integration.common.paths import SSO_ROOT, ensure_on_sys_path
from integration.common.solver_client import (
    SolverConfig,
    call_solver_raw,
    solver_config_from_env,
    usage_from_response,
)

ensure_on_sys_path(SSO_ROOT)

from sso.core.types import Execution, ExecutionRef, Skill, Task  # noqa: E402
from sso.executors.base import Executor  # noqa: E402

LlmCall = Callable[..., str]  # (system: str, user: str) -> str


class LiveMathExecutor(Executor):
    """f_exec for LiveMath: render skill as system prompt, task as user prompt.

    `llm_call` defaults to `integration.common.solver_client.call_solver`
    against the shared proxy, but accepts any `(system, user) -> str`
    callable -- tests inject a deterministic fake here instead of making a
    real call.
    """

    def __init__(
        self,
        llm_call: LlmCall | None = None,
        *,
        solver_config: SolverConfig | None = None,
    ) -> None:
        self._llm_call = llm_call
        self._solver_config = solver_config or solver_config_from_env()
        # Purely additive: cumulative solver-side usage across every real
        # (non-injected) execute() call on this instance. A caller that
        # injects a fake llm_call (dry-run/tests) gets no usage data here --
        # there is nothing real to count. Mirrors sso.llm.client's
        # ChatCompletionsClient.usage_snapshot()/reset_usage() shape so a
        # driver can combine solver-side and optimizer-side totals the same
        # way.
        self._usage_totals = {
            "calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "cached_tokens": 0, "total_tokens": 0,
        }

    def usage_snapshot(self) -> dict:
        return dict(self._usage_totals)

    def reset_usage(self) -> None:
        for key in self._usage_totals:
            self._usage_totals[key] = 0

    def execute(self, skill: Skill, task: Task) -> Execution:
        if self._llm_call is not None:
            raw = self._llm_call(system=skill.content, user=task.content)
        else:
            resp = call_solver_raw(skill.content, task.content, config=self._solver_config)
            raw = resp.choices[0].message.content or ""
            usage = usage_from_response(resp)
            self._usage_totals["calls"] += 1
            self._usage_totals["prompt_tokens"] += usage["prompt_tokens"] or 0
            self._usage_totals["completion_tokens"] += usage["completion_tokens"] or 0
            self._usage_totals["cached_tokens"] += usage.get("cached_tokens") or 0
            self._usage_totals["total_tokens"] += (
                (usage["prompt_tokens"] or 0) + (usage["completion_tokens"] or 0)
            )
        # A local placeholder ref; sso.core._execution.execute_as rebinds it
        # to the caller's contextual ref (anchor vs. probe k) after the call,
        # per Executor.execute's own contract docstring.
        return Execution(ExecutionRef.anchor(task.task_id), raw)
