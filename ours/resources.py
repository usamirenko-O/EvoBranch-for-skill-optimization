"""Provide metered chat and embedding clients to search components."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Sequence

ROLE_PARTITIONER = "ours_partitioner"
ROLE_OPERATOR = "ours_operator"
ROLE_TIE_BREAK = "ours_tie_break"
ROLE_ISSUE_TRACKER = "ours_issue_tracker"

_ZERO = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}


class ResourcesUnavailable(RuntimeError):
    """A component needs model access and the run provided none."""


@dataclass
class RunResources:
    """Model access for framework components, metered into one phase."""

    optimizer_model: str
    proxy_base_url: str
    api_key: str
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout_seconds: float = 300.0
    max_retries: int = 2
    embedding_model: str | None = None
    #: The run's OPTIMIZATION `PhaseCostAccumulator`. None means "meter
    #: locally only" (tests); a real run always passes one.
    accumulator: Any = None
    seed: int | None = None
    last_finish_reason: str | None = field(default=None, repr=False)
    _client: Any = field(default=None, repr=False)
    _usage: dict = field(default_factory=dict, repr=False)
    _lock: Any = field(default_factory=threading.Lock, repr=False)

    # ── client ───────────────────────────────────────────────────────────

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.proxy_base_url,
                api_key=self.api_key,
                timeout=self.timeout_seconds,
                max_retries=self.max_retries,
            )
        return self._client

    # ── metering ─────────────────────────────────────────────────────────

    def _meter(self, *, role: str, usage: Any) -> None:
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0) if usage is not None else None
        completion = int(getattr(usage, "completion_tokens", 0) or 0) if usage is not None else None
        cached = None
        if usage is not None:
            details = getattr(usage, "prompt_tokens_details", None)
            if details is not None:
                cached = getattr(details, "cached_tokens", None)
        with self._lock:
            bucket = self._usage.setdefault(role, dict(_ZERO))
            bucket["calls"] += 1
            bucket["prompt_tokens"] += prompt or 0
            bucket["completion_tokens"] += completion or 0
            bucket["cached_tokens"] += cached or 0
        if self.accumulator is not None:
            self.accumulator.add(
                llm_calls=1,
                input_tokens=prompt,
                output_tokens=completion,
                cached_tokens=cached,
                token_source="provider_usage",
                role=role,
            )

    def usage_snapshot(self) -> dict:
        """role -> totals metered so far through this object."""
        with self._lock:
            return {role: dict(bucket) for role, bucket in self._usage.items()}

    @staticmethod
    def usage_delta(before: dict, after: dict) -> dict:
        """Per-role difference between two snapshots (roles with no change are
        omitted)."""
        out = {}
        for role, bucket in after.items():
            prev = before.get(role, _ZERO)
            delta = {k: bucket[k] - prev.get(k, 0) for k in _ZERO}
            if any(delta.values()):
                out[role] = delta
        return out

    # ── calls ────────────────────────────────────────────────────────────

    def chat(self, messages: list, *, role: str, model: str | None = None,
             max_tokens: int | None = None, temperature: float | None = None,
             **kwargs) -> Any:
        """One chat completion on the optimizer-side alias. Returns the SDK
        message object; usage is metered before it is returned."""
        client = self._ensure_client()
        resp = client.chat.completions.create(
            model=model or self.optimizer_model,
            messages=messages,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
            **kwargs,
        )
        self._meter(role=role, usage=getattr(resp, "usage", None))
        if not resp.choices:
            raise RuntimeError("optimizer-side model returned no choices")
        choice = resp.choices[0]
        #: finish_reason of the last chat call ("length" == the output hit
        #: max_tokens and is truncated); components record it per call.
        self.last_finish_reason = getattr(choice, "finish_reason", None)
        return choice.message

    def embed(self, texts: Sequence[str], *, role: str, model: str | None = None) -> list:
        """Embeddings for ``texts`` on the run's embedding alias, in order.
        Usage is metered before the vectors are returned."""
        alias = model or self.embedding_model
        if not alias:
            raise ResourcesUnavailable(
                "no embedding alias configured for this run "
                "(main_experiment/baseline_hparams.json#ours.embedding_model / "
                "--extra ours_embedding_model=<alias>)"
            )
        client = self._ensure_client()
        resp = client.embeddings.create(model=alias, input=list(texts))
        self._meter(role=role, usage=getattr(resp, "usage", None))
        rows = sorted(resp.data, key=lambda d: getattr(d, "index", 0))
        return [list(d.embedding) for d in rows]

    def describe(self) -> dict:
        return {
            "optimizer_model": self.optimizer_model,
            "embedding_model": self.embedding_model,
            "proxy_base_url": self.proxy_base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "metered_into": "optimization" if self.accumulator is not None else None,
        }


class RunBound:
    """Mixin for framework components that may need run resources.

    The design builds a component from its registered name and static options;
    :meth:`~ours.design.OursDesign.build_controller` then calls :meth:`bind`
    with the run's :class:`RunResources`. Components that need no model ignore
    it; components that do call :meth:`require_resources`.
    """

    resources: RunResources | None = None

    def bind(self, resources: RunResources | None) -> None:
        self.resources = resources

    def require_resources(self, what: str = "this component") -> RunResources:
        if self.resources is None:
            raise ResourcesUnavailable(
                f"{what} needs model access but the run provided no RunResources; "
                "build the controller through OursDesign.build_controller(resources=...) "
                "(the run drivers do) or pick a component that makes no model call"
            )
        return self.resources


__all__ = [
    "ROLE_ISSUE_TRACKER", "ROLE_OPERATOR", "ROLE_PARTITIONER", "ROLE_TIE_BREAK",
    "ResourcesUnavailable", "RunBound", "RunResources",
]
