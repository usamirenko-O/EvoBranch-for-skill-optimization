"""Shared text-completion boundary and optional HTTP transport.

this protocol, Chat Completions wire format, JSON mode,
UTF-8, synchronous single requests, and error policy are engineering choices,
not SSO paper requirements. No SDK or third-party runtime dependency is used."""

from __future__ import annotations

import json
import math
from typing import Optional, Protocol
from urllib import error, request
from urllib.parse import urlsplit


class LLMClient(Protocol):
    """one stateless system/user request -> response text.
    
    Implementations must fail visibly on incomplete/refused responses and
    transport errors. Modules do not depend on provider-specific arguments."""

    def complete(self, *, system: str, user: str) -> str:
        ...


class LLMClientError(RuntimeError):
    """Transport failure, refusal, or incomplete/invalid provider envelope."""


class ChatCompletionsClient:
    """explicit OpenAI-compatible endpoint and model.
    
    `endpoint` is the full chat/completions URL; `api_key` is optional for
    local compatible servers. JSON-object mode is required of the endpoint.
    Decoding settings are omitted (provider defaults); timeout defaults to
    120 seconds. No retries, repair, streaming, tools, or conversation memory.
    Model selection is explicit: this adapter does not claim model parity
    with the paper. Keep the judge's client/model fixed across comparisons."""

    def __init__(
        self, *, endpoint: str, model: str, api_key: Optional[str] = None,
        timeout: float = 120.0, max_tokens: Optional[int] = None,
        temperature: Optional[float] = None, max_retries: int = 2,
    ) -> None:
        url = urlsplit(endpoint)
        if url.scheme not in ("http", "https") or not url.netloc:
            raise ValueError("endpoint must be a full HTTP(S) URL")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be nonempty text")
        if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        if max_tokens is not None and (isinstance(max_tokens, bool) or max_tokens <= 0):
            raise ValueError("max_tokens must be a positive int when given")
        if temperature is not None and (isinstance(temperature, bool) or not math.isfinite(temperature)):
            raise ValueError("temperature must be finite when given")
        if isinstance(max_retries, bool) or max_retries < 0:
            raise ValueError("max_retries must be a non-negative int")
        self.endpoint = endpoint
        self.model = model
        self._api_key = api_key
        self.timeout = timeout
        self._max_retries = max_retries


        self._max_tokens = max_tokens
        self._temperature = temperature
        # REPRODUCTION_CHOICE, purely additive: the paper specifies none of
        # this. Cumulative usage across every successful complete() call on
        # this instance, for external cost accounting. Does not change the
        # LLMClient Protocol's `-> str` contract; a caller that never reads
        # usage_snapshot()/reset_usage() sees identical behavior to before.
        self._usage_totals = {
            "calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "cached_tokens": 0, "total_tokens": 0,


            "failed_attempts": 0,
        }

    def usage_snapshot(self) -> dict:
        """Non-destructive copy of cumulative usage since the last reset."""
        return dict(self._usage_totals)

    def reset_usage(self) -> None:
        for key in self._usage_totals:
            self._usage_totals[key] = 0

    def _record_usage(self, usage: object) -> None:
        # A call that reached here already passed complete()'s own
        # well-formedness checks -- it is a real, successful call and must
        # always be counted, even if this particular provider/response
        # omits usage (some do). Undercounting `calls` when tokens are
        # merely unavailable would be worse than reporting 0 tokens for it.
        self._usage_totals["calls"] += 1
        if not isinstance(usage, dict):
            return
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        total = int(usage.get("total_tokens") or (prompt + completion))
        details = usage.get("prompt_tokens_details")
        cached = 0
        if isinstance(details, dict):
            cached = int(details.get("cached_tokens") or 0)
        self._usage_totals["prompt_tokens"] += prompt
        self._usage_totals["completion_tokens"] += completion
        self._usage_totals["cached_tokens"] += cached
        self._usage_totals["total_tokens"] += total

    def complete(self, *, system: str, user: str) -> str:
        """Request a completion with bounded retries for transport errors or truncated responses."""
        last_err: LLMClientError | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return self._complete_once(system=system, user=user)
            except LLMClientError as exc:
                last_err = exc
                self._usage_totals["failed_attempts"] += 1
        assert last_err is not None
        raise last_err

    def _complete_once(self, *, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        if self._max_tokens is not None:
            payload["max_tokens"] = self._max_tokens
        if self._temperature is not None:
            payload["temperature"] = self._temperature
        headers = {"Content-Type": "application/json"}
        if self._api_key is not None:
            headers["Authorization"] = "Bearer " + self._api_key
        req = request.Request(
            self.endpoint, data=json.dumps(payload, allow_nan=False).encode("utf-8"),
            headers=headers, method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            raise LLMClientError(f"LLM HTTP request failed (status {exc.code})") from None
        except (error.URLError, OSError, UnicodeError):
            raise LLMClientError("LLM transport failed or returned invalid UTF-8") from None
        try:
            envelope = json.loads(body)
            choices = envelope["choices"]
            if not isinstance(choices, list) or len(choices) != 1:
                raise ValueError
            choice = choices[0]
            message = choice["message"]
            if (choice["finish_reason"] != "stop" or message.get("refusal")
                    or message.get("tool_calls") or message.get("function_call")):
                raise ValueError
            content = message["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError
        except (ValueError, KeyError, TypeError, AttributeError):
            raise LLMClientError("LLM response is incomplete, refused, or malformed") from None
        # Recorded only once the envelope is confirmed well-formed above, so
        # a malformed/refused response (which raises and never returns) is
        # never counted as a successful call.
        self._record_usage(envelope.get("usage"))
        return content
