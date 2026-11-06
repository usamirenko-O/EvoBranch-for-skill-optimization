"""Call a shared OpenAI-compatible solver endpoint and record token usage."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

# Defaults describe *shape*, not a live account. Nothing here calls Vertex,
# GCP project ids, or region names -- those live only in
# integration/common/litellm_proxy_config.yaml, which a human fills in and
# runs by hand (see PHASE1_REPORT.md "Real Vertex smoke test commands").
DEFAULT_BASE_URL = "http://localhost:4000/v1"
DEFAULT_MODEL = "livemath-solver"
DEFAULT_API_KEY = "dummy"  # LiteLLM's local proxy accepts any non-empty key.
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TIMEOUT_SECONDS = float(__import__("os").environ.get("SKILLOPTEVO_SOLVER_TIMEOUT", "300.0"))  


DEFAULT_MAX_RETRIES = 2


@dataclass(frozen=True)
class SolverConfig:
    """The one shared "solver call contract" (checklist item A.2).

    Every method that calls the frozen solver for LiveMath should build its
    request through this config rather than picking its own
    temperature/max_tokens, so a cross-method comparison isn't silently
    confounded by differing generation settings.
    """

    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    api_key: str = DEFAULT_API_KEY
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES


def solver_config_from_env(prefix: str = "LIVEMATH_SOLVER") -> SolverConfig:
    """Build a SolverConfig from `{prefix}_*` env vars, falling back to defaults.

    No secret or project-specific value is hardcoded here; set
    `LIVEMATH_SOLVER_BASE_URL` / `_MODEL` / `_API_KEY` to point at a real
    running proxy before making an actual call.
    """
    return SolverConfig(
        base_url=os.environ.get(f"{prefix}_BASE_URL", DEFAULT_BASE_URL),
        model=os.environ.get(f"{prefix}_MODEL", DEFAULT_MODEL),
        api_key=os.environ.get(f"{prefix}_API_KEY", DEFAULT_API_KEY),
        temperature=float(os.environ.get(f"{prefix}_TEMPERATURE", DEFAULT_TEMPERATURE)),
        max_tokens=int(os.environ.get(f"{prefix}_MAX_TOKENS", DEFAULT_MAX_TOKENS)),
        timeout_seconds=float(
            os.environ.get(f"{prefix}_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)
        ),
        max_retries=int(os.environ.get(f"{prefix}_MAX_RETRIES", DEFAULT_MAX_RETRIES)),
    )


def _client(config: SolverConfig) -> OpenAI:
    return OpenAI(
        base_url=config.base_url, api_key=config.api_key, timeout=config.timeout_seconds,
        max_retries=config.max_retries,
    )


def call_solver_raw(system: str, user: str, *, config: SolverConfig | None = None) -> Any:
    """Call the shared solver endpoint and return the response with token usage."""
    cfg = config or solver_config_from_env()
    client = _client(cfg)
    return client.chat.completions.create(
        model=cfg.model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )


def call_solver(system: str, user: str, *, config: SolverConfig | None = None) -> str:
    """Convenience wrapper returning only the response text."""
    resp = call_solver_raw(system, user, config=config)
    return resp.choices[0].message.content or ""


def usage_from_response(resp: Any) -> dict[str, int | None]:
    """Extract a run-logger-friendly usage dict from an OpenAI ChatCompletion.

    `cached_tokens` is best-effort: only some OpenAI-compatible backends
    (including LiteLLM's Vertex passthrough for some models) populate
    `usage.prompt_tokens_details.cached_tokens`; absence is not an error.
    """
    usage = getattr(resp, "usage", None)
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": None}
    cached = None
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        cached = getattr(details, "cached_tokens", None)
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "cached_tokens": cached,
    }
