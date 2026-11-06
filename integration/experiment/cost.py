"""Track optimization and deployment costs separately."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

TOKEN_SOURCES = ("provider_usage", "runtime_tracker", "mixed", "unavailable")
"""Where a phase's token counts came from.

provider_usage  -- the model provider's own `usage` block (authoritative).
runtime_tracker -- an optimizer's own in-process token tracker.
mixed           -- more than one of the above contributed to this phase.
unavailable     -- no authoritative count existed. Tokens stay None.
"""

PHASES = ("optimization", "deployment")


@dataclass(frozen=True)
class PhaseCost:
    llm_calls: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cached_tokens: int | None = None
    token_source: str = "unavailable"
    #: finer-grained accounting an optimizer already produces, preserved
    #: verbatim (e.g. {"solver_side": {...}, "optimizer_side": {...}} or
    #: SkillGen's per-stage rows). Never inspected by the summary.
    per_role: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.token_source not in TOKEN_SOURCES:
            raise ValueError(
                f"token_source must be one of {TOKEN_SOURCES}, got {self.token_source!r}"
            )
        if self.llm_calls < 0:
            raise ValueError("llm_calls must be >= 0")

    @staticmethod
    def derive_total(input_tokens: int | None, output_tokens: int | None) -> int | None:
        """Sum only when both halves are actually known."""
        if input_tokens is None or output_tokens is None:
            return None
        return int(input_tokens) + int(output_tokens)

    @classmethod
    def build(
        cls,
        *,
        llm_calls: int,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cached_tokens: int | None = None,
        token_source: str = "unavailable",
        per_role: dict | None = None,
    ) -> "PhaseCost":
        return cls(
            llm_calls=int(llm_calls),
            input_tokens=None if input_tokens is None else int(input_tokens),
            output_tokens=None if output_tokens is None else int(output_tokens),
            total_tokens=cls.derive_total(input_tokens, output_tokens),
            cached_tokens=None if cached_tokens is None else int(cached_tokens),
            token_source=token_source,
            per_role=dict(per_role or {}),
        )

    @classmethod
    def from_legacy_bucket(
        cls,
        bucket: dict | None,
        *,
        token_source: str = "runtime_tracker",
        per_role: dict | None = None,
    ) -> "PhaseCost":
        """Convert a calls-and-token usage dictionary to PhaseCost."""
        if not bucket:
            return cls(llm_calls=0, token_source="unavailable", per_role=dict(per_role or {}))
        return cls.build(
            llm_calls=int(bucket.get("calls", 0) or 0),
            input_tokens=bucket.get("prompt_tokens"),
            output_tokens=bucket.get("completion_tokens"),
            cached_tokens=bucket.get("cached_tokens"),
            token_source=token_source,
            per_role=per_role,
        )

    def to_dict(self) -> dict:
        return {
            "llm_calls": self.llm_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens,
            "token_source": self.token_source,
            "per_role": dict(self.per_role),
        }


def empty_cost() -> PhaseCost:
    return PhaseCost(llm_calls=0, token_source="unavailable")


class PhaseCostAccumulator:
    """Reset-then-snapshot accumulator, one instance per phase.

    Deliberately dumb: a driver calls `add()` once per authoritative usage
    block it receives, and `snapshot()` at the phase boundary. There is no
    inference, no proration, and no way to add a count that did not come from
    a provider/runtime -- `token_source` must be declared on every `add()`.
    """

    def __init__(self, phase: str) -> None:
        if phase not in PHASES:
            raise ValueError(f"phase must be one of {PHASES}, got {phase!r}")
        self.phase = phase
        # Harnesses may execute tasks on a thread pool; a phase total that
        # loses calls to a race would be silently wrong, which is worse than
        # the negligible cost of a lock around an integer add.
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        self._calls = 0
        self._input = 0
        self._output = 0
        self._cached = 0
        self._saw_input = False
        self._saw_output = False
        self._saw_cached = False
        self._sources: set = set()
        self._per_role: dict = {}

    def add(
        self,
        *,
        llm_calls: int = 1,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cached_tokens: int | None = None,
        token_source: str,
        role: str | None = None,
    ) -> None:
        if token_source not in TOKEN_SOURCES or token_source == "unavailable":
            raise ValueError(
                "add() requires an authoritative token_source "
                f"(one of {[s for s in TOKEN_SOURCES if s != 'unavailable']}); got {token_source!r}"
            )
        with self._lock:
            self._calls += int(llm_calls)
            self._sources.add(token_source)
            if input_tokens is not None:
                self._input += int(input_tokens)
                self._saw_input = True
            if output_tokens is not None:
                self._output += int(output_tokens)
                self._saw_output = True
            if cached_tokens is not None:
                self._cached += int(cached_tokens)
                self._saw_cached = True
            if role:
                bucket = self._per_role.setdefault(
                    role, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}
                )
                bucket["calls"] += int(llm_calls)
                bucket["prompt_tokens"] += int(input_tokens or 0)
                bucket["completion_tokens"] += int(output_tokens or 0)
                bucket["cached_tokens"] += int(cached_tokens or 0)

    def adopt_native_totals(
        self,
        *,
        llm_calls: int,
        input_tokens: int | None,
        output_tokens: int | None,
        cached_tokens: int | None = None,
        role: str,
    ) -> None:
        """Fold an optimizer's OWN token tracker totals into this phase.

        Used when the optimizer ran its solver calls itself (SkillOpt runs its
        native rollouts in a subprocess), so the authoritative counts come from
        that optimizer's tracker rather than from a `SolverSession`. Source is
        `runtime_tracker`, which is why a phase that mixes both reports
        `token_source="mixed"` rather than claiming provider-level precision
        for all of it.
        """
        self.add(
            llm_calls=llm_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            token_source="runtime_tracker",
            role=role,
        )

    def snapshot(self) -> PhaseCost:
        if not self._sources:
            source = "unavailable"
        elif len(self._sources) == 1:
            source = next(iter(self._sources))
        else:
            source = "mixed"
        return PhaseCost.build(
            llm_calls=self._calls,
            input_tokens=self._input if self._saw_input else None,
            output_tokens=self._output if self._saw_output else None,
            cached_tokens=self._cached if self._saw_cached else None,
            token_source=source,
            per_role=self._per_role,
        )
