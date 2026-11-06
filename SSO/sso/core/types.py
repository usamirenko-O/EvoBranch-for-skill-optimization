"""Typed data structures for SSO's Evidence Aggregation and Ranking step."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, List, Optional


class Activation(Enum):
    """alpha_ij(r) / alpha_ic(r) in {1, 0, bot} (eq. 7).

    PRESENT / ABSENT are the two "clear" states (numeric 1 / 0). UNCLEAR is
    the paper's bot ("unknown / unclear") state and must never be coerced to
    0 -- eq. 9's changed-cluster set explicitly requires both compared
    activations to be clear before a behavior can count as "changed".
    """

    PRESENT = "present"
    ABSENT = "absent"
    UNCLEAR = "unclear"


class Intent(Enum):
    """Update direction assigned to a cluster from sign(Q_c) (eq. 14)."""

    ENCOURAGE = "encourage"  # Q_c > 0
    SUPPRESS = "suppress"    # Q_c < 0
    NEUTRAL = "neutral"      # Q_c == 0; discarded before ranking (eq. 14)


@dataclass(frozen=True)
class ExecutionRef:
    """Identifies one observable execution r for a task (eq. 1-2, 4).

    Either the current-skill anchor execution r_i^o, or the k-th probe
    execution r_i^k, for task i. This is the key type used to index
    `BehaviorRecord.activations` and to look up alpha_ic(r).
    """

    task_id: str
    is_anchor: bool
    probe_index: Optional[int] = None  # k; required iff is_anchor is False

    def __post_init__(self) -> None:
        if self.is_anchor and self.probe_index is not None:
            raise ValueError("an anchor ExecutionRef must not carry a probe_index")
        if not self.is_anchor and self.probe_index is None:
            raise ValueError("a probe ExecutionRef must carry a probe_index (k)")

    @staticmethod
    def anchor(task_id: str) -> "ExecutionRef":
        """r_i^o."""
        return ExecutionRef(task_id=task_id, is_anchor=True)

    @staticmethod
    def probe(task_id: str, probe_index: int) -> "ExecutionRef":
        """r_i^k for the given k. NOTE: this is one execution of the k-th
        skill *probe* P_k -- never the rendered candidate skill produced by
        f_render. Probes are exploratory and are never promoted directly
        (see docs/paper_mapping.md, "Probes vs. the rendered candidate")."""
        return ExecutionRef(task_id=task_id, is_anchor=False, probe_index=probe_index)


@dataclass(frozen=True)
class PairwiseDecision:
    """y_i^k in {-1, 0, +1}: the judge's anchored decision for task i,
    probe k (eq. 5).

    +1: probe execution r_i^k preferred over the anchor.
    -1: anchor execution r_i^o preferred over the probe.
     0: tie.

    This type only carries the *outcome* of f_judge; how f_judge is queried
    (including the paper's position-randomization to mitigate order bias) is
    out of scope for this deterministic core.
    """

    task_id: str
    probe_index: int
    y: int

    def __post_init__(self) -> None:
        if self.y not in (-1, 0, 1):
            raise ValueError(f"y must be in {{-1, 0, 1}}, got {self.y!r}")


@dataclass
class BehaviorRecord:
    """One (beta_ij, alpha_ij) pair produced by f_beh for task x_i (eq. 6-7).

    `activations` gives alpha_ij(r) for every execution r the behavior
    extractor was shown for this task (the anchor r_i^o and each probe
    execution r_i^k, k=1..K). An ExecutionRef missing from this mapping is
    treated the same as an explicit UNCLEAR entry by
    `evidence.resolve_cluster_activation` (defensive default; see
    docs/assumptions.md).
    """

    task_id: str
    record_id: str  # local id for beta_ij within task i (e.g. "i:j")
    description: str  # beta_ij, verbatim natural-language text
    activations: Dict[ExecutionRef, Activation] = field(default_factory=dict)


@dataclass
class BehaviorCluster:
    """One cluster c in C_t: a group of semantically-equivalent behavior
    records produced by f_cluster (eq. 8).

    The paper requires `representative_description` to be one of the
    original member descriptions, never a newly generated summary. This
    module takes clustering as already done (an input), so it does not
    enforce that constraint -- f_cluster's implementation (a later phase)
    is responsible for it.
    """

    cluster_id: str
    representative_description: str
    member_records: List[BehaviorRecord] = field(default_factory=list)

    def records_for_task(self, task_id: str) -> List[BehaviorRecord]:
        """Task-local behavior records assigned to this cluster, used by
        `evidence.resolve_cluster_activation` to compute alpha_ic(r)."""
        return [r for r in self.member_records if r.task_id == task_id]


@dataclass
class ClusterScore:
    """Intermediate per-cluster aggregate before top-U selection.

    Holds Q_c, E_c (eq. 13), the resulting intent (eq. 14), and the set of
    tasks that contributed nonzero evidence to c, used as the paper's final
    ranking tie-breaker ("evidence from more distinct tasks").
    """

    cluster_id: str
    description: str
    q: float
    e: float
    contributing_tasks: FrozenSet[str]
    intent: Intent

    @property
    def contributing_task_count(self) -> int:
        return len(self.contributing_tasks)


@dataclass(frozen=True)
class Skill:
    """A complete skill document with optional identity metadata."""

    content: str
    skill_id: Optional[str] = None


@dataclass(frozen=True)
class Task:
    """A task x_i (eq. 1).
    
    tasks are the unit f_exec executes a skill against, and
    the unit f_judge/f_beh are shown alongside executions.
    modeled as opaque `content` text (e.g. the task
    prompt/instructions) keyed by `task_id`, matching the `task_id` string
    already used throughout `ExecutionRef`/`PairwiseDecision`/
    `BehaviorRecord`. No structured task schema is assumed."""

    task_id: str
    content: str


@dataclass(frozen=True)
class Execution:
    """Execution text and its task, skill, and probe reference."""

    ref: ExecutionRef
    content: str


@dataclass(frozen=True)
class Probe:
    """One generated probe P_k (eq. 3-4), paired with its index k.
    
    the paper never names this pairing as a type -- it
    is pure bookkeeping introduced here because downstream code needs k
    attached to its probe skill (e.g. to build `ExecutionRef.probe(task_id,
    k)` once the probe is executed against a task). `probe_index` carries no
    algorithmic meaning of its own beyond identifying which of the K probes
    this is."""

    probe_index: int
    skill: Skill

    def __post_init__(self) -> None:
        if self.probe_index < 1:
            raise ValueError(f"probe_index must be >= 1, got {self.probe_index!r}")


@dataclass
class BehaviorUnit:
    """One element of U_t: a ranked, non-neutral behavior cluster retained
    within the update budget U (eq. 14, "Retain up to U highest-ranked
    clusters as the behavior units U_t").

    U_t is the *input* to f_render (eq. 15), not a skill itself. Rendering
    U_t into a candidate skill S~_{t+1}, and validating that candidate, are
    both out of scope for this module (later phases).
    """

    cluster_id: str
    description: str
    intent: Intent
    q: float
    e: float
    contributing_task_count: int
