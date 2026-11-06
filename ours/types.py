"""Data types for skills, evidence, proposals, scores, and search state."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any


def skill_fingerprint(skill: str) -> str:
    """Content address of a skill document. Two candidates with the same text
    are the same candidate for scoring purposes."""
    return hashlib.sha256(skill.encode("utf-8")).hexdigest()[:16]


class Origin(str, Enum):
    """How a *competitive* candidate entered the pool."""

    PARENT = "parent"
    OPERATOR = "operator"


class Stage(str, Enum):
    """Ordered stages of one search round. The order is authoritative.

    ``PROPOSE`` produces raw operator material, not candidates.
    ``OPERATOR`` is where the round's actual challengers are built.
    ``POOL_COMPLETE`` is the competitive pool ``C_t = Beam_t ∪ O_t`` being
    sealed -- raw proposals are excluded from it by construction.
    ``SCORE`` reuses each incumbent's fixed-validation score and measures only
    the candidates that do not have one.
    """

    COLLECT_EVIDENCE = "collect_evidence"
    PARTITION = "partition"
    PROPOSE = "propose"
    OPERATOR = "operator"
    POOL_COMPLETE = "pool_complete"
    SCORE = "score"
    SELECT = "select"


ROUND_STAGE_ORDER: tuple[Stage, ...] = (
    Stage.COLLECT_EVIDENCE,
    Stage.PARTITION,
    Stage.PROPOSE,
    Stage.OPERATOR,
    Stage.POOL_COMPLETE,
    Stage.SCORE,
    Stage.SELECT,
)


@dataclass(frozen=True)
class CostAccounting:
    """Aggregate model calls, tokens, and costs associated with a candidate."""

    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    wall_time_s: float = 0.0

    def merged(self, other: CostAccounting) -> CostAccounting:
        return CostAccounting(
            llm_calls=self.llm_calls + other.llm_calls,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            wall_time_s=self.wall_time_s + other.wall_time_s,
        )


@dataclass(frozen=True)
class Score:
    """A *comparable* validation score: the one number candidates are ranked by."""

    value: float
    components: dict[str, Any] = field(default_factory=dict)
    evaluator: str = ""
    cache_hit: bool = False
    n_items: int = 0
    validation_context: str = ""
    measured_in_round: int | None = None

    def matches_context(self, validation_context: str) -> bool:
        return self.validation_context == validation_context


@dataclass(frozen=True)
class Evidence:
    """Task-level evidence gathered for one parent skill.

    ``payload`` is whatever the adapter needs (for SkillOpt: rollout result
    dicts plus a prediction directory). The framework only ever reads
    ``task_ids`` and hands ``payload`` back to the adapter.
    """

    evidence_id: str
    parent_id: str
    optimizer_name: str
    task_ids: tuple[str, ...]
    payload: Any = None
    workspace_dir: str = ""
    cost: CostAccounting = field(default_factory=CostAccounting)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceSubset:
    """One partition of an :class:`Evidence` — the E1/E2/E3 of the flow.

    Produced by an :class:`~ours.partition.EvidencePartitioner` (which picks
    the task ids) and materialised by the adapter (which builds ``payload``).
    """

    subset_id: str
    evidence_id: str
    parent_id: str
    task_ids: tuple[str, ...]
    payload: Any = None
    partitioner: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RawProposal:
    """A raw native proposal: OPERATOR MATERIAL, never a competitive candidate."""

    proposal_id: str
    optimizer_name: str
    round_id: int
    skill: str
    parent_id: str
    evidence_subset_id: str
    skill_hash: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)
    workspace_dir: str = ""
    cost: CostAccounting = field(default_factory=CostAccounting)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.skill_hash:
            object.__setattr__(self, "skill_hash", skill_fingerprint(self.skill))


class CandidateStateError(RuntimeError):
    """A CandidateState invariant was violated."""


@dataclass
class CandidateState:
    """One COMPETITIVE candidate: a parent or an operator output."""

    candidate_id: str
    optimizer_name: str
    round_id: int
    skill: str
    origin: Origin
    parent_id: str | None = None
    skill_hash: str = ""
    evidence_subset_id: str | None = None
    proposal_provenance: dict[str, Any] = field(default_factory=dict)
    operator_provenance: dict[str, Any] = field(default_factory=dict)
    score: Score | None = None
    first_scored_in_round: int | None = None
    workspace_dir: str = ""
    cost: CostAccounting = field(default_factory=CostAccounting)
    lineage: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.skill_hash:
            self.skill_hash = skill_fingerprint(self.skill)

    @property
    def is_scored(self) -> bool:
        return self.score is not None

    def has_valid_score(self, validation_context: str) -> bool:
        """True when the stored score may be reused as-is.

        Reuse is allowed exactly when the candidate already holds a score
        measured under *this* fixed evaluation condition. Same skill, same
        fixed validation set, same evaluator/solver condition => same number,
        so re-running it would burn budget to reproduce a value we have.
        """
        return self.score is not None and self.score.matches_context(
            validation_context
        )

    def attach_score(self, score: Score) -> None:
        if self.score is not None:
            raise CandidateStateError(
                f"candidate {self.candidate_id} already scored "
                f"({self.score.value}); refusing to score it twice"
            )
        self.score = score
        if self.first_scored_in_round is None:
            self.first_scored_in_round = (
                score.measured_in_round
                if score.measured_in_round is not None
                else self.round_id
            )

    @property
    def lineage_including_self(self) -> tuple[str, ...]:
        """Ancestry ending in this candidate, without repeating it.

        Used both when carrying a survivor forward and when an operator
        candidate records the incumbent it descends from, so an id appears at
        most once however many rounds a skill survives.
        """
        if self.lineage[-1:] == (self.candidate_id,):
            return self.lineage
        return self.lineage + (self.candidate_id,)

    def clear_score(self, reason: str) -> Score | None:
        """Drop a stored score that is no longer valid, loudly and on purpose.

        The only legitimate caller is the scorer, when the stored score was
        measured under a different evaluation condition than the one now in
        force. ``reason`` is kept in ``metadata`` so a run can show why a
        candidate was re-evaluated.
        """
        dropped, self.score = self.score, None
        if dropped is not None:
            self.metadata.setdefault("score_invalidations", []).append({
                "reason": reason,
                "dropped_value": dropped.value,
                "dropped_context": dropped.validation_context,
            })
        return dropped

    def carried_forward(self, round_id: int) -> CandidateState:
        """This candidate as a validated parent of the next round."""
        from ours import history as _history

        return replace(
            self,
            round_id=round_id,
            origin=Origin.PARENT,
            lineage=self.lineage_including_self,
            metadata=_history.carry_forward_metadata(self.metadata),
        )
