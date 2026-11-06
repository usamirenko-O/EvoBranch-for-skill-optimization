"""Select the global Top-M candidates by validation score and a deterministic tie-break."""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from ours.pool import CandidatePool
from ours.registry import register
from ours.resources import RunBound
from ours.types import CandidateState


class BeamSelectionError(RuntimeError):
    """Selection was asked to run on a pool it cannot rank."""


class UnresolvedTiePolicy(RuntimeError):
    """A tie straddled the Top-M cutoff and no policy has been decided."""


class TiePolicy(str, Enum):
    PROVISIONAL_ID_ASC = "provisional_candidate_id_ascending"


class UnresolvedTieWarning(UserWarning):
    """A placeholder tie-break decided which candidate survived."""


class TieBreakPolicy(RunBound, ABC):
    """Orders candidates that hold exactly the same validation score.

    SCIENTIFIC SLOT. The global Top-M ranks by score; this decides the order
    inside one equal-score group, which matters whenever such a group straddles
    the Top-M cutoff. Register an implementation with
    ``@register("tie_break", "<name>")`` and select it by name in
    :class:`~ours.design.OursDesign`.

    ``order`` receives one equal-score group (never the whole pool) and must
    return a permutation of it. It sees full candidates, so a policy may use
    origin, lineage, skill text or cost.
    """

    name: str = "abstract"
    is_placeholder: bool = False

    @abstractmethod
    def order(self, tied: Sequence[CandidateState]) -> list[CandidateState]:
        """Return ``tied`` in survival-priority order."""


@register("tie_break", TiePolicy.PROVISIONAL_ID_ASC.value)
class CandidateIdAscendingTieBreak(TieBreakPolicy):
    """PLACEHOLDER. Ascending candidate id: deterministic and meaningless."""

    name = TiePolicy.PROVISIONAL_ID_ASC.value
    is_placeholder = True

    def order(self, tied):
        return sorted(tied, key=lambda c: c.candidate_id)


@register("tie_break", "skill_text_hash")
class SkillTextHashTieBreak(TieBreakPolicy):
    """Break exact validation-score ties by the SHA-256 hash of the full skill text."""

    name = "skill_text_hash"
    is_placeholder = False

    @staticmethod
    def key(candidate: CandidateState) -> str:
        import hashlib

        return hashlib.sha256(candidate.skill.encode("utf-8")).hexdigest()

    def order(self, tied):
        return sorted(tied, key=self.key)


@dataclass
class BeamSelection:
    """Outcome of one Top-M selection."""

    survivors: tuple[CandidateState, ...] = ()
    ranked: tuple[CandidateState, ...] = ()
    m: int = 0
    tie_at_cutoff: bool = False
    tie_group_ids: tuple[str, ...] = ()
    tie_policy: str = ""
    pool_size: int = 0

    @property
    def survivor_ids(self) -> tuple[str, ...]:
        return tuple(c.candidate_id for c in self.survivors)

    @property
    def ranked_ids(self) -> tuple[str, ...]:
        """Exactly the candidates that competed -- the selection input."""
        return tuple(c.candidate_id for c in self.ranked)

    @property
    def survivor_origins(self) -> tuple[str, ...]:
        return tuple(c.origin.value for c in self.survivors)


@dataclass
class BeamManager:
    """Ranks a sealed, fully scored pool and returns the Top-M survivors."""

    m: int = 1
    tie_policy: TiePolicy = TiePolicy.PROVISIONAL_ID_ASC
    strict_ties: bool = False
    #: The tie-break implementation. None means the placeholder named by
    #: ``tie_policy`` (kept for backward compatibility).
    tie_break: TieBreakPolicy | None = None

    def __post_init__(self) -> None:
        if self.m < 1:
            raise ValueError(f"beam width M must be >= 1, got {self.m}")
        if self.tie_break is None:
            self.tie_break = CandidateIdAscendingTieBreak()

    def rank(self, pool: CandidatePool) -> tuple[CandidateState, ...]:
        """One descending ranking of the whole competitive pool.

        The input is ``pool.competitive()`` -- parents plus operator
        candidates. Raw proposals are a different type living in a different
        store and are structurally absent.
        """
        members = pool.competitive()       # refuses an unsealed pool
        unscored = [c.candidate_id for c in members if not c.is_scored]
        if unscored:
            raise BeamSelectionError(
                "refusing to select from a partially scored pool; unscored: "
                f"{unscored}"
            )
        contexts = {c.score.validation_context for c in members}
        if len(contexts) > 1:
            raise BeamSelectionError(
                "refusing to rank scores measured under different evaluation "
                f"conditions: {sorted(contexts)}. Top-M is only meaningful "
                "over one fixed validation set."
            )
        # Descending score; each equal-score group ordered by the tie-break.
        by_score = sorted(members, key=lambda c: -float(c.score.value))
        ranked: list[CandidateState] = []
        i = 0
        while i < len(by_score):
            j = i
            value = float(by_score[i].score.value)
            while j < len(by_score) and float(by_score[j].score.value) == value:
                j += 1
            group = by_score[i:j]
            ordered = list(self.tie_break.order(group)) if len(group) > 1 else group
            if sorted(map(id, ordered)) != sorted(map(id, group)):
                raise BeamSelectionError(
                    f"tie-break {self.tie_break.name!r} did not return a "
                    "permutation of the tied group"
                )
            ranked.extend(ordered)
            i = j
        return tuple(ranked)

    def select(self, pool: CandidatePool) -> BeamSelection:
        ranked = self.rank(pool)
        m = min(self.m, len(ranked))
        survivors = ranked[:m]

        tie_at_cutoff = False
        tie_group: tuple[str, ...] = ()
        if 0 < m < len(ranked):
            cutoff_value = float(ranked[m - 1].score.value)
            if float(ranked[m].score.value) == cutoff_value:
                tie_at_cutoff = True
                tie_group = tuple(
                    c.candidate_id for c in ranked
                    if float(c.score.value) == cutoff_value
                )

        if tie_at_cutoff and self.tie_break.is_placeholder:
            message = (
                f"tie at the Top-{self.m} cutoff (score="
                f"{float(ranked[m - 1].score.value)!r}, tied candidates="
                f"{list(tie_group)}). Survivors were chosen by the PLACEHOLDER "
                f"policy {self.tie_break.name!r}. The scientific tie-break "
                "policy is an open design question."
            )
            if self.strict_ties:
                raise UnresolvedTiePolicy(message)
            warnings.warn(message, UnresolvedTieWarning, stacklevel=2)

        return BeamSelection(
            survivors=survivors,
            ranked=ranked,
            m=self.m,
            tie_at_cutoff=tie_at_cutoff,
            tie_group_ids=tie_group,
            tie_policy=self.tie_break.name,
            pool_size=len(ranked),
        )
