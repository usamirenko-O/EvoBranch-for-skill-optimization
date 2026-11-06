"""Keep raw proposals separate from the competitive parent and operator candidates."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ours.types import CandidateState, Origin, RawProposal


class PoolStageError(RuntimeError):
    """An operation was attempted in the wrong pool stage."""


class PoolIntegrityError(RuntimeError):
    """A stored object is missing, or the wrong kind of object got in."""


class PoolStage(str, Enum):
    PROPOSING = "proposing"
    OPERATOR = "operator"
    COMPLETE = "complete"


@dataclass
class RawProposalStore:
    """Append-only store of this round's raw native proposals.

    Frozen when the operator stage opens, so "the operator saw ALL of them" is
    enforced: nothing can be added behind the operator's back, and the ids
    present at freeze time are re-checked when the pool seals.
    """

    round_id: int = 0
    _proposals: list[RawProposal] = field(default_factory=list, repr=False)
    _frozen: bool = False
    _ids_at_freeze: tuple[str, ...] = ()

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def add(self, proposal: RawProposal) -> RawProposal:
        if self._frozen:
            raise PoolStageError(
                "cannot add a raw proposal after the operator stage opened; "
                "every proposal must exist before the operator runs so the "
                "operator sees all of them"
            )
        if not isinstance(proposal, RawProposal):
            raise PoolIntegrityError(
                f"the raw proposal store holds RawProposal only, got "
                f"{type(proposal).__name__}"
            )
        self._proposals.append(proposal)
        return proposal

    def freeze(self) -> tuple[RawProposal, ...]:
        """Close the store and return everything in it, for the operator."""
        self._frozen = True
        self._ids_at_freeze = tuple(p.proposal_id for p in self._proposals)
        return self.all()

    def all(self) -> tuple[RawProposal, ...]:
        return tuple(self._proposals)

    def ids(self) -> tuple[str, ...]:
        return tuple(p.proposal_id for p in self._proposals)

    def assert_intact(self) -> None:
        present = {p.proposal_id for p in self._proposals}
        missing = [pid for pid in self._ids_at_freeze if pid not in present]
        if missing:
            raise PoolIntegrityError(
                f"raw proposals disappeared between the operator stage and "
                f"sealing: {missing}"
            )

    def __len__(self) -> int:
        return len(self._proposals)


@dataclass
class CandidatePool:
    """The competitive candidate pool: ``C_t = Beam_t ∪ O_t``.

    Members are validated parents and operator outputs. Raw proposals are not
    members and have no way to become members; ``proposal_material`` is only a
    back-reference so the pool can prove, at seal time, that nothing leaked
    across the boundary.
    """

    round_id: int
    proposal_material: RawProposalStore = field(default_factory=RawProposalStore)
    _parents: list[CandidateState] = field(default_factory=list, repr=False)
    _operator: list[CandidateState] = field(default_factory=list, repr=False)
    _stage: PoolStage = PoolStage.PROPOSING

    # ── stage machine ────────────────────────────────────────────────────

    @property
    def stage(self) -> PoolStage:
        return self._stage

    def open_operator_stage(self) -> tuple[RawProposal, ...]:
        """Freeze the raw proposals and hand them all to the caller."""
        if self._stage is not PoolStage.PROPOSING:
            raise PoolStageError(
                f"cannot open the operator stage from {self._stage.value}"
            )
        self._stage = PoolStage.OPERATOR
        return self.proposal_material.freeze()

    def seal(self) -> None:
        """Freeze the competitive pool. After this it is ready to be scored."""
        if self._stage is not PoolStage.OPERATOR:
            raise PoolStageError(
                f"cannot seal the pool from {self._stage.value}; the operator "
                "stage must run before the competitive pool is complete"
            )
        self.proposal_material.assert_intact()
        if not self._parents:
            raise PoolIntegrityError(
                "the competitive pool must contain the current/parent "
                "skill(s); none was added"
            )
        self._assert_no_raw_proposal_leaked()
        self._stage = PoolStage.COMPLETE

    def _assert_no_raw_proposal_leaked(self) -> None:
        proposal_ids = set(self.proposal_material.ids())
        leaked = [
            c.candidate_id for c in self.members()
            if c.candidate_id in proposal_ids
        ]
        if leaked:
            raise PoolIntegrityError(
                f"raw proposals entered the competitive pool: {leaked}. Raw "
                "proposals are operator material; only parents and operator "
                "outputs compete."
            )

    # ── membership ───────────────────────────────────────────────────────

    def add_parent(self, candidate: CandidateState) -> CandidateState:
        # Allowed in PROPOSING or OPERATOR: the authoritative flow assembles
        # the competitive pool after the operator stage, so the wording of the
        # flow and the code match exactly.
        if self._stage is PoolStage.COMPLETE:
            raise PoolStageError("cannot add a parent to a sealed pool")
        self._check_member(candidate, Origin.PARENT)
        self._parents.append(candidate)
        return candidate

    def add_operator_candidate(self, candidate: CandidateState) -> CandidateState:
        self._require(PoolStage.OPERATOR, "add an operator candidate")
        self._check_member(candidate, Origin.OPERATOR)
        self._operator.append(candidate)
        return candidate

    @staticmethod
    def _check_member(candidate, expected: Origin) -> None:
        if not isinstance(candidate, CandidateState):
            raise PoolIntegrityError(
                f"the competitive pool holds CandidateState only, got "
                f"{type(candidate).__name__}; raw proposals are not "
                "competitive candidates"
            )
        if candidate.origin is not expected:
            raise PoolStageError(
                f"{expected.value} member {candidate.candidate_id} has origin "
                f"{candidate.origin.value}"
            )

    # ── views ────────────────────────────────────────────────────────────

    def parents(self) -> tuple[CandidateState, ...]:
        return tuple(self._parents)

    def operator_candidates(self) -> tuple[CandidateState, ...]:
        return tuple(self._operator)

    def raw_proposals(self) -> tuple[RawProposal, ...]:
        """The round's operator material. NOT pool members -- for inspection,
        tracing and reporting only."""
        return self.proposal_material.all()

    def members(self) -> tuple[CandidateState, ...]:
        """Every competitive candidate currently in the pool, at any stage."""
        return tuple(self._parents) + tuple(self._operator)

    def competitive(self) -> tuple[CandidateState, ...]:
        """``C_t = Beam_t ∪ O_t`` -- parents plus operator outputs, sealed.

        Refuses to answer before the pool is sealed, so nothing downstream can
        accidentally score or select a partial pool.
        """
        if self._stage is not PoolStage.COMPLETE:
            raise PoolStageError(
                f"the competitive pool is not available in stage "
                f"{self._stage.value}; seal the pool first"
            )
        return self.members()

    @property
    def all_scored(self) -> bool:
        return all(c.is_scored for c in self.members())

    def unscored(self) -> tuple[CandidateState, ...]:
        return tuple(c for c in self.members() if not c.is_scored)

    def needing_evaluation(self, validation_context: str) -> tuple[CandidateState, ...]:
        """Competitive members with no reusable score for this condition."""
        return tuple(
            c for c in self.members() if not c.has_valid_score(validation_context)
        )

    def _require(self, stage: PoolStage, what: str) -> None:
        if self._stage is not stage:
            raise PoolStageError(
                f"cannot {what} in stage {self._stage.value} "
                f"(requires {stage.value})"
            )
