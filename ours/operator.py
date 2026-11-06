"""Interfaces for turning raw proposals into competitive candidates."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

from ours.registry import register
from ours.resources import RunBound
from ours.types import CandidateState, CostAccounting, Evidence, RawProposal


@dataclass
class OperatorProposal:
    """One new competitive candidate synthesised by the operator.

    Provenance is split by role on purpose: ``source_proposal_ids`` are the
    raw proposals this was built from (operator material), while
    ``source_parent_ids`` are the incumbents it drew on. Blurring the two is
    exactly the confusion this design avoids.
    """

    skill: str
    provenance: dict[str, Any] = field(default_factory=dict)
    source_proposal_ids: tuple[str, ...] = ()
    source_parent_ids: tuple[str, ...] = ()
    cost: CostAccounting = field(default_factory=CostAccounting)


class SkillOperator(RunBound, ABC):
    """Builds the round's new challengers out of the raw proposal set."""

    name: str = "abstract"
    is_placeholder: bool = False

    @abstractmethod
    def expand(
        self,
        *,
        parents: Sequence[CandidateState],
        proposals: Sequence[RawProposal],
        evidence: Sequence[Evidence],
        round_id: int,
        allocate_dir,
    ) -> list[OperatorProposal]:
        """Return zero or more new candidates.

        ``parents`` is ALL of ``Beam_t``, ``proposals`` is ALL raw proposals
        from all parents, ``evidence`` is ALL evidence from all parents.
        ``allocate_dir(tag) -> str`` hands back a fresh, isolated directory if
        the operator needs to do work on disk.
        """


@register("operator", "noop_placeholder")
class NoOpOperator(SkillOperator):
    """PLACEHOLDER. Builds nothing.

    With this operator the competitive pool is exactly ``Beam_t`` -- the round
    produces raw proposals, hands them to the operator, and the operator
    declines to make anything of them, so no challenger exists and the
    incumbents trivially survive.

    That is the honest default until the real operator is designed: it keeps
    the stage in the control flow, and in the trace, without pretending to do
    science. It also makes the corrected semantics visible, since under the
    old (incorrect) reading the raw proposals would have competed here.
    """

    name = "noop_placeholder"
    is_placeholder = True

    def expand(self, *, parents, proposals, evidence, round_id, allocate_dir):
        return []


@register("operator", "pass_through_placeholder")
class PassThroughOperator(SkillOperator):
    """PLACEHOLDER. Promotes each raw proposal, unchanged, to one candidate."""

    name = "pass_through_placeholder"
    is_placeholder = True

    def __init__(self, *, max_candidates: int | None = None) -> None:
        #: Purely a cost guard for smoke runs. The real budget policy -- how
        #: many candidates an operator may emit -- is undecided.
        self.max_candidates = max_candidates
        self.seen_parent_ids: list[tuple[str, ...]] = []
        self.seen_proposal_ids: list[tuple[str, ...]] = []
        self.seen_evidence_ids: list[tuple[str, ...]] = []

    def expand(self, *, parents, proposals, evidence, round_id, allocate_dir):
        self.seen_parent_ids.append(tuple(p.candidate_id for p in parents))
        self.seen_proposal_ids.append(tuple(q.proposal_id for q in proposals))
        self.seen_evidence_ids.append(tuple(e.evidence_id for e in evidence))

        selected = proposals
        if self.max_candidates is not None:
            selected = proposals[: max(0, int(self.max_candidates))]

        out: list[OperatorProposal] = []
        for index, proposal in enumerate(selected):
            allocate_dir(f"passthrough_{index:02d}")
            out.append(OperatorProposal(
                skill=proposal.skill,
                provenance={
                    "policy": "pass_through_unchanged",
                    "is_placeholder": True,
                    "source_evidence_subset_id": proposal.evidence_subset_id,
                    "n_parents_seen": len(parents),
                    "n_proposals_seen": len(proposals),
                    "n_evidence_seen": len(evidence),
                    **dict(proposal.provenance),
                },
                source_proposal_ids=(proposal.proposal_id,),
                source_parent_ids=(proposal.parent_id,),
            ))
        return out
