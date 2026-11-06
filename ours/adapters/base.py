"""Interfaces for optimizer evidence, proposal generation, and candidate evaluation."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from ours.types import (
    CandidateState,
    CostAccounting,
    Evidence,
    EvidenceSubset,
    Score,
)


class ProposalResult:
    """What one native proposal call produced: RAW OPERATOR MATERIAL.

    ``skill`` is ``None`` when the optimizer's native mechanism declined to
    produce anything for this evidence subset (SkillOpt's
    ``skip_no_patches`` / ``skip_no_rewrite``). That is not an error and not a
    rejection -- there is simply no material from this subset.

    A produced ``skill`` becomes a :class:`~ours.types.RawProposal`, which the
    operator reads and synthesises new candidates from. It does **not** become
    a competitive candidate, is never scored on the validation set, and can
    never survive into the next beam on its own.
    """

    def __init__(
        self,
        skill: str | None,
        *,
        status: str = "ok",
        provenance: dict[str, Any] | None = None,
        cost: CostAccounting | None = None,
        workspace_dir: str = "",
    ) -> None:
        self.skill = skill
        self.status = status
        self.provenance = provenance or {}
        self.cost = cost or CostAccounting()
        self.workspace_dir = workspace_dir

    @property
    def produced_proposal(self) -> bool:
        """The native mechanism returned raw material for the operator."""
        return bool(self.skill)


class OptimizerAdapter(ABC):
    """Optimizer-specific native behaviour, and nothing else."""

    name: str = "abstract"

    # ── round lifecycle ──────────────────────────────────────────────────

    def begin_round(self, round_id: int) -> None:
        """Hook for per-round native state that must be resolved ONCE.

        SkillOpt uses it to consume exactly one LR-scheduler tick for the
        whole round instead of one per candidate. Default is a no-op.
        """

    def end_round(self, round_id: int) -> None:
        """Symmetric teardown hook. Default is a no-op."""


    def adopt_native_state(self, skill: str, anchor: CandidateState) -> tuple[str, dict]:
        """A NEW competitive candidate built from ``anchor``'s branch takes over
        that branch's native state: returns ``(skill, metadata_update)``.
        Default: the skill unchanged and no metadata (no native state)."""
        return skill, {}

    def finish_round(self, survivors: Sequence[CandidateState], *, parents_by_id: dict,
                     round_id: int, out_dir: str) -> dict | None:
        """After the global Top-M of a round that is not the last: the native
        optimizer's epoch-level update of the survivors (SkillOpt: slow update +
        meta skill per NEW survivor). May mutate the survivors' skill / metadata
        in place; returns a report for the round trace, or None when the
        optimizer has no such mechanism (default)."""
        return None

    # ── the three interfaces the flow is built from ──────────────────────

    @abstractmethod
    def collect_evidence(
        self,
        parent: CandidateState,
        tasks: Sequence[Any],
        *,
        out_dir: str,
        seed: int | None = None,
    ) -> Evidence:
        """Run the optimizer's native evidence collection for one parent."""

    @abstractmethod
    def propose(
        self,
        parent: CandidateState,
        subset: EvidenceSubset,
        *,
        out_dir: str,
        round_id: int,
        history_context: str = "",
    ) -> ProposalResult:
        """Generate one raw proposal from a parent, an evidence subset, and optional history.
        
        Candidate scoring and selection belong to the outer search."""

    @abstractmethod
    def slice_evidence(
        self,
        evidence: Evidence,
        subset_id: str,
        task_ids: Sequence[str],
        *,
        partitioner: str = "",
    ) -> EvidenceSubset:
        """Materialise the subset payload for a set of task ids.

        The partitioner chooses *which* task ids belong together using the
        optimizer-agnostic view from :meth:`evidence_features`; only the
        adapter knows how to build the payload those ids correspond to.
        """

    def evidence_features(self, evidence: Evidence) -> dict[str, dict[str, Any]]:
        """Per-task feature view, keyed by task id, for the partitioner.

        Deliberately generic (outcome, error signal, task type, ...) so a
        future clustering method can be written once against every optimizer.
        Every real adapter also provides ``native_text``: a faithful textual
        serialization of that task's native evidence record (the rollout
        result, the anchor execution, the case analysis), so a
        representation-based partitioner (text -> embedding -> clustering)
        has its raw material without importing the optimizer. Which fields or
        representation a partitioner should use is the partitioner's own
        (open) choice, not fixed here.
        Default: no features, which the trivial partitioners do not need.
        """
        return {task_id: {} for task_id in evidence.task_ids}


class EvaluatorAdapter(ABC):
    """Scores any skill on ONE fixed validation set, on a comparable scale.

    Contract, and the reason this is separate from ``OptimizerAdapter``:
    it returns a *number*, never a decision. It must not invoke the
    optimizer's native accept/reject gate -- in OURS that gate is not the
    search policy, the :class:`~ours.beam.BeamManager` is.

    The validation set is fixed for the whole optimization run. Because it is
    fixed, a score measured in round ``t`` is still the right number in round
    ``t+k``, which is what lets OURS carry an incumbent's score forward
    instead of re-measuring an unchanged skill every round.
    """

    name: str = "abstract"

    @property
    def validation_context_id(self) -> str:
        """Identity of the fixed evaluation condition this adapter measures.

        A stored score may be reused only while this string is unchanged. It
        must therefore fold in everything that would make two numbers
        incomparable: the evaluator, the fixed validation split and its size,
        the seed, the metric and the solver condition.

        The default is the adapter name, which is correct for an adapter whose
        condition cannot vary; an adapter with knobs must override it.
        """
        return self.name

    @abstractmethod
    def score(self, candidate: CandidateState, *, out_dir: str) -> Score:
        """Return this candidate's score on the fixed validation set.

        Callers stamp the returned :class:`~ours.types.Score` with
        :attr:`validation_context_id`, so an implementation does not have to
        remember to.
        """
