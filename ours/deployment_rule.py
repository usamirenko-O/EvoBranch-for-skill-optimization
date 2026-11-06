"""Select a final skill from the surviving beam and an optional merged candidate."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence

from ours.registry import register
from ours.types import CandidateState


class DeploymentRuleError(RuntimeError):
    """The rule cannot map this Beam_T to a deployment."""


@dataclass(frozen=True)
class MemberDeployment:
    """The official held-out result of one deployed Beam_T member."""

    candidate_id: str
    metric_name: str
    metric_value: float | None
    n_tasks: int
    validation_score: float | None = None
    secondary_metrics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class AggregateResult:
    """The one reported deployment metric, plus how it was obtained."""

    metric_value: float | None
    n_tasks: int
    method: str
    details: dict = field(default_factory=dict)


@dataclass
class FinalSelection:
    """What the rule decided BEFORE any deployment task ran.

    ``considered`` are the candidates that competed for deployment (Beam_T,
    plus whatever the rule added and scored on the same fixed VAL);
    ``selected`` the ones to deploy. Everything here was decided on VAL only.
    """

    considered: tuple = ()
    selected: tuple = ()
    added: tuple = ()
    report: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        def _c(c):
            return {"candidate_id": c.candidate_id, "origin": c.origin.value,
                    "skill_hash": c.skill_hash,
                    "validation_score": float(c.score.value) if c.score else None}
        return {
            "considered": [_c(c) for c in self.considered],
            "added": [_c(c) for c in self.added],
            "selected": [_c(c) for c in self.selected],
            "report": dict(self.report),
        }


class DeploymentRule(ABC):
    name: str = "abstract"
    is_placeholder: bool = False

    def check(self, beam_m: int) -> list[str]:
        """Blocking problems for a run with beam width ``beam_m``."""
        return []

    def finalize(self, final_beam: Sequence[CandidateState], *, controller, round_id: int,
                 allocate_dir) -> FinalSelection:
        """Decide what to deploy, INSIDE the optimization phase (before any
        deployment task runs; anything scored here is scored on the fixed
        VAL through ``controller`` and charged to optimization).

        Default: no extra work, ``select`` on Beam_T as given."""
        return FinalSelection(considered=tuple(final_beam), selected=tuple(self.select(final_beam)))

    @abstractmethod
    def select(self, final_beam: Sequence[CandidateState]) -> tuple[CandidateState, ...]:
        """The Beam_T members to deploy (non-empty)."""

    def aggregate(self, results: Sequence[MemberDeployment]) -> AggregateResult:
        """Default: exactly one member was deployed, and its result is THE result."""
        if len(results) != 1:
            raise DeploymentRuleError(
                f"rule {self.name!r} deployed {len(results)} members but does not "
                "define how to aggregate them; override aggregate()"
            )
        only = results[0]
        return AggregateResult(
            metric_value=only.metric_value, n_tasks=only.n_tasks,
            method="single_member", details={"candidate_id": only.candidate_id},
        )


@register("deployment_rule", "top_validation_score_placeholder")
class TopValidationScoreRule(DeploymentRule):
    """PLACEHOLDER. Deploy the head of Beam_T -- the best fixed-validation score,
    ties resolved by the run's tie-break. Not a decided rule."""

    name = "top_validation_score_placeholder"
    is_placeholder = True

    def select(self, final_beam):
        if not final_beam:
            raise DeploymentRuleError("Beam_T is empty")
        return (final_beam[0],)


@register("deployment_rule", "beam_top1_no_merge")
class BeamTop1NoMergeRule(DeploymentRule):
    """Deploy the highest-ranked beam member without a final merge."""

    name = "beam_top1_no_merge"
    is_placeholder = False

    def finalize(self, final_beam, *, controller, round_id, allocate_dir):
        survivors = tuple(final_beam)
        if not survivors:
            raise DeploymentRuleError("Beam_T is empty")
        head = survivors[0]
        return FinalSelection(
            considered=survivors, selected=(head,), added=(),
            report={
                "rule": self.name, "final_merge": False, "merged_candidate_id": None,
                "ranked_ids": [c.candidate_id for c in survivors],
                "scores": {c.candidate_id: float(c.score.value) for c in survivors if c.score},
                "tie_policy": controller.beam_manager.tie_break.name,
                "selected_is_merged": False,
            },
        )

    def select(self, final_beam):
        if not final_beam:
            raise DeploymentRuleError("Beam_T is empty")
        return (final_beam[0],)


@register("deployment_rule", "require_single_member")
class RequireSingleMemberRule(DeploymentRule):
    """Deploy Beam_T only when it has exactly one member; refuse M > 1.

    Makes no choice at all, which is what makes it useful as a guard while no
    multi-member rule has been decided.
    """

    name = "require_single_member"
    is_placeholder = False

    def check(self, beam_m):
        if beam_m > 1:
            return [
                "deployment_rule=require_single_member refuses beam_m > 1: Beam_T "
                "may hold several survivors and this rule decides none of them"
            ]
        return []

    def select(self, final_beam):
        if len(final_beam) != 1:
            raise DeploymentRuleError(
                f"require_single_member: Beam_T has {len(final_beam)} members"
            )
        return (final_beam[0],)


@register("deployment_rule", "final_merge_top1")
class FinalMergeTop1Rule(DeploymentRule):
    """Generate one merged skill and select Top-1 from it and the surviving beam using VAL."""

    name = "final_merge_top1"
    is_placeholder = False

    def check(self, beam_m):
        return []

    def finalize(self, final_beam, *, controller, round_id, allocate_dir):
        from ours.beam import BeamManager
        from ours.pool import CandidatePool, RawProposalStore
        from ours.types import Origin

        survivors = tuple(final_beam)
        if not survivors:
            raise DeploymentRuleError("Beam_T is empty")
        operator = controller.operator
        if not hasattr(operator, "final_merge"):
            raise DeploymentRuleError(
                f"deployment rule {self.name!r} needs an operator with final_merge(); "
                f"{operator.name!r} has none"
            )
        # ① ONE merged candidate from ALL survivors, via the frozen operator.
        merged_proposal = operator.final_merge(survivors, round_id=round_id, allocate_dir=allocate_dir)
        merge_report = dict(getattr(operator, "last_report", None) or {})
        pool = CandidatePool(round_id=round_id, proposal_material=RawProposalStore(round_id=round_id))
        pool.open_operator_stage()
        merged = None
        if merged_proposal is not None:
            anchor = survivors[0]


            merged_skill, native_meta = controller.adapter.adopt_native_state(
                merged_proposal.skill, anchor)
            merged = CandidateState(
                candidate_id=f"r{round_id:03d}-op-000",
                optimizer_name=anchor.optimizer_name, round_id=round_id,
                skill=merged_skill, origin=Origin.OPERATOR,
                metadata=dict(native_meta),
                parent_id=anchor.candidate_id,
                operator_provenance={
                    "operator": operator.name, "is_placeholder": False,
                    "final_merge": True,
                    "source_proposal_ids": list(merged_proposal.source_proposal_ids),
                    "source_parent_ids": list(merged_proposal.source_parent_ids),
                    **dict(merged_proposal.provenance),
                },
                workspace_dir=allocate_dir("final_merge_candidate"),
                lineage=anchor.lineage_including_self,
            )
            pool.add_operator_candidate(merged)
        for c in survivors:
            pool.add_parent(c.carried_forward(round_id))
        pool.seal()
        # ② the merged candidate is scored on the SAME fixed VAL (survivors
        #    reuse their persistent scores); charged to optimization.
        scoring = controller.scorer.score_pool(
            pool, allocate_dir=lambda c: allocate_dir(f"final_eval/{c.candidate_id}"),
            round_id=round_id,
        )
        # ③ ONE Top-1 by VAL score, then the run's tie-break.
        selection = BeamManager(m=1, strict_ties=controller.beam_manager.strict_ties,
                                tie_break=controller.beam_manager.tie_break).select(pool)
        winner = selection.survivors[0]
        return FinalSelection(
            considered=tuple(pool.competitive()),
            selected=(winner,),
            added=(merged,) if merged is not None else (),
            report={
                "rule": self.name,
                "merged_candidate_id": merged.candidate_id if merged is not None else None,
                "merged_from": [c.candidate_id for c in survivors],
                "merge": merge_report,
                "scoring": {"n_evaluated": scoring.n_evaluated, "n_reused": scoring.n_reused,
                            "n_cache_hits": scoring.n_cache_hits,
                            "validation_context": scoring.validation_context},
                "ranked_ids": list(selection.ranked_ids),
                "scores": {c.candidate_id: float(c.score.value) for c in selection.ranked},
                "tie_at_cutoff": selection.tie_at_cutoff,
                "tie_policy": selection.tie_policy,
                "selected_is_merged": merged is not None and winner.candidate_id == merged.candidate_id,
            },
        )

    def select(self, final_beam):
        if len(final_beam) != 1:
            raise DeploymentRuleError(
                f"{self.name}: deploy exactly the finalized Top-1; got {len(final_beam)} "
                "candidates (finalize() must run first)"
            )
        return (final_beam[0],)


__all__ = [
    "AggregateResult", "BeamTop1NoMergeRule", "DeploymentRule", "DeploymentRuleError", "FinalMergeTop1Rule",
    "FinalSelection", "MemberDeployment", "RequireSingleMemberRule", "TopValidationScoreRule",
]
