"""Run the search rounds and select the final skill for deployment."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from ours.controller import RoundResult, SearchController
from ours.types import CandidateState


class SearchError(RuntimeError):
    """The search was asked to do something it cannot do."""


@dataclass
class SearchResult:
    """Everything one multi-round run produced."""

    rounds: tuple[RoundResult, ...] = ()
    final_beam: tuple[CandidateState, ...] = ()
    seed_beam: tuple[CandidateState, ...] = ()
    n_rounds: int = 0
    wall_time_s: float = 0.0
    metadata: dict = field(default_factory=dict)
    #: Filled by `ours.experiment.common.finalize_search`: the deployment
    #: rule's decision over Beam_T (a `FinalSelection`). None until then.
    final_selection: object = None

    @property
    def best(self) -> CandidateState | None:
        """The highest-scoring survivor of the last round.

        ``final_beam`` is already in Top-M order -- the round's single global
        ranking -- so this is its head, not a fresh comparison.
        """
        return self.final_beam[0] if self.final_beam else None

    @property
    def best_skill(self) -> str:
        best = self.best
        return best.skill if best is not None else ""

    def carry_forward_report(self) -> list[dict]:
        """Per round: which members were re-measured and which reused a score.

        This is the observable form of "an unchanged incumbent is not
        re-evaluated on the fixed validation set", which is otherwise only
        visible as an absence of spend.
        """
        return [
            {
                "round_id": r.round_id,
                "n_members": r.scoring.n_members,
                "evaluated_ids": list(r.scoring.evaluated_ids),
                "reused_ids": list(r.scoring.reused_ids),
                "n_cache_hits": r.scoring.n_cache_hits,
                "survivor_ids": list(r.selection.survivor_ids),
                "survivor_origins": list(r.selection.survivor_origins),
            }
            for r in self.rounds
        ]


def run_search(
    controller: SearchController,
    seed_parents: Sequence[CandidateState],
    tasks: Sequence[Any],
    *,
    n_rounds: int,
    first_round_id: int = 0,
    on_round=None,
) -> SearchResult:
    """Run ``n_rounds`` OURS rounds, feeding each round's Top-M into the next.

    ``on_round(round_result)`` is an optional observer (logging, artifact
    dumping). It is called after a round completes and may not mutate
    anything; it is deliberately not a hook that can influence selection.
    """
    import time

    if n_rounds < 1:
        raise SearchError(f"n_rounds must be >= 1, got {n_rounds}")
    if not seed_parents:
        raise SearchError("run_search needs at least one seed parent for Beam_0")

    t0 = time.time()
    beam = tuple(seed_parents)
    results: list[RoundResult] = []
    for offset in range(n_rounds):
        round_id = first_round_id + offset
        result = controller.run_round(beam, tasks, round_id=round_id,
                                      final_round=(offset == n_rounds - 1))
        results.append(result)
        if on_round is not None:
            on_round(result)
        # The round already selected. Carrying its output forward verbatim is
        # the whole contract of this loop.
        beam = result.next_beam

    return SearchResult(
        rounds=tuple(results),
        final_beam=beam,
        seed_beam=tuple(seed_parents),
        n_rounds=n_rounds,
        wall_time_s=round(time.time() - t0, 3),
        metadata={
            "adapter": controller.adapter.name,
            "partitioner": controller.partitioner.name,
            "partitioner_is_placeholder": controller.partitioner.is_placeholder,
            "operator": controller.operator.name,
            "operator_is_placeholder": controller.operator.is_placeholder,
            "evaluator": controller.evaluator.name,
            "validation_context": controller.evaluator.validation_context_id,
            "beam_m": controller.beam_manager.m,
            "k_subsets": controller.k_subsets,
        },
    )
