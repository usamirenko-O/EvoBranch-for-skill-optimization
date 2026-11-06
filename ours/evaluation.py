"""Score competitive candidates on a fixed validation set and reuse cached scores."""
from __future__ import annotations

from dataclasses import dataclass, replace

from ours.adapters.base import EvaluatorAdapter
from ours.pool import CandidatePool
from ours.types import Score


@dataclass
class ScoringReport:
    """Audit of one scoring pass over the competitive pool.

    ``n_reused`` is the interesting number: it is how many candidates already
    held a valid fixed-validation score and therefore cost nothing this round.
    """

    n_members: int = 0
    n_evaluated: int = 0
    n_cache_hits: int = 0
    n_reused: int = 0
    n_invalidated: int = 0
    scored_ids: tuple[str, ...] = ()
    evaluated_ids: tuple[str, ...] = ()
    reused_ids: tuple[str, ...] = ()
    evaluator: str = ""
    validation_context: str = ""

    @property
    def every_member_scored_once(self) -> bool:
        """Every competitive member ended the pass holding exactly one score,
        obtained in exactly one of the three legitimate ways."""
        return (
            len(self.scored_ids) == self.n_members
            and len(set(self.scored_ids)) == self.n_members
            and self.n_evaluated + self.n_cache_hits + self.n_reused
            == self.n_members
        )

    @property
    def no_candidate_evaluated_twice(self) -> bool:
        return len(set(self.evaluated_ids)) == len(self.evaluated_ids)


class PoolScorer:
    """Gives every competitive candidate a comparable fixed-validation score.

    The scorer is owned by the run, not by the round, so its duplicate-skill
    cache spans rounds exactly as the persistent per-candidate scores do.
    """

    def __init__(self, evaluator: EvaluatorAdapter) -> None:
        self.evaluator = evaluator
        self._by_hash: dict[tuple[str, str], Score] = {}

    @property
    def validation_context(self) -> str:
        """The current fixed evaluation condition (validation set + evaluator
        + solver condition), as declared by the evaluator adapter."""
        return self.evaluator.validation_context_id

    def score_pool(
        self,
        pool: CandidatePool,
        *,
        allocate_dir,
        round_id: int | None = None,
    ) -> ScoringReport:
        members = pool.competitive()
        context = self.validation_context
        report = ScoringReport(
            n_members=len(members),
            evaluator=self.evaluator.name,
            validation_context=context,
        )
        scored_ids: list[str] = []
        evaluated_ids: list[str] = []
        reused_ids: list[str] = []

        for candidate in members:
            # ① the candidate already holds a score for THIS condition ────
            if candidate.has_valid_score(context):
                report.n_reused += 1
                reused_ids.append(candidate.candidate_id)
                scored_ids.append(candidate.candidate_id)
                continue

            # a score from a different condition is not comparable: drop it
            if candidate.is_scored:
                candidate.clear_score(
                    f"validation context changed to {context!r}"
                )
                report.n_invalidated += 1

            key = (context, candidate.skill_hash)
            cached = self._by_hash.get(key)
            if cached is not None:
                # ② identical skill text, already measured under this
                #    condition -- share the number, do not pay twice.
                candidate.attach_score(replace(
                    cached,
                    components=dict(cached.components),
                    cache_hit=True,
                    measured_in_round=round_id,
                ))
                report.n_cache_hits += 1
            else:
                # ③ genuinely new: measure it on the fixed validation set.
                score = self.evaluator.score(
                    candidate, out_dir=allocate_dir(candidate),
                )
                score = replace(
                    score,
                    validation_context=context,
                    measured_in_round=round_id,
                )
                candidate.attach_score(score)
                self._by_hash[key] = score
                report.n_evaluated += 1
                evaluated_ids.append(candidate.candidate_id)
            scored_ids.append(candidate.candidate_id)

        report.scored_ids = tuple(scored_ids)
        report.evaluated_ids = tuple(evaluated_ids)
        report.reused_ids = tuple(reused_ids)
        return report
