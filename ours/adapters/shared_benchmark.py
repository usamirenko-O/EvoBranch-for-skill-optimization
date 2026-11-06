"""Evaluate candidate skills on the shared fixed validation split."""
from __future__ import annotations

from typing import Sequence

# `run_tasks` is the shared execution loop the deployment path uses;
# `_score_input` is its rule for "a task the harness could not complete is
# still scored, as a failure, rather than dropped". Both are imported rather
# than re-implemented so validation and deployment cannot drift apart -- a
# second copy of that rule is exactly how a denominator silently shrinks.
from integration.experiment.deployment import _score_input, run_tasks
from integration.harness.registry import HARNESS_KIND, build_harness, max_parallel_tasks
from ours.adapters.base import EvaluatorAdapter
from ours.types import CandidateState, Score
from ours.workspace import assert_fresh_dir


class ValidationSplitError(RuntimeError):
    """The evaluator was pointed at tasks that are not the validation split."""


class SharedBenchmarkEvaluator(EvaluatorAdapter):
    """Official-scorer accuracy on the run's fixed validation split."""

    name = "shared_direct_validation"

    def __init__(
        self,
        *,
        adapter,
        manifest,
        solver_config,
        accumulator,
        harness_options: dict | None = None,
        max_workers: int = 1,
        role: str = "ours_validation",
    ) -> None:
        self.adapter = adapter
        self.manifest = manifest
        self.solver_config = solver_config
        self.accumulator = accumulator
        self.harness_options = dict(harness_options or {})
        self.max_workers = max(1, min(int(max_workers), max_parallel_tasks(adapter.name)))
        self.role = role

        self.task_ids: tuple[str, ...] = tuple(manifest.task_ids("val"))
        if not self.task_ids:
            raise ValidationSplitError(
                f"manifest {manifest.manifest_id!r} has an empty validation "
                "split; OURS has nothing to select on"
            )
        deployment = set(manifest.task_ids("deployment"))
        leaked = [t for t in self.task_ids if t in deployment]
        if leaked:
            raise ValidationSplitError(
                f"validation ids overlap the deployment split: {leaked[:5]}"
            )
        self.n_scored = 0
        self.scored_candidate_ids: list[str] = []

    @property
    def validation_context_id(self) -> str:
        """Everything that would make two of these numbers incomparable.

        Fixed for the whole run, which is what makes a parent's score reusable
        in later rounds. If any of it changed mid-run, stored scores would
        stop matching and be re-measured rather than silently compared.
        """
        return ":".join(str(x) for x in (
            self.name,
            self.adapter.name,
            self.manifest.manifest_id,
            self.manifest.content_hash[:16],
            len(self.task_ids),
            HARNESS_KIND,
            self.adapter.scorer_ref,
            self.adapter.metric_name,
            self.solver_config.model,
            self.solver_config.temperature,
            self.solver_config.max_tokens,
        ))

    def score(self, candidate: CandidateState, *, out_dir: str) -> Score:
        """Run this candidate's skill over the fixed validation ids and score
        it with the benchmark's own scorer."""
        from integration.harness.base import SolverSession

        work_dir = assert_fresh_dir(
            f"{out_dir}/harness", f"validation harness for {candidate.candidate_id}",
        )
        session = SolverSession(
            config=self.solver_config, accumulator=self.accumulator, role=self.role,
        )
        harness = build_harness(
            self.adapter.name, work_dir=work_dir, options=self.harness_options,
        )
        try:
            runs = run_tasks(
                adapter=self.adapter,
                harness=harness,
                task_ids=list(self.task_ids),
                skill_text=candidate.skill,
                session=session,
                role="val",
                max_workers=self.max_workers,
            )
        finally:
            harness.close()

        outputs = {r.task_id: _score_input(r, self.adapter.name) for r in runs}
        result = self.adapter.evaluate(outputs, role="val")
        self.n_scored += 1
        self.scored_candidate_ids.append(candidate.candidate_id)

        value = result.metric_value
        return Score(
            # A split on which nothing was scorable has no metric. 0.0 keeps
            # the pool rankable and `metric_value_is_none` says it happened,
            # rather than a None silently becoming a winner or a crash.
            value=float(value) if value is not None else 0.0,
            components={
                "metric_name": result.metric_name,
                "metric_value": value,
                "metric_value_is_none": value is None,
                "scorer_ref": result.scorer_ref,
                "secondary_metrics": dict(result.secondary_metrics),
                "n_errors": sum(1 for r in runs if r.error),
            },
            evaluator=self.name,
            n_items=result.n_tasks,
        )
