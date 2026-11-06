"""Benchmark-independent SSO step and explicit-schedule loop (Algorithm 1).

PAPER_SPECIFIED control flow is annotated with Algorithm 1 line numbers.
dependency injection, preselected history ids, result
records, sequential ordering, and boundary validation. No selection policies."""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

from sso.core._execution import compare, execute_as, task_ids
from sso.core.evidence import compute_behavior_units
from sso.core.types import (
    BehaviorCluster, BehaviorRecord, BehaviorUnit, Execution, ExecutionRef,
    PairwiseDecision, Probe, Skill, Task,
)
from sso.core.validation import ValidationResult, validate_candidate
from sso.executors import Executor
from sso.llm import (
    BehaviorClusterer, BehaviorExtractor, PairwiseJudge, ProbeGenerator, Renderer,
    assemble_clusters,
)


@dataclass(frozen=True)
class StepInputs:
    """explicit X_t, history membership, and V_t.
    
    history_task_ids chooses only membership/order, never cached executions.
    H_t is materialized from this step's anchors (eq. 3). Input sequences are
    snapshotted to tuples. No random/default selection, including for H_t.
    Task ids identify dataset instances and must be consistent across steps."""

    training_batch: Sequence[Task]
    history_task_ids: Sequence[str]
    validation_batch: Sequence[Task]

    def __post_init__(self) -> None:
        for name in ("training_batch", "history_task_ids", "validation_batch"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        training = task_ids(self.training_batch)
        validation = task_ids(self.validation_batch)
        history = self.history_task_ids
        if any(not isinstance(key, str) for key in history):
            raise ValueError("history ids must be strings")
        if len(set(history)) != len(history) or not set(history) <= training:
            raise ValueError("history ids must be a unique subset of training task ids")
        # PAPER_SPECIFIED: D_tr and D_val are disjoint (Problem Setup).
        # REPRODUCTION_CHOICE: enforce observable membership via task ids.
        if training & validation:
            raise ValueError("training and validation task ids must be disjoint")


@dataclass(frozen=True)
class StepResult:
    """in-memory inspection trace, scoped to one step.
    
    candidate_skill/validation are None iff U_t was empty. Existing mutable
    records/maps are retained without deep-copying; this is not an immutable
    audit log. Validation comparisons remain separate from training decisions."""

    selected_skill: Skill
    anchor_executions: Dict[str, Execution]
    probes: Tuple[Probe, ...]
    probe_executions: Dict[ExecutionRef, Execution]
    decisions: Tuple[PairwiseDecision, ...]
    behavior_records: Tuple[BehaviorRecord, ...]
    clusters: Tuple[BehaviorCluster, ...]
    behavior_units: Tuple[BehaviorUnit, ...]
    candidate_skill: Optional[Skill]
    validation: Optional[ValidationResult]


@dataclass(frozen=True)
class OptimizationResult:
    """final S_T plus per-step inspection results."""

    selected_skill: Skill
    steps: Tuple[StepResult, ...]


@dataclass
class SSOOptimizer:
    """Run SSO with an executor, LLM components, probe count, and update budget."""

    executor: Executor
    generator: ProbeGenerator
    judge: PairwiseJudge
    extractor: BehaviorExtractor
    clusterer: BehaviorClusterer
    renderer: Renderer
    num_probes: int
    update_budget: int

    def __post_init__(self) -> None:
        if type(self.num_probes) is not int or self.num_probes < 1:
            raise ValueError("num_probes must be a positive integer")
        if type(self.update_budget) is not int or self.update_budget < 0:
            raise ValueError("update_budget must be a nonnegative integer")

    def step(self, current_skill: Skill, inputs: StepInputs) -> StepResult:
        """One complete Algorithm 1 iteration, including candidate validation."""
        # PAPER_SPECIFIED lines 3-6 / eq. 2-3: anchors, H_t, joint probes.
        tasks = inputs.training_batch
        by_id = {task.task_id: task for task in tasks}
        anchors = {
            task.task_id: execute_as(
                self.executor, current_skill, task, ExecutionRef.anchor(task.task_id),
            ) for task in tasks
        }
        history = tuple((by_id[key], anchors[key]) for key in inputs.history_task_ids)
        probes = tuple(self.generator.generate(current_skill, history, self.num_probes))
        # REPRODUCTION_CHOICE: validate injected generators' count/indices;
        # retain their skill documents and order without renumbering or repair.
        indices = [probe.probe_index for probe in probes]
        if (len(probes) != self.num_probes or any(type(k) is not int for k in indices)
                or set(indices) != set(range(1, self.num_probes + 1))):
            raise ValueError("generator must return K probes with indices 1..K exactly once")

        # PAPER_SPECIFIED lines 7-10 / eq. 4-5: same X_t for every probe.
        executions = {}
        decisions = []
        for probe in probes:
            for task in tasks:
                ref = ExecutionRef.probe(task.task_id, probe.probe_index)
                executions[ref] = execute_as(self.executor, probe.skill, task, ref)
            for task in tasks:
                execution = executions[ExecutionRef.probe(task.task_id, probe.probe_index)]
                decisions.append(compare(
                    self.judge, task, anchors[task.task_id], execution, probe.probe_index,
                ))

        # PAPER_SPECIFIED lines 12-15 / eq. 6-8: decisive-task filter; all K
        # executions together; no y to extractor and descriptions-only clusterer.
        decisive = {d.task_id for d in decisions if d.y != 0}
        records = []
        record_ids = set()
        for task in tasks:
            if task.task_id not in decisive:
                continue
            all_probes = tuple(
                executions[ExecutionRef.probe(task.task_id, probe.probe_index)] for probe in probes
            )
            for record in self.extractor.extract(task, anchors[task.task_id], all_probes):
                # REPRODUCTION_CHOICE: prevent id collisions and cross-task routing.
                if record.task_id != task.task_id or record.record_id in record_ids:
                    raise ValueError("extractor returned wrong task or duplicate record id")
                record_ids.add(record.record_id)
                records.append(record)
        assignments = self.clusterer.cluster(tuple((r.record_id, r.description) for r in records))
        # REPRODUCTION_CHOICE: check injected clusterers too, before the existing
        # assembly helper could attach overlapping/incomplete membership.
        seen = set()
        for assignment in assignments:
            members = assignment.member_record_ids
            if (not members or not members <= record_ids or seen & members
                    or assignment.representative_record_id not in members):
                raise ValueError("clusterer returned an invalid partition")
            seen.update(members)
        if seen != record_ids:
            raise ValueError("clusterer omitted records")
        clusters = assemble_clusters(assignments, records)

        # PAPER_SPECIFIED lines 17-30 / eq. 9-14: unchanged deterministic core.
        units = compute_behavior_units(decisions, clusters, self.update_budget)
        candidate = None
        validation = None
        selected = current_skill
        # PAPER_SPECIFIED lines 32-42 / eq. 15-18: no U_t => no renderer or
        # validation execution; otherwise only a validated candidate can advance.
        if units:
            candidate = self.renderer.render(current_skill, units)
            validation = validate_candidate(
                current_skill, candidate, inputs.validation_batch,
                executor=self.executor, judge=self.judge,
            )
            selected = validation.selected_skill
        return StepResult(
            selected, anchors, probes, executions, tuple(decisions), tuple(records),
            tuple(clusters), tuple(units), candidate, validation,
        )

    def optimize(self, initial_skill: Skill, steps: Sequence[StepInputs]) -> OptimizationResult:
        """iterate t=0..T-1 and return S_T (Algorithm 1).
        
        explicit finite per-step schedule; T=len(steps).
        No callbacks/default selection, early stopping, best-skill search, or
        cache. An empty schedule returns S_0. Check train/validation id
        disjointness across the whole supplied schedule before executing.
        Dataset membership beyond this schedule remains the caller's concern."""
        schedule = tuple(steps)
        training = {task.task_id for step in schedule for task in step.training_batch}
        validation = {task.task_id for step in schedule for task in step.validation_batch}
        if training & validation:
            raise ValueError("training/validation task ids overlap across the schedule")
        selected = initial_skill
        results = []
        for inputs in schedule:
            result = self.step(selected, inputs)
            results.append(result)
            selected = result.selected_skill
        return OptimizationResult(selected, tuple(results))
