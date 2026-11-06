"""Adapt SSO evidence collection and proposal generation to OURS."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Sequence

from integration.common.paths import SSO_ROOT, ensure_on_sys_path
from ours.adapters.base import OptimizerAdapter, ProposalResult
from ours.types import CandidateState, CostAccounting, Evidence, EvidenceSubset

ensure_on_sys_path(SSO_ROOT)

from sso.core._execution import execute_as  # noqa: E402  native role binding
from sso.core.algorithm import SSOOptimizer, StepInputs, StepResult  # noqa: E402
from sso.core.types import Execution, ExecutionRef, Skill, Task  # noqa: E402
from sso.executors.base import Executor  # noqa: E402


class SSOAdapterError(RuntimeError):
    """The SSO adapter was misconfigured or used out of order."""


class _PayloadAugmentingClient:
    """Add branch history to the probe generator's JSON payload."""

    def __init__(self, real, extra: dict) -> None:
        self._real = real
        self._extra = dict(extra)

    def complete(self, *, system: str, user: str) -> str:
        try:
            payload = json.loads(user)
            if isinstance(payload, dict):
                payload.update(self._extra)
                user = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        except (ValueError, TypeError):
            pass
        return self._real.complete(system=system, user=user)


# ─────────────────────────── native components ──────────────────────────────


@dataclass
class SSONativeComponents:
    """f_exec plus SSO's five optimizer-side modules and its K / U.

    ``num_probes`` is SSO's own K (probes per step, paper K). It is NOT the
    OURS partition count, which is ``--k-subsets``.
    """

    executor: Executor
    generator: Any
    judge: Any
    extractor: Any
    clusterer: Any
    renderer: Any
    num_probes: int
    update_budget: int
    client: Any = None

    def optimizer_for(self, executor: Executor, *, generator=None) -> SSOOptimizer:
        return SSOOptimizer(
            executor=executor,
            generator=generator or self.generator,
            judge=self.judge,
            extractor=self.extractor,
            clusterer=self.clusterer,
            renderer=self.renderer,
            num_probes=self.num_probes,
            update_budget=self.update_budget,
        )


def build_native_components(
    *,
    adapter,
    harness,
    session,
    proxy_base_url: str,
    optimizer_model: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
    num_probes: int,
    update_budget: int,
) -> SSONativeComponents:
    """Wire SSO exactly as the native runner does (`runners/sso.py::optimize`).

    Same executor class (the shared Direct harness), same one
    ``ChatCompletionsClient`` on the optimizer alias shared by all five
    modules, same module classes.
    """
    from integration.experiment.runners.sso import SharedDirectExecutor
    from sso.llm.client import ChatCompletionsClient
    from sso.llm.clusterer import LLMBehaviorClusterer
    from sso.llm.extractor import LLMBehaviorExtractor
    from sso.llm.judge import LLMPairwiseJudge
    from sso.llm.probe_generator import LLMProbeGenerator
    from sso.llm.renderer import LLMRenderer

    client = ChatCompletionsClient(
        endpoint=proxy_base_url.rstrip("/") + "/chat/completions",
        model=optimizer_model,
        api_key=api_key,
        max_tokens=int(max_tokens),
        temperature=float(temperature),
        timeout=timeout,
    )
    client.reset_usage()
    return SSONativeComponents(
        executor=SharedDirectExecutor(adapter=adapter, harness=harness, session=session),
        generator=LLMProbeGenerator(client),
        judge=LLMPairwiseJudge(client),
        extractor=LLMBehaviorExtractor(client),
        clusterer=LLMBehaviorClusterer(client),
        renderer=LLMRenderer(client),
        num_probes=int(num_probes),
        update_budget=int(update_budget),
        client=client,
    )


def optimizer_side_usage(components: SSONativeComponents) -> dict:
    """SSO's own client usage totals (optimizer side only)."""
    if components.client is None:
        return {}
    return dict(components.client.usage_snapshot())


def tasks_from_benchmark(benchmark_tasks) -> list[Task]:
    """SSO ``Task`` objects, built exactly as the native runner builds them
    (`runners/sso.py::optimize::_tasks`)."""
    return [
        Task(task_id=t.task_id, content=t.payload.get("prompt") or t.payload.get("question")
             or t.payload.get("instruction") or t.task_id)
        for t in benchmark_tasks
    ]


# ─────────────────────────── anchor replay ──────────────────────────────────


class AnchorReplayExecutor(Executor):
    """Replays a Parent's recorded anchor executions, each exactly once.

    For ``(parent skill text, task id)`` the first request returns the anchor
    execution recorded during evidence collection; every other request -- any
    other skill, or a second request for the same task -- is forwarded to the
    real executor. This does not infer a role from skill text beyond "is this
    the parent whose anchors I hold", and keeps no probe counter.
    """

    def __init__(self, real: Executor, *, parent_skill: str, anchors: dict) -> None:
        self._real = real
        self._parent_skill = parent_skill
        self._pending: dict = dict(anchors)
        self.n_replayed = 0
        self.n_forwarded = 0

    def execute(self, skill: Skill, task: Task) -> Execution:
        if skill.content == self._parent_skill and task.task_id in self._pending:
            self.n_replayed += 1
            return self._pending.pop(task.task_id)
        self.n_forwarded += 1
        return self._real.execute(skill, task)

    def assert_all_replayed(self) -> None:
        if self._pending:
            raise SSOAdapterError(
                "native step did not consume the recorded Parent anchors for "
                f"{sorted(self._pending)}; the proposal would not be conditioned "
                "on the partitioned evidence"
            )


# ─────────────────────────────── evidence ───────────────────────────────────


@dataclass
class SSOEvidence:
    """One Parent's real TRAIN executions for one round.

    ``training_task_ids`` is X_t in native schedule order; ``anchors`` holds
    the executions r_i^o of the Parent on exactly those tasks.
    ``native_history_task_ids`` is the native H_t for the whole X_t, recorded
    for audit only -- each subset derives its own H from its own tasks.
    """

    parent_id: str
    parent_skill: str
    round_id: int
    training_task_ids: tuple[str, ...]
    native_history_task_ids: tuple[str, ...]
    tasks: dict = field(default_factory=dict)
    anchors: dict = field(default_factory=dict)
    is_subset: bool = False


class SSOAdapter(OptimizerAdapter):
    """Native SSO evidence collection and one-step proposal generation."""

    name = "sso"

    def __init__(
        self,
        *,
        components: SSONativeComponents,
        tasks_by_id: dict,
        schedule: Sequence[dict],
        history_per_step: int,
        history_seed: int = 0,
    ) -> None:
        if history_per_step < 0:
            raise SSOAdapterError("history_per_step must be >= 0")
        self.components = components
        self.tasks_by_id = dict(tasks_by_id)
        self.schedule = list(schedule)
        self.history_per_step = int(history_per_step)
        self.history_seed = int(history_seed)
        self._round_id: int | None = None
        # Observability; never read by the flow.
        self.evidence_calls: list[str] = []
        self.propose_calls: list[tuple[str, str]] = []
        self.step_traces: list[dict] = []
        self.history_contexts: list[str] = []

    # ── round lifecycle ──────────────────────────────────────────────────

    def begin_round(self, round_id: int) -> None:
        if round_id >= len(self.schedule):
            raise SSOAdapterError(
                f"round {round_id} has no native training batch; the schedule "
                f"covers {len(self.schedule)} round(s)"
            )
        self._round_id = round_id

    def end_round(self, round_id: int) -> None:
        self._round_id = None

    def _require_round(self) -> int:
        if self._round_id is None:
            raise SSOAdapterError("SSO adapter used outside a round")
        return self._round_id

    # ── ① evidence, per Parent ───────────────────────────────────────────

    def collect_evidence(
        self,
        parent: CandidateState,
        tasks: Sequence[Any],
        *,
        out_dir: str,
        seed: int | None = None,
    ) -> Evidence:
        """Execute THIS Parent on this round's native X_t (TRAIN only).

        X_t comes from the native schedule for the round, so every Parent of a
        round is executed on the same sampled batch, as a single native SSO
        step would be. ``tasks`` is the TRAIN pool the controller was given;
        a scheduled id outside it is refused, which is what keeps validation
        and deployment ids out of evidence even if a schedule were wrong.
        """
        round_id = self._require_round()
        step = self.schedule[round_id]
        pool = {_task_id(t) for t in tasks}
        x_t = [str(t) for t in step["training_task_ids"]]
        outside = [t for t in x_t if t not in pool or t not in self.tasks_by_id]
        if outside:
            raise SSOAdapterError(
                f"round {round_id} schedule names ids outside the TRAIN pool: {outside[:5]}"
            )
        skill = Skill(parent.skill)
        anchors = {
            tid: execute_as(
                self.components.executor, skill, self.tasks_by_id[tid],
                ExecutionRef.anchor(tid),
            )
            for tid in x_t
        }
        self.evidence_calls.append(parent.candidate_id)
        native = SSOEvidence(
            parent_id=parent.candidate_id,
            parent_skill=parent.skill,
            round_id=round_id,
            training_task_ids=tuple(x_t),
            native_history_task_ids=tuple(str(t) for t in step["history_task_ids"]),
            tasks={tid: self.tasks_by_id[tid] for tid in x_t},
            anchors=anchors,
        )
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "anchor_executions.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "parent_id": parent.candidate_id,
                    "round_id": round_id,
                    "training_task_ids": list(x_t),
                    "native_history_task_ids": list(native.native_history_task_ids),
                    "anchors": {tid: e.content for tid, e in anchors.items()},
                },
                f, indent=2, ensure_ascii=False,
            )
        return Evidence(
            evidence_id=f"{parent.candidate_id}:ev",
            parent_id=parent.candidate_id,
            optimizer_name=self.name,
            task_ids=tuple(x_t),
            payload=native,
            workspace_dir=out_dir,
            cost=CostAccounting(),
            metadata={
                "round_id": round_id,
                "n_training_tasks": len(x_t),
                "native_sampling": "integration/experiment/runners/sso.py::build_schedule",
                "evidence_kind": "parent anchor executions r_i^o on native X_t",
            },
        )

    def evidence_features(self, evidence: Evidence) -> dict[str, dict[str, Any]]:
        """Per-task generic view for the partitioner.

        Deliberately OUTCOME-FREE. Native SSO is label-free: its training
        signal is the pairwise judge, and it never sees a benchmark score on a
        training task. Exposing the official scorer's verdict here would hand
        the partition a supervised signal SSO itself does not have.

        ``native_text`` is the native evidence pair (x_i, r_i^o) exactly as
        SSO's own probe generator receives it (`sso.llm.probe_generator
        .LLMProbeGenerator.generate` builds ``{"task": x.content, "execution":
        r.content}`` per history item): the task text and the Parent's anchor
        execution, serialised as that JSON object. Nothing is re-interpreted.
        """
        import json

        native: SSOEvidence = evidence.payload
        out: dict[str, dict[str, Any]] = {}
        for tid in evidence.task_ids:
            content = native.anchors[tid].content
            error = ""
            if "[harness error]" in content:
                error = content.split("[harness error]", 1)[1].strip()
            out[tid] = {
                "error_signal": error,
                "task_type": "",
                "execution_chars": len(content),
                "native_text": json.dumps(
                    {"task": native.tasks[tid].content, "execution": content},
                    ensure_ascii=False,
                ),
            }
        return out

    # ── ② subsets ────────────────────────────────────────────────────────

    def slice_evidence(
        self,
        evidence: Evidence,
        subset_id: str,
        task_ids: Sequence[str],
        *,
        partitioner: str = "",
    ) -> EvidenceSubset:
        """Select subset executions and sample history pairs using the native seeded rule."""
        native: SSOEvidence = evidence.payload
        wanted = {str(t) for t in task_ids}
        unknown = wanted - set(native.training_task_ids)
        if unknown:
            raise SSOAdapterError(f"subset {subset_id} names ids outside X_t: {sorted(unknown)}")
        ordered = tuple(t for t in native.training_task_ids if t in wanted)
        if not ordered:
            raise SSOAdapterError(f"subset {subset_id} selected no evidence")
        from integration.experiment.runners.sso import select_history_task_ids

        history = tuple(select_history_task_ids(
            list(ordered), self.history_per_step,
            rng_key=f"sso-ours-history:{self.history_seed}:round{native.round_id}:{subset_id}",
        ))
        sliced = SSOEvidence(
            parent_id=native.parent_id,
            parent_skill=native.parent_skill,
            round_id=native.round_id,
            training_task_ids=ordered,
            native_history_task_ids=history,
            tasks={t: native.tasks[t] for t in ordered},
            anchors={t: native.anchors[t] for t in ordered},
            is_subset=True,
        )
        return EvidenceSubset(
            subset_id=subset_id,
            evidence_id=evidence.evidence_id,
            parent_id=evidence.parent_id,
            task_ids=ordered,
            payload=sliced,
            partitioner=partitioner,
            metadata={
                "n_tasks": len(ordered),
                "history_task_ids": list(history),
                "history_rule": "random history_per_step of the subset's executed pairs "
                                "(all if fewer), seeded; runners/sso.py::select_history_task_ids",
            },
        )

    # ── ③ ONE native SSO step per (Parent, subset) -> RAW MATERIAL ───────

    def propose(
        self,
        parent: CandidateState,
        subset: EvidenceSubset,
        *,
        out_dir: str,
        round_id: int,
        history_context: str = "",
    ) -> ProposalResult:
        """Generate one raw SSO proposal with an empty validation batch.
        
        Pass branch history through the probe client when supported."""
        native: SSOEvidence = subset.payload
        if native.parent_id != parent.candidate_id or native.parent_skill != parent.skill:
            raise SSOAdapterError(
                f"subset {subset.subset_id} belongs to {native.parent_id}, not "
                f"{parent.candidate_id}; a proposal must start from its own Parent"
            )
        self.propose_calls.append((parent.candidate_id, subset.subset_id))
        self.history_contexts.append(history_context)

        replay = AnchorReplayExecutor(
            self.components.executor,
            parent_skill=parent.skill,
            anchors=native.anchors,
        )
        history_supported = False
        generator = None
        if history_context and self.components.client is not None:
            from sso.llm.probe_generator import LLMProbeGenerator

            generator = LLMProbeGenerator(_PayloadAugmentingClient(
                self.components.client, {"branch_history": history_context}))
            history_supported = True
        optimizer = self.components.optimizer_for(replay, generator=generator)
        inputs = StepInputs(
            training_batch=[native.tasks[t] for t in native.training_task_ids],
            history_task_ids=list(native.native_history_task_ids),
            validation_batch=(),
        )
        result: StepResult = optimizer.step(Skill(parent.skill), inputs)
        replay.assert_all_replayed()
        if result.validation is not None and result.validation.comparisons:
            raise SSOAdapterError(  # pragma: no cover - structurally impossible
                "the native validation gate compared executions inside an OURS "
                "proposal; OURS selection must not depend on it"
            )

        trace = _step_trace(parent, subset, result, replay)
        self.step_traces.append(trace)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "sso_step.json"), "w", encoding="utf-8") as f:
            json.dump(trace, f, indent=2, ensure_ascii=False)

        history_provenance = {
            "history_context_chars": len(history_context or ""),
            "history_context_supported": history_supported,
            "history_context_via": ("native_probe_generator_payload.branch_history"
                                    if history_supported else "unsupported"),
        }
        if result.candidate_skill is None:
            # U_t was empty: native SSO renders nothing. Not a rejection --
            # there is simply no material from this subset.
            return ProposalResult(None, status="skip_no_behavior_units", workspace_dir=out_dir,
                                  provenance=history_provenance)
        return ProposalResult(
            result.candidate_skill.content,
            status="ok",
            provenance={
                "native_mechanism": "sso.core.algorithm.SSOOptimizer.step",
                "native_step": "sso_step_render",
                "native_gate_executed": False,
                "native_gate_discarded_verdict": (
                    None if result.validation is None else result.validation.accepted
                ),
                "parent_candidate_id": parent.candidate_id,
                "subset_task_ids": list(subset.task_ids),
                "history_task_ids": list(native.native_history_task_ids),
                "n_probes": len(result.probes),
                "n_training_decisions": len(result.decisions),
                "n_decisive_decisions": sum(1 for d in result.decisions if d.y != 0),
                "n_behavior_records": len(result.behavior_records),
                "n_clusters": len(result.clusters),
                "behavior_units": trace["behavior_units"],
                "changed_parent_body": result.candidate_skill.content != parent.skill,
                **history_provenance,
            },
            workspace_dir=out_dir,
        )


def _step_trace(parent, subset, result: StepResult, replay: AnchorReplayExecutor) -> dict:
    return {
        "parent_id": parent.candidate_id,
        "parent_skill": parent.skill,
        "subset_id": subset.subset_id,
        "training_task_ids": list(subset.task_ids),
        "n_parent_anchors_replayed": replay.n_replayed,
        "n_executions_forwarded": replay.n_forwarded,
        "probes": [{"probe_index": p.probe_index, "content": p.skill.content}
                   for p in result.probes],
        "decisions": [{"task_id": d.task_id, "probe_index": d.probe_index, "y": d.y}
                      for d in result.decisions],
        "behavior_units": [
            {"cluster_id": u.cluster_id, "description": u.description,
             "intent": u.intent.value, "q": u.q, "e": u.e,
             "contributing_task_count": u.contributing_task_count}
            for u in result.behavior_units
        ],
        "candidate_skill": (result.candidate_skill.content
                            if result.candidate_skill is not None else None),
        "native_gate": {
            "executed_comparisons": (0 if result.validation is None
                                     else len(result.validation.comparisons)),
            "discarded_verdict_accepted": (None if result.validation is None
                                           else result.validation.accepted),
            "used_by_ours": False,
        },
    }


def _task_id(task: Any) -> str:
    if isinstance(task, str):
        return task
    if isinstance(task, dict):
        return str(task.get("id") or task.get("task_id"))
    for attr in ("task_id", "id"):
        if hasattr(task, attr):
            return str(getattr(task, attr))
    raise SSOAdapterError(f"cannot read a task id from {task!r}")


__all__ = [
    "AnchorReplayExecutor",
    "SSOAdapter",
    "SSOAdapterError",
    "SSOEvidence",
    "SSONativeComponents",
    "build_native_components",
    "optimizer_side_usage",
    "tasks_from_benchmark",
]
