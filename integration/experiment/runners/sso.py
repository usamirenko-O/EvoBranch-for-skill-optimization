"""Run native SSO with shared splits and a seeded training schedule."""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

from integration.common.paths import SSO_ROOT, ensure_on_sys_path
from integration.experiment import main_experiment_config
from integration.experiment.cost import PhaseCostAccumulator
from integration.experiment.runners.base import OptimizationResult, OptimizerRunner, RunContext
from integration.experiment.runners.skills import NATIVE_START_CONDITION, initial_skill_text
from integration.harness.base import SolverSession
from integration.harness.registry import build_harness

ensure_on_sys_path(SSO_ROOT)

from sso.core.algorithm import SSOOptimizer, StepInputs  # noqa: E402
from sso.core.types import Execution, ExecutionRef, Skill, Task  # noqa: E402
from sso.executors.base import Executor  # noqa: E402

# SSO 的主实验取值 (T、|X_t|、seed、|H_t|、K、U) 不在代码里：全部只从
# main_experiment/baseline_hparams.json 读取 (integration/experiment/
# main_experiment_config.py)，native SSO 与 SSO+OURS 共用
# resolve_training_schedule / resolve_sso_knobs。


class SharedDirectExecutor(Executor):
    """SSO's f_exec, backed by the shared Direct harness.

    The `Execution.content` SSO reasons over is the harness transcript, which
    is the full record of what the solver did. The benchmark-scorable output
    is kept alongside in `last_outputs` for inspection, but SSO never sees it
    and never scores -- the benchmark metric is computed only at deployment.
    """

    def __init__(self, *, adapter, harness, session: SolverSession) -> None:
        self._adapter = adapter
        self._harness = harness
        self._session = session
        self.last_outputs: dict = {}

    def execute(self, skill: Skill, task: Task) -> Execution:
        bench_tasks = self._adapter.tasks_for_ids([task.task_id], role="optimization")
        run = self._harness.run_task(skill.content, bench_tasks[0], self._session)
        self.last_outputs[task.task_id] = run.output
        content = run.transcript or str(run.output)
        if run.error:
            content = f"{content}\n\n[harness error] {run.error}"
        if not content.strip():
            content = "[empty execution]"
        return Execution(ExecutionRef.anchor(task.task_id), content)


def select_history_task_ids(executed_task_ids: list, history_per_step: int, *, rng_key: str) -> list:
    """Sample history task IDs with a dedicated seeded RNG and retain execution order."""
    ids = list(executed_task_ids)
    n = max(0, int(history_per_step))
    if n >= len(ids):
        return ids
    chosen = set(random.Random(rng_key).sample(ids, n))
    return [t for t in ids if t in chosen]


# ───────────────────── main-experiment training schedule ─────────────────────

#: Where the SSO training schedule (T, |X_t|, schedule seed) comes from.
HPARAMS_SOURCE = main_experiment_config.source("optimizer.sso.training_schedule")
SCHEDULE_KEYS = ("sso_steps", "sso_train_per_step", "sso_schedule_seed")


def main_experiment_training_schedule(benchmark: str) -> dict:
    """Read SSO step count, batch size, and schedule seed for the benchmark."""
    return main_experiment_config.sso_training_schedule(benchmark)


@dataclass(frozen=True)
class SSOKnobs:
    """|H_t|, K probes and update budget U of one run, with their sources."""

    history_per_step: int
    num_probes: int
    update_budget: int
    sources: dict = field(default_factory=dict)

    def record(self) -> dict:
        return {
            "sso_history_per_step": self.history_per_step,
            "sso_num_probes": self.num_probes,
            "sso_update_budget": self.update_budget,
            "sources": dict(self.sources),
        }


def resolve_sso_knobs(extra: dict) -> SSOKnobs:
    """SSO's |H_t| / K / U. The ONLY reader of these values, for native SSO and
    SSO+OURS: the explicit ``--extra`` override when given, else the
    main-experiment configuration."""
    defaults = main_experiment_config.sso_knobs()
    values, sources = {}, {}
    for key, (value, where) in defaults.items():
        if key in extra:
            values[key], sources[key] = int(extra[key]), f"override: --extra {key}"
        else:
            values[key], sources[key] = int(value), where
    if values["sso_num_probes"] < 1 or values["sso_update_budget"] < 0 or values["sso_history_per_step"] < 0:
        raise ValueError(
            "sso_num_probes must be >= 1, sso_update_budget and sso_history_per_step >= 0, "
            f"got {values}"
        )
    return SSOKnobs(
        history_per_step=values["sso_history_per_step"], num_probes=values["sso_num_probes"],
        update_budget=values["sso_update_budget"], sources=sources,
    )


@dataclass(frozen=True)
class TrainingSchedule:
    """The resolved T / |X_t| / seed of one run, and where each value came from."""

    benchmark: str
    steps: int
    train_per_step: int
    seed: int
    sources: dict = field(default_factory=dict)

    def record(self, schedule: list, *, steps_name: str) -> dict:
        """What an artifact / dry-run records: the values, their sources and
        every step's X_t task ids."""
        return {
            "benchmark": self.benchmark,
            steps_name: self.steps,
            "sso_train_per_step": self.train_per_step,
            "sso_schedule_seed": self.seed,
            "sources": {(steps_name if k == "sso_steps" else k): v for k, v in self.sources.items()},
            "selection_rule": "build_schedule: seeded shuffle of TRAIN ids once, sequential "
                              "non-overlapping slices of sso_train_per_step, wrap-around",
            "x_t_task_ids": [list(step["training_task_ids"]) for step in schedule],
        }


def resolve_training_schedule(benchmark: str, extra: dict, *, steps: tuple | None = None) -> TrainingSchedule:
    """The SSO training schedule of one run. The ONLY reader of these values.

    Each of ``sso_steps``, ``sso_train_per_step``, ``sso_schedule_seed`` is the
    explicit ``--extra`` override when given, else the main-experiment value for
    ``benchmark``. SSO+OURS passes ``steps=(value, source)``: its T is its
    resolved ``rounds``, never ``--extra sso_steps``.
    """
    defaults = main_experiment_training_schedule(benchmark)
    values, sources = {}, {}
    for key in SCHEDULE_KEYS:
        if key in extra:
            values[key], sources[key] = int(extra[key]), f"override: --extra {key}"
        else:
            values[key], sources[key] = defaults[key], HPARAMS_SOURCE
    if steps is not None:
        values["sso_steps"], sources["sso_steps"] = int(steps[0]), steps[1]
    if values["sso_steps"] < 1 or values["sso_train_per_step"] < 1:
        raise ValueError(
            f"SSO training schedule needs sso_steps >= 1 and sso_train_per_step >= 1, got {values}"
        )
    return TrainingSchedule(
        benchmark=benchmark, steps=values["sso_steps"],
        train_per_step=values["sso_train_per_step"], seed=values["sso_schedule_seed"],
        sources=sources,
    )


def schedule_from(training: TrainingSchedule, train_ids: list, val_ids: list, *,
                  history_per_step: int) -> list:
    """`build_schedule` for a resolved training schedule (shared by native SSO
    and SSO+OURS, so equal inputs give equal X_t)."""
    return build_schedule(
        train_ids, val_ids,
        steps=training.steps, train_per_step=training.train_per_step,
        history_per_step=history_per_step, seed=training.seed,
    )


def build_schedule(
    train_ids: list,
    val_ids: list,
    *,
    steps: int,
    train_per_step: int,
    history_per_step: int,
    seed: int,
) -> list:
    """Deterministic X_t/H_t/V_t schedule over the manifest's train/val ids.

    One shuffled pass over the train ids in non-overlapping chunks (wrapping
    if more steps than one pass covers); H_t is a random N of that step's own
    executed pairs (`select_history_task_ids`; all of them if fewer than N);
    V_t is the whole validation split every step, so the gate signal is
    consistent step to step.
    """
    if not train_ids:
        raise ValueError("manifest train split is empty; SSO cannot build a schedule")
    rng = random.Random(seed)
    shuffled = list(train_ids)
    rng.shuffle(shuffled)
    schedule = []
    for i in range(steps):
        start = (i * train_per_step) % len(shuffled)
        batch = [shuffled[(start + j) % len(shuffled)] for j in range(min(train_per_step, len(shuffled)))]
        seen: set = set()
        deduped = [t for t in batch if not (t in seen or seen.add(t))]
        schedule.append(
            {
                "training_task_ids": deduped,
                "history_task_ids": select_history_task_ids(
                    deduped, history_per_step, rng_key=f"sso-history:{seed}:step{i}",
                ),
                "validation_task_ids": list(val_ids),
            }
        )
    return schedule


class SSORunner(OptimizerRunner):
    optimizer = "sso"

    @staticmethod
    def preflight(ctx: RunContext) -> list:
        errors = []
        if not ctx.train_ids:
            errors.append("manifest train split is empty; SSO cannot optimize")
        if not ctx.val_ids:
            errors.append("manifest validation split is empty; SSO's gate needs V_t")
        try:
            resolve_training_schedule(ctx.adapter.name, ctx.config.extra)
            resolve_sso_knobs(ctx.config.extra)
        except (TypeError, ValueError, KeyError) as exc:
            errors.append(f"SSO main-experiment settings invalid: {exc}")
        # `optimizer_model` itself is enforced by ExperimentConfig for every
        # optimizer. SSO's extra requirement is that the endpoint behind it
        # supports JSON-object mode -- its five modules parse strict JSON and
        # reject anything else rather than repairing it.
        if not (ctx.config.optimizer_model or "").strip():
            errors.append("SSO requires an optimizer-side alias (JSON-object mode capable)")
        return errors

    def optimize(self, ctx: RunContext) -> OptimizationResult:
        from sso.llm.client import ChatCompletionsClient
        from sso.llm.clusterer import LLMBehaviorClusterer
        from sso.llm.extractor import LLMBehaviorExtractor
        from sso.llm.judge import LLMPairwiseJudge
        from sso.llm.probe_generator import LLMProbeGenerator
        from sso.llm.renderer import LLMRenderer

        cfg = ctx.config
        extra = cfg.extra
        accumulator = PhaseCostAccumulator("optimization")
        session = SolverSession(config=ctx.solver_config, accumulator=accumulator, role="sso_solver_side")
        harness = build_harness(ctx.adapter.name, work_dir=ctx.workdir("sso_harness"), options=ctx.harness_options)

        endpoint = cfg.proxy_base_url.rstrip("/") + "/chat/completions"
        client = ChatCompletionsClient(
            endpoint=endpoint,
            model=cfg.optimizer_model,
            api_key=cfg.api_key,
            max_tokens=int(extra.get("sso_optimizer_max_tokens", cfg.optimizer_max_tokens)),
            temperature=float(extra.get("sso_optimizer_temperature", cfg.temperature)),
            timeout=cfg.timeout_seconds,
        )
        client.reset_usage()

        knobs = resolve_sso_knobs(extra)
        executor = SharedDirectExecutor(adapter=ctx.adapter, harness=harness, session=session)
        optimizer = SSOOptimizer(
            executor=executor,
            generator=LLMProbeGenerator(client),
            judge=LLMPairwiseJudge(client),
            extractor=LLMBehaviorExtractor(client),
            clusterer=LLMBehaviorClusterer(client),
            renderer=LLMRenderer(client),
            num_probes=knobs.num_probes,
            update_budget=knobs.update_budget,
        )

        training = resolve_training_schedule(ctx.adapter.name, extra)
        seed = training.seed
        schedule_spec = schedule_from(
            training, ctx.train_ids, ctx.val_ids,
            history_per_step=knobs.history_per_step,
        )
        # Guard: a schedule may only ever name train/validation ids.
        deployment = set(ctx.deployment_ids)
        for i, step in enumerate(schedule_spec):
            used = set(step["training_task_ids"]) | set(step["history_task_ids"]) | set(step["validation_task_ids"])
            leaked = used & deployment
            if leaked:
                raise ValueError(f"SSO schedule step {i} names deployment ids: {sorted(leaked)[:5]}")

        def _tasks(ids, role):
            return [
                Task(task_id=t.task_id, content=t.payload.get("prompt") or t.payload.get("question")
                     or t.payload.get("instruction") or t.task_id)
                for t in ctx.adapter.tasks_for_ids(ids, role=role)
            ]

        steps = [
            StepInputs(
                training_batch=_tasks(step["training_task_ids"], "train"),
                history_task_ids=step["history_task_ids"],
                validation_batch=_tasks(step["validation_task_ids"], "val"),
            )
            for step in schedule_spec
        ]

        # S_0 = no-skill，论文 Implementation Details 明确规定
        # ("We initialize S_0 as the no-skill condition")。SSO 自己的
        # probe-generator prompt 也是按空 skill 写的。
        initial = Skill(initial_skill_text("sso", ctx.adapter.name))
        t0 = time.time()
        try:
            result = optimizer.optimize(initial, steps)
        finally:
            harness.close()
        wall = time.time() - t0

        optimizer_usage = client.usage_snapshot()
        accumulator.add(
            llm_calls=int(optimizer_usage.get("calls", 0)),
            input_tokens=optimizer_usage.get("prompt_tokens"),
            output_tokens=optimizer_usage.get("completion_tokens"),
            cached_tokens=optimizer_usage.get("cached_tokens"),
            token_source="provider_usage",
            role="sso_optimizer_side",
        )

        skill_path = ctx.run_dir / "sso_selected_skill.md"
        skill_path.write_text(result.selected_skill.content, encoding="utf-8")
        accepted = result.selected_skill.content.strip() != initial.content.strip()

        gate_steps = []
        previous = initial.content
        for i, step in enumerate(result.steps):
            gate_steps.append(
                {
                    "step_idx": i,
                    "had_candidate": step.validation is not None,
                    "wins": step.validation.wins if step.validation else None,
                    "losses": step.validation.losses if step.validation else None,
                    "ties": step.validation.ties if step.validation else None,
                    # Accepted == the gate advanced the skill past what went
                    # INTO this step. Comparing against `step.selected_skill`
                    # would be vacuous: that field *is* the gate's output.
                    "accepted": (step.selected_skill.content != previous)
                    if step.validation
                    else None,
                }
            )
            previous = step.selected_skill.content

        return OptimizationResult(
            final_skill_text=result.selected_skill.content,
            cost=accumulator.snapshot(),
            wall_time_s=wall,
            skill_accepted=accepted,
            final_skill_path=str(skill_path),
            internal_gate={
                "mechanism": "SSO pairwise-judge validation gate (sso.core.validation)",
                "steps": gate_steps,
                "note": "this is SSO's own internal relative win/loss gate, NOT the benchmark metric",
            },
            native_summary={
                "n_steps": len(steps),
                "schedule": schedule_spec,
                "training_schedule": training.record(schedule_spec, steps_name="sso_steps"),
                "sso_knobs": knobs.record(),
                "schedule_policy": {
                    "source": "integration/experiment/runners/sso.py::build_schedule",
                    "seed": seed,
                    "steps": len(steps),
                    "train_per_step": training.train_per_step,
                    "history_per_step": knobs.history_per_step,
                    "history_rule": "random N of the step's executed (x_i, r_i^o) pairs, all if fewer "
                                    "(runners/sso.py::select_history_task_ids)",
                    "validation_per_step": "all validation ids",
                },
                "num_probes": optimizer.num_probes,
                "update_budget": optimizer.update_budget,
                "optimizer_side_usage": optimizer_usage,
                "initial_skill": {
                    "condition": NATIVE_START_CONDITION["sso"],
                    "chars": len(initial.content),
                    "source": "paper Implementation Details: S_0 = the no-skill condition",
                },
                "sso_provenance": "unofficial reproduction of arXiv:2607.28777",
            },
        )


def planned_training_schedule(config, manifest) -> dict:
    """The native run's resolved training schedule and X_t, for a dry-run."""
    training = resolve_training_schedule(config.benchmark, config.extra)
    knobs = resolve_sso_knobs(config.extra)
    spec = schedule_from(
        training, manifest.task_ids("train"), manifest.task_ids("val"),
        history_per_step=knobs.history_per_step,
    )
    return {**training.record(spec, steps_name="sso_steps"), "sso_knobs": knobs.record()}
