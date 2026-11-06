"""Shared Direct-harness adapter for SkillGen."""

from __future__ import annotations

import time
import uuid

_CONTEXT = None


class SharedDirectContext:
    """Everything the shared harness needs, installed once per run."""

    def __init__(self, *, adapter, harness, session, skill_text_hook=None):
        self.adapter = adapter
        self.harness = harness
        self.session = session
        self.skill_text_hook = skill_text_hook


def install_context(context) -> None:
    global _CONTEXT
    _CONTEXT = context


def clear_context() -> None:
    global _CONTEXT
    _CONTEXT = None


def _require_context():
    if _CONTEXT is None:
        raise RuntimeError(
            "shared_direct benchmark instances were used without a run context. "
            "The integration layer must call "
            "benchmarks.shared_direct_adapter.install_context(...) first; there is "
            "deliberately no fallback execution or scoring path."
        )
    return _CONTEXT


def _skill_text(skill_bundle) -> str:
    if skill_bundle is None:
        return ""
    body = getattr(skill_bundle, "body", "") or ""
    return body


def run_shared_direct_agent(instance, skill_bundle, config):
    """Execute one instance through the shared Direct harness."""
    from models import Trajectory  # SkillGen-internal

    ctx = _require_context()
    task_id = instance.metadata.get("task_id") or instance.instance_id
    tasks = ctx.adapter.tasks_for_ids([task_id], role="optimization")
    skill_text = _skill_text(skill_bundle)

    t0 = time.time()
    run = ctx.harness.run_task(skill_text, tasks[0], ctx.session)
    latency = time.time() - t0

    agent_config = config.to_dict() if hasattr(config, "to_dict") else {}
    agent_config["skill_id"] = getattr(skill_bundle, "skill_id", None)
    agent_config["benchmark"] = "shared_direct"
    agent_config["shared_benchmark"] = ctx.adapter.name
    # The benchmark-scorable output rides on the trajectory itself, so grading
    # does not depend on any module-level lookup table that a concurrent run
    # could race.
    agent_config["shared_direct_output"] = run.output
    agent_config["shared_direct_error"] = run.error
    agent_config["shared_direct_llm_calls"] = run.llm_calls

    return Trajectory(
        trajectory_id=str(uuid.uuid4()),
        instance_id=instance.instance_id,
        agent_config=agent_config,
        messages=[{"role": "assistant", "content": run.transcript}],
        final_output=run.transcript,
        latency=latency,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


def evaluate_shared_direct_trajectory(traj, instance, task_type):
    """Score one trajectory with the benchmark's OFFICIAL scorer."""
    ctx = _require_context()
    task_id = instance.metadata.get("task_id") or instance.instance_id
    output = (traj.agent_config or {}).get("shared_direct_output")
    if output is None:
        name = ctx.adapter.name
        output = {"won": False, "steps": 0} if name == "alfworld" else ({} if name == "spreadsheetbench" else "")

    result = ctx.adapter.evaluate({task_id: output}, role="deployment")
    verdict = result.per_task.get(task_id, {})
    score = float(verdict.get("score") or 0.0)

    traj.success = score >= 1.0
    traj.score = score
    if traj.metadata is None:
        traj.metadata = {}
    traj.metadata["shared_direct_eval"] = {
        "source": "shared_direct_official_scorer",
        "scorer_ref": ctx.adapter.scorer_ref,
        "metric_name": ctx.adapter.metric_name,
        "verdict": {k: v for k, v in verdict.items() if k != "cases"},
    }
    if not traj.success:
        traj.error_summary = (traj.agent_config or {}).get("shared_direct_error") or (
            f"official scorer returned score={score}"
        )
    return traj
