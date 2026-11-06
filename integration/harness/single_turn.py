"""Execute single-turn benchmark tasks through a shared solver session."""
from __future__ import annotations

from integration.common.paths import SKILLOPT_ROOT, ensure_on_sys_path
from integration.harness.base import DirectHarness, SolverSession, TaskRun

ensure_on_sys_path(SKILLOPT_ROOT)

from skillopt.prompts import load_prompt  # noqa: E402


def build_system_prompt(env: str, skill_text: str) -> str:
    template = load_prompt("rollout_system", env=env)
    skill_section = f"## Skill\n{skill_text.strip()}\n\n" if skill_text.strip() else ""
    return template.format(skill_section=skill_section)


class SingleTurnHarness(DirectHarness):
    def __init__(self, benchmark: str) -> None:
        self.benchmark = benchmark

    def run_task(self, skill_text: str, task, session: SolverSession) -> TaskRun:
        system = build_system_prompt(self.benchmark, skill_text)
        user = task.payload["prompt"]
        msg = session.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}]
        )
        text = msg.content or ""
        return TaskRun(
            task_id=task.task_id,
            output=text,
            transcript=f"[system]\n{system}\n\n[user]\n{user}\n\n[assistant]\n{text}",
            llm_calls=1,
        )
