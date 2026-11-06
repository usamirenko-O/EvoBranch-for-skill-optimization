"""Execute one LiveMath question with a skill as the system prompt."""
from __future__ import annotations

from integration.harness.base import DirectHarness, SolverSession, TaskRun

DEFAULT_SYSTEM = (
    "You are an expert mathematician. Answer the multiple-choice question."
)


class LiveMathHarness(DirectHarness):
    benchmark = "livemathematicianbench"

    def run_task(self, skill_text: str, task, session: SolverSession) -> TaskRun:
        system = skill_text.strip() or DEFAULT_SYSTEM
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
