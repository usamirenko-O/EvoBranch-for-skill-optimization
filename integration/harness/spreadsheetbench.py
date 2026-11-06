"""Generate and execute spreadsheet code, then score the resulting workbook."""
from __future__ import annotations

import os

from integration.common.paths import SKILLOPT_ROOT, ensure_on_sys_path
from integration.harness.base import DirectHarness, SolverSession, TaskRun

ensure_on_sys_path(SKILLOPT_ROOT)

from skillopt.envs.spreadsheetbench.codegen_agent import (  # noqa: E402
    _build_system,
    _build_user,
    extract_code,
)
from skillopt.envs.spreadsheetbench.executor import run_generated_code  # noqa: E402
from skillopt.envs.spreadsheetbench.rollout import _find_test_cases  # noqa: E402

DEFAULT_MAX_TURNS = 3
DEFAULT_EXEC_TIMEOUT = 120


class SpreadsheetBenchHarness(DirectHarness):
    benchmark = "spreadsheetbench"

    def __init__(
        self,
        *,
        work_root: str,
        max_turns: int = DEFAULT_MAX_TURNS,
        exec_timeout: int = DEFAULT_EXEC_TIMEOUT,
    ) -> None:
        self.work_root = work_root
        self.max_turns = int(max_turns)
        self.exec_timeout = int(exec_timeout)

    def run_task(self, skill_text: str, task, session: SolverSession) -> TaskRun:
        payload = task.payload
        task_dir = payload["task_dir"]
        cases = _find_test_cases(task_dir)
        out_dir = os.path.join(self.work_root, str(task.task_id))
        os.makedirs(out_dir, exist_ok=True)

        system = _build_system(skill_text)
        case_outputs: dict = {}
        transcript: list = [f"[system]\n{system}"]
        calls = 0
        errors: list = []

        for case_no, init_path, _golden_path in cases:
            user = _build_user(
                payload["instruction"],
                init_path,
                instruction_type=payload.get("instruction_type", ""),
                answer_position=payload.get("answer_position_eval") or payload.get("answer_position", ""),
            )
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            transcript.append(f"[user case {case_no}]\n{user[:2000]}")
            output_path = os.path.join(out_dir, f"{case_no}_output.xlsx")
            ok, err = False, "no attempt"
            for turn in range(self.max_turns):
                msg = session.chat(messages)
                calls += 1
                content = msg.content or ""
                transcript.append(f"[assistant case {case_no} turn {turn}]\n{content[:2000]}")
                code = extract_code(content)
                ok, err = run_generated_code(
                    code, init_path, output_path, timeout=self.exec_timeout
                )
                if ok:
                    break
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your code failed to execute. Fix it and return one complete "
                            f"```python``` block.\n\n# Error\n{err[:2000]}"
                        ),
                    }
                )
                transcript.append(f"[exec error case {case_no} turn {turn}]\n{err[:1000]}")
            if ok:
                case_outputs[case_no] = output_path
            else:
                errors.append(f"case {case_no}: {err[:200]}")

        return TaskRun(
            task_id=task.task_id,
            output=case_outputs,
            transcript="\n\n".join(transcript),
            llm_calls=calls,
            error="; ".join(errors),
            extra={"n_cases": len(cases), "n_executed": len(case_outputs)},
        )
