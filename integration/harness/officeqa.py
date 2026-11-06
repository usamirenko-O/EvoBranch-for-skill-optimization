"""Execute OfficeQA tool-use tasks over the local document corpus."""
from __future__ import annotations

import json

from integration.common.paths import SKILLOPT_ROOT, ensure_on_sys_path
from integration.harness.base import DirectHarness, SolverSession, TaskRun

ensure_on_sys_path(SKILLOPT_ROOT)

from skillopt.envs.officeqa.tool_runtime import (  # noqa: E402
    resolve_candidate_files,
    resolve_docs_roots,
    run_tool,
)
from skillopt.prompts import load_prompt  # noqa: E402

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find candidate local document files by filename or relative-path glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a local text document excerpt by path and line window.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start": {"type": "integer"},
                    "limit": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search a local text document for a literal pattern and return matching lines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["pattern", "path"],
            },
        },
    },
]

DEFAULT_MAX_TOOL_TURNS = 12


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return []
    if "\n" in text:
        return [p.strip() for p in text.splitlines() if p.strip()]
    if "," in text and not text.lower().endswith(".txt"):
        return [p.strip() for p in text.split(",") if p.strip()]
    return [text]


def build_system_prompt(skill_text: str) -> str:
    """SkillOpt's own OfficeQA rollout system prompt, with the skill spliced in
    at the template's own `{skill_section}` placeholder."""
    template = load_prompt("rollout_system", env="officeqa")
    skill_section = f"## Skill\n{skill_text.strip()}\n\n" if skill_text.strip() else ""
    return template.format(skill_section=skill_section)


class OfficeQAHarness(DirectHarness):
    benchmark = "officeqa"

    def __init__(self, *, docs_root: str | None = None, max_tool_turns: int = DEFAULT_MAX_TOOL_TURNS) -> None:
        self._docs_roots = resolve_docs_roots([docs_root] if docs_root else None)
        self.max_tool_turns = int(max_tool_turns)

    def run_task(self, skill_text: str, task, session: SolverSession) -> TaskRun:
        payload = task.payload
        allowed_files = _as_list(payload.get("source_files"))
        allowed_roots = self._docs_roots
        # Restricting to the task's declared source files is SkillOpt's own
        # offline-mode behavior; when a task declares none, the whole corpus
        # stays visible.
        candidate_files = resolve_candidate_files(allowed_files, allowed_roots)

        system = build_system_prompt(skill_text)
        user = f"## Question\n{payload['question']}\n"
        if candidate_files:
            listed = "\n".join(candidate_files[:20])
            user += f"\n## Candidate local files\n{listed}\n"
        user += "\nUse the tools to gather evidence, then answer inside <answer>...</answer>."

        messages: list = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        transcript = [f"[system]\n{system}", f"[user]\n{user}"]
        calls = 0
        final_text = ""
        error = ""

        for _ in range(self.max_tool_turns):
            msg = session.chat(messages, tools=TOOL_SCHEMAS, tool_choice="auto")
            calls += 1
            content = msg.content or ""
            tool_calls = getattr(msg, "tool_calls", None) or []
            messages.append(
                {
                    "role": "assistant",
                    "content": content,
                    **(
                        {
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    },
                                }
                                for tc in tool_calls
                            ]
                        }
                        if tool_calls
                        else {}
                    ),
                }
            )
            transcript.append(f"[assistant]\n{content}")
            if not tool_calls:
                final_text = content
                break
            for tc in tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                try:
                    label, result = run_tool(
                        tc.function.name,
                        args,
                        allowed_roots=allowed_roots,
                        allowed_files=allowed_files,
                    )
                except Exception as exc:  # noqa: BLE001 - a tool error is task data, not a crash
                    label, result = tc.function.name, f"[tool error: {exc}]"
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                transcript.append(f"[tool {label}]\n{result[:1000]}")
        else:
            error = f"no final answer within {self.max_tool_turns} tool turns"
            final_text = messages[-1].get("content") or "" if messages else ""

        return TaskRun(
            task_id=task.task_id,
            output=final_text,
            transcript="\n\n".join(transcript),
            llm_calls=calls,
            error=error,
            extra={"n_candidate_files": len(candidate_files)},
        )
