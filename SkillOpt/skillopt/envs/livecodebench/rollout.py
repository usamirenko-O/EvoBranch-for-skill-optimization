"""LiveCodeBench rollout -- one solver call, one Python solution."""
from __future__ import annotations

from skillopt.envs.livecodebench.evaluator import evaluate
from skillopt.envs.single_turn_rollout import run_batch as _run_batch
from skillopt.prompts import load_prompt


def build_system(skill_content: str) -> str:
    skill_section = f"## Skill\n{skill_content.strip()}\n\n" if skill_content.strip() else ""
    return load_prompt("rollout_system", env="livecodebench").format(skill_section=skill_section)


def build_user(item: dict) -> str:
    """The task rendering. Same wording as SkillGen's
    `build_livecodebench_prompt`, so every method (and every phase) presents
    the problem identically."""
    title = str(item.get("question_title") or "").strip()
    content = str(item.get("question_content") or "").strip()
    starter = str(item.get("starter_code") or "").strip()
    parts = [
        "Solve the following competitive programming problem in Python 3.",
        f"Title: {title}" if title else "",
        "Problem:\n" + content,
    ]
    if starter:
        parts.append("Starter code (use it only if it helps):\n" f"```python\n{starter}\n```")
    parts.append(
        "Return only the final Python 3 solution code. "
        "Do not include prose, analysis, or surrounding markdown unless a code fence is unavoidable."
    )
    return "\n\n".join(p for p in parts if p)


def task_type_of(item: dict) -> str:
    return str(item.get("difficulty") or "unknown")


def run_batch(items: list[dict], out_root: str, skill_content: str, **kwargs) -> list[dict]:
    return _run_batch(
        items,
        out_root,
        skill_content,
        build_system=build_system,
        build_user=build_user,
        evaluate=evaluate,
        task_type_of=task_type_of,
        **kwargs,
    )
