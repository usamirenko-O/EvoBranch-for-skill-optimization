"""PubMedQA rollout -- one solver call, one yes/no/maybe answer."""
from __future__ import annotations

from skillopt.envs.pubmedqa.evaluator import evaluate
from skillopt.envs.single_turn_rollout import run_batch as _run_batch
from skillopt.prompts import load_prompt


def build_system(skill_content: str) -> str:
    skill_section = f"## Skill\n{skill_content.strip()}\n\n" if skill_content.strip() else ""
    return load_prompt("rollout_system", env="pubmedqa").format(skill_section=skill_section)


def build_user(item: dict) -> str:
    """PubMedQA reasoning-required setting: structured abstract + question.
    Wording follows SkillGen's `_convert_pubmedqa` so every method sees the
    same task text."""
    contexts = list(item.get("contexts") or [])
    labels = list(item.get("labels") or [])
    if labels and len(labels) == len(contexts):
        context_block = "\n\n".join(f"### {lab}\n{txt}" for lab, txt in zip(labels, contexts))
    else:
        context_block = "\n\n".join(contexts)
    return (
        "Answer the biomedical research question using ONLY the provided abstract.\n\n"
        f"## Abstract\n{context_block}\n\n"
        f"## Question\n{item['question']}\n\n"
        "On the last line, write exactly one of: yes, no, maybe (lowercase, no punctuation)."
    )


def task_type_of(item: dict) -> str:
    return str(item.get("final_decision") or "unknown")


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
