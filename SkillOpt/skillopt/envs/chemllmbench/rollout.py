"""ChemLLMBench yield-prediction rollout -- one solver call, Yes or No."""
from __future__ import annotations

from skillopt.envs.chemllmbench.dataloader import SUBSET_NAMES
from skillopt.envs.chemllmbench.evaluator import evaluate
from skillopt.envs.single_turn_rollout import run_batch as _run_batch
from skillopt.prompts import load_prompt


def build_system(skill_content: str) -> str:
    skill_section = f"## Skill\n{skill_content.strip()}\n\n" if skill_content.strip() else ""
    return load_prompt("rollout_system", env="chemllmbench").format(skill_section=skill_section)


def build_user(item: dict) -> str:
    """Same wording as SkillGen's `convert_yield_prediction`, so every method
    sees the task exactly as the SkillGen paper posed it."""
    rxn_name = SUBSET_NAMES.get(str(item.get("subset")), "chemical")
    return (
        f"You are an expert chemist. For the following {rxn_name} reaction "
        "(reactants/reagents/catalysts separated by '.', product separated by '>>'), "
        "predict whether the reaction will give a high yield (> 30%). "
        "Answer with only 'Yes' or 'No' on the last line.\n\n"
        f"Reaction: {item['reaction']}"
    )


def task_type_of(item: dict) -> str:
    return str(item.get("subset") or "unknown")


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
