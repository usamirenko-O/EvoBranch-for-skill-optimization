"""Load and render LiveMath multiple-choice tasks."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from integration.common.paths import LIVEMATH_DATA_ROOT, SKILLOPT_ROOT, ensure_on_sys_path

ensure_on_sys_path(SKILLOPT_ROOT)

from skillopt.envs.livemathematicianbench.dataloader import load_items  # noqa: E402


@dataclass(frozen=True)
class LiveMathItem:
    task_id: str  # "<month>:<no>"
    month: str
    no: int
    question: str
    choices: tuple[tuple[str, str], ...]  # ((label, text), ...)
    correct_label: str
    correct_text: str
    theorem_type: tuple[str, ...]
    theorem: str
    sketch: str
    paper_link: str


def _to_item(raw: dict) -> LiveMathItem:
    return LiveMathItem(
        task_id=str(raw["id"]),
        month=str(raw.get("month", "")),
        no=int(raw.get("no", 0)),
        question=str(raw["question"]),
        choices=tuple((c["label"], c["text"]) for c in raw["choices"]),
        correct_label=str(raw["correct_choice"]["label"]),
        correct_text=str(raw["correct_choice"].get("text", "")),
        theorem_type=tuple(raw.get("theorem_type") or ()),
        theorem=str(raw.get("theorem", "")),
        sketch=str(raw.get("sketch", "")),
        paper_link=str(raw.get("paper_link", "")),
    )


def load_livemath_items(data_root: Path | str = LIVEMATH_DATA_ROOT) -> list[LiveMathItem]:
    """Load and normalize every LiveMath item under `data_root`.

    `data_root` may be a single `qa_*_final.json` file or a directory
    containing one or more monthly files (matches `load_items`'s own
    contract). Defaults to the full locally-downloaded 8-month corpus.
    """
    raw_items = load_items(str(data_root))
    return [_to_item(item) for item in raw_items]


def render_prompt(item: LiveMathItem) -> str:
    """Render one LiveMathItem as a solver-facing prompt.

    Matches SkillOpt's own rendering shape (`rollout.py::_build_user` +
    `_format_choices`) closely enough that a solver sees an equivalent task
    regardless of which optimizer is driving it, without importing
    SkillOpt's rollout module itself (that module pulls in the backend
    router, which this shared module intentionally avoids depending on).
    """
    choice_lines = "\n".join(f"{label}. {text}" for label, text in item.choices)
    return (
        f"## Question\n{item.question}\n\n"
        f"## Choices\n{choice_lines}\n\n"
        "Think step by step, then respond with your final choice as "
        "<answer>LABEL</answer>, where LABEL is a single choice letter."
    )
