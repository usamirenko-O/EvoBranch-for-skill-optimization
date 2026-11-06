"""LiveMathematicianBench grader: deterministic MCQ label extraction + match."""
from __future__ import annotations

import re
from typing import Any

_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def _extract_answer(text: str) -> str:
    matches = _ANSWER_TAG_RE.findall(text)
    if matches:
        return matches[-1].strip()
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if lines:
        return lines[-1]
    return text.strip()


def _normalize_label(text: str) -> str:
    return str(text).strip().upper().rstrip(".):")


def _parse_choice_label(prediction_text: str, choices: list[dict]) -> str:
    answer = _extract_answer(prediction_text)
    label = _normalize_label(answer)
    valid_labels = {_normalize_label(choice.get("label", "")) for choice in choices}
    if label in valid_labels:
        return label

    answer_lower = answer.lower()
    for choice in choices:
        choice_label = _normalize_label(choice.get("label", ""))
        choice_text = str(choice.get("text", "")).strip()
        if choice_text and choice_text.lower() == answer_lower:
            return choice_label

    first_token = _normalize_label(answer.split()[0]) if answer.split() else ""
    if first_token in valid_labels:
        return first_token
    return label


def evaluate_livemath_output(
    model_output: str,
    instance_metadata: dict[str, Any],
) -> dict[str, Any]:
    choices = instance_metadata.get("choices") or []
    gt_label = _normalize_label(instance_metadata.get("correct_choice_label", ""))
    predicted_label = _parse_choice_label(str(model_output or ""), choices)

    if not gt_label:
        return {
            "passed": False,
            "score": 0.0,
            "predicted_label": predicted_label,
            "gt_label": gt_label,
            "error_message": "missing correct_choice_label in instance metadata",
        }

    passed = predicted_label == gt_label
    return {
        "passed": bool(passed),
        "score": 1.0 if passed else 0.0,
        "predicted_label": predicted_label,
        "gt_label": gt_label,
        "error_message": None if passed else f"answer mismatch (pred={predicted_label} gt={gt_label})",
    }


__all__ = ["evaluate_livemath_output"]
