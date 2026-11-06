"""Parse LiveMath answers and score them against the correct choice label."""
from __future__ import annotations

import re

from integration.common.paths import SKILLOPT_ROOT, ensure_on_sys_path

ensure_on_sys_path(SKILLOPT_ROOT)

from skillopt.envs.livemathematicianbench.evaluator import (  # noqa: E402
    extract_answer,
    normalize_label,
    parse_choice_label,
)

__all__ = ["extract_answer", "parse_choice_label", "recover_choice_label", "score"]

#: "Answer: B" / "Final answer is (C)" / "option D" -- the label after a
#: keyword. Scanned for the LAST match, because a response states its final
#: choice last.
_KEYWORD_LABEL = re.compile(
    r"(?:final\s+)?(?:answer|choice|option)\s*(?:is\s*)?[::\-]?\s*\(?\s*([A-Za-z])\s*\)?(?![A-Za-z])",
    re.IGNORECASE,
)
#: A bare label sitting at the very end: "... so B", "(C)", "D."
_TRAILING_LABEL = re.compile(r"(?:^|[\s:：(\[])\(?\s*([A-Za-z])\s*\)?\s*[.)\]]?\s*$")


def recover_choice_label(raw_text: str, choices: list[dict]) -> str | None:
    """Recover a valid choice label after an answer keyword or at the end of the response."""
    valid = {normalize_label(c.get("label", "")) for c in choices}
    valid.discard("")
    if not valid:
        return None
    answer = extract_answer(raw_text)
    for pattern, take_last in ((_KEYWORD_LABEL, True), (_TRAILING_LABEL, False)):
        found = pattern.findall(answer) if take_last else pattern.findall(answer)
        for candidate in (reversed(found) if take_last else found):
            label = normalize_label(candidate)
            if label in valid:
                return label
    return None


def score(raw_text: str, choices: list[tuple[str, str]], correct_label: str) -> dict:
    """Score a response against the normalized correct label; choices are (label, text) pairs."""
    choice_dicts = [{"label": label, "text": text} for label, text in choices]
    predicted = parse_choice_label(raw_text, choice_dicts)
    valid_labels = {normalize_label(c["label"]) for c in choice_dicts}
    if predicted not in valid_labels:
        recovered = recover_choice_label(raw_text, choice_dicts)
        if recovered is not None:
            predicted = recovered
    correct_label_normalized = normalize_label(correct_label)
    correct = predicted == correct_label_normalized
    return {
        "predicted_label": predicted,
        "correct_label": correct_label_normalized,
        "score": float(correct),
        "passed": correct,
    }
