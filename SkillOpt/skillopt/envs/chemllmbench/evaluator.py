"""ChemLLMBench yield prediction scoring: exact Yes / No match."""
from __future__ import annotations

import re


def _last_nonempty_line(text: str) -> str:
    for line in reversed((text or "").splitlines()):
        line = line.strip()
        if line:
            return line
    return ""


def extract_yes_no(text: str) -> str | None:
    if not text:
        return None
    last = _last_nonempty_line(text).strip(" .\t,\"'")
    low = last.lower()
    if low in ("yes", "no"):
        return "Yes" if low == "yes" else "No"
    m = re.fullmatch(r"(yes|no)[\.\,\!\?]*", low)
    if m:
        return "Yes" if m.group(1) == "yes" else "No"
    tokens = re.findall(r"\b(yes|no)\b", (text or "").lower())
    if tokens:
        return "Yes" if tokens[-1] == "yes" else "No"
    return None


def evaluate(response_text: str, item: dict) -> dict:
    gt = str(item.get("label") or "").strip()
    pred = extract_yes_no(response_text or "")
    passed = pred is not None and pred.lower() == gt.lower()
    if pred is None:
        fail_reason = "could not parse Yes/No from the output"
    elif not passed:
        fail_reason = f"label mismatch: predicted {pred!r}, expected {gt!r}"
    else:
        fail_reason = ""
    return {
        "score": 1.0 if passed else 0.0,
        "passed": bool(passed),
        "predicted": pred or "",
        "correct_label": gt,
        "fail_reason": fail_reason,
        "detail": f"Predicted: {pred!r}\nCorrect: {gt!r}\nExact match: {float(passed)}",
    }
