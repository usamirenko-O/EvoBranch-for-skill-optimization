"""PubMedQA scoring: exact match on the official yes / no / maybe label."""
from __future__ import annotations

import re

VALID = ("yes", "no", "maybe")
_TOKEN_RE = re.compile(r"\b(yes|no|maybe)\b", re.IGNORECASE)
_BOXED_RE = re.compile(r"\\boxed\{\s*([^}]+?)\s*\}", re.IGNORECASE)


def _normalise(tok: str) -> str | None:
    t = tok.strip().strip(".,:;!?\"'`*()[]").lower()
    return t if t in VALID else None


def extract_label(raw: str) -> tuple[str | None, str]:
    text = (raw or "").strip()
    if not text:
        return None, "none"
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        cand = _normalise(line)
        if cand:
            return cand, "last_line"
        m = _TOKEN_RE.findall(line)
        if m:
            cand = _normalise(m[-1])
            if cand:
                return cand, "last_line"
        break
    boxed = _BOXED_RE.findall(text)
    if boxed:
        cand = _normalise(boxed[-1])
        if cand:
            return cand, "boxed"
    all_tokens = _TOKEN_RE.findall(text)
    if all_tokens:
        cand = _normalise(all_tokens[-1])
        if cand:
            return cand, "regex_scan"
    return None, "none"


def evaluate(response_text: str, item: dict) -> dict:
    gt = str(item.get("final_decision") or "").strip().lower()
    pred, rule = extract_label(response_text or "")
    passed = pred is not None and pred == gt
    if pred is None:
        fail_reason = "could not extract yes/no/maybe from the output"
    elif not passed:
        fail_reason = f"label mismatch: predicted {pred!r}, expected {gt!r}"
    else:
        fail_reason = ""
    return {
        "score": 1.0 if passed else 0.0,
        "passed": bool(passed),
        "predicted": pred or "",
        "correct_label": gt,
        "extraction_rule": rule,
        "fail_reason": fail_reason,
        "detail": f"Predicted: {pred!r}\nCorrect: {gt!r}\nExact match: {float(passed)}",
    }
