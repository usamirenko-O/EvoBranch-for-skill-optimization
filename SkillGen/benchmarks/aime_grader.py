"""AIME grader: deterministic integer extraction + exact match."""

from __future__ import annotations

import re
from typing import Any


_INT_ONLY_LINE = re.compile(r"^\s*0*(\d{1,3})\s*$")
_BOXED_RE = re.compile(r"\\boxed\s*\{\s*0*(\d{1,3})\s*\}", re.IGNORECASE)
_ANSWER_NEAR_END = re.compile(
    r"(?:final\s+answer|answer)\s*(?:is|:|=)?\s*\**\s*0*(\d{1,3})\b",
    re.IGNORECASE,
)
_ANY_INT = re.compile(r"\b0*(\d{1,3})\b")


def _in_range(n: int) -> bool:
    return 0 <= n <= 999


def _extract_int(raw: str) -> tuple[int | None, str]:
    text = (raw or "").strip()
    if not text:
        return None, "none"

    # Rule 1: last non-empty line that is just an integer.
    for line in reversed(text.splitlines()):
        line = line.strip().strip(".,;:*`\"'")
        if not line:
            continue
        m = _INT_ONLY_LINE.match(line)
        if m:
            n = int(m.group(1))
            if _in_range(n):
                return n, "last_line"
        break  # only the truly-last non-empty line qualifies for rule 1

    # Rule 2: \boxed{NNN}
    boxed_hits = _BOXED_RE.findall(text)
    if boxed_hits:
        n = int(boxed_hits[-1])
        if _in_range(n):
            return n, "boxed"

    # Rule 3: "final answer is NNN" / "answer: NNN" - prefer the LAST
    # occurrence since models often restate the answer at the end.
    ans_hits = _ANSWER_NEAR_END.findall(text)
    if ans_hits:
        n = int(ans_hits[-1])
        if _in_range(n):
            return n, "answer_pattern"

    # Rule 4: fall back to the very last 1-3 digit integer.
    all_ints = _ANY_INT.findall(text)
    for tok in reversed(all_ints):
        n = int(tok)
        if _in_range(n):
            return n, "final_integer"

    return None, "none"


def _gt_int(instance_metadata: dict[str, Any]) -> int | None:
    """Prefer the normalised ground_truth carried by the pipeline; fall back
    to raw_answer in metadata. Return None if neither parses as an integer."""
    for key in ("ground_truth", "raw_answer"):
        val = (instance_metadata or {}).get(key)
        if val is None:
            continue
        s = str(val).strip()
        # strip common boxed wrapping
        if s.startswith("\\boxed"):
            m = re.search(r"\d{1,3}", s)
            if m:
                s = m.group(0)
        try:
            n = int(s)
            if _in_range(n):
                return n
        except ValueError:
            continue
    return None


def evaluate_aime_output(
    model_output: str,
    instance_metadata: dict[str, Any],
) -> dict[str, Any]:
    gt = _gt_int(instance_metadata)
    pred, rule = _extract_int(str(model_output or ""))

    if gt is None:
        return {
            "passed": False,
            "score": 0.0,
            "predicted_int": pred,
            "gt_int": None,
            "error_message": "missing or unparseable ground_truth",
            "extraction_rule": rule,
        }

    passed = pred is not None and pred == gt
    err: str | None = None
    if pred is None:
        err = "could not extract integer from agent output"
    elif not passed:
        err = f"answer mismatch (pred={pred} gt={gt})"

    return {
        "passed": bool(passed),
        "score": 1.0 if passed else 0.0,
        "predicted_int": pred,
        "gt_int": gt,
        "error_message": err,
        "extraction_rule": rule,
    }


__all__ = ["evaluate_aime_output"]
