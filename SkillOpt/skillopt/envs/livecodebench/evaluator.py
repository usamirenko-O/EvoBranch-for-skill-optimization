"""LiveCodeBench scoring: pass@1 over public + private tests.

One function, `evaluate(response_text, item)`, used by BOTH SkillOpt's own
rollout and the shared experiment layer. The heavy lifting is the vendored
official grader (`lcb_grader.evaluate_livecodebench_output`); this module
only fetches the private tests for the item and shapes the verdict.
"""
from __future__ import annotations

from skillopt.envs.livecodebench.dataloader import read_raw_line
from skillopt.envs.livecodebench.lcb_grader import (
    _DEFAULT_TIMEOUT_SECONDS,
    evaluate_livecodebench_output,
    extract_livecodebench_code,
)

PER_TEST_TIMEOUT_SECONDS = _DEFAULT_TIMEOUT_SECONDS  # LiveCodeBench's own 6 s/test


def load_private_tests(item: dict) -> str:
    """The still-encoded `private_test_cases` string for `item`."""
    raw = read_raw_line(item["source_file"], item["source_offset"])
    if str(raw.get("question_id")) != str(item["id"]):
        raise ValueError(
            f"offset {item['source_offset']} in {item['source_file']} holds "
            f"{raw.get('question_id')!r}, expected {item['id']!r}"
        )
    return raw.get("private_test_cases", "[]") or "[]"


def evaluate(response_text: str, item: dict) -> dict:
    instance_metadata = {
        "public_test_cases": list(item.get("public_test_cases") or []),
        "private_test_cases": load_private_tests(item),
        "lcb_metadata": {"func_name": item.get("func_name") or None},
        "timeout_seconds": PER_TEST_TIMEOUT_SECONDS,
    }
    result = evaluate_livecodebench_output(
        model_output=response_text or "",
        instance_metadata=instance_metadata,
    )
    passed = bool(result.get("passed"))
    code = result.get("extracted_code") or extract_livecodebench_code(response_text or "")
    fail_reason = ""
    if not passed:
        fail_reason = (
            f"{result.get('error_message') or 'failed'} "
            f"(passed {result.get('passed_tests', 0)}/{result.get('total_tests', 0)} tests)"
        )
        for key in ("inputs", "expected", "output"):
            if result.get(key) is not None:
                fail_reason += f"\n  {key}: {str(result[key])[:300]}"
    return {
        "score": 1.0 if passed else 0.0,
        "passed": passed,
        "predicted": code[:2000],
        "passed_tests": int(result.get("passed_tests", 0) or 0),
        "total_tests": int(result.get("total_tests", 0) or 0),
        "error_code": result.get("error_code"),
        "error_message": result.get("error_message"),
        "fail_reason": fail_reason,
        "detail": (
            f"Passed: {passed}\n"
            f"Tests passed: {result.get('passed_tests', 0)}/{result.get('total_tests', 0)}\n"
            f"Error: {result.get('error_message') or ''}\n"
            + ("" if passed else fail_reason)
        ),
    }
