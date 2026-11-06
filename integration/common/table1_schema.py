"""Validate the shared experiment summary schema."""
from __future__ import annotations

from typing import Any

_COST_FIELDS = ("calls", "prompt_tokens", "completion_tokens", "cached_tokens")

# field_name -> predicate(value) -> True if valid. `None` is accepted for a
# field whenever the docstring below says so (e.g. no held-out D_test run
# happened, or a rate genuinely could not be computed) -- callers must
# still include the key, just possibly with value None, never omit it.
_REQUIRED_FIELDS: dict[str, str] = {
    "optimizer": "one of 'skillopt', 'sso', 'skillgen'",
    "solver_model_alias": "the shared-proxy alias used for the task-solving model",
    "resolved_models": "dict[alias -> physical model string], from resolve_models.resolve_model_aliases",
    "split_manifest_hash": "sha256 hex digest (64 chars) over the three frozen items.json files' content",
    "n_train": "int >= 0",
    "n_val": "int >= 0",
    "n_test": "int >= 0",
    "test_accuracy": "float in [0, 1], or None if D_test was never evaluated",
    "test_n": "int >= 0",
    "optimization_cost": "dict with calls/prompt_tokens/completion_tokens/cached_tokens (all int >= 0)",
    "deployment_cost": "dict with calls/prompt_tokens/completion_tokens/cached_tokens (all int >= 0)",
    "truncation_rate": "float in [0, 1] over deployment calls, or None if finish_reason wasn't available",
    "no_answer_tag_rate": "float in [0, 1] over deployment calls, or None if not computed",
    "final_skill_path": "str path to the persisted final skill/state, or None if none was produced",
    "skill_accepted": "bool -- SkillOpt: best != initial skill; SSO: any step accepted; SkillGen: status == active",
    "launch_command": "str -- the exact command line that produced this run",
    "wall_time_s": "float >= 0, total wall-clock seconds for the run",
    "optimizer_specific": "dict -- method-specific extra fields; contents not validated here",
}

_VALID_OPTIMIZERS = {"skillopt", "sso", "skillgen"}


def _is_rate_or_none(v: Any) -> bool:
    return v is None or (isinstance(v, (int, float)) and not isinstance(v, bool) and 0.0 <= float(v) <= 1.0)


def _is_nonneg_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def _is_cost_dict(v: Any) -> bool:
    if not isinstance(v, dict):
        return False
    return all(field in v and _is_nonneg_int(v[field]) for field in _COST_FIELDS)


def validate_table1_summary(summary: dict) -> list[str]:
    """Return a list of validation error strings; empty list == valid.

    Never raises on a malformed `summary` -- a caller checking "is this
    Table-1-ready" wants a report, not an exception mid-audit.
    """
    errors: list[str] = []
    if not isinstance(summary, dict):
        return ["summary is not a dict"]

    for field in _REQUIRED_FIELDS:
        if field not in summary:
            errors.append(f"missing required field: {field!r}")

    if "optimizer" in summary and summary["optimizer"] not in _VALID_OPTIMIZERS:
        errors.append(f"optimizer must be one of {sorted(_VALID_OPTIMIZERS)}, got {summary['optimizer']!r}")

    if "solver_model_alias" in summary and not isinstance(summary["solver_model_alias"], str):
        errors.append("solver_model_alias must be a str")

    if "resolved_models" in summary:
        rm = summary["resolved_models"]
        if not isinstance(rm, dict) or not rm or not all(isinstance(k, str) and isinstance(v, str) for k, v in rm.items()):
            errors.append("resolved_models must be a non-empty dict[str, str]")
        elif all(v == "unresolved" for v in rm.values()):
            errors.append("resolved_models: every alias is 'unresolved' -- physical model was never confirmed")

    if "split_manifest_hash" in summary:
        h = summary["split_manifest_hash"]
        if not isinstance(h, str) or len(h) != 64:
            errors.append("split_manifest_hash must be a 64-character sha256 hex digest")

    for field in ("n_train", "n_val", "n_test", "test_n"):
        if field in summary and not _is_nonneg_int(summary[field]):
            errors.append(f"{field} must be a non-negative int")

    if "test_accuracy" in summary:
        v = summary["test_accuracy"]
        if not (v is None or (isinstance(v, (int, float)) and not isinstance(v, bool) and 0.0 <= float(v) <= 1.0)):
            errors.append("test_accuracy must be a float in [0, 1] or None")

    for field in ("optimization_cost", "deployment_cost"):
        if field in summary and not _is_cost_dict(summary[field]):
            errors.append(f"{field} must be a dict with non-negative int calls/prompt_tokens/completion_tokens/cached_tokens")

    for field in ("truncation_rate", "no_answer_tag_rate"):
        if field in summary and not _is_rate_or_none(summary[field]):
            errors.append(f"{field} must be a float in [0, 1] or None")

    if "final_skill_path" in summary and not (summary["final_skill_path"] is None or isinstance(summary["final_skill_path"], str)):
        errors.append("final_skill_path must be a str or None")

    if "skill_accepted" in summary and not isinstance(summary["skill_accepted"], bool):
        errors.append("skill_accepted must be a bool")

    if "launch_command" in summary and not isinstance(summary["launch_command"], str):
        errors.append("launch_command must be a str")

    if "wall_time_s" in summary:
        v = summary["wall_time_s"]
        if not (isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0):
            errors.append("wall_time_s must be a non-negative number")

    if "optimizer_specific" in summary and not isinstance(summary["optimizer_specific"], dict):
        errors.append("optimizer_specific must be a dict")

    return errors
