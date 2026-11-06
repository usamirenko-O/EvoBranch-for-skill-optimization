"""Serialize native-baseline configuration, metrics, provenance, and costs."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from integration.experiment.cost import TOKEN_SOURCES, PhaseCost

SCHEMA_VERSION = "1.0"

_EXPERIMENT_FIELDS = (
    "optimizer",
    "benchmark",
    "harness",
    "solver_model",
    "solver_model_resolved",
    "split_manifest_id",
    "split_manifest_version",
    "split_role",
)
_PERFORMANCE_FIELDS = ("metric_name", "metric_value", "n_tasks", "secondary_metrics", "scorer_ref")
_COST_FIELDS = (
    "llm_calls",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cached_tokens",
    "token_source",
    "per_role",
)
_PROVENANCE_FIELDS = (
    "run_id",
    "created_utc",
    "code_freeze",
    "benchmark_manifest_hash",
    "launch_command",
    "notes",
)

_VALID_OPTIMIZERS = {"skillopt", "sso", "skillgen"}


def build_record(
    *,
    optimizer: str,
    benchmark: str,
    harness: str,
    solver_model: str,
    solver_model_resolved: str | None,
    split_manifest_id: str,
    split_manifest_version: str,
    split_manifest_hash: str,
    split_role: str,
    metric_name: str,
    metric_value: float | None,
    n_tasks: int,
    scorer_ref: str,
    optimization_cost: PhaseCost,
    deployment_cost: PhaseCost,
    run_id: str,
    code_freeze: dict,
    launch_command: str,
    secondary_metrics: dict | None = None,
    notes: list | None = None,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": {
            "optimizer": optimizer,
            "benchmark": benchmark,
            "harness": harness,
            "solver_model": solver_model,
            "solver_model_resolved": solver_model_resolved,
            "split_manifest_id": split_manifest_id,
            "split_manifest_version": split_manifest_version,
            "split_role": split_role,
        },
        "performance": {
            "metric_name": metric_name,
            "metric_value": metric_value,
            "n_tasks": int(n_tasks),
            "secondary_metrics": dict(secondary_metrics or {}),
            "scorer_ref": scorer_ref,
        },
        "optimization_cost": optimization_cost.to_dict(),
        "deployment_cost": deployment_cost.to_dict(),
        "provenance": {
            "run_id": run_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "code_freeze": dict(code_freeze),
            "benchmark_manifest_hash": split_manifest_hash,
            "launch_command": launch_command,
            "notes": list(notes or []),
        },
    }


# ------------------------------------------------------------- validation --


def _is_nonneg_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def _is_opt_nonneg_int(v: Any) -> bool:
    return v is None or _is_nonneg_int(v)


def _validate_cost(section: Any, label: str) -> list:
    errors = []
    if not isinstance(section, dict):
        return [f"{label} must be a dict"]
    for f in _COST_FIELDS:
        if f not in section:
            errors.append(f"{label}: missing field {f!r}")
    if not _is_nonneg_int(section.get("llm_calls")):
        errors.append(f"{label}.llm_calls must be a non-negative int")
    for f in ("input_tokens", "output_tokens", "total_tokens", "cached_tokens"):
        if f in section and not _is_opt_nonneg_int(section[f]):
            errors.append(f"{label}.{f} must be a non-negative int or None")
    src = section.get("token_source")
    if src not in TOKEN_SOURCES:
        errors.append(f"{label}.token_source must be one of {TOKEN_SOURCES}, got {src!r}")
    # total must agree with its parts, or be absent because a part was absent.
    i, o, t = section.get("input_tokens"), section.get("output_tokens"), section.get("total_tokens")
    if i is not None and o is not None:
        if t != i + o:
            errors.append(f"{label}.total_tokens ({t}) != input+output ({i}+{o})")
    elif t is not None:
        errors.append(
            f"{label}.total_tokens is set but input/output are not both known -- "
            "a total must never be estimated"
        )
    if "per_role" in section and not isinstance(section["per_role"], dict):
        errors.append(f"{label}.per_role must be a dict")
    return errors


def validate_record(record: Any) -> list:
    """Return a list of error strings; empty list == matrix-ready."""
    errors: list = []
    if not isinstance(record, dict):
        return ["record is not a dict"]

    if record.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION!r}, got {record.get('schema_version')!r}")

    exp = record.get("experiment")
    if not isinstance(exp, dict):
        errors.append("experiment must be a dict")
    else:
        for f in _EXPERIMENT_FIELDS:
            if f not in exp:
                errors.append(f"experiment: missing field {f!r}")
        if exp.get("optimizer") not in _VALID_OPTIMIZERS:
            errors.append(
                f"experiment.optimizer must be one of {sorted(_VALID_OPTIMIZERS)}, "
                f"got {exp.get('optimizer')!r} -- native baselines only; 'ours' must "
                "never be recorded through this artifact"
            )
        for f in ("benchmark", "harness", "solver_model", "split_manifest_id", "split_role"):
            if f in exp and not isinstance(exp[f], str):
                errors.append(f"experiment.{f} must be a str")

    perf = record.get("performance")
    if not isinstance(perf, dict):
        errors.append("performance must be a dict")
    else:
        for f in _PERFORMANCE_FIELDS:
            if f not in perf:
                errors.append(f"performance: missing field {f!r}")
        if not isinstance(perf.get("metric_name"), str) or not perf.get("metric_name"):
            errors.append("performance.metric_name must be a non-empty str")
        mv = perf.get("metric_value")
        if not (mv is None or (isinstance(mv, (int, float)) and not isinstance(mv, bool))):
            errors.append("performance.metric_value must be a number or None")
        if not _is_nonneg_int(perf.get("n_tasks")):
            errors.append("performance.n_tasks must be a non-negative int")
        if not isinstance(perf.get("scorer_ref"), str) or not perf.get("scorer_ref"):
            errors.append("performance.scorer_ref must be a non-empty str")

    errors += _validate_cost(record.get("optimization_cost"), "optimization_cost")
    errors += _validate_cost(record.get("deployment_cost"), "deployment_cost")

    prov = record.get("provenance")
    if not isinstance(prov, dict):
        errors.append("provenance must be a dict")
    else:
        for f in _PROVENANCE_FIELDS:
            if f not in prov:
                errors.append(f"provenance: missing field {f!r}")
        h = prov.get("benchmark_manifest_hash")
        if not isinstance(h, str) or len(h) != 64:
            errors.append("provenance.benchmark_manifest_hash must be a 64-char sha256 hex digest")
        if not isinstance(prov.get("run_id"), str) or not prov.get("run_id"):
            errors.append("provenance.run_id must be a non-empty str")
        if not isinstance(prov.get("code_freeze"), dict):
            errors.append("provenance.code_freeze must be a dict")

    return errors


def write_record(path: Path | str, record: dict) -> Path:
    """Persist a record, refusing to write one that fails validation.

    A silently-invalid artifact is worse than a missing one: the matrix
    script would read it and produce a wrong table.
    """
    errors = validate_record(record)
    if errors:
        raise ValueError("refusing to write an invalid experiment record:\n  " + "\n  ".join(errors))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(record)
    record["provenance"] = dict(record["provenance"])
    record["provenance"]["artifact_path"] = str(path)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def read_record(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# Table-1 summary conversion


def from_table1_summary(
    summary: dict,
    *,
    benchmark: str,
    harness: str,
    split_manifest_id: str,
    split_manifest_version: str,
    split_manifest_hash: str,
    metric_name: str,
    scorer_ref: str,
    run_id: str,
    code_freeze: dict | None = None,
    split_role: str = "deployment",
) -> dict:
    """Convert a summary_table1 record to the shared schema using the supplied manifest hash."""
    resolved = summary.get("resolved_models") or {}
    alias = summary.get("solver_model_alias", "")
    return build_record(
        optimizer=summary.get("optimizer", ""),
        benchmark=benchmark,
        harness=harness,
        solver_model=alias,
        solver_model_resolved=resolved.get(alias),
        split_manifest_id=split_manifest_id,
        split_manifest_version=split_manifest_version,
        split_manifest_hash=split_manifest_hash,
        split_role=split_role,
        metric_name=metric_name,
        metric_value=summary.get("test_accuracy"),
        n_tasks=int(summary.get("test_n", 0) or 0),
        scorer_ref=scorer_ref,
        optimization_cost=PhaseCost.from_legacy_bucket(summary.get("optimization_cost")),
        deployment_cost=PhaseCost.from_legacy_bucket(summary.get("deployment_cost")),
        run_id=run_id,
        code_freeze=dict(code_freeze or {}),
        launch_command=summary.get("launch_command", ""),
        secondary_metrics={
            k: summary.get(k)
            for k in ("truncation_rate", "no_answer_tag_rate")
            if k in summary
        },
        notes=[
            "lifted from a Phase-1 summary_table1.json; no number was recomputed",
        ],
    )
