"""Serialize OURS configuration, search results, provenance, and costs."""
from __future__ import annotations

from pathlib import Path

from integration.experiment.cost import PhaseCost
from integration.experiment.record import (
    SCHEMA_VERSION,
    build_record,
    read_record,
    validate_record,
    write_record,
)

OURS_ARTIFACT_NAME = "ours_experiment_record.json"

_REQUIRED_OURS_FIELDS = (
    "rounds", "beam_m", "k_subsets", "partitioner", "operator",
    "placeholders", "placeholders_in_force", "search_trace",
)


def build_ours_record(
    *,
    optimizer: str,
    method: str,
    benchmark: str,
    harness: str,
    solver_model: str,
    solver_model_resolved: str | None,
    optimizer_model: str,
    optimizer_model_resolved: str | None,
    split_manifest_id: str,
    split_manifest_version: str,
    split_manifest_hash: str,
    split_counts: dict,
    metric_name: str,
    metric_value: float | None,
    n_tasks: int,
    scorer_ref: str,
    optimization_cost: PhaseCost,
    deployment_cost: PhaseCost,
    run_id: str,
    code_freeze: dict,
    launch_command: str,
    ours: dict,
    secondary_metrics: dict | None = None,
    notes: list | None = None,
) -> dict:
    record = build_record(
        optimizer=optimizer,
        benchmark=benchmark,
        harness=harness,
        solver_model=solver_model,
        solver_model_resolved=solver_model_resolved,
        split_manifest_id=split_manifest_id,
        split_manifest_version=split_manifest_version,
        split_manifest_hash=split_manifest_hash,
        split_role="deployment",
        metric_name=metric_name,
        metric_value=metric_value,
        n_tasks=n_tasks,
        scorer_ref=scorer_ref,
        optimization_cost=optimization_cost,
        deployment_cost=deployment_cost,
        run_id=run_id,
        code_freeze=code_freeze,
        launch_command=launch_command,
        secondary_metrics=secondary_metrics,
        notes=list(notes or []) + [
            "OURS run: the deployment number was produced by the benchmark's "
            "official scorer on the deployment split via the shared Direct "
            "harness -- the same code path a native baseline uses.",
            "The optimizer's own accept/reject gate was NOT the search policy; "
            "survivors were chosen by one global Top-M on the fixed validation "
            "split.",
        ],
    )
    record["experiment"]["method"] = method
    record["experiment"]["optimizer_model"] = optimizer_model
    record["experiment"]["optimizer_model_resolved"] = optimizer_model_resolved
    record["experiment"]["split_counts"] = dict(split_counts)
    record["ours"] = dict(ours)
    return record


def validate_ours_record(record) -> list:
    """Native validation, plus the OURS-only requirements. Never raises."""
    errors = list(validate_record(record))
    if not isinstance(record, dict):
        return errors

    exp = record.get("experiment")
    if isinstance(exp, dict):
        if exp.get("method") != "ours":
            errors.append(
                f"experiment.method must be 'ours' in an OURS record, got "
                f"{exp.get('method')!r} -- an OURS run must never be readable "
                "as a native baseline row"
            )
    else:
        errors.append("experiment must be a dict")

    ours = record.get("ours")
    if not isinstance(ours, dict):
        errors.append("ours must be a dict describing the outer search")
        return errors
    for f in _REQUIRED_OURS_FIELDS:
        if f not in ours:
            errors.append(f"ours: missing field {f!r}")
    placeholders = ours.get("placeholders")
    if not isinstance(placeholders, dict):
        errors.append("ours.placeholders must be a dict")
    in_force = ours.get("placeholders_in_force")
    if not isinstance(in_force, list):
        errors.append("ours.placeholders_in_force must be a list (empty once every slot is frozen)")
    elif isinstance(placeholders, dict):
        # every slot still in force must be declared, so a result is never read
        # as if the method were settled when it was not
        missing = [slot for slot in in_force if slot not in placeholders]
        if missing:
            errors.append(f"ours.placeholders does not declare the placeholders in force: {missing}")
    trace = ours.get("search_trace")
    if not isinstance(trace, list):
        errors.append("ours.search_trace must be a list, one entry per round")
    return errors


def write_ours_record(path: Path | str, record: dict) -> Path:
    """Persist an OURS record, refusing an invalid or misnamed one."""
    path = Path(path)
    if path.name != OURS_ARTIFACT_NAME:
        raise ValueError(
            f"an OURS record must be written as {OURS_ARTIFACT_NAME!r}, not "
            f"{path.name!r}: sharing a filename with the native-baseline "
            "artifact would let a matrix glob read an OURS run as a baseline"
        )
    errors = validate_ours_record(record)
    if errors:
        raise ValueError(
            "refusing to write an invalid OURS experiment record:\n  "
            + "\n  ".join(errors)
        )
    return write_record(path, record)


__all__ = [
    "OURS_ARTIFACT_NAME",
    "SCHEMA_VERSION",
    "build_ours_record",
    "read_record",
    "validate_ours_record",
    "write_ours_record",
]
