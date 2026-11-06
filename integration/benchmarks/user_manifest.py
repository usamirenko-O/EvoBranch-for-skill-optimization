"""Validate user-specified train, validation, and deployment task lists."""
from __future__ import annotations

import json
import random
from pathlib import Path

from integration.benchmarks.manifest import (
    SPLIT_ROLES,
    SplitEntry,
    SplitManifest,
    load_manifest,
    manifest_from_dict,
)
from integration.benchmarks.registry import canonical_benchmark_name, get_adapter

_ROLE_KEYS = {
    "train": ("train_ids", "train"),
    "val": ("validation_ids", "val_ids", "validation", "val"),
    "deployment": ("deployment_ids", "test_ids", "deployment", "test"),
}


class ManifestValidationError(ValueError):
    """Raised before any execution when a manifest cannot govern a run."""


def _role_ids(spec: dict, role: str) -> list:
    for key in _ROLE_KEYS[role]:
        if key in spec:
            value = spec[key]
            if not isinstance(value, list):
                raise ManifestValidationError(f"{key!r} must be a list of task ids")
            return [str(v) for v in value]
    raise ManifestValidationError(
        f"manifest is missing the {role} split (expected one of {_ROLE_KEYS[role]})"
    )


def _sampling(spec: dict, benchmark: str) -> dict:
    """The `sampling` block, plus any benchmark-specific presentation knob.

    LiveMath needs one: its raw corpus lists the correct option first in every
    item, so the choices are deterministically shuffled and the seed has to
    travel with the manifest -- otherwise two runs of "the same split" could
    present different questions. Defaults to the shared protocol value.
    """
    sampling = {
        "method": str(spec.get("sampling_method") or "user_specified"),
        "seed": spec.get("seed"),
    }
    if benchmark == "livemathematicianbench":
        from integration.benchmarks.livemath_choices import DEFAULT_CHOICE_SHUFFLE_SEED

        seed = spec.get("choice_shuffle_seed")
        sampling["choice_shuffle_seed"] = int(
            DEFAULT_CHOICE_SHUFFLE_SEED if seed is None else seed
        )
    return sampling


def build_user_manifest(spec: dict, *, source_path: str | None = None) -> SplitManifest:  # noqa: ARG001
    """Turn a user spec into a content-hashed `SplitManifest`. No validation
    against the benchmark population happens here -- that is
    `validate_manifest`, which the launcher calls with the live adapter."""
    if "benchmark" not in spec:
        raise ManifestValidationError("manifest must name a 'benchmark'")
    benchmark = canonical_benchmark_name(str(spec["benchmark"]))
    manifest_id = str(spec.get("manifest_id") or "user_manifest")
    splits = {
        role: [
            SplitEntry(task_id=task_id, native_id=task_id, native_split="", extra={})
            for task_id in _role_ids(spec, role)
        ]
        for role in SPLIT_ROLES
    }
    return SplitManifest(
        benchmark=benchmark,
        manifest_id=manifest_id,
        version=str(spec.get("version", "1")),
        source={
            "kind": "user_specified",
            # Deliberately NOT the file path: a manifest copied elsewhere must
            # hash the same, or the hash stops being a stable split identity.
            # Where the file lived is recorded in the run artifact's
            # provenance.split_manifest_path instead.
            "inputs": [],
            "official_scorer": {"ref": get_adapter(benchmark).scorer_ref},
        },
        role_map={
            "train": "user-specified train_ids",
            "val": "user-specified validation_ids",
            "deployment": "user-specified deployment_ids",
            "native_splits": "not applicable -- the user's partition is authoritative",
        },
        sampling=_sampling(spec, benchmark),
        splits=splits,
        notes=list(spec.get("notes") or []),
    ).with_hash()


def load_split_manifest(path: Path | str) -> SplitManifest:
    """Load either a user spec or a fully-expanded manifest document."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "splits" in payload:
        # Fully-expanded document: verify its own hash on the way in.
        return load_manifest(path)
    return build_user_manifest(payload, source_path=str(path))


def validate_manifest(manifest: SplitManifest, *, benchmark: str | None = None) -> list:
    """Return a list of blocking errors; empty list == safe to execute."""
    errors: list = []
    if benchmark and canonical_benchmark_name(benchmark) != manifest.benchmark:
        errors.append(
            f"manifest is for benchmark {manifest.benchmark!r} but the run requested "
            f"{canonical_benchmark_name(benchmark)!r}"
        )
        return errors

    adapter = get_adapter(manifest.benchmark)
    population = set(adapter.population())
    if not population:
        errors.append(
            f"{manifest.benchmark}: benchmark population is empty -- its local resources "
            "are missing, so no manifest can be validated against it"
        )
        return errors

    seen: dict = {}
    for role in SPLIT_ROLES:
        ids = manifest.task_ids(role)
        if not ids:
            errors.append(f"{role} split is empty; all three splits must be non-empty")
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            errors.append(f"{role} split repeats task id(s): {duplicates[:5]}")
        unknown = sorted(set(ids) - population)
        if unknown:
            errors.append(
                f"{role} split names {len(unknown)} task id(s) not in the "
                f"{manifest.benchmark} population (first 5: {unknown[:5]})"
            )
        for task_id in set(ids):
            if task_id in seen:
                errors.append(
                    f"task id {task_id!r} appears in both {seen[task_id]!r} and {role!r}; "
                    "splits must be disjoint"
                )
            else:
                seen[task_id] = role
    return errors


def sample_manifest(
    benchmark: str,
    *,
    n_train: int,
    n_val: int,
    n_deployment: int,
    seed: int,
    manifest_id: str | None = None,
    restrict_to: list | None = None,
) -> SplitManifest:
    """Build a reproducible random manifest from the benchmark population.

    Provided for convenience (and for the smoke tests); a real experiment may
    equally well hand-write its three id lists. The seed and the population
    size are recorded in `sampling`, so the partition can be reproduced.
    """
    adapter = get_adapter(benchmark)
    population = adapter.population()
    if restrict_to is not None:
        allowed = set(str(t) for t in restrict_to)
        population = [t for t in population if t in allowed]
    total = n_train + n_val + n_deployment
    if total > len(population):
        raise ManifestValidationError(
            f"{adapter.name}: requested {total} tasks but the population has {len(population)}"
        )
    rng = random.Random(seed)
    chosen = rng.sample(sorted(population), total)
    spec = {
        "benchmark": adapter.name,
        "manifest_id": manifest_id or f"{adapter.name}_sampled_{n_train}_{n_val}_{n_deployment}_s{seed}",
        "version": "1",
        "train_ids": sorted(chosen[:n_train]),
        "validation_ids": sorted(chosen[n_train : n_train + n_val]),
        "deployment_ids": sorted(chosen[n_train + n_val :]),
        "sampling_method": "random_sample_from_population",
        "seed": seed,
        "notes": [
            f"sampled from a population of {len(population)} tasks with seed {seed}",
        ],
    }
    return build_user_manifest(spec)


def manifest_to_user_spec(manifest: SplitManifest) -> dict:
    """Render a manifest back into the small user-facing spec shape."""
    spec = {
        "benchmark": manifest.benchmark,
        "manifest_id": manifest.manifest_id,
        "version": manifest.version,
        "train_ids": manifest.task_ids("train"),
        "validation_ids": manifest.task_ids("val"),
        "deployment_ids": manifest.task_ids("deployment"),
        "sampling_method": manifest.sampling.get("method"),
        "seed": manifest.sampling.get("seed"),
        "notes": list(manifest.notes),
    }
    if "choice_shuffle_seed" in (manifest.sampling or {}):
        spec["choice_shuffle_seed"] = manifest.sampling["choice_shuffle_seed"]
    return spec


__all__ = [
    "ManifestValidationError",
    "build_user_manifest",
    "load_split_manifest",
    "manifest_to_user_spec",
    "sample_manifest",
    "validate_manifest",
]
