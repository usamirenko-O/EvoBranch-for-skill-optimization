"""Serialize shared task splits and verify their provenance and content hashes."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from integration.common.paths import WORKSPACE_ROOT

SPLIT_ROLES = ("train", "val", "deployment")
"""The three roles the experiment layer knows about.

`deployment` is the held-out set the final reported number comes from. It is
named `deployment` rather than `test` because the paper table separates
*optimization cost* from *deployment cost*, and using one word for both the
split and the cost phase keeps the artifact readable. Adapters that wrap a
benchmark whose native vocabulary says `test` record that in `role_map`.
"""

_HASH_EXCLUDED_FIELDS = ("content_hash",)


def _portable_source(value):
    """Keep repository-local provenance independent of the checkout location."""
    if isinstance(value, dict):
        return {key: _portable_source(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable_source(item) for item in value]
    if isinstance(value, str) and Path(value).is_absolute():
        try:
            return Path(value).relative_to(WORKSPACE_ROOT).as_posix()
        except ValueError:
            pass
    return value


@dataclass(frozen=True)
class SplitEntry:
    """One task in one split."""

    task_id: str
    native_id: str
    native_split: str = ""
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SplitManifest:
    benchmark: str
    manifest_id: str
    version: str
    source: dict
    role_map: dict
    sampling: dict
    splits: dict
    notes: list = field(default_factory=list)
    content_hash: str = ""

    # ---- construction -------------------------------------------------

    def with_hash(self) -> "SplitManifest":
        """Return a copy whose `content_hash` is filled in."""
        return SplitManifest(
            benchmark=self.benchmark,
            manifest_id=self.manifest_id,
            version=self.version,
            source=_portable_source(self.source),
            role_map=self.role_map,
            sampling=self.sampling,
            splits=self.splits,
            notes=list(self.notes),
            content_hash=self.compute_hash(),
        )

    # ---- accessors ----------------------------------------------------

    def entries(self, role: str) -> list[SplitEntry]:
        if role not in SPLIT_ROLES:
            raise KeyError(f"unknown split role {role!r}; expected one of {SPLIT_ROLES}")
        return list(self.splits[role])

    def task_ids(self, role: str) -> list[str]:
        return [e.task_id for e in self.entries(role)]

    def native_ids(self, role: str) -> list[str]:
        return [e.native_id for e in self.entries(role)]

    def counts(self) -> dict:
        return {role: len(self.splits[role]) for role in SPLIT_ROLES}

    def all_task_ids(self) -> list[str]:
        return [e.task_id for role in SPLIT_ROLES for e in self.splits[role]]

    # ---- serialization ------------------------------------------------

    def to_dict(self) -> dict:
        d = asdict(self)
        d["source"] = _portable_source(d["source"])
        d["splits"] = {role: [asdict(e) for e in self.splits[role]] for role in SPLIT_ROLES}
        d["counts"] = self.counts()
        return d

    def compute_hash(self) -> str:
        payload = self.to_dict()
        for excluded in _HASH_EXCLUDED_FIELDS:
            payload.pop(excluded, None)
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def write(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        stamped = self if self.content_hash else self.with_hash()
        path.write_text(
            json.dumps(stamped.to_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path


def manifest_from_dict(payload: dict) -> SplitManifest:
    splits = {
        role: [SplitEntry(**entry) for entry in payload["splits"][role]] for role in SPLIT_ROLES
    }
    return SplitManifest(
        benchmark=payload["benchmark"],
        manifest_id=payload["manifest_id"],
        version=payload["version"],
        source=payload["source"],
        role_map=payload["role_map"],
        sampling=payload["sampling"],
        splits=splits,
        notes=list(payload.get("notes", [])),
        content_hash=payload.get("content_hash", ""),
    )


def load_manifest(path: Path | str) -> SplitManifest:
    """Load a persisted manifest and verify its own content hash.

    Raises `ValueError` on a hash mismatch rather than returning a manifest
    that silently disagrees with the hash a past run recorded in its result
    artifact -- a stale/edited split file is exactly the failure this hash
    exists to catch.
    """
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest = manifest_from_dict(payload)
    recorded = manifest.content_hash
    recomputed = manifest.compute_hash()
    if recorded and recorded != recomputed:
        raise ValueError(
            f"{path}: content_hash mismatch (recorded {recorded[:12]}..., "
            f"recomputed {recomputed[:12]}...). The manifest file was edited "
            "after it was written; refusing to load it."
        )
    return manifest


def file_digest(path: Path | str) -> str:
    """sha256 of one file's bytes, used for `source.inputs` provenance."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_inputs(paths: list[Path]) -> list[dict]:
    """Build the `source.inputs` provenance list for the given upstream files.

    A file that does not exist is recorded with `"sha256": None` and
    `"present": false` rather than omitted -- a manifest that was built while
    an input was missing must say so, not look like it never needed it.
    """
    out = []
    for p in paths:
        p = Path(p)
        present = p.is_file()
        out.append(
            {
                "path": str(p),
                "present": present,
                "sha256": file_digest(p) if present else None,
            }
        )
    return out
