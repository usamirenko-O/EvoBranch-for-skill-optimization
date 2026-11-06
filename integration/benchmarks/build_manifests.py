"""Build shared split manifests from local data; use --check to verify saved files."""
from __future__ import annotations

import argparse
import json
import sys

from integration.benchmarks import (
    alfworld,
    chemllmbench,
    livecodebench,
    livemath,
    officeqa,
    pubmedqa,
    spreadsheetbench,
)
from integration.benchmarks.manifest import SplitManifest
from integration.benchmarks.paths import MANIFEST_DIR

BUILDERS = {
    livemath.MANIFEST_ID: livemath.build_manifest,
    officeqa.MANIFEST_ID: officeqa.build_manifest,
    spreadsheetbench.MANIFEST_ID: spreadsheetbench.build_manifest,
    alfworld.MANIFEST_ID: alfworld.build_manifest,
    livecodebench.MANIFEST_ID: livecodebench.build_manifest,
    pubmedqa.MANIFEST_ID: pubmedqa.build_manifest,
    chemllmbench.MANIFEST_ID: chemllmbench.build_manifest,
}


def build_all() -> dict:
    return {manifest_id: builder() for manifest_id, builder in BUILDERS.items()}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true", help="verify persisted manifests match a fresh build")
    args = p.parse_args()

    failures = []
    for manifest_id, manifest in build_all().items():
        path = MANIFEST_DIR / f"{manifest_id}.json"
        if args.check:
            if not path.exists():
                failures.append(f"{manifest_id}: not persisted at {path}")
                continue
            on_disk = json.loads(path.read_text(encoding="utf-8"))
            if on_disk.get("content_hash") != manifest.content_hash:
                failures.append(
                    f"{manifest_id}: persisted hash {on_disk.get('content_hash', '')[:12]}... "
                    f"!= rebuilt {manifest.content_hash[:12]}..."
                )
            else:
                print(f"[ok]    {manifest_id}  {manifest.content_hash[:12]}  {manifest.counts()}")
        else:
            manifest.write(path)
            print(f"[write] {manifest_id}  {manifest.content_hash[:12]}  {manifest.counts()}  -> {path}")

    for f in failures:
        print(f"[FAIL]  {f}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
