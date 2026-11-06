"""Create a shared split manifest from a user-supplied task specification."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from integration.benchmarks.registry import BENCHMARK_NAMES, get_adapter
from integration.benchmarks.user_manifest import (
    manifest_to_user_spec,
    sample_manifest,
    validate_manifest,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", required=True, help=f"one of: {', '.join(BENCHMARK_NAMES)}")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--from-default", action="store_true",
                   help="export the checked-in default manifest as an editable user spec")
    p.add_argument("--train", type=int, default=0)
    p.add_argument("--val", type=int, default=0)
    p.add_argument("--deployment", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--manifest-id", default=None)
    p.add_argument("--restrict-prefix", default=None,
                   help="only sample task ids starting with this prefix "
                        "(e.g. 'valid_unseen/' for ALFWorld)")
    args = p.parse_args()

    adapter = get_adapter(args.benchmark)
    if args.from_default:
        manifest = adapter.manifest()
    else:
        if not (args.train and args.deployment):
            p.error("--train and --deployment must be > 0 (or use --from-default)")
        restrict = None
        if args.restrict_prefix:
            restrict = [t for t in adapter.population() if t.startswith(args.restrict_prefix)]
        manifest = sample_manifest(
            args.benchmark,
            n_train=args.train,
            n_val=args.val,
            n_deployment=args.deployment,
            seed=args.seed,
            manifest_id=args.manifest_id,
            restrict_to=restrict,
        )

    errors = validate_manifest(manifest)
    if errors:
        for e in errors:
            print(f"[FAIL] {e}")
        return 1

    spec = manifest_to_user_spec(manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"[write] {args.out}  benchmark={manifest.benchmark} "
        f"train={len(spec['train_ids'])} val={len(spec['validation_ids'])} "
        f"deployment={len(spec['deployment_ids'])} hash={manifest.content_hash[:12]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
