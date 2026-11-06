"""Materialize LiveMath ID splits from the raw task records."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # workspace root, for `integration.*`

from integration.common.paths import BENCHMARKS_ROOT, SKILLOPT_ROOT, ensure_on_sys_path

ensure_on_sys_path(SKILLOPT_ROOT)

from skillopt.envs.livemathematicianbench.dataloader import load_items  # noqa: E402

MANIFEST_DIR = SKILLOPT_ROOT / "data" / "livemathematicianbench_id_split"
RAW_DATA_ROOT = BENCHMARKS_ROOT / "livemathematicianbench" / "data"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "generated_split"
SPLITS = ("train", "val", "test")


def _load_manifest_ids(split: str) -> list[str]:
    with open(MANIFEST_DIR / split / "items.json", encoding="utf-8") as f:
        manifest_items = json.load(f)
    return [str(item["id"]) for item in manifest_items]


def join_split(*, manifest_dir: Path = MANIFEST_DIR, raw_data_root: Path = RAW_DATA_ROOT) -> dict[str, list[dict]]:
    """Return `{split_name: [normalized_item, ...]}` for train/val/test.

    Raises `KeyError` (via a clear message) if any manifest id is not found
    among the normalized raw items -- a silent partial join would be worse
    than failing loudly here.
    """
    all_items_by_id = {item["id"]: item for item in load_items(str(raw_data_root))}

    result: dict[str, list[dict]] = {}
    for split in SPLITS:
        with open(manifest_dir / split / "items.json", encoding="utf-8") as f:
            manifest_items = json.load(f)
        joined = []
        missing = []
        for manifest_item in manifest_items:
            item_id = str(manifest_item["id"])
            full_item = all_items_by_id.get(item_id)
            if full_item is None:
                missing.append(item_id)
                continue
            joined.append(full_item)
        if missing:
            raise KeyError(
                f"{split}: {len(missing)} manifest id(s) not found in raw data "
                f"under {raw_data_root} (first 5: {missing[:5]}). The manifest "
                "may reference a month not present locally."
            )
        result[split] = joined
    return result


def write_split(out_dir: Path, joined: dict[str, list[dict]]) -> None:
    for split, items in joined.items():
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        with open(split_dir / "items.json", "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--manifest-dir", type=Path, default=MANIFEST_DIR)
    parser.add_argument("--raw-data-root", type=Path, default=RAW_DATA_ROOT)
    args = parser.parse_args()

    joined = join_split(manifest_dir=args.manifest_dir, raw_data_root=args.raw_data_root)
    write_split(args.out_dir, joined)
    counts = {split: len(items) for split, items in joined.items()}
    print(f"Joined LiveMath split written to {args.out_dir}: {counts}")


if __name__ == "__main__":
    main()
