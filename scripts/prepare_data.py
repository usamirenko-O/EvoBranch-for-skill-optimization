"""Download, import, or verify the four main benchmark snapshots."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="verify local files (default)")
    mode.add_argument("--download", action="store_true", help="download missing files from upstream")
    mode.add_argument("--from", dest="source", type=Path, help="import from an existing benchmarks directory")
    args = parser.parse_args()
    assets = json.loads((ROOT / "benchmarks/assets.json").read_text())["files"]
    failed = 0
    for asset in assets:
        relative = Path(asset["path"])
        target = ROOT / "benchmarks" / relative
        temporary = None
        try:
            if not target.exists() and (args.download or args.source):
                target.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".tmp", delete=False) as out:
                    temporary = Path(out.name)
                    if args.source:
                        with (args.source / relative).open("rb") as src:
                            shutil.copyfileobj(src, out)
                    else:
                        request = urllib.request.Request(asset["url"], headers={"User-Agent": "EvoBranch-data"})
                        with urllib.request.urlopen(request, timeout=120) as src:
                            shutil.copyfileobj(src, out)
                if sha256(temporary) != asset["sha256"]:
                    raise ValueError("source checksum differs from the fixed snapshot")
                temporary.replace(target)
            if not target.is_file():
                raise ValueError("missing; use --download or --from BENCHMARKS_DIR")
            if sha256(target) != asset["sha256"]:
                raise ValueError("checksum mismatch; existing file was left unchanged")
            print(f"OK {relative}")
        except (OSError, ValueError) as error:
            failed += 1
            print(f"FAIL {relative}: {error}")
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    print(f"{len(assets) - failed}/{len(assets)} benchmark assets verified")
    return bool(failed)


if __name__ == "__main__":
    raise SystemExit(main())
