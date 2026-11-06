"""Build a deterministic source ZIP without local data, credentials, or run output."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("ours", "integration", "main_experiment", "ablation", "SSO", "SkillGen", "SkillOpt", "configs", "scripts")
SOURCE_FILES = ("README.md", "requirements.txt", "requirements-services.txt", ".gitignore", "benchmarks/README.md", "benchmarks/assets.json")
EXCLUDED_DIRS = {
    ".git", ".agents", ".aws", ".claude", ".codex", ".vscode", ".idea",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".cache",
    ".venv", "venv", "tests", "outputs", "mini_train_output", "smoke_test_output",
    "generated_dataset", "generated_split", "mini_split", "_superseded", "skill_output",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".log", ".bak", ".tmp", ".pem", ".key"}


def include(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if any(part in EXCLUDED_DIRS or part.endswith(".egg-info") for part in rel.parts):
        return False
    if rel.parts[:2] == ("SkillGen", "data"):
        return False
    if rel.parts[0] == "integration" and len(rel.parts) > 1 and rel.parts[1].startswith("runs"):
        return False
    if path.name == ".DS_Store" or path.suffix in EXCLUDED_SUFFIXES:
        return False
    return not (path.name.startswith(".env") and path.name != ".env.example")


def source_files() -> list[Path]:
    files = [ROOT / name for name in SOURCE_FILES]
    for directory in SOURCE_DIRS:
        for current, dirs, names in os.walk(ROOT / directory, followlinks=False):
            for name in dirs + names:
                path = Path(current) / name
                if include(path) and path.is_symlink():
                    raise ValueError(f"Source symlink must be reviewed: {path.relative_to(ROOT)}")
            dirs[:] = [name for name in dirs if include(Path(current) / name)]
            files.extend(Path(current) / name for name in names if include(Path(current) / name))
    return sorted(files, key=lambda path: path.relative_to(ROOT).as_posix())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "dist/evobranch-anonymous-source.zip")
    args = parser.parse_args()
    files = source_files()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            info = zipfile.ZipInfo(f"evobranch/{relative}", date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (0o100755 if path.suffix == ".sh" else 0o100644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    with args.out.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    print(f"{len(files)} files; {args.out.stat().st_size:,} bytes; sha256 {digest}")
    print(args.out)


if __name__ == "__main__":
    main()
