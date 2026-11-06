"""Convert shared LiveMath splits to SkillGen dataset files."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from integration.skillopt_livemath.join_livemath_split import join_split  # noqa: E402

DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "generated_dataset"


def _to_skillgen_instance(item: dict) -> dict:
    choices = [{"label": c["label"], "text": c["text"]} for c in item["choices"]]
    prompt = _render_from_dict(item)
    return {
        "instance_id": item["id"],
        "input": prompt,
        "ground_truth": item["correct_choice"]["label"],
        "metadata": {
            "benchmark": "livemath",
            "month": item.get("month", ""),
            "no": item.get("no"),
            "choices": choices,
            "correct_choice_label": item["correct_choice"]["label"],
            "correct_choice_text": item["correct_choice"].get("text", ""),
            "paper_link": item.get("paper_link", ""),
        },
    }


def _render_from_dict(item: dict) -> str:
    """Render straight from SkillOpt's own dict item shape (render_prompt in
    common/livemath_data.py takes a LiveMathItem dataclass, not this dict --
    duplicated here as a 3-line formatting shim rather than round-tripping
    dict -> dataclass -> dict for no benefit)."""
    choice_lines = "\n".join(f"{c['label']}. {c['text']}" for c in item["choices"])
    return (
        f"## Question\n{item['question']}\n\n"
        f"## Choices\n{choice_lines}\n\n"
        "Think step by step, then respond with your final choice as "
        "<answer>LABEL</answer>, where LABEL is a single choice letter."
    )


def build_dataset(split: str) -> dict:
    joined = join_split()
    if split not in joined:
        raise ValueError(f"split must be one of {list(joined)}, got {split!r}")
    instances = [_to_skillgen_instance(item) for item in joined[split]]
    return {
        "dataset_id": f"livemathematicianbench_{split}",
        "task_name": "LiveMathematicianBench",
        "task_type": "binary",
        "instances": instances,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    dataset = build_dataset(args.split)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.split}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(dataset['instances'])} instances to {out_path}")


if __name__ == "__main__":
    main()
