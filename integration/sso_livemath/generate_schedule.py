"""Generate SSO training, history, and validation schedules from shared splits."""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from integration.experiment.runners.sso import select_history_task_ids
from integration.sso_livemath.livemath_loader import canonical_split_task_ids


def generate_schedule(
    *, steps: int | None, train_per_step: int, history_per_step: int,
    val_per_step: int, seed: int,
) -> dict:
    ids = canonical_split_task_ids()
    train_ids = list(ids["train"])
    val_ids = list(ids["val"])
    if val_per_step > len(val_ids):
        raise ValueError(f"--val-per-step={val_per_step} exceeds D_val size={len(val_ids)}")

    rng = random.Random(seed)
    shuffled_train = list(train_ids)
    rng.shuffle(shuffled_train)

    n_chunks = steps if steps is not None else -(-len(shuffled_train) // train_per_step)  # ceil div
    schedule_steps = []
    for i in range(n_chunks):
        start = (i * train_per_step) % len(shuffled_train)
        # Wrap around if the caller asked for more steps than one full pass
        # over D_train covers -- deterministic, not a fresh random draw.
        batch = [shuffled_train[(start + j) % len(shuffled_train)] for j in range(train_per_step)]
        # Guard against duplicate ids within one batch when train_per_step
        # is close to (or exceeds) len(shuffled_train) and wrap-around would
        # otherwise repeat an id inside the SAME step's training_task_ids.
        seen = set()
        deduped = [t for t in batch if not (t in seen or seen.add(t))]
        history = select_history_task_ids(
            deduped, history_per_step, rng_key=f"sso-history:{seed}:step{i}",
        )
        val_sample = rng.sample(val_ids, val_per_step) if val_per_step < len(val_ids) else list(val_ids)
        schedule_steps.append({
            "training_task_ids": deduped,
            "history_task_ids": history,
            "validation_task_ids": val_sample,
        })

    return {
        "generation_params": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "steps": n_chunks,
            "train_per_step": train_per_step,
            "history_per_step": history_per_step,
            "val_per_step": val_per_step,
            "source": "integration/sso_livemath/generate_schedule.py",
            "note": "REPRODUCTION_CHOICE -- sampling policy lives in this integration-layer "
                    "script, not inside SSO/. Not an official experimental protocol until a "
                    "human reviews and (optionally) freezes this specific output file.",
        },
        "steps": schedule_steps,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=None,
                    help="Number of steps T. Default: ceil(35 / --train-per-step), one full pass over D_train.")
    p.add_argument("--train-per-step", type=int, default=5)
    p.add_argument("--history-per-step", type=int, default=2)
    p.add_argument("--val-per-step", type=int, default=17)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    schedule = generate_schedule(
        steps=args.steps, train_per_step=args.train_per_step,
        history_per_step=args.history_per_step, val_per_step=args.val_per_step,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(schedule, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(schedule['steps'])}-step schedule to {args.out}")
    print(json.dumps(schedule["generation_params"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
