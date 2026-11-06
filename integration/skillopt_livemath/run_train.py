"""Launch native SkillOpt training on shared LiveMath splits."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
_SKILLOPT_ROOT = _WORKSPACE_ROOT / "SkillOpt"
_DEFAULT_SKILL_INIT = (
    _SKILLOPT_ROOT / "skillopt" / "envs" / "livemathematicianbench" / "skills" / "initial.md"
)
_CONFIG_OVERRIDE = _WORKSPACE_ROOT / "integration" / "skillopt_livemath" / "config.override.yaml"

sys.path.insert(0, str(_WORKSPACE_ROOT))

from integration.common.resolve_models import resolve_model_aliases  # noqa: E402
from integration.common.table1_schema import validate_table1_summary  # noqa: E402


def _git(args: list[str], cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False,
    ).stdout.strip()


def _split_manifest_hash(split_dir: Path) -> str:
    h = hashlib.sha256()
    for part in ("train", "val", "test"):
        h.update((split_dir / part / "items.json").read_bytes())
    return h.hexdigest()


def _split_counts(split_dir: Path) -> dict[str, int]:
    counts = {}
    for part in ("train", "val", "test"):
        with open(split_dir / part / "items.json") as f:
            counts[part] = len(json.load(f))
    return counts


def _iter_result_lines(results_jsonl: Path):
    if not results_jsonl.exists():
        return
    with open(results_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _truncation_and_answer_tag_rates(results_jsonl: Path) -> tuple[float | None, float | None, int]:
    finish_reasons: list[str | None] = []
    answer_tags: list[bool] = []
    for rec in _iter_result_lines(results_jsonl):
        if "finish_reason" in rec:
            finish_reasons.append(rec["finish_reason"])
        if "answer_tag_found" in rec:
            answer_tags.append(bool(rec["answer_tag_found"]))
    n = len(finish_reasons)
    truncation_rate = (
        sum(1 for r in finish_reasons if r == "length") / n if n else None
    )
    n_tags = len(answer_tags)
    no_answer_tag_rate = (
        sum(1 for found in answer_tags if not found) / n_tags if n_tags else None
    )
    return truncation_rate, no_answer_tag_rate, n


def _cost_bucket(summary_dict: dict | None) -> dict:
    total = (summary_dict or {}).get("_total", {}) if summary_dict else {}
    return {
        "calls": total.get("calls", 0),
        "prompt_tokens": total.get("prompt_tokens", 0),
        "completion_tokens": total.get("completion_tokens", 0),
        "cached_tokens": total.get("cached_tokens", 0),
    }


def _reconcile_cost_buckets(summary: dict) -> bool:
    "Check that optimization, deployment, and diagnostic usage sum to the reported total."
    fields = ("calls", "prompt_tokens", "completion_tokens", "cached_tokens", "total_tokens")
    grand = (summary.get("token_summary") or {}).get("_total")
    if grand is None:
        return True  # nothing to reconcile (eval_test was off)
    buckets = [
        summary.get("optimization_token_summary"),
        summary.get("baseline_test_token_summary_diagnostic_only"),
        summary.get("deployment_token_summary"),
        summary.get("final_skill_test_token_summary_diagnostic_only"),
    ]
    reconciled = {f: sum((b or {}).get("_total", {}).get(f, 0) for b in buckets) for f in fields}
    return all(reconciled[f] == grand.get(f, 0) for f in fields)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split-dir", required=True, type=Path,
                    help="Frozen split dir with train/val/test/items.json (e.g. mini_split or generated_split)")
    p.add_argument("--out-root", required=True, type=Path)
    p.add_argument("--target-model-alias", required=True)
    p.add_argument("--optimizer-model-alias", required=True)
    p.add_argument("--target-max-tokens", required=True, type=int,
                    help="Required maximum output tokens for solver calls.")
    p.add_argument("--optimizer-max-tokens", required=True, type=int)
    p.add_argument("--proxy-base-url", default="http://localhost:4000/v1")
    p.add_argument("--skill-init", type=Path, default=_DEFAULT_SKILL_INIT)
    p.add_argument("--num-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=None,
                    help="Must be <= the train split size. Defaults to the train split size itself.")
    p.add_argument("--extra-cfg-options", nargs="*", default=[])
    args = p.parse_args()

    # ---- Fail-loud preflight (before spending any money) ----------------
    if not args.skill_init.exists():
        raise SystemExit(
            f"--skill-init file does not exist: {args.skill_init}"
            "\nAn initial skill file is required."
        )
    for part in ("train", "val", "test"):
        f = args.split_dir / part / "items.json"
        if not f.exists():
            raise SystemExit(f"split dir missing {f}")

    counts = _split_counts(args.split_dir)
    batch_size = args.batch_size or counts["train"]
    if batch_size > counts["train"]:
        raise SystemExit(
            f"--batch-size={batch_size} exceeds train split size={counts['train']}"
        )

    args.out_root.mkdir(parents=True, exist_ok=True)

    skill_init_abs = str(args.skill_init.resolve())
    split_dir_abs = str(args.split_dir.resolve())

    env = os.environ.copy()
    env["TARGET_OPENAI_COMPATIBLE_BASE_URL"] = args.proxy_base_url
    env["TARGET_OPENAI_COMPATIBLE_MODEL"] = args.target_model_alias
    env["TARGET_OPENAI_COMPATIBLE_MAX_TOKENS"] = str(args.target_max_tokens)
    env["TARGET_OPENAI_COMPATIBLE_API_KEY"] = env.get("TARGET_OPENAI_COMPATIBLE_API_KEY", "dummy")
    env["OPTIMIZER_OPENAI_COMPATIBLE_BASE_URL"] = args.proxy_base_url
    env["OPTIMIZER_OPENAI_COMPATIBLE_MODEL"] = args.optimizer_model_alias
    env["OPTIMIZER_OPENAI_COMPATIBLE_MAX_TOKENS"] = str(args.optimizer_max_tokens)
    env["OPTIMIZER_OPENAI_COMPATIBLE_API_KEY"] = env.get("OPTIMIZER_OPENAI_COMPATIBLE_API_KEY", "dummy")

    cfg_options = [
        f"env.split_dir={split_dir_abs}",
        f"env.skill_init={skill_init_abs}",
        f"model.target={args.target_model_alias}",
        f"model.optimizer={args.optimizer_model_alias}",
        f"train.batch_size={batch_size}",
        f"train.num_epochs={args.num_epochs}",
    ] + list(args.extra_cfg_options)

    argv = [
        sys.executable, str(_SKILLOPT_ROOT / "scripts" / "train.py"),
        "--config", str(_CONFIG_OVERRIDE),
        "--cfg-options", *cfg_options,
        "--out_root", str(args.out_root.resolve()),
    ]

    launch_record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "argv": argv,
        "cfg_options": cfg_options,
        "env_redacted": {
            # "API_KEY"/"SECRET", not the bare substring "KEY"/"TOKEN" --
            # the latter false-positive-matched *_MAX_TOKENS (a plain
            # non-secret integer), redacting genuinely useful debug info.
            k: ("***" if "API_KEY" in k or "SECRET" in k else v)
            for k, v in env.items()
            if k.startswith(("TARGET_", "OPTIMIZER_", "VERTEX_"))
        },
        "skillopt_git_head": _git(["rev-parse", "HEAD"], _SKILLOPT_ROOT),
        "skillopt_git_diff_stat": _git(["diff", "--stat"], _SKILLOPT_ROOT),
        "split_counts": counts,
        "split_manifest_hash": _split_manifest_hash(args.split_dir),
    }
    (args.out_root / "launch.json").write_text(
        json.dumps(launch_record, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    print("[run_train] launch.json written. Resolved config:")
    print(f"  target_model_alias={args.target_model_alias} max_tokens={args.target_max_tokens}")
    print(f"  optimizer_model_alias={args.optimizer_model_alias} max_tokens={args.optimizer_max_tokens}")
    print(f"  split_dir={split_dir_abs} counts={counts}")
    print(f"  skill_init={skill_init_abs}")
    print(f"  argv={' '.join(argv)}")

    t0 = time.time()
    proc = subprocess.run(argv, cwd=_WORKSPACE_ROOT, env=env)
    wall_time_s = time.time() - t0
    if proc.returncode != 0:
        raise SystemExit(f"SkillOpt/scripts/train.py exited {proc.returncode}")

    # ---- Post-hoc Table-1 summary (never modifies trainer.py's own summary.json) ----
    summary_path = args.out_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    resolved_models = resolve_model_aliases(
        args.proxy_base_url, [args.target_model_alias, args.optimizer_model_alias],
    )

    deployment_results = args.out_root / "test_eval" / "results.jsonl"
    truncation_rate, no_answer_tag_rate, n_rated = _truncation_and_answer_tag_rates(deployment_results)

    best_skill_path = args.out_root / "best_skill.md"
    initial_skill_path = args.out_root / "skills" / "skill_v0000.md"
    skill_accepted = (
        best_skill_path.exists()
        and initial_skill_path.exists()
        and best_skill_path.read_bytes() != initial_skill_path.read_bytes()
    )

    cost_reconciliation_failed = not _reconcile_cost_buckets(summary)

    table1_summary = {
        "optimizer": "skillopt",
        "solver_model_alias": args.target_model_alias,
        "resolved_models": resolved_models,
        "split_manifest_hash": launch_record["split_manifest_hash"],
        "n_train": counts["train"], "n_val": counts["val"], "n_test": counts["test"],
        "test_accuracy": summary.get("test_hard"),
        "test_n": counts["test"],
        "optimization_cost": _cost_bucket(summary.get("optimization_token_summary")),
        "deployment_cost": _cost_bucket(summary.get("deployment_token_summary")),
        "truncation_rate": truncation_rate,
        "no_answer_tag_rate": no_answer_tag_rate,
        "final_skill_path": str(best_skill_path) if best_skill_path.exists() else None,
        "skill_accepted": skill_accepted,
        "launch_command": " ".join(argv),
        "wall_time_s": round(wall_time_s, 1),
        "optimizer_specific": {
            "best_step": summary.get("best_step"),
            "best_selection_hard": summary.get("best_selection_hard"),
            "baseline_selection_hard": summary.get("baseline_selection_hard"),
            "cost_reconciliation_failed": cost_reconciliation_failed,
            "truncation_rate_sample_size": n_rated,
        },
    }
    schema_errors = validate_table1_summary(table1_summary)
    if schema_errors:
        table1_summary["_table1_schema_errors"] = schema_errors

    (args.out_root / "summary_table1.json").write_text(
        json.dumps(table1_summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    print("\n[run_train] summary_table1.json:")
    print(json.dumps(table1_summary, indent=2, ensure_ascii=False))
    if schema_errors:
        print(f"\n[run_train] WARNING: table1 schema errors: {schema_errors}")
    if cost_reconciliation_failed:
        print("\n[run_train] WARNING: cost_reconciliation_failed=true -- the four cost "
              "buckets do not sum to trainer.py's own grand total.")


if __name__ == "__main__":
    main()
