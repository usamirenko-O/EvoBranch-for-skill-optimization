"""Evaluate a saved final skill on the held-out deployment split."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from integration.benchmarks.manifest import load_manifest  # noqa: E402
from integration.benchmarks.registry import get_adapter  # noqa: E402
from integration.experiment.config import DEFAULT_API_KEY, ExperimentConfig  # noqa: E402
from integration.experiment.deployment import run_deployment  # noqa: E402
from integration.harness.registry import build_harness, max_parallel_tasks  # noqa: E402

ARTIFACTS = ("ours_experiment_record.json", "experiment_record.json")


def _base_from_record(record: dict) -> dict:
    """Solver contract of a native run_baseline record (experiment block + --flags of its launch command)."""
    exp = dict(record["experiment"])
    argv = record["provenance"]["launch_command"].split()
    def flag(name, cast, default):
        return cast(argv[argv.index(name) + 1]) if name in argv else default
    exp["proxy_base_url"] = flag("--proxy-base-url", str, "http://localhost:4000/v1")
    exp["max_tokens"] = flag("--max-tokens", int, 4096)
    exp["temperature"] = flag("--temperature", float, 0.0)
    exp["max_retries"] = 2
    return exp


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--deployment-workers", type=int, default=4)
    ap.add_argument("--timeout-seconds", type=float, default=float(os.environ.get("SKILLOPTEVO_SOLVER_TIMEOUT", 1800)))
    ap.add_argument("--out-root", type=Path, default=_ROOT / "integration" / "runs_redeploy")
    ap.add_argument("--reason", default="Re-evaluate the saved final skill.")
    ap.add_argument("--skill-file", type=Path, default=None,
                    help="DIAGNOSTIC: deploy this skill text instead of the run's final_skill.md (e.g. a "
                         "VAL-tied candidate that was not selected). The record is written as "
                         "diagnostic_deployment_record.json so no collector can mistake it for a result.")
    ap.add_argument("--label", default="", help="short tag for the output directory name (with --skill-file)")
    args = ap.parse_args(argv)

    src = args.run_dir.resolve()
    artifact_name = next((n for n in ARTIFACTS if (src / n).exists()), None)
    if artifact_name is None:
        raise SystemExit(f"{src} holds no experiment record")
    original = json.loads((src / artifact_name).read_text(encoding="utf-8"))
    diagnostic = args.skill_file is not None
    skill = (args.skill_file if diagnostic else src / "final_skill.md").read_text(encoding="utf-8")
    manifest = load_manifest(src / "split_manifest.json")
    cfg_name = "ours_run_config.json" if (src / "ours_run_config.json").exists() else "run_config.json"
    if (src / cfg_name).exists():
        run_cfg = json.loads((src / cfg_name).read_text(encoding="utf-8"))
        base = run_cfg.get("base", run_cfg)
    else:   # native baseline runs keep no config file: rebuild it from the record + recorded launch command
        base = _base_from_record(original)
    api_key = base.get("api_key")
    if not api_key or api_key == "***":          # recorded configs redact the key
        api_key = os.environ.get("SKILLOPTEVO_API_KEY") or DEFAULT_API_KEY

    config = ExperimentConfig(
        optimizer=base["optimizer"], benchmark=base["benchmark"], solver_model=base["solver_model"],
        optimizer_model=base.get("optimizer_model", "optimizer-llm"), harness=base.get("harness", "direct"),
        proxy_base_url=base["proxy_base_url"], api_key=api_key,
        temperature=base["temperature"], max_tokens=base["max_tokens"],
        timeout_seconds=args.timeout_seconds, max_retries=base.get("max_retries", 2),
        harness_options=dict(base.get("harness_options") or {}),
    )
    adapter = get_adapter(config.benchmark)
    from ours.experiment.common import harness_options as _harness_options
    options = _harness_options(config, adapter)
    task_ids = manifest.task_ids("deployment")
    if original["experiment"].get("split_manifest_hash") not in (None, manifest.content_hash):
        raise SystemExit("split_manifest.json in run_dir does not match the record's manifest hash")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    tag = f"diag_{args.label or args.skill_file.stem}" if diagnostic else "redeploy"
    out_dir = args.out_root / f"{src.name}__{tag}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    if diagnostic:
        (out_dir / "deployed_skill.md").write_text(skill, encoding="utf-8")
        artifact_name = "diagnostic_deployment_record.json"
        args.reason = f"DIAGNOSTIC deployment of a non-selected skill ({args.skill_file}); not a result"
    else:
        shutil.copy(src / "final_skill.md", out_dir / "final_skill.md")
    manifest.write(out_dir / "split_manifest.json")
    workers = min(int(args.deployment_workers), max_parallel_tasks(adapter.name))
    print(f"redeploy {src.name}: benchmark={config.benchmark} solver={config.solver_model} "
          f"n={len(task_ids)} workers={workers} timeout={args.timeout_seconds}s -> {out_dir}")

    harness = build_harness(adapter.name, work_dir=out_dir / "deployment_harness", options=options)
    t0 = time.time()
    try:
        result = run_deployment(adapter=adapter, harness=harness, task_ids=task_ids, skill_text=skill,
                                solver_config=config.solver_config(), max_workers=workers,
                                transcripts_path=out_dir / "deployment_transcripts.jsonl")
    finally:
        harness.close()
    wall = time.time() - t0

    errors = 0
    with open(out_dir / "deployment_transcripts.jsonl", encoding="utf-8") as f:
        for line in f:
            if (json.loads(line).get("error") or "").strip():
                errors += 1
    (out_dir / "deployment_result.json").write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    record = json.loads(json.dumps(original))
    record["performance"] = {
        "metric_name": result.result.metric_name, "metric_value": result.result.metric_value,
        "n_tasks": result.result.n_tasks, "scorer_ref": result.result.scorer_ref,
        "secondary_metrics": dict(result.result.secondary_metrics),
    }
    record["deployment_cost"] = result.cost.to_dict()
    record["provenance"] = {
        **record.get("provenance", {}),
        "redeploy": {
            "of_run_dir": str(src), "of_run_id": original.get("provenance", {}).get("run_id") or src.name,
            "reason": args.reason, "timestamp_utc": stamp,
            "original_performance": original["performance"],
            "original_deployment_cost": original.get("deployment_cost"),
            "timeout_seconds": args.timeout_seconds, "deployment_workers": workers,
            "deployment_error_tasks": errors,
            "unchanged": ["final_skill", "split_manifest", "solver alias", "temperature", "max_tokens",
                          "harness", "scorer", "optimization_cost", "search trace"],
        },
        "deployment_wall_time_s": round(wall, 2), "run_dir": str(out_dir),
    }
    (out_dir / artifact_name).write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str),
                                         encoding="utf-8")
    print(f"{record['performance']['metric_name']} = {record['performance']['metric_value']:.4f} over "
          f"n={record['performance']['n_tasks']} | errors={errors} | original={original['performance']['metric_value']:.4f} "
          f"| deployment {record['deployment_cost'].get('llm_calls')} calls | {wall:.0f}s")
    print(f"artifact: {out_dir / artifact_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
