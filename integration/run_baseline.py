"""Configure and run a native optimizer baseline."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

from integration.benchmarks.manifest import SPLIT_ROLES  # noqa: E402
from integration.benchmarks.registry import BENCHMARK_NAMES, OPTIMIZERS  # noqa: E402
from integration.experiment.config import (  # noqa: E402
    DEFAULT_OPTIMIZER_MODEL,
    DEFAULT_OUT_ROOT,
    ExperimentConfig,
    code_freeze_reference,
    load_config,
)
from integration.experiment.execute import (  # noqa: E402
    PreflightError,
    execute,
    preflight,
    resolve_manifest,
)
from integration.experiment.native_paths import NATIVE_CAVEATS  # noqa: E402
from integration.harness.registry import HARNESS_KIND  # noqa: E402


def build_plan(config: ExperimentConfig) -> dict:
    """Everything a dry-run reports, as data (so tests can assert on it)."""
    adapter = config.adapter()
    manifest = resolve_manifest(config)
    resources = adapter.resource_status()
    caveat = NATIVE_CAVEATS.get((config.optimizer, config.benchmark), "")
    blockers = preflight(config, manifest)

    plan = {
        "experiment": {
            "optimizer": config.optimizer,
            "benchmark": config.benchmark,
            "harness": config.harness,
            "harness_kind": HARNESS_KIND,
            "solver_model": config.solver_model,
            "optimizer_model": config.optimizer_model,
        },
        "split_manifest": {
            "manifest_id": manifest.manifest_id,
            "version": manifest.version,
            "source": config.split_manifest_path or "benchmark default",
            "content_hash": manifest.content_hash,
            "counts": manifest.counts(),
            "sampling": manifest.sampling,
            "role_map": manifest.role_map,
        },
        "scorer": {
            "metric_name": adapter.metric_name,
            "scorer_ref": adapter.scorer_ref,
        },
        "solver_call_contract": {
            "base_url": config.proxy_base_url,
            "model_alias": config.solver_model,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "timeout_seconds": config.timeout_seconds,
            "max_retries": config.max_retries,
        },
        "output": {
            "run_dir": str(config.run_dir()),
            "artifact_path": str(config.artifact_path()),
        },
        "readiness": {
            "benchmark_resources_ready": resources.ready,
            "benchmark_degraded": resources.degraded,
            "benchmark_details": resources.details,
            "comparability_caveat": caveat,
            "blockers": blockers,
            "can_execute": not blockers,
        },
        "manifest_notes": list(manifest.notes),
    }
    if config.optimizer == "sso" and not blockers:
        from integration.experiment.runners.sso import planned_training_schedule

        plan["sso_training_schedule"] = planned_training_schedule(config, manifest)
    return plan


def print_plan(plan: dict) -> None:
    e, m, s, c, r = (
        plan["experiment"],
        plan["split_manifest"],
        plan["scorer"],
        plan["solver_call_contract"],
        plan["readiness"],
    )
    print("=" * 74)
    print("DRY RUN -- no LLM call, no network call, no subprocess")
    print("=" * 74)
    print(f"  optimizer         : {e['optimizer']}")
    print(f"  benchmark         : {e['benchmark']}")
    print(f"  harness           : {e['harness']} ({e['harness_kind']})")
    print(f"  solver model      : {e['solver_model']}  (proxy alias @ {c['base_url']})")
    print(f"  optimizer model   : {e['optimizer_model'] or '(not set)'}")
    print()
    print(f"  split manifest    : {m['manifest_id']} v{m['version']}")
    print(f"    source          : {m['source']}")
    print(f"    content hash    : {m['content_hash']}")
    print(f"    counts          : train={m['counts']['train']} val={m['counts']['val']} "
          f"deployment={m['counts']['deployment']}")
    print(f"    sampling        : {m['sampling'].get('method')} seed={m['sampling'].get('seed')}")
    print()
    print(f"  scorer            : {s['scorer_ref']}")
    print(f"  metric            : {s['metric_name']}")
    print(f"  solver contract   : temperature={c['temperature']} max_tokens={c['max_tokens']} "
          f"timeout={c['timeout_seconds']}s retries={c['max_retries']}")
    print()
    print(f"  output artifact   : {plan['output']['artifact_path']}")
    if "sso_training_schedule" in plan:
        print_sso_training_schedule(plan["sso_training_schedule"], steps_name="sso_steps")
    print("-" * 74)
    print(f"  can execute now   : {r['can_execute']}")
    for note in r["benchmark_degraded"]:
        print(f"  [degraded]  {note}")
    for b in r["blockers"]:
        print(f"  [BLOCKER]   {b}")
    if r["comparability_caveat"]:
        print(f"  [CAVEAT]    {r['comparability_caveat']}")
    if plan["manifest_notes"]:
        print("-" * 74)
        print("  manifest notes:")
        for n in plan["manifest_notes"]:
            print(f"    - {n}")
    print("=" * 74)


def print_sso_training_schedule(ts: dict, *, steps_name: str) -> None:
    """The resolved SSO training schedule, as a dry-run shows it (shared by the
    native and the OURS launcher so both print the same thing)."""
    print("-" * 74)
    print("  SSO training schedule (resolved)")
    print(f"    benchmark         : {ts['benchmark']}")
    for key in (steps_name, "sso_train_per_step", "sso_schedule_seed"):
        print(f"    {key:<18s}: {ts[key]}   [{ts['sources'][key]}]")
    knobs = ts.get("sso_knobs") or {}
    for key in ("sso_history_per_step", "sso_num_probes", "sso_update_budget"):
        if key in knobs:
            print(f"    {key:<18s}: {knobs[key]}   [{knobs['sources'][key]}]")
    for t, ids in enumerate(ts["x_t_task_ids"]):
        print(f"    X_{t} ({len(ids)})  : {', '.join(ids)}")


def _build_config(args, argv_for_record: list) -> ExperimentConfig:
    if args.config:
        config = load_config(args.config)
    else:
        kwargs = dict(
            optimizer=args.optimizer,
            benchmark=args.benchmark,
            solver_model=args.solver_model,
            optimizer_model=args.optimizer_model,
            harness=args.harness,
            split_role=args.split,
            out_root=args.out_root,
            run_id=args.run_id,
            split_manifest_path=str(args.split_manifest) if args.split_manifest else None,
            deployment_workers=args.deployment_workers,
        )
        if args.proxy_base_url:
            kwargs["proxy_base_url"] = args.proxy_base_url
        if args.api_key:
            kwargs["api_key"] = args.api_key
        if args.max_tokens:
            kwargs["max_tokens"] = args.max_tokens
        if args.optimizer_max_tokens:
            kwargs["optimizer_max_tokens"] = args.optimizer_max_tokens
        if args.temperature is not None:
            kwargs["temperature"] = args.temperature
        if args.harness_option:
            kwargs["harness_options"] = dict(_parse_kv(args.harness_option))
        extra = dict(_parse_kv(args.extra or []))
        extra["launch_command"] = " ".join(argv_for_record)
        kwargs["extra"] = extra
        config = ExperimentConfig(**kwargs)
    return config


def _parse_kv(pairs):
    for pair in pairs:
        key, _, raw = pair.partition("=")
        if not key or not _:
            raise ValueError(f"expected key=value, got {pair!r}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        yield key, value


def main(argv: list | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", type=Path, help="JSON config file instead of the flags below")
    p.add_argument("--optimizer", choices=OPTIMIZERS)
    p.add_argument("--benchmark", help=f"one of: {', '.join(BENCHMARK_NAMES)} (aliases accepted)")
    p.add_argument("--solver-model", help="proxy alias for the frozen solver (see solver_models.md)")
    p.add_argument("--optimizer-model", default=DEFAULT_OPTIMIZER_MODEL,
                   help=f"proxy alias for the optimizer-side LLM, shared by every "
                        f"optimizer-side role in all three optimizers "
                        f"(default: {DEFAULT_OPTIMIZER_MODEL})")
    p.add_argument("--split-manifest", type=Path, default=None,
                   help="user-specified split manifest; omit to use the benchmark default")
    p.add_argument("--split", default="deployment", choices=list(SPLIT_ROLES),
                   help="which split the artifact's metric refers to (deployment)")
    p.add_argument("--harness", default="direct", choices=("direct",))
    p.add_argument("--proxy-base-url", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--optimizer-max-tokens", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--deployment-workers", type=int, default=1)
    p.add_argument("--harness-option", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--extra", action="append", default=[], metavar="KEY=VALUE",
                   help="optimizer knobs, e.g. sso_steps=2 skillopt_num_epochs=1")
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--run-id", default="")
    p.add_argument("--mode", default="dry-run", choices=("dry-run", "execute"))
    p.add_argument("--json", action="store_true", help="emit the dry-run plan as JSON")
    p.add_argument("--show-code-freeze", action="store_true")
    args = p.parse_args(argv)

    if not args.config:
        missing = [
            name
            for name, value in (
                ("--optimizer", args.optimizer),
                ("--benchmark", args.benchmark),
                ("--solver-model", args.solver_model),
            )
            if not value
        ]
        if missing:
            p.error(f"--config, or all of --optimizer/--benchmark/--solver-model (missing: {missing})")

    argv_for_record = ["python", "-m", "integration.run_baseline"] + list(argv or sys.argv[1:])
    config = _build_config(args, argv_for_record)

    if args.mode == "execute":
        try:
            outcome = execute(config)
        except PreflightError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        record = outcome.record
        print(json.dumps(record, indent=2, ensure_ascii=False, default=str))
        print(f"\nartifact: {outcome.artifact_path}")
        perf = record["performance"]
        print(
            f"{record['experiment']['optimizer']} x {record['experiment']['benchmark']}: "
            f"{perf['metric_name']}={perf['metric_value']} over n={perf['n_tasks']} | "
            f"optimization {record['optimization_cost']['llm_calls']} calls / "
            f"{record['optimization_cost']['total_tokens']} tokens | "
            f"deployment {record['deployment_cost']['llm_calls']} calls / "
            f"{record['deployment_cost']['total_tokens']} tokens"
        )
        return 0

    plan = build_plan(config)
    if args.show_code_freeze:
        plan["code_freeze"] = code_freeze_reference()
    if args.json:
        print(json.dumps(plan, indent=2, ensure_ascii=False, default=str))
    else:
        print_plan(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
