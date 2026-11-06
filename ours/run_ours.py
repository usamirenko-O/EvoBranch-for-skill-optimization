"""Configure and run OURS with a selected optimizer and benchmark."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

from integration.benchmarks.registry import BENCHMARK_NAMES  # noqa: E402
from integration.experiment.config import (  # noqa: E402
    DEFAULT_OPTIMIZER_MODEL,
    DEFAULT_OUT_ROOT,
    code_freeze_reference,
)
from integration.experiment.execute import PreflightError, resolve_manifest  # noqa: E402
from integration.harness.registry import HARNESS_KIND  # noqa: E402
from ours import registry  # noqa: E402
from ours.design import OursDesign  # noqa: E402
from ours.experiment.config import (  # noqa: E402
    METHODS,
    OURS_OPTIMIZERS,
    OursRunConfig,
)


#: optimizer -> its OURS run driver module. Each driver exposes
#: `ours_preflight(config, manifest)` and `run(config)`.
_DRIVERS = {
    "skillgen": "ours.experiment.skillgen_ours",
    "skillopt": "ours.experiment.skillopt_ours",
    "sso": "ours.experiment.sso_ours",
}

#: What TRAIN is used for, per optimizer, for the dry-run plan.
_TRAIN_ROLE = {
    "skillgen": "native SkillGen: Beam_0 = no skill, so round 0 clusters the no-skill "
                "trajectories and runs induction + generation per subset; skilled "
                "Parents give contrastive evidence vs that fixed no-skill baseline "
                "and one refine_once per subset",
    "skillopt": "native SkillOpt: each Parent rolled out in SkillOpt's own env on the "
                "round's native TRAIN batch, then one independent native one-step "
                "propose_candidate per evidence subset (same one-step edit budget "
                "every round; no SkillOpt state across rounds)",
    "sso": "native SSO: each Parent executed on the native X_t batch, then one "
           "native SSO step (probes/judge/behaviors/render) per evidence subset",
}


def driver_for(optimizer: str):
    import importlib

    return importlib.import_module(_DRIVERS[optimizer])


def build_plan(config: OursRunConfig) -> dict:
    """Everything a dry-run reports, as data (so tests can assert on it)."""
    from ours.adapters.skillgen import EVIDENCE_SCOPES

    adapter = config.adapter()
    manifest = resolve_manifest(config.base)
    blockers = driver_for(config.optimizer).ours_preflight(config, manifest)
    resources = adapter.resource_status()

    plan = {
        "experiment": {
            "optimizer": config.optimizer,
            "method": config.method,
            "benchmark": config.benchmark,
            "harness": config.base.harness,
            "harness_kind": HARNESS_KIND,
            "solver_model": config.base.solver_model,
            "optimizer_model": config.base.optimizer_model,
        },
        "search": config.search_settings(),
        "split_manifest": {
            "manifest_id": manifest.manifest_id,
            "version": manifest.version,
            "source": config.base.split_manifest_path or "benchmark default",
            "content_hash": manifest.content_hash,
            "counts": manifest.counts(),
        },
        "split_roles": {
            "train": _TRAIN_ROLE[config.optimizer],
            "val": "OURS selection -- global Top-M over the competitive pool",
            "deployment": "reported once, after the search has finished",
        },
        "scorer": {"metric_name": adapter.metric_name, "scorer_ref": adapter.scorer_ref},
        "output": {
            "run_dir": str(config.run_dir()),
            "artifact_path": str(config.artifact_path()),
        },
        "design": config.design.to_dict(),
        "placeholders": {
            "partitioner": config.partitioner,
            "operator": config.operator,
            "tie_break": config.design.tie_break,
            "deployment_rule": config.design.deployment_rule,
            "in_force": config.placeholders_in_force(),
            "components": config.design_report(),
            "evidence_scope": config.evidence_scope,
            "evidence_scope_options": list(EVIDENCE_SCOPES),
            "note": "OURS experimental design lives in ours/design.py::OursDesign; "
                    "every slot is swappable and applies to every optimizer",
        },
        "readiness": {
            "benchmark_resources_ready": resources.ready,
            "benchmark_degraded": resources.degraded,
            "blockers": blockers,
            "can_execute": not blockers,
        },
    }
    driver = driver_for(config.optimizer)
    if hasattr(driver, "training_schedule_plan") and not blockers:
        plan["sso_training_schedule"] = driver.training_schedule_plan(config, manifest)
    return plan


def print_plan(plan: dict) -> None:
    e, s, m, r = plan["experiment"], plan["search"], plan["split_manifest"], plan["readiness"]
    print("=" * 74)
    print("OURS DRY RUN -- no LLM call, no network call")
    print("=" * 74)
    print(f"  optimizer x method : {e['optimizer']} + {e['method']}")
    print(f"  benchmark          : {e['benchmark']}")
    print(f"  harness            : {e['harness']} ({e['harness_kind']})")
    print(f"  solver model       : {e['solver_model']}")
    print(f"  optimizer model    : {e['optimizer_model']}")
    print()
    print(f"  rounds T           : {s['rounds']}")
    print(f"  beam M             : {s['beam_m']}")
    print(f"  subsets K          : {s['k_subsets']}   -> up to M*K raw proposals/round")
    comps = s["design_components"]
    for slot in ("partitioner", "operator", "tie_break", "deployment_rule"):
        flag = "   [PLACEHOLDER]" if comps[slot]["is_placeholder"] else ""
        print(f"  {slot:<19s}: {comps[slot]['name']}{flag}")
    for name, entry in comps["hyperparameters"].items():
        tag = "[PLACEHOLDER]" if entry["is_placeholder"] else "[FROZEN main-experiment default]"
        extra_note = " (overridden)" if entry["overridden"] else ""
        print(f"  {name + ' status':<19s}: {entry['value']}  {tag}{extra_note}  [{entry['source']}]")
    if s.get("main_experiment_config"):
        print("  main-experiment config (resolved):")
        for key, entry in s["main_experiment_config"].items():
            print(f"    {key:<26s}: {entry['value']}   [{entry['source']}]")
    print(f"  evidence scope     : {s['evidence_scope']}")
    print()
    print(f"  split manifest     : {m['manifest_id']} v{m['version']}  ({m['source']})")
    print(f"    content hash     : {m['content_hash']}")
    print(f"    counts           : train={m['counts']['train']} val={m['counts']['val']} "
          f"deployment={m['counts']['deployment']}")
    for role, purpose in plan["split_roles"].items():
        print(f"    {role:<11s}    {purpose}")
    print()
    print(f"  scorer             : {plan['scorer']['scorer_ref']}")
    print(f"  metric             : {plan['scorer']['metric_name']}")
    print(f"  output artifact    : {plan['output']['artifact_path']}")
    if "sso_training_schedule" in plan:
        from integration.run_baseline import print_sso_training_schedule

        print_sso_training_schedule(plan["sso_training_schedule"], steps_name="rounds")
    print("-" * 74)
    print(f"  can execute now    : {r['can_execute']}")
    for note in r["benchmark_degraded"]:
        print(f"  [degraded]  {note}")
    for b in r["blockers"]:
        print(f"  [BLOCKER]   {b}")
    print("=" * 74)


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


def build_config(args, argv_for_record: list) -> OursRunConfig:
    extra = dict(_parse_kv(args.extra or []))
    extra["launch_command"] = " ".join(argv_for_record)
    design = OursDesign.load(args.design) if args.design else OursDesign()
    file_fields = json.loads(args.design.read_text(encoding="utf-8")) if args.design else {}
    file_rounds = file_fields.get("rounds")
    explicit_rounds = args.rounds if args.rounds is not None else file_rounds
    design_sources = {k: f"override: --design {args.design}" for k in file_fields}
    for flag, name in (("rounds", "rounds"), ("beam_m", "beam_m"), ("k_subsets", "k_subsets"),
                       ("partitioner", "partitioner"), ("operator", "operator"),
                       ("tie_break", "tie_break"), ("deployment_rule", "deployment_rule")):
        if getattr(args, flag) is not None:
            design_sources[name] = f"override: --{flag.replace('_', '-')}"
    design = design.with_overrides(
        beam_m=args.beam_m, k_subsets=args.k_subsets,
        partitioner=args.partitioner, operator=args.operator,
        tie_break=args.tie_break, deployment_rule=args.deployment_rule,
        strict_ties=True if args.strict_ties else None,
        **dict(_parse_kv(args.design_option or [])),
    )
    kwargs = dict(
        optimizer=args.optimizer,
        method=args.method,
        benchmark=args.benchmark,
        solver_model=args.solver_model,
        optimizer_model=args.optimizer_model,
        harness=args.harness,
        split_manifest_path=str(args.split_manifest) if args.split_manifest else None,
        out_root=args.out_root,
        run_id=args.run_id,
        harness_options=dict(_parse_kv(args.harness_option)) if args.harness_option else None,
        extra=extra,
        deployment_workers=args.deployment_workers,
        design=design,
        design_sources=design_sources,
        **({"rounds": explicit_rounds} if explicit_rounds is not None else {}),
        evidence_scope=args.evidence_scope,
        seed=args.seed,
        validation_workers=args.validation_workers,
    )
    for flag, key in (("proxy_base_url", "proxy_base_url"), ("api_key", "api_key"),
                      ("max_tokens", "max_tokens"),
                      ("optimizer_max_tokens", "optimizer_max_tokens")):
        value = getattr(args, flag, None)
        if value:
            kwargs[key] = value
    if args.temperature is not None:
        kwargs["temperature"] = args.temperature
    return OursRunConfig.build(**kwargs)


def main(argv: list | None = None) -> int:
    from ours.adapters.skillgen import EVIDENCE_SCOPES

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--optimizer", choices=OURS_OPTIMIZERS, required=True)
    p.add_argument("--method", choices=METHODS, default="ours")
    p.add_argument("--benchmark", required=True,
                   help=f"one of: {', '.join(BENCHMARK_NAMES)} (aliases accepted)")
    p.add_argument("--solver-model", required=True,
                   help="proxy alias for the frozen solver (see solver_models.md)")
    p.add_argument("--optimizer-model", default=DEFAULT_OPTIMIZER_MODEL)
    p.add_argument("--split-manifest", type=Path, default=None,
                   help="user-specified split manifest; omit to use the benchmark default")
    p.add_argument("--harness", default="direct", choices=("direct",))
    # ── OURS experimental design (ours/design.py). Unset flags fall back to
    #    --design, then to OursDesign's defaults. Same meaning for every optimizer.
    p.add_argument("--design", type=Path, default=None,
                   help="JSON file with OursDesign fields; flags below override it")
    p.add_argument("--rounds", type=int, default=None, help="T -- number of OURS rounds")
    p.add_argument("--beam-m", type=int, default=None, help="M -- beam width (global Top-M)")
    p.add_argument("--k-subsets", type=int, default=None,
                   help="K -- evidence subsets per parent")
    p.add_argument("--partitioner", default=None, choices=registry.names("partitioner"))
    p.add_argument("--operator", default=None, choices=registry.names("operator"))
    p.add_argument("--tie-break", default=None, choices=registry.names("tie_break"))
    p.add_argument("--deployment-rule", default=None, choices=registry.names("deployment_rule"))
    p.add_argument("--design-option", action="append", default=[], metavar="SLOT.KEY=VALUE",
                   help="constructor option for a design component, e.g. partitioner.n_init=10")
    p.add_argument("--strict-ties", action="store_true",
                   help="fail instead of using a placeholder tie-break at the Top-M cutoff")
    p.add_argument("--evidence-scope", default="diagnostic_cases", choices=EVIDENCE_SCOPES,
                   help="SkillGen only")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--validation-workers", type=int, default=1)
    p.add_argument("--deployment-workers", type=int, default=None,
                   help="default: the main-experiment config for the benchmark")
    p.add_argument("--proxy-base-url", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--optimizer-max-tokens", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--harness-option", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--extra", action="append", default=[], metavar="KEY=VALUE",
                   help="optimizer knobs; overrides the value inherited from the "
                        "main-experiment config for that key only")
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--run-id", default="")
    p.add_argument("--mode", default="dry-run", choices=("dry-run", "execute"))
    p.add_argument("--json", action="store_true")
    p.add_argument("--show-code-freeze", action="store_true")
    args = p.parse_args(argv)

    argv_for_record = ["python", "-m", "ours.run_ours"] + list(argv or sys.argv[1:])
    config = build_config(args, argv_for_record)

    if args.mode == "execute":
        run = driver_for(config.optimizer).run

        try:
            outcome = run(config)
        except PreflightError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        record = outcome.record
        print(json.dumps(record, indent=2, ensure_ascii=False, default=str))
        print(f"\nartifact: {outcome.artifact_path}")
        perf = record["performance"]
        print(
            f"{record['experiment']['optimizer']}+{record['experiment']['method']} x "
            f"{record['experiment']['benchmark']}: "
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
