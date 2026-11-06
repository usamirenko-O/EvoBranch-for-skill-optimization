"""Display result tables and details for individual experiment runs."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

DEFAULT_RUNS_ROOT = _WORKSPACE_ROOT / "integration" / "runs"

#: 一格结果 = 这两种文件之一。native 与 +OURS 故意用不同文件名，
#: 这样一个 glob 永远不会把 +OURS 的数字当成 baseline 收进来。
RECORD_FILES = {
    "experiment_record.json": "native",
    "ours_experiment_record.json": "ours",
}

#: 这些文件是"跑的时候到底发生了什么"的记录，详情视图会把路径列出来。
_LOG_FILES = [
    ("标准化结果 (native)", "experiment_record.json"),
    ("标准化结果 (+OURS)", "ours_experiment_record.json"),
    ("OURS 运行配置", "ours_run_config.json"),
    ("OURS 搜索轨迹", "ours_optimization_summary.json"),
    ("OURS 逐轮记录", "rounds"),
    ("本次用的 split", "split_manifest.json"),
    ("最终 skill", "final_skill.md"),
    ("held-out 逐题判定", "deployment_result.json"),
    ("held-out 每题完整轨迹", "deployment_transcripts.jsonl"),
    ("SkillOpt 启动参数", "skillopt_launch.json"),
    ("SkillOpt stdout", "skillopt_stdout.log"),
    ("SkillOpt stderr", "skillopt_stderr.log"),
    ("SkillOpt 原生产物", "skillopt_out"),
    ("SSO 选中的 skill", "sso_selected_skill.md"),
    ("SkillGen 生效配置", "skillgen_effective_config.yaml"),
    ("SkillGen 实例池", "skillgen_instances.json"),
    ("SkillGen 分阶段用量", "skillgen_token_stats.json"),
    ("SkillGen 原生产物", "skillgen_artifacts"),
]


def _method_of(record: dict, filename: str) -> str:
    """native / ours。文件名与记录里的 method 字段必须一致，否则这条记录不可信。"""
    expected = RECORD_FILES[filename]
    declared = record["experiment"].get("method", "native")
    if declared != expected:
        raise ValueError(
            f"{filename} declares method={declared!r} but the filename means {expected!r}"
        )
    return expected


def load_runs(runs_root: Path) -> list:
    out = []
    if not runs_root.is_dir():
        return out
    for filename in RECORD_FILES:
        for path in sorted(runs_root.rglob(filename)):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                method = _method_of(record, filename)
            except (json.JSONDecodeError, OSError, KeyError, ValueError):
                continue
            out.append({"record": record, "run_dir": path.parent, "method": method})
    out.sort(key=lambda r: r["record"]["provenance"].get("created_utc", ""))
    return out


def _method_label(entry: dict) -> str:
    return entry["record"]["experiment"]["optimizer"] + ("+ours" if entry["method"] == "ours" else "")


def _cell_key(entry: dict) -> tuple:
    e = entry["record"]["experiment"]
    return (entry["method"], e["optimizer"], e["benchmark"], e["solver_model"])


def _fmt(value, width: int = 0) -> str:
    if value is None:
        return "—".rjust(width) if width else "—"
    if isinstance(value, float):
        text = f"{value:.4f}"
    elif isinstance(value, int):
        text = f"{value:,}"
    else:
        text = str(value)
    return text.rjust(width) if width else text


def print_table(runs: list, *, detail: bool) -> None:
    if not runs:
        print("还没有跑过任何格子。\n\n  ./main_experiment/run_cell.sh <optimizer> <benchmark> <solver>")
        return

    header = (
        f"{'method':<14} {'benchmark':<24} {'solver':<24} "
        f"{'metric':<38} {'value':>9} {'n':>5} "
        f"{'opt calls':>10} {'opt tokens':>12} {'dep calls':>10} {'dep tokens':>12}"
    )
    print(header)
    print("─" * len(header))
    for entry in runs:
        r = entry["record"]
        e, p, o, d = r["experiment"], r["performance"], r["optimization_cost"], r["deployment_cost"]
        print(
            f"{_method_label(entry):<14} {e['benchmark']:<24} {e['solver_model']:<24} "
            f"{p['metric_name']:<38} {_fmt(p['metric_value'], 9)} {_fmt(p['n_tasks'], 5)} "
            f"{_fmt(o['llm_calls'], 10)} {_fmt(o['total_tokens'], 12)} "
            f"{_fmt(d['llm_calls'], 10)} {_fmt(d['total_tokens'], 12)}"
        )
    print()
    if detail:
        for entry in runs:
            print_detail(entry)


def print_detail(entry: dict) -> None:
    r, run_dir = entry["record"], entry["run_dir"]
    e, p, prov = r["experiment"], r["performance"], r["provenance"]
    gate = r.get("optimizer_internal_gate", {})

    print("═" * 78)
    print(f"  {_method_label(entry)} × {e['benchmark']} × {e['solver_model']}")
    print("═" * 78)
    print(f"  run_id            : {prov['run_id']}")
    print(f"  跑完时间          : {prov.get('created_utc', '—')}")
    print(f"  harness           : {e['harness']}")
    print(f"  optimizer 侧模型  : {e.get('optimizer_model', '—')}")
    print(f"  solver 实际后端   : {e.get('solver_model_resolved', '—')}")
    print(f"  split             : {e['split_manifest_id']}  {e.get('split_counts', '')}")
    print(f"  split hash        : {prov['benchmark_manifest_hash']}")
    print()
    print(f"  【performance】 官方打分器 = {p['scorer_ref']}")
    print(f"      {p['metric_name']} = {_fmt(p['metric_value'])}   (held-out n={p['n_tasks']})")
    for key, value in (p.get("secondary_metrics") or {}).items():
        print(f"      · {key}: {_fmt(value) if not isinstance(value, dict) else value}")
    print()
    for label, section in (("optimization", "optimization_cost"), ("deployment", "deployment_cost")):
        c = r[section]
        print(f"  【{label} cost】 {_fmt(c['llm_calls'])} 次调用 · "
              f"in {_fmt(c['input_tokens'])} / out {_fmt(c['output_tokens'])} / "
              f"总 {_fmt(c['total_tokens'])} token  ({c['token_source']})")
        for role, bucket in (c.get("per_role") or {}).items():
            print(f"      · {role}: {bucket.get('calls')} 次 · "
                  f"in {bucket.get('prompt_tokens'):,} / out {bucket.get('completion_tokens'):,}")
    print()
    if entry["method"] == "ours":
        ours = r.get("ours", {})
        comps = (ours.get("design_components") or {})
        print(f"  【OURS design】 T={ours.get('rounds')} M={ours.get('beam_m')} K={ours.get('k_subsets')}  "
              f"partitioner={ours.get('partitioner')} operator={ours.get('operator')} "
              f"tie_break={ours.get('tie_break')} deployment_rule={ours.get('deployment_rule')}")
        placeholders = ours.get("placeholders_in_force") or []
        print(f"      placeholders in force: {', '.join(placeholders) if placeholders else '(none)'}")
        for name, entry_ in (comps.get("hyperparameters") or {}).items():
            print(f"      · {name}: {entry_.get('value')}  [{entry_.get('status')}; {entry_.get('source')}]")
    else:
        print(f"  【optimizer 内部 gate】 {gate.get('mechanism', '—')}")
        print(f"      skill 被接受 = {gate.get('skill_accepted')}   "
              f"(这不是 benchmark 指标: is_benchmark_metric={gate.get('is_benchmark_metric')})")
    print()
    print(f"  【日志与产物】 {run_dir}")
    for label, name in _LOG_FILES:
        path = run_dir / name
        if path.exists():
            size = f"{path.stat().st_size:,}B" if path.is_file() else "目录"
            print(f"      {label:<22} {name:<32} {size}")
    print(f"  【启动命令】 {prov.get('launch_command', '—')}")
    print()


def write_csv(runs: list, path: Path) -> None:
    fields = [
        "method", "optimizer", "benchmark", "solver_model", "metric_name", "metric_value", "n_tasks",
        "optimization_llm_calls", "optimization_tokens",
        "deployment_llm_calls", "deployment_tokens",
        "split_manifest_id", "split_manifest_hash", "run_id", "run_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for entry in runs:
            r = entry["record"]
            e, p, o, d, prov = (
                r["experiment"], r["performance"], r["optimization_cost"],
                r["deployment_cost"], r["provenance"],
            )
            writer.writerow({
                "method": _method_label(entry),
                "optimizer": e["optimizer"], "benchmark": e["benchmark"],
                "solver_model": e["solver_model"],
                "metric_name": p["metric_name"], "metric_value": p["metric_value"],
                "n_tasks": p["n_tasks"],
                "optimization_llm_calls": o["llm_calls"], "optimization_tokens": o["total_tokens"],
                "deployment_llm_calls": d["llm_calls"], "deployment_tokens": d["total_tokens"],
                "split_manifest_id": e["split_manifest_id"],
                "split_manifest_hash": prov["benchmark_manifest_hash"],
                "run_id": prov["run_id"], "run_dir": str(entry["run_dir"]),
            })
    print(f"写出 {len(runs)} 行 -> {path}")


def main(argv: list | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    p.add_argument("--detail", action="store_true", help="附成本明细、内部 gate、日志路径")
    p.add_argument("--latest", action="store_true", help="只看最近一次运行")
    p.add_argument("--all-runs", action="store_true", help="同一格的历史运行全部显示")
    p.add_argument("--method", choices=("native", "ours"), help="只看 native 或只看 +OURS")
    p.add_argument("--optimizer"); p.add_argument("--benchmark"); p.add_argument("--solver")
    p.add_argument("--csv", type=Path, help="导出 CSV")
    args = p.parse_args(argv)

    runs = load_runs(args.runs_root)
    if args.method:
        runs = [r for r in runs if r["method"] == args.method]
    for key, value in (("optimizer", args.optimizer), ("benchmark", args.benchmark),
                       ("solver_model", args.solver)):
        if value:
            runs = [r for r in runs if value in r["record"]["experiment"][key]]

    if not args.all_runs:
        latest: dict = {}
        for entry in runs:                      # 已按时间排序，后来的覆盖先前的
            latest[_cell_key(entry)] = entry
        runs = list(latest.values())
    if args.latest:
        runs = runs[-1:]

    if args.csv:
        write_csv(runs, args.csv)
        return 0
    print_table(runs, detail=args.detail)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
