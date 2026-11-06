#!/usr/bin/env bash
# 主表实验：跑一个格子。
#
#   ./main_experiment/run_cell.sh <optimizer> <benchmark> <solver>
#
# 例：
#   ./main_experiment/run_cell.sh sso livemath opus5
#   ./main_experiment/run_cell.sh skillopt officeqa sonnet5
#   ./main_experiment/run_cell.sh sso+ours spreadsheet gemini
#   ./main_experiment/run_cell.sh skillopt+ours officeqa opus5
#
# split 用冻结的主实验 manifest（main_experiment/main_experiment_protocol.json），
# 不需要也不应该在这里指定。超参数读 main_experiment/baseline_hparams.json，
# 唯一读取入口是 integration/experiment/main_experiment_config.py。
#
# 常用开关：
#   --dry-run     只打印计划，不调用任何模型
#   --pilot N     用 N 个 deployment 任务的临时小 manifest 试跑（不是主表数据）
#   --out-root D  输出目录（默认 integration/runs）
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# 模型调用默认超时 1800 秒，可通过环境变量覆盖。
export SKILLOPTEVO_SOLVER_TIMEOUT="${SKILLOPTEVO_SOLVER_TIMEOUT:-1800}"
export TARGET_OPENAI_COMPATIBLE_TIMEOUT_SECONDS="${TARGET_OPENAI_COMPATIBLE_TIMEOUT_SECONDS:-1800}"

PROXY_BASE_URL="${PROXY_BASE_URL:-http://localhost:4000/v1}"
OUT_ROOT="integration/runs"
MODE="execute"
PILOT=""
VAL_WORKERS=""
DEP_WORKERS=""
DESIGN=""

die() { echo "" >&2; echo "✗ $*" >&2; exit 1; }

# 使用当前环境；可显式指定 Python 解释器。
ENV_PY="${SKILLOPTEVO_PYTHON:-$(command -v python3)}"
if [[ -x "$ENV_PY" ]]; then
  export PATH="$(dirname "$ENV_PY"):$PATH"
else
  die "找不到 Python 解释器: $ENV_PY。请激活已安装 requirements.txt 的环境。"
fi

usage() {
  cat >&2 <<EOF
用法: $0 <optimizer> <benchmark> <solver> [选项]

  optimizer : skillopt | sso | skillgen | skillopt+ours | sso+ours | skillgen+ours
  benchmark : livemath | officeqa | spreadsheet | alfworld
              | livecode | pubmedqa | chem
  solver    : astra | terra | sol | luna | gpt55      (CLIProxyAPI 配置)
              opus5 | sonnet5 | gemini | qwen        (Vertex 配置)
              (或直接给 proxy 上的任意 alias)

选项:
  --dry-run       只打印计划，不花钱
  --pilot N       用 N 个 deployment 任务试跑（不是主表数据）
  --out-root DIR  输出目录（默认 $OUT_ROOT）
  --validation-workers N   仅 +ours：并行评分 val 任务数（默认 1，上限 8）
  --deployment-workers N   部署阶段并行任务数（上限 8）
  --design FILE            仅 +ours：OursDesign JSON（消融实验用，见 ablation/designs/）；
                           主表格子不要传，主表用 baseline_hparams.json 里冻结的设计
EOF
  exit 1
}

[[ $# -ge 3 ]] || usage
OPTIMIZER="$1"; BENCHMARK_IN="$2"; SOLVER_IN="$3"; shift 3
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)  MODE="dry-run"; shift ;;
    --pilot)    PILOT="${2:-}"; [[ -n "$PILOT" ]] || die "--pilot 需要一个数字"; shift 2 ;;
    --out-root) OUT_ROOT="${2:-}"; shift 2 ;;
    --validation-workers) VAL_WORKERS="${2:-}"; [[ -n "$VAL_WORKERS" ]] || die "--validation-workers 需要一个数字"; shift 2 ;;
    --deployment-workers) DEP_WORKERS="${2:-}"; [[ -n "$DEP_WORKERS" ]] || die "--deployment-workers 需要一个数字"; shift 2 ;;
    --design)   DESIGN="${2:-}"; [[ -f "$DESIGN" ]] || die "--design 需要一个存在的 JSON 文件: '$DESIGN'"; shift 2 ;;
    *) die "未知选项: $1" ;;
  esac
done

# ---- 参数归一化 -----------------------------------------------------------
case "$OPTIMIZER" in
  skillopt|sso|skillgen)                      METHOD="native"; BASE_OPTIMIZER="$OPTIMIZER" ;;
  skillopt+ours|sso+ours|skillgen+ours)       METHOD="ours";   BASE_OPTIMIZER="${OPTIMIZER%+ours}" ;;
  *) die "optimizer 必须是 skillopt / sso / skillgen / skillopt+ours / sso+ours / skillgen+ours，你给的是 '$OPTIMIZER'" ;;
esac

case "$BENCHMARK_IN" in
  livemath|livemathematicianbench) BENCHMARK="livemathematicianbench" ;;
  officeqa)                        BENCHMARK="officeqa" ;;
  spreadsheet|spreadsheetbench|ssb) BENCHMARK="spreadsheetbench" ;;
  alfworld)                        BENCHMARK="alfworld" ;;
  livecode|livecodebench|lcb)      BENCHMARK="livecodebench" ;;
  pubmedqa|pubmed)                 BENCHMARK="pubmedqa" ;;
  chem|chemllmbench|chemllm|chemyield) BENCHMARK="chemllmbench" ;;
  *) die "benchmark 必须是 livemath / officeqa / spreadsheet / alfworld / livecode / pubmedqa / chem，你给的是 '$BENCHMARK_IN'" ;;
esac

case "$SOLVER_IN" in
  opus5)   SOLVER="solver-opus-5" ;;
  sonnet5) SOLVER="solver-sonnet-5" ;;
  gemini)  SOLVER="solver-gemini-2-5-pro" ;;
  qwen)    SOLVER="solver-qwen3-4b" ;;
  astra|gpt6|gpt-6-astra)          SOLVER="solver-gpt-6-astra" ;;
  terra|gpt56|gpt-5.6-terra)       SOLVER="solver-gpt-5-6-terra" ;;
  sol|gpt-5.6-sol)                 SOLVER="solver-gpt-5-6-sol" ;;
  luna|gpt-5.6-luna)               SOLVER="solver-gpt-5-6-luna" ;;
  gpt55|gpt-5.5)                   SOLVER="solver-gpt-5-5" ;;
  *)       SOLVER="$SOLVER_IN" ;;   # 直接当成 proxy alias
esac

# ---- 起跑前检查 -----------------------------------------------------------
if [[ "$MODE" == "execute" ]]; then
  echo "检查 proxy: $PROXY_BASE_URL ..."
  INFO="$(curl -sS --max-time 10 "${PROXY_BASE_URL%/}/model/info" 2>/dev/null || true)"
  [[ -n "$INFO" ]] || die "连不上 proxy ($PROXY_BASE_URL)。请启动与模型配置对应的 LiteLLM proxy，例如:
    litellm --config configs/proxy.example.yaml --port 4000"

  for ALIAS in "$SOLVER" "optimizer-llm"; do
    echo "$INFO" | grep -q "\"$ALIAS\"" \
      || die "proxy 上没有 alias '$ALIAS'。请检查所选模型与 proxy 配置。当前提供的 alias:
$(echo "$INFO" | python3 -c 'import json,sys; print("      " + ", ".join(m["model_name"] for m in json.load(sys.stdin).get("data",[])))' 2>/dev/null || echo '      (解析失败)')"
  done
  echo "  ✓ proxy 可达，alias 齐全"
fi

# ---- 组装参数 -------------------------------------------------------------
if [[ "$METHOD" == "native" ]]; then
  ENTRY="integration.run_baseline"
  ARGS=( --optimizer "$OPTIMIZER" --benchmark "$BENCHMARK" --solver-model "$SOLVER"
         --harness direct --proxy-base-url "$PROXY_BASE_URL"
         --out-root "$OUT_ROOT" --mode "$MODE" )
  # 正式超参数：由唯一读取入口给出 (SSO 的 schedule 与 |H_t|/K/U 由 runner 自己从同一文件解析)
  while IFS= read -r line; do ARGS+=( "$line" ); done < <(
    python3 -m integration.experiment.main_experiment_config launch-args "$OPTIMIZER" "$BENCHMARK" "$SOLVER"
  )
  if [[ -n "$DEP_WORKERS" ]]; then ARGS+=( --deployment-workers "$DEP_WORKERS" ); fi
else
  # +OURS：不传任何超参数 flag —— OursRunConfig.build 通过同一个读取入口对每个
  # optimizer 自动解析完整正式配置 (solver call / harness contract / workers /
  # 继承的 native knobs / SSO schedule)，并把每个值的来源写进 dry-run 与 artifact。
  ENTRY="ours.run_ours"
  ARGS=( --optimizer "$BASE_OPTIMIZER" --benchmark "$BENCHMARK" --solver-model "$SOLVER"
         --harness direct --proxy-base-url "$PROXY_BASE_URL"
         --out-root "$OUT_ROOT" --mode "$MODE" )
  if [[ -n "$VAL_WORKERS" ]]; then ARGS+=( --validation-workers "$VAL_WORKERS" ); fi
  if [[ -n "$DEP_WORKERS" ]]; then ARGS+=( --deployment-workers "$DEP_WORKERS" ); fi
  if [[ -n "$DESIGN" ]]; then ARGS+=( --design "$DESIGN" ); fi
fi
if [[ -n "$DESIGN" && "$METHOD" != "ours" ]]; then die "--design 只对 +ours 有意义"; fi

if [[ -n "$PILOT" ]]; then
  PILOT_MANIFEST="$OUT_ROOT/_pilot_manifests/${BENCHMARK}_pilot${PILOT}.json"
  mkdir -p "$(dirname "$PILOT_MANIFEST")"
  python3 -m integration.benchmarks.make_manifest \
      --benchmark "$BENCHMARK" --train 2 --val 2 --deployment "$PILOT" \
      --seed 0 --manifest-id "pilot_${BENCHMARK}_${PILOT}" --out "$PILOT_MANIFEST" >/dev/null
  ARGS+=( --split-manifest "$PILOT_MANIFEST" )
  echo ""
  echo "⚠  PILOT 模式：用的是临时 manifest（deployment=$PILOT），"
  echo "   这不是主表数据。主表跑请去掉 --pilot。"
fi

echo ""
echo "════════════════════════════════════════════════════════"
echo "  optimizer : $OPTIMIZER   (method: $METHOD)"
echo "  benchmark : $BENCHMARK"
echo "  solver    : $SOLVER        (optimizer 侧固定 optimizer-llm)"
echo "  mode      : $MODE"
echo "════════════════════════════════════════════════════════"
echo ""

set +e
python3 -m "$ENTRY" "${ARGS[@]}"
RC=$?
set -e

if [[ "$MODE" == "execute" && $RC -eq 0 ]]; then
  echo ""
  echo "════════════════════════════════════════════════════════"
  echo "  跑完了。看结果："
  echo ""
  echo "    python3 -m integration.show_results"
  echo ""
  echo "  看这一格的详细内容（日志 / 逐题判定 / 最终 skill）："
  echo ""
  echo "    python3 -m integration.show_results --latest --detail"
  echo "════════════════════════════════════════════════════════"
fi
exit $RC
