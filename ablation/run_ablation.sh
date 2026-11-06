#!/usr/bin/env bash
# 消融实验：4 个 variant × 2 benchmark = 8 格 SkillOpt+OURS × Luna。
#
#   ./ablation/run_ablation.sh                 # 2 车道全部开跑（Luna 使用低并发）
#   ./ablation/run_ablation.sh --dry-run       # 8 格只解析配置、不花钱
#   ./ablation/run_ablation.sh --only wo_global_evolution:pubmedqa   # 只跑一格（可重复给）
#
# 每格 = ./main_experiment/run_cell.sh skillopt+ours <bench> luna --design ablation/designs/<variant>.json
#        --out-root integration/runs_ablation/<variant>
# 输出与主表 integration/runs 完全分开，show_results 的 glob 不会碰到它们。
# 此脚本只运行四个消融配置；Native / Full OURS 使用主实验入口运行。
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

VARIANTS=(wo_history_state wo_history_partition wo_attempt_proposal wo_global_evolution)
BENCHES=(livemath pubmedqa)
SOLVER="${SOLVER:-luna}"
VW="${VW:-2}"; DW="${DW:-4}"          # Luna 专用低并发
OUT_BASE="integration/runs_ablation"
MODE_FLAG=""
ONLY=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) MODE_FLAG="--dry-run"; shift ;;
    --only)    ONLY+=("$2"); shift 2 ;;
    *) echo "unknown option $1" >&2; exit 1 ;;
  esac
done

cell() {   # cell <variant> <bench>
  local v="$1" b="$2"
  local design="ablation/designs/${v}.json"
  [[ -f "$design" ]] || { echo "missing $design" >&2; return 1; }
  mkdir -p "$OUT_BASE/$v" "$OUT_BASE/_logs"
  local log="$OUT_BASE/_logs/${v}__${b}__${SOLVER}__$(date +%Y%m%d-%H%M%S).log"
  echo "[$(date +%F\ %T)] START $v $b -> $log"
  if ./main_experiment/run_cell.sh skillopt+ours "$b" "$SOLVER" --design "$design" \
        --out-root "$OUT_BASE/$v" --validation-workers "$VW" --deployment-workers "$DW" \
        $MODE_FLAG > "$log" 2>&1; then
    echo "[$(date +%F\ %T)] OK    $v $b"
  else
    echo "[$(date +%F\ %T)] FAIL  $v $b  (see $log)"; return 1
  fi
}

if [[ ${#ONLY[@]} -gt 0 ]]; then
  for spec in "${ONLY[@]}"; do cell "${spec%%:*}" "${spec##*:}"; done
  exit 0
fi

if [[ -n "$MODE_FLAG" ]]; then
  for v in "${VARIANTS[@]}"; do for b in "${BENCHES[@]}"; do cell "$v" "$b"; done; done
  exit 0
fi

# 两车道，将两个 benchmark 分配到各车道。
lane() { local name="$1"; shift
  for spec in "$@"; do cell "${spec%%:*}" "${spec##*:}" || true; done
  echo "[$(date +%F\ %T)] lane $name done"
}
lane A wo_history_state:livemath     wo_attempt_proposal:livemath \
       wo_history_state:pubmedqa     wo_attempt_proposal:pubmedqa &
lane B wo_history_partition:livemath wo_global_evolution:livemath \
       wo_history_partition:pubmedqa wo_global_evolution:pubmedqa &
wait
echo "ALL ABLATION CELLS DONE -> $OUT_BASE"
