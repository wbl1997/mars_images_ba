#!/usr/bin/env bash
# 批量生成 mosaic 视差图：对每个立体相对执行
#   python script/grid_to_mosaic_disparity_paper_panels.py <cfg> --source <grid|match>
#
# 用法：
#   bash script/run_grid_to_mosaic_disparity.sh
#   bash script/run_grid_to_mosaic_disparity.sh --dry-run
#   bash script/run_grid_to_mosaic_disparity.sh --continue
#   bash script/run_grid_to_mosaic_disparity.sh --source grid
#   bash script/run_grid_to_mosaic_disparity.sh --cfg-dir /path --log-dir /path
#   bash script/run_grid_to_mosaic_disparity.sh --output-mode png+pdf
#   bash script/run_grid_to_mosaic_disparity.sh --extra "--dpi 300 --crop 0 1000 0 2000"
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MARS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---------- 路径配置 ----------
PY_SCRIPT="${SCRIPT_DIR}/grid_to_mosaic_disparity_paper_panels.py"
CFG_DIR="${MARS_ROOT}"
LOG_DIR="${MARS_ROOT}/logs/grid_to_mosaic_disparity"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# ---------- 立体相对列表（与 run_mars_batch.sh 保持一致） ----------
# 每项格式："ID1 ID2" → 对应 cfg_${ID1}_${ID2}.yaml
PAIRS=(
  # "ESP_013001_1415 ESP_013146_1415"
  # "ESP_013110_1325 ESP_013611_1325"
  # "ESP_067718_1630 ESP_068444_1630"
  # "PSP_009561_2325 PSP_010260_2325"

  # "ESP_019190_1560 ESP_019335_1560"
  # "ESP_018701_1775 ESP_018846_1775"
  # "ESP_037586_1725 ESP_037731_1725"
  # "ESP_017266_1715 ESP_017411_1715"
  # "ESP_031916_1730 ESP_031982_1730"

  "PSP_009149_1750 PSP_009294_1750"
  "ESP_012993_1725 ESP_013059_1725"
  # "ESP_023957_1755 ESP_024023_1755"

  "PSP_001777_1650 PSP_001513_1655"
  "PSP_008469_2040 PSP_008825_2040"
  "ESP_075625_2055 ESP_075559_2055"
)

# ---------- 可选开关 ----------
DRY_RUN=0
CONTINUE_ON_ERROR=0
SOURCE="match"
OUTPUT_MODE="png"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)       DRY_RUN=1; shift ;;
    --continue)      CONTINUE_ON_ERROR=1; shift ;;
    --source)        SOURCE="$2"; shift 2 ;;
    --output-mode)   OUTPUT_MODE="$2"; shift 2 ;;
    --cfg-dir)       CFG_DIR="$2"; shift 2 ;;
    --log-dir)       LOG_DIR="$2"; shift 2 ;;
    --python)        PYTHON_BIN="$2"; shift 2 ;;
    --script)        PY_SCRIPT="$2"; shift 2 ;;
    --extra)
      # 额外参数整串传给 python，例如: --extra "--dpi 300 --marker-size 5"
      # shellcheck disable=SC2206
      EXTRA_ARGS+=($2)
      shift 2
      ;;
    -h|--help)
      sed -n '2,15p' "$0"
      exit 0
      ;;
    *)
      echo "[error] 未知参数: $1" >&2
      exit 1
      ;;
  esac
done

if [[ "$SOURCE" != "grid" && "$SOURCE" != "match" ]]; then
  echo "[error] --source 必须是 grid 或 match，当前: $SOURCE" >&2
  exit 1
fi

if [[ "$OUTPUT_MODE" != "png" && "$OUTPUT_MODE" != "png+pdf" ]]; then
  echo "[error] --output-mode 必须是 png 或 png+pdf，当前: $OUTPUT_MODE" >&2
  exit 1
fi

if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "[error] Python 脚本不存在: $PY_SCRIPT" >&2
  exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[error] Python 不可用: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ${#PAIRS[@]} -eq 0 ]]; then
  echo "[error] PAIRS 为空，请在脚本中填写立体相对列表" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY_LOG="${LOG_DIR}/batch_${TIMESTAMP}.summary.log"

echo "=============================================="
echo " grid_to_mosaic_disparity 批量"
echo " MARS_ROOT = $MARS_ROOT"
echo " PY_SCRIPT = $PY_SCRIPT"
echo " PYTHON    = $PYTHON_BIN"
echo " CFG_DIR   = $CFG_DIR"
echo " LOG_DIR   = $LOG_DIR"
echo " SOURCE    = $SOURCE"
echo " OUT_MODE  = $OUTPUT_MODE"
echo " 相对数量  = ${#PAIRS[@]}"
echo "==============================================" | tee "$SUMMARY_LOG"

ok_count=0
fail_count=0
skip_count=0
failed_pairs=()

idx=0
total_pairs=0
for pair in "${PAIRS[@]}"; do
  [[ -z "${pair//[[:space:]]/}" ]] && continue
  [[ "$pair" =~ ^[[:space:]]*# ]] && continue
  total_pairs=$((total_pairs + 1))
done

for pair in "${PAIRS[@]}"; do
  [[ -z "${pair//[[:space:]]/}" ]] && continue
  [[ "$pair" =~ ^[[:space:]]*# ]] && continue

  read -r ID1 ID2 _ <<< "$pair"
  if [[ -z "${ID1:-}" || -z "${ID2:-}" ]]; then
    echo "[error] 无效相对条目: '$pair'（需要 \"ID1 ID2\"）" >&2
    exit 1
  fi

  idx=$((idx + 1))
  CFG="${CFG_DIR}/cfg_${ID1}_${ID2}.yaml"
  PAIR_LOG="${LOG_DIR}/${ID1}_${ID2}_${SOURCE}_${TIMESTAMP}.log"

  echo ""
  echo "######## [${idx}/${total_pairs}] ${ID1}  <->  ${ID2}  (source=${SOURCE}) ########" | tee -a "$SUMMARY_LOG"

  if [[ ! -f "$CFG" ]]; then
    echo "[skip] 找不到 cfg: $CFG" | tee -a "$SUMMARY_LOG"
    skip_count=$((skip_count + 1))
    failed_pairs+=("${ID1}/${ID2} (missing cfg)")
    if [[ "$CONTINUE_ON_ERROR" -eq 0 ]]; then
      exit 1
    fi
    continue
  fi

  echo "[info] cfg = $CFG" | tee -a "$SUMMARY_LOG"
  echo "[info] log = $PAIR_LOG" | tee -a "$SUMMARY_LOG"

  CMD=(
    "$PYTHON_BIN" "$PY_SCRIPT" "$CFG"
    --source "$SOURCE"
    --output-mode "$OUTPUT_MODE"
    --fill-gaps
  )
  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    CMD+=("${EXTRA_ARGS[@]}")
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] ${CMD[*]}" | tee -a "$SUMMARY_LOG"
    ok_count=$((ok_count + 1))
    continue
  fi

  set +e
  "${CMD[@]}" 2>&1 | tee "$PAIR_LOG"
  rc=${PIPESTATUS[0]}
  set -e

  if [[ "$rc" -eq 0 ]]; then
    echo "[ok] ${ID1} / ${ID2} 完成 (exit=$rc)" | tee -a "$SUMMARY_LOG"
    ok_count=$((ok_count + 1))
  else
    echo "[fail] ${ID1} / ${ID2} 失败 (exit=$rc)，详见 $PAIR_LOG" | tee -a "$SUMMARY_LOG"
    fail_count=$((fail_count + 1))
    failed_pairs+=("${ID1}/${ID2} (exit=$rc)")
    if [[ "$CONTINUE_ON_ERROR" -eq 0 ]]; then
      echo "[abort] 已中止批量任务（可用 --continue 跳过失败项）" | tee -a "$SUMMARY_LOG"
      exit "$rc"
    fi
  fi
done

echo ""
echo "========== 批量结果 ==========" | tee -a "$SUMMARY_LOG"
echo "  成功: $ok_count" | tee -a "$SUMMARY_LOG"
echo "  失败: $fail_count" | tee -a "$SUMMARY_LOG"
echo "  跳过: $skip_count" | tee -a "$SUMMARY_LOG"
echo "  汇总: $SUMMARY_LOG" | tee -a "$SUMMARY_LOG"
if [[ ${#failed_pairs[@]} -gt 0 ]]; then
  echo "  失败相对:" | tee -a "$SUMMARY_LOG"
  for p in "${failed_pairs[@]}"; do
    echo "    - $p" | tee -a "$SUMMARY_LOG"
  done
fi

if [[ "$fail_count" -gt 0 || "$skip_count" -gt 0 ]]; then
  exit 1
fi
echo "[done] 全部相对视差图生成完成"
