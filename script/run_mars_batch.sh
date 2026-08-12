#!/usr/bin/env bash
# 批量运行 Mars 重建任务：对每个立体相对执行
#   cd <mars_root>/bin && ./Mars <cfg_xxx.yaml>
#
# 用法：
#   bash script/run_mars_batch.sh
#   bash script/run_mars_batch.sh --dry-run
#   bash script/run_mars_batch.sh --continue          # 某对失败后继续下对
#   bash script/run_mars_batch.sh --cfg-dir /path     # cfg 所在目录
#   bash script/run_mars_batch.sh --log-dir /path     # 日志目录
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MARS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---------- 路径配置 ----------
BIN_DIR="${MARS_ROOT}/bin"
MARS_BIN="${BIN_DIR}/Mars"
CFG_DIR="${MARS_ROOT}"
LOG_DIR="${MARS_ROOT}/logs/mars_batch"
# OpenCV 等动态库（与工程编译路径一致；可被环境变量覆盖）
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${HOME}/pkgs/opencv455/opencv/install/lib"

# ---------- 立体相对列表（与 gen_pair_configs.sh 保持一致） ----------
# 每项格式："ID1 ID2" → 对应 cfg_${ID1}_${ID2}.yaml
PAIRS=(
  # "ESP_013001_1415 ESP_013146_1415"   # _1415
  # "ESP_013110_1325 ESP_013611_1325"   # _1325
  # "ESP_067718_1630 ESP_068444_1630"   # _1630
  # "PSP_009561_2325 PSP_010260_2325"   # _2325

  # "ESP_019190_1560 ESP_019335_1560"   # Eberswalde Crater deposits
  # "ESP_018701_1775 ESP_018846_1775"   # Endeavour Crater western rim
  # "ESP_037586_1725 ESP_037731_1725"   # Candor Chasma
  # "ESP_017266_1715 ESP_017411_1715"   # Layered mesa in Candor Chasma
  # "ESP_031916_1730 ESP_031982_1730"   # East Candor Chasma / Nia Mensa

  "PSP_009149_1750 PSP_009294_1750"   # Gale Crater inverted riverbed
  "ESP_012993_1725 ESP_013059_1725"   # Ganges Mensa layered deposits
  "ESP_023957_1755 ESP_024023_1755"   # Gale Crater / MSL region

  "PSP_001777_1650 PSP_001513_1655"
  "PSP_008469_2040 PSP_008825_2040"
  "ESP_075625_2055 ESP_075559_2055"
)

# ---------- 可选开关 ----------
DRY_RUN=0
CONTINUE_ON_ERROR=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)       DRY_RUN=1; shift ;;
    --continue)      CONTINUE_ON_ERROR=1; shift ;;
    --cfg-dir)       CFG_DIR="$2"; shift 2 ;;
    --log-dir)       LOG_DIR="$2"; shift 2 ;;
    --mars)          MARS_BIN="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,14p' "$0"
      exit 0
      ;;
    *)
      echo "[error] 未知参数: $1" >&2
      exit 1
      ;;
  esac
done

if [[ ! -x "$MARS_BIN" ]]; then
  echo "[error] Mars 可执行文件不存在或不可执行: $MARS_BIN" >&2
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
echo " Mars 批量重建"
echo " MARS_ROOT = $MARS_ROOT"
echo " MARS_BIN  = $MARS_BIN"
echo " CFG_DIR   = $CFG_DIR"
echo " LOG_DIR   = $LOG_DIR"
echo " 相对数量  = ${#PAIRS[@]}"
echo "==============================================" | tee "$SUMMARY_LOG"

ok_count=0
fail_count=0
skip_count=0
failed_pairs=()

idx=0
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
  PAIR_LOG="${LOG_DIR}/${ID1}_${ID2}_${TIMESTAMP}.log"

  echo ""
  echo "######## [${idx}/${#PAIRS[@]}] ${ID1}  <->  ${ID2} ########" | tee -a "$SUMMARY_LOG"

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

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] (cd \"$BIN_DIR\" && \"$MARS_BIN\" \"$CFG\")" | tee -a "$SUMMARY_LOG"
    ok_count=$((ok_count + 1))
    continue
  fi

  # 必须在 bin/ 下启动，EO/IO 等相对路径 ../data/... 才正确
  set +e
  (
    cd "$BIN_DIR"
    # 绝对路径传 cfg，避免依赖 cwd 找配置文件
    "$MARS_BIN" "$CFG"
  ) 2>&1 | tee "$PAIR_LOG"
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
echo "[done] 全部相对重建完成"
