#!/usr/bin/env bash
# Batch denoise configured *_FI.txt DEM point clouds.
#
# Usage:
#   bash script/run_denoise_fi_dem_pointcloud.sh
#   bash script/run_denoise_fi_dem_pointcloud.sh --dry-run
#   bash script/run_denoise_fi_dem_pointcloud.sh --output-dir data/result_denoised --extra "--cell-size 150 --sigma 3"
#   bash script/run_denoise_fi_dem_pointcloud.sh --cfg-dir /path --input-dir /path --log-dir /path
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MARS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PY_SCRIPT="${SCRIPT_DIR}/denoise_fi_dem_pointcloud.py"
CFG_DIR="${MARS_ROOT}"
INPUT_DIR="${MARS_ROOT}/data/result"
OUTPUT_DIR=""
LOG_DIR="${MARS_ROOT}/logs/fi_denoise"

# ---------- 立体相对列表（与 run_mars_batch.sh / run_grid_to_mosaic_disparity.sh 同格式） ----------
# 每项格式："ID1 ID2"
# 对应：
#   config: ${CFG_DIR}/cfg_${ID1}_${ID2}.yaml
#   FI:     ${INPUT_DIR}/${ID1}_${ID2}_FI.txt
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

DRY_RUN=0
CONTINUE_ON_ERROR=0
WRITE_NOISE=0
EXTRA_ARGS=()

usage() {
  sed -n '2,9p' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --continue) CONTINUE_ON_ERROR=1; shift ;;
    --write-noise) WRITE_NOISE=1; shift ;;
    --cfg-dir) CFG_DIR="$2"; shift 2 ;;
    --input-dir) INPUT_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --script) PY_SCRIPT="$2"; shift 2 ;;
    --extra)
      # shellcheck disable=SC2206
      EXTRA_ARGS+=($2)
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[error] 未知参数: $1" >&2; exit 1 ;;
  esac
done

if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "[error] Python 脚本不存在: $PY_SCRIPT" >&2
  exit 1
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[error] Python 不可用: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -d "$CFG_DIR" ]]; then
  echo "[error] cfg 目录不存在: $CFG_DIR" >&2
  exit 1
fi
if [[ ! -d "$INPUT_DIR" ]]; then
  echo "[error] FI 输入目录不存在: $INPUT_DIR" >&2
  exit 1
fi
if [[ ${#PAIRS[@]} -eq 0 ]]; then
  echo "[error] PAIRS 为空，请在脚本中启用至少一个立体相对" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
if [[ -n "$OUTPUT_DIR" ]]; then
  mkdir -p "$OUTPUT_DIR"
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY_LOG="${LOG_DIR}/fi_denoise_${TIMESTAMP}.summary.log"

declare -a PAIR_ID1=()
declare -a PAIR_ID2=()
declare -a PAIR_CFG=()
declare -a PAIR_FI=()

for pair in "${PAIRS[@]}"; do
  [[ -z "${pair//[[:space:]]/}" ]] && continue
  [[ "$pair" =~ ^[[:space:]]*# ]] && continue

  read -r id1 id2 _ <<< "$pair"
  if [[ -z "${id1:-}" || -z "${id2:-}" ]]; then
    echo "[error] 无效相对条目: '$pair'（需要 \"ID1 ID2\"）" >&2
    exit 1
  fi

  PAIR_ID1+=("$id1")
  PAIR_ID2+=("$id2")
  PAIR_CFG+=("${CFG_DIR}/cfg_${id1}_${id2}.yaml")
  PAIR_FI+=("${INPUT_DIR}/${id1}_${id2}_FI.txt")
done

if [[ ${#PAIR_ID1[@]} -eq 0 ]]; then
  echo "[error] 没有启用的立体相对，请修改脚本中的 PAIRS" >&2
  exit 1
fi

echo "==============================================" | tee "$SUMMARY_LOG"
echo " FI DEM point cloud denoise batch" | tee -a "$SUMMARY_LOG"
echo " MARS_ROOT  = $MARS_ROOT" | tee -a "$SUMMARY_LOG"
echo " CFG_DIR    = $CFG_DIR" | tee -a "$SUMMARY_LOG"
echo " INPUT_DIR  = $INPUT_DIR" | tee -a "$SUMMARY_LOG"
echo " OUTPUT_DIR = ${OUTPUT_DIR:-<alongside input>}" | tee -a "$SUMMARY_LOG"
echo " PYTHON     = $PYTHON_BIN" | tee -a "$SUMMARY_LOG"
echo " PY_SCRIPT  = $PY_SCRIPT" | tee -a "$SUMMARY_LOG"
echo " PAIRS      = ${#PAIR_ID1[@]}" | tee -a "$SUMMARY_LOG"
echo " DRY_RUN    = $DRY_RUN" | tee -a "$SUMMARY_LOG"
echo "==============================================" | tee -a "$SUMMARY_LOG"

ok_count=0
skip_count=0
fail_count=0

for idx in "${!PAIR_ID1[@]}"; do
  id1="${PAIR_ID1[$idx]}"
  id2="${PAIR_ID2[$idx]}"
  cfg="${PAIR_CFG[$idx]}"
  in_file="${PAIR_FI[$idx]}"
  stem="${id1}_${id2}"

  if [[ -n "$OUTPUT_DIR" ]]; then
    out_file="${OUTPUT_DIR}/${stem}_FI_denoised.txt"
  else
    out_file="${INPUT_DIR}/${stem}_FI_denoised.txt"
  fi
  stats_file="${out_file}.stats.json"

  noise_args=()
  if [[ "$WRITE_NOISE" -eq 1 ]]; then
    noise_args=(--noise-out "${out_file%_FI_denoised.txt}_FI_noise.txt")
  fi

  echo "" | tee -a "$SUMMARY_LOG"
  echo "######## [$((idx + 1))/${#PAIR_ID1[@]}] ${id1} <-> ${id2} ########" | tee -a "$SUMMARY_LOG"
  echo "[cfg] $cfg" | tee -a "$SUMMARY_LOG"
  echo "[fi]  $in_file" | tee -a "$SUMMARY_LOG"

  if [[ ! -f "$cfg" ]]; then
    echo "[skip] config 不存在: $cfg" | tee -a "$SUMMARY_LOG"
    skip_count=$((skip_count + 1))
    if [[ "$CONTINUE_ON_ERROR" -eq 0 ]]; then
      exit 1
    fi
    continue
  fi
  if [[ ! -f "$in_file" ]]; then
    echo "[skip] FI 文件不存在: $in_file" | tee -a "$SUMMARY_LOG"
    skip_count=$((skip_count + 1))
    if [[ "$CONTINUE_ON_ERROR" -eq 0 ]]; then
      exit 1
    fi
    continue
  fi

  cmd=("$PYTHON_BIN" "$PY_SCRIPT" "$in_file" -o "$out_file" --stats-out "$stats_file" "${noise_args[@]}" "${EXTRA_ARGS[@]}")
  printf '[cmd]' | tee -a "$SUMMARY_LOG"
  printf ' %q' "${cmd[@]}" | tee -a "$SUMMARY_LOG"
  printf '\n' | tee -a "$SUMMARY_LOG"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    ok_count=$((ok_count + 1))
    continue
  fi

  pair_log="${LOG_DIR}/${stem}_${TIMESTAMP}.log"
  if "${cmd[@]}" >"$pair_log" 2>&1; then
    cat "$pair_log" | tee -a "$SUMMARY_LOG"
    ok_count=$((ok_count + 1))
  else
    cat "$pair_log" | tee -a "$SUMMARY_LOG"
    echo "[fail] ${stem}_FI.txt, log=$pair_log" | tee -a "$SUMMARY_LOG"
    fail_count=$((fail_count + 1))
    if [[ "$CONTINUE_ON_ERROR" -eq 0 ]]; then
      exit 1
    fi
  fi
done

echo "" | tee -a "$SUMMARY_LOG"
echo "==============================================" | tee -a "$SUMMARY_LOG"
echo " done: ok=$ok_count skip=$skip_count fail=$fail_count summary=$SUMMARY_LOG" | tee -a "$SUMMARY_LOG"
echo "==============================================" | tee -a "$SUMMARY_LOG"

if [[ "$fail_count" -gt 0 ]]; then
  exit 1
fi
