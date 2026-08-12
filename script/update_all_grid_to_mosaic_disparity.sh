#!/usr/bin/env bash
# 基于 auto_updated_results/match 中的匹配结果，批量生成论文用稀疏匹配视差图。
#
# 新版主图：
#   (a) 左影像 mosaic 上的稀疏匹配点 + 视差箭头；
#   (b,c) 一个局部区域的左右影像（纵向排布）。
# 分图：*_sparse_match_global（顶栏横向色条 + 右下外侧纵向比例尺，无标题/编号）
#       *_sparse_match_region（左右纵向，无标题/编号）
#
# 背景图默认由每个 cfg 中的 dataset.filepath 自动定位：
#   {dataset.filepath}/{ID}/downsample/0/mosaic_ds4.tif
#
# 输入：
#   {MATCH_ROOT}/{ID1}_{ID2}/{ID1}/downsample/0/
#   {MATCH_ROOT}/{ID1}_{ID2}/{ID2}/downsample/0/
#
# 输出：
#   {PLOT_ROOT}/{ID1}_{ID2}/match/
#       {ID1}_match_sparse_match_disparity_local.png
#       {ID1}_match_sparse_match_global.png
#       {ID1}_match_sparse_match_region.png
#
# 用法：
#   bash script/update_all_grid_to_mosaic_disparity.sh
#   bash script/update_all_grid_to_mosaic_disparity.sh --dry-run
#   bash script/update_all_grid_to_mosaic_disparity.sh --continue
#   bash script/update_all_grid_to_mosaic_disparity.sh --output-mode png+pdf
#   bash script/update_all_grid_to_mosaic_disparity.sh --sources match,grid
#   bash script/update_all_grid_to_mosaic_disparity.sh --sources match
#   bash script/update_all_grid_to_mosaic_disparity.sh --sources grid
#
# 手工指定局部区域时，可通过 --extra 继续向 Python 传参，例如：
#   --extra "--roi1 18000 23000 3000 8000"
# 坐标轴/legend 字号：默认 --font-scale 2.4（不影响图幅布局）
#   bash script/update_all_grid_to_mosaic_disparity.sh --font-scale 2.8
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MARS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------- 路径配置 ----------
PY_SCRIPT="${SCRIPT_DIR}/grid_to_mosaic_disparity_paper_panels.py"
CFG_DIR="${MARS_ROOT}"
MATCH_ROOT="/media/wbl/Elements/paper_experiments/Mars/auto_updated_results/match"
PLOT_ROOT="${MATCH_ROOT}/0plot"
LOG_DIR="${MARS_ROOT}/logs/update_all_grid_to_mosaic_disparity"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# ---------- 立体像对列表 ----------
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
  "ESP_023957_1755 ESP_024023_1755"

  "PSP_001777_1650 PSP_001513_1655"
  "PSP_008469_2040 PSP_008825_2040"
  "ESP_075625_2055 ESP_075559_2055"
)

# ---------- 默认绘图参数 ----------
DRY_RUN=0
CONTINUE_ON_ERROR=0
OUTPUT_MODE="png"

# 默认同时跑 match + grid；可用 --sources 覆盖（如 match 或 grid）。
# grid 按旧方式输出标量面板（--fill-gaps --despike）。
SOURCES=("match" "grid")

MOSAIC_NAME="mosaic_ds4.tif"
FIGURE_WIDTH_MM="178"
# 仅放大坐标轴/legend/colorbar 字号（不改图幅与标注位置）
FONT_SCALE="2.4"
GLOBAL_MAX_ARROWS="1400"
LOCAL_MAX_POINTS="120"
ROI_HEIGHT="5000"
ROI_WIDTH="5000"
GLOBAL_ARROW_TARGET_FRACTION="0.05"
LOCAL_ARROW_TARGET_FRACTION="0.13"

EXTRA_ARGS=()

need_value() {
  local opt="$1"
  local value="${2:-}"
  if [[ -z "${value}" ]]; then
    echo "[error] ${opt} 缺少参数" >&2
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --continue)
      CONTINUE_ON_ERROR=1
      shift
      ;;
    --output-mode)
      need_value "$1" "${2:-}"
      OUTPUT_MODE="$2"
      shift 2
      ;;
    --match-root)
      need_value "$1" "${2:-}"
      MATCH_ROOT="$2"
      PLOT_ROOT="${MATCH_ROOT}/0plot"
      shift 2
      ;;
    --plot-root)
      need_value "$1" "${2:-}"
      PLOT_ROOT="$2"
      shift 2
      ;;
    --cfg-dir)
      need_value "$1" "${2:-}"
      CFG_DIR="$2"
      shift 2
      ;;
    --log-dir)
      need_value "$1" "${2:-}"
      LOG_DIR="$2"
      shift 2
      ;;
    --python)
      need_value "$1" "${2:-}"
      PYTHON_BIN="$2"
      shift 2
      ;;
    --script)
      need_value "$1" "${2:-}"
      PY_SCRIPT="$2"
      shift 2
      ;;
    --sources)
      need_value "$1" "${2:-}"
      IFS=',' read -r -a SOURCES <<< "$2"
      shift 2
      ;;
    --mosaic-name)
      need_value "$1" "${2:-}"
      MOSAIC_NAME="$2"
      shift 2
      ;;
    --max-arrows)
      need_value "$1" "${2:-}"
      GLOBAL_MAX_ARROWS="$2"
      shift 2
      ;;
    --local-max-points)
      need_value "$1" "${2:-}"
      LOCAL_MAX_POINTS="$2"
      shift 2
      ;;
    --roi-height)
      need_value "$1" "${2:-}"
      ROI_HEIGHT="$2"
      shift 2
      ;;
    --roi-width)
      need_value "$1" "${2:-}"
      ROI_WIDTH="$2"
      shift 2
      ;;
    --font-scale)
      need_value "$1" "${2:-}"
      FONT_SCALE="$2"
      shift 2
      ;;
    --figure-width-mm)
      need_value "$1" "${2:-}"
      FIGURE_WIDTH_MM="$2"
      shift 2
      ;;
    --extra)
      need_value "$1" "${2:-}"
      # 将一个字符串按 shell 单词拆分，便于传入多个 Python 参数。
      # shellcheck disable=SC2206
      _tmp_extra=($2)
      EXTRA_ARGS+=("${_tmp_extra[@]}")
      shift 2
      ;;
    -h|--help)
      sed -n '2,35p' "$0"
      exit 0
      ;;
    *)
      echo "[error] 未知参数: $1" >&2
      exit 1
      ;;
  esac
done

if [[ "${OUTPUT_MODE}" != "png" && "${OUTPUT_MODE}" != "png+pdf" ]]; then
  echo "[error] --output-mode 必须是 png 或 png+pdf，当前: ${OUTPUT_MODE}" >&2
  exit 1
fi

for src in "${SOURCES[@]}"; do
  if [[ "${src}" != "grid" && "${src}" != "match" ]]; then
    echo "[error] --sources 仅支持 match / grid，当前含: ${src}" >&2
    exit 1
  fi
done

if [[ ! -f "${PY_SCRIPT}" ]]; then
  echo "[error] Python 脚本不存在: ${PY_SCRIPT}" >&2
  exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[error] Python 不可用: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -d "${MATCH_ROOT}" ]]; then
  echo "[error] 输入目录不存在: ${MATCH_ROOT}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}" "${PLOT_ROOT}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY_LOG="${LOG_DIR}/batch_${TIMESTAMP}.summary.log"

{
  echo "=============================================="
  echo " update_all_grid_to_mosaic_disparity"
  echo " MATCH_ROOT   = ${MATCH_ROOT}"
  echo " PLOT_ROOT    = ${PLOT_ROOT}"
  echo " PY_SCRIPT    = ${PY_SCRIPT}"
  echo " CFG_DIR      = ${CFG_DIR}"
  echo " SOURCES      = ${SOURCES[*]}"
  echo " OUT_MODE     = ${OUTPUT_MODE}"
  echo " MOSAIC_NAME  = ${MOSAIC_NAME}"
  echo " FIG_WIDTH_MM = ${FIGURE_WIDTH_MM}"
  echo " FONT_SCALE   = ${FONT_SCALE}"
  echo " MAX_ARROWS   = ${GLOBAL_MAX_ARROWS}"
  echo " LOCAL_POINTS = ${LOCAL_MAX_POINTS}"
  echo " ROI_SIZE     = ${ROI_HEIGHT} x ${ROI_WIDTH}"
  echo "=============================================="
} | tee "${SUMMARY_LOG}"

ok_count=0
fail_count=0
skip_count=0
failed_jobs=()

total_pairs=0
for pair in "${PAIRS[@]}"; do
  [[ -z "${pair//[[:space:]]/}" ]] && continue
  [[ "${pair}" =~ ^[[:space:]]*# ]] && continue
  total_pairs=$((total_pairs + 1))
done

idx=0
for pair in "${PAIRS[@]}"; do
  [[ -z "${pair//[[:space:]]/}" ]] && continue
  [[ "${pair}" =~ ^[[:space:]]*# ]] && continue

  read -r ID1 ID2 EXTRA <<< "${pair}"
  if [[ -z "${ID1:-}" || -z "${ID2:-}" || -n "${EXTRA:-}" ]]; then
    echo "[error] 无效像对条目: '${pair}'（需要严格的 \"ID1 ID2\"）" >&2
    exit 1
  fi

  idx=$((idx + 1))
  CFG="${CFG_DIR}/cfg_${ID1}_${ID2}.yaml"
  PAIR_TAG="${ID1}_${ID2}"
  PAIR_IN="${MATCH_ROOT}/${PAIR_TAG}"
  LEFT_DIR="${PAIR_IN}/${ID1}/downsample/0"
  RIGHT_DIR="${PAIR_IN}/${ID2}/downsample/0"

  echo ""
  echo "######## [${idx}/${total_pairs}] ${ID1}  <->  ${ID2} ########" \
    | tee -a "${SUMMARY_LOG}"

  if [[ ! -f "${CFG}" ]]; then
    echo "[skip] 找不到 cfg: ${CFG}" | tee -a "${SUMMARY_LOG}"
    skip_count=$((skip_count + 1))
    failed_jobs+=("${PAIR_TAG} (missing cfg)")
    if [[ "${CONTINUE_ON_ERROR}" -eq 0 ]]; then
      exit 1
    fi
    continue
  fi

  if [[ ! -d "${LEFT_DIR}" ]]; then
    echo "[skip] 左 level-0 不存在: ${LEFT_DIR}" | tee -a "${SUMMARY_LOG}"
    skip_count=$((skip_count + 1))
    failed_jobs+=("${PAIR_TAG} (missing left_dir)")
    if [[ "${CONTINUE_ON_ERROR}" -eq 0 ]]; then
      exit 1
    fi
    continue
  fi

  if [[ ! -d "${RIGHT_DIR}" ]]; then
    echo "[skip] 右 level-0 不存在: ${RIGHT_DIR}" | tee -a "${SUMMARY_LOG}"
    skip_count=$((skip_count + 1))
    failed_jobs+=("${PAIR_TAG} (missing right_dir)")
    if [[ "${CONTINUE_ON_ERROR}" -eq 0 ]]; then
      exit 1
    fi
    continue
  fi

  if [[ ! -f "${LEFT_DIR}/mosaic.txt" || ! -f "${RIGHT_DIR}/mosaic.txt" ]]; then
    echo "[skip] 缺少 mosaic.txt（left/right）" | tee -a "${SUMMARY_LOG}"
    skip_count=$((skip_count + 1))
    failed_jobs+=("${PAIR_TAG} (missing mosaic.txt)")
    if [[ "${CONTINUE_ON_ERROR}" -eq 0 ]]; then
      exit 1
    fi
    continue
  fi

  for SOURCE in "${SOURCES[@]}"; do
    OUT_DIR="${PLOT_ROOT}/${PAIR_TAG}/${SOURCE}"
    PAIR_LOG="${LOG_DIR}/${PAIR_TAG}_${SOURCE}_${TIMESTAMP}.log"
    mkdir -p "${OUT_DIR}"

    echo ""
    echo "---- source=${SOURCE} → ${OUT_DIR} ----" | tee -a "${SUMMARY_LOG}"

    CMD=(
      "${PYTHON_BIN}" "${PY_SCRIPT}" "${CFG}"
      --source "${SOURCE}"
      --left-dir "${LEFT_DIR}"
      --right-dir "${RIGHT_DIR}"
      --out-dir "${OUT_DIR}"
      --output-mode "${OUTPUT_MODE}"
      --prefix "${ID1}_${SOURCE}"
      --figure-width-mm "${FIGURE_WIDTH_MM}"
      --font-scale "${FONT_SCALE}"
    )

    if [[ "${SOURCE}" == "match" ]]; then
      # 新的审稿回复图：只输出全局箭头 + 两个局部左右匹配区域。
      CMD+=(
        --mosaic-name "${MOSAIC_NAME}"
        --max-arrows "${GLOBAL_MAX_ARROWS}"
        --local-max-points "${LOCAL_MAX_POINTS}"
        --roi-height "${ROI_HEIGHT}"
        --roi-width "${ROI_WIDTH}"
        --global-arrow-target-fraction "${GLOBAL_ARROW_TARGET_FRACTION}"
        --local-arrow-target-fraction "${LOCAL_ARROW_TARGET_FRACTION}"
      )
    else
      # 保留 grid 的旧标量面板逻辑。
      CMD+=(
        --fill-gaps
        # --despike
      )
    fi

    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
      CMD+=("${EXTRA_ARGS[@]}")
    fi

    {
      echo "[info] cfg       = ${CFG}"
      echo "[info] left_dir  = ${LEFT_DIR}"
      echo "[info] right_dir = ${RIGHT_DIR}"
      echo "[info] out_dir   = ${OUT_DIR}"
      echo "[info] log       = ${PAIR_LOG}"
      printf "[info] command   ="
      printf " %q" "${CMD[@]}"
      printf "\n"
    } | tee -a "${SUMMARY_LOG}"

    if [[ "${DRY_RUN}" -eq 1 ]]; then
      ok_count=$((ok_count + 1))
      continue
    fi

    set +e
    "${CMD[@]}" 2>&1 | tee "${PAIR_LOG}"
    rc=${PIPESTATUS[0]}
    set -e

    if [[ "${rc}" -eq 0 ]]; then
      echo "[ok] ${PAIR_TAG} / ${SOURCE} 完成" | tee -a "${SUMMARY_LOG}"
      ok_count=$((ok_count + 1))
    else
      echo "[fail] ${PAIR_TAG} / ${SOURCE} 失败 (exit=${rc})，详见 ${PAIR_LOG}" \
        | tee -a "${SUMMARY_LOG}"
      fail_count=$((fail_count + 1))
      failed_jobs+=("${PAIR_TAG}/${SOURCE} (exit=${rc})")
      if [[ "${CONTINUE_ON_ERROR}" -eq 0 ]]; then
        echo "[abort] 已中止（可用 --continue）" | tee -a "${SUMMARY_LOG}"
        exit "${rc}"
      fi
    fi
  done
done

echo ""
echo "========== 批量结果 ==========" | tee -a "${SUMMARY_LOG}"
echo "  成功: ${ok_count}" | tee -a "${SUMMARY_LOG}"
echo "  失败: ${fail_count}" | tee -a "${SUMMARY_LOG}"
echo "  跳过: ${skip_count}" | tee -a "${SUMMARY_LOG}"
echo "  输出: ${PLOT_ROOT}" | tee -a "${SUMMARY_LOG}"
echo "  汇总: ${SUMMARY_LOG}" | tee -a "${SUMMARY_LOG}"

if [[ ${#failed_jobs[@]} -gt 0 ]]; then
  echo "  失败项:" | tee -a "${SUMMARY_LOG}"
  for p in "${failed_jobs[@]}"; do
    echo "    - ${p}" | tee -a "${SUMMARY_LOG}"
  done
fi

if [[ "${fail_count}" -gt 0 || "${skip_count}" -gt 0 ]]; then
  exit 1
fi

echo "[done] 全部稀疏匹配视差图生成完成 → ${PLOT_ROOT}"