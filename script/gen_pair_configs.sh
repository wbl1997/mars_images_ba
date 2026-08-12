#!/usr/bin/env bash
# 批量生成立体相对的 EO / SPICE / cfg 配置文件。
# 调用三个 Python 脚本：
#   1) gen_eo_inputfile.py   -> data/EO/{pid}.txt + data/EO/{pid}/
#   2) hirise2mk_online.py   -> data/EO/{pid}_pz.txt (+ 下载内核)
#   3) gen_cfg.py            -> cfg_{id1}_{id2}.yaml
#
# cfg 格式对齐：cfg_ESP_013110_1325_ESP_013611_1325.yaml
#   - dataset 含 CCD_begin / CCD_end
#   - feature_match 含 local_affine / densify_*
#   - 新相对默认 steps 全开（preprocess→ba）；可用 --keep-steps 沿用模板 steps
#
# 用法：
#   bash script/gen_pair_configs.sh
#   bash script/gen_pair_configs.sh --skip-download   # 不重新下载 SPICE 内核
#   bash script/gen_pair_configs.sh --keep-steps      # 保留模板中的 steps 开关
#   bash script/gen_pair_configs.sh --dry-run
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MARS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---------- 路径配置（按需修改） ----------
IMG_DIR="/media/wbl/Elements/data/Mars"
FILEPATH="/media/wbl/Elements/paper_experiments/Mars/new"
KERNEL_DEST="/media/wbl/Elements/paper_experiments/Mars/kernals/MRO"
EO_DIR="${MARS_ROOT}/data/EO"
CFG_OUT_DIR="${MARS_ROOT}"          # 生成的 cfg_{id1}_{id2}.yaml 输出目录
# 算法参数模板：与大起伏优化配置一致
CFG_TEMPLATE="${MARS_ROOT}/cfg_ESP_013110_1325_ESP_013611_1325.yaml"

# hirise2mk_online.py 需要 requests；与 process.sh 一致，默认用 radar_env
CONDA_ENV="${CONDA_ENV:-radar_env}"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "${HOME}/anaconda3/envs/${CONDA_ENV}/bin/python" ]]; then
    PYTHON="${HOME}/anaconda3/envs/${CONDA_ENV}/bin/python"
  elif command -v conda >/dev/null 2>&1; then
    # 兜底：尝试激活环境后再取 python
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
    PYTHON="$(command -v python)"
  else
    PYTHON="python3"
  fi
fi

# ---------- 立体相对列表（在此添加 / 注释） ----------
# 每项格式："ID1 ID2"；按产品号末位（如 _1325）分组
PAIRS=(
  # "PSP_009149_1750 PSP_009294_1750"   # Gale Crater inverted riverbed
  # "ESP_019190_1560 ESP_019335_1560"   # Eberswalde Crater deposits
  # "ESP_018701_1775 ESP_018846_1775"   # Endeavour Crater western rim
  # "ESP_012993_1725 ESP_013059_1725"   # Ganges Mensa layered deposits
  # "ESP_031916_1730 ESP_031982_1730"   # East Candor Chasma / Nia Mensa
  # "ESP_037586_1725 ESP_037731_1725"   # Candor Chasma
  # "ESP_017266_1715 ESP_017411_1715"   # Layered mesa in Candor Chasma
  "ESP_023957_1755 ESP_024023_1755"   # Gale Crater / MSL region

  # 已有或此前测试的像对：
  # "ESP_013001_1415 ESP_013146_1415"
  # "ESP_013110_1325 ESP_013611_1325"
  # "ESP_067718_1630 ESP_068444_1630"
  # "PSP_009561_2325 PSP_010260_2325"
)

# ---------- 可选开关 ----------
SKIP_DOWNLOAD=0
DRY_RUN=0
COPY_AS_ACTIVE=0   # 1 时把生成的 cfg 再复制为工程根目录 cfg.yaml
KEEP_STEPS=0       # 1 时保留模板 steps；默认新相对全流程 true
CCD_BEGIN=0
CCD_END=-1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-download) SKIP_DOWNLOAD=1; shift ;;
    --dry-run)       DRY_RUN=1; shift ;;
    --copy-active)   COPY_AS_ACTIVE=1; shift ;;
    --keep-steps)    KEEP_STEPS=1; shift ;;
    --ccd-begin)     CCD_BEGIN="$2"; shift 2 ;;
    --ccd-end)       CCD_END="$2"; shift 2 ;;
    --template)      CFG_TEMPLATE="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,18p' "$0"
      exit 0
      ;;
    *)
      echo "[error] 未知参数: $1" >&2
      exit 1
      ;;
  esac
done

GEN_EO="${SCRIPT_DIR}/gen_eo_inputfile.py"
GEN_MK="${SCRIPT_DIR}/mro_spice/hirise2mk_online.py"
GEN_CFG="${SCRIPT_DIR}/gen_cfg.py"

for f in "$GEN_EO" "$GEN_MK" "$GEN_CFG"; do
  [[ -f "$f" ]] || { echo "[error] 找不到脚本: $f" >&2; exit 1; }
done

if [[ ! -f "$CFG_TEMPLATE" ]]; then
  echo "[error] cfg 模板不存在: $CFG_TEMPLATE" >&2
  exit 1
fi

if ! "$PYTHON" -c "import math, requests" >/dev/null 2>&1; then
  echo "[error] Python 环境不可用或缺少依赖 (math/requests): $PYTHON" >&2
  echo "        请安装 requests，或指定: PYTHON=/path/to/radar_env/bin/python $0" >&2
  exit 1
fi

mkdir -p "$EO_DIR" "$CFG_OUT_DIR" "$FILEPATH"

run() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] $*"
  else
    echo "[run] $*"
    "$@"
  fi
}

gen_one_product() {
  local pid="$1"

  # 若存在 RED0_0.IMG，让 gen_eo 自己读 LINES；无需额外传 --rows
  if [[ ! -d "${IMG_DIR}/${pid}" ]]; then
    echo "[error] 影像目录不存在: ${IMG_DIR}/${pid}" >&2
    return 1
  fi

  echo "========== EO 输入: ${pid} =========="
  run "$PYTHON" "$GEN_EO" "$pid" \
    -r "$MARS_ROOT" \
    -i "$IMG_DIR"

  echo "========== SPICE MK: ${pid} =========="
  local mk_args=(
    --pid "$pid"
    --out "${EO_DIR}/${pid}_pz.txt"
    --dest "$KERNEL_DEST"
  )
  if [[ "$SKIP_DOWNLOAD" -eq 0 ]]; then
    mk_args+=(--download)
  fi
  run "$PYTHON" "$GEN_MK" "${mk_args[@]}"
}

echo "=============================================="
echo " Mars 立体相对配置批量生成"
echo " MARS_ROOT     = $MARS_ROOT"
echo " IMG_DIR       = $IMG_DIR"
echo " FILEPATH      = $FILEPATH"
echo " KERNEL_DEST   = $KERNEL_DEST"
echo " CFG_TEMPLATE  = $CFG_TEMPLATE"
echo " CCD_range     = [${CCD_BEGIN}, ${CCD_END})"
echo " KEEP_STEPS    = $KEEP_STEPS"
echo " PYTHON        = $PYTHON"
echo " 相对数量      = ${#PAIRS[@]}"
echo "=============================================="

if [[ ${#PAIRS[@]} -eq 0 ]]; then
  echo "[error] PAIRS 为空，请在脚本中填写立体相对列表" >&2
  exit 1
fi

idx=0
for pair in "${PAIRS[@]}"; do
  # 跳过空行 / 纯注释（数组里用 # 注释的项不会进入，这里再防一下）
  [[ -z "${pair//[[:space:]]/}" ]] && continue
  [[ "$pair" =~ ^[[:space:]]*# ]] && continue

  read -r ID1 ID2 _ <<< "$pair"
  if [[ -z "${ID1:-}" || -z "${ID2:-}" ]]; then
    echo "[error] 无效相对条目: '$pair'（需要 \"ID1 ID2\"）" >&2
    exit 1
  fi

  idx=$((idx + 1))
  echo ""
  echo "######## [${idx}/${#PAIRS[@]}] ${ID1}  <->  ${ID2} ########"

  gen_one_product "$ID1"
  gen_one_product "$ID2"

  CFG_OUT="${CFG_OUT_DIR}/cfg_${ID1}_${ID2}.yaml"
  echo "========== cfg: ${ID1} / ${ID2} =========="
  cfg_args=(
    "$PYTHON" "$GEN_CFG" "$ID1" "$ID2"
    -r "$MARS_ROOT"
    -i "$IMG_DIR"
    -f "$FILEPATH"
    -t "$CFG_TEMPLATE"
    -o "$CFG_OUT"
    --ccd-begin "$CCD_BEGIN"
    --ccd-end "$CCD_END"
  )
  if [[ "$KEEP_STEPS" -eq 1 ]]; then
    cfg_args+=(--keep-steps)
  fi
  run "${cfg_args[@]}"

  if [[ "$COPY_AS_ACTIVE" -eq 1 ]]; then
    if [[ "$DRY_RUN" -eq 1 ]]; then
      echo "[dry-run] cp \"$CFG_OUT\" \"${MARS_ROOT}/cfg.yaml\""
    else
      cp "$CFG_OUT" "${MARS_ROOT}/cfg.yaml"
      echo "[info] 已复制为活动配置: ${MARS_ROOT}/cfg.yaml"
    fi
  fi

  echo "[ok] ${ID1} / ${ID2} 配置完成"
  echo "     EO : ${EO_DIR}/${ID1}.txt , ${EO_DIR}/${ID1}_pz.txt"
  echo "     EO : ${EO_DIR}/${ID2}.txt , ${EO_DIR}/${ID2}_pz.txt"
  echo "     cfg: ${CFG_OUT}"
done

echo ""
echo "[done] 全部相对处理完成（共 ${idx} 对）"
