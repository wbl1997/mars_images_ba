#!/usr/bin/env bash
# 将非原始影像的算法输出汇总到 auto_updated_results：
#   match/  —— 特征/格网/CCD 间匹配及视差可视化等
#   BA/     —— observedata + result（EO / FI / 观测）
#
# 用法：
#   bash script/update_all_result.sh
#   bash script/update_all_result.sh --dry-run
#   bash script/update_all_result.sh --dest /path/to/auto_updated_results
#   bash script/update_all_result.sh --cfg-dir /path
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MARS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DEST_ROOT="/media/wbl/Elements/paper_experiments/Mars/auto_updated_results"
CFG_DIR="${MARS_ROOT}"
DATA_DIR="${MARS_ROOT}/data"
DRY_RUN=0

# ---------- 立体相对列表（与 run_mars_batch.sh 同格式） ----------
# 每项："ID1 ID2" → 读 cfg_${ID1}_${ID2}.yaml 取 filepath
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
  # "ESP_012993_1725 ESP_013059_1725"
  # "ESP_023957_1755 ESP_024023_1755"

  # "PSP_001777_1650 PSP_001513_1655"
  # "PSP_008469_2040 PSP_008825_2040"
  # "ESP_075625_2055 ESP_075559_2055"
)

# 不拷贝的原始/中间影像扩展名
IMAGE_EXCLUDES=(
  --exclude='*.tif'
  --exclude='*.tiff'
  --exclude='*.TIF'
  --exclude='*.TIFF'
  --exclude='*.jp2'
  --exclude='*.JP2'
  --exclude='*.img'
  --exclude='*.IMG'
  --exclude='*.cub'
  --exclude='*.CUB'
  --exclude='*.png.aux.xml'
)

usage() {
  sed -n '2,12p' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)  DRY_RUN=1; shift ;;
    --dest)     DEST_ROOT="$2"; shift 2 ;;
    --cfg-dir)  CFG_DIR="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    -h|--help)  usage; exit 0 ;;
    *)
      echo "[error] 未知参数: $1" >&2
      exit 1
      ;;
  esac
done

MATCH_DEST="${DEST_ROOT}/match"
BA_DEST="${DEST_ROOT}/BA"

# ---------- 从 cfg 解析 filepath ----------
parse_cfg_filepath() {
  local cfg="$1"
  local filepath
  filepath="$(grep -E '^\s*filepath\s*:' "$cfg" | head -1 | sed -E 's/^\s*filepath\s*:\s*"?([^"#]+)"?.*/\1/' | xargs)"
  if [[ -z "${filepath:-}" ]]; then
    echo "[warn] cfg 缺少 filepath: $cfg" >&2
    return 1
  fi
  printf '%s\n' "$filepath"
}

# ---------- 按 PAIRS 收集待处理相对 ----------
declare -a PAIRS_FILEPATH=()
declare -a PAIRS_ID1=()
declare -a PAIRS_ID2=()

for pair in "${PAIRS[@]}"; do
  [[ -z "${pair//[[:space:]]/}" ]] && continue
  [[ "$pair" =~ ^[[:space:]]*# ]] && continue

  read -r id1 id2 _ <<< "$pair"
  if [[ -z "${id1:-}" || -z "${id2:-}" ]]; then
    echo "[error] 无效相对条目: '$pair'（需要 \"ID1 ID2\"）" >&2
    exit 1
  fi

  cfg="${CFG_DIR}/cfg_${id1}_${id2}.yaml"
  if [[ ! -f "$cfg" ]]; then
    echo "[error] 找不到 cfg: $cfg" >&2
    exit 1
  fi
  if ! filepath="$(parse_cfg_filepath "$cfg")"; then
    exit 1
  fi
  PAIRS_FILEPATH+=("$filepath")
  PAIRS_ID1+=("$id1")
  PAIRS_ID2+=("$id2")
done

if [[ ${#PAIRS_ID1[@]} -eq 0 ]]; then
  echo "[error] PAIRS 为空或全部被注释，请在脚本中启用至少一对" >&2
  exit 1
fi

run_rsync() {
  local src="$1"
  local dst="$2"
  shift 2
  if [[ ! -e "$src" ]]; then
    echo "[skip] 源不存在: $src"
    return 0
  fi
  mkdir -p "$dst"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] rsync -a $* \"$src\" \"$dst\""
    rsync -a -n --info=stats2 "$@" "$src" "$dst" 2>/dev/null | tail -5 || true
    return 0
  fi
  echo "[copy] $src  →  $dst"
  rsync -a --info=name0,stats1 "$@" "$src" "$dst"
}

echo "=============================================="
echo " update_all_result"
echo " DEST       = $DEST_ROOT"
echo " MATCH      = $MATCH_DEST"
echo " BA         = $BA_DEST"
echo " CFG_DIR    = $CFG_DIR"
echo " DATA_DIR   = $DATA_DIR"
echo " 相对数量   = ${#PAIRS_ID1[@]}"
echo " DRY_RUN    = $DRY_RUN"
echo "=============================================="

mkdir -p "$MATCH_DEST" "$BA_DEST"

# ---------- 1) 匹配相关：downsample 下非影像算法输出 ----------
for i in "${!PAIRS_ID1[@]}"; do
  filepath="${PAIRS_FILEPATH[$i]}"
  id1="${PAIRS_ID1[$i]}"
  id2="${PAIRS_ID2[$i]}"
  pair_tag="${id1}_${id2}"
  pair_match="${MATCH_DEST}/${pair_tag}"

  echo ""
  echo "######## match: ${id1} <-> ${id2} ########"
  echo "[info] filepath = $filepath"

  for sid in "$id1" "$id2"; do
    src_ds="${filepath}/${sid}/downsample"
    if [[ ! -d "$src_ds" ]]; then
      echo "[skip] 无 downsample: $src_ds"
      continue
    fi
    # 保持 {pair}/ID/downsample/... 结构
    dst_ds="${pair_match}/${sid}/"
    mkdir -p "$dst_ds"
    # 只同步 downsample 目录本身到 ID/ 下
    run_rsync "${src_ds}" "$dst_ds" "${IMAGE_EXCLUDES[@]}"
  done
done

# ---------- 2) BA 相关：observedata + result ----------
echo ""
echo "######## BA: observedata / result ########"

if [[ -d "${DATA_DIR}/observedata" ]]; then
  run_rsync "${DATA_DIR}/observedata/" "${BA_DEST}/observedata/"
else
  echo "[skip] 无 observedata: ${DATA_DIR}/observedata"
fi

if [[ -d "${DATA_DIR}/result" ]]; then
  run_rsync "${DATA_DIR}/result/" "${BA_DEST}/result/" \
    --exclude='result（复件）' \
    --exclude='新建文件夹'
else
  echo "[skip] 无 result: ${DATA_DIR}/result"
fi

# EO 初值有时也在 data/EO；BA 更新后的在 result/*_EOre.txt。
# 若需要同步输入 EO，取消下面注释：
# if [[ -d "${DATA_DIR}/EO" ]]; then
#   run_rsync "${DATA_DIR}/EO/" "${BA_DEST}/EO/"
# fi

echo ""
echo "========== 完成 =========="
echo "  match → ${MATCH_DEST}"
echo "  BA    → ${BA_DEST}/observedata , ${BA_DEST}/result"
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "  (dry-run，未实际写入)"
fi
