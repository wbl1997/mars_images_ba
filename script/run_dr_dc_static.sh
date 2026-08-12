#!/usr/bin/env bash
set -euo pipefail

source activate radar_env

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# INPUT_PATH="${1:-/media/wbl/KESU/硕士相关材料/MARS_zl/代码/重建/data/new/PSP_008469_2040/downsample/0}"
INPUT_PATH="${1:-/media/wbl/Elements/paper_experiments/Mars/new/ESP_075625_2055/downsample/0}"
OUT_DIR="${2:-$SCRIPT_DIR/dr_dc_plots}"
JSON_OUT="${3:-$OUT_DIR/dr_dc_stats.json}"

python3 "$SCRIPT_DIR/dr_dc_static.py" \
  "$INPUT_PATH" \
  --pattern "*_intra_match.txt" \
  --out-dir "$OUT_DIR" \
  --json-out "$JSON_OUT" --filter-outliers
