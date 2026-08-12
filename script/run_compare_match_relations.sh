#!/usr/bin/env bash
source activate radar_env
python3 "$(dirname "$0")/compare_match_relations.py" \
  "/media/wbl/KESU/硕士相关材料/MARS_zl/代码/重建/data/new/PSP_008469_2040/downsample/0" \
  "/media/wbl/Elements/paper_experiments/Mars/new/PSP_008469_2040/downsample/0"
