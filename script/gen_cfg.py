#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为立体相对生成 Mars 程序所用的 cfg.yaml。

根据 srcfilepath 下两景 IMG 自动读取 LINES / CCD 数，算法参数默认从模板 cfg 复制。
模板格式对齐 cfg_ESP_013110_1325_ESP_013611_1325.yaml（含 CCD_begin/end、local_affine、densify 等）。

用法：
  python gen_cfg.py <id1> <id2> [options]

  -i, --img-dir     原始 IMG 父目录（拼 {img-dir}/{id}/）
  -f, --filepath    处理输出根目录（cfg.dataset.filepath）
  -o, --out         输出 cfg 路径（默认：<mars_root>/cfg_{id1}_{id2}.yaml）
  -t, --template    算法参数模板（默认：cfg_ESP_013110_1325_ESP_013611_1325.yaml）
  -r, --root        Mars 工程根目录
  --rows1/--rows2   覆盖 LINES
  --cols            列数（默认 2048）
  --ccd-num         覆盖 CCD 数量
  --ccd-begin/--ccd-end  CCD 处理区间（默认 0 / -1）
  --preprocess ...  覆盖 steps 开关（true/false）
  --keep-steps      保留模板中的 steps（默认重置为全流程开启）
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MARS_ROOT = SCRIPT_DIR.parent
DEFAULT_TEMPLATE_NAME = "cfg_ESP_013110_1325_ESP_013611_1325.yaml"

# 无模板可用时的算法参数兜底（与 cfg_ESP_013110_1325_ESP_013611_1325.yaml 对齐）
DEFAULT_ALGO = """\
# 大起伏优化：放宽搜索/阈值，略缩小匹配窗口；先重跑匹配再 BA
steps:
  preprocess: true
  intra_ccd_mosaic: true
  downsample: true
  feature_mosaic: true
  feature_extract: true
  feature_match: true
  grid_match: true
  inter_ccd_match: true
  ba: true

mosaic:
  overlap_samples: 48

feature_extract:
  channel: 6
  localmax_win: [64, 51, 51, 31, 15]

feature_match:
  # --- 1) fenfu_match ---
  pyramid_open: true
  coarse_window_size: 17
  guided_window_factor: 6
  final_layer_window_size: 21
  search_range_factor: 140
  coarse_cc_threshold: 0.50
  guided_cc_threshold: 0.45
  final_layer_cc_threshold: 0.42
  ransac_sigma_factor: 3.5
  ransac_iterations: 30000
  use_robust_scoreb: false
  use_affine_patch_score: false
  use_local_affine: true
  local_tiles_r: 12
  local_tiles_c: 10
  local_affine_sigma: 220.0
  local_min_points: 8
  local_keep_score_min: 0.42
  # --- 2) densify_grid ---
  densify_grid: true
  densify_batch_r: 64
  densify_batch_c: 64
  densify_window_size: 15
  densify_search_range: 48
  densify_cc_threshold: 0.45
  densify_knn: 8
  # --- 3) iterative_refinement (Fea IR / limit_fea) ---
  iterative_refinement_iterations: 3
  iterative_window_size: 15
  iterative_search_range: 192
  iterative_cc_threshold: 0.65
  lambda2: 0.1
  iterative_seed_radius: 500
  iterative_seed_min: 3

grid_match:
  refinement_iterations: 3
  batch_size: 48
  window_size: 15
  search_range: 128
  cc_threshold: 0.55
  use_robust_scoreb: false
  lambda2: 0.1
  followup_window_size: 15
  followup_search_range: 4
  followup_cc_threshold: 0.55

inter_ccd_match:
  intra_window_size: 15
  intra_search_range: 80
  intra_batch_size_r: 10
  intra_batch_size_c: 48
  intra_cc_threshold: 0.30
  cross_window_size: 15
  cross_search_range: 150
  cross_cc_threshold: 0.55
  use_robust_scoreb: false
  run_global_refine: true

ba:
  # FI 观测来源: feature(_match) | grid(_grid)
  fi_source: grid
  # 是否先做特征点 BA 更新 EO
  run_feature_ba: true
  # true=优先 Jitter，失败回退 Block；false=直接 Block
  use_jitter: true
  block_max_iterations: 100
  jitter_max_iterations: 50
  fi_max_iterations: 30
  cauchy_loss_ba: 1.0
  cauchy_loss_fi: 0.5
  function_tolerance: 1.0e-10
  # Tichu_CX 粗差剔除（mark=0/1 特征/格网；大起伏可放宽）
  tichu_affine_max_residual: 500.0
  tichu_affine_max_vx: 150.0
  tichu_affine_max_vy: 250.0
  tichu_local_radius: 500.0
  tichu_sigma_factor: 3.0
"""

STEP_KEYS = (
    "preprocess",
    "intra_ccd_mosaic",
    "downsample",
    "feature_mosaic",
    "feature_extract",
    "feature_match",
    "grid_match",
    "inter_ccd_match",
    "ba",
)


def read_pds3_label(img_path: Path, max_bytes: int = 2_000_000) -> str:
    with open(img_path, "rb") as f:
        text = f.read(max_bytes).decode("latin-1", errors="ignore")
    end = re.search(r"^\s*END\s*$", text, re.M)
    return text[: end.end()] if end else text


def read_lines_from_img(img_path: Path) -> int:
    label = read_pds3_label(img_path)
    txt = re.sub(r"/\*.*?\*/", " ", label, flags=re.S)
    m_img = re.search(r"OBJECT\s*=\s*IMAGE\b(.*?)END_OBJECT\s*=\s*IMAGE", txt, re.S | re.I)
    block = m_img.group(1) if m_img else txt
    m = re.search(r"\bLINES\s*=\s*(\d+)", block, re.I)
    if not m:
        raise ValueError(f"无法从 {img_path} 解析 LINES")
    return int(m.group(1))


def count_ccds(img_dir: Path, product_id: str) -> int:
    n = 0
    for i in range(10):
        p0 = img_dir / product_id / f"{product_id}_RED{i}_0.IMG"
        p1 = img_dir / product_id / f"{product_id}_RED{i}_0.img"
        if p0.exists() or p1.exists():
            n += 1
    return n


def find_red0(img_dir: Path, product_id: str) -> Path:
    for name in (f"{product_id}_RED0_0.IMG", f"{product_id}_RED0_0.img"):
        p = img_dir / product_id / name
        if p.exists():
            return p
    raise FileNotFoundError(f"未找到 {img_dir}/{product_id}/{product_id}_RED0_0.IMG")


def extract_algo_from_template(template_path: Path, keep_steps: bool = False) -> str:
    """从模板中截取算法参数段。

    默认丢弃模板里的 steps（避免把旧相对的流程开关带过来），
    只保留 mosaic / feature_* / grid_match / inter_ccd_match。
    """
    text = template_path.read_text(encoding="utf-8")
    m = re.search(r"(?m)^steps:\s*$", text)
    if not m:
        # 允许模板在 steps 前有注释行
        m_comment = re.search(r"(?m)^(?:#.*\n)*steps:\s*$", text)
        if m_comment:
            m = re.search(r"(?m)^steps:\s*$", text)
    if not m:
        print(f"[warn] 模板 {template_path} 中未找到 steps:，使用内置默认算法参数")
        return DEFAULT_ALGO

    # 若 steps 前有说明注释，一并保留（与 013110 模板一致）
    start = m.start()
    line_start = text.rfind("\n", 0, start) + 1
    prefix = text[line_start:start]
    if prefix.lstrip().startswith("#"):
        # 向上吞并连续注释行
        block_start = line_start
        while block_start > 0:
            prev_nl = text.rfind("\n", 0, block_start - 1)
            prev_line = text[prev_nl + 1 : block_start]
            if prev_line.lstrip().startswith("#"):
                block_start = prev_nl + 1
            else:
                break
        start = block_start

    algo = text[start:].rstrip() + "\n"
    if keep_steps:
        return algo

    m_rest = re.search(
        r"(?m)^(mosaic|feature_extract|feature_match|grid_match|inter_ccd_match):\s*$",
        algo,
    )
    if not m_rest:
        return DEFAULT_ALGO

    m_default_mosaic = re.search(r"(?m)^mosaic:\s*$", DEFAULT_ALGO)
    if not m_default_mosaic:
        return DEFAULT_ALGO
    default_steps = DEFAULT_ALGO[: m_default_mosaic.start()]
    return default_steps + algo[m_rest.start() :]


def apply_step_overrides(algo: str, overrides: dict) -> str:
    if not any(v is not None for v in overrides.values()):
        return algo
    out = algo
    for key, val in overrides.items():
        if val is None:
            continue
        yaml_val = "true" if val else "false"
        new_out, n = re.subn(
            rf"(?m)^(\s*{re.escape(key)}:\s*)(true|false)\s*$",
            rf"\g<1>{yaml_val}",
            out,
            count=1,
        )
        if n == 0:
            print(f"[warn] 未在算法段找到 steps.{key}，跳过覆盖")
        else:
            out = new_out
    return out


def build_dataset_block(
    srcfilepath: str,
    filepath: str,
    id1: str,
    id2: str,
    rows1: int,
    rows2: int,
    cols: int,
    ccd_num: int,
    ccd_begin: int = 0,
    ccd_end: int = -1,
) -> str:
    src = srcfilepath.rstrip("/")
    fp = filepath.rstrip("/")
    return (
        "dataset:\n"
        f'  srcfilepath: "{src}"\n'
        f'  filepath: "{fp}"\n'
        f'  xulie_ID1: "{id1}"\n'
        f'  xulie_ID2: "{id2}"\n'
        f"  rows1: {rows1}\n"
        f"  rows2: {rows2}\n"
        f"  cols: {cols}\n"
        f"  CCD_num: {ccd_num}\n"
        f"  CCD_begin: {ccd_begin}\n"
        f"  CCD_end: {ccd_end}\n"
    )


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"无效布尔值: {v}")


def main():
    ap = argparse.ArgumentParser(description="生成 Mars 立体相对 cfg.yaml")
    ap.add_argument("id1", help="左影像序列号，如 ESP_013110_1325")
    ap.add_argument("id2", help="右影像序列号，如 ESP_013611_1325")
    ap.add_argument("-r", "--root", default=None, help="Mars 工程根目录")
    ap.add_argument("-i", "--img-dir", required=True, help="原始 IMG 父目录")
    ap.add_argument("-f", "--filepath", required=True, help="处理输出根目录")
    ap.add_argument("-o", "--out", default=None, help="输出 cfg 路径")
    ap.add_argument(
        "-t",
        "--template",
        default=None,
        help=f"算法参数模板 cfg（默认：<mars_root>/{DEFAULT_TEMPLATE_NAME}）",
    )
    ap.add_argument(
        "--keep-steps",
        action="store_true",
        help="保留模板中的 steps 开关（默认重置为全流程 true）",
    )
    ap.add_argument("--rows1", type=int, default=None)
    ap.add_argument("--rows2", type=int, default=None)
    ap.add_argument("--cols", type=int, default=2048)
    ap.add_argument("--ccd-num", type=int, default=None)
    ap.add_argument("--ccd-begin", type=int, default=0)
    ap.add_argument("--ccd-end", type=int, default=-1)
    for k in STEP_KEYS:
        ap.add_argument(f"--{k.replace('_', '-')}", dest=k, type=str2bool, default=None)
    args = ap.parse_args()

    mars_root = Path(args.root).resolve() if args.root else DEFAULT_MARS_ROOT
    img_dir = Path(args.img_dir).resolve()
    filepath = str(Path(args.filepath).resolve())
    srcfilepath = str(img_dir)

    red0_1 = find_red0(img_dir, args.id1)
    red0_2 = find_red0(img_dir, args.id2)
    rows1 = args.rows1 if args.rows1 is not None else read_lines_from_img(red0_1)
    rows2 = args.rows2 if args.rows2 is not None else read_lines_from_img(red0_2)

    ccd1 = count_ccds(img_dir, args.id1)
    ccd2 = count_ccds(img_dir, args.id2)
    if ccd1 == 0 or ccd2 == 0:
        print(f"[error] CCD 数为 0：{args.id1}={ccd1}, {args.id2}={ccd2}", file=sys.stderr)
        sys.exit(1)
    if ccd1 != ccd2:
        print(f"[warn] 两景 CCD 数不一致：{args.id1}={ccd1}, {args.id2}={ccd2}，取较小值")
    ccd_num = args.ccd_num if args.ccd_num is not None else min(ccd1, ccd2)

    template = (
        Path(args.template).resolve()
        if args.template
        else (mars_root / DEFAULT_TEMPLATE_NAME)
    )
    if not template.exists():
        # 兼容旧默认
        fallback = mars_root / "cfg.yaml"
        if fallback.exists():
            print(f"[warn] 模板不存在：{template}，回退到 {fallback}")
            template = fallback

    if template.exists():
        print(f"[info] 算法参数模板：{template}")
        algo = extract_algo_from_template(template, keep_steps=args.keep_steps)
    else:
        print(f"[warn] 模板不存在：{template}，使用内置默认算法参数")
        algo = DEFAULT_ALGO

    step_overrides = {k: getattr(args, k) for k in STEP_KEYS}
    algo = apply_step_overrides(algo, step_overrides)

    dataset = build_dataset_block(
        srcfilepath=srcfilepath,
        filepath=filepath,
        id1=args.id1,
        id2=args.id2,
        rows1=rows1,
        rows2=rows2,
        cols=args.cols,
        ccd_num=ccd_num,
        ccd_begin=args.ccd_begin,
        ccd_end=args.ccd_end,
    )

    out_path = (
        Path(args.out).resolve()
        if args.out
        else mars_root / f"cfg_{args.id1}_{args.id2}.yaml"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# Auto-generated by gen_cfg.py for stereo pair "
        f"{args.id1} / {args.id2}\n"
    )
    out_path.write_text(header + "\n" + dataset + "\n" + algo, encoding="utf-8")

    print(
        f"[info] rows1={rows1}, rows2={rows2}, cols={args.cols}, "
        f"CCD_num={ccd_num}, CCD_range=[{args.ccd_begin},{args.ccd_end})"
    )
    print(f"[done] 已生成：{out_path}")


if __name__ == "__main__":
    main()
