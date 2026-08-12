#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制 fenfu_match0 匹配结果：正确匹配 + 错误匹配 + 未匹配特征。

- 正确：*_RED{j}_match.txt（层末筛选后），默认绿色
- 错误：*.txt 中 bj==1 但不在 *_match.txt 里的点，红色点
- 未匹配：*.txt 中 bj==0，黄色点（只画在所在侧影像上）

坐标与 C++ fenfu_match1 一致：mosaic × (1/16) 叠到 mosaic_ds4.tif。

用法：
  python script/draw_fenfu_match0_inliers.py cfg_ESP_....yaml
  python script/draw_fenfu_match0_inliers.py cfg_....yaml --no-draw-lines
  python script/draw_fenfu_match0_inliers.py cfg_....yaml --no-draw-outliers
  python script/draw_fenfu_match0_inliers.py cfg_....yaml --no-draw-unmatched
  python script/draw_fenfu_match0_inliers.py cfg_....yaml --random-color
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import cv2
import numpy as np

Pt = Tuple[float, float, int, float, float, float]  # row,col,imgID,mrow,mcol,score
FeaPt = Tuple[float, float]  # row, col


def parse_cfg(path: Path) -> Dict[str, str]:
    """解析仓库 cfg：取 dataset 下 filepath / xulie_ID* / CCD_num。"""
    text = path.read_text(encoding="utf-8")
    out: Dict[str, str] = {}

    def grab(key: str) -> None:
        m = re.search(rf'(?m)^\s*{re.escape(key)}\s*:\s*"?([^"\n#]+)"?', text)
        if m:
            out[key] = m.group(1).strip().strip('"').strip("'")

    for k in ("filepath", "xulie_ID1", "xulie_ID2", "CCD_num"):
        grab(k)
    return out


def load_mosaic_offsets(mosaic_txt: Path, ccd_num: int) -> np.ndarray:
    """返回 shape (CCD_num, 4): off_r, ?, off_c, ? 与 C++ mosaic_c[j*4+0/2] 一致。"""
    offs = np.zeros((ccd_num, 4), dtype=np.int32)
    with mosaic_txt.open("r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            ccd_id = int(parts[0])
            if 0 <= ccd_id < ccd_num:
                offs[ccd_id, 0] = int(float(parts[1]))
                offs[ccd_id, 1] = int(float(parts[2]))
                offs[ccd_id, 2] = int(float(parts[3]))
                offs[ccd_id, 3] = int(float(parts[4]))
    return offs


def pt_key(row: float, col: float, img_id: int, mrow: float, mcol: float) -> Tuple[int, int, int, int, int]:
    """匹配点近似键（亚像素 densify 四舍五入到 0.1）。"""
    return (
        int(round(row * 10)),
        int(round(col * 10)),
        int(img_id),
        int(round(mrow * 10)),
        int(round(mcol * 10)),
    )


def read_matched_points(path: Path) -> List[Pt]:
    """读取 bj==1 的匹配行：bj row col imgID mrow mcol score。"""
    pts: List[Pt] = []
    if not path.is_file():
        return pts
    with path.open("r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 7:
                continue
            if int(float(parts[0])) != 1:
                continue
            pts.append(
                (
                    float(parts[1]),
                    float(parts[2]),
                    int(float(parts[3])),
                    float(parts[4]),
                    float(parts[5]),
                    float(parts[6]),
                )
            )
    return pts


def read_unmatched_points(path: Path) -> List[FeaPt]:
    """读取 bj==0 未匹配特征：bj row col。"""
    pts: List[FeaPt] = []
    if not path.is_file():
        return pts
    with path.open("r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            if int(float(parts[0])) != 0:
                continue
            pts.append((float(parts[1]), float(parts[2])))
    return pts


def to_ds4_xy(
    pts: List[Pt],
    left_ccd: int,
    mosaic1: np.ndarray,
    mosaic2: np.ndarray,
    ccd_num: int,
    sfr: float,
) -> Tuple[np.ndarray, np.ndarray]:
    left_xy: List[Tuple[float, float]] = []
    right_xy: List[Tuple[float, float]] = []
    for row, col, img_id, mrow, mcol, _score in pts:
        if img_id < 0 or img_id >= ccd_num:
            continue
        r1 = (row + mosaic1[left_ccd, 0]) * sfr
        c1 = (col + mosaic1[left_ccd, 2]) * sfr
        r2 = (mrow + mosaic2[img_id, 0]) * sfr
        c2 = (mcol + mosaic2[img_id, 2]) * sfr
        left_xy.append((c1, r1))
        right_xy.append((c2, r2))
    if not left_xy:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    return np.asarray(left_xy, np.float32), np.asarray(right_xy, np.float32)


def to_ds4_xy_single(
    pts: List[FeaPt],
    ccd: int,
    mosaic: np.ndarray,
    sfr: float,
) -> np.ndarray:
    """单侧特征点 CCD 坐标 -> mosaic_ds4 (col, row)。"""
    xy: List[Tuple[float, float]] = []
    for row, col in pts:
        r = (row + mosaic[ccd, 0]) * sfr
        c = (col + mosaic[ccd, 2]) * sfr
        xy.append((c, r))
    if not xy:
        return np.zeros((0, 2), np.float32)
    return np.asarray(xy, np.float32)


def _as_arr(xs: List[Tuple[float, float]]) -> np.ndarray:
    if not xs:
        return np.zeros((0, 2), np.float32)
    return np.asarray(xs, np.float32)


def collect_points(
    filepath: Path,
    id1: str,
    id2: str,
    ccd_num: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    正确 / 错误 / 未匹配。
    返回 (inl_l, inl_r, out_l, out_r, un_l, un_r)，坐标 (col, row) in mosaic_ds4。
    """
    mosaic1 = load_mosaic_offsets(filepath / id1 / "downsample" / "0" / "mosaic.txt", ccd_num)
    mosaic2 = load_mosaic_offsets(filepath / id2 / "downsample" / "0" / "mosaic.txt", ccd_num)
    sfr = 1.0 / 16.0

    all_inl: List[Tuple[float, float]] = []
    all_inr: List[Tuple[float, float]] = []
    all_outl: List[Tuple[float, float]] = []
    all_outr: List[Tuple[float, float]] = []
    all_un_l: List[Tuple[float, float]] = []
    all_un_r: List[Tuple[float, float]] = []

    for j in range(ccd_num):
        fea_l = filepath / id1 / "downsample" / "0" / f"{id1}_RED{j}.txt"
        fea_r = filepath / id2 / "downsample" / "0" / f"{id2}_RED{j}.txt"
        match_path = filepath / id1 / "downsample" / "0" / f"{id1}_RED{j}_match.txt"

        fea_pts = read_matched_points(fea_l)
        match_pts = read_matched_points(match_path)
        un_l_pts = read_unmatched_points(fea_l)
        un_r_pts = read_unmatched_points(fea_r)

        inlier_keys: Set[Tuple[int, int, int, int, int]] = set()
        for row, col, img_id, mrow, mcol, _ in match_pts:
            inlier_keys.add(pt_key(row, col, img_id, mrow, mcol))

        outlier_pts: List[Pt] = []
        for p in fea_pts:
            if pt_key(p[0], p[1], p[2], p[3], p[4]) not in inlier_keys:
                outlier_pts.append(p)

        inl_l, inl_r = to_ds4_xy(match_pts, j, mosaic1, mosaic2, ccd_num, sfr)
        out_l, out_r = to_ds4_xy(outlier_pts, j, mosaic1, mosaic2, ccd_num, sfr)
        un_l = to_ds4_xy_single(un_l_pts, j, mosaic1, sfr)
        un_r = to_ds4_xy_single(un_r_pts, j, mosaic2, sfr)

        print(
            f"[INFO] RED{j}: fea_match={len(fea_pts)}, inlier={len(match_pts)}, "
            f"outlier={len(outlier_pts)}, unmatched_L={len(un_l_pts)}, unmatched_R={len(un_r_pts)}"
        )

        if inl_l.shape[0]:
            all_inl.extend(map(tuple, inl_l.tolist()))
            all_inr.extend(map(tuple, inl_r.tolist()))
        if out_l.shape[0]:
            all_outl.extend(map(tuple, out_l.tolist()))
            all_outr.extend(map(tuple, out_r.tolist()))
        if un_l.shape[0]:
            all_un_l.extend(map(tuple, un_l.tolist()))
        if un_r.shape[0]:
            all_un_r.extend(map(tuple, un_r.tolist()))

    return (
        _as_arr(all_inl),
        _as_arr(all_inr),
        _as_arr(all_outl),
        _as_arr(all_outr),
        _as_arr(all_un_l),
        _as_arr(all_un_r),
    )


def _sample_idxs(n: int, max_draw: int, rng: random.Random) -> List[int]:
    idxs = list(range(n))
    if max_draw > 0 and n > max_draw:
        idxs = rng.sample(idxs, max_draw)
    return idxs


def draw_matches(
    img_l: np.ndarray,
    img_r: np.ndarray,
    inl_l: np.ndarray,
    inl_r: np.ndarray,
    out_l: np.ndarray,
    out_r: np.ndarray,
    un_l: np.ndarray,
    un_r: np.ndarray,
    max_draw: int,
    max_outliers: int,
    max_unmatched: int,
    thickness: int,
    radius: int,
    seed: int,
    draw_lines: bool = True,
    draw_outliers: bool = True,
    draw_unmatched: bool = True,
    random_color: bool = False,
    color_bgr: Tuple[int, int, int] = (0, 255, 0),
    outlier_color_bgr: Tuple[int, int, int] = (0, 0, 255),
    unmatched_color_bgr: Tuple[int, int, int] = (0, 255, 255),
) -> np.ndarray:
    """未匹配黄点 → 错误红点 → 正确绿点(+可选线)。"""
    if img_l.ndim == 2:
        img_l = cv2.cvtColor(img_l, cv2.COLOR_GRAY2BGR)
    if img_r.ndim == 2:
        img_r = cv2.cvtColor(img_r, cv2.COLOR_GRAY2BGR)

    h = max(img_l.shape[0], img_r.shape[0])
    w = img_l.shape[1] + img_r.shape[1]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[: img_l.shape[0], : img_l.shape[1]] = img_l
    canvas[: img_r.shape[0], img_l.shape[1] : img_l.shape[1] + img_r.shape[1]] = img_r

    rng = random.Random(seed)
    offset_x = img_l.shape[1]

    if draw_unmatched:
        for i in _sample_idxs(un_l.shape[0], max_unmatched, rng):
            p1 = (int(round(float(un_l[i, 0]))), int(round(float(un_l[i, 1]))))
            cv2.circle(canvas, p1, radius, unmatched_color_bgr, 1, lineType=cv2.LINE_AA)
        for i in _sample_idxs(un_r.shape[0], max_unmatched, rng):
            p2 = (
                int(round(float(un_r[i, 0]) + offset_x)),
                int(round(float(un_r[i, 1]))),
            )
            cv2.circle(canvas, p2, radius, unmatched_color_bgr, 1, lineType=cv2.LINE_AA)

    if draw_outliers and out_l.shape[0] > 0:
        for i in _sample_idxs(out_l.shape[0], max_outliers, rng):
            p1 = (int(round(float(out_l[i, 0]))), int(round(float(out_l[i, 1]))))
            p2 = (
                int(round(float(out_r[i, 0]) + offset_x)),
                int(round(float(out_r[i, 1]))),
            )
            cv2.circle(canvas, p1, radius, outlier_color_bgr, 1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, p2, radius, outlier_color_bgr, 1, lineType=cv2.LINE_AA)

    for i in _sample_idxs(inl_l.shape[0], max_draw, rng):
        x1, y1 = float(inl_l[i, 0]), float(inl_l[i, 1])
        x2, y2 = float(inl_r[i, 0]) + offset_x, float(inl_r[i, 1])
        if random_color:
            color = (rng.randint(32, 255), rng.randint(32, 255), rng.randint(32, 255))
        else:
            color = color_bgr
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        cv2.circle(canvas, p1, radius, color, 1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, p2, radius, color, 1, lineType=cv2.LINE_AA)
        if draw_lines:
            cv2.line(canvas, p1, p2, color, thickness, lineType=cv2.LINE_AA)

    return canvas


def main() -> int:
    ap = argparse.ArgumentParser(description="绘制 fenfu_match0 正确/错误/未匹配点")
    ap.add_argument("cfg", type=Path, help="Mars cfg yaml，如 cfg_ESP_....yaml")
    ap.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="输出图路径（默认: out/fenfu_match0_inliers.jpg）",
    )
    ap.add_argument("--max", type=int, default=8000, help="最多绘制正确匹配数（0=全部）")
    ap.add_argument(
        "--max-outliers",
        type=int,
        default=8000,
        help="最多绘制错误匹配数（0=全部）",
    )
    ap.add_argument(
        "--max-unmatched",
        type=int,
        default=8000,
        help="每侧最多绘制未匹配特征数（0=全部）",
    )
    ap.add_argument("--thickness", type=int, default=1, help="正确匹配连线线宽")
    ap.add_argument("--radius", type=int, default=3, help="圆点半径")
    ap.add_argument("--seed", type=int, default=0, help="随机颜色/抽样种子")
    ap.add_argument("--jpg-quality", type=int, default=90, help="jpg 质量")

    line_group = ap.add_mutually_exclusive_group()
    line_group.add_argument(
        "--draw-lines", dest="draw_lines", action="store_true", help="绘制正确匹配连线（默认）"
    )
    line_group.add_argument(
        "--no-draw-lines", dest="draw_lines", action="store_false", help="不画连线，只画点"
    )
    ap.set_defaults(draw_lines=True)

    out_group = ap.add_mutually_exclusive_group()
    out_group.add_argument(
        "--draw-outliers",
        dest="draw_outliers",
        action="store_true",
        help="绘制错误匹配红点（默认）",
    )
    out_group.add_argument(
        "--no-draw-outliers",
        dest="draw_outliers",
        action="store_false",
        help="不绘制错误匹配",
    )
    ap.set_defaults(draw_outliers=True)

    un_group = ap.add_mutually_exclusive_group()
    un_group.add_argument(
        "--draw-unmatched",
        dest="draw_unmatched",
        action="store_true",
        help="绘制未匹配黄点（默认）",
    )
    un_group.add_argument(
        "--no-draw-unmatched",
        dest="draw_unmatched",
        action="store_false",
        help="不绘制未匹配特征",
    )
    ap.set_defaults(draw_unmatched=True)

    color_group = ap.add_mutually_exclusive_group()
    color_group.add_argument(
        "--random-color", dest="random_color", action="store_true", help="正确匹配用随机色"
    )
    color_group.add_argument(
        "--no-random-color",
        dest="random_color",
        action="store_false",
        help="正确匹配用固定色（默认）",
    )
    ap.set_defaults(random_color=False)
    ap.add_argument(
        "--color",
        type=str,
        default="0,255,0",
        help="正确匹配 BGR，默认绿色 0,255,0",
    )
    ap.add_argument(
        "--outlier-color",
        type=str,
        default="0,0,255",
        help="错误匹配 BGR，默认红色 0,0,255",
    )
    ap.add_argument(
        "--unmatched-color",
        type=str,
        default="0,255,255",
        help="未匹配 BGR，默认黄色 0,255,255",
    )
    args = ap.parse_args()

    cfg = parse_cfg(args.cfg)
    filepath = Path(cfg.get("filepath", ""))
    id1 = cfg.get("xulie_ID1", "")
    id2 = cfg.get("xulie_ID2", "")
    ccd_num = int(float(cfg.get("CCD_num", "0")))
    if not filepath or not id1 or not id2 or ccd_num <= 0:
        print("[ERROR] cfg 缺少 filepath / xulie_ID1 / xulie_ID2 / CCD_num", file=sys.stderr)
        print(f"  parsed={cfg}", file=sys.stderr)
        return 1

    def parse_bgr(s: str, name: str) -> Tuple[int, int, int]:
        try:
            bgr = tuple(int(x.strip()) for x in s.split(","))
            if len(bgr) != 3:
                raise ValueError
            return (bgr[0], bgr[1], bgr[2])
        except ValueError:
            print(f"[ERROR] {name} 格式应为 B,G,R，收到: {s}", file=sys.stderr)
            raise

    try:
        color_bgr = parse_bgr(args.color, "--color")
        outlier_bgr = parse_bgr(args.outlier_color, "--outlier-color")
        unmatched_bgr = parse_bgr(args.unmatched_color, "--unmatched-color")
    except ValueError:
        return 1

    img_l_path = filepath / id1 / "downsample" / "0" / "mosaic_ds4.tif"
    img_r_path = filepath / id2 / "downsample" / "0" / "mosaic_ds4.tif"
    if not img_l_path.is_file() or not img_r_path.is_file():
        print(f"[ERROR] mosaic_ds4 不存在:\n  {img_l_path}\n  {img_r_path}", file=sys.stderr)
        return 1

    print(f"[INFO] L: {img_l_path}")
    print(f"[INFO] R: {img_r_path}")
    print(
        f"[INFO] CCD_num={ccd_num}, draw_lines={args.draw_lines}, "
        f"draw_outliers={args.draw_outliers}, draw_unmatched={args.draw_unmatched}, "
        f"inlier={color_bgr}, outlier={outlier_bgr}, unmatched={unmatched_bgr}"
    )

    inl_l, inl_r, out_l, out_r, un_l, un_r = collect_points(filepath, id1, id2, ccd_num)
    print(
        f"[INFO] total inliers: {inl_l.shape[0]}, outliers: {out_l.shape[0]}, "
        f"unmatched_L: {un_l.shape[0]}, unmatched_R: {un_r.shape[0]}"
    )
    if inl_l.shape[0] == 0 and out_l.shape[0] == 0 and un_l.shape[0] == 0 and un_r.shape[0] == 0:
        print("[ERROR] 没有可绘制的点", file=sys.stderr)
        return 1

    img_l = cv2.imread(str(img_l_path), cv2.IMREAD_GRAYSCALE)
    img_r = cv2.imread(str(img_r_path), cv2.IMREAD_GRAYSCALE)
    if img_l is None or img_r is None:
        print("[ERROR] 影像读取失败", file=sys.stderr)
        return 1

    canvas = draw_matches(
        img_l,
        img_r,
        inl_l,
        inl_r,
        out_l,
        out_r,
        un_l,
        un_r,
        args.max,
        args.max_outliers,
        args.max_unmatched,
        args.thickness,
        args.radius,
        args.seed,
        draw_lines=args.draw_lines,
        draw_outliers=args.draw_outliers,
        draw_unmatched=args.draw_unmatched,
        random_color=args.random_color,
        color_bgr=color_bgr,
        outlier_color_bgr=outlier_bgr,
        unmatched_color_bgr=unmatched_bgr,
    )

    out = args.out
    if out is None:
        out = Path(__file__).resolve().parents[1] / "out" / "fenfu_match0_inliers.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)

    ext = out.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        cv2.imwrite(str(out), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpg_quality])
    else:
        cv2.imwrite(str(out), canvas)

    n_in = inl_l.shape[0] if args.max <= 0 else min(inl_l.shape[0], args.max)
    n_out = out_l.shape[0] if args.max_outliers <= 0 else min(out_l.shape[0], args.max_outliers)
    n_un_l = un_l.shape[0] if args.max_unmatched <= 0 else min(un_l.shape[0], args.max_unmatched)
    n_un_r = un_r.shape[0] if args.max_unmatched <= 0 else min(un_r.shape[0], args.max_unmatched)
    print(
        f"[INFO] drew inliers {n_in}/{inl_l.shape[0]}, "
        f"outliers {n_out if args.draw_outliers else 0}/{out_l.shape[0]}, "
        f"unmatched L+R {n_un_l + n_un_r if args.draw_unmatched else 0}/"
        f"{un_l.shape[0] + un_r.shape[0]} -> {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
