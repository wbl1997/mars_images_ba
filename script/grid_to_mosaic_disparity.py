#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nature-style dense grid/feature matching disparity visualization.

功能
----
1. 将 level-0 grid/match 同名点转换到统一 CCD mosaic 坐标系：
       dx = C2 - C1  (horizontal disparity)
       dy = R2 - R1  (vertical disparity)
2. 保存 dx、dy、score、valid mask、局部视差不一致性等原始结果。
3. 生成论文级图件：
   - 单结果：1×3（或 1×4）；
   - 修正前/后对比：2×3（或 2×4）；
   - before/after 使用相同空间范围和共享色标；
   - dx 使用 cividis，dy 使用以 0 为中心的 RdBu_r；
   - 无效区域为浅灰色，可选叠加灰度 mosaic 背景；
   - 输出 PDF、PNG、JPG，默认 600 dpi。
4. 输出稳健统计：
   valid coverage、median、MAD、P95 |dy|、local inconsistency 等。

默认文件命名
------------
grid 模式：
    {id1}_RED{ccd}_grid.txt
match 模式：
    {id1}_RED{ccd}_match.txt

每行格式沿用原工程：
    bj row col right_ccd mrow mcol score
仅读取 bj == 1 的有效匹配。

常用示例
--------
A. 只绘制改进后的结果：
    python grid_to_mosaic_disparity_nature.py cfg_after.yaml \
        --source grid \
        --out-dir results/figure11

B. 修正前/后对比，分别使用两个 cfg：
    python grid_to_mosaic_disparity_nature.py cfg_after.yaml \
        --before-cfg cfg_before.yaml \
        --source grid \
        --out-dir results/figure11

C. 同一个 cfg，但修正前匹配文件位于另一个 level-0 目录：
    python grid_to_mosaic_disparity_nature.py cfg_after.yaml \
        --before-left-dir /path/to/before/level0 \
        --source grid \
        --out-dir results/figure11

D. 修正前后使用不同文件名模板：
    python grid_to_mosaic_disparity_nature.py cfg_after.yaml \
        --before-left-dir /path/to/level0 \
        --before-pattern "{id1}_RED{ccd}_grid_before.txt" \
        --after-pattern  "{id1}_RED{ccd}_grid_after.txt" \
        --source grid

E. 叠加统一 mosaic 背景，并裁剪到重点区域：
    python grid_to_mosaic_disparity_nature.py cfg_after.yaml \
        --before-cfg cfg_before.yaml \
        --background /path/to/left_mosaic_preview.jpg \
        --crop 2000 12000 5000 25000 \
        --out-dir results/figure11

F. 增加第四列局部视差不一致性：
    python grid_to_mosaic_disparity_nature.py cfg_after.yaml \
        --before-cfg cfg_before.yaml \
        --include-inconsistency-panel
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, Normalize, TwoSlopeNorm
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "该脚本需要 matplotlib。请安装：pip install matplotlib"
    ) from exc


DEFAULT_PATTERN = "{id1}_RED{ccd}_{suffix}"


@dataclass
class PipelineConfig:
    path: Path
    filepath: Path
    id1: str
    id2: str
    ccd_num: int
    ccd_begin: int
    ccd_end: int
    batch_size: int


@dataclass
class StageData:
    label: str
    cfg: PipelineConfig
    source: str
    left_dir: Path
    right_dir: Path
    pattern: str
    H: int
    W: int
    R1: np.ndarray
    C1: np.ndarray
    DX: np.ndarray
    DY: np.ndarray
    score: np.ndarray
    left_ccd: List[int]
    right_ccd: List[int]
    disp_x: np.ndarray
    disp_y: np.ndarray
    score_map: np.ndarray
    mask: np.ndarray
    inconsistency: np.ndarray
    cell_size: float
    grid_step: int
    raster_scale: float
    metrics: Dict[str, float]


# ---------------------------------------------------------------------------
# Configuration and input
# ---------------------------------------------------------------------------

def parse_cfg(path: Path) -> PipelineConfig:
    """Parse the small subset of cfg keys required by this script."""
    if not path.is_file():
        raise FileNotFoundError(path)

    text = path.read_text(encoding="utf-8")

    def grab_str(key: str) -> Optional[str]:
        match = re.search(
            rf'(?m)^\s*{re.escape(key)}\s*:\s*"?([^"\n#]+)"?',
            text,
        )
        if not match:
            return None
        return match.group(1).strip().strip('"').strip("'")

    def grab_int(key: str, default: int) -> int:
        match = re.search(
            rf"(?m)^\s*{re.escape(key)}\s*:\s*(-?\d+)",
            text,
        )
        return int(match.group(1)) if match else default

    filepath = grab_str("filepath")
    id1 = grab_str("xulie_ID1")
    id2 = grab_str("xulie_ID2")
    if filepath is None or id1 is None or id2 is None:
        missing = [
            key
            for key, val in (
                ("filepath", filepath),
                ("xulie_ID1", id1),
                ("xulie_ID2", id2),
            )
            if val is None
        ]
        raise RuntimeError(f"cfg 缺少字段 {missing}: {path}")

    ccd_num = grab_int("CCD_num", 10)
    ccd_begin = grab_int("CCD_begin", 0)
    ccd_end_raw = grab_int("CCD_end", -1)
    ccd_end = ccd_num if ccd_end_raw < 0 else ccd_end_raw

    if not (0 <= ccd_begin < ccd_num):
        raise ValueError(f"非法 CCD_begin={ccd_begin}, CCD_num={ccd_num}")
    if not (ccd_begin < ccd_end <= ccd_num):
        raise ValueError(
            f"非法 CCD 范围 [{ccd_begin}, {ccd_end}), CCD_num={ccd_num}"
        )

    return PipelineConfig(
        path=path,
        filepath=Path(filepath),
        id1=id1,
        id2=id2,
        ccd_num=ccd_num,
        ccd_begin=ccd_begin,
        ccd_end=ccd_end,
        batch_size=grab_int("batch_size", 0),
    )


def load_mosaic_offsets(mosaic_txt: Path, ccd_num: int) -> np.ndarray:
    """Return shape (CCD_num, 4): beginR, endR, beginC, endC."""
    if not mosaic_txt.is_file():
        raise FileNotFoundError(mosaic_txt)

    offsets = np.zeros((ccd_num, 4), dtype=np.int64)
    seen = np.zeros(ccd_num, dtype=bool)

    with mosaic_txt.open("r", encoding="utf-8") as stream:
        for line in stream:
            parts = line.split()
            if len(parts) < 5:
                continue
            ccd_id = int(float(parts[0]))
            if 0 <= ccd_id < ccd_num:
                offsets[ccd_id] = [
                    int(float(parts[1])),
                    int(float(parts[2])),
                    int(float(parts[3])),
                    int(float(parts[4])),
                ]
                seen[ccd_id] = True

    if not np.any(seen):
        raise RuntimeError(f"mosaic.txt 中未解析到任何 CCD 偏移：{mosaic_txt}")

    missing = np.flatnonzero(~seen)
    if missing.size:
        print(
            f"[WARN] {mosaic_txt}: 缺少 CCD offsets: "
            + ", ".join(map(str, missing.tolist()))
        )
    return offsets


def mosaic_size_from_offsets(offsets: np.ndarray) -> Tuple[int, int]:
    if offsets.size == 0:
        return 0, 0
    H = int(np.max(offsets[:, 1]))
    W = int(np.max(offsets[:, 3]))
    if H <= 0 or W <= 0:
        raise RuntimeError(f"非法 mosaic 尺寸：H={H}, W={W}")
    return H, W


def read_matched_points(
    path: Path,
) -> List[Tuple[float, float, int, float, float, float]]:
    """Read bj==1 rows as row, col, right_ccd, mrow, mcol, score."""
    points: List[Tuple[float, float, int, float, float, float]] = []
    if not path.is_file():
        return points

    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                if int(float(parts[0])) != 1:
                    continue
                points.append(
                    (
                        float(parts[1]),
                        float(parts[2]),
                        int(float(parts[3])),
                        float(parts[4]),
                        float(parts[5]),
                        float(parts[6]),
                    )
                )
            except ValueError:
                print(f"[WARN] 跳过无法解析的行 {path}:{line_no}")
    return points


def render_pattern(
    pattern: str,
    id1: str,
    ccd: int,
    source: str,
) -> str:
    suffix = "grid.txt" if source == "grid" else "match.txt"
    try:
        return pattern.format(
            id1=id1,
            ccd=ccd,
            source=source,
            suffix=suffix,
        )
    except KeyError as exc:
        raise ValueError(
            f"文件模板包含未知占位符 {exc}: {pattern}"
        ) from exc


def collect_mosaic_matches(
    cfg: PipelineConfig,
    left_dir: Path,
    right_dir: Path,
    source: str,
    pattern: str,
    allow_grid_fallback: bool,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[int],
    List[int],
    int,
    int,
]:
    """Collect matched points in the unified mosaic frame."""
    offsets_left = load_mosaic_offsets(
        left_dir / "mosaic.txt",
        cfg.ccd_num,
    )
    offsets_right = load_mosaic_offsets(
        right_dir / "mosaic.txt",
        cfg.ccd_num,
    )
    H, W = mosaic_size_from_offsets(offsets_left)

    R1_values: List[float] = []
    C1_values: List[float] = []
    DX_values: List[float] = []
    DY_values: List[float] = []
    scores: List[float] = []
    left_ccds: List[int] = []
    right_ccds: List[int] = []

    for left_ccd in range(cfg.ccd_begin, cfg.ccd_end):
        filename = render_pattern(pattern, cfg.id1, left_ccd, source)
        path = left_dir / filename

        if (
            source == "grid"
            and allow_grid_fallback
            and not path.is_file()
            and pattern == DEFAULT_PATTERN
        ):
            fallback = left_dir / f"{cfg.id1}_RED{left_ccd}_match.txt"
            if fallback.is_file():
                print(
                    f"[WARN] 缺少 {path.name}，临时回退到 {fallback.name}"
                )
                path = fallback

        if not path.is_file():
            print(f"[WARN] 缺少 {path}，跳过 CCD{left_ccd}")
            continue

        points = read_matched_points(path)
        print(
            f"[INFO] CCD{left_ccd}: {len(points)} matches from {path.name}"
        )
        begin_r_left, _, begin_c_left, _ = offsets_left[left_ccd]

        for row, col, right_ccd, mrow, mcol, score in points:
            if right_ccd < 0 or right_ccd >= cfg.ccd_num:
                continue
            begin_r_right, _, begin_c_right, _ = offsets_right[right_ccd]

            R1 = row + begin_r_left
            C1 = col + begin_c_left
            R2 = mrow + begin_r_right
            C2 = mcol + begin_c_right

            R1_values.append(R1)
            C1_values.append(C1)
            DX_values.append(C2 - C1)
            DY_values.append(R2 - R1)
            scores.append(score)
            left_ccds.append(left_ccd)
            right_ccds.append(right_ccd)

    if not R1_values:
        raise RuntimeError(
            f"未读取到任何有效匹配点：source={source}, left_dir={left_dir}"
        )

    return (
        np.asarray(R1_values, dtype=np.float64),
        np.asarray(C1_values, dtype=np.float64),
        np.asarray(DX_values, dtype=np.float32),
        np.asarray(DY_values, dtype=np.float32),
        np.asarray(scores, dtype=np.float32),
        left_ccds,
        right_ccds,
        H,
        W,
    )


# ---------------------------------------------------------------------------
# Rasterization and statistics
# ---------------------------------------------------------------------------

def estimate_grid_step(
    rows: Sequence[float],
    cols: Sequence[float],
) -> int:
    """Estimate a dominant regular step from unique coordinates."""

    def step_1d(values: Sequence[float]) -> int:
        unique = np.unique(
            np.round(np.asarray(values, dtype=np.float64), 3)
        )
        if unique.size < 2:
            return 0
        diffs = np.diff(unique)
        diffs = diffs[diffs > 1e-3]
        if diffs.size == 0:
            return 0
        rounded = np.round(diffs).astype(np.int64)
        rounded = rounded[rounded > 0]
        if rounded.size == 0:
            return 0
        values_u, counts = np.unique(rounded, return_counts=True)
        return int(values_u[np.argmax(counts)])

    step_r = step_1d(rows)
    step_c = step_1d(cols)
    if step_r > 0 and step_c > 0:
        if abs(step_r - step_c) <= 2:
            return int(round(0.5 * (step_r + step_c)))
        return max(step_r, step_c)
    return step_r or step_c or 0


def rasterize_best_score(
    R1: np.ndarray,
    C1: np.ndarray,
    DX: np.ndarray,
    DY: np.ndarray,
    score: np.ndarray,
    H: int,
    W: int,
    cell_size: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Rasterize points to a regular canvas.

    Each output cell keeps the observation with the highest score. No spatial
    dilation or interpolation is performed.
    """
    if cell_size <= 0:
        raise ValueError(f"cell_size 必须 > 0，当前为 {cell_size}")

    Hr = int(math.ceil(H / cell_size)) + 1
    Wr = int(math.ceil(W / cell_size)) + 1
    ir = np.rint(R1 / cell_size).astype(np.int64)
    ic = np.rint(C1 / cell_size).astype(np.int64)

    disp_x = np.full((Hr, Wr), np.nan, dtype=np.float32)
    disp_y = np.full((Hr, Wr), np.nan, dtype=np.float32)
    score_map = np.full((Hr, Wr), np.nan, dtype=np.float32)

    for index in range(R1.size):
        row = int(ir[index])
        col = int(ic[index])
        if row < 0 or col < 0 or row >= Hr or col >= Wr:
            continue
        current_score = float(score[index])
        if (
            np.isnan(score_map[row, col])
            or current_score >= float(score_map[row, col])
        ):
            disp_x[row, col] = DX[index]
            disp_y[row, col] = DY[index]
            score_map[row, col] = current_score

    mask = np.isfinite(disp_x) & np.isfinite(disp_y)
    return disp_x, disp_y, score_map, mask


def shifted_nan_stack(array: np.ndarray) -> np.ndarray:
    """Build a 3×3 shifted stack without scipy."""
    H, W = array.shape
    stack: List[np.ndarray] = []

    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            shifted = np.full((H, W), np.nan, dtype=np.float32)

            src_r0 = max(0, -dr)
            src_r1 = min(H, H - dr)
            src_c0 = max(0, -dc)
            src_c1 = min(W, W - dc)

            dst_r0 = max(0, dr)
            dst_r1 = min(H, H + dr)
            dst_c0 = max(0, dc)
            dst_c1 = min(W, W + dc)

            shifted[dst_r0:dst_r1, dst_c0:dst_c1] = array[
                src_r0:src_r1,
                src_c0:src_c1,
            ]
            stack.append(shifted)

    return np.stack(stack, axis=0)


def local_disparity_inconsistency(
    disp_x: np.ndarray,
    disp_y: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Local inconsistency:
        || d_i - median_{j in N_3x3(i)} d_j ||_2

    Only valid neighbors are used. Invalid positions remain NaN.
    """
    x = np.where(mask, disp_x, np.nan).astype(np.float32)
    y = np.where(mask, disp_y, np.nan).astype(np.float32)

    with np.errstate(all="ignore"):
        median_x = np.nanmedian(shifted_nan_stack(x), axis=0)
        median_y = np.nanmedian(shifted_nan_stack(y), axis=0)

    inconsistency = np.sqrt(
        (disp_x - median_x) ** 2
        + (disp_y - median_y) ** 2
    ).astype(np.float32)
    inconsistency[~mask] = np.nan
    return inconsistency


def robust_mad(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    median = np.median(finite)
    return float(np.median(np.abs(finite - median)))


def robust_outlier_ratio(values: np.ndarray) -> float:
    """
    Percentage outside median ± 3 × 1.4826 × MAD.
    Returns 0 when the robust scale degenerates.
    """
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    median = float(np.median(finite))
    mad = robust_mad(finite)
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-12:
        return 0.0
    outlier = np.abs(finite - median) > 3.0 * scale
    return float(np.mean(outlier) * 100.0)


def compute_metrics(
    disp_x: np.ndarray,
    disp_y: np.ndarray,
    mask: np.ndarray,
    inconsistency: np.ndarray,
    n_points: int,
) -> Dict[str, float]:
    valid_x = disp_x[mask]
    valid_y = disp_y[mask]
    valid_inc = inconsistency[np.isfinite(inconsistency)]

    return {
        "n_input_points": float(n_points),
        "n_valid_cells": float(np.sum(mask)),
        "valid_coverage_percent": float(np.mean(mask) * 100.0),
        "dx_median_px": float(np.nanmedian(valid_x)),
        "dx_mad_px": robust_mad(valid_x),
        "dy_median_px": float(np.nanmedian(valid_y)),
        "dy_mad_px": robust_mad(valid_y),
        "dy_p95_abs_px": float(np.nanpercentile(np.abs(valid_y), 95)),
        "dy_outlier_ratio_percent": robust_outlier_ratio(valid_y),
        "local_inconsistency_median_px": (
            float(np.nanmedian(valid_inc))
            if valid_inc.size
            else float("nan")
        ),
        "local_inconsistency_p95_px": (
            float(np.nanpercentile(valid_inc, 95))
            if valid_inc.size
            else float("nan")
        ),
        "local_inconsistency_outlier_ratio_percent": (
            robust_outlier_ratio(valid_inc)
            if valid_inc.size
            else float("nan")
        ),
    }


def build_stage(
    label: str,
    cfg_path: Path,
    source: str,
    left_dir_override: Optional[Path],
    right_dir_override: Optional[Path],
    pattern: str,
    grid_step_override: int,
    scatter_scale: float,
    allow_grid_fallback: bool,
) -> StageData:
    cfg = parse_cfg(cfg_path)

    left_dir = (
        left_dir_override
        if left_dir_override is not None
        else cfg.filepath / cfg.id1 / "downsample" / "0"
    )
    right_dir = (
        right_dir_override
        if right_dir_override is not None
        else cfg.filepath / cfg.id2 / "downsample" / "0"
    )

    (
        R1,
        C1,
        DX,
        DY,
        score,
        left_ccd,
        right_ccd,
        H,
        W,
    ) = collect_mosaic_matches(
        cfg=cfg,
        left_dir=left_dir,
        right_dir=right_dir,
        source=source,
        pattern=pattern,
        allow_grid_fallback=allow_grid_fallback,
    )

    if source == "grid":
        grid_step = int(grid_step_override)
        if grid_step <= 0:
            grid_step = cfg.batch_size if cfg.batch_size > 0 else 0
        if grid_step <= 0:
            grid_step = estimate_grid_step(R1, C1)
        if grid_step <= 0:
            grid_step = 48
            print("[WARN] 无法估计 grid step，使用默认值 48 px")
        cell_size = float(grid_step)
        raster_scale = 1.0
        print(f"[INFO] {label}: grid_step={grid_step} px")
    else:
        if scatter_scale < 1.0:
            raise ValueError("--scatter-scale 必须 >= 1")
        grid_step = 0
        cell_size = float(scatter_scale)
        raster_scale = float(scatter_scale)
        print(
            f"[INFO] {label}: feature-match raster scale="
            f"{scatter_scale:g}; publication figure uses original points"
        )

    disp_x, disp_y, score_map, mask = rasterize_best_score(
        R1=R1,
        C1=C1,
        DX=DX,
        DY=DY,
        score=score,
        H=H,
        W=W,
        cell_size=cell_size,
    )
    inconsistency = local_disparity_inconsistency(
        disp_x,
        disp_y,
        mask,
    )
    metrics = compute_metrics(
        disp_x,
        disp_y,
        mask,
        inconsistency,
        n_points=R1.size,
    )

    print(
        f"[STAT] {label}: "
        f"valid={int(metrics['n_valid_cells'])}, "
        f"coverage={metrics['valid_coverage_percent']:.2f}%, "
        f"median(dx)={metrics['dx_median_px']:.3f}px, "
        f"median(dy)={metrics['dy_median_px']:.3f}px, "
        f"P95|dy|={metrics['dy_p95_abs_px']:.3f}px, "
        f"median(local)={metrics['local_inconsistency_median_px']:.3f}px"
    )

    return StageData(
        label=label,
        cfg=cfg,
        source=source,
        left_dir=left_dir,
        right_dir=right_dir,
        pattern=pattern,
        H=H,
        W=W,
        R1=R1,
        C1=C1,
        DX=DX,
        DY=DY,
        score=score,
        left_ccd=left_ccd,
        right_ccd=right_ccd,
        disp_x=disp_x,
        disp_y=disp_y,
        score_map=score_map,
        mask=mask,
        inconsistency=inconsistency,
        cell_size=cell_size,
        grid_step=grid_step,
        raster_scale=raster_scale,
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Saving raw results
# ---------------------------------------------------------------------------

def scalar_display_range(array: np.ndarray) -> Tuple[float, float]:
    finite = np.asarray(array, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(finite, [1.0, 99.0])
    vmin = float(vmin)
    vmax = float(vmax)
    if abs(vmax - vmin) < 1e-12:
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1.0
    return vmin, vmax


def upsample_nearest(array: np.ndarray, scale: int) -> np.ndarray:
    """Nearest-neighbor upsample for a larger on-disk preview."""
    if scale <= 1:
        return array
    return np.repeat(np.repeat(array, scale, axis=0), scale, axis=1)


def dilate_scalar_cells(array: np.ndarray, radius: int) -> np.ndarray:
    """
    Expand each finite cell to a disk of the given radius (in cell units).

    Rasterization stores one value per match cell; without dilation each
    correspondence is only 1 cell wide. Stage PNG previews use this to make
    matched points visually larger. Overlaps keep the nearest source cell.
    """
    if radius <= 0:
        return np.asarray(array, dtype=np.float32)

    src = np.asarray(array, dtype=np.float32)
    H, W = src.shape
    out = np.full((H, W), np.nan, dtype=np.float32)
    best_dist2 = np.full((H, W), np.inf, dtype=np.float32)
    valid_r, valid_c = np.where(np.isfinite(src))
    r2_max = radius * radius

    for y, x in zip(valid_r.tolist(), valid_c.tolist()):
        value = src[y, x]
        r0 = max(0, y - radius)
        r1 = min(H, y + radius + 1)
        c0 = max(0, x - radius)
        c1 = min(W, x + radius + 1)
        for yy in range(r0, r1):
            dy = yy - y
            for xx in range(c0, c1):
                dx = xx - x
                dist2 = float(dy * dy + dx * dx)
                if dist2 > r2_max:
                    continue
                if dist2 < best_dist2[yy, xx]:
                    best_dist2[yy, xx] = dist2
                    out[yy, xx] = value
    return out


def dilate_binary_cells(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary disk dilation in cell units (stage PNG preview only)."""
    if radius <= 0:
        return (np.asarray(mask) > 0).astype(np.float32)
    src = (np.asarray(mask) > 0)
    H, W = src.shape
    out = np.zeros((H, W), dtype=np.float32)
    valid_r, valid_c = np.where(src)
    r2_max = radius * radius
    for y, x in zip(valid_r.tolist(), valid_c.tolist()):
        r0 = max(0, y - radius)
        r1 = min(H, y + radius + 1)
        c0 = max(0, x - radius)
        c1 = min(W, x + radius + 1)
        for yy in range(r0, r1):
            dy = yy - y
            for xx in range(c0, c1):
                dx = xx - x
                if dy * dy + dx * dx <= r2_max:
                    out[yy, xx] = 1.0
    return out


def save_png(
    path: Path,
    array: np.ndarray,
    *,
    cmap: str = "gray",
    scale: int = 1,
    binary: bool = False,
    dilate_radius: int = 0,
) -> Path:
    path = path.with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(array)
    if arr.ndim != 2:
        raise ValueError(f"仅支持二维数组，当前 shape={arr.shape}")

    if binary:
        preview = dilate_binary_cells(arr, dilate_radius)
        preview = upsample_nearest(preview, scale)
        plt.imsave(str(path), preview, cmap="gray", vmin=0.0, vmax=1.0)
    else:
        preview = dilate_scalar_cells(arr, dilate_radius)
        vmin, vmax = scalar_display_range(preview)
        preview = upsample_nearest(preview, scale)
        masked = np.ma.masked_invalid(preview)
        cmap_obj = copy_cmap(cmap, bad_color="#d9d9d9")
        plt.imsave(
            str(path),
            masked,
            cmap=cmap_obj,
            vmin=vmin,
            vmax=vmax,
        )

    print(f"[OK] {path}")
    return path


def save_stage_outputs(stage: StageData, out_dir: Path) -> None:
    stage_dir = out_dir / stage.label.lower().replace(" ", "_")
    stage_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{stage.cfg.id1}_{stage.cfg.id2}_{stage.source}_disp"
    # 仅 stage 目录预览：彩虹色；每个匹配点在 cell 上膨胀，再轻微放大
    # 外层论文三图不受影响
    disp_scale = 3
    dilate_radius = 2  # 半径 2 cell → 直径约 5 cell
    disp_cmap = "rainbow"

    save_png(
        stage_dir / f"{stem}_dx.png",
        stage.disp_x,
        cmap=disp_cmap,
        scale=disp_scale,
        dilate_radius=dilate_radius,
    )
    save_png(
        stage_dir / f"{stem}_dy.png",
        stage.disp_y,
        cmap=disp_cmap,
        scale=disp_scale,
        dilate_radius=dilate_radius,
    )
    save_png(
        stage_dir / f"{stem}_score.png",
        stage.score_map,
        scale=disp_scale,
        dilate_radius=dilate_radius,
    )
    save_png(
        stage_dir / f"{stem}_mask.png",
        stage.mask.astype(np.uint8),
        binary=True,
        scale=disp_scale,
        dilate_radius=dilate_radius,
    )
    save_png(
        stage_dir / f"{stem}_local_inconsistency.png",
        stage.inconsistency,
        cmap="magma",
        scale=disp_scale,
        dilate_radius=dilate_radius,
    )

    # Backward-compatible aliases:
    save_png(
        stage_dir / f"{stem}_dc.png",
        stage.disp_x,
        cmap=disp_cmap,
        scale=disp_scale,
        dilate_radius=dilate_radius,
    )
    save_png(
        stage_dir / f"{stem}_dr.png",
        stage.disp_y,
        cmap=disp_cmap,
        scale=disp_scale,
        dilate_radius=dilate_radius,
    )

    points_path = stage_dir / f"{stem}_points.txt"
    with points_path.open("w", encoding="utf-8") as stream:
        stream.write(
            "R1 C1 dx dy score left_ccd right_ccd\n"
        )
        for index in range(stage.R1.size):
            stream.write(
                f"{stage.R1[index]:.3f} "
                f"{stage.C1[index]:.3f} "
                f"{stage.DX[index]:.3f} "
                f"{stage.DY[index]:.3f} "
                f"{stage.score[index]:.6f} "
                f"{stage.left_ccd[index]} "
                f"{stage.right_ccd[index]}\n"
            )
    print(f"[OK] {points_path}")

    meta_path = stage_dir / f"{stem}_meta.txt"
    with meta_path.open("w", encoding="utf-8") as stream:
        stream.write(f"label={stage.label}\n")
        stream.write(f"cfg={stage.cfg.path}\n")
        stream.write(f"id1={stage.cfg.id1}\n")
        stream.write(f"id2={stage.cfg.id2}\n")
        stream.write(f"source={stage.source}\n")
        stream.write(f"left_dir={stage.left_dir}\n")
        stream.write(f"right_dir={stage.right_dir}\n")
        stream.write(f"pattern={stage.pattern}\n")
        stream.write(f"mosaic_H={stage.H}\n")
        stream.write(f"mosaic_W={stage.W}\n")
        stream.write(f"grid_step={stage.grid_step}\n")
        stream.write(f"raster_scale={stage.raster_scale}\n")
        stream.write(f"cell_size={stage.cell_size}\n")
        stream.write(
            "definition: dx=C2-C1, dy=R2-R1 "
            "in full-resolution mosaic pixels\n"
        )
        stream.write(
            "local_inconsistency: L2 distance from the "
            "3x3 valid-neighbor median disparity vector\n"
        )
        for key, value in stage.metrics.items():
            stream.write(f"{key}={value}\n")
    print(f"[OK] {meta_path}")


def save_metrics_csv(stages: Sequence[StageData], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["label"] + list(stages[0].metrics.keys())

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for stage in stages:
            row = {"label": stage.label}
            row.update(stage.metrics)
            writer.writerow(row)
    print(f"[OK] {path}")


# ---------------------------------------------------------------------------
# Publication figure
# ---------------------------------------------------------------------------

def load_background(path: Optional[Path]) -> Optional[np.ndarray]:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(path)

    image = np.asarray(plt.imread(str(path)))
    if image.ndim == 3:
        image = image[..., :3].astype(np.float32)
        image = (
            0.2126 * image[..., 0]
            + 0.7152 * image[..., 1]
            + 0.0722 * image[..., 2]
        )
    image = np.squeeze(image).astype(np.float32)

    finite = image[np.isfinite(image)]
    if finite.size:
        low, high = np.percentile(finite, [1, 99])
        if high > low:
            image = np.clip((image - low) / (high - low), 0.0, 1.0)
    return image


def valid_values(
    stages: Sequence[StageData],
    attribute: str,
) -> np.ndarray:
    values: List[np.ndarray] = []
    for stage in stages:
        array = getattr(stage, attribute)
        finite = array[np.isfinite(array)]
        if finite.size:
            values.append(finite.astype(np.float64))
    if not values:
        return np.asarray([], dtype=np.float64)
    return np.concatenate(values)


def safe_percentile_range(
    values: np.ndarray,
    low_q: float,
    high_q: float,
) -> Tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(finite, [low_q, high_q])
    vmin = float(vmin)
    vmax = float(vmax)
    if abs(vmax - vmin) < 1e-9:
        delta = max(1.0, abs(vmin) * 0.01)
        vmin -= delta
        vmax += delta
    return vmin, vmax


def copy_cmap(name: str, bad_color: str = "#d9d9d9"):
    cmap = plt.get_cmap(name)
    try:
        cmap = cmap.copy()
    except AttributeError:  # older matplotlib
        pass
    cmap.set_bad(bad_color)
    return cmap


def set_axis_extent(
    ax,
    stage: StageData,
    crop: Optional[Tuple[float, float, float, float]],
) -> None:
    if crop is None:
        ax.set_xlim(0, stage.W)
        ax.set_ylim(stage.H, 0)
        return

    r_min, r_max, c_min, c_max = crop
    ax.set_xlim(c_min, c_max)
    ax.set_ylim(r_max, r_min)


def draw_background(
    ax,
    background: Optional[np.ndarray],
    stage: StageData,
    alpha: float,
) -> None:
    if background is None:
        ax.set_facecolor("#efefef")
        return
    ax.imshow(
        background,
        cmap="gray",
        vmin=0,
        vmax=1,
        extent=[0, stage.W, stage.H, 0],
        interpolation="nearest",
        alpha=alpha,
        zorder=0,
    )


def draw_mask_panel(
    ax,
    stage: StageData,
    background: Optional[np.ndarray],
    crop: Optional[Tuple[float, float, float, float]],
    marker_size: float,
    background_alpha: float,
) -> None:
    draw_background(ax, background, stage, background_alpha)

    if stage.source == "match":
        ax.scatter(
            stage.C1,
            stage.R1,
            s=marker_size,
            c="#111111",
            linewidths=0,
            alpha=0.85,
            rasterized=True,
            zorder=2,
        )
    else:
        mask_float = np.ma.masked_where(
            ~stage.mask,
            np.ones(stage.mask.shape, dtype=np.float32),
        )
        valid_cmap = ListedColormap(["#111111"])
        valid_cmap.set_bad((0, 0, 0, 0))
        ax.imshow(
            mask_float,
            cmap=valid_cmap,
            vmin=0,
            vmax=1,
            extent=[0, stage.W, stage.H, 0],
            interpolation="nearest",
            alpha=0.9,
            zorder=2,
        )

    set_axis_extent(ax, stage, crop)


def draw_scalar_panel(
    ax,
    stage: StageData,
    background: Optional[np.ndarray],
    crop: Optional[Tuple[float, float, float, float]],
    quantity: str,
    cmap,
    norm,
    marker_size: float,
    background_alpha: float,
):
    draw_background(ax, background, stage, background_alpha)

    if quantity == "dx":
        point_values = stage.DX
        raster_values = stage.disp_x
    elif quantity == "dy":
        point_values = stage.DY
        raster_values = stage.disp_y
    elif quantity == "inconsistency":
        point_values = None
        raster_values = stage.inconsistency
    else:
        raise ValueError(quantity)

    if stage.source == "match" and quantity in {"dx", "dy"}:
        artist = ax.scatter(
            stage.C1,
            stage.R1,
            c=point_values,
            s=marker_size,
            cmap=cmap,
            norm=norm,
            linewidths=0,
            alpha=0.9,
            rasterized=True,
            zorder=2,
        )
    else:
        masked = np.ma.masked_invalid(raster_values)
        artist = ax.imshow(
            masked,
            cmap=cmap,
            norm=norm,
            extent=[0, stage.W, stage.H, 0],
            interpolation="nearest",
            alpha=0.88,
            zorder=2,
        )

    set_axis_extent(ax, stage, crop)
    return artist


def add_panel_letter(ax, letter: str) -> None:
    ax.text(
        -0.10,
        1.035,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        clip_on=False,
    )


def add_metrics_annotation(ax, metrics: Dict[str, float]) -> None:
    text = (
        f"Coverage {metrics['valid_coverage_percent']:.1f}%\n"
        f"P95 |dy| {metrics['dy_p95_abs_px']:.2f} px\n"
        f"Median $E_{{loc}}$ "
        f"{metrics['local_inconsistency_median_px']:.2f} px"
    )
    ax.text(
        0.985,
        0.985,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.8,
        color="black",
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.78,
        },
        zorder=5,
    )


def build_publication_figure(
    stages: Sequence[StageData],
    backgrounds: Sequence[Optional[np.ndarray]],
    out_dir: Path,
    figure_name: str,
    include_inconsistency_panel: bool,
    crop: Optional[Tuple[float, float, float, float]],
    color_percentiles: Tuple[float, float],
    dx_zero_centered: bool,
    annotate_metrics: bool,
    marker_size: float,
    background_alpha: float,
    dpi: int,
) -> None:
    n_rows = len(stages)
    n_cols = 4 if include_inconsistency_panel else 3

    # Shared color scales across before/after.
    low_q, high_q = color_percentiles
    dx_values = valid_values(stages, "disp_x")
    dy_values = valid_values(stages, "disp_y")
    inc_values = valid_values(stages, "inconsistency")

    dx_vmin, dx_vmax = safe_percentile_range(
        dx_values,
        low_q,
        high_q,
    )
    if dx_zero_centered:
        dx_limit = max(abs(dx_vmin), abs(dx_vmax), 1e-6)
        dx_norm = TwoSlopeNorm(
            vmin=-dx_limit,
            vcenter=0.0,
            vmax=dx_limit,
        )
    else:
        dx_norm = Normalize(vmin=dx_vmin, vmax=dx_vmax)

    finite_dy = dy_values[np.isfinite(dy_values)]
    if finite_dy.size:
        dy_limit = float(np.percentile(np.abs(finite_dy), high_q))
    else:
        dy_limit = 1.0
    dy_limit = max(dy_limit, 1e-6)
    dy_norm = TwoSlopeNorm(
        vmin=-dy_limit,
        vcenter=0.0,
        vmax=dy_limit,
    )

    finite_inc = inc_values[np.isfinite(inc_values)]
    if finite_inc.size:
        inc_vmax = float(np.percentile(finite_inc, high_q))
    else:
        inc_vmax = 1.0
    inc_norm = Normalize(vmin=0.0, vmax=max(inc_vmax, 1e-6))

    dx_cmap = copy_cmap("cividis")
    dy_cmap = copy_cmap("RdBu_r")
    inc_cmap = copy_cmap("magma")

    # Nature-style compact typography and full two-column width.
    plt.rcParams.update(
        {
            "font.family": ["Arial", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    width_in = 178.0 / 25.4
    height_in = width_in * (n_rows / n_cols) * 0.98
    height_in = max(height_in, 2.15 if n_rows == 1 else 3.65)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(width_in, height_in),
        squeeze=False,
        sharex=True,
        sharey=True,
        layout="constrained",
    )

    column_titles = [
        "Valid matches",
        "Horizontal disparity",
        "Vertical disparity",
    ]
    if include_inconsistency_panel:
        column_titles.append("Local inconsistency")

    dx_artists = []
    dy_artists = []
    inc_artists = []
    panel_index = 0

    for row, (stage, background) in enumerate(
        zip(stages, backgrounds)
    ):
        draw_mask_panel(
            axes[row, 0],
            stage,
            background,
            crop,
            marker_size,
            background_alpha,
        )
        if annotate_metrics:
            add_metrics_annotation(axes[row, 0], stage.metrics)

        dx_artist = draw_scalar_panel(
            axes[row, 1],
            stage,
            background,
            crop,
            "dx",
            dx_cmap,
            dx_norm,
            marker_size,
            background_alpha,
        )
        dy_artist = draw_scalar_panel(
            axes[row, 2],
            stage,
            background,
            crop,
            "dy",
            dy_cmap,
            dy_norm,
            marker_size,
            background_alpha,
        )
        dx_artists.append(dx_artist)
        dy_artists.append(dy_artist)

        if include_inconsistency_panel:
            inc_artist = draw_scalar_panel(
                axes[row, 3],
                stage,
                background,
                crop,
                "inconsistency",
                inc_cmap,
                inc_norm,
                marker_size,
                background_alpha,
            )
            inc_artists.append(inc_artist)

        axes[row, 0].set_ylabel("Mosaic row (pixel)")
        axes[row, 0].text(
            -0.22,
            0.5,
            stage.label,
            transform=axes[row, 0].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold",
            clip_on=False,
        )

        for col in range(n_cols):
            add_panel_letter(
                axes[row, col],
                chr(ord("a") + panel_index),
            )
            panel_index += 1
            axes[row, col].tick_params(direction="out")
            if row == n_rows - 1:
                axes[row, col].set_xlabel(
                    "Mosaic column (pixel)"
                )
            if col > 0:
                axes[row, col].tick_params(
                    labelleft=False
                )

    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title, pad=3)

    dx_colorbar = fig.colorbar(
        dx_artists[-1],
        ax=axes[:, 1],
        orientation="vertical",
        fraction=0.035,
        pad=0.018,
    )
    dx_colorbar.set_label(
        "Horizontal disparity $d_x$ (pixel)"
    )

    dy_colorbar = fig.colorbar(
        dy_artists[-1],
        ax=axes[:, 2],
        orientation="vertical",
        fraction=0.035,
        pad=0.018,
    )
    dy_colorbar.set_label(
        "Vertical disparity $d_y$ (pixel)"
    )

    if include_inconsistency_panel:
        inc_colorbar = fig.colorbar(
            inc_artists[-1],
            ax=axes[:, 3],
            orientation="vertical",
            fraction=0.035,
            pad=0.018,
        )
        inc_colorbar.set_label(
            "Local inconsistency $E_{loc}$ (pixel)"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{figure_name}.pdf"
    png_path = out_dir / f"{figure_name}.png"
    jpg_path = out_dir / f"{figure_name}.jpg"

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        png_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        jpg_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"quality": 95},
    )
    plt.close(fig)

    print(f"[OK] {pdf_path}")
    print(f"[OK] {png_path}")
    print(f"[OK] {jpg_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def positive_float(value: str) -> float:
    result = float(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("必须为正数")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Dense grid/feature matches -> mosaic disparity data and "
            "Nature-style publication figure"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "after_cfg",
        type=Path,
        help="改进后结果对应的 pipeline cfg",
    )
    parser.add_argument(
        "--before-cfg",
        type=Path,
        default=None,
        help=(
            "改进前结果对应 cfg。若仅提供 --before-left-dir，"
            "则默认复用 after_cfg"
        ),
    )
    parser.add_argument(
        "--source",
        choices=("grid", "match"),
        default="grid",
        help="改进后数据类型，默认 grid",
    )
    parser.add_argument(
        "--before-source",
        choices=("grid", "match"),
        default=None,
        help="改进前数据类型，默认与 --source 相同",
    )

    parser.add_argument(
        "--after-left-dir",
        type=Path,
        default=None,
        help="覆盖改进后左影像 level-0 目录",
    )
    parser.add_argument(
        "--after-right-dir",
        type=Path,
        default=None,
        help="覆盖改进后右影像 level-0 目录",
    )
    parser.add_argument(
        "--before-left-dir",
        type=Path,
        default=None,
        help="覆盖改进前左影像 level-0 目录",
    )
    parser.add_argument(
        "--before-right-dir",
        type=Path,
        default=None,
        help="覆盖改进前右影像 level-0 目录",
    )

    parser.add_argument(
        "--after-pattern",
        default=DEFAULT_PATTERN,
        help=(
            "改进后匹配文件模板，可用占位符："
            "{id1}, {ccd}, {source}, {suffix}"
        ),
    )
    parser.add_argument(
        "--before-pattern",
        default=None,
        help="改进前匹配文件模板，默认与 --after-pattern 相同",
    )

    parser.add_argument(
        "--grid-step",
        type=int,
        default=0,
        help=(
            "改进后 grid 步长；0=cfg batch_size/自动估计"
        ),
    )
    parser.add_argument(
        "--before-grid-step",
        type=int,
        default=None,
        help="改进前 grid 步长；默认与 --grid-step 相同",
    )
    parser.add_argument(
        "--scatter-scale",
        type=positive_float,
        default=16.0,
        help=(
            "match 模式原始栅格倍率（stage 目录输出 PNG），论文图仍用散点；"
            "默认 16"
        ),
    )

    parser.add_argument(
        "--background",
        type=Path,
        default=None,
        help="before/after 共用的左 mosaic 灰度背景",
    )
    parser.add_argument(
        "--before-background",
        type=Path,
        default=None,
        help="单独指定改进前背景",
    )
    parser.add_argument(
        "--after-background",
        type=Path,
        default=None,
        help="单独指定改进后背景",
    )
    parser.add_argument(
        "--background-alpha",
        type=float,
        default=0.42,
        help="灰度背景透明度，默认 0.42",
    )

    parser.add_argument(
        "--crop",
        nargs=4,
        type=float,
        metavar=("R_MIN", "R_MAX", "C_MIN", "C_MAX"),
        default=None,
        help=(
            "按 mosaic 坐标裁剪重点区域："
            "R_MIN R_MAX C_MIN C_MAX"
        ),
    )
    parser.add_argument(
        "--color-percentiles",
        nargs=2,
        type=float,
        metavar=("LOW", "HIGH"),
        default=(2.0, 98.0),
        help="共享色标分位数，默认 2 98",
    )
    parser.add_argument(
        "--dx-zero-centered",
        action="store_true",
        help="令水平视差也以 0 为色图中心",
    )
    parser.add_argument(
        "--include-inconsistency-panel",
        action="store_true",
        help="增加第四列局部视差不一致性",
    )
    parser.add_argument(
        "--no-metric-annotation",
        action="store_true",
        help="不在 valid mask 面板标注关键统计量",
    )
    parser.add_argument(
        "--marker-size",
        type=positive_float,
        default=3.0,
        help="match 散点标记面积，默认 3",
    )

    parser.add_argument(
        "--before-label",
        default="Before correction",
        help="改进前行标签",
    )
    parser.add_argument(
        "--after-label",
        default="After correction",
        help="改进后行标签",
    )
    parser.add_argument(
        "--figure-name",
        default="figure11_disparity",
        help="论文图文件名，不含扩展名",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "输出目录；默认 "
            "<after level0>/disparity_publication"
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="PNG/JPG 分辨率，默认 600",
    )
    parser.add_argument(
        "--no-grid-fallback",
        action="store_true",
        help="grid 文件缺失时不回退读取 match 文件",
    )

    return parser


def validate_args(args: argparse.Namespace) -> None:
    low, high = args.color_percentiles
    if not (0.0 <= low < high <= 100.0):
        raise ValueError(
            "--color-percentiles 必须满足 0 <= LOW < HIGH <= 100"
        )
    if not (0.0 <= args.background_alpha <= 1.0):
        raise ValueError("--background-alpha 必须在 [0, 1] 内")
    if args.dpi <= 0:
        raise ValueError("--dpi 必须 > 0")

    if args.crop is not None:
        r_min, r_max, c_min, c_max = args.crop
        if not (r_min < r_max and c_min < c_max):
            raise ValueError(
                "--crop 必须满足 R_MIN<R_MAX 且 C_MIN<C_MAX"
            )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        validate_args(args)

        before_requested = (
            args.before_cfg is not None
            or args.before_left_dir is not None
            or args.before_right_dir is not None
            or args.before_pattern is not None
        )

        after_stage = build_stage(
            label=args.after_label,
            cfg_path=args.after_cfg,
            source=args.source,
            left_dir_override=args.after_left_dir,
            right_dir_override=args.after_right_dir,
            pattern=args.after_pattern,
            grid_step_override=args.grid_step,
            scatter_scale=args.scatter_scale,
            allow_grid_fallback=not args.no_grid_fallback,
        )

        if args.out_dir is None:
            out_dir = (
                after_stage.left_dir
                / "disparity_publication"
            )
        else:
            out_dir = args.out_dir

        stages: List[StageData] = []
        backgrounds: List[Optional[np.ndarray]] = []

        if before_requested:
            before_cfg = (
                args.before_cfg
                if args.before_cfg is not None
                else args.after_cfg
            )
            before_stage = build_stage(
                label=args.before_label,
                cfg_path=before_cfg,
                source=(
                    args.before_source
                    if args.before_source is not None
                    else args.source
                ),
                left_dir_override=args.before_left_dir,
                right_dir_override=args.before_right_dir,
                pattern=(
                    args.before_pattern
                    if args.before_pattern is not None
                    else args.after_pattern
                ),
                grid_step_override=(
                    args.before_grid_step
                    if args.before_grid_step is not None
                    else args.grid_step
                ),
                scatter_scale=args.scatter_scale,
                allow_grid_fallback=not args.no_grid_fallback,
            )
            stages.append(before_stage)
            backgrounds.append(
                load_background(
                    args.before_background
                    or args.background
                )
            )

        stages.append(after_stage)
        backgrounds.append(
            load_background(
                args.after_background
                or args.background
            )
        )

        for stage in stages:
            save_stage_outputs(stage, out_dir)

        save_metrics_csv(
            stages,
            out_dir / f"{args.figure_name}_metrics.csv",
        )

        crop = (
            tuple(args.crop)
            if args.crop is not None
            else None
        )
        build_publication_figure(
            stages=stages,
            backgrounds=backgrounds,
            out_dir=out_dir,
            figure_name=args.figure_name,
            include_inconsistency_panel=(
                args.include_inconsistency_panel
            ),
            crop=crop,
            color_percentiles=tuple(args.color_percentiles),
            dx_zero_centered=args.dx_zero_centered,
            annotate_metrics=not args.no_metric_annotation,
            marker_size=args.marker_size,
            background_alpha=args.background_alpha,
            dpi=args.dpi,
        )

    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())