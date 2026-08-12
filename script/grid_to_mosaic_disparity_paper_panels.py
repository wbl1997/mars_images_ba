#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export separate publication panels from mosaic disparity matches.

This script reuses grid_to_mosaic_disparity.py for parsing MARS level-0
*_grid.txt / *_match.txt files and for converting CCD-local coordinates into
the unified mosaic frame.  It writes each panel as an independent figure:

  - horizontal disparity dx
  - vertical disparity dy
  - matching score
  - disparity vector field drawn by arrow direction and length
  - statistical distribution of dx / dy (histograms + joint density)

Also writes mosaic-frame point list (same as grid_to_mosaic_disparity.py):

  - {prefix}_points.txt   columns: R1 C1 dx dy score left_ccd right_ccd

Output modes:
  --output-mode png       write PNG only (default)
  --output-mode png+pdf   write PNG and PDF

Gap fill (visualization only):
  --fill-gaps             also write filled panels (*_filled_dx/dy/score);
                          raw (unfilled) panels are always written
  A blank cell is filled if either:
    (1) it is an inland hole (fully enclosed by valid cells), or
    (2) it lies within --fill-max-gap cells of a valid sample.
  Values use Gaussian-weighted moving-plane (面元) interpolation.
  --fill-max-gap N        proximity criterion radius in cells (default 8)
  --fill-sigma S          Gaussian σ in raster cells (default 2)
  --fill-radius R         kernel support radius (default max(3σ, max-gap))

Local outlier despike (before panels / fill):
  --despike               remove local gross-error cells/points
  --despike-k K           MAD multiplier (default 3)
  --despike-abs T         absolute inconsistency floor in px (default 0=off)

Example:
  python script/grid_to_mosaic_disparity_paper_panels.py cfg.yaml \
      --source match \
      --despike \
      --fill-gaps \
      --output-mode png+pdf \
      --out-dir results/paper_disparity_panels
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import deque
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

try:
    from scipy import ndimage as _ndimage

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _ndimage = None
    _HAS_SCIPY = False

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("该脚本需要 matplotlib。请安装：pip install matplotlib") from exc

try:
    from matplotlib.colors import TwoSlopeNorm
except ImportError:  # pragma: no cover - compatibility with older server matplotlib

    class TwoSlopeNorm(Normalize):
        def __init__(self, vcenter: float, vmin: Optional[float] = None, vmax: Optional[float] = None, clip: bool = False):
            super().__init__(vmin=vmin, vmax=vmax, clip=clip)
            self.vcenter = float(vcenter)

        def __call__(self, value, clip=None):
            result, is_scalar = self.process_value(value)
            self.autoscale_None(result)
            vmin = float(self.vmin)
            vmax = float(self.vmax)
            if not (vmin < self.vcenter < vmax):
                return super().__call__(value, clip=clip)

            data = np.asarray(result.data, dtype=np.float64)
            mapped = np.interp(data, [vmin, self.vcenter, vmax], [0.0, 0.5, 1.0])
            mapped = np.ma.array(mapped, mask=result.mask, copy=False)
            if is_scalar:
                mapped = mapped[0]
            return mapped

    mcolors.TwoSlopeNorm = TwoSlopeNorm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from grid_to_mosaic_disparity import (  # noqa: E402
    DEFAULT_PATTERN,
    StageData,
    build_stage,
    copy_cmap,
    draw_background,
    load_background,
    local_disparity_inconsistency,
    set_axis_extent,
    shifted_nan_stack,
)


def positive_float(value: str) -> float:
    result = float(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("必须为正数")
    return result


def valid_values(stage: StageData, quantity: str) -> np.ndarray:
    if quantity == "dx":
        values = stage.DX if stage.source == "match" else stage.disp_x
    elif quantity == "dy":
        values = stage.DY if stage.source == "match" else stage.disp_y
    elif quantity == "score":
        values = stage.score if stage.source == "match" else stage.score_map
    elif quantity == "magnitude":
        values = np.hypot(stage.DX, stage.DY)
    else:
        raise ValueError(quantity)
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def safe_minmax_range(values: np.ndarray) -> Tuple[float, float]:
    """Color / axis range from finite data min/max (with a tiny pad if degenerate)."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if abs(vmax - vmin) < 1e-9:
        delta = max(1.0, abs(vmin) * 0.01 + 1e-6)
        vmin -= delta
        vmax += delta
    return vmin, vmax


def make_scalar_norm(quantity: str, vmin: float, vmax: float):
    """Build a colormap norm spanning the adaptive [vmin, vmax]."""
    if quantity == "dy" and vmin < 0.0 < vmax:
        return TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    if quantity == "score":
        return Normalize(vmin=vmin, vmax=vmax)
    return Normalize(vmin=vmin, vmax=vmax)


# Scales only axis labels / ticks / legend / colorbar (see --font-scale).
# Does not change figure geometry or overlay annotation sizes.
_AXIS_FONT_SCALE = 1.0


def axis_fs(nominal: float) -> float:
    """Point size for axis/legend text under the active --font-scale."""
    return float(nominal) * float(_AXIS_FONT_SCALE)


def setup_publication_style(font_scale: float = 1.0) -> None:
    """
    Publication rcParams. ``font_scale`` enlarges only:
      - axes labels, tick labels
      - legend / colorbar text
    Titles, panel letters, and in-figure annotation boxes stay at fixed sizes
    so layout and image framing are unchanged.
    """
    global _AXIS_FONT_SCALE
    _AXIS_FONT_SCALE = max(float(font_scale), 0.5)
    label = axis_fs(9.0)
    tick = axis_fs(8.0)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "Nimbus Roman No9 L",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "font.size": 7,
            "axes.labelsize": label,
            "axes.titlesize": 7,
            "xtick.labelsize": tick,
            "ytick.labelsize": tick,
            "legend.fontsize": tick,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_colorbar(cbar) -> None:
    """Apply axis font scale to a colorbar label and ticks (H or V)."""
    cbar.ax.tick_params(labelsize=axis_fs(8.0))
    for lab in (cbar.ax.xaxis.label, cbar.ax.yaxis.label):
        if lab is not None and str(lab.get_text()):
            lab.set_size(axis_fs(9.0))


def figure_size(width_mm: float, stage: StageData, crop: Optional[Tuple[float, float, float, float]]) -> Tuple[float, float]:
    """
    Figure size that preserves mosaic pixel aspect (1 row px == 1 col px).

    If the strip is extremely tall, shrink width together with height so the
    data aspect ratio stays unchanged.
    """
    width_in = width_mm / 25.4
    if crop is None:
        rows = max(float(stage.H), 1.0)
        cols = max(float(stage.W), 1.0)
    else:
        r_min, r_max, c_min, c_max = crop
        rows = max(float(r_max - r_min), 1.0)
        cols = max(float(c_max - c_min), 1.0)

    # data aspect = Δrow / Δcol  →  figure height/width
    height_in = width_in * (rows / cols)
    # Practical page cap: shrink both sides uniformly (keep aspect)
    max_height_in = 11.0
    if height_in > max_height_in:
        scale = max_height_in / height_in
        height_in = max_height_in
        width_in *= scale
    min_height_in = 1.6
    if height_in < min_height_in:
        scale = min_height_in / height_in
        height_in = min_height_in
        width_in *= scale
    return width_in, height_in


def apply_equal_aspect(ax) -> None:
    """Keep mosaic row/column pixel scales identical on screen."""
    ax.set_aspect("equal", adjustable="box")


def apply_local_outlier_removal(
    stage: StageData,
    k: float = 3.0,
    abs_thr: float = 0.0,
    min_neighbors: int = 3,
) -> StageData:
    """
    Remove local gross-error samples (局部粗差).

    Criterion (raster, 3×3 neighborhood):
        E_loc = || d_i - median_{N_3x3(i)} d_j ||_2
        reject if E_loc > max(abs_thr, median(E) + k · 1.4826 · MAD(E))
        and the 3×3 window has at least ``min_neighbors`` other valid cells.

    Rejected raster cells become NaN; matching points that fall into rejected
    cells are dropped (affects scatter / quiver / histograms).
    """
    mask = np.asarray(stage.mask, dtype=bool)
    if not np.any(mask):
        print("[WARN] despike: no valid cells, skipped")
        return stage

    inc = local_disparity_inconsistency(stage.disp_x, stage.disp_y, mask)
    # Neighbor count excluding self.
    valid_f = mask.astype(np.float32)
    with np.errstate(all="ignore"):
        n_win = np.nansum(shifted_nan_stack(np.where(mask, valid_f, np.nan)), axis=0)
    n_others = np.maximum(n_win - 1.0, 0.0)

    finite = inc[np.isfinite(inc)]
    if finite.size < 8:
        print("[WARN] despike: too few samples for robust threshold, skipped")
        return stage

    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med)))
    scale = 1.4826 * mad
    if scale > 1e-12:
        thr = med + float(k) * scale
    else:
        # Degenerate MAD (e.g. mostly identical disparities): reject any E > median.
        thr = med
    if abs_thr > 0.0:
        thr = max(thr, float(abs_thr))

    bad = (
        np.isfinite(inc)
        & (inc > thr)
        & (n_others >= float(min_neighbors))
    )
    n_bad = int(np.sum(bad))
    n_valid = int(np.sum(mask))
    print(
        f"[INFO] despike: thr={thr:.4g} (median={med:.4g}, MAD={mad:.4g}, k={k}, "
        f"abs={abs_thr}); reject {n_bad}/{n_valid} cells "
        f"({100.0 * n_bad / max(n_valid, 1):.2f}%)"
    )

    if n_bad == 0:
        return stage

    stage.disp_x = np.array(stage.disp_x, copy=True)
    stage.disp_y = np.array(stage.disp_y, copy=True)
    stage.score_map = np.array(stage.score_map, copy=True)
    stage.disp_x[bad] = np.nan
    stage.disp_y[bad] = np.nan
    stage.score_map[bad] = np.nan
    stage.mask = np.isfinite(stage.disp_x) & np.isfinite(stage.disp_y)
    stage.inconsistency = local_disparity_inconsistency(
        stage.disp_x, stage.disp_y, stage.mask
    )

    # Drop points whose rasterized cell was rejected.
    cell = float(stage.cell_size) if stage.cell_size > 0 else 1.0
    ir = np.rint(stage.R1 / cell).astype(np.int64)
    ic = np.rint(stage.C1 / cell).astype(np.int64)
    Hr, Wr = stage.disp_x.shape
    in_b = (ir >= 0) & (ic >= 0) & (ir < Hr) & (ic < Wr)
    keep = np.ones(stage.R1.shape[0], dtype=bool)
    keep[in_b] = ~bad[ir[in_b], ic[in_b]]
    n_drop = int(np.sum(~keep))
    if n_drop:
        stage.R1 = stage.R1[keep]
        stage.C1 = stage.C1[keep]
        stage.DX = stage.DX[keep]
        stage.DY = stage.DY[keep]
        stage.score = stage.score[keep]
        left = np.asarray(stage.left_ccd)[keep]
        right = np.asarray(stage.right_ccd)[keep]
        stage.left_ccd = left.tolist()
        stage.right_ccd = right.tolist()
        print(f"[INFO] despike: dropped {n_drop} points mapped to rejected cells")

    return stage


def inland_holes(valid: np.ndarray) -> np.ndarray:
    """
    Inland (fully enclosed) NaN holes: True where cell is invalid but lies
    inside a closed valid contour (not connected to the image exterior).
    """
    valid = np.asarray(valid, dtype=bool)
    if _HAS_SCIPY:
        enclosed = _ndimage.binary_fill_holes(valid)
    else:
        # Flood-fill background from the border through invalid cells.
        H, W = valid.shape
        exterior = np.zeros((H, W), dtype=bool)
        q: deque = deque()
        for r in range(H):
            for c in (0, W - 1):
                if not valid[r, c] and not exterior[r, c]:
                    exterior[r, c] = True
                    q.append((r, c))
        for c in range(W):
            for r in (0, H - 1):
                if not valid[r, c] and not exterior[r, c]:
                    exterior[r, c] = True
                    q.append((r, c))
        while q:
            r, c = q.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and (not valid[nr, nc]) and (not exterior[nr, nc]):
                    exterior[nr, nc] = True
                    q.append((nr, nc))
        enclosed = valid | (~exterior)
    return enclosed & (~valid)


def _dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary dilation by Chebyshev radius (square footprint)."""
    if radius <= 0:
        return np.asarray(mask, dtype=bool).copy()
    mask = np.asarray(mask, dtype=bool)
    if _HAS_SCIPY:
        structure = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
        return _ndimage.binary_dilation(mask, structure=structure)
    out = mask.copy()
    H, W = out.shape
    for _ in range(radius):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        stacked = np.stack(
            [
                padded[0:H, 0:W],
                padded[0:H, 1 : W + 1],
                padded[0:H, 2 : W + 2],
                padded[1 : H + 1, 0:W],
                padded[1 : H + 1, 1 : W + 1],
                padded[1 : H + 1, 2 : W + 2],
                padded[2 : H + 2, 0:W],
                padded[2 : H + 2, 1 : W + 1],
                padded[2 : H + 2, 2 : W + 2],
            ],
            axis=0,
        )
        out = np.any(stacked, axis=0)
    return out


def proximity_holes(valid: np.ndarray, max_gap: int) -> np.ndarray:
    """Invalid cells within Euclidean ``max_gap`` of any valid sample."""
    valid = np.asarray(valid, dtype=bool)
    if max_gap <= 0:
        return np.zeros_like(valid, dtype=bool)
    if _HAS_SCIPY:
        dist = _ndimage.distance_transform_edt(~valid)
        return (dist > 0) & (dist <= float(max_gap))
    return _dilate_mask(valid, max_gap) & (~valid)


def _closing_mask_fast(mask: np.ndarray, radius: int) -> np.ndarray:
    """Euclidean closing via distance transforms (O(HW), not O(HW·R²))."""
    mask = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return mask.copy()
    if _HAS_SCIPY:
        dilated = _ndimage.distance_transform_edt(~mask) <= float(radius)
        # Erode dilated: keep pixels farther than ``radius`` from background.
        dist_in = _ndimage.distance_transform_edt(dilated)
        return dilated & (dist_in > float(radius))
    return _erode_mask(_dilate_mask(mask, radius), radius)


def _erode_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return np.asarray(mask, dtype=bool).copy()
    return ~_dilate_mask(~np.asarray(mask, dtype=bool), radius)


def coverage_holes(valid: np.ndarray, max_gap: int) -> np.ndarray:
    """
    Invalid cells inside the closed footprint of valid samples.

    Sparse match points leave porous gaps that connect to the image border, so
    plain inland-hole detection fails. Closing bridges those gaps first; then
    fill_holes recovers the true interior of the match coverage.
    """
    valid = np.asarray(valid, dtype=bool)
    if max_gap <= 0 or not np.any(valid):
        return np.zeros_like(valid, dtype=bool)
    closed = _closing_mask_fast(valid, max_gap)
    if _HAS_SCIPY:
        footprint = _ndimage.binary_fill_holes(closed)
    else:
        footprint = closed | inland_holes(closed)
    return footprint & (~valid)


def median_valid_nn_gap(valid: np.ndarray) -> float:
    """Median nearest-neighbor distance (cells) among valid samples."""
    coords = np.column_stack(np.nonzero(valid))
    n = coords.shape[0]
    if n < 2:
        return 0.0
    # Subsample for speed on dense valid masks.
    if n > 8000:
        rng = np.random.default_rng(0)
        coords = coords[rng.choice(n, size=8000, replace=False)]
    if _HAS_SCIPY:
        from scipy.spatial import cKDTree

        dists, _ = cKDTree(coords.astype(np.float64)).query(coords.astype(np.float64), k=2)
        nn = dists[:, 1]
        nn = nn[np.isfinite(nn)]
        return float(np.median(nn)) if nn.size else 0.0
    step = max(1, coords.shape[0] // 2000)
    sample = coords[::step]
    dmin = [
        math.sqrt(float(np.partition(np.sum((coords - p) ** 2, axis=1), 1)[1]))
        for p in sample
    ]
    return float(np.median(dmin)) if dmin else 0.0


def fill_candidate_holes(
    valid: np.ndarray,
    max_gap: int,
    connect_sparse: bool = False,
) -> np.ndarray:
    """Fill if inland OR proximity OR (optional) closed-coverage interior."""
    holes = inland_holes(valid) | proximity_holes(valid, max_gap)
    if connect_sparse:
        holes = holes | coverage_holes(valid, max_gap)
    return holes


def _gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur — much faster than dense truncated convolution."""
    sigma = max(float(sigma), 1e-6)
    if _HAS_SCIPY:
        return _ndimage.gaussian_filter(
            image, sigma=sigma, mode="constant", cval=0.0, truncate=3.0
        )
    radius = max(1, int(math.ceil(3.0 * sigma)))
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    dist2 = yy.astype(np.float64) ** 2 + xx.astype(np.float64) ** 2
    kern = np.exp(-dist2 / (2.0 * sigma * sigma))
    kern[dist2 > float(radius * radius)] = 0.0
    s = float(kern.sum())
    if s > 0:
        kern /= s
    # Fallback 2D convolve.
    pr = pc = radius
    padded = np.pad(image, ((pr, pr), (pc, pc)), mode="constant", constant_values=0.0)
    H, W = image.shape
    out = np.zeros((H, W), dtype=np.float64)
    for i in range(2 * radius + 1):
        for j in range(2 * radius + 1):
            w = kern[i, j]
            if w == 0.0:
                continue
            out += w * padded[i : i + H, j : j + W]
    return out


def _solve_symmetric_3x3(
    a00: np.ndarray,
    a01: np.ndarray,
    a02: np.ndarray,
    a11: np.ndarray,
    a12: np.ndarray,
    a22: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized solve of symmetric 3×3 systems → (x0, x1, x2, ok)."""
    c00 = a11 * a22 - a12 * a12
    c01 = a02 * a12 - a01 * a22
    c02 = a01 * a12 - a02 * a11
    c11 = a00 * a22 - a02 * a02
    c12 = a02 * a01 - a00 * a12
    c22 = a00 * a11 - a01 * a01
    det = a00 * c00 + a01 * c01 + a02 * c02
    scale = (
        np.abs(a00) + np.abs(a11) + np.abs(a22) + np.abs(a01) + np.abs(a02) + np.abs(a12) + 1e-30
    )
    ok = np.isfinite(det) & (np.abs(det) > 1e-12 * (scale**3))
    inv_det = np.zeros_like(det)
    inv_det[ok] = 1.0 / det[ok]
    x0 = (c00 * b0 + c01 * b1 + c02 * b2) * inv_det
    x1 = (c01 * b0 + c11 * b1 + c12 * b2) * inv_det
    x2 = (c02 * b0 + c12 * b1 + c22 * b2) * inv_det
    ok &= np.isfinite(x0) & np.isfinite(x1) & np.isfinite(x2)
    return x0, x1, x2, ok


def _geometry_moments(
    valid: np.ndarray, sigma: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shared Gaussian geometry moments (independent of z)."""
    H, W = valid.shape
    rr, cc = np.mgrid[0:H, 0:W]
    rr = rr.astype(np.float64)
    cc = cc.astype(np.float64)
    m = valid.astype(np.float64)
    sw = _gaussian_blur(m, sigma)
    sr = _gaussian_blur(m * rr, sigma)
    sc = _gaussian_blur(m * cc, sigma)
    srr = _gaussian_blur(m * rr * rr, sigma)
    scc = _gaussian_blur(m * cc * cc, sigma)
    src = _gaussian_blur(m * rr * cc, sigma)
    return rr, cc, sw, sr, sc, srr, scc, src


def gaussian_surface_fill(
    array: np.ndarray,
    holes: np.ndarray,
    sigma: float,
    geom: Optional[Tuple] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, Tuple]:
    """
    Fast Gaussian-weighted moving-plane fill via separable gaussian_filter.

    ``geom`` caches geometry moments so dx/dy/score can share them.
    Returns (filled_array, geom).
    """
    arr = np.asarray(array, dtype=np.float64)
    valid = np.isfinite(arr)
    need = np.asarray(holes, dtype=bool) & (~valid)
    out = np.array(array, dtype=np.float32, copy=True)
    if not np.any(need) or not np.any(valid):
        if geom is None:
            geom = _geometry_moments(valid, sigma)
        return out, geom

    sigma = max(float(sigma), 1e-6)
    if geom is None:
        geom = _geometry_moments(valid, sigma)
    rr, cc, sw, sr, sc, srr, scc, src = geom

    m = valid.astype(np.float64)
    z = np.where(valid, arr, 0.0)
    sz = _gaussian_blur(m * z, sigma)
    srz = _gaussian_blur(m * rr * z, sigma)
    scz = _gaussian_blur(m * cc * z, sigma)

    rows, cols = np.nonzero(need)
    r0 = rows.astype(np.float64)
    c0 = cols.astype(np.float64)
    w = sw[rows, cols]
    w_min = 1e-6
    has_w = w > w_min

    mean_z = np.full(rows.shape, np.nan, dtype=np.float64)
    mean_z[has_w] = sz[rows[has_w], cols[has_w]] / w[has_w]

    sr_l = sr[rows, cols] - r0 * w
    sc_l = sc[rows, cols] - c0 * w
    srr_l = srr[rows, cols] - 2.0 * r0 * sr[rows, cols] + r0 * r0 * w
    scc_l = scc[rows, cols] - 2.0 * c0 * sc[rows, cols] + c0 * c0 * w
    src_l = src[rows, cols] - r0 * sc[rows, cols] - c0 * sr[rows, cols] + r0 * c0 * w
    srz_l = srz[rows, cols] - r0 * sz[rows, cols]
    scz_l = scz[rows, cols] - c0 * sz[rows, cols]

    a0, _b0, _c0, plane_ok = _solve_symmetric_3x3(
        w,
        sr_l,
        sc_l,
        srr_l,
        src_l,
        scc_l,
        sz[rows, cols],
        srz_l,
        scz_l,
    )
    plane_ok &= has_w
    pred = a0
    local_spread = np.abs(mean_z) + 1.0
    plane_ok &= np.isfinite(pred) & (np.abs(pred - mean_z) <= 10.0 * local_spread)

    use_plane = plane_ok
    use_mean = has_w & (~use_plane)
    out[rows[use_plane], cols[use_plane]] = pred[use_plane].astype(np.float32)
    out[rows[use_mean], cols[use_mean]] = mean_z[use_mean].astype(np.float32)

    if verbose:
        print(
            f"[INFO] gaussian plane fill: plane={int(np.sum(use_plane))} "
            f"mean_fallback={int(np.sum(use_mean))} skip={int(np.sum(~has_w))}"
        )
    return out, geom


def apply_stage_gap_fill(
    stage: StageData,
    sigma: float,
    radius: Optional[int],
    max_gap: int = 8,
) -> StageData:
    """In-place fill of visualization rasters; point arrays (DX/DY) unchanged."""
    del radius  # kept in signature for CLI compatibility; blur uses σ only
    n_before = int(np.sum(~np.isfinite(stage.disp_x)))
    valid0 = np.isfinite(stage.disp_x)
    connect_sparse = stage.source == "match"

    eff_gap = int(max_gap)
    sigma_eff = max(float(sigma), 1e-6)
    if connect_sparse:
        nn = median_valid_nn_gap(valid0)
        bridge = int(math.ceil(0.5 * nn)) + 1 if nn > 0 else eff_gap
        if bridge > eff_gap:
            print(
                f"[INFO] match sparse nn_gap≈{nn:.2f} cells → "
                f"raise fill-max-gap {eff_gap} → {bridge}"
            )
            eff_gap = bridge
        # Larger σ so separable blur reaches across sparse gaps (replaces slow kNN).
        sigma_eff = max(sigma_eff, 0.5 * nn, 1.0)
        print(f"[INFO] match fill σ={sigma_eff:.3f} (blur path)")

    holes = fill_candidate_holes(valid0, max_gap=eff_gap, connect_sparse=connect_sparse)
    n_cand = int(np.sum(holes))
    if n_cand == 0:
        print(f"[INFO] fill-gaps: no candidate holes (source={stage.source})")
        return stage

    # Geometry moments once; z-moments per channel.
    stage.disp_x, geom = gaussian_surface_fill(
        stage.disp_x, holes, sigma=sigma_eff, geom=None, verbose=True
    )
    stage.disp_y, geom = gaussian_surface_fill(
        stage.disp_y, holes, sigma=sigma_eff, geom=geom, verbose=False
    )
    stage.score_map, _ = gaussian_surface_fill(
        stage.score_map, holes, sigma=sigma_eff, geom=geom, verbose=False
    )
    stage.mask = np.isfinite(stage.disp_x) & np.isfinite(stage.disp_y)
    n_after = int(np.sum(~np.isfinite(stage.disp_x)))
    mode = "inland|proximity|closed-coverage" if connect_sparse else "inland|proximity"
    print(
        f"[INFO] fill-gaps source={stage.source} ({mode}) gap={eff_gap} σ={sigma_eff:.3f}: "
        f"holes={n_cand}, disp_x NaN {n_before} -> {n_after}"
    )
    return stage


def save_figure(fig, out_base: Path, dpi: int, formats: Sequence[str]) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = out_base.with_suffix(f".{fmt}")
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if fmt.lower() in {"png", "jpg", "jpeg", "tif", "tiff"}:
            kwargs["dpi"] = dpi
        if fmt.lower() in {"jpg", "jpeg"}:
            kwargs["pil_kwargs"] = {"quality": 95}
        fig.savefig(path, **kwargs)
        print(f"[OK] {path}")
    plt.close(fig)


def draw_scalar_values(
    ax,
    stage: StageData,
    background: Optional[np.ndarray],
    crop: Optional[Tuple[float, float, float, float]],
    quantity: str,
    cmap,
    norm,
    marker_size: float,
    background_alpha: float,
    use_raster: bool = False,
):
    draw_background(ax, background, stage, background_alpha)

    if quantity == "dx":
        point_values = stage.DX
        raster_values = stage.disp_x
    elif quantity == "dy":
        point_values = stage.DY
        raster_values = stage.disp_y
    elif quantity == "score":
        point_values = stage.score
        raster_values = stage.score_map
    else:
        raise ValueError(quantity)

    # Gap-filled panels always draw the raster so interpolated cells are visible.
    if stage.source == "match" and not use_raster:
        artist = ax.scatter(
            stage.C1,
            stage.R1,
            c=point_values,
            s=marker_size,
            cmap=cmap,
            norm=norm,
            linewidths=0,
            alpha=0.92,
            rasterized=True,
            zorder=2,
        )
    else:
        artist = ax.imshow(
            np.ma.masked_invalid(raster_values),
            cmap=cmap,
            norm=norm,
            extent=[0, stage.W, stage.H, 0],
            interpolation="nearest",
            alpha=0.90,
            zorder=2,
        )

    set_axis_extent(ax, stage, crop)
    apply_equal_aspect(ax)
    return artist


def save_scalar_panel(
    stage: StageData,
    background: Optional[np.ndarray],
    out_dir: Path,
    prefix: str,
    quantity: str,
    title: str,
    colorbar_label: str,
    crop: Optional[Tuple[float, float, float, float]],
    marker_size: float,
    background_alpha: float,
    width_mm: float,
    dpi: int,
    formats: Sequence[str],
    use_raster: bool = False,
) -> None:
    if use_raster:
        if quantity == "dx":
            values = stage.disp_x[np.isfinite(stage.disp_x)]
        elif quantity == "dy":
            values = stage.disp_y[np.isfinite(stage.disp_y)]
        elif quantity == "score":
            values = stage.score_map[np.isfinite(stage.score_map)]
        else:
            raise ValueError(quantity)
        values = np.asarray(values, dtype=np.float64)
    else:
        values = valid_values(stage, quantity)
    vmin, vmax = safe_minmax_range(values)

    if quantity == "dx":
        cmap = copy_cmap("cividis")
    elif quantity == "dy":
        cmap = copy_cmap("RdBu_r")
    elif quantity == "score":
        cmap = copy_cmap("viridis")
    else:
        raise ValueError(quantity)
    norm = make_scalar_norm(quantity, vmin, vmax)
    print(f"[INFO] {quantity} color range: [{vmin:.6g}, {vmax:.6g}]")

    fig, ax = plt.subplots(figsize=figure_size(width_mm, stage, crop), constrained_layout=True)
    artist = draw_scalar_values(
        ax,
        stage,
        background,
        crop,
        quantity,
        cmap,
        norm,
        marker_size,
        background_alpha,
        use_raster=use_raster,
    )
    ax.set_title(title, pad=3)
    ax.set_xlabel("Mosaic column (pixel)")
    ax.set_ylabel("Mosaic row (pixel)")
    cbar = fig.colorbar(artist, ax=ax, fraction=0.035, pad=0.018)
    cbar.set_label(colorbar_label)
    style_colorbar(cbar)
    save_figure(fig, out_dir / f"{prefix}_{quantity}", dpi, formats)


def quiver_sample_points(
    stage: StageData,
    max_arrows: int,
    min_magnitude: float,
    crop: Optional[Tuple[float, float, float, float]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    R = stage.R1.astype(np.float64)
    C = stage.C1.astype(np.float64)
    DX = stage.DX.astype(np.float64)
    DY = stage.DY.astype(np.float64)
    mag = np.hypot(DX, DY)

    mask = np.isfinite(R) & np.isfinite(C) & np.isfinite(DX) & np.isfinite(DY)
    mask &= mag >= float(min_magnitude)
    if crop is not None:
        r_min, r_max, c_min, c_max = crop
        mask &= (R >= r_min) & (R <= r_max) & (C >= c_min) & (C <= c_max)

    R, C, DX, DY, mag = R[mask], C[mask], DX[mask], DY[mask], mag[mask]
    if R.size == 0:
        raise RuntimeError("没有可用于箭头图的有效视差点")

    if max_arrows > 0 and R.size > max_arrows:
        # Deterministic spatial thinning: sort by row/column and take an even stride.
        order = np.lexsort((C, R))
        keep = np.linspace(0, R.size - 1, max_arrows).round().astype(np.int64)
        idx = order[keep]
        R, C, DX, DY, mag = R[idx], C[idx], DX[idx], DY[idx], mag[idx]

    return R, C, DX, DY, mag


def save_quiver_panel(
    stage: StageData,
    background: Optional[np.ndarray],
    out_dir: Path,
    prefix: str,
    crop: Optional[Tuple[float, float, float, float]],
    max_arrows: int,
    min_magnitude: float,
    arrow_scale: Optional[float],
    background_alpha: float,
    width_mm: float,
    dpi: int,
    formats: Sequence[str],
) -> None:
    R, C, DX, DY, mag = quiver_sample_points(stage, max_arrows, min_magnitude, crop)
    mag_vmin, mag_vmax = safe_minmax_range(mag)
    mag_vmax = max(mag_vmax, 1e-6)
    print(f"[INFO] quiver |d| color range: [{mag_vmin:.6g}, {mag_vmax:.6g}]")

    if arrow_scale is None or arrow_scale <= 0:
        # Matplotlib quiver scale: larger value gives shorter arrows.
        arrow_scale = max(mag_vmax * 20.0, 1.0)

    fig, ax = plt.subplots(figsize=figure_size(width_mm, stage, crop), constrained_layout=True)
    draw_background(ax, background, stage, background_alpha)
    artist = ax.quiver(
        C,
        R,
        DX,
        DY,
        mag,
        angles="xy",
        scale_units="xy",
        scale=arrow_scale,
        cmap=copy_cmap("plasma"),
        norm=Normalize(vmin=0.0, vmax=mag_vmax),
        width=0.003,
        headwidth=3.2,
        headlength=4.2,
        headaxislength=3.8,
        alpha=0.92,
        zorder=3,
    )
    set_axis_extent(ax, stage, crop)
    apply_equal_aspect(ax)
    ax.set_title("Disparity vector field", pad=3)
    ax.set_xlabel("Mosaic column (pixel)")
    ax.set_ylabel("Mosaic row (pixel)")
    cbar = fig.colorbar(artist, ax=ax, fraction=0.035, pad=0.018)
    cbar.set_label("Disparity magnitude $|d|$ (pixel)")
    style_colorbar(cbar)

    ref = float(np.median(mag)) if mag.size else 1.0
    ref = max(ref, 1e-6)
    try:
        ax.quiverkey(
            artist,
            X=0.88,
            Y=1.035,
            U=ref,
            label=f"{ref:.1f} px",
            labelpos="E",
            coordinates="axes",
            fontproperties={"size": axis_fs(8.0)},
        )
    except Exception:
        pass

    save_figure(fig, out_dir / f"{prefix}_quiver", dpi, formats)


def _finite_dx_dy(
    stage: StageData,
    crop: Optional[Tuple[float, float, float, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect finite (dx, dy) pairs, optionally limited to a mosaic crop."""
    dx = stage.DX.astype(np.float64)
    dy = stage.DY.astype(np.float64)
    mask = np.isfinite(dx) & np.isfinite(dy)
    if crop is not None:
        r_min, r_max, c_min, c_max = crop
        r = stage.R1.astype(np.float64)
        c = stage.C1.astype(np.float64)
        mask &= (
            np.isfinite(r)
            & np.isfinite(c)
            & (r >= r_min)
            & (r <= r_max)
            & (c >= c_min)
            & (c <= c_max)
        )
    dx = dx[mask]
    dy = dy[mask]
    if dx.size == 0:
        raise RuntimeError("没有可用于视差分布统计的有效点")
    return dx, dy


def save_disparity_distribution_panel(
    stage: StageData,
    out_dir: Path,
    prefix: str,
    crop: Optional[Tuple[float, float, float, float]],
    width_mm: float,
    dpi: int,
    formats: Sequence[str],
    n_bins: int = 80,
) -> None:
    """
    Statistical distribution of mosaic-space disparity:
      (a) histogram of dx
      (b) histogram of dy
      (c) 2D joint density in (dx, dy) space

    Axis and color ranges follow data min/max.
    """
    dx, dy = _finite_dx_dy(stage, crop)
    dx_lo, dx_hi = safe_minmax_range(dx)
    dy_lo, dy_hi = safe_minmax_range(dy)
    print(
        f"[INFO] distribution ranges: dx=[{dx_lo:.6g}, {dx_hi:.6g}], "
        f"dy=[{dy_lo:.6g}, {dy_hi:.6g}]"
    )

    width_in = (width_mm * 2.0) / 25.4  # two-column style for 1×3
    height_in = width_in * 0.38
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(width_in, height_in),
        constrained_layout=True,
    )

    axes[0].hist(
        dx,
        bins=n_bins,
        range=(dx_lo, dx_hi),
        color="#3b6ea5",
        edgecolor="none",
        alpha=0.90,
    )
    axes[0].axvline(float(np.median(dx)), color="#111111", lw=0.8, ls="--")
    axes[0].set_xlim(dx_lo, dx_hi)
    axes[0].set_xlabel("Horizontal disparity $d_x$ (pixel)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of $d_x$", pad=3)
    axes[0].text(
        0.03,
        0.97,
        f"$N$={dx.size}\n"
        f"min={dx_lo:.2f}\nmax={dx_hi:.2f}\nmed={np.median(dx):.2f}",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=5.8,
        bbox={
            "boxstyle": "round,pad=0.2",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.75,
        },
    )

    axes[1].hist(
        dy,
        bins=n_bins,
        range=(dy_lo, dy_hi),
        color="#c44e52",
        edgecolor="none",
        alpha=0.90,
    )
    axes[1].axvline(float(np.median(dy)), color="#111111", lw=0.8, ls="--")
    if dy_lo < 0.0 < dy_hi:
        axes[1].axvline(0.0, color="#666666", lw=0.6, ls=":")
    axes[1].set_xlim(dy_lo, dy_hi)
    axes[1].set_xlabel("Vertical disparity $d_y$ (pixel)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Distribution of $d_y$", pad=3)
    axes[1].text(
        0.03,
        0.97,
        f"$N$={dy.size}\n"
        f"min={dy_lo:.2f}\nmax={dy_hi:.2f}\nmed={np.median(dy):.2f}",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=5.8,
        bbox={
            "boxstyle": "round,pad=0.2",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.75,
        },
    )

    # Joint distribution in disparity space (not mosaic geography)
    hb = axes[2].hexbin(
        dx,
        dy,
        gridsize=60,
        extent=(dx_lo, dx_hi, dy_lo, dy_hi),
        mincnt=1,
        cmap="magma",
        linewidths=0.0,
    )
    if dy_lo < 0.0 < dy_hi:
        axes[2].axhline(0.0, color="#888888", lw=0.5, ls=":")
    axes[2].axvline(float(np.median(dx)), color="#cccccc", lw=0.5, ls="--")
    axes[2].set_xlim(dx_lo, dx_hi)
    axes[2].set_ylim(dy_lo, dy_hi)
    axes[2].set_aspect("equal", adjustable="box")
    axes[2].set_xlabel("Horizontal disparity $d_x$ (pixel)")
    axes[2].set_ylabel("Vertical disparity $d_y$ (pixel)")
    axes[2].set_title("Joint $(d_x,d_y)$ density", pad=3)
    cbar = fig.colorbar(hb, ax=axes[2], fraction=0.046, pad=0.02)
    cbar.set_label("Count")
    style_colorbar(cbar)

    for i, letter in enumerate("abc"):
        axes[i].text(
            -0.12,
            1.04,
            letter,
            transform=axes[i].transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            clip_on=False,
        )

    save_figure(fig, out_dir / f"{prefix}_disp_distribution", dpi, formats)



# ============================================================================
# Sparse-match publication figure:
#   (a) global sparse point distribution + directed disparity vectors
#   (b,c) two local left/right patch pairs with corresponding points and
#         locally scaled disparity arrows.
# ============================================================================

def _read_yaml_cfg(cfg_path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("自动解析 cfg 需要 PyYAML：pip install pyyaml") from exc
    with Path(cfg_path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _resolve_background_paths(
    cfg_path: Path,
    stage: StageData,
    left_override: Optional[Path],
    right_override: Optional[Path],
    mosaic_name: str,
) -> Tuple[Path, Path]:
    """
    Resolve left/right display mosaics.

    Default:
      cfg.dataset.filepath/{ID}/downsample/0/mosaic_ds4.tif

    The image may be downsampled. It is displayed with the full mosaic
    coordinate extent, so sparse-match coordinates remain in their original
    level-0 coordinate system.
    """
    if left_override is not None and right_override is not None:
        return Path(left_override), Path(right_override)

    cfg_raw = _read_yaml_cfg(cfg_path)
    dataset = cfg_raw.get("dataset", {})
    root = Path(dataset.get("filepath", "")) if dataset.get("filepath") else None
    id1 = str(dataset.get("xulie_ID1", getattr(stage.cfg, "id1", "")))
    id2 = str(dataset.get("xulie_ID2", getattr(stage.cfg, "id2", "")))

    def candidates(pid: str, override: Optional[Path]) -> Sequence[Path]:
        if override is not None:
            return [Path(override)]
        if root is None:
            return []
        base = root / pid / "downsample" / "0"
        names = [
            mosaic_name,
            "mosaic_ds4.tif",
            "mosaic_ds4.tiff",
            "mosaic.tif",
            "mosaic.tiff",
            "mosaic.png",
        ]
        # Preserve order while removing duplicates.
        out = []
        seen = set()
        for name in names:
            p = base / name
            if str(p) not in seen:
                out.append(p)
                seen.add(str(p))
        return out

    def choose(pid: str, override: Optional[Path], side: str) -> Path:
        tried = candidates(pid, override)
        for p in tried:
            if p.is_file():
                print(f"[INFO] {side} background: {p}")
                return p
        tried_text = "\n".join(f"        - {p}" for p in tried) or "        (no candidates)"
        raise FileNotFoundError(
            f"找不到 {side} mosaic 背景。已尝试：\n{tried_text}\n"
            f"可显式传入 --{side}-background /path/to/mosaic_ds4.tif"
        )

    return choose(id1, left_override, "left"), choose(id2, right_override, "right")


def _load_gray_mosaic(path: Path) -> np.ndarray:
    """Load TIFF/PNG and robustly normalize it for grayscale display."""
    path = Path(path)
    image = None
    errors = []

    try:
        import tifffile

        image = tifffile.imread(str(path))
    except Exception as exc:  # pragma: no cover
        errors.append(f"tifffile: {exc}")

    if image is None:
        try:
            from PIL import Image

            with Image.open(path) as im:
                image = np.asarray(im)
        except Exception as exc:  # pragma: no cover
            errors.append(f"PIL: {exc}")

    if image is None:
        try:
            image = plt.imread(str(path))
        except Exception as exc:  # pragma: no cover
            errors.append(f"matplotlib: {exc}")

    if image is None:
        raise RuntimeError(f"无法读取图像 {path}: {'; '.join(errors)}")

    arr = np.asarray(image)
    if arr.ndim == 3:
        # Avoid a dependency on skimage; standard luminance conversion.
        if arr.shape[-1] >= 3:
            arr = (
                0.299 * arr[..., 0].astype(np.float64)
                + 0.587 * arr[..., 1].astype(np.float64)
                + 0.114 * arr[..., 2].astype(np.float64)
            )
        else:
            arr = arr[..., 0]
    arr = np.asarray(arr, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise RuntimeError(f"图像中没有有效像素: {path}")

    # Robust contrast stretch. Keep zero-valued nodata black.
    nonzero = finite[np.abs(finite) > 1e-12]
    ref = nonzero if nonzero.size > 100 else finite
    lo, hi = np.percentile(ref, [1.0, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(ref)), float(np.max(ref))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)

    out = (arr - float(lo)) / float(hi - lo)
    out = np.clip(out, 0.0, 1.0)
    out[~np.isfinite(arr)] = 0.0
    return out.astype(np.float32)


def _full_image_extent(
    image: np.ndarray,
    reference_h: float,
    reference_w: float,
    left_image: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    Full-coordinate height/width represented by a downsampled background.

    For the left image, reference_h/reference_w are stage.H/stage.W.
    For the right image, use the left downsampling scale inferred from the
    left display mosaic.
    """
    if left_image is None:
        return float(reference_h), float(reference_w)
    sr = float(reference_h) / max(float(left_image.shape[0]), 1.0)
    sc = float(reference_w) / max(float(left_image.shape[1]), 1.0)
    return float(image.shape[0]) * sr, float(image.shape[1]) * sc


def _clip_roi(
    roi: Tuple[float, float, float, float],
    H: float,
    W: float,
) -> Tuple[float, float, float, float]:
    r0, r1, c0, c1 = [float(v) for v in roi]
    hh = max(r1 - r0, 1.0)
    ww = max(c1 - c0, 1.0)

    r0 = max(0.0, min(r0, max(H - hh, 0.0)))
    c0 = max(0.0, min(c0, max(W - ww, 0.0)))
    r1 = min(H, r0 + hh)
    c1 = min(W, c0 + ww)
    return r0, r1, c0, c1


def _auto_local_rois(
    stage: StageData,
    roi_height: float,
    roi_width: float,
) -> Tuple[
    Tuple[float, float, float, float],
    Tuple[float, float, float, float],
]:
    """
    Select two representative, vertically separated regions containing
    sufficient sparse matches. The user can override them with --roi1/--roi2.
    """
    R = np.asarray(stage.R1, dtype=np.float64)
    C = np.asarray(stage.C1, dtype=np.float64)
    good = np.isfinite(R) & np.isfinite(C)
    R, C = R[good], C[good]
    if R.size < 2:
        raise RuntimeError("有效稀疏匹配点不足，无法自动选择两个局部区域")

    roi_height = min(max(float(roi_height), 200.0), float(stage.H))
    roi_width = min(max(float(roi_width), 200.0), float(stage.W))

    rois = []
    for q in (0.32, 0.68):
        target_r = float(np.quantile(R, q))
        band = np.abs(R - target_r) <= max(roi_height, 0.06 * float(stage.H))
        if np.any(band):
            center_r = float(np.median(R[band]))
            center_c = float(np.median(C[band]))
        else:
            idx = int(np.argmin(np.abs(R - target_r)))
            center_r = float(R[idx])
            center_c = float(C[idx])

        roi = (
            center_r - 0.5 * roi_height,
            center_r + 0.5 * roi_height,
            center_c - 0.5 * roi_width,
            center_c + 0.5 * roi_width,
        )
        rois.append(_clip_roi(roi, float(stage.H), float(stage.W)))

    return rois[0], rois[1]


def _points_in_left_roi(
    stage: StageData,
    roi: Tuple[float, float, float, float],
) -> np.ndarray:
    r0, r1, c0, c1 = roi
    R = np.asarray(stage.R1, dtype=np.float64)
    C = np.asarray(stage.C1, dtype=np.float64)
    DX = np.asarray(stage.DX, dtype=np.float64)
    DY = np.asarray(stage.DY, dtype=np.float64)
    return (
        np.isfinite(R)
        & np.isfinite(C)
        & np.isfinite(DX)
        & np.isfinite(DY)
        & (R >= r0)
        & (R <= r1)
        & (C >= c0)
        & (C <= c1)
    )


def _spatial_subsample_indices(
    R: np.ndarray,
    C: np.ndarray,
    max_points: int,
) -> np.ndarray:
    n = int(R.size)
    if max_points <= 0 or n <= max_points:
        return np.arange(n, dtype=np.int64)
    order = np.lexsort((C, R))
    take = np.linspace(0, n - 1, max_points).round().astype(np.int64)
    return order[take]


def _scaled_vectors(
    dx: np.ndarray,
    dy: np.ndarray,
    target_length: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Scale vectors only for display while preserving direction/relative size."""
    mag = np.hypot(dx, dy)
    finite = mag[np.isfinite(mag) & (mag > 1e-9)]
    ref = float(np.median(finite)) if finite.size else 1.0
    gain = float(target_length) / max(ref, 1e-9)
    return dx * gain, dy * gain, gain


def _set_image_crop(
    ax,
    image: np.ndarray,
    H: float,
    W: float,
    roi: Tuple[float, float, float, float],
) -> None:
    ax.imshow(
        image,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        extent=[0.0, W, H, 0.0],
        interpolation="nearest",
        zorder=0,
    )
    r0, r1, c0, c1 = roi
    ax.set_xlim(c0, c1)
    ax.set_ylim(r1, r0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])


def _right_roi_from_matches(
    stage: StageData,
    left_roi: Tuple[float, float, float, float],
    right_h: float,
    right_w: float,
) -> Tuple[Tuple[float, float, float, float], float, float]:
    mask = _points_in_left_roi(stage, left_roi)
    if not np.any(mask):
        raise RuntimeError(f"局部区域中没有匹配点: {left_roi}")
    dx_med = float(np.median(np.asarray(stage.DX)[mask]))
    dy_med = float(np.median(np.asarray(stage.DY)[mask]))
    r0, r1, c0, c1 = left_roi
    right_roi = (r0 + dy_med, r1 + dy_med, c0 + dx_med, c1 + dx_med)
    return _clip_roi(right_roi, right_h, right_w), dx_med, dy_med


def _draw_local_match_pair(
    ax_left,
    ax_right,
    stage: StageData,
    left_image: np.ndarray,
    right_image: np.ndarray,
    left_h: float,
    left_w: float,
    right_h: float,
    right_w: float,
    left_roi: Tuple[float, float, float, float],
    box_color: str,
    label: str,
    max_points: int,
    arrow_target_fraction: float,
    show_annotations: bool = True,
    mag_norm: Optional[Normalize] = None,
    mag_cmap=None,
) -> None:
    mask = _points_in_left_roi(stage, left_roi)
    ids = np.flatnonzero(mask)
    if ids.size == 0:
        raise RuntimeError(f"{label} 中没有有效匹配点")

    local_pick = _spatial_subsample_indices(
        np.asarray(stage.R1)[ids],
        np.asarray(stage.C1)[ids],
        max_points,
    )
    ids = ids[local_pick]

    R1 = np.asarray(stage.R1, dtype=np.float64)[ids]
    C1 = np.asarray(stage.C1, dtype=np.float64)[ids]
    DX = np.asarray(stage.DX, dtype=np.float64)[ids]
    DY = np.asarray(stage.DY, dtype=np.float64)[ids]
    R2 = R1 + DY
    C2 = C1 + DX
    mag = np.hypot(DX, DY)

    right_roi, dx_med, dy_med = _right_roi_from_matches(
        stage, left_roi, right_h, right_w
    )
    _set_image_crop(ax_left, left_image, left_h, left_w, left_roi)
    _set_image_crop(ax_right, right_image, right_h, right_w, right_roi)

    n = ids.size
    # Same colormap / scale as global arrows: color encodes |d|.
    cmap = mag_cmap if mag_cmap is not None else copy_cmap("plasma")
    if mag_norm is None:
        mag_norm = Normalize(
            vmin=float(np.nanmin(mag)),
            vmax=float(np.nanmax(mag)) + 1e-9,
        )
    colors = cmap(mag_norm(mag))
    marker_size = 52.0 if n <= 100 else 36.0

    ax_left.scatter(
        C1,
        R1,
        s=marker_size,
        facecolors="none",
        edgecolors=colors,
        linewidths=1.1,
        zorder=4,
        rasterized=True,
    )
    ax_right.scatter(
        C2,
        R2,
        s=marker_size,
        facecolors="none",
        edgecolors=colors,
        linewidths=1.1,
        zorder=4,
        rasterized=True,
    )

    r0, r1, c0, c1 = left_roi
    target = max(1.0, float(arrow_target_fraction) * min(r1 - r0, c1 - c0))
    U, V, gain = _scaled_vectors(DX, DY, target)
    ax_left.quiver(
        C1,
        R1,
        U,
        V,
        color=colors,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.0064,
        headwidth=3.5,
        headlength=4.5,
        headaxislength=4.0,
        alpha=0.92,
        zorder=3,
    )

    if show_annotations:
        ax_left.set_title(f"{label}: left image", pad=2)
        ax_right.set_title(f"{label}: right image", pad=2)
        ax_left.text(
            0.02,
            0.02,
            f"$N$={n}\nmedian $(d_x,d_y)$=({dx_med:.1f}, {dy_med:.1f}) px\n"
            f"arrow display gain={gain:.3g}",
            transform=ax_left.transAxes,
            ha="left",
            va="bottom",
            fontsize=5.5,
            color="white",
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "black",
                "edgecolor": "none",
                "alpha": 0.58,
            },
            zorder=6,
        )

    for ax in (ax_left, ax_right):
        for spine in ax.spines.values():
            spine.set_linewidth(1.4)
            spine.set_edgecolor(box_color)


def _add_vertical_scale_bar(
    ax,
    length_data: float,
    label: str,
    x_axes: float = 1.045,
    y0_frac: float = 0.10,
) -> None:
    """
    Vertical scale arrow outside the right border.

    Length is in mosaic data units (same as quiver with ``scale_units='xy'``),
    so it matches on-figure arrow lengths after display gain.
    """
    from matplotlib.transforms import blended_transform_factory

    length_data = float(abs(length_data))
    if not np.isfinite(length_data) or length_data <= 0.0:
        return

    # Anchor near the visual bottom (large row), tip toward smaller row (up on screen).
    y_lim = ax.get_ylim()
    y_span = float(y_lim[0] - y_lim[1])  # typically H - 0 > 0 with inverted y
    y0 = float(y_lim[0] - y0_frac * y_span)
    y1 = y0 - length_data  # upward on screen when y increases downward

    trans = blended_transform_factory(ax.transAxes, ax.transData)
    ax.annotate(
        "",
        xy=(x_axes, y1),
        xytext=(x_axes, y0),
        xycoords=trans,
        textcoords=trans,
        arrowprops={
            "arrowstyle": "->",
            "color": "black",
            "lw": 1.35,
            "mutation_scale": 10,
        },
        clip_on=False,
        zorder=10,
    )
    ax.text(
        x_axes + 0.035,
        0.5 * (y0 + y1),
        label,
        transform=trans,
        rotation=90,
        ha="left",
        va="center",
        fontsize=axis_fs(8.0),
        clip_on=False,
        zorder=10,
    )


def _draw_global_sparse_panel(
    ax,
    left_image: np.ndarray,
    left_h: float,
    left_w: float,
    roi: Tuple[float, float, float, float],
    R_all: np.ndarray,
    C_all: np.ndarray,
    Rq: np.ndarray,
    Cq: np.ndarray,
    Uq: np.ndarray,
    Vq: np.ndarray,
    magq: np.ndarray,
    global_gain: float,
    *,
    show_title: bool = True,
    show_info: bool = True,
    show_roi_label: bool = True,
    show_scale: bool = True,
    scale_mode: str = "quiverkey",
    mag_norm: Optional[Normalize] = None,
    mag_cmap=None,
):
    """
    Draw global sparse matches.

    ``scale_mode``:
      - ``quiverkey``: horizontal key near the top (combined figure)
      - ``vertical``: vertical scale bar outside bottom-right (standalone global)
    """
    ax.imshow(
        left_image,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        extent=[0.0, left_w, left_h, 0.0],
        interpolation="nearest",
        zorder=0,
    )
    cmap = mag_cmap if mag_cmap is not None else copy_cmap("plasma")
    if mag_norm is None:
        mag_norm = Normalize(vmin=float(np.min(magq)), vmax=float(np.max(magq)) + 1e-9)
    q = ax.quiver(
        Cq,
        Rq,
        Uq,
        Vq,
        magq,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        cmap=cmap,
        norm=mag_norm,
        width=0.0032,
        headwidth=3.2,
        headlength=4.0,
        headaxislength=3.6,
        alpha=0.96,
        zorder=2,
    )
    ax.set_xlim(0.0, left_w)
    ax.set_ylim(left_h, 0.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Mosaic column (pixel)")
    ax.set_ylabel("Mosaic row (pixel)")
    if show_title:
        ax.set_title("Global sparse correspondences and disparity vectors", pad=3)

    r0, r1, c0, c1 = roi
    ax.add_patch(
        Rectangle(
            (c0, r0),
            c1 - c0,
            r1 - r0,
            fill=False,
            edgecolor="#e31a1c",
            linewidth=1.8,
            zorder=5,
        )
    )
    if show_roi_label:
        ax.text(
            c0,
            max(0.0, r0 - 0.006 * left_h),
            "Region",
            color="#e31a1c",
            fontsize=6,
            fontweight="bold",
            ha="left",
            va="bottom",
            zorder=6,
        )

    ref = float(np.median(magq)) if magq.size else 1.0
    # Display length of a ``ref``-pixel disparity after gain (matches quiver xy scale).
    scale_len = float(ref * global_gain)
    if show_scale:
        if scale_mode == "vertical":
            _add_vertical_scale_bar(
                ax,
                length_data=scale_len,
                label=f"{ref:.0f} px",
            )
        else:
            try:
                ax.quiverkey(
                    q,
                    X=0.58,
                    Y=1.018,
                    U=scale_len,
                    label=f"{ref:.0f} px",
                    labelpos="E",
                    coordinates="axes",
                    fontproperties={"size": axis_fs(8.0)},
                )
            except Exception:
                pass

    if show_info:
        ax.text(
            0.02,
            0.015,
            f"Displayed arrows: {Rq.size}/{R_all.size}\n"
            f"arrow display gain={global_gain:.3g}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.5,
            color="white",
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "black",
                "edgecolor": "none",
                "alpha": 0.55,
            },
            zorder=6,
        )
    return q


def _save_global_sparse_panel(
    out_dir: Path,
    prefix: str,
    left_image: np.ndarray,
    left_h: float,
    left_w: float,
    roi: Tuple[float, float, float, float],
    R_all: np.ndarray,
    C_all: np.ndarray,
    Rq: np.ndarray,
    Cq: np.ndarray,
    Uq: np.ndarray,
    Vq: np.ndarray,
    magq: np.ndarray,
    global_gain: float,
    width_mm: float,
    dpi: int,
    formats: Sequence[str],
    mag_norm: Optional[Normalize] = None,
    mag_cmap=None,
) -> None:
    """Standalone global panel: no titles/letters/info text; top H-colorbar; BR V-scale."""
    fig, ax = plt.subplots(
        figsize=(width_mm / 25.4, max(5.8, min(9.4, (width_mm / 25.4) * 1.18))),
        constrained_layout=True,
    )
    q = _draw_global_sparse_panel(
        ax,
        left_image,
        left_h,
        left_w,
        roi,
        R_all,
        C_all,
        Rq,
        Cq,
        Uq,
        Vq,
        magq,
        global_gain,
        show_title=False,
        show_info=False,
        show_roi_label=False,
        show_scale=True,
        scale_mode="vertical",
        mag_norm=mag_norm,
        mag_cmap=mag_cmap,
    )
    cbar = fig.colorbar(
        q,
        ax=ax,
        orientation="horizontal",
        location="top",
        fraction=0.046,
        pad=0.10,
        aspect=40,
    )
    cbar.set_label("Disparity magnitude $|d|$ (pixel)")
    style_colorbar(cbar)
    # Extra right pad so the vertical scale bar sits outside the frame.
    fig.set_layout_engine("constrained")
    save_figure(fig, out_dir / f"{prefix}_sparse_match_global", dpi, formats)


def _save_single_local_region_panel(
    stage: StageData,
    out_dir: Path,
    prefix: str,
    left_image: np.ndarray,
    right_image: np.ndarray,
    left_h: float,
    left_w: float,
    right_h: float,
    right_w: float,
    roi: Tuple[float, float, float, float],
    box_color: str,
    label: str,
    max_local_points: int,
    local_arrow_target_fraction: float,
    width_mm: float,
    dpi: int,
    formats: Sequence[str],
    mag_norm: Optional[Normalize] = None,
    mag_cmap=None,
) -> None:
    """Standalone local pair: left/right stacked vertically; no titles/letters."""
    width_in = width_mm / 25.4
    fig = plt.figure(figsize=(width_in * 0.55, max(4.8, width_in * 0.95)), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=(1.0, 1.0), hspace=0.04)
    ax_l = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[1, 0])
    _draw_local_match_pair(
        ax_l,
        ax_r,
        stage,
        left_image,
        right_image,
        left_h,
        left_w,
        right_h,
        right_w,
        roi,
        box_color,
        label,
        max_local_points,
        local_arrow_target_fraction,
        show_annotations=False,
        mag_norm=mag_norm,
        mag_cmap=mag_cmap,
    )
    save_figure(fig, out_dir / f"{prefix}_sparse_match_region", dpi, formats)


def save_sparse_local_match_panel(
    stage: StageData,
    cfg_path: Path,
    out_dir: Path,
    prefix: str,
    left_background: Optional[Path],
    right_background: Optional[Path],
    mosaic_name: str,
    roi1: Optional[Tuple[float, float, float, float]],
    roi2: Optional[Tuple[float, float, float, float]],
    roi_height: float,
    roi_width: float,
    max_global_arrows: int,
    max_local_points: int,
    global_arrow_target_fraction: float,
    local_arrow_target_fraction: float,
    width_mm: float,
    dpi: int,
    formats: Sequence[str],
) -> None:
    """
    Publication sparse-match figure:
      (a) global sparse positions + directed disparity arrows;
      (b,c) one local left/right pair stacked vertically.
    Also writes standalone ``*_sparse_match_global`` and ``*_sparse_match_region``
    without titles / panel letters / info overlays.
    """
    del roi2  # only one local region is kept
    if stage.source != "match":
        raise ValueError(
            "稀疏匹配局部面板仅适用于 --source match；"
            "grid 结果请继续使用原 dx/dy 栅格面板。"
        )

    left_path, right_path = _resolve_background_paths(
        cfg_path,
        stage,
        left_background,
        right_background,
        mosaic_name,
    )
    left_image = _load_gray_mosaic(left_path)
    right_image = _load_gray_mosaic(right_path)

    left_h, left_w = float(stage.H), float(stage.W)
    right_h, right_w = _full_image_extent(
        right_image, left_h, left_w, left_image=left_image
    )

    auto1, _auto2 = _auto_local_rois(stage, roi_height, roi_width)
    roi = _clip_roi(roi1 if roi1 is not None else auto1, left_h, left_w)
    print(f"[INFO] local ROI (R0,R1,C0,C1): {roi}")

    R = np.asarray(stage.R1, dtype=np.float64)
    C = np.asarray(stage.C1, dtype=np.float64)
    DX = np.asarray(stage.DX, dtype=np.float64)
    DY = np.asarray(stage.DY, dtype=np.float64)
    good = (
        np.isfinite(R)
        & np.isfinite(C)
        & np.isfinite(DX)
        & np.isfinite(DY)
    )
    R, C, DX, DY = R[good], C[good], DX[good], DY[good]
    if R.size == 0:
        raise RuntimeError("没有可用于全局稀疏匹配面板的有效点")

    keep = _spatial_subsample_indices(R, C, max_global_arrows)
    Rq, Cq, DXq, DYq = R[keep], C[keep], DX[keep], DY[keep]
    magq = np.hypot(DXq, DYq)
    target = max(
        1.0,
        float(global_arrow_target_fraction) * min(left_h, left_w),
    )
    Uq, Vq, global_gain = _scaled_vectors(DXq, DYq, target)

    # Shared |d| color scale for global arrows + local points/arrows.
    mag_cmap = copy_cmap("plasma")
    mag_norm = Normalize(
        vmin=float(np.min(magq)),
        vmax=float(np.max(magq)) + 1e-9,
    )

    # Combined figure: global | local-left/local-right (vertical stack).
    width_in = float(width_mm) / 25.4
    height_in = max(5.8, min(9.6, width_in * 1.15))
    fig = plt.figure(figsize=(width_in, height_in), constrained_layout=True)
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.45, 1.0),
        height_ratios=(1.0, 1.0),
    )
    ax_global = fig.add_subplot(gs[:, 0])
    ax_l = fig.add_subplot(gs[0, 1])
    ax_r = fig.add_subplot(gs[1, 1])

    q = _draw_global_sparse_panel(
        ax_global,
        left_image,
        left_h,
        left_w,
        roi,
        R,
        C,
        Rq,
        Cq,
        Uq,
        Vq,
        magq,
        global_gain,
        show_title=True,
        show_info=True,
        show_roi_label=True,
        show_scale=True,
        scale_mode="quiverkey",
        mag_norm=mag_norm,
        mag_cmap=mag_cmap,
    )
    cbar = fig.colorbar(q, ax=ax_global, fraction=0.038, pad=0.02)
    cbar.set_label("Disparity magnitude $|d|$ (pixel)")
    style_colorbar(cbar)

    _draw_local_match_pair(
        ax_l,
        ax_r,
        stage,
        left_image,
        right_image,
        left_h,
        left_w,
        right_h,
        right_w,
        roi,
        "#e31a1c",
        "Region",
        max_local_points,
        local_arrow_target_fraction,
        show_annotations=True,
        mag_norm=mag_norm,
        mag_cmap=mag_cmap,
    )

    ax_global.text(
        -0.12, 1.02, "(a)", transform=ax_global.transAxes,
        fontsize=9, fontweight="bold", ha="left", va="bottom"
    )
    ax_l.text(
        -0.10, 1.03, "(b)", transform=ax_l.transAxes,
        fontsize=9, fontweight="bold", ha="left", va="bottom"
    )
    ax_r.text(
        -0.10, 1.03, "(c)", transform=ax_r.transAxes,
        fontsize=9, fontweight="bold", ha="left", va="bottom"
    )

    save_figure(
        fig,
        out_dir / f"{prefix}_sparse_match_disparity_local",
        dpi,
        formats,
    )

    _save_global_sparse_panel(
        out_dir,
        prefix,
        left_image,
        left_h,
        left_w,
        roi,
        R,
        C,
        Rq,
        Cq,
        Uq,
        Vq,
        magq,
        global_gain,
        width_mm,
        dpi,
        formats,
        mag_norm=mag_norm,
        mag_cmap=mag_cmap,
    )
    _save_single_local_region_panel(
        stage,
        out_dir,
        prefix,
        left_image,
        right_image,
        left_h,
        left_w,
        right_h,
        right_w,
        roi,
        "#e31a1c",
        "Region",
        max_local_points,
        local_arrow_target_fraction,
        width_mm,
        dpi,
        formats,
        mag_norm=mag_norm,
        mag_cmap=mag_cmap,
    )

def formats_from_output_mode(mode: str) -> Tuple[str, ...]:
    if mode == "png":
        return ("png",)
    if mode == "png+pdf":
        return ("png", "pdf")
    raise ValueError(f"未知输出模式: {mode}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate separate publication-ready disparity panels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("cfg", type=Path, help="pipeline cfg")
    parser.add_argument("--source", choices=("grid", "match"), default="grid")
    parser.add_argument("--left-dir", type=Path, default=None, help="覆盖左影像 level-0 目录")
    parser.add_argument("--right-dir", type=Path, default=None, help="覆盖右影像 level-0 目录")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN, help="匹配文件模板")
    parser.add_argument("--grid-step", type=int, default=0, help="grid 步长；0=cfg batch_size/自动估计")
    parser.add_argument("--scatter-scale", type=positive_float, default=16.0, help="match 栅格化倍率")
    parser.add_argument("--no-grid-fallback", action="store_true", help="grid 文件缺失时不回退 match")
    parser.add_argument("--background", type=Path, default=None, help="左 mosaic 灰度背景")
    parser.add_argument("--background-alpha", type=float, default=0.35)
    parser.add_argument(
        "--crop",
        nargs=4,
        type=float,
        metavar=("R_MIN", "R_MAX", "C_MIN", "C_MAX"),
        default=None,
        help="按 mosaic 坐标裁剪重点区域",
    )
    parser.add_argument("--marker-size", type=positive_float, default=3.0)
    parser.add_argument("--max-arrows", type=int, default=2500, help="箭头图最大箭头数；<=0 表示不抽稀")
    parser.add_argument("--min-arrow-magnitude", type=float, default=0.0, help="箭头图最小视差长度")
    parser.add_argument("--arrow-scale", type=float, default=0.0, help="matplotlib quiver scale；0=自动")
    parser.add_argument("--figure-width-mm", type=positive_float, default=89.0, help="单栏图宽；双栏可用 178")
    parser.add_argument(
        "--font-scale",
        type=positive_float,
        default=1.0,
        help="仅放大坐标轴标签/刻度与 legend/colorbar 文字，不改变图幅与标注位置；默认建议 2.4",
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--prefix", default=None, help="输出文件名前缀；默认 <id1>_<source>")
    parser.add_argument("--out-dir", type=Path, default=None, help="输出目录")
    parser.add_argument(
        "--output-mode",
        choices=("png", "png+pdf"),
        default="png",
        help="输出模式：png=仅 PNG；png+pdf=同时输出 PNG 与 PDF",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=80,
        help="视差分布直方图 bin 数",
    )
    parser.add_argument(
        "--fill-gaps",
        action="store_true",
        help="额外输出高斯面元插值后的 dx/dy/score（*_filled_*）；未内插结果始终保存。"
        "空白点在「内陆」或「距有效点≤fill-max-gap」满足其一即补偿",
    )
    parser.add_argument(
        "--fill-max-gap",
        type=int,
        default=8,
        help="邻近判据：距有效样本的最大距离（栅格格数，Chebyshev）",
    )
    parser.add_argument(
        "--fill-sigma",
        type=positive_float,
        default=2.0,
        help="高斯面元插值 σ（栅格格数）",
    )
    parser.add_argument(
        "--fill-radius",
        type=int,
        default=0,
        help="高斯核截断半径（栅格格数）；0 表示使用 ceil(3σ)",
    )
    parser.add_argument(
        "--despike",
        action="store_true",
        help="局部粗差去除：按 3×3 邻域视差不一致性剔除异常栅格/点（在出图与填洞之前）",
    )
    parser.add_argument(
        "--despike-k",
        type=positive_float,
        default=3.0,
        help="粗差阈值：median(E)+k·1.4826·MAD(E) 中的 k",
    )
    parser.add_argument(
        "--despike-abs",
        type=float,
        default=0.0,
        help="不一致性绝对下限（像素）；>0 时 thr=max(稳健阈值, abs)",
    )
    parser.add_argument(
        "--despike-min-neighbors",
        type=int,
        default=3,
        help="3×3 内至少多少个其他有效邻域才允许判粗差",
    )

    parser.add_argument(
        "--sparse-local-panel",
        action="store_true",
        help="输出稀疏匹配图：(a) 全局点+箭头；(b,c) 一个局部左右影像纵向子块",
    )
    parser.add_argument(
        "--only-sparse-local-panel",
        action="store_true",
        help="仅输出 sparse-local 组合图，不再输出旧的 dx/dy/score/quiver/hist 面板",
    )
    parser.add_argument(
        "--left-background",
        type=Path,
        default=None,
        help="左影像 mosaic 背景；默认由 cfg.dataset.filepath 自动定位",
    )
    parser.add_argument(
        "--right-background",
        type=Path,
        default=None,
        help="右影像 mosaic 背景；默认由 cfg.dataset.filepath 自动定位",
    )
    parser.add_argument(
        "--mosaic-name",
        default="mosaic_ds4.tif",
        help="自动背景文件名，目录为 dataset.filepath/ID/downsample/0/",
    )
    parser.add_argument(
        "--roi1",
        nargs=4,
        type=float,
        metavar=("R_MIN", "R_MAX", "C_MIN", "C_MAX"),
        default=None,
        help="局部区域（左 mosaic 坐标）；不提供则自动选择。仅使用一个子块",
    )
    parser.add_argument(
        "--roi2",
        nargs=4,
        type=float,
        metavar=("R_MIN", "R_MAX", "C_MIN", "C_MAX"),
        default=None,
        help="(已弃用) 仅保留一个局部子块，此参数忽略",
    )
    parser.add_argument(
        "--roi-height",
        type=positive_float,
        default=5000.0,
        help="自动局部框高度（level-0 mosaic pixel）",
    )
    parser.add_argument(
        "--roi-width",
        type=positive_float,
        default=5000.0,
        help="自动局部框宽度（level-0 mosaic pixel）",
    )
    parser.add_argument(
        "--local-max-points",
        type=int,
        default=120,
        help="每个局部左右影像最多显示的对应点数量",
    )
    parser.add_argument(
        "--global-arrow-target-fraction",
        type=positive_float,
        default=0.045,
        help="全局箭头中位长度占 min(H,W) 的目标比例（仅显示缩放）",
    )
    parser.add_argument(
        "--local-arrow-target-fraction",
        type=positive_float,
        default=0.13,
        help="局部箭头中位长度占局部框短边的目标比例（仅显示缩放）",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.crop is not None:
        r_min, r_max, c_min, c_max = args.crop
        if not (r_min < r_max and c_min < c_max):
            raise ValueError("--crop 必须满足 R_MIN<R_MAX 且 C_MIN<C_MAX")
    if not (0.0 <= args.background_alpha <= 1.0):
        raise ValueError("--background-alpha 必须在 [0, 1] 内")
    if args.dpi <= 0:
        raise ValueError("--dpi 必须 > 0")
    if args.hist_bins < 5:
        raise ValueError("--hist-bins 必须 >= 5")
    if args.fill_max_gap < 0:
        raise ValueError("--fill-max-gap 必须 >= 0")
    if args.fill_radius < 0:
        raise ValueError("--fill-radius 必须 >= 0")
    if args.despike_abs < 0:
        raise ValueError("--despike-abs 必须 >= 0")
    if args.despike_min_neighbors < 1:
        raise ValueError("--despike-min-neighbors 必须 >= 1")

    for name in ("roi1", "roi2"):
        roi = getattr(args, name)
        if roi is not None:
            r_min, r_max, c_min, c_max = roi
            if not (r_min < r_max and c_min < c_max):
                raise ValueError(f"--{name} 必须满足 R_MIN<R_MAX 且 C_MIN<C_MAX")
    if args.local_max_points == 0 or args.local_max_points < -1:
        raise ValueError("--local-max-points 必须为正数或 -1（不限制）")
    if args.only_sparse_local_panel:
        args.sparse_local_panel = True


def save_points_txt(stage: StageData, out_dir: Path, prefix: str) -> Path:
    """Write mosaic-frame matches: R1 C1 dx dy score left_ccd right_ccd."""
    out_dir.mkdir(parents=True, exist_ok=True)
    points_path = out_dir / f"{prefix}_points.txt"
    n = int(stage.R1.size)
    with points_path.open("w", encoding="utf-8") as stream:
        stream.write("R1 C1 dx dy score left_ccd right_ccd\n")
        for index in range(n):
            stream.write(
                f"{float(stage.R1[index]):.3f} "
                f"{float(stage.C1[index]):.3f} "
                f"{float(stage.DX[index]):.3f} "
                f"{float(stage.DY[index]):.3f} "
                f"{float(stage.score[index]):.6f} "
                f"{int(stage.left_ccd[index])} "
                f"{int(stage.right_ccd[index])}\n"
            )
    print(f"[OK] {points_path}  (N={n})")
    return points_path


def _save_scalar_panels(
    stage: StageData,
    background: Optional[np.ndarray],
    out_dir: Path,
    prefix: str,
    crop: Optional[Tuple[float, float, float, float]],
    marker_size: float,
    background_alpha: float,
    width_mm: float,
    dpi: int,
    formats: Sequence[str],
    use_raster: bool,
) -> None:
    panels = (
        ("dx", "Horizontal disparity", "Horizontal disparity $d_x$ (pixel)"),
        ("dy", "Vertical disparity", "Vertical disparity $d_y$ (pixel)"),
        ("score", "Matching score", "Matching score"),
    )
    for quantity, title, cbar in panels:
        save_scalar_panel(
            stage,
            background,
            out_dir,
            prefix,
            quantity,
            title,
            cbar,
            crop,
            marker_size,
            background_alpha,
            width_mm,
            dpi,
            formats,
            use_raster=use_raster,
        )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        validate_args(args)
        setup_publication_style(font_scale=args.font_scale)
        print(f"[INFO] font_scale={args.font_scale}")
        formats = formats_from_output_mode(args.output_mode)

        stage = build_stage(
            label="Disparity",
            cfg_path=args.cfg,
            source=args.source,
            left_dir_override=args.left_dir,
            right_dir_override=args.right_dir,
            pattern=args.pattern,
            grid_step_override=args.grid_step,
            scatter_scale=args.scatter_scale,
            allow_grid_fallback=not args.no_grid_fallback,
        )

        if args.despike:
            apply_local_outlier_removal(
                stage,
                k=args.despike_k,
                abs_thr=args.despike_abs,
                min_neighbors=args.despike_min_neighbors,
            )

        out_dir = args.out_dir
        if out_dir is None:
            out_dir = stage.left_dir / "disparity_paper_panels"
        prefix = args.prefix or f"{stage.cfg.id1}_{args.source}"
        crop = tuple(args.crop) if args.crop is not None else None
        background = load_background(args.background)

        # Always export point list (same schema as grid_to_mosaic_disparity.py).
        save_points_txt(stage, out_dir, prefix)

        if args.sparse_local_panel:
            roi1 = tuple(args.roi1) if args.roi1 is not None else None
            roi2 = tuple(args.roi2) if args.roi2 is not None else None
            save_sparse_local_match_panel(
                stage=stage,
                cfg_path=args.cfg,
                out_dir=out_dir,
                prefix=prefix,
                left_background=args.left_background,
                right_background=args.right_background,
                mosaic_name=args.mosaic_name,
                roi1=roi1,
                roi2=roi2,
                roi_height=args.roi_height,
                roi_width=args.roi_width,
                max_global_arrows=args.max_arrows,
                max_local_points=args.local_max_points,
                global_arrow_target_fraction=args.global_arrow_target_fraction,
                local_arrow_target_fraction=args.local_arrow_target_fraction,
                width_mm=max(args.figure_width_mm, 178.0),
                dpi=args.dpi,
                formats=formats,
            )
            if args.only_sparse_local_panel:
                print(f"[DONE] 仅输出 sparse-local 组合图: {out_dir}")
                return 0

        print(f"[INFO] output_mode={args.output_mode} formats={list(formats)}")
        print(f"[INFO] despike={args.despike} fill_gaps={args.fill_gaps}")

        # Always write unfilled panels first (match=scatter / grid=raw raster).
        _save_scalar_panels(
            stage,
            background,
            out_dir,
            prefix,
            crop,
            args.marker_size,
            args.background_alpha,
            args.figure_width_mm,
            args.dpi,
            formats,
            use_raster=False,
        )

        if args.fill_gaps:
            apply_stage_gap_fill(
                stage,
                sigma=args.fill_sigma,
                radius=None if args.fill_radius <= 0 else args.fill_radius,
                max_gap=args.fill_max_gap,
            )
            _save_scalar_panels(
                stage,
                background,
                out_dir,
                f"{prefix}_filled",
                crop,
                args.marker_size,
                args.background_alpha,
                args.figure_width_mm,
                args.dpi,
                formats,
                use_raster=True,
            )

        save_quiver_panel(
            stage,
            background,
            out_dir,
            prefix,
            crop,
            args.max_arrows,
            args.min_arrow_magnitude,
            args.arrow_scale if args.arrow_scale > 0 else None,
            args.background_alpha,
            args.figure_width_mm,
            args.dpi,
            formats,
        )
        save_disparity_distribution_panel(
            stage,
            out_dir,
            prefix,
            crop,
            args.figure_width_mm,
            args.dpi,
            formats,
            n_bins=args.hist_bins,
        )

        print(f"[DONE] 输出目录: {out_dir}")
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
