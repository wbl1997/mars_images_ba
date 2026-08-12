#!/usr/bin/env python3
"""Denoise *_FI.txt DEM point clouds by local continuous surface fitting.

Uses robust local quadratic surface fits (MLS-style) in a local tangent frame.
Residual is the signed height above the fitted surface along the local normal
(approx. orthogonal distance). A second discrete-noise pass uses local Z MAD.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


Cell = Tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove outliers from *_FI.txt DEM point clouds using local quadratic "
            "surface fitting and surface-normal residual distance."
        )
    )
    parser.add_argument("input", type=Path, help="Input *_FI.txt file")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Denoised output TXT")
    parser.add_argument("--noise-out", type=Path, default=None, help="Optional removed-point output TXT")
    parser.add_argument("--stats-out", type=Path, default=None, help="Optional JSON statistics output")
    parser.add_argument("--cell-size", type=float, default=100.0, help="XY grid cell size in point-cloud units (default 100)")
    parser.add_argument(
        "--neighbor-cells",
        type=int,
        default=0,
        help="Surface-fit neighborhood radius in grid cells (default 0 = current cell only)",
    )
    parser.add_argument("--min-points", type=int, default=20, help="Minimum local points required to test a cell")
    parser.add_argument(
        "--sigma",
        type=float,
        default=4.0,
        help="MAD sigma multiplier for surface residual rejection (default 3.0)",
    )
    parser.add_argument(
        "--abs-threshold",
        type=float,
        default=0.0,
        help=(
            "Optional absolute surface residual threshold. "
            "0 disables it; otherwise the stricter of sigma/absolute gates is used."
        ),
    )
    parser.add_argument(
        "--discrete-sigma",
        type=float,
        default=3,
        help=(
            "Second-pass discrete-noise MAD multiplier vs local Z median. "
            "0 disables it. Default 2.5"
        ),
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=2,
        help="Robust surface fitting iterations inside each local neighborhood",
    )
    parser.add_argument("--no-header", action="store_true", help="Treat first line as point data, not a count")
    parser.add_argument("--fmt", default="%.10f", help="Output numeric format")
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=None,
        help="Optional path to also save a static 3D PNG snapshot",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable interactive 3D visualization (enabled by default)",
    )
    parser.add_argument(
        "--plot-max-points",
        type=int,
        default=300000,
        help="Max points drawn in visualization (subsample if larger; default 300000)",
    )
    return parser.parse_args()


def first_data_line(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                return stripped
    raise ValueError(f"empty input file: {path}")


def has_count_header(path: Path, no_header: bool) -> bool:
    if no_header:
        return False
    parts = first_data_line(path).split()
    if len(parts) != 1:
        return False
    try:
        int(float(parts[0]))
    except ValueError:
        return False
    return True


def load_points(path: Path, no_header: bool) -> Tuple[np.ndarray, bool]:
    count_header = has_count_header(path, no_header)
    data = np.loadtxt(path, dtype=np.float64, comments="#", skiprows=1 if count_header else 0)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        raise ValueError(f"input must have at least 3 columns, got {data.shape[1]}: {path}")
    finite = np.isfinite(data[:, :3]).all(axis=1)
    if not np.all(finite):
        data = data[finite]
    return data, count_header


def save_points(path: Path, points: np.ndarray, count_header: bool, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        if count_header:
            f.write(f"{points.shape[0]}\n")
        np.savetxt(f, points, fmt=fmt)


def robust_sigma(values: np.ndarray) -> Tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    sigma = 1.4826 * mad
    if sigma <= 1e-12:
        sigma = float(np.std(values))
    return med, sigma


def build_cells(xy: np.ndarray, cell_size: float) -> Tuple[Dict[Cell, np.ndarray], np.ndarray]:
    xy_min = xy.min(axis=0)
    ij = np.floor((xy - xy_min) / cell_size).astype(np.int64)
    buckets: Dict[Cell, List[int]] = {}
    for idx, (i, j) in enumerate(ij):
        buckets.setdefault((int(i), int(j)), []).append(idx)
    cells = {cell: np.asarray(indices, dtype=np.int64) for cell, indices in buckets.items()}
    return cells, ij


def neighbor_indices(cells: Dict[Cell, np.ndarray], cell: Cell, radius: int) -> np.ndarray:
    indices: List[np.ndarray] = []
    ci, cj = cell
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            arr = cells.get((ci + di, cj + dj))
            if arr is not None:
                indices.append(arr)
    if not indices:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(indices)


def point_to_plane_signed_distance(xyz: np.ndarray, normal: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """Orthogonal signed point-to-plane distance for unit normal."""
    return (xyz - centroid) @ normal


def _local_tangent_frame(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """PCA local frame: centroid, t1, t2, normal (normal forced to have +Z)."""
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    t1 = vh[0].copy()
    normal = vh[-1].copy()
    nrm = float(np.linalg.norm(normal))
    if nrm <= 1e-12:
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        normal /= nrm
    if normal[2] < 0:
        normal = -normal
    t1 = t1 - normal * float(np.dot(t1, normal))
    t1_n = float(np.linalg.norm(t1))
    if t1_n <= 1e-12:
        # degenerate: pick an axis not parallel to normal
        axis = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        t1 = np.cross(normal, axis)
        t1 /= float(np.linalg.norm(t1))
    else:
        t1 /= t1_n
    t2 = np.cross(normal, t1)
    t2 /= float(np.linalg.norm(t2))
    return centroid, t1, t2, normal


def _to_local_uvw(
    xyz: np.ndarray,
    centroid: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    normal: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = xyz - centroid
    return d @ t1, d @ t2, d @ normal


def _eval_quadratic(coef: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """w ≈ a0 + a1 u + a2 v + a3 u² + a4 u v + a5 v²."""
    return (
        coef[0]
        + coef[1] * u
        + coef[2] * v
        + coef[3] * (u * u)
        + coef[4] * (u * v)
        + coef[5] * (v * v)
    )


def _fit_quadratic_or_plane(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Least-squares quadratic; fall back to plane (linear in u,v) if too few points."""
    n = int(u.size)
    if n >= 6:
        a = np.column_stack(
            (np.ones(n), u, v, u * u, u * v, v * v)
        )
        coef, *_ = np.linalg.lstsq(a, w, rcond=None)
        return np.asarray(coef, dtype=np.float64)

    # plane in local frame: w = a0 + a1 u + a2 v
    a = np.column_stack((np.ones(n), u, v))
    lin, *_ = np.linalg.lstsq(a, w, rcond=None)
    coef = np.zeros(6, dtype=np.float64)
    coef[:3] = lin
    return coef


def fit_local_quadratic_surface(
    xyz: np.ndarray,
    max_iterations: int,
    sigma: float,
) -> np.ndarray:
    """Robust local quadratic surface fit; residual = signed height along local normal.

    Continuity: within each neighborhood the surface is a smooth paraboloid
    ``w(u,v)`` in the local tangent frame (MLS-style continuous patch), not a plane.
    """
    n_pts = int(xyz.shape[0])
    active = np.ones(n_pts, dtype=bool)
    residual = np.zeros(n_pts, dtype=np.float64)

    for _ in range(max(1, max_iterations)):
        n_active = int(active.sum())
        if n_active < 3:
            break
        centroid, t1, t2, normal = _local_tangent_frame(xyz[active])
        u, v, w = _to_local_uvw(xyz, centroid, t1, t2, normal)
        coef = _fit_quadratic_or_plane(u[active], v[active], w[active])
        residual = w - _eval_quadratic(coef, u, v)

        med, rsigma = robust_sigma(residual[active])
        if rsigma <= 1e-12:
            break
        active = np.abs(residual - med) <= sigma * rsigma

    # Final residual with last inlier-derived frame/coef
    if int(active.sum()) >= 3:
        centroid, t1, t2, normal = _local_tangent_frame(xyz[active])
        u, v, w = _to_local_uvw(xyz, centroid, t1, t2, normal)
        coef = _fit_quadratic_or_plane(u[active], v[active], w[active])
        residual = w - _eval_quadratic(coef, u, v)
    return residual


def threshold_from_sigma(local_sigma: float, sigma_factor: float, abs_threshold: float) -> float:
    sigma_gate = sigma_factor * local_sigma if local_sigma > 1e-12 else math.inf
    if abs_threshold > 0.0:
        return min(sigma_gate, abs_threshold)
    return sigma_gate


def denoise(points: np.ndarray, args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, dict]:
    if args.cell_size <= 0:
        raise ValueError("--cell-size must be > 0")
    if args.neighbor_cells < 0:
        raise ValueError("--neighbor-cells must be >= 0")
    if args.min_points < 3:
        raise ValueError("--min-points must be >= 3")

    xyz = points[:, :3]
    cells, _cell_ids = build_cells(xyz[:, :2], args.cell_size)
    keep = np.ones(points.shape[0], dtype=bool)
    reasons = np.zeros(points.shape[0], dtype=np.int8)

    tested_cells = 0
    skipped_cells = 0
    for cell, cell_point_indices in cells.items():
        local_idx = neighbor_indices(cells, cell, args.neighbor_cells)
        if local_idx.size < args.min_points:
            skipped_cells += 1
            continue
        tested_cells += 1
        residual_all = fit_local_quadratic_surface(
            xyz[local_idx],
            args.max_iterations,
            args.sigma,
        )
        local_med, local_sigma = robust_sigma(residual_all)
        threshold = threshold_from_sigma(local_sigma, args.sigma, args.abs_threshold)
        if not math.isfinite(threshold):
            continue

        local_lookup = {int(v): k for k, v in enumerate(local_idx)}
        pos = np.asarray([local_lookup[int(v)] for v in cell_point_indices], dtype=np.int64)
        cell_residual = residual_all[pos]
        bad_surf = np.abs(cell_residual - local_med) > threshold
        if np.any(bad_surf):
            keep[cell_point_indices[bad_surf]] = False
            reasons[cell_point_indices[bad_surf]] = 1

        # Second pass: discrete spike removal vs local Z median (MAD)
        if args.discrete_sigma > 0:
            local_keep = keep[local_idx]
            ref_idx = local_idx[local_keep] if np.any(local_keep) else local_idx
            z_med, z_sigma = robust_sigma(xyz[ref_idx, 2])
            if z_sigma > 1e-12:
                cell_z = xyz[cell_point_indices, 2]
                bad_disc = np.abs(cell_z - z_med) > args.discrete_sigma * z_sigma
                still_kept = keep[cell_point_indices]
                bad_disc = bad_disc & still_kept
                if np.any(bad_disc):
                    keep[cell_point_indices[bad_disc]] = False
                    reasons[cell_point_indices[bad_disc]] = 2

    stats = {
        "input_points": int(points.shape[0]),
        "kept_points": int(keep.sum()),
        "removed_points": int((~keep).sum()),
        "removed_ratio": float((~keep).sum() / max(1, points.shape[0])),
        "grid_cells": int(len(cells)),
        "tested_cells": int(tested_cells),
        "skipped_cells": int(skipped_cells),
        "surface_residual_removed": int(np.count_nonzero(reasons == 1)),
        "discrete_noise_removed": int(np.count_nonzero(reasons == 2)),
        "cell_size": float(args.cell_size),
        "neighbor_cells": int(args.neighbor_cells),
        "fit_window": float(args.cell_size * (2 * args.neighbor_cells + 1)),
        "min_points": int(args.min_points),
        "sigma": float(args.sigma),
        "abs_threshold": float(args.abs_threshold),
        "discrete_sigma": float(args.discrete_sigma),
        "surface_model": "local_quadratic_mls",
        "residual": "signed_height_along_local_normal",
    }
    return points[keep], points[~keep], stats


def default_output_path(input_path: Path) -> Path:
    if input_path.name.endswith("_FI.txt"):
        return input_path.with_name(input_path.name[:-7] + "_FI_denoised.txt")
    return input_path.with_name(input_path.stem + "_denoised" + input_path.suffix)


def _subsample_points(xyz: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if xyz.shape[0] <= max_points:
        return xyz
    idx = rng.choice(xyz.shape[0], size=int(max_points), replace=False)
    return xyz[idx]


def _prepare_vis_points(
    kept: np.ndarray,
    removed: np.ndarray,
    max_points: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    n_keep = int(kept.shape[0])
    n_rem = int(removed.shape[0])
    if n_keep + n_rem <= max_points:
        keep_pts = kept[:, :3] if n_keep else np.empty((0, 3))
        rem_pts = removed[:, :3] if n_rem else np.empty((0, 3))
        return keep_pts, rem_pts

    rem_budget = min(n_rem, max(8000, int(0.12 * max_points))) if n_rem else 0
    keep_budget = max_points - rem_budget
    keep_pts = _subsample_points(kept[:, :3], keep_budget, rng) if n_keep else np.empty((0, 3))
    rem_pts = _subsample_points(removed[:, :3], rem_budget, rng) if n_rem else np.empty((0, 3))
    return keep_pts, rem_pts


def _height_colors(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    lo = float(np.percentile(z, 2)) if z.size else 0.0
    hi = float(np.percentile(z, 98)) if z.size else 1.0
    if hi <= lo:
        hi = lo + 1.0
    t = np.clip((z - lo) / (hi - lo), 0.0, 1.0)
    # simple viridis-like RGB without matplotlib dependency for open3d path
    r = np.clip(1.5 - 4.0 * np.abs(t - 0.75), 0.0, 1.0)
    g = np.clip(1.5 - 4.0 * np.abs(t - 0.50), 0.0, 1.0)
    b = np.clip(1.5 - 4.0 * np.abs(t - 0.25), 0.0, 1.0)
    return np.column_stack([r, g, b])


def _save_matplotlib_3d_snapshot(
    plot_out: Path,
    keep_pts: np.ndarray,
    rem_pts: np.ndarray,
    stats: dict,
) -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    plot_out = Path(plot_out)
    plot_out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    if keep_pts.size:
        ax.scatter(
            keep_pts[:, 0],
            keep_pts[:, 1],
            keep_pts[:, 2],
            c=keep_pts[:, 2],
            cmap="viridis",
            s=1.0,
            alpha=0.35,
            linewidths=0,
            label="kept",
        )
    if rem_pts.size:
        ax.scatter(
            rem_pts[:, 0],
            rem_pts[:, 1],
            rem_pts[:, 2],
            c="#d62728",
            s=8.0,
            alpha=0.9,
            linewidths=0,
            label="removed",
        )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    removed_ratio = float(stats.get("removed_ratio", 0.0))
    ax.set_title(
        f"FI denoise kept={stats.get('kept_points')} "
        f"removed={stats.get('removed_points')} ({removed_ratio:.3%})"
    )
    if rem_pts.size or keep_pts.size:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plot_out, dpi=160)
    plt.close(fig)


def show_denoise_visualization(
    kept: np.ndarray,
    removed: np.ndarray,
    stats: dict,
    max_points: int = 300000,
    plot_out: Path | None = None,
    show: bool = True,
) -> None:
    """Interactive 3D visualization: kept (height-colored) vs removed (red)."""
    keep_pts, rem_pts = _prepare_vis_points(kept, removed, max_points)
    removed_ratio = float(stats.get("removed_ratio", 0.0))
    window_name = (
        f"FI denoise  kept={stats.get('kept_points', keep_pts.shape[0])}  "
        f"removed={stats.get('removed_points', rem_pts.shape[0])} ({removed_ratio:.3%})  "
        f"sigma={stats.get('sigma', '?')}"
    )

    if plot_out is not None:
        # Snapshot uses a lighter subsample for speed.
        rng = np.random.default_rng(1)
        snap_keep = _subsample_points(keep_pts, min(80000, keep_pts.shape[0] or 1), rng) if keep_pts.size else keep_pts
        snap_rem = _subsample_points(rem_pts, min(20000, rem_pts.shape[0] or 1), rng) if rem_pts.size else rem_pts
        _save_matplotlib_3d_snapshot(plot_out, snap_keep, snap_rem, stats)

    if not show:
        return

    try:
        import open3d as o3d
    except ImportError:
        o3d = None

    if o3d is not None:
        geoms = []
        if keep_pts.size:
            pcd_keep = o3d.geometry.PointCloud()
            pcd_keep.points = o3d.utility.Vector3dVector(keep_pts.astype(np.float64))
            pcd_keep.colors = o3d.utility.Vector3dVector(_height_colors(keep_pts[:, 2]))
            geoms.append(pcd_keep)
        if rem_pts.size:
            pcd_rem = o3d.geometry.PointCloud()
            pcd_rem.points = o3d.utility.Vector3dVector(rem_pts.astype(np.float64))
            pcd_rem.paint_uniform_color([0.90, 0.10, 0.10])
            geoms.append(pcd_rem)
        if not geoms:
            print("[FI denoise] plot skipped: empty point cloud")
            return
        print(
            "[FI denoise] Open3D 3D view: rotate=LMB  pan=Shift+LMB  zoom=wheel/RMB  "
            "close window to continue"
        )
        o3d.visualization.draw_geometries(
            geoms,
            window_name=window_name,
            width=1400,
            height=900,
            point_show_normal=False,
        )
        return

    # Fallback: matplotlib interactive 3D (slower, still rotatable)
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    if keep_pts.size:
        ax.scatter(
            keep_pts[:, 0],
            keep_pts[:, 1],
            keep_pts[:, 2],
            c=keep_pts[:, 2],
            cmap="viridis",
            s=1.0,
            alpha=0.35,
            linewidths=0,
            label="kept",
        )
    if rem_pts.size:
        ax.scatter(
            rem_pts[:, 0],
            rem_pts[:, 1],
            rem_pts[:, 2],
            c="#d62728",
            s=8.0,
            alpha=0.9,
            linewidths=0,
            label="removed",
        )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(window_name)
    if keep_pts.size or rem_pts.size:
        ax.legend(loc="best")
    print("[FI denoise] matplotlib 3D fallback (install open3d for better interaction)")
    plt.show()


def main() -> int:
    args = parse_args()
    out_path = args.output or default_output_path(args.input)
    stats_path = args.stats_out or out_path.with_suffix(out_path.suffix + ".stats.json")
    do_plot = not bool(args.no_plot)

    points, count_header = load_points(args.input, args.no_header)
    kept, removed, stats = denoise(points, args)
    stats["input"] = str(args.input)
    stats["output"] = str(out_path)
    if args.noise_out is not None:
        stats["noise_output"] = str(args.noise_out)

    save_points(out_path, kept, count_header, args.fmt)
    if args.noise_out is not None:
        save_points(args.noise_out, removed, count_header, args.fmt)

    if do_plot:
        plot_out = args.plot_out
        show_denoise_visualization(
            kept=kept,
            removed=removed,
            stats=stats,
            max_points=max(1000, int(args.plot_max_points)),
            plot_out=plot_out,
            show=True,
        )
        stats["plot"] = "open3d_3d_popup"
        if plot_out is not None:
            stats["plot_out"] = str(plot_out)

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        "[FI denoise] "
        f"{args.input} -> {out_path} kept={stats['kept_points']}/{stats['input_points']} "
        f"removed={stats['removed_points']} ({stats['removed_ratio']:.3%}) stats={stats_path}"
        + (" plot=3d-popup" if do_plot else " plot=off")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
