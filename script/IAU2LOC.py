#!/usr/bin/env python3
"""IAU 坐标到局部坐标转换（对应 MATLAB 脚本 IAU2LOC.m）。

主要流程：
1. 从文本读取 IAU 笛卡尔坐标点云（X, Y, Z）。
2. 以给定局部中心（未提供时使用输入点云均值）建立局部坐标系。
3. 将点转换到局部坐标，并将 Z 替换为相对火星半径的高程 H。
4. 导出为 ASCII PLY（默认 xyz.ply）。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np


R_MARS = 3_396_190.0

# 与 IAU2LOC.m 中的示例中心一致，可用于自测点生成
DEFAULT_CENTER_XYZ = np.array(
    [-3.272960367755164e06, 2.572781525152316e05, -8.546371929324134e05],
    dtype=np.float64,
)


def _local_rotation_from_center(center_xyz: np.ndarray, r_mars: float = R_MARS) -> tuple[np.ndarray, np.ndarray]:
    """根据局部中心（IAU XYZ）计算球面投影中心和旋转矩阵 R。"""
    cx, cy, cz = center_xyz.astype(np.float64)

    norm = np.sqrt(cx * cx + cy * cy + cz * cz)
    if norm == 0:
        raise ValueError("局部中心坐标不能为零向量。")

    x0 = r_mars * cx / norm
    y0 = r_mars * cy / norm
    z0 = r_mars * cz / norm

    lat = np.arcsin(z0 / np.sqrt(x0 * x0 + y0 * y0 + z0 * z0))
    lon = np.arctan(y0 / x0)
    if x0 < 0:
        lon = lon + np.pi

    r = np.array(
        [
            [-np.sin(lon), np.cos(lon), 0.0],
            [-np.cos(lon) * np.sin(lat), -np.sin(lon) * np.sin(lat), np.cos(lat)],
            [np.cos(lon) * np.cos(lat), np.sin(lon) * np.cos(lat), np.sin(lat)],
        ],
        dtype=np.float64,
    )

    origin = np.array([x0, y0, z0], dtype=np.float64)
    return origin, r


def iau2loc(
    gt_xyz: np.ndarray,
    center_xyz: Optional[np.ndarray] = None,
    r_mars: float = R_MARS,
) -> np.ndarray:
    """将 IAU XYZ 点转换为局部坐标（输出 Z 为高程 H）。

    参数
    ----
    gt_xyz: (N,3) IAU 笛卡尔坐标。
    center_xyz: 局部中心 IAU 坐标；为 None 时使用 gt_xyz 的均值。
    r_mars: 火星半径。

    返回
    ----
    local_cc: (N,3) [X_local, Y_local, H]。
    """
    if gt_xyz.ndim != 2 or gt_xyz.shape[1] != 3:
        raise ValueError("gt_xyz 必须是形状为 (N,3) 的数组。")

    xyz = np.asarray(gt_xyz, dtype=np.float64)
    if xyz.shape[0] == 0:
        raise ValueError("gt_xyz 不能为空。")

    if center_xyz is None:
        center = np.mean(xyz, axis=0)
    else:
        center = np.asarray(center_xyz, dtype=np.float64)
        if center.shape != (3,):
            raise ValueError("center_xyz 必须为长度为 3 的向量。")

    h = 5*(np.sqrt(np.sum(xyz * xyz, axis=1)) - r_mars)

    origin, rot = _local_rotation_from_center(center, r_mars=r_mars)
    local_xyz = (rot @ (xyz - origin).T).T

    local_cc = np.column_stack((local_xyz[:, 0], local_xyz[:, 1], h))
    return local_cc


def write_ascii_ply(points_xyz: np.ndarray, out_path: Path) -> None:
    """写出最小 ASCII PLY 点云。"""
    pts = np.asarray(points_xyz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points_xyz 必须是形状为 (N,3) 的数组。")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        np.savetxt(f, pts, fmt="%.10f %.10f %.10f")


def load_gt_xyz(txt_path: Path, skip_first_row: bool = True) -> np.ndarray:
    """读取 GT 文本并提取 XYZ（对应 MATLAB: GT(2:end,1:3)）。"""
    rows: list[list[float]] = []
    with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_idx, line in enumerate(f, start=1):
            if skip_first_row and line_idx == 1:
                continue

            s = line.strip()
            if not s:
                continue

            # 兼容空格/逗号分隔与 Fortran D 指数
            tokens = s.replace(",", " ").split()
            vals: list[float] = []
            for t in tokens:
                try:
                    vals.append(float(t.replace("D", "E").replace("d", "e")))
                except ValueError:
                    continue

            if len(vals) >= 3:
                rows.append(vals[:3])

    if not rows:
        raise ValueError(
            "未能从输入文件解析出任何 XYZ 点（每行需至少包含 3 个数值列）。"
        )

    return np.asarray(rows, dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IAU2LOC.m 的 Python 对应实现")
    parser.add_argument("--input", type=str, help="输入 txt 文件路径（至少包含 XYZ 三列）")
    parser.add_argument("--output", type=str, default="xyz.ply", help="输出 PLY 路径，默认 xyz.ply")
    parser.add_argument(
        "--center",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
    help="局部坐标系中心 XYZ（默认使用输入点云均值）",
    )
    parser.add_argument(
        "--no-skip-first-row",
        action="store_true",
        help="不跳过输入文件首行（默认行为与 MATLAB 脚本一致：跳过首行）",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="运行内置自测，不读取外部文件",
    )
    return parser.parse_args()


def _run_self_test() -> None:
    """轻量自测：构造随机点，验证维度与有限值。"""
    rng = np.random.default_rng(0)
    base = DEFAULT_CENTER_XYZ
    pts = base + rng.normal(scale=100.0, size=(100, 3))
    out = iau2loc(pts)

    assert out.shape == (100, 3)
    assert np.isfinite(out).all()
    # h 与半径差定义一致
    h_ref = 1*(np.sqrt(np.sum(pts * pts, axis=1)) - R_MARS)
    assert np.allclose(out[:, 2], h_ref)
    print("Self-test passed.")


def main() -> None:
    args = parse_args()

    if args.self_test:
        _run_self_test()
        return

    if not args.input:
        raise SystemExit("请提供 --input，或使用 --self-test。")

    in_path = Path(args.input)
    out_path = Path(args.output)

    xyz = load_gt_xyz(in_path, skip_first_row=not args.no_skip_first_row)
    center = None if args.center is None else np.array(args.center, dtype=np.float64)

    local_cc = iau2loc(xyz, center_xyz=center)
    write_ascii_ply(local_cc, out_path)

    print(f"Input points: {xyz.shape[0]}")
    print(f"Output written to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
