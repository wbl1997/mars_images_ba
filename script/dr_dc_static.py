#!/usr/bin/env python3
"""统计 intra 匹配成功点的 dr/dc，并按文件绘制 row-dr / row-dc 图。"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PLACEHOLDER_TOKEN = "REDi_i+1"
DEFAULT_BATCH_GLOB = "*_intra__.txt"


@dataclass(frozen=True)
class MatchPoint:
    line_no: int
    src_row: float
    src_col: float
    img_id: int
    dst_row: float
    dst_col: float
    score: float
    dr: float
    dc: float


@dataclass
class FileStatResult:
    file: str
    total_nonempty_lines: int
    malformed_lines: int
    matched_success_count: int
    used_point_count: int
    outlier_removed_count: int
    filter_enabled: bool
    matched_success_ratio: float
    dr_stats: dict[str, float | int | None]
    dc_stats: dict[str, float | int | None]
    by_img_id: dict[str, dict[str, object]]
    plot_path: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="统计匹配成功点并绘制 row-dr / row-dc 图（按 img_id 区分）。"
    )
    parser.add_argument("input_path", type=Path, help="输入文件、目录或带 REDi_i+1 占位符的路径")
    parser.add_argument(
        "--pattern",
        default=DEFAULT_BATCH_GLOB,
        help=f"目录模式下文件匹配模式（默认: {DEFAULT_BATCH_GLOB}）",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./dr_dc_plots"),
        help="图像输出目录（默认: ./dr_dc_plots）",
    )
    parser.add_argument("--dpi", type=int, default=150, help="输出图像 DPI（默认: 150）")
    parser.add_argument(
        "--filter-outliers",
        action="store_true",
        help="启用粗差滤波后再统计/绘图（默认关闭）",
    )
    parser.add_argument(
        "--mad-k",
        type=float,
        default=3.5,
        help="MAD 粗差判定阈值 k（默认: 3.5，越小越严格）",
    )
    parser.add_argument("--json-out", type=Path, help="将统计结果写入 JSON 文件")
    return parser


def _robust_zscore(values: list[float]) -> list[float]:
    if not values:
        return []

    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n % 2 == 1:
        median_v = sorted_vals[n // 2]
    else:
        median_v = 0.5 * (sorted_vals[n // 2 - 1] + sorted_vals[n // 2])

    abs_dev = [abs(v - median_v) for v in values]
    sorted_abs = sorted(abs_dev)
    m = len(sorted_abs)
    if m % 2 == 1:
        mad = sorted_abs[m // 2]
    else:
        mad = 0.5 * (sorted_abs[m // 2 - 1] + sorted_abs[m // 2])

    if mad == 0:
        return [0.0 for _ in values]

    # 与标准差近似一致的鲁棒 z-score
    return [0.6745 * (v - median_v) / mad for v in values]


def filter_outlier_points(points: list[MatchPoint], mad_k: float) -> list[MatchPoint]:
    """按 dr/dc 的鲁棒 z-score 过滤粗差点。"""
    if len(points) < 5:
        return points

    drs = [p.dr for p in points]
    dcs = [p.dc for p in points]
    z_dr = _robust_zscore(drs)
    z_dc = _robust_zscore(dcs)

    kept: list[MatchPoint] = []
    for p, zr, zc in zip(points, z_dr, z_dc):
        if abs(zr) <= mad_k and abs(zc) <= mad_k:
            kept.append(p)
    return kept


def calc_stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    n = len(values)
    mean_v = sum(values) / n
    var_v = sum((v - mean_v) ** 2 for v in values) / n
    return {
        "count": n,
        "mean": mean_v,
        "std": math.sqrt(var_v),
        "min": min(values),
        "max": max(values),
    }


def collect_input_files(input_path: Path, pattern: str) -> list[Path]:
    if input_path.is_file():
        return [input_path]

    if input_path.is_dir():
        return sorted([p for p in input_path.glob(pattern) if p.is_file()])

    if PLACEHOLDER_TOKEN in input_path.name and input_path.parent.is_dir():
        template = input_path.name
        regex_text = re.escape(template).replace(re.escape(PLACEHOLDER_TOKEN), r"RED\d+_\d+")
        regex = re.compile(rf"^{regex_text}$")
        return sorted(
            [p for p in input_path.parent.iterdir() if p.is_file() and regex.fullmatch(p.name)]
        )

    raise FileNotFoundError(f"路径不存在或不支持: {input_path}")


def parse_matched_points(path: Path) -> tuple[list[MatchPoint], int, int]:
    points: list[MatchPoint] = []
    total_nonempty = 0
    malformed = 0

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            total_nonempty += 1
            parts = line.split()

            if len(parts) not in (3, 7):
                malformed += 1
                continue

            if len(parts) == 3:
                # 三列视为未匹配点
                continue

            # 7列视为匹配成功：
            # [col1 col2 col3 col4 col5 col6 col7]
            # 其中 dr = col5-col2, dc = col6-col3
            try:
                _left_img_id = int(float(parts[0]))  # 仅保留，不参与“是否匹配成功”判断
                src_row = float(parts[1])            # col2
                src_col = float(parts[2])            # col3
                img_id = int(float(parts[3]))        # col4（匹配图像id）
                dst_row = float(parts[4])            # col5
                dst_col = float(parts[5])            # col6
                score = float(parts[6])              # col7
            except ValueError:
                malformed += 1
                continue

            points.append(
                MatchPoint(
                    line_no=line_no,
                    src_row=src_row,
                    src_col=src_col,
                    img_id=img_id,
                    dst_row=dst_row,
                    dst_col=dst_col,
                    score=score,
                    dr=dst_row - src_row,
                    dc=dst_col - src_col,
                )
            )

    return points, total_nonempty, malformed


def plot_row_dr_dc(points: list[MatchPoint], file_path: Path, out_dir: Path, dpi: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / f"{file_path.stem}_row_dr_dc.png"

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    grouped: dict[int, list[MatchPoint]] = defaultdict(list)
    for p in points:
        grouped[p.img_id].append(p)

    for img_id in sorted(grouped):
        grp = grouped[img_id]
        rows = [p.src_row for p in grp]
        drs = [p.dr for p in grp]
        dcs = [p.dc for p in grp]
        axes[0].scatter(rows, drs, s=8, alpha=0.7, label=f"img_id={img_id}")
        axes[1].scatter(rows, dcs, s=8, alpha=0.7, label=f"img_id={img_id}")

    axes[0].set_ylabel("dr (col5-col2)")
    axes[1].set_ylabel("dc (col6-col3)")
    axes[1].set_xlabel("row (col2)")
    axes[0].set_title(f"{file_path.name}: row-dr")
    axes[1].set_title(f"{file_path.name}: row-dc")
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)
    axes[1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=dpi)
    plt.close(fig)
    return plot_path


def summarize_file(path: Path, out_dir: Path, dpi: int, filter_outliers: bool, mad_k: float) -> FileStatResult:
    points, total_nonempty, malformed = parse_matched_points(path)
    used_points = filter_outlier_points(points, mad_k) if filter_outliers else points

    dr_values = [p.dr for p in used_points]
    dc_values = [p.dc for p in used_points]

    by_img_id: dict[str, dict[str, object]] = {}
    grouped: dict[int, list[MatchPoint]] = defaultdict(list)
    for p in used_points:
        grouped[p.img_id].append(p)

    for img_id in sorted(grouped):
        grp = grouped[img_id]
        by_img_id[str(img_id)] = {
            "count": len(grp),
            "dr_stats": calc_stats([p.dr for p in grp]),
            "dc_stats": calc_stats([p.dc for p in grp]),
        }

    plot_path = plot_row_dr_dc(used_points, path, out_dir, dpi) if used_points else out_dir / f"{path.stem}_row_dr_dc.png"

    ratio = (len(points) / total_nonempty) if total_nonempty > 0 else 0.0
    return FileStatResult(
        file=str(path),
        total_nonempty_lines=total_nonempty,
        malformed_lines=malformed,
        matched_success_count=len(points),
        used_point_count=len(used_points),
        outlier_removed_count=len(points) - len(used_points),
        filter_enabled=filter_outliers,
        matched_success_ratio=ratio,
        dr_stats=calc_stats(dr_values),
        dc_stats=calc_stats(dc_values),
        by_img_id=by_img_id,
        plot_path=str(plot_path),
    )


def print_file_summary(stat: FileStatResult) -> None:
    print(f"\n=== {Path(stat.file).name} ===")
    print(f"总非空行: {stat.total_nonempty_lines}, 异常行: {stat.malformed_lines}")
    print(f"匹配成功数: {stat.matched_success_count}, 成功率: {stat.matched_success_ratio:.4f}")
    if stat.filter_enabled:
        print(f"粗差滤波: 开启, 保留点数: {stat.used_point_count}, 剔除点数: {stat.outlier_removed_count}")
    else:
        print("粗差滤波: 关闭")
    print(f"dr统计: {stat.dr_stats}")
    print(f"dc统计: {stat.dc_stats}")
    print(f"图像输出: {stat.plot_path}")
    if stat.by_img_id:
        print("按 img_id 统计:")
        for img_id, info in stat.by_img_id.items():
            print(f"  img_id={img_id}, count={info['count']}, dr={info['dr_stats']}, dc={info['dc_stats']}")


def main() -> int:
    args = build_parser().parse_args()

    try:
        files = collect_input_files(args.input_path, args.pattern)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        print(f"错误: {exc}")
        return 2

    if not files:
        print("错误: 未找到可处理文件")
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_stats: list[FileStatResult] = []
    for file_path in files:
        stat = summarize_file(
            file_path,
            args.out_dir,
            args.dpi,
            filter_outliers=args.filter_outliers,
            mad_k=args.mad_k,
        )
        all_stats.append(stat)
        print_file_summary(stat)

    payload = {
        "input_path": str(args.input_path),
        "pattern": args.pattern,
        "filter_outliers": args.filter_outliers,
        "mad_k": args.mad_k,
        "file_count": len(all_stats),
        "files": [asdict(s) for s in all_stats],
    }

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n完整 JSON 已写入: {args.json_out}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
