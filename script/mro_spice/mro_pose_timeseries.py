#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于 SPICE meta-kernel(.tm)，在给定时间段生成 MRO 相对火星地固(IAU_MARS)的位姿序列。

输出 CSV 字段：
time_utc, et, x_km, y_km, z_km, vx_kms, vy_kms, vz_kms,
q_w, q_x, q_y, q_z,
r00, r01, r02, r10, r11, r12, r20, r21, r22

使用示例：
python mro_pose_timeseries.py \
  --tm mro_pose.tm \
  --start "2006-11-21T00:00:00" \
  --end   "2006-11-27T23:59:59" \
  --step 60 \
  --sc-frame MRO_SPACECRAFT \
  --out mro_pose_20061121_27_60s.csv
"""

import argparse
import csv
from datetime import datetime, timedelta, timezone
from typing import Tuple, List

import spiceypy as spice
import numpy as np


def parse_args():
    ap = argparse.ArgumentParser(description="MRO pose w.r.t. IAU_MARS over time")
    ap.add_argument("--tm", required=True, help="SPICE meta-kernel (.tm)")
    ap.add_argument("--start", required=True, help="UTC start, e.g. 2006-11-21T00:00:00[Z]")
    ap.add_argument("--end", required=True, help="UTC end,   e.g. 2006-11-27T23:59:59[Z]")
    ap.add_argument("--step", type=float, default=60.0, help="step in seconds (default 60)")
    ap.add_argument("--sc-frame", default="MRO_SPACECRAFT",
                    help="spacecraft body frame name (default: MRO_SPACECRAFT)")
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--with-vel", action="store_true", help="also compute velocity (uses spkezr)")
    return ap.parse_args()


def to_iso_z(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def parse_utc_any(s: str) -> str:
    s = s.strip()
    # 如果是 ISO + Z，换成空格 + UTC（STR2ET 对 'Z' 也OK，但统一到 ' UTC' 更稳）
    if s.endswith(("Z", "z")):
        return s[:-1].replace("T", " ") + " UTC"
    # 已带时区偏移（+HH:MM 或 -HH:MM）则直接去掉 T
    import re
    if re.search(r"[+\-]\d{2}:\d{2}$", s):
        return s.replace("T", " ")
    # 无时区就按 UTC 处理，并把 T 换空格
    return s.replace("T", " ") + " UTC"


def dcm_to_quat(r: np.ndarray) -> Tuple[float, float, float, float]:
    """
    将方向余弦矩阵(IAU_MARS<-SC)转为四元数，标量在前 (w,x,y,z)。
    采用数值稳健的分支实现。
    """
    t = np.trace(r)
    if t > 0.0:
        s = np.sqrt(max(t + 1.0, 0.0)) * 2.0
        w = 0.25 * s
        x = (r[2,1] - r[1,2]) / s
        y = (r[0,2] - r[2,0]) / s
        z = (r[1,0] - r[0,1]) / s
    else:
        # 选主对角最大者
        if r[0,0] > r[1,1] and r[0,0] > r[2,2]:
            s = np.sqrt(max(1.0 + r[0,0] - r[1,1] - r[2,2], 0.0)) * 2.0
            w = (r[2,1] - r[1,2]) / s
            x = 0.25 * s
            y = (r[0,1] + r[1,0]) / s
            z = (r[0,2] + r[2,0]) / s
        elif r[1,1] > r[2,2]:
            s = np.sqrt(max(1.0 + r[1,1] - r[0,0] - r[2,2], 0.0)) * 2.0
            w = (r[0,2] - r[2,0]) / s
            x = (r[0,1] + r[1,0]) / s
            y = 0.25 * s
            z = (r[1,2] + r[2,1]) / s
        else:
            s = np.sqrt(max(1.0 + r[2,2] - r[0,0] - r[1,1], 0.0)) * 2.0
            w = (r[1,0] - r[0,1]) / s
            x = (r[0,2] + r[2,0]) / s
            y = (r[1,2] + r[2,1]) / s
            z = 0.25 * s
    # 归一化
    q = np.array([w, x, y, z], dtype=float)
    n = np.linalg.norm(q)
    if n > 0:
        q = q / n
    # 保证 w>=0（统一符号约定）
    if q[0] < 0:
        q = -q
    return tuple(q.tolist())


def main():
    args = parse_args()

    # 1) 载入 kernels
    spice.kclear()
    spice.furnsh(args.tm)

    # 2) 时间网格
    et_start = spice.str2et(parse_utc_any(args.start))
    et_end   = spice.str2et(parse_utc_any(args.end))
    if et_end < et_start:
        raise ValueError("end < start")
    step = max(args.step, 1.0)

    ets: List[float] = []
    t = et_start
    while t <= et_end + 1e-9:
        ets.append(t)
        t += step

    # 3) 位姿计算（全部在 IAU_MARS 下）
    target_sc = "MRO"              # 航天器目标名（由 FK/SPK 决定，常用 'MRO'）
    observer  = "MARS"             # 观测者为火星质心
    ref_frame = "IAU_MARS"         # 地固坐标系
    corr      = "NONE"             # 无光行差，几何量
    sc_frame  = args.sc_frame      # 航天器机体系（例如 MRO_SPACECRAFT）

    rows = []
    for et in ets:
        # 位置/速度：MRO 相对 MARS，在 IAU_MARS 坐标系
        if args.with_vel:
            state, _lt = spice.spkezr(target_sc, et, ref_frame, corr, observer)  # 6x1
            pos = state[:3]  # km
            vel = state[3:]  # km/s
        else:
            pos, _lt = spice.spkpos(target_sc, et, ref_frame, corr, observer)    # 3x1
            vel = (0.0, 0.0, 0.0)

        # 姿态：从航天器机体坐标系 -> IAU_MARS 的 DCM
        # 若只要相机帧，可把 sc_frame 换成相机帧（例如 MRO_HIRISE_OPTICAL_AXIS 等）
        r_iaumars_from_sc = np.array(spice.pxform(sc_frame, ref_frame, et))  # 3x3
        q = dcm_to_quat(r_iaumars_from_sc)

        utc = spice.et2utc(et, "ISOC", 0)  # ISO 格式 UTC
        rows.append([
            utc, et,
            pos[0], pos[1], pos[2],
            vel[0], vel[1], vel[2],
            q[0], q[1], q[2], q[3],
            r_iaumars_from_sc[0,0], r_iaumars_from_sc[0,1], r_iaumars_from_sc[0,2],
            r_iaumars_from_sc[1,0], r_iaumars_from_sc[1,1], r_iaumars_from_sc[1,2],
            r_iaumars_from_sc[2,0], r_iaumars_from_sc[2,1], r_iaumars_from_sc[2,2],
        ])

    # 4) 写 CSV
    header = [
        "time_utc","et",
        "x_km","y_km","z_km","vx_kms","vy_kms","vz_kms",
        "q_w","q_x","q_y","q_z",
        "r00","r01","r02","r10","r11","r12","r20","r21","r22"
    ]
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    # 清理
    spice.kclear()
    print(f"[ok] wrote {args.out}  ({len(rows)} samples)")


if __name__ == "__main__":
    main()
