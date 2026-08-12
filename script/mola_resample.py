#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import numpy as np
from osgeo import gdal

MARS_RADIUS = 3396190.0


# =========================
# 坐标转换
# =========================
def rect_to_latlon_h(x, y, z, r_mars=MARS_RADIUS):
    rho = math.sqrt(x * x + y * y + z * z)
    if rho == 0:
        raise ValueError("Invalid zero-length coordinate.")

    lat = math.asin(z / rho)
    lon = math.atan2(y, x)
    h = rho - r_mars

    return math.degrees(lat), math.degrees(lon), h


def latlon_h_to_rect(lat_rad, lon_rad, h, r_mars=MARS_RADIUS):
    radius = r_mars + h
    x = radius * math.cos(lat_rad) * math.cos(lon_rad)
    y = radius * math.cos(lat_rad) * math.sin(lon_rad)
    z = radius * math.sin(lat_rad)
    return x, y, z


# =========================
# 读取XYZ范围
# =========================
def read_xyz_range(xyz_file):
    with open(xyz_file, "r") as f:
        n = int(f.readline())

        lat_min, lat_max = 1e9, -1e9
        lon_min, lon_max = 1e9, -1e9

        for _ in range(n):
            vals = f.readline().split()
            if len(vals) < 3:
                continue

            x, y, z = map(float, vals[:3])
            lat, lon, _ = rect_to_latlon_h(x, y, z)

            lat_min = min(lat_min, lat)
            lat_max = max(lat_max, lat)
            lon_min = min(lon_min, lon)
            lon_max = max(lon_max, lon)

    return lat_min, lat_max, lon_min, lon_max


# =========================
# 扩展范围（核心）
# =========================
def expand_range(lat_min, lat_max, lon_min, lon_max, ratio):
    lat_c = 0.5 * (lat_min + lat_max)
    lon_c = 0.5 * (lon_min + lon_max)

    lat_half = 0.5 * (lat_max - lat_min)
    lon_half = 0.5 * (lon_max - lon_min)

    lat_half_new = lat_half * ratio
    lon_half_new = lon_half * ratio

    lat_min_new = lat_c - lat_half_new
    lat_max_new = lat_c + lat_half_new
    lon_min_new = lon_c - lon_half_new
    lon_max_new = lon_c + lon_half_new

    # 防止越界
    lat_min_new = max(-90.0, lat_min_new)
    lat_max_new = min(90.0, lat_max_new)
    lon_min_new = max(-180.0, lon_min_new)
    lon_max_new = min(180.0, lon_max_new)

    return lat_min_new, lat_max_new, lon_min_new, lon_max_new


# =========================
# DEM采样
# =========================
def sample_dem_3x3_weighted(dem, row, col):
    n_rows, n_cols = dem.shape

    row0 = int(row + 0.5)
    col0 = int(col + 0.5)

    val, wt_sum = 0.0, 0.0

    for dr in range(-1, 2):
        for dc in range(-1, 2):
            rr = max(0, min(n_rows - 1, row0 + dr))
            cc = (col0 + dc) % n_cols

            dist2 = (cc - col) ** 2 + (rr - row) ** 2
            w = math.exp(-dist2)

            val += w * float(dem[rr, cc])
            wt_sum += w

    return val / wt_sum if wt_sum > 0 else np.nan


# =========================
# 主函数
# =========================
def mola_resample(xyz_file, dem_file, out_file, pixel_size, expand_ratio):
    gdal.AllRegister()

    ds = gdal.Open(dem_file)
    if ds is None:
        raise RuntimeError("DEM open failed")

    dem = ds.GetRasterBand(1).ReadAsArray()
    n_rows, n_cols = dem.shape

    # ===== 原始范围 =====
    lat_min, lat_max, lon_min, lon_max = read_xyz_range(xyz_file)

    print("\nOriginal range:")
    print(lat_min, lat_max, lon_min, lon_max)

    # ===== 扩展 =====
    lat_min, lat_max, lon_min, lon_max = expand_range(
        lat_min, lat_max, lon_min, lon_max, expand_ratio
    )

    print("\nExpanded range:")
    print(lat_min, lat_max, lon_min, lon_max)

    lat_min = math.radians(lat_min)
    lat_max = math.radians(lat_max)
    lon_min = math.radians(lon_min)
    lon_max = math.radians(lon_max)

    step = pixel_size / MARS_RADIUS

    lats = np.arange(lat_min, lat_max, step)
    lons = np.arange(lon_min, lon_max, step)

    count = len(lats) * len(lons)

    with open(out_file, "w") as f:
        f.write(f"{count}\n")

        for lat in lats:
            for lon in lons:
                lat_deg = math.degrees(lat)
                lon_deg = math.degrees(lon)

                row = (90 - lat_deg) / 180 * n_rows
                col = (lon_deg + 180) / 360 * n_cols

                h = sample_dem_3x3_weighted(dem, row, col)

                x, y, z = latlon_h_to_rect(lat, lon, h)
                f.write(f"{x} {y} {z} {h}\n")

    print("\nOutput:", out_file)


def print_range_info(name, lat_min, lat_max, lon_min, lon_max):
    lat_center = 0.5 * (lat_min + lat_max)
    lon_center = 0.5 * (lon_min + lon_max)

    lat_span = lat_max - lat_min
    lon_span = lon_max - lon_min

    print("\n" + "="*60)
    print(f"{name}")
    print("="*60)

    print("Degrees:")
    print(f"  Latitude : [{lat_min:.6f}, {lat_max:.6f}]  span={lat_span:.6f}")
    print(f"  Longitude: [{lon_min:.6f}, {lon_max:.6f}]  span={lon_span:.6f}")
    print(f"  Center   : lat={lat_center:.6f}, lon={lon_center:.6f}")

    print("Radians:")
    print(f"  Latitude : [{math.radians(lat_min):.6f}, {math.radians(lat_max):.6f}]")
    print(f"  Longitude: [{math.radians(lon_min):.6f}, {math.radians(lon_max):.6f}]")


# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("xyz_file")
    parser.add_argument("dem_file")
    parser.add_argument("out_file")

    parser.add_argument("--pixel-size", type=float, default=30.0)
    parser.add_argument("--expand-ratio", type=float, default=1.0)

    args = parser.parse_args()

    mola_resample(
        args.xyz_file,
        args.dem_file,
        args.out_file,
        args.pixel_size,
        args.expand_ratio,
    )


if __name__ == "__main__":
    main()