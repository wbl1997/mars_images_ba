#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 HiRISE 影像观测信息汇总表
输入： base 文件夹路径，序列号 (如 ESP_069731_2055)
输出：
10
../data/EO/PSP_001777_1650/
PSP_001777_1650_RED0_0 850427621:44577 131 1 128 40000
...
"""

import os
import re
import sys
from pathlib import Path

# -------------------------------------------------------------------
# 读取 .IMG 文件头部（ASCII 标签区）
# -------------------------------------------------------------------
def read_pds3_label(img_path, max_bytes=2_000_000):
    with open(img_path, "rb") as f:
        text = f.read(max_bytes).decode("latin-1", errors="ignore")
        endpos = re.search(r"^\s*END\s*$", text, re.M)
        return text[: endpos.end()] if endpos else text

# -------------------------------------------------------------------
# 从标签中提取字段
# -------------------------------------------------------------------
def parse_hirise_fields(text):
    def find_num(key):
        m = re.search(rf"{key}\s*=\s*([0-9E\.\+\-]+)", text)
        return float(m.group(1)) if m else None
    def find_str(key):
        m = re.search(rf'{key}\s*=\s*"?(.*?)"?(?:\s|<|$)', text)
        return m.group(1).strip('"') if m else None

    info = {}
    info["SCLK"] = find_str("SPACECRAFT_CLOCK_START_COUNT")
    info["DLINE"] = find_num("MRO:DELTA_LINE_TIMER_COUNT") or 0
    info["BIN"]   = int(find_num("MRO:BINNING") or 1)
    info["TDI"]   = int(find_num("MRO:TDI") or 0)

    m = re.search(r"OBJECT\s*=\s*IMAGE.*?LINES\s*=\s*(\d+)", text, re.S)
    info["LINES"] = int(m.group(1)) if m else 0

    # 计算 Line Rate (秒)
    info["LINE_RATE"] = (74.0 + info["DLINE"] / 16.0) / 1_000_000
    return info

# -------------------------------------------------------------------
# 主程序
# -------------------------------------------------------------------
def main():
    if len(sys.argv) < 5:
        print("Usage: python generate_hirise_info.py <base_dir> <product_id> <output_file> <output_path>")
        sys.exit(0)

    base_dir = Path(sys.argv[1]).expanduser()
    product_id = sys.argv[2]
    output_file = Path(sys.argv[3])
    output_path = sys.argv[4]  # 新增输出路径参数

    # HiRISE RED0–RED9 通道
    channels = [f"RED{i}_0" for i in range(10)]
    found_files = []
    results = []

    for ch in channels:
        img_file = base_dir / product_id / f"{product_id}_{ch}.IMG"
        if img_file.exists():
            found_files.append(img_file)

    if not found_files:
        print(f"[error] no .IMG found under {base_dir / product_id}")
        sys.exit(1)

    for f in found_files:
        label = read_pds3_label(f)
        info = parse_hirise_fields(label)
        name = f.stem
        results.append(f"{name} {info['SCLK']} {int(info['DLINE'])} {info['BIN']} {info['TDI']} {info['LINES']}")

    # 输出到文件
    with open(output_file, 'w') as f:
        f.write(f"{len(results)}\n")
        f.write(f"{output_path}\n")  # 使用命令行指定的输出路径
        for line in results:
            f.write(f"{line}\n")
    
    # 同时输出到 stdout
    print(len(results))
    print(f"{output_path}")
    for line in results:
        print(line)

# -------------------------------------------------------------------
if __name__ == "__main__":
    main()
