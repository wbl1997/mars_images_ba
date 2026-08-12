#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 Mars 程序所需的 EO 输入配置文件。

数据来源（按优先级）：
  1. eo_setup 目录下已有的提取结果（{product_id}.txt）
  2. PDS3 .IMG 原始影像文件（通过 extract_hirise_from_img.py 中的解析逻辑）

输出：
  <mars_root>/data/EO/{product_id}.txt        —— EO 输入配置文件
  <mars_root>/data/EO/{product_id}/           —— EO 输出目录（自动创建）

文件格式：
  <CCD 数量>
  ../data/EO/{product_id}/
  {name} {SCLK} {DLINE} {BIN} {TDI} {LINES}
  ...

用法：
  python gen_eo_inputfile.py <product_id> [options]

  -r, --root      Mars 工程根目录（默认：脚本所在目录的上一级）
  -s, --eo-setup  eo_setup 目录路径（默认：在 root/data/EO/eo_setup 或通过搜索自动找到）
  -i, --img-dir   包含 .IMG 文件的目录（备用数据源）
  --rows          LINE 行数（覆盖从文件中读取的值）
"""

import argparse
import os
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# 从 eo_setup 文件解析
# ---------------------------------------------------------------------------
def parse_eo_setup_file(path: Path):
    """
    解析 eo_setup 格式的文件，返回 (count, entries) 列表。
    每个 entry 是 dict: name, SCLK, DLINE, BIN, TDI, LINES
    """
    lines = path.read_text().strip().splitlines()
    if len(lines) < 2:
        raise ValueError(f"文件格式错误：{path}")

    count = int(lines[0].strip())
    entries = []
    for line in lines[2:]:          # 跳过 count 行和路径行
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 6:
            print(f"  [警告] 跳过格式不匹配的行：{line}")
            continue
        entries.append({
            "name":  parts[0],
            "SCLK":  parts[1],
            "DLINE": int(parts[2]),
            "BIN":   int(parts[3]),
            "TDI":   int(parts[4]),
            "LINES": int(parts[5]),
        })
    return count, entries


# ---------------------------------------------------------------------------
# 从 PDS3 .IMG 文件解析（备用）
# ---------------------------------------------------------------------------
def read_pds3_label(img_path: Path, max_bytes: int = 2_000_000) -> str:
    with open(img_path, "rb") as f:
        text = f.read(max_bytes).decode("latin-1", errors="ignore")
    end = re.search(r"^\s*END\s*$", text, re.M)
    return text[: end.end()] if end else text


def parse_img_fields(label_text: str) -> dict:
    txt = re.sub(r"/\*.*?\*/", " ", label_text, flags=re.S)

    def fstr(key):
        m = re.search(rf'{key}\s*=\s*"?([^"\s<]+)"?', txt, re.I)
        return m.group(1) if m else None

    def fnum(key):
        m = re.search(rf"{key}\s*=\s*([0-9E.+\-]+)", txt, re.I)
        return float(m.group(1)) if m else None

    m_img = re.search(r"OBJECT\s*=\s*IMAGE\b(.*?)END_OBJECT\s*=\s*IMAGE", txt, re.S | re.I)
    lines_block = m_img.group(1) if m_img else txt
    m_lines = re.search(r"\bLINES\s*=\s*(\d+)", lines_block, re.I)

    return {
        "SCLK":  fstr("SPACECRAFT_CLOCK_START_COUNT"),
        "DLINE": int(fnum("MRO:DELTA_LINE_TIMER_COUNT") or 0),
        "BIN":   int(fnum("MRO:BINNING") or 1),
        "TDI":   int(fnum("MRO:TDI") or 0),
        "LINES": int(m_lines.group(1)) if m_lines else 0,
    }


def parse_from_img_dir(img_dir: Path, product_id: str) -> list:
    """扫描 img_dir 下的 {product_id}_RED*_0.IMG 文件，解析并返回 entries 列表。"""
    channels = [f"RED{i}_0" for i in range(10)]
    entries = []
    for ch in channels:
        img_file = img_dir / product_id / f"{product_id}_{ch}.IMG"
        if not img_file.exists():
            print(f"  [跳过] 未找到 {img_file.name}")
            continue
        print(f"  解析 {img_file.name} ...")
        label = read_pds3_label(img_file)
        info = parse_img_fields(label)
        entries.append({"name": f"{product_id}_{ch}", **info})
    return entries


# ---------------------------------------------------------------------------
# 搜索 eo_setup 目录
# ---------------------------------------------------------------------------
def find_eo_setup_file(product_id: str, mars_root: Path, extra_hint: Path = None) -> Path:
    """按几个常见位置查找 eo_setup/{product_id}.txt。"""
    candidates = []
    if extra_hint:
        candidates.append(extra_hint / f"{product_id}.txt")
    candidates += [
        mars_root / "data" / "EO" / "eo_setup" / f"{product_id}.txt",
        mars_root / "data" / "EO" / f"{product_id}_setup.txt",
        Path("/media/wbl/Elements/paper_experiments/Mars/new/eo_setup") / f"{product_id}.txt",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# 生成输出文件
# ---------------------------------------------------------------------------
def pad_to_10_ccds(entries: list, product_id: str) -> list:
    """
    若 entries 少于 10 个 CCD，补充缺失的 CCD 条目。
    HiRISE 奇数 CCD (1,3,5,7,9) 和偶数 CCD (0,2,4,6,8) 各自共用相同的 SCLK 时间。
    用已有同奇偶的 CCD SCLK 推断缺失 CCD 的参数。
    """
    if len(entries) >= 10:
        return entries

    existing_names = {e["name"] for e in entries}
    # 按 CCD 编号建立映射
    ccd_map = {}
    for e in entries:
        import re as _re
        m = _re.search(r"RED(\d+)_", e["name"])
        if m:
            ccd_map[int(m.group(1))] = e

    result = list(entries)
    for i in range(10):
        name = f"{product_id}_RED{i}_0"
        if name in existing_names:
            continue
        # 用同奇偶的已知 CCD 作为参考
        parity = i % 2
        ref = None
        for j in sorted(ccd_map.keys()):
            if j % 2 == parity:
                ref = ccd_map[j]
                break
        if ref is None:
            # fallback: 用最近的 CCD
            ref = entries[-1]
        new_entry = dict(ref)
        new_entry["name"] = name
        result.append(new_entry)
        print(f"  [补全] 从 {ref['name']} 推断 {name} (SCLK={ref['SCLK']})")

    # 按 CCD 编号排序
    import re as _re
    result.sort(key=lambda e: int(_re.search(r"RED(\d+)_", e["name"]).group(1)))
    return result


def generate(product_id: str, entries: list, mars_root: Path, rows_override: int = None):
    # 补全到 10 个 CCD（代码中硬编码了 CCD_num=10）
    entries = pad_to_10_ccds(entries, product_id)

    eo_dir = mars_root / "data" / "EO"
    out_txt = eo_dir / f"{product_id}.txt"
    out_subdir = eo_dir / product_id

    # 创建 EO 输出子目录
    out_subdir.mkdir(parents=True, exist_ok=True)
    print(f"  已创建目录：{out_subdir}")

    # 相对路径（相对于 bin/ 目录）
    rel_out_path = f"../data/EO/{product_id}/"

    lines_out = [str(len(entries)), rel_out_path]
    for e in entries:
        line_count = rows_override if rows_override is not None else e["LINES"]
        lines_out.append(f"{e['name']} {e['SCLK']} {e['DLINE']} {e['BIN']} {e['TDI']} {line_count}")

    out_txt.write_text("\n".join(lines_out) + "\n")
    print(f"  已生成：{out_txt}")
    print(f"  内容预览：")
    for l in lines_out:
        print(f"    {l}")


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="生成 Mars EO 输入配置文件")
    parser.add_argument("product_id", help="HiRISE 序列号，如 ESP_069731_2055")
    parser.add_argument("-r", "--root",      default=None, help="Mars 工程根目录")
    parser.add_argument("-s", "--eo-setup",  default=None, help="eo_setup 目录路径")
    parser.add_argument("-i", "--img-dir",   default=None, help=".IMG 文件所在目录（备用）")
    parser.add_argument("--rows",            type=int, default=None, help="覆盖 LINE 行数")
    args = parser.parse_args()

    product_id = args.product_id

    # 确定工程根目录
    script_dir = Path(__file__).resolve().parent
    mars_root  = Path(args.root).resolve() if args.root else script_dir.parent
    print(f"[info] 工程根目录：{mars_root}")

    # 1) 优先从 eo_setup 文件读取
    eo_setup_hint = Path(args.eo_setup) if args.eo_setup else None
    setup_file = find_eo_setup_file(product_id, mars_root, eo_setup_hint)

    if setup_file:
        print(f"[info] 从 eo_setup 文件读取：{setup_file}")
        _, entries = parse_eo_setup_file(setup_file)
    elif args.img_dir:
        # 2) 备用：从 .IMG 文件提取
        img_dir = Path(args.img_dir).resolve()
        print(f"[info] 从 .IMG 目录提取：{img_dir}")
        entries = parse_from_img_dir(img_dir, product_id)
        if not entries:
            print(f"[error] 未找到任何 .IMG 文件，退出。")
            sys.exit(1)
    else:
        print(f"[error] 未找到 eo_setup 文件，也未指定 --img-dir。")
        print(f"        请用 -s 指定 eo_setup 目录，或用 -i 指定 .IMG 目录。")
        sys.exit(1)

    generate(product_id, entries, mars_root, args.rows)
    print("[done]")


if __name__ == "__main__":
    main()
