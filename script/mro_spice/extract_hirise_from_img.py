#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 PDS3 .IMG 读取标签头，并安全解析关键字段。
修复点：仅当行匹配 ^\s*END\s*$ 时才认为标签结束，避免被 'EXTENDED' 等误触发。
"""

import re
import sys
from pathlib import Path
from typing import Optional
import requests


END_LINE_RE = re.compile(r'^\s*END\s*$', re.IGNORECASE)

def _detect_label_size_prefix(text_chunk: str) -> Optional[int]:
    """
    尝试在已读的前若干文本中解析标签长度：
    - 情况A：RECORD_TYPE = FIXED_LENGTH 且 RECORD_BYTES + LABEL_RECORDS（单位=记录数）
    - 情况B：RECORD_TYPE = UNDEFINED 且 LABEL_RECORDS 以 <BYTES> 给出
    返回需要读取的总字节数（仅标签部分）。若无法判定返回 None。
    """
    # 去掉注释再匹配
    cleaned = re.sub(r'/\*.*?\*/', ' ', text_chunk, flags=re.S)
    rec_type = re.search(r'RECORD_TYPE\s*=\s*([A-Z_]+)', cleaned)
    if not rec_type:
        return None
    rec_type = rec_type.group(1).upper()

    if rec_type == 'FIXED_LENGTH':
        m1 = re.search(r'RECORD_BYTES\s*=\s*(\d+)', cleaned, re.I)
        m2 = re.search(r'LABEL_RECORDS\s*=\s*(\d+)\b(?!\s*<BYTES>)', cleaned, re.I)
        if m1 and m2:
            record_bytes = int(m1.group(1))
            label_records = int(m2.group(1))
            return record_bytes * label_records

    # UNDEFINED 或其他：LABEL_RECORDS 可能直接给 <BYTES>
    m3 = re.search(r'LABEL_RECORDS\s*=\s*(\d+)\s*<BYTES>', cleaned, re.I)
    if m3:
        return int(m3.group(1))

    return None


def read_pds3_label(path_or_url: str, head_probe_bytes: int = 4096) -> str:
    """
    读取 PDS3 标签文本。
    优先依据 LABEL_RECORDS/RECORD_BYTES 计算字节数；否则逐行读到匹配 ^END$ 为止。
    """
    if path_or_url.startswith(('http://', 'https://')):
        # 先取一小段探测标签尺寸
        with requests.get(path_or_url, stream=True, timeout=60) as r:
            r.raise_for_status()
            it = r.iter_content(1024)
            buf = bytearray()
            while len(buf) < head_probe_bytes:
                try:
                    chunk = next(it)
                except StopIteration:
                    break
                buf += chunk
            # 尝试解析标签总长度
            label_size = _detect_label_size_prefix(buf.decode('latin-1', errors='ignore'))
            if label_size is not None:
                # 直接按字节读取完整标签
                need = label_size - len(buf)
                data = bytes(buf)
                while need > 0:
                    try:
                        chunk = next(it)
                    except StopIteration:
                        break
                    data += chunk
                    need -= len(chunk)
                return data[:label_size].decode('latin-1', errors='ignore')
            else:
                # 回退：逐行拼接，直到遇到独立 END 行
                text = []
                # 把已读的 buf 先按行处理
                for line in buf.decode('latin-1', errors='ignore').splitlines():
                    text.append(line)
                    if END_LINE_RE.match(line):
                        return '\n'.join(text)
                # 继续流式读剩余部分
                for chunk in it:
                    s = chunk.decode('latin-1', errors='ignore')
                    for line in s.splitlines():
                        text.append(line)
                        if END_LINE_RE.match(line):
                            return '\n'.join(text)
                # 若未遇到 END，也返回已收集的文本
                return '\n'.join(text)
    else:
        p = Path(path_or_url)
        # 先读一小段探测
        with p.open('rb') as f:
            buf = f.read(head_probe_bytes)
            label_size = _detect_label_size_prefix(buf.decode('latin-1', errors='ignore'))
            if label_size is not None:
                f.seek(0)
                data = f.read(label_size)
                return data.decode('latin-1', errors='ignore')
            # 回退：逐行到 END
            f.seek(0)
            lines = []
            for raw in f:
                line = raw.decode('latin-1', errors='ignore').rstrip('\r\n')
                lines.append(line)
                if END_LINE_RE.match(line):
                    break
            return '\n'.join(lines)


def parse_hirise_fields(label_text: str):
    """提取你关心的字段：SCLK, DLINE, BIN, TDI, LINES"""
    # 安全做法：忽略块注释
    txt = re.sub(r'/\*.*?\*/', ' ', label_text, flags=re.S)

    def fstr(key):
        m = re.search(rf'{key}\s*=\s*"([^"]+)"', txt, re.I)
        if m: return m.group(1)
        m = re.search(rf'{key}\s*=\s*([^\s<]+)', txt, re.I)
        return m.group(1) if m else None

    def fnum(key):
        m = re.search(rf'{key}\s*=\s*([0-9E\.\+\-]+)', txt, re.I)
        return float(m.group(1)) if m else None

    info = {}
    info['SCLK_START'] = fstr('SPACECRAFT_CLOCK_START_COUNT')
    info['SCLK_STOP']  = fstr('SPACECRAFT_CLOCK_STOP_COUNT')
    # DLINE：优先秒值，其次某些标注用计数（DELTA_LINE_TIMER_COUNT）
    dline = fnum('DELTA_LINE_TIMER_COUNT')
    if dline is None:
        # 有些 EDR 用 MRO:LINE_EXPOSURE_DURATION 表达（微秒），也可做近似
        lexp_us = fnum(r'MRO:LINE_EXPOSURE_DURATION')
        if lexp_us is not None: dline = lexp_us / 1e6
    info['DLINE'] = dline

    info['BIN']   = fnum(r'MRO:BINNING') or fnum('BINNING_MODE')
    info['TDI']   = fnum(r'MRO:TDI') or fnum('TDI')

    # LINES 取 IMAGE 对象里的值
    m_img = re.search(r'OBJECT\s*=\s*IMAGE\b(.*?)END_OBJECT\s*=\s*IMAGE', txt, re.S | re.I)
    if m_img:
        mm = re.search(r'LINES\s*=\s*(\d+)', m_img.group(1), re.I)
        info['LINES'] = int(mm.group(1)) if mm else None
    else:
        # 没有明显的块时，退化为全局第一次出现的 LINES
        mm = re.search(r'\bLINES\s*=\s*(\d+)', txt, re.I)
        info['LINES'] = int(mm.group(1)) if mm else None

    return info


def main():
    if len(sys.argv) < 2:
        print("Usage: python pds3_read_label_safe.py <file.IMG or URL>")
        sys.exit(1)
    src = sys.argv[1]
    label = read_pds3_label(src)
    info = parse_hirise_fields(label)
    print("[source]", src)
    for k, v in info.items():
        print(f"{k:12s} = {v}")


if __name__ == "__main__":
    main()
