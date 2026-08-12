#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HiRISE 编号 → 自动生成用于位姿计算的 MRO meta-kernel (.tm)
仅包含必要的 SPICE 内核（航天器轨道 SPK、姿态 CK、FK、IK、IAK、LSK、SCLK、PCK）

示例：
  # 仅生成在线 URL 版清单
  python hirise2mk_pose.py --pid ESP_069731_2055 --out mro_pose_ESP_069731_2055.tm

  # 下载所有内核到本地并生成可 furnsh() 的 .tm
  python hirise2mk_pose.py --pid PSP_010502_2090 --download --dest ./mro_kernels --out mro_pose_local.tm
"""

import argparse, os, re, sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dateutil import parser as dtparser

# ---------- 常量 ----------
# BASE = "https://naif.jpl.nasa.gov/pub/naif/MRO/kernels"
BASE = "https://naif.jpl.nasa.gov/pub/naif/pds/data/mro-m-spice-6-v1.0/mrosp_1000/data/"
PDS_ATLAS_API = "https://pds-imaging.jpl.nasa.gov/tools/atlas/api/search"
USGS_PYGEOAPI_ITEMS = (
    "https://astrogeology.usgs.gov/pygeoapi/collections/mars/hirise-observation-footprints-equatorial/items"
)
HRISE_PDS_BASE = "https://hirise.lpl.arizona.edu/PDS"
UAHIRISE_PAGE = "https://www.uahirise.org/{pid}"
HIPO_BASE = "https://www.uahirise.org/hipod/{pid}"
SUFFIXES = ["", "_RED", "_IRB", "_BG"]

# ---------- HTTP 会话 ----------
def build_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=3, connect=3, read=3,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    ad = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    s.mount("http://", ad); s.mount("https://", ad)
    s.headers.update({
        "User-Agent": "hirise2mk-pose/1.0 (+python; requests)",
        "Accept": "application/json, text/html;q=0.9, */*;q=0.1"
    })
    # 本地代理 MITM 时常缺 CA；INSECURE_SSL=1 时跳过校验
    if os.environ.get("INSECURE_SSL", "0") == "1":
        s.verify = False
        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except Exception:
            pass
    return s

S = build_session()

def http_get_json(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    r = S.get(url, params=params, timeout=(8, 12))
    r.raise_for_status()
    return r.json()

def http_get_text(url: str) -> str:
    r = S.get(url, timeout=(8, 20))
    r.raise_for_status()
    r.encoding = r.apparent_encoding or "utf-8"
    return r.text

# ---------- 工具 ----------
def y_doy_to_dt(year: int, doy: int) -> datetime:
    return datetime(year,1,1,tzinfo=timezone.utc)+timedelta(days=doy-1)

# ---------- 观测时间解析 ----------
@dataclass
class ObsInfo:
    product_id: str
    start_time_utc: datetime
    label_url: Optional[str]

def best_variants(pid: str) -> List[str]:
    pid = re.sub(r"\s+","",pid).strip()
    if re.search(r"_(RED|IRB|BG)$",pid,re.IGNORECASE):
        return [pid]
    return [pid]+[pid+s for s in SUFFIXES[1:]]

def extract_start_time_from_lbl(lbl_url: str) -> Optional[ObsInfo]:
    try: txt=http_get_text(lbl_url)
    except Exception: return None
    m=re.search(r"START_TIME\s*=\s*([0-9T:\.\-Z\+]+)",txt)
    if not m: return None
    try:
        t=dtparser.parse(m.group(1))
        if t.tzinfo is None: t=t.replace(tzinfo=timezone.utc)
        return ObsInfo("",t.astimezone(timezone.utc),lbl_url)
    except Exception:
        return None

def query_start_time_from_pds_atlas(pid:str)->Optional[ObsInfo]:
    try: js=http_get_json(PDS_ATLAS_API,{"product_id":pid,"instrument":"HIRISE"})
    except Exception: return None
    docs=js.get("docs") or js.get("data") or []
    if not docs and isinstance(js,dict): docs=[js]
    for d in docs:
        st=d.get("start_time") or d.get("START_TIME")
        lbl=d.get("labelurl") or d.get("label_url")
        if st:
            try:
                t=dtparser.parse(st)
                if t.tzinfo is None: t=t.replace(tzinfo=timezone.utc)
                return ObsInfo(pid,t.astimezone(timezone.utc),lbl)
            except: pass
    for d in docs:
        lbl=d.get("labelurl") or d.get("label_url")
        if lbl:
            info=extract_start_time_from_lbl(lbl)
            if info: info.product_id=pid; return info
    return None

def query_start_time_from_pygeoapi(pid:str)->Optional[ObsInfo]:
    for flt in [f"productid = '{pid}'", f"productid LIKE '{pid}%'"]:
        try:
            js=http_get_json(USGS_PYGEOAPI_ITEMS,{
                "filter-lang":"cql2-text","filter":flt,"limit":1,"f":"json"})
        except Exception: continue
        feats=js.get("features") or []
        if not feats: continue
        props=feats[0].get("properties",{}) or {}
        st=props.get("utcstart") or props.get("UTCSTART")
        lbl=props.get("labelurl") or props.get("LABELURL")
        pid2=props.get("productid") or pid
        if st:
            try:
                t=dtparser.parse(st)
                if t.tzinfo is None: t=t.replace(tzinfo=timezone.utc)
                return ObsInfo(pid2,t.astimezone(timezone.utc),lbl)
            except: pass
        if lbl:
            info=extract_start_time_from_lbl(lbl)
            if info: info.product_id=pid2; return info
    return None

def orbit_from_pid(pid:str)->Optional[int]:
    m=re.search(r"_(\d{6})_",pid)
    return int(m.group(1)) if m else None

def build_lbl_urls(pid:str)->List[str]:
    phase=pid.split("_",1)[0].upper()
    if phase not in ("PSP","ESP","TRA"): phase="ESP"
    orb=orbit_from_pid(pid)
    if orb is None: return []
    lo=(orb//100)*100; hi=lo+99
    bucket=f"ORB_{lo:06d}_{hi:06d}"
    base=f"{HRISE_PDS_BASE}/RDR/{phase}/{bucket}/{pid}/{pid}"
    return [f"{base}{s}.LBL" for s in SUFFIXES[1:]]+[f"{base}.LBL"]

def try_lbl_paths(pid:str)->Optional[ObsInfo]:
    for url in build_lbl_urls(pid):
        info=extract_start_time_from_lbl(url)
        if info: info.product_id=pid; return info
    return None

def parse_date_from_uahirise(pid:str)->Optional[datetime]:
    for url in [UAHIRISE_PAGE.format(pid=pid), HIPO_BASE.format(pid=pid)]:
        try: html=http_get_text(url)
        except Exception: continue
        m=re.search(r"(\d{1,2}\s+[A-Za-z]+\s+\d{4})",html)
        if m:
            dt=dtparser.parse(m.group(1),dayfirst=True,
                              default=datetime(2000,1,1,12,0,0,tzinfo=timezone.utc))
            return dt
    return None

def resolve_obs_info(pid_user:str)->ObsInfo:
    last_err=None
    for pid in best_variants(pid_user):
        try:
            x=query_start_time_from_pds_atlas(pid)
            if x: return x
        except Exception as e: last_err=e
        try:
            x=query_start_time_from_pygeoapi(pid)
            if x: return x
        except Exception as e: last_err=e
        x=try_lbl_paths(pid)
        if x: return x
        dt=parse_date_from_uahirise(pid)
        if dt: return ObsInfo(pid,dt,None)
    raise RuntimeError(f"未能解析 {pid_user} 的 START_TIME；最后错误: {last_err}")

# ---------- NAIF 目录 ----------
def list_dir(sub:str)->List[str]:
    html=http_get_text(f"{BASE}/{sub}/")
    return [m for m in re.findall(r'href="([^"/]+)"',html) if not m.endswith("/")]

# ---------- 文件名解析 ----------
def parse_range_any(name:str)->Optional[Tuple[datetime,datetime]]:
    m=re.search(r"(\d{4})_(\d{3})_(\d{4})_(\d{3})",name)
    if m:
        y1,d1,y2,d2=map(int,m.groups())
        t1=y_doy_to_dt(y1,d1); t2=y_doy_to_dt(y2,d2)+timedelta(hours=23,minutes=59)
        return t1,t2
    m=re.search(r"(\d{6})_(\d{6})",name)
    if m:
        yymmdd1,yymmdd2=m.groups()
        yy1=int(yymmdd1[:2]); mm1=int(yymmdd1[2:4]); dd1=int(yymmdd1[4:6])
        yy2=int(yymmdd2[:2]); mm2=int(yymmdd2[2:4]); dd2=int(yymmdd2[4:6])
        year1=2000+yy1; year2=2000+yy2
        t1=datetime(year1,mm1,dd1,tzinfo=timezone.utc)
        t2=datetime(year2,mm2,dd2,23,59,0,tzinfo=timezone.utc)
        return t1,t2
    return None

def pick_covering(files:List[str],when:datetime,keep:int=1)->List[str]:
    hits=[]
    for f in files:
        r=parse_range_any(f)
        if r and r[0]<=when<=r[1]: hits.append((f,r))
    if not hits:
        dated=[(f,parse_range_any(f)) for f in files if parse_range_any(f)]
        if not dated: return []
        dated.sort(key=lambda x:abs((x[1][0]-when).days))
        return [dated[0][0]]
    # 优先重建 CK（无尾缀 p），避免 predictive CK 文件名覆盖但首日实际无数据
    def _ck_rank(name: str):
        n = name.lower()
        predictive = 1 if n.endswith("p.bc") else 0
        return (predictive, n)
    hits.sort(key=lambda x: _ck_rank(x[0]))
    return [f for f, _ in hits[:keep]]

def pick_latest(files:List[str],pattern:str)->Optional[str]:
    lst=[f for f in files if re.search(pattern,f,re.IGNORECASE)]
    return sorted(lst)[-1] if lst else None

def parse_lbl_time_range(lbl_url: str) -> Optional[Tuple[datetime, datetime]]:
    """从 SPK .lbl 文件中提取时间范围"""
    # print("lbl_url",lbl_url)
    try:
        txt = http_get_text(lbl_url)
        # print("txt",txt)
        start = re.search(r"START_TIME\s*=\s*[\"']?([0-9T:\.\-Z\+]+)", txt)
        stop = re.search(r"STOP_TIME\s*=\s*[\"']?([0-9T:\.\-Z\+]+)", txt)
        if not (start and stop):
            return None
        t1 = dtparser.parse(start.group(1))
        t2 = dtparser.parse(stop.group(1))
        if t1.tzinfo is None:
            t1 = t1.replace(tzinfo=timezone.utc)
        if t2.tzinfo is None:
            t2 = t2.replace(tzinfo=timezone.utc)
        return t1.astimezone(timezone.utc), t2.astimezone(timezone.utc)
    except Exception:
        return None

def pick_spacecraft_orbit_spk(spk_files: List[str], when: datetime) -> Optional[str]:
    """选择合适的航天器轨道 SPK 文件"""
    orbit_like = [f for f in spk_files if re.match(r"mro_(psp|orbit)", f, re.IGNORECASE)]
    if not orbit_like:
        return None
    
    # 先尝试从 .lbl 文件获取精确时间范围
    hits = []
    for f in orbit_like:
        if not f.lower().endswith('.bsp'):
            continue
        lbl_file = f[:-4] + '.lbl'
        try:
            lbl_url = f"{BASE}/spk/{lbl_file}"
            time_range = parse_lbl_time_range(lbl_url)
            if time_range and time_range[0] <= when <= time_range[1]:
                hits.append((f, time_range))
        except Exception:
            continue
    
    if hits:
        # 如果找到精确匹配的文件，返回最新的那个
        hits.sort(key=lambda x: x[1][1])  # 按结束时间排序
        return hits[-1][0]
    
    # 如果没有从 .lbl 找到，回退到文件名解析方法
    for f in orbit_like:
        r = parse_range_any(f)
        if r and r[0] <= when <= r[1]:
            return f
            
    # 如果还是没找到，返回最接近的或最新的文件
    dated = [(f, parse_range_any(f)) for f in orbit_like if parse_range_any(f)]
    if dated:
        dated.sort(key=lambda x: abs((x[1][0] - when).days))
        return dated[0][0]
    
    return sorted(orbit_like)[-1]

# ---------- 生成 .tm ----------
def make_tm(paths:List[str],path_values:List[str],symbol="A")->str:
    lines = [
        "KPL/MK\n",
        "\\begindata\n\n",
        f"PATH_VALUES  = ( {', '.join([repr(p) for p in path_values])}\n)\n\n",
        f"PATH_SYMBOLS = ( '{symbol}'\n)\n\n",
        "KERNELS_TO_LOAD = (\n"
    ]
    for p in paths: lines.append(f"'{p}',\n")
    lines.append(")\n\\begintext\n")
    return "".join(lines)

def download_to(url_path:str,dest_root:Path)->Path:
    if url_path.startswith("$A/"):
        rel=url_path[3:]
        url=f"{BASE}/{rel}"
        local=dest_root/rel
    else:
        url=url_path
        rel=url.split("/MRO/kernels/")[-1]
        local=dest_root/rel
    local.parent.mkdir(parents=True,exist_ok=True)

    name = local.name.lower()
    # 航天器轨道 SPK / 姿态 CK：残缺文件常 >1MB 但仍 DAF 损坏（DAFBEGGTEND）
    if local.suffix.lower() == ".bc" or (
        local.suffix.lower() == ".bsp" and name.startswith("mro_") and "mar" not in name[:6]
    ):
        min_bytes = 40_000_000
    elif local.suffix.lower() in (".bsp", ".bc"):
        min_bytes = 1_000_000
    else:
        min_bytes = 100

    expected = None
    try:
        hr = S.head(url, timeout=(15, 60), allow_redirects=True)
        if hr.ok and hr.headers.get("Content-Length"):
            expected = int(hr.headers["Content-Length"])
    except Exception as e:
        print(f"[warn] HEAD 失败，跳过远端大小预检: {e}")

    def too_small(path: Path) -> bool:
        sz = path.stat().st_size
        if expected is not None and sz < expected:
            return True
        return sz < min_bytes

    if local.is_file() and not too_small(local):
        print(f"[skip] 已存在且完整: {local} ({local.stat().st_size} bytes"
              f"{'' if expected is None else f', expect {expected}'})")
        return local
    if local.is_file():
        print(f"[warn] 本地不完整，重新下载: {local} ({local.stat().st_size} bytes"
              f"{'' if expected is None else f', expect {expected}'})")
        local.unlink()

    print(f"[download] {url} -> {local}")
    tmp = local.with_suffix(local.suffix + ".part")
    if tmp.is_file():
        tmp.unlink()
    with S.get(url, stream=True, timeout=(30, 1200)) as r:
        r.raise_for_status()
        cl = r.headers.get("Content-Length")
        if cl is not None:
            expected = int(cl)
        written = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
                    written += len(chunk)
    if expected is not None and written != expected:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"下载字节数不匹配: got {written}, Content-Length={expected}: {local}"
        )
    if written < min_bytes:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"下载后文件仍过小 ({written} bytes < {min_bytes}): {local}"
        )
    tmp.replace(local)
    print(f"[ok] {local} ({local.stat().st_size} bytes)")
    return local

# ---------- 主 ----------
def main():
    ap=argparse.ArgumentParser(description="HiRISE编号 -> 位姿计算用 MRO meta-kernel (.tm)")
    ap.add_argument("--pid",required=True,help="如 ESP_069731_2055 或 PSP_010502_2090[_RED]")
    ap.add_argument("--out",required=True,help="输出 .tm 路径")
    ap.add_argument("--download",action="store_true",help="下载选中内核到本地")
    ap.add_argument("--dest",default="./mro_kernels",help="--download 时存储根目录")
    ap.add_argument("--path-symbol",default="A",help="PATH_SYMBOL 名称（默认 A）")
    args=ap.parse_args()

    obs=resolve_obs_info(args.pid)
    when=obs.start_time_utc
    print(f"[info] 产品: {obs.product_id}")
    print(f"[info] 观测时间: {when.isoformat()}")
    if obs.label_url: print(f"[info] 标签: {obs.label_url}")

    # 在线列目录
    lsk=list_dir("lsk"); sclk=list_dir("sclk"); pck=list_dir("pck")
    fk=list_dir("fk"); ik=list_dir("ik")
    ck=list_dir("ck"); spk=list_dir("spk")
    try: iak=list_dir("iak")
    except Exception: iak=[]

    chosen=[]
    l=pick_latest(lsk,r"naif\d+\.tls");        
    if l: 
        chosen.append(f"$A/lsk/{l}")
    s=pick_latest(sclk,r"MRO_SCLKSCET.*\.tsc");
    if s: 
        chosen.append(f"$A/sclk/{s}")
    p=pick_latest(pck,r"pck\d+\.tpc");         
    if p: 
        chosen.append(f"$A/pck/{p}")
    f=pick_latest(fk,r"mro_v\d+\.tf");         
    if f: 
        chosen.append(f"$A/fk/{f}")
    i=pick_latest(ik,r"mro_hirise_v\d+\.ti");  
    if i: 
        chosen.append(f"$A/ik/{i}")
    ia=pick_latest(iak,r"hiriseAddendum\d+\.ti"); 
    if ia: 
        chosen.append(f"$A/iak/{ia}")

    # CK
    ck_hit = pick_covering(
        [f for f in ck if f.lower().startswith("mro_sc_") and f.lower().endswith(".bc")],
        when,
        keep=1
    )
    if ck_hit: chosen.append(f"$A/ck/{ck_hit[0]}")
    else: print("[warn] 未找到覆盖观测时刻的 CK")

    # 航天器轨道 SPK
    print("[info] 选择覆盖观测时刻的航天器轨道 SPK...")
    spk_dyn=pick_spacecraft_orbit_spk(spk,when)
    if not spk_dyn:
        raise RuntimeError("未找到航天器轨道 SPK（mro_psp* 或 mro_orbit_*）")
    chosen.append(f"$A/spk/{spk_dyn}")
    
    # 行星轨道 SPK（优先用本地验证可用的 mar063/mar080；
    # mar097 等新文件偶发损坏会触发 SPICE(DAFBEGGTEND)）
    print("[info] 添加行星轨道 SPK...")
    planet_spk = (
        pick_latest(spk, r"mar080\.bsp")
        or pick_latest(spk, r"mar063\.bsp")
        or pick_latest(spk, r"mar\d+\.bsp")
    )
    if planet_spk:
        if planet_spk.startswith("mar097"):
            print("[warn] 选用 mar097.bsp；若出现 DAFBEGGTEND，请改用 mar063/mar080")
        chosen.append(f"$A/spk/{planet_spk}")
    else:
        print("[warn] 未找到行星轨道 SPK (mar*.bsp)")

    # 生成 .tm
    if args.download:
        dest = Path(args.dest).resolve()
        local = []
        for p in chosen:
            # 解析本地路径
            if p.startswith("$A/"):
                rel = p[3:]
                local_path = dest / rel
            else:
                rel = p.split("/MRO/kernels/")[-1]
                local_path = dest / rel
            # 判断文件是否已存在
            if local_path.exists():
                print(f"[info] 已存在: {local_path}")
            else:
                local_path = download_to(p, dest)
            rel_path = local_path.relative_to(dest).as_posix()
            local.append(f"$A/{rel_path}")
        tm_text = make_tm(local, [dest.as_posix()], symbol=args.path_symbol)
    else:
        tm_text = make_tm(chosen, [BASE], symbol=args.path_symbol)

    Path(args.out).write_text(tm_text,encoding="utf-8")
    print(f"[ok] 输出 TM: {args.out}")
    if args.download: print(f"[ok] 内核已下载到: {Path(args.dest).resolve()}")

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt:
        print("\n[warn] 用户中断"); sys.exit(130)
    except Exception as e:
        print(f"[error] {e}"); sys.exit(1)
