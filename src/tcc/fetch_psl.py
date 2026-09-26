"""PSL + Open-Meteo 取数主线（绕开 CDS 队列）。

CDS 在 2026 年长期过载：单账号有排队配额，实测一晚只落 2 个文件。
这条主线改用两个免认证、免排队的公共服务，全量约 1 GB / 30—50 分钟：

  * NOAA PSL THREDDS NCSS —— 服务端按 区域 + 层次 + 时间 裁剪，返回即用。
    实测 Z500 一个暖季 512 KB / 2.3 秒；不裁层（vertCoord）要 8.7 MB。
  * Open-Meteo Archive API —— 目标变量 Tmax。实测 325 个点一次拿一整年：
    0.8 MB / 3.2 秒，URL 3.7 KB（nginx 上限 8 KB）。

**产品口径与纯 ERA5 方案不同，写数据字典时必须注明**：
    环流  NCEP/NCAR R1 日平均 2.5°（不是 ERA5 0.25°；副高指数本来就用 2.5° 网格算）
    SST   NOAA OISST v2.1 0.25° 日（1981-09 起 → 1981 年缺，脚本会自动跳过并提示）
    Tmax  ERA5-Land 日最高（与 CDS 路线同一产品，文件格式完全同构、可直接接上）

用法：
    uv run python src/tcc/fetch_psl.py --dry-run
    uv run python src/tcc/fetch_psl.py                      # 全部
    uv run python src/tcc/fetch_psl.py --group circ --years 1981-1985
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("TCC_DATA_DIR") or ROOT / "data") / "raw"

YEARS = list(range(1981, 2026))
PSL = "https://psl.noaa.gov/thredds/ncss/grid/Datasets"

POLITE_SLEEP = 1.0     # PSL 是学术公共服务，慢慢来
RETRIES = 4
MIN_BYTES = 2048       # 小于这个体积一定是错误页而不是数据

# PSL 请求项。area = (北, 西, 南, 东)；时间为整个暖季，一个请求一年。
PSL_ITEMS = [
    dict(group="circ", dataset="ncep.reanalysis.dailyavgs/pressure",
         nc="hgt.{year}.nc", var="hgt", level="500",
         area=(60, 60, 0, 160), res=2.5),
    dict(group="circ", dataset="ncep.reanalysis.dailyavgs/pressure",
         nc="uwnd.{year}.nc", var="uwnd", level="850",
         area=(60, 60, 0, 160), res=2.5),
    dict(group="circ", dataset="ncep.reanalysis.dailyavgs/pressure",
         nc="vwnd.{year}.nc", var="vwnd", level="850",
         area=(60, 60, 0, 160), res=2.5),
    dict(group="mslp", dataset="ncep.reanalysis.dailyavgs/surface",
         nc="slp.{year}.nc", var="slp", level=None,
         area=(55, 95, 0, 160), res=2.5),
    dict(group="sst", dataset="noaa.oisst.v2.highres",
         nc="sst.day.mean.{year}.nc", var="sst", level=None,
         area=(40, 100, 0, 160), res=0.25),
]

# 目标变量走 Open-Meteo，输出与 openmeteo_fetch.py 完全同构（t2m / 13×25 / 年月文件名）
TMAX = dict(group="tmax", area=(34, 110, 28, 122), spacing=0.5)

# 体积估算用的实测字节/值（PSL 返回 netcdf3 float32 = 4；Open-Meteo JSON ≈ 20）
BYTES_PER_VALUE = {"psl": 4, "openmeteo": 20}


# ---------------------------------------------------------------- 工具

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def axis(area, res):
    """按区域和分辨率生成经纬度轴（含端点）。"""
    north, west, south, east = area
    lats = np.round(np.arange(south, north + res / 2, res), 4)
    lons = np.round(np.arange(west, east + res / 2, res), 4)
    return lats, lons


def download(url, dest, sleep=POLITE_SLEEP):
    """下载到 dest。返回字节数；返回 None 表示该年数据不存在（跳过，不算错）。"""
    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=300) as resp:
                data = resp.read()
            if len(data) < MIN_BYTES:
                raise ValueError(f"响应只有 {len(data)} 字节，不是数据")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            time.sleep(sleep)
            return len(data)
        except urllib.error.HTTPError as exc:
            # 400/404 = 该年份该数据集确实没有（如 OISST 1981 年 5—8 月），重试无意义
            if exc.code in (400, 404):
                time.sleep(sleep)
                return None
            if attempt == RETRIES:
                raise
            time.sleep(2 ** attempt)
        except Exception:
            if attempt == RETRIES:
                raise
            time.sleep(2 ** attempt)
    return None


def is_valid(path, var, min_kb=8):
    """断点续传判据：文件存在、够大、打得开、含目标变量。"""
    if not path.exists() or path.stat().st_size < min_kb * 1024:
        return False
    try:
        with xr.open_dataset(path) as ds:
            return var in ds.data_vars and ds[var].size > 0
    except Exception:
        return False


def parse_years(text):
    if not text:
        return YEARS
    if "-" in text:
        start, end = text.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(text)]


# ---------------------------------------------------------------- PSL

def psl_url(item, year):
    north, west, south, east = item["area"]
    # time_end 必须给 T00:00:00Z。给 T23:59:59Z 会多带出 9 月 1 日（实测 124 天 vs 123 天）
    url = (f"{PSL}/{item['dataset']}/{item['nc'].format(year=year)}"
           f"?var={item['var']}"
           f"&north={north}&west={west}&south={south}&east={east}"
           f"&time_start={year}-05-01T00:00:00Z&time_end={year}-08-31T00:00:00Z"
           f"&accept=netcdf")
    if item.get("level"):
        url += f"&vertCoord={item['level']}"      # 只取一层，体积 ÷17
    return url


def tidy_psl(dest, item):
    """规整 PSL 返回的 netcdf3：压掉长度为 1 的 level 维、补属性、顺带压缩。

    NCSS 返回的形状是 (time, level, lat, lon)，level 恒为 1。统一成 (time, lat, lon)
    才能和 tmax 的文件放在一起用，不至于下游 glob 出两种形状。
    """
    with xr.open_dataset(dest) as ds:
        ds = ds.squeeze(drop=True)
        # 整体替换属性：PSL 的文件里同时有 history 和 History，
        # netCDF 属性名大小写不敏感，用 update() 会报 "String match to name in use"
        ds.attrs = {
            "source": f"NOAA PSL THREDDS NCSS: {item['dataset']}",
            "underlying_data": f"NCEP/NCAR Reanalysis 1 daily average, {item['var']}",
            "level_hPa": item.get("level") or "surface",
            "area_nwse": str(item["area"]),
            "reference": "https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html",
            "history": f"fetched {pd.Timestamp.now():%Y-%m-%d %H:%M:%S}",
        }
        tmp = dest.with_suffix(".tmp.nc")
        ds.to_netcdf(tmp, encoding={item["var"]: {"zlib": True, "complevel": 4, "shuffle": True}})
    tmp.replace(dest)


def fetch_psl(item, years):
    out_dir = DATA_DIR / item["group"]
    tag = f"{item['var']}{item.get('level') or ''}"
    done = skipped = missing = 0
    for year in years:
        dest = out_dir / f"{tag}_{year}.nc"
        if is_valid(dest, item["var"]):
            skipped += 1
            continue
        nbytes = download(psl_url(item, year), dest)
        if nbytes is None:
            missing += 1
            log(f"    {item['group']}/{tag} {year}: 该年无数据，跳过")
        else:
            tidy_psl(dest, item)
            done += 1
    log(f"  {item['group']}/{tag}: 新下 {done}  已存在 {skipped}  无数据 {missing}")


# ---------------------------------------------------------------- Open-Meteo

def fetch_tmax(years):
    out_dir = DATA_DIR / TMAX["group"]
    out_dir.mkdir(parents=True, exist_ok=True)
    lats, lons = axis(TMAX["area"], TMAX["spacing"])
    pairs = [(float(a), float(b)) for a in lats for b in lons]

    done = skipped = 0
    for year in years:
        targets = {m: out_dir / f"tmax_{year}_{m:02d}.nc" for m in (5, 6, 7, 8)}
        if all(is_valid(t, "t2m", min_kb=4) for t in targets.values()):
            skipped += 1
            continue

        url = ("https://archive-api.open-meteo.com/v1/archive"
               f"?latitude={','.join(str(a) for a, _ in pairs)}"
               f"&longitude={','.join(str(b) for _, b in pairs)}"
               f"&start_date={year}-05-01&end_date={year}-08-31"
               "&daily=temperature_2m_max&timezone=Asia%2FShanghai&models=era5_land")
        raw = download(url, out_dir / f".tmax_{year}.json", sleep=0.5)
        if raw is None:
            log(f"    tmax {year}: 无数据，跳过")
            continue

        payload = json.loads((out_dir / f".tmax_{year}.json").read_text(encoding="utf-8"))
        payload = payload if isinstance(payload, list) else [payload]
        dates = pd.to_datetime(payload[0]["daily"]["time"])
        cube = np.array([p["daily"]["temperature_2m_max"] for p in payload],
                        dtype="float32").reshape(len(lats), len(lons), len(dates))
        cube += 273.15      # Open-Meteo 返回摄氏度；既有 tmax 文件按 ERA5 惯例存开尔文

        for month, target in targets.items():
            mask = dates.month == month
            ds = xr.Dataset(
                {"t2m": (("time", "latitude", "longitude"),
                         np.transpose(cube[:, :, mask], (2, 0, 1)))},
                coords={"time": dates[mask], "latitude": lats, "longitude": lons},
                attrs={
                    "source": "Open-Meteo Archive API (models=era5_land)",
                    "underlying_data": "ERA5-Land (0.1 deg) 2m_temperature daily maximum",
                    "units": "K",
                    "time_zone": "Asia/Shanghai",
                    "note": f"区域采样点 {len(lats)}×{len(lons)} = {len(pairs)} 个"
                            f"（间隔 {TMAX['spacing']} 度）；下游按 cos(lat) 面积加权平均",
                    "history": f"fetched {pd.Timestamp.now():%Y-%m-%d %H:%M:%S}",
                },
            )
            # 编码与既有 100 个文件保持一致（zlib/complevel=4/dtype），避免同一套数据两种写法
            ds.to_netcdf(target, encoding={"t2m": {"zlib": True, "complevel": 4,
                                                   "dtype": "float32"}})
        (out_dir / f".tmax_{year}.json").unlink()
        done += 1
        log(f"    tmax {year}: 写好 4 个月")

    log(f"  tmax: 新下 {done} 年  已存在 {skipped} 年")


# ---------------------------------------------------------------- 计划与入口

def show_plan(groups, years):
    days = 123                                   # 5—8 月
    rows = {}                                    # (分组, 来源) -> [格点, 请求数, 字节]

    def add(group, source, npts, nbytes):
        row = rows.setdefault((group, source), [npts, 0, 0])
        row[1] += len(years)
        row[2] += nbytes

    for item in PSL_ITEMS:
        if item["group"] in groups:
            lats, lons = axis(item["area"], item["res"])
            npts = lats.size * lons.size
            add(item["group"], "PSL NCSS", npts,
                npts * days * BYTES_PER_VALUE["psl"] * len(years))
    if "tmax" in groups:
        lats, lons = axis(TMAX["area"], TMAX["spacing"])
        npts = lats.size * lons.size
        add("tmax", "Open-Meteo", npts,
            npts * days * BYTES_PER_VALUE["openmeteo"] * len(years))

    print(f"年份 {years[0]}—{years[-1]}（{len(years)} 年）")
    print(f"{'分组':10s}{'来源':>12s}{'格点':>10s}{'请求数':>8s}{'预估体积':>11s}")
    total_req = total_bytes = 0
    for (group, source), (pts, req, nbytes) in rows.items():
        total_req += req
        total_bytes += nbytes
        print(f"{group:10s}{source:>12s}{pts:>10,}{req:>8d}{nbytes / 1e6:>10.0f}M")
    print(f"{'合计':10s}{'':>12s}{'':>10s}{total_req:>8d}{total_bytes / 1e6:>10.0f}M")
    print("体积按实测：PSL netcdf3 float32 = 4 字节/值；Open-Meteo JSON ≈ 20 字节/值")


def main():
    parser = argparse.ArgumentParser(description="PSL + Open-Meteo 取数主线")
    parser.add_argument("--group", action="append",
                        choices=["circ", "mslp", "sst", "tmax"], help="只下指定分组（可重复）")
    parser.add_argument("--years", help="如 1981-1985 或 2024（默认全部 45 年）")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划和预估体积")
    args = parser.parse_args()

    groups = args.group or ["circ", "mslp", "sst", "tmax"]
    years = parse_years(args.years)
    show_plan(groups, years)
    if args.dry_run:
        print("--dry-run：未发起任何请求")
        return

    print()
    for item in PSL_ITEMS:
        if item["group"] in groups:
            log(f"=== {item['group']} / {item['var']} "
                f"{item.get('level') or ''} hPa ===")
            fetch_psl(item, years)
    if "tmax" in groups:
        log("=== tmax / ERA5-Land 日最高（Open-Meteo）===")
        fetch_tmax(years)
    log("完成。重跑本脚本会自动跳过已完成的文件。")


if __name__ == "__main__":
    main()
