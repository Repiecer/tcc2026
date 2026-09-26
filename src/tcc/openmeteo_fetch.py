"""从 Open-Meteo Archive API 拉取 ERA5-Land 区域日最高气温。

**为什么走这条路**

CDS 在 2026 年长期过载（全局排队 8000+，单账号还有排队配额），实测一晚只落盘 2 个文件；
ARCO-ERA5 的 Zarr 每个 chunk 是「全球一整层」，我们区域只占 0.12%，
逐小时取要 305 GB —— 两条路都不通。

Open-Meteo 的 archive API 提供 **ERA5-Land 的日最高气温**，且支持
``timezone=Asia/Shanghai``（当地日界），与 CDS 那条 ``derived-era5-land-daily-statistics``
是**同一产品**，但不需要认证、不需要排队。

**方法说明（必须写进数据字典）**

* 产品：ERA5-Land（0.1°）日最高 2 米气温，与之前 CDS 路线的产品一致。
* 取数方式：在区域内按固定间隔取**采样点网格**，而不是逐格点全取。
  默认 0.25° 间隔 → 25×49 = 1225 个点，覆盖 ERA5-Land 在研究区内的格点。
* 区域平均：下游对 ``latitude`` 维做 cos φ 面积加权平均（与其他变量口径一致）。
* 单位：API 返回 ℃，这里统一转成 **K**，与 CDS 路线保持同一单位约定。

**请求策略**

API 的限制是「单次请求的数据量」（实测约 5×10^5 个数值），不是点数。
所以按**每年一个请求、只取暖季月份**来切：123 天 × 1225 点 ≈ 1.5×10^5，远在限内。
一年一个请求 → 45 个请求，配合限流间隔约十几分钟即可拉完全部。

**署名要求**：使用 Open-Meteo 需注明来源；底层数据为 ERA5-Land © ECMWF/C3S。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

API = "https://archive-api.open-meteo.com/v1/archive"
MIN_VALID_BYTES = 1024

# 实测到的两条硬约束：
#   1) URL 过长会被 nginx 挡（HTTP 414）—— 1225 个点对约 15 KB，超出 8 KB 上限
#   2) 点数 × 天数超过约 5e5 会被 400 拒绝（"requests too much data"）
#   3) 429 是每分钟限流，需要退避
MAX_POINTS_PER_REQUEST = 250      # URL 安全上限（实测 600 点仍触发 nginx 414，8KB 限制）
MAX_VALUES_PER_REQUEST = 300_000  # 数据量安全上限
SLEEP_BETWEEN = 3.0
RETRY_429_WAIT = 65.0
MAX_RETRIES = 5


def log(msg: str, logfile: Path | None = None) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    if logfile:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        with logfile.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def is_valid(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < MIN_VALID_BYTES:
        return False
    try:
        with xr.open_dataset(path) as ds:
            return "t2m" in ds and ds.sizes.get("time", 0) > 0
    except Exception:  # noqa: BLE001
        return False


def build_points(area: list[float], spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """在 [N,W,S,E] 区域内按 spacing 生成规则的采样点网格。"""
    n, w, s, e = area
    lats = np.round(np.arange(s, n + 1e-9, spacing), 4)
    lons = np.round(np.arange(w, e + 1e-9, spacing), 4)
    return lats, lons


def expand_pairs(lats: np.ndarray, lons: np.ndarray) -> tuple[list, list]:
    """把经纬度网格展开成**成对**的点列表（纬度为主序）。

    ⚠️ Open-Meteo 的 latitude / longitude 参数要求**元素个数相同**，
    是按点配对，不是叉积。之前传网格直接踩了这个坑。
    展平顺序必须与后续 reshape(len(lats), len(lons)) 一致。
    """
    pair_lats = [float(a) for a in lats for _ in lons]
    pair_lons = [float(b) for _ in lats for b in lons]
    return pair_lats, pair_lons


def _seconds_to_next_hour() -> float:
    """距离下一个整点的秒数。Open-Meteo 的小时配额按整点重置。"""
    now = time.time()
    return 3600.0 - (now % 3600.0)


def _request(lats: list[float], lons: list[float], start: str, end: str) -> dict:
    params = {
        "latitude": ",".join(str(x) for x in lats),
        "longitude": ",".join(str(x) for x in lons),
        "start_date": start,
        "end_date": end,
        "daily": "temperature_2m_max",
        "models": "era5_land",
        "timezone": "Asia/Shanghai",
    }
    url = API + "?" + urllib.parse.urlencode(params)
    last = ""
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=180) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode()[:160]
            last = f"HTTP {exc.code}: {body}"
            if exc.code == 429:
                # 两种限流：分钟级（等一分钟）和**小时级**（等到下一个整点）
                if "Hourly" in body or "hour" in body.lower():
                    wait = _seconds_to_next_hour() + 30
                    log(f"    ⏳ 触发小时级限流，等待 {wait:.0f}s 到下一个整点")
                    time.sleep(wait)
                else:
                    time.sleep(RETRY_429_WAIT)
            elif exc.code == 400:
                raise RuntimeError("请求过大，需要减小分块: " + body) from exc
            else:
                time.sleep(5 * (attempt + 1))
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"重试 {MAX_RETRIES} 次仍失败: {last}")


def fetch_group(cfg: dict, group: dict, logfile: Path | None = None,
                dry_run: bool = False) -> list[Path]:
    """按年拉取，写成与 CDS 路线同构的 (time, latitude, longitude) NetCDF。"""
    name = group["name"]
    area = group["area"]
    spacing = float(group.get("point_spacing", cfg.get("point_spacing", 0.25)))
    out_dir = cfg["data_dir"] / "raw" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    lats, lons = build_points(area, spacing)
    npts = len(lats) * len(lons)
    months = sorted(int(m) for m in cfg["months"])
    years = cfg["_years"]
    batch = max(1, int(cfg.get("year_batch", 1)))
    blocks = [years[i:i + batch] for i in range(0, len(years), batch)]

    log(f"=== Open-Meteo 组 [{name}] ===", logfile)
    log(f"    产品: ERA5-Land daily temperature_2m_max, timezone=Asia/Shanghai", logfile)
    log(f"    采样点: {len(lats)}×{len(lons)} = {npts} 个（间隔 {spacing}°）", logfile)
    log(f"    年份: {years[0]}—{years[-1]}  月份: {months}  年块: {batch}", logfile)

    if dry_run:
        tg = [out_dir / (f"{name}_{b[0]}-{b[-1]}_{m:02d}.nc" if len(b) > 1
                         else f"{name}_{b[0]}_{m:02d}.nc") for b in blocks for m in months]
        log(f"    [dry-run] 将写出 {len(tg)} 个文件；每个请求 {len(months)*31}天×{npts}点 "
            f"≈ {len(months)*31*npts/1000:.0f}k 数值", logfile)
        return tg

    # ---- 二维分块：点块 × 单年；**每完成一年立即写盘**
    #
    # 两个设计都是被实测教训逼出来的：
    #   * Open-Meteo 的 start/end 是**连续区间**，写 7 年区间会返回 2300+ 天而非 7 个暖季
    #   * 它是**每小时**限流；如果先把 45 年全拉进内存再写，一旦中途撞限流，
    #     前面所有成功的请求全部作废（第一版就是这样白扔了 18 个请求）
    # 所以：一年一个请求，一年一落盘。
    point_chunks = [list(range(i, min(i + MAX_POINTS_PER_REQUEST, npts)))
                    for i in range(0, npts, MAX_POINTS_PER_REQUEST)]
    all_pairs_lat, all_pairs_lon = expand_pairs(lats, lons)
    m0, m1 = months[0], months[-1]
    log(f"    分块: {len(point_chunks)} 个点块 × {len(years)} 年 "
        f"= {len(point_chunks)*len(years)} 个请求；每年一落盘", logfile)

    written: list[Path] = []
    for year in years:
        yseries: dict[int, dict[str, float]] = {}
        for ci, chunk in enumerate(point_chunks, 1):
            clats = [all_pairs_lat[i] for i in chunk]
            clons = [all_pairs_lon[i] for i in chunk]
            log(f"    {year} 点块{ci}/{len(point_chunks)} ({len(chunk)}点)", logfile)
            payload = _request(clats, clons, f"{year}-{m0:02d}-01", f"{year}-{m1:02d}-31")
            if isinstance(payload, dict):
                payload = [payload]
            for li, item in enumerate(payload):
                d = item["daily"]
                gi = chunk[li] if li < len(chunk) else chunk[-1]
                b = yseries.setdefault(gi, {})
                for ds_, v in zip(d["time"], d["temperature_2m_max"]):
                    if v is not None:
                        b[ds_] = v
            time.sleep(SLEEP_BETWEEN)

        # 该年数据到手 → 立刻写 4 个月的文件
        for month in months:
            dates = [f"{year}-{month:02d}-{d:02d}" for d in range(1, 32)
                     if f"{year}-{month:02d}-{d:02d}" in yseries.get(0, {})]
            if not dates:
                continue
            grid = np.full((len(dates), len(lats), len(lons)), np.nan, dtype="float32")
            for gi in range(npts):
                b = yseries.get(gi)
                if not b:
                    continue
                grid[:, gi // len(lons), gi % len(lons)] = np.array(
                    [b.get(dt, np.nan) for dt in dates], dtype="float32")
            grid += 273.15
            target = out_dir / f"{name}_{year}_{month:02d}.nc"
            dso = xr.Dataset(
                {"t2m": (("time", "latitude", "longitude"), grid)},
                coords={"time": pd.to_datetime(dates),
                        "latitude": lats, "longitude": lons},
                attrs={
                    "source": "Open-Meteo Archive API (models=era5_land)",
                    "underlying_data": "ERA5-Land (0.1 deg) 2m_temperature daily maximum",
                    "time_zone": "Asia/Shanghai",
                    "note": f"区域采样点 {npts} 个（间隔 {spacing} 度）；下游按 cos(lat) 面积加权平均",
                    "history": f"fetched {datetime.now():%Y-%m-%d %H:%M:%S}",
                },
            )
            dso.to_netcdf(target, encoding={"t2m": {"zlib": True, "complevel": 4,
                                                    "dtype": "float32"}})
            dso.close()
            written.append(target)
        log(f"    ✓ {year} 年落盘完成（累计 {len(written)} 个文件）", logfile)

    return written


def source_group_names(cfg: dict, source: str, only: list[str] | None = None) -> list[str]:
    return [n for n, g in cfg["groups"].items()
            if g.get("source") == source and (not only or n in only)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="从 Open-Meteo 拉取区域日最高气温")
    ap.add_argument("--config", default="configs/download.yaml")
    ap.add_argument("--group", action="append")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from download_workers import load_config  # type: ignore

    cfg = load_config(Path(args.config))
    logfile = cfg["data_dir"] / "logs" / "download.log"
    names = args.group or source_group_names(cfg, "openmeteo", None)
    if not names:
        log("配置里没有 source: openmeteo 的分组", logfile)
        return 0
    for n in names:
        fetch_group(cfg, cfg["groups"][n], logfile, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
