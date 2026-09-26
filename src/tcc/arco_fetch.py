"""从 ARCO-ERA5（Google Cloud 公共数据集）直接拉取区域日最高气温。

**为什么需要这条路径**

CDS（Climate Data Store）在 2026 年长期处于过载状态：全局排队 8000+，
且对单账号的"排队中请求数"有硬配额。实测一晚（8 小时）只成功落盘 2 个文件。
而 ARCO 是 ECMWF 在 Google Cloud 上的官方 Zarr 镜像，支持**按块读取、无需排队**，
实测本机 1.2—1.4 MB/s，本项目研究区子集仅约 0.65 GB。

**与 CDS 路径的差异（必须在数据字典里写明）**

* 数据集：ARCO 是 **ERA5**（0.25°），CDS 那条用的是 **ERA5-Land**（0.1°，仅陆地）。
  换产品不是错误——很多研究就用 ERA5 的 2m_temperature——但两者不能混用，
  必须全项目统一，并在数据字典中登记。
* 日统计：CDS 由服务端算好 `daily_maximum`；这里由本地按**北京时间日界**算，
  口径与 CDS 的 `time_zone=utc+08:00` 一致（当地日 = UTC [D-1 16:00, D 15:59]）。

**输出**

与 CDS 路径同样的命名和粒度：``<data_dir>/raw/tmax/tmax_<年起>-<年止>_<月>.nc``，
变量名 ``t2m``（单位 K），时间坐标为**当地日**。这样下游拼接逻辑无需改动。
"""

from __future__ import annotations

import argparse
import calendar
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

BUCKET = "gcp-public-data-arco-era5"
PREFIX = "ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
# 走 gs:// 让 gcsfs 处理匿名访问。
# ⚠️ 若改用 https://storage.googleapis.com/...，必须**不传** storage_options，
#    否则 fsspec 的 HTTP 后端会因收到 `token` 参数而报 TypeError。
STORE = f"gs://{BUCKET}/{PREFIX}"
STORAGE_OPTIONS = {"token": "anon"}

MIN_VALID_BYTES = 1024


def log(msg: str, logfile: Path | None = None) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    if logfile:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        with logfile.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def is_valid(path: Path) -> bool:
    """能不能当作已完成的产物跳过：存在、够大、能打开、有时间维。"""
    if not path.exists() or path.stat().st_size < MIN_VALID_BYTES:
        return False
    try:
        with xr.open_dataset(path) as ds:
            return len(ds.data_vars) > 0 and ds.sizes.get("time", 0) > 0
    except Exception:  # noqa: BLE001
        return False


def _year_blocks(years: list[int], batch: int) -> list[list[int]]:
    batch = max(1, int(batch))
    return [years[i:i + batch] for i in range(0, len(years), batch)]


def fetch_group(
    cfg: dict,
    group: dict,
    logfile: Path | None = None,
    dry_run: bool = False,
) -> list[Path]:
    """拉取一个分组的区域日最高气温，返回写出的文件路径列表。

    逐 (年块, 月) 处理：先把该月的 UTC 时间窗读进来，平移到北京时间后
    按日历日取最大值 —— 这样"当地日"的定义与 CDS 的 ``time_zone=utc+08:00`` 一致。
    """
    name = group["name"]
    area = group["area"]                     # [N, W, S, E]
    var = group.get("arco_variable", "2m_temperature")
    scale = group.get("scale_to_celsius", False)
    out_dir = cfg["data_dir"] / "raw" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    years = cfg["_years"]
    months = [int(m) for m in cfg["months"]]
    blocks = _year_blocks(years, cfg.get("year_batch", 1))

    log(f"=== ARCO 组 [{name}] ===", logfile)
    log(f"    存储: {STORE}", logfile)
    log(f"    变量: {var}   区域: {area}   年块: {[f'{b[0]}-{b[-1]}' for b in blocks]}", logfile)

    if dry_run:
        targets = [
            out_dir / f"{name}_{b[0]}-{b[-1]}_{m:02d}.nc" if len(b) > 1
            else out_dir / f"{name}_{b[0]}_{m:02d}.nc"
            for b in blocks for m in months
        ]
        log(f"    [dry-run] 将写出 {len(targets)} 个文件，例如 {targets[0].name}", logfile)
        return targets

    log("    打开 Zarr（首次会拉取元数据，稍慢）...", logfile)
    ds = xr.open_zarr(STORE, chunks={"time": 240}, storage_options=STORAGE_OPTIONS)
    da = ds[var]
    # ARCO 纬度是 90 → -90 递减，所以 slice 要 (北, 南)
    lat_slice = slice(area[0], area[2])
    lon_slice = slice(area[1], area[3])

    written: list[Path] = []
    for block in blocks:
        for month in months:
            tag = f"{block[0]}-{block[-1]}" if len(block) > 1 else f"{block[0]}"
            target = out_dir / f"{name}_{tag}_{month:02d}.nc"
            if is_valid(target):
                written.append(target)
                continue

            pieces = []
            for year in block:
                last = calendar.monthrange(year, month)[1]
                # 当地日 D = UTC [D-1 16:00, D 15:59]，所以 UTC 窗口要前后各放宽
                utc_start = pd.Timestamp(f"{year}-{month:02d}-01") - pd.Timedelta(hours=8)
                utc_end = pd.Timestamp(f"{year}-{month:02d}-{last:02d}") + pd.Timedelta(days=1)
                sub = da.sel(latitude=lat_slice, longitude=lon_slice,
                             time=slice(utc_start, utc_end))
                sub = sub.assign_coords(time=sub.time + pd.Timedelta(hours=8))
                daily = sub.resample(time="1D").max()
                daily = daily.sel(time=slice(f"{year}-{month:02d}-01",
                                             f"{year}-{month:02d}-{last:02d}"))
                pieces.append(daily.compute())

            out = xr.concat(pieces, dim="time").rename("t2m")
            if scale:
                out = out - 273.15
                out.attrs["units"] = "degC"
            else:
                out.attrs.setdefault("units", "K")
            out.attrs["long_name"] = "2 metre temperature, daily maximum (local day)"
            out.attrs["source"] = f"ARCO-ERA5 {PREFIX}"
            out.attrs["time_zone"] = cfg["time_zone"]
            out.attrs["history"] = f"fetched {datetime.now():%Y-%m-%d %H:%M:%S}"

            enc = {"t2m": {"zlib": True, "complevel": 4, "dtype": "float32"}}
            out.to_netcdf(target, encoding=enc)
            log(f"    ✓ {target.name}  {out.sizes['time']} 天  "
                f"{target.stat().st_size / 1e6:.2f} MB", logfile)
            written.append(target)

    return written


def arco_group_names(cfg: dict, only: list[str] | None = None) -> list[str]:
    names = []
    for name, g in cfg["groups"].items():
        if g.get("source") != "arco":
            continue
        if only and name not in only:
            continue
        names.append(name)
    return names


def main(argv: list[str] | None = None) -> int:
    """独立运行入口，便于单独测试 ARCO 取数而不用跑通整个下载流程。"""
    ap = argparse.ArgumentParser(description="从 ARCO-ERA5 拉取区域日最高气温")
    ap.add_argument("--config", default="configs/download.yaml")
    ap.add_argument("--group", action="append")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    # 复用 download_workers 的配置加载，避免两套解析逻辑
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from download_workers import load_config  # type: ignore

    cfg = load_config(Path(args.config))
    logfile = cfg["data_dir"] / "logs" / "download.log"

    names = args.group or arco_group_names(cfg, None)
    if not names:
        log("配置里没有 source: arco 的分组", logfile)
        return 0
    for name in names:
        g = cfg["groups"][name]
        fetch_group(cfg, g, logfile, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
