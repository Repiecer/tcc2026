#!/usr/bin/env python3
"""tcc 取数公共件：区域定义、参数解析、重试、落盘，以及两个后端。

两个后端：
  planette   ERA5（含气压层）日均，0.25°，AWS us-east-2 匿名读取。
             依赖 icechunk / zarr / s3fs。
  openmeteo  Open-Meteo Archive API（ERA5-Land / ERA5），逐小时，无需注册。
             只用 requests + pandas。

外部只需用到 Factor、run、GPM_PER_M2S2；其余都是内部实现。
被 src/tcc/fetch_*.py 引用，不建议直接运行。
"""

from __future__ import annotations

import argparse
import calendar
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

# ---------------------------------------------------------------- 路径

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
LOG_DIR = ROOT / "logs"

# ---------------------------------------------------------------- 区域

# step 为 None 表示该区域只用于格点数据源（不做点采样）
BOXES: dict[str, dict] = {
    # 长江中下游陆地：与 tmax 的 13x25 / 0.5° 网格一致
    "yrb": dict(south=28.0, north=34.0, west=110.0, east=122.0, step=0.5),
    # 大尺度环流：覆盖西太副高、南亚高压、热带对流
    "asia": dict(south=-10.0, north=60.0, west=40.0, east=180.0, step=None),
}

# ---------------------------------------------------------------- 数据源常量

PLANETTE = dict(
    bucket="planette-era5",
    prefix="ERA5_ic/day",
    region="us-east-2",
    single="single/0p25latx0p25lon",
    pressure="pressure/0p25latx0p25lon",
)

OM_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

GPM_PER_M2S2 = 1.0 / 9.80665  # 位势 -> 位势高度


# ---------------------------------------------------------------- 因子描述

@dataclass
class Factor:
    key: str                      # z500，用于文件名
    out_var: str                  # 输出变量名
    outdir: str                   # data/raw/ 下的子目录
    source: str                   # planette | openmeteo
    units: str
    box: str                      # BOXES 的键
    long_name: str = ""
    # planette 专用
    group: str | None = None      # pressure | single
    pm_var: str | None = None     # Planette / ERA5 变量名
    level: int | None = None      # 气压层 hPa
    scale: float = 1.0            # 乘性换算，如 kg m-2 s-1 -> mm/day 取 86400
    shift: float = 0.0            # 加性换算，如 K -> degC 取 -273.15
    # openmeteo 专用
    om_var: str | None = None
    om_agg: str = "mean"          # mean | sum
    om_models: str = "era5_land"
    extra_attrs: dict = field(default_factory=dict)


# ---------------------------------------------------------------- 小工具

def parse_spec(spec: str, lo: int, hi: int, what: str) -> list[int]:
    """'1981-2025' / '3-8' / '3,6,8' / '3-8,12' -> 有序去重列表。"""
    out: set[int] = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            if a > b:
                a, b = b, a
            out.update(range(a, b + 1))
        else:
            out.add(int(part))
    vals = sorted(out)
    bad = [v for v in vals if v < lo or v > hi]
    if bad:
        raise SystemExit(f"{what} 取值越界: {bad}（允许 {lo}-{hi}）")
    return vals


def setup_logging(name: str) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(name)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOG_DIR / "fetch.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.setLevel(logging.INFO)
    log.handlers[:] = [fh, sh]
    return log


def grid_points(box_name: str):
    """返回 (lats, lons, pts)，pts 按纬度外循环、经度内循环排列（行主序）。"""
    b = BOXES[box_name]
    step = b["step"]
    if step is None:
        raise SystemExit(f"区域 {box_name} 未定义 step，不能用于点采样源")
    lats = np.round(np.arange(b["south"], b["north"] + 1e-9, step), 4)
    lons = np.round(np.arange(b["west"], b["east"] + 1e-9, step), 4)
    pts = [(float(la), float(lo)) for la in lats for lo in lons]
    return lats, lons, pts


def _lon_indices(lon: np.ndarray, west: float, east: float) -> np.ndarray:
    """挑出 [west, east]（统一按 0-360 语义给出）对应的索引。

    做法：先把数据自身的经度统一映射到 0-360 再比较，这样 -180 与 180
    这两个等价的日界线写法不会误判成「跨 180°的框」。
    """
    lon = np.asarray(lon, dtype="float64")
    lon360 = lon % 360.0                   # -180..180 与 0..360 都归一
    w, e = west % 360.0, east % 360.0
    if w <= e:
        mask = (lon360 >= w - 1e-6) & (lon360 <= e + 1e-6)
    else:                                  # 真正跨 180°
        mask = (lon360 >= w - 1e-6) | (lon360 <= e + 1e-6)
    idx = np.where(mask)[0]
    if idx.size == 0:
        raise SystemExit(
            f"经度范围 [{west}, {east}] 没有命中任何格点；"
            f"数据经度范围 [{lon.min()}, {lon.max()}], 归一后 [{lon360.min()}, {lon360.max()}]")
    return idx


def region_indexers(ds: xr.Dataset, box_name: str):
    """按数据集自身坐标约定，返回经纬度维度的 isel 索引。"""
    b = BOXES[box_name]
    latname = "latitude" if "latitude" in ds.coords else "lat"
    lonname = "longitude" if "longitude" in ds.coords else "lon"
    lat = np.asarray(ds[latname].values, dtype="float64")
    lon = np.asarray(ds[lonname].values, dtype="float64")
    lat_idx = np.where((lat >= b["south"] - 1e-6) & (lat <= b["north"] + 1e-6))[0]
    if lat_idx.size == 0:
        raise SystemExit(
            f"纬度范围 [{b['south']}, {b['north']}] 没有命中任何格点；"
            f"数据纬度范围 [{lat.min()}, {lat.max()}]")
    idx = {latname: lat_idx, lonname: _lon_indices(lon, b["west"], b["east"])}
    return idx, latname, lonname


def orient(da: xr.DataArray, latname: str, lonname: str) -> xr.DataArray:
    """统一成纬度升序、经度 0-360 升序，好和 tmax 等既有文件对齐。"""
    lat = np.asarray(da[latname].values, dtype="float64")
    if lat.size > 1 and lat[0] > lat[-1]:
        da = da.isel({latname: slice(None, None, -1)})
    lon = np.asarray(da[lonname].values, dtype="float64")
    if np.any(lon < 0):
        da = da.assign_coords({lonname: (da[lonname].dims, lon % 360.0)})
        lon = lon % 360.0
    if lon.size > 1 and np.any(np.diff(lon) < 0):
        da = da.isel({lonname: np.argsort(lon)})
    return da


def dst_path(factor: Factor, year: int) -> Path:
    return RAW / factor.outdir / f"{factor.key}_{year}.nc"


def write_year(da: xr.DataArray, factor: Factor, year: int,
               force: bool, log: logging.Logger, source_note: str):
    outdir = RAW / factor.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    dst = dst_path(factor, year)
    if dst.exists() and not force:
        log.info("跳过 %s（已存在，用 --force 覆盖）", dst.name)
        return None

    da = da.rename(factor.out_var)
    da.attrs = {
        "units": factor.units,
        "long_name": factor.long_name or factor.out_var,
        "source": source_note,
        "region": factor.box,
        "notes": "daily statistics; tcc project",
        **factor.extra_attrs,
    }
    ds = da.to_dataset()
    ds.attrs["history"] = f"fetched by fetch_{factor.key}.py at {time.strftime('%Y-%m-%d %H:%M:%S')}"
    enc = {factor.out_var: {"zlib": True, "complevel": 4, "dtype": "float32"}}
    tmp = dst.with_suffix(".nc.tmp")
    ds.to_netcdf(tmp, engine="netcdf4", encoding=enc)
    tmp.replace(dst)
    log.info("写出 %s  (%.1f MB, %d 天)",
             dst.name, dst.stat().st_size / 1e6, ds.sizes.get("time", 0))
    return dst


def _tail(done: int, skipped: int, failed: int) -> None:
    print(f"\n完成：新下载 {done}，跳过 {skipped}，失败 {failed}")


# ---------------------------------------------------------------- Planette 后端

def open_planette(group: str) -> xr.Dataset:
    try:
        import icechunk as ic
    except ImportError as e:
        raise SystemExit(
            "需要先安装依赖：.venv/bin/pip install 'icechunk>=0.2' zarr 's3fs>=2024'\n"
            "国内可加：-i https://pypi.tuna.tsinghua.edu.cn/simple"
        ) from e
    storage = ic.s3_storage(bucket=PLANETTE["bucket"], prefix=PLANETTE["prefix"],
                            region=PLANETTE["region"], anonymous=True)
    repo = ic.Repository.open(storage=storage)
    session = repo.readonly_session("main")
    return xr.open_zarr(session.store, group=group, consolidated=False,
                        decode_timedelta=True, chunks={})


def _level_dim(da: xr.DataArray):
    for n in ("level", "plev", "pressure_level", "isobaricInhPa"):
        if n in da.dims or n in da.coords:
            return n
    for d in da.dims:
        if d not in ("time", "latitude", "longitude", "lat", "lon"):
            return d
    return None


def _pick_var(ds: xr.Dataset, factor: Factor) -> xr.DataArray:
    if factor.pm_var not in ds:
        raise SystemExit(f"变量 {factor.pm_var} 不存在，实际变量: {list(ds.data_vars)}")
    da = ds[factor.pm_var]
    if factor.level is not None:
        ln = _level_dim(da)
        if ln is None:
            raise SystemExit(f"变量 {factor.pm_var} 没有气压层维度")
        da = da.sel({ln: factor.level}, method="nearest")
    return da


def _fetch_planette_year(factor: Factor, ds: xr.Dataset, idx: dict,
                         full: xr.DataArray, ren: dict, latname: str, lonname: str,
                         y: int, args, log: logging.Logger, source_note: str) -> bool:
    """取并写出某一年（Planette）。调用方保证该年文件尚不存在。"""
    da = full.sel(time=slice(f"{y}-01-01", f"{y}-12-31")).isel(**idx)
    da = da.sel(time=da.time.dt.month.isin(args.months)).load()
    if factor.scale != 1.0:
        da = da * factor.scale
    if factor.shift != 0.0:
        da = da + factor.shift
    da = orient(da, latname, lonname)
    if ren:
        da = da.rename(ren)
    return bool(write_year(da, factor, y, args.force, log, source_note))


def _year_engine(factor: Factor, years, args, log, fetch_one) -> None:
    """逐年跑 fetch_one，统一处理跳过/计数/报错收尾。"""
    done = skipped = failed = 0
    for y in years:
        dst = dst_path(factor, y)
        if dst.exists() and not args.force:
            log.info("跳过 %s（已存在）", dst.name)
            skipped += 1
            continue
        try:
            done += bool(fetch_one(y))
        except Exception as e:                                    # noqa: BLE001
            log.error("%d 失败：%s", y, e)
            failed += 1
        time.sleep(args.sleep)
    _tail(done, skipped, failed)


def fetch_planette(factor: Factor, years: list[int], months: list[int],
                   args, log: logging.Logger) -> None:
    ds = open_planette(PLANETTE[factor.group])
    idx, latname, lonname = region_indexers(ds, factor.box)
    # 变量与气压层只取一次，不要放在年份循环里反复取
    full = _pick_var(ds, factor)
    ren = {k: v for k, v in ((latname, "latitude"), (lonname, "longitude")) if k != v}
    source_note = ("ERA5 daily statistics via Planette ERA5 Archive "
                   "(AWS us-east-2, anonymous)")
    try:
        _year_engine(factor, years, args, log,
                     lambda y: _fetch_planette_year(factor, ds, idx, full, ren,
                                                    latname, lonname, y, args, log,
                                                    source_note))
    finally:
        ds.close()


# ---------------------------------------------------------------- Open-Meteo 后端

def om_get(url: str, params: dict, timeout: int, attempts: int = 4):
    last_err: Exception | None = None
    for i in range(1, attempts + 1):
        try:
            r = requests.get(url, params=params, timeout=(15, timeout))
            if r.status_code == 400:      # 参数写错，重试也没用
                raise SystemExit(f"Open-Meteo 拒绝请求 400：{r.text[:400]}")
            r.raise_for_status()
            return r.json()
        except SystemExit:
            raise
        except Exception as e:                                    # noqa: BLE001
            last_err = e
            wait = 5 * i
            print(f"第 {i}/{attempts} 次失败（{e}），{wait}s 后重试")
            time.sleep(wait)
    raise RuntimeError(f"下载失败: {url}") from last_err


def om_request_kwargs(factor: Factor, chunk, year: int, months: list[int],
                      timezone: str) -> dict:
    m0, m1 = min(months), max(months)
    last = calendar.monthrange(year, m1)[1]
    return {
        "latitude": ",".join(f"{la:g}" for la, _ in chunk),
        "longitude": ",".join(f"{lo:g}" for _, lo in chunk),
        "start_date": f"{year}-{m0:02d}-01",
        "end_date": f"{year}-{m1:02d}-{last:02d}",
        "hourly": factor.om_var,
        "models": factor.om_models,
        "cell_selection": "nearest",
        "timezone": timezone,
    }


def om_fetch_year(factor: Factor, year: int, months: list[int],
                  args, log: logging.Logger) -> xr.DataArray:
    lats, lons, pts = grid_points(factor.box)
    nlat, nlon = len(lats), len(lons)
    cols: dict[int, pd.Series] = {}
    index = None

    for b0 in range(0, len(pts), args.batch):
        chunk = pts[b0:b0 + args.batch]
        params = om_request_kwargs(factor, chunk, year, months, args.timezone)
        js = om_get(OM_ARCHIVE, params, args.timeout)
        if isinstance(js, dict):
            js = [js]
        if len(js) != len(chunk):
            raise RuntimeError(f"返回点数 {len(js)} != 请求点数 {len(chunk)}")
        if index is None:
            index = pd.to_datetime(js[0]["hourly"]["time"])
        for k, loc in enumerate(js):
            vals = loc["hourly"].get(factor.om_var)
            if vals is None:
                raise RuntimeError(f"响应里没有 {factor.om_var}：{list(loc['hourly'])}")
            cols[b0 + k] = pd.Series(vals, index=index, dtype="float64")
        time.sleep(args.sleep)

    df = pd.DataFrame(cols).reindex(columns=range(len(pts)))

    # ⚠️ 全 NaN 保险丝（2026-09-30 新增）
    # Open-Meteo 对"在指定模型下不提供的变量"不报错，而是返回
    # {"变量名": [null, null, ...]}。此时下面的 Series 全是 NaN，
    # 却依然能写出一个"看起来正常"的 nc 文件 —— 属静默失败。
    # 实测踩过两次：mslp 的 JSON key 写错、ssrd 误用 era5_land。
    if not np.isfinite(df.to_numpy(dtype="float64")).any():
        raise RuntimeError(
            f"Open-Meteo 返回的 `{factor.om_var}` 全部为 null（models={factor.om_models}）。"
            "常见原因：该变量在指定模型下不提供。已实测："
            "shortwave_radiation 需用 models=era5（era5_land 不支持）；"
            "latent_heat_flux 在 era5 与 era5_land 下都不提供。"
            "请先用一个小请求验证变量名与模型，再重跑。")

    daily = df.resample("1D").sum(min_count=1) if factor.om_agg == "sum" else df.resample("1D").mean()
    daily = daily.loc[daily.index.month.isin(months)]

    values = daily.to_numpy(dtype="float32").reshape(len(daily), nlat, nlon)
    return xr.DataArray(
        values,
        dims=("time", "latitude", "longitude"),
        coords={"time": daily.index.values,
                "latitude": lats.astype("float64"),
                "longitude": lons.astype("float64")},
    )


def _fetch_openmeteo_year(factor: Factor, y: int, args, log: logging.Logger,
                          source_note: str) -> bool:
    """取并写出某一年（Open-Meteo）。调用方保证该年文件尚不存在。"""
    da = om_fetch_year(factor, y, args.months, args, log)
    return bool(write_year(da, factor, y, args.force, log, source_note))


def fetch_openmeteo(factor: Factor, years: list[int], months: list[int],
                    args, log: logging.Logger) -> None:
    source_note = ("ERA5-Land / ERA5 via Open-Meteo Archive API "
                   f"(models={factor.om_models})")
    _year_engine(factor, years, args, log,
                 lambda y: _fetch_openmeteo_year(factor, y, args, log, source_note))


# ---------------------------------------------------------------- dry-run / check

def _plan_header(factor: Factor, years: list[int], months: list[int]) -> None:
    print("[%s] %s" % (factor.source, "计划"))
    print("  years   : %d-%d (%d)" % (years[0], years[-1], len(years)))
    print("  months  : %s" % months)
    print("  output  : data/raw/%s/%s_<year>.nc  var=%s units=%s"
          % (factor.outdir, factor.key, factor.out_var, factor.units))


def _plan_openmeteo(factor: Factor, years: list[int], months: list[int], args) -> None:
    lats, lons, pts = grid_points(factor.box)
    print("  url     : %s" % OM_ARCHIVE)
    print("  variable: %s (models=%s, agg=%s)"
          % (factor.om_var, factor.om_models, factor.om_agg))
    print("  grid    : %d x %d = %d 点，lat %s..%s, lon %s..%s, step %s"
          % (len(lats), len(lons), len(pts), lats[0], lats[-1], lons[0], lons[-1],
             BOXES[factor.box]["step"]))
    print("  batches : %d 点/次 -> %d 次/年，共 %d 次"
          % (args.batch, -(-len(pts) // args.batch), -(-len(pts) // args.batch) * len(years)))
    print("  timezone: %s" % args.timezone)
    print("  示例参数:", om_request_kwargs(factor, pts[:3], years[0], months, args.timezone))


def dry_run(factor: Factor, years: list[int], months: list[int], args) -> None:
    _plan_header(factor, years, months)
    if factor.source == "planette":
        print("  store   : s3://%s/%s/%s/" % (PLANETTE["bucket"], PLANETTE["prefix"],
                                              factor.group))
        print("  variable: %s%s" % (factor.pm_var,
                                    "" if factor.level is None else f" @ {factor.level} hPa"))
        print("  region  : %s %s" % (factor.box, BOXES[factor.box]))
    else:
        _plan_openmeteo(factor, years, months, args)


def check(factor: Factor, args) -> None:
    """连通性与结构自检；变量/模型不匹配时明确报错，不靠肉眼看输出。"""
    if factor.source == "planette":
        ds = open_planette(PLANETTE[factor.group])
        idx, latname, lonname = region_indexers(ds, factor.box)
        da = _pick_var(ds, factor)
        print(ds)
        print("\n区域子集形状:", dict(da.isel(time=0, **idx).sizes))
        print("纬度:", ds[latname].values[:3], "...", ds[latname].values[-3:])
        print("经度:", ds[lonname].values[:3], "...", ds[lonname].values[-3:])
        print("命中纬度点:", idx[latname].size, " 命中经度点:", idx[lonname].size)
        ds.close()
        return

    _, _, pts = grid_points(factor.box)
    y = int(str(args.years).split("-")[0])
    params = om_request_kwargs(factor, pts[:4], y, [6], args.timezone)
    params["start_date"] = f"{y}-06-01"
    params["end_date"] = f"{y}-06-03"
    js = om_get(OM_ARCHIVE, params, args.timeout)
    if isinstance(js, dict):
        js = [js]
    h = js[0]["hourly"]
    print("返回点数:", len(js))
    print("请求模型:", factor.om_models)
    print("返回字段:", list(h))
    print("时间:", h["time"][:3], "...", h["time"][-1])
    if factor.om_var not in h:
        raise SystemExit(f"✗ 失败：models={factor.om_models} 没有返回 {factor.om_var}\n"
                         f"  可用字段：{list(h)}\n"
                         f"  → 换一个模型重试，例如把 Factor 的 om_models 改成 "
                         f"\"era5\" 或 \"best_match\"")
    print(f"取值({factor.om_var}):", h[factor.om_var][:5])
    print(f"✓ 通过：models={factor.om_models} 提供 {factor.om_var}，"
          f"单位 {js[0].get('hourly_units', {}).get(factor.om_var)}")


# ---------------------------------------------------------------- 统一入口

def run(factor: Factor, argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog=f"fetch_{factor.key}.py",
        description=f"下载 {factor.out_var}（{factor.units}），源={factor.source}")
    ap.add_argument("--years", default="1981-2025", help="如 1981-2025")
    ap.add_argument("--months", default="5-8", help="如 5-8 或 1-12 或 3,6,8（项目暖季为 5-8）")
    ap.add_argument("--force", action="store_true", help="覆盖已存在的年份文件")
    ap.add_argument("--check", action="store_true", help="只做连通性/结构检查")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划，不联网")
    ap.add_argument("--timeout", type=int, default=120, help="单次请求读超时（秒）")
    ap.add_argument("--sleep", type=float, default=1.0, help="请求间隔（秒）")
    ap.add_argument("--batch", type=int, default=40, help="Open-Meteo 每次请求的点数")
    ap.add_argument("--timezone", default="Asia/Shanghai",
                    help="Open-Meteo 日界时区（仅 openmeteo 源）")
    args = ap.parse_args(argv)

    log = setup_logging(f"fetch_{factor.key}")
    years = parse_spec(args.years, 1979, 2100, "years")
    months = parse_spec(args.months, 1, 12, "months")
    args.months = months            # 后端只需拿 args，避免到处传 months
    log.info("%s  源=%s  年=%d-%d  月=%s", factor.key, factor.source,
             years[0], years[-1], months)

    if args.dry_run:
        dry_run(factor, years, months, args)
        return 0
    if args.check:
        check(factor, args)
        return 0

    try:
        if factor.source == "planette":
            fetch_planette(factor, years, months, args, log)
        else:
            fetch_openmeteo(factor, years, months, args, log)
    except SystemExit:
        raise
    except Exception as e:                                        # noqa: BLE001
        log.error("失败：%s", e)
        log.error("若提示 Network is unreachable，说明当前环境连不到该主机；"
                  "请在你自己能上网的终端里重跑。")
        return 2
    return 0
