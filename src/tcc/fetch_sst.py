from __future__ import annotations

import argparse
import calendar
import logging
import sys
import time
from pathlib import Path

import numpy as np
import requests
import xarray as xr

# ---------------------------------------------------------------- 路径与常量

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "raw" / "sst"
LOG_DIR = ROOT / "logs"

VAR_NAME = "sea_surface_temperature_mean"

# 热带带，0-360 经度约定
LAT_MIN, LAT_MAX = -30.0, 30.0
LON_MIN, LON_MAX = 30.0, 290.0

OISST_BASE = "https://psl.noaa.gov/thredds/ncss/grid/Datasets/noaa.oisst.v2.highres"

PLANETTE_BUCKET = "planette-era5"
PLANETTE_PREFIX = "ERA5_ic/day"
PLANETTE_REGION = "us-east-2"
PLANETTE_GROUP = "single/0p25latx0p25lon"

log = logging.getLogger("fetch_sst")


# ---------------------------------------------------------------- 参数解析

def parse_spec(spec: str, lo: int, hi: int, what: str) -> list[int]:
    """把 '1981-2025' / '3-8' / '3,6,8' / '3-8,12' 解析成有序去重的整数列表。"""
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


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOG_DIR / "fetch_sst.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.setLevel(logging.INFO)
    log.handlers[:] = [fh, sh]


# ---------------------------------------------------------------- 公共工具

def region_slice(ds: xr.Dataset):
    """按数据集自身的坐标方向，构造经纬度切片。"""
    latname = "latitude" if "latitude" in ds.coords else "lat"
    lonname = "longitude" if "longitude" in ds.coords else "lon"
    lat = ds[latname].values
    lon = ds[lonname].values

    if lat[0] > lat[-1]:                       # 纬度递减（ERA5 惯例）
        lat_sel = slice(LAT_MAX, LAT_MIN)
    else:
        lat_sel = slice(LAT_MIN, LAT_MAX)

    if lon.min() < 0:
        log.warning("经度是 -180..180 约定，本脚本按 0-360 处理，请先确认区域是否跨 180°")
        lon_sel = slice(LON_MIN, LON_MAX)
    else:
        lon_sel = slice(LON_MIN, LON_MAX)

    return {latname: lat_sel, lonname: lon_sel}, latname, lonname


def finalize(da: xr.DataArray, latname: str, lonname: str,
             units: str, source_note: str, native_celsius: bool) -> xr.DataArray:
    """统一变量名、坐标名、单位与属性。"""
    da = da.rename(VAR_NAME)
    ren = {}
    if latname != "latitude":
        ren[latname] = "latitude"
    if lonname != "longitude":
        ren[lonname] = "longitude"
    if ren:
        da = da.rename(ren)

    if units == "kelvin" and native_celsius:
        da = da + 273.15
    elif units == "celsius" and not native_celsius:
        da = da - 273.15

    da.attrs = {
        "units": "K" if units == "kelvin" else "degrees_C",
        "long_name": "Daily mean sea surface temperature",
        "source": source_note,
        "region_lat": f"{LAT_MIN} to {LAT_MAX}",
        "region_lon": f"{LON_MIN} to {LON_MAX} (0-360)",
    }
    da.name = VAR_NAME
    return da


def write_year(da: xr.DataArray, year: int, force: bool) -> Path | None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dst = OUT_DIR / f"sst_daily_{year}.nc"
    if dst.exists() and not force:
        log.info("跳过 %s（已存在，用 --force 覆盖）", dst.name)
        return None

    ds = da.to_dataset()
    ds.attrs["history"] = f"fetched by fetch_sst.py at {time.strftime('%Y-%m-%d %H:%M:%S')}"
    enc = {VAR_NAME: {"zlib": True, "complevel": 4, "dtype": "float32"}}
    tmp = dst.with_suffix(".nc.tmp")
    ds.to_netcdf(tmp, engine="netcdf4", encoding=enc)
    tmp.replace(dst)
    mb = dst.stat().st_size / 1e6
    log.info("写出 %s  (%.1f MB, %d 天)", dst.name, mb, ds.sizes.get("time", 0))
    return dst


# ---------------------------------------------------------------- OISST 源

def oisst_url(year: int, months: list[int], stride: int) -> tuple[str, dict]:
    m0, m1 = min(months), max(months)
    last = calendar.monthrange(year, m1)[1]
    params = {
        "var": "sst",
        "north": LAT_MAX, "west": LON_MIN, "east": LON_MAX, "south": LAT_MIN,
        "horizStride": stride,
        "time_start": f"{year}-{m0:02d}-01T00:00:00Z",
        "time_end": f"{year}-{m1:02d}-{last:02d}T23:59:59Z",
        "accept": "netcdf4",
    }
    return f"{OISST_BASE}/sst.day.mean.{year}.nc", params


def oisst_get(url: str, params: dict, timeout: int, attempts: int = 4) -> bytes:
    last_err: Exception | None = None
    for i in range(1, attempts + 1):
        try:
            r = requests.get(url, params=params, timeout=(15, timeout), stream=True)
            if r.status_code == 404:
                raise FileNotFoundError(f"该年文件不存在: {url}")
            r.raise_for_status()
            buf = bytearray()
            for chunk in r.iter_content(1 << 20):
                buf.extend(chunk)
            return bytes(buf)
        except FileNotFoundError:
            raise
        except Exception as e:                                   # noqa: BLE001
            last_err = e
            wait = 5 * i
            log.warning("第 %d/%d 次失败（%s），%ds 后重试", i, attempts, e, wait)
            time.sleep(wait)
    raise RuntimeError(f"下载失败: {url}") from last_err


def fetch_oisst_year(year: int, months: list[int], stride: int,
                     units: str, timeout: int, force: bool) -> Path | None:
    dst = OUT_DIR / f"sst_daily_{year}.nc"
    if dst.exists() and not force:
        log.info("跳过 %s（已存在）", dst.name)
        return None

    url, params = oisst_url(year, months, stride)
    log.info("请求 OISST %d  区域 lat[%s,%s] lon[%s,%s] stride=%d",
             year, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, stride)
    raw = oisst_get(url, params, timeout)

    tmp = OUT_DIR / f".{year}.oisst.tmp.nc"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(raw)
    try:
        ds = xr.open_dataset(tmp, engine="netcdf4")
    except Exception as e:                                        # noqa: BLE001
        head = raw[:300].decode("utf-8", "replace")
        raise RuntimeError(f"返回内容不是 netCDF，前 300 字符:\n{head}") from e

    if "sst" not in ds:
        raise RuntimeError(f"变量 sst 不存在，实际变量: {list(ds.data_vars)}")

    sel, latname, lonname = region_slice(ds)
    da = ds["sst"].sel(**sel)
    da = da.sel(time=da.time.dt.month.isin(months))

    note = "NOAA OISST v2.1 (0.25 deg daily mean), via NOAA PSL NCSS"
    da = finalize(da, latname, lonname, units, note, native_celsius=True)
    da.load()
    ds.close()
    tmp.unlink(missing_ok=True)
    return write_year(da, year, force)


def check_oisst(timeout: int) -> None:
    url = f"{OISST_BASE}/sst.day.mean.2020.nc"
    params = {"var": "sst", "north": 31, "west": 110, "east": 112,
              "south": 30, "horizStride": 1,
              "time_start": "2020-06-01T00:00:00Z",
              "time_end": "2020-06-03T23:59:59Z", "accept": "netcdf4"}
    log.info("连通性检查: %s", url)
    raw = oisst_get(url, params, timeout)
    tmp = OUT_DIR / ".check.nc"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(raw)
    ds = xr.open_dataset(tmp, engine="netcdf4")
    print(ds)
    print("sst units =", ds["sst"].attrs.get("units"))
    print("时间范围 =", ds.time.values.min(), "->", ds.time.values.max())
    print("lat:", ds.lat.values[:3], "...", "lon:", ds.lon.values[:3], "...")
    ds.close()
    tmp.unlink(missing_ok=True)
    log.info("OISST 可用。")


# ---------------------------------------------------------------- Planette 源

def planette_store():
    try:
        import icechunk as ic
    except ImportError as e:                                      # noqa: BLE001
        raise SystemExit(
            "需要先安装依赖：.venv/bin/pip install 'icechunk>=0.2' zarr 's3fs>=2024' "
            "(如网络受限可加 -i https://pypi.tuna.tsinghua.edu.cn/simple)"
        ) from e
    storage = ic.s3_storage(bucket=PLANETTE_BUCKET, prefix=PLANETTE_PREFIX,
                            region=PLANETTE_REGION, anonymous=True)
    repo = ic.Repository.open(storage=storage)
    session = repo.readonly_session("main")
    return session.store


def fetch_planette_year(ds: xr.Dataset, year: int, months: list[int],
                        units: str, force: bool) -> Path | None:
    dst = OUT_DIR / f"sst_daily_{year}.nc"
    if dst.exists() and not force:
        log.info("跳过 %s（已存在）", dst.name)
        return None

    sel, latname, lonname = region_slice(ds)
    da = ds["sst"].sel(time=slice(f"{year}-01-01", f"{year}-12-31"), **sel)
    da = da.sel(time=da.time.dt.month.isin(months))
    da = da.load()

    note = "ERA5 daily mean sea surface temperature, via Planette ERA5 Archive (AWS, anonymous)"
    da = finalize(da, latname, lonname, units, note, native_celsius=False)
    return write_year(da, year, force)


def check_planette() -> None:
    store = planette_store()
    ds = xr.open_zarr(store, group=PLANETTE_GROUP, consolidated=False,
                      decode_timedelta=True, chunks={})
    print(ds)
    if "sst" not in ds:
        raise SystemExit(f"single 组里没有 sst，实际变量: {list(ds.data_vars)}")
    sel, latname, lonname = region_slice(ds)
    sub = ds["sst"].isel(time=0).sel(**sel)
    print("\n区域子集形状:", dict(sub.sizes))
    print("纬度方向:", ds[latname].values[:3], "...", ds[latname].values[-3:])
    print("经度方向:", ds[lonname].values[:3], "...", ds[lonname].values[-3:])
    log.info("Planette store 可打开。")


# ---------------------------------------------------------------- 主流程

def main() -> int:
    ap = argparse.ArgumentParser(description="下载日平均 SST")
    ap.add_argument("--source", choices=["oisst", "planette"], default="oisst")
    ap.add_argument("--years", default="1981-2025", help="如 1981-2025")
    ap.add_argument("--months", default="3-8", help="如 3-8 或 1-12 或 3,6,8")
    ap.add_argument("--stride", type=int, default=1,
                    help="OISST 水平抽稀：1=0.25°, 2=0.5°, 4=1°")
    ap.add_argument("--units", choices=["kelvin", "celsius"], default="kelvin",
                    help="输出单位，默认 K（与 ERA5 族一致）")
    ap.add_argument("--timeout", type=int, default=900, help="单次请求读超时（秒）")
    ap.add_argument("--sleep", type=float, default=2.0, help="每次请求之间的间隔（秒）")
    ap.add_argument("--force", action="store_true", help="覆盖已存在的年份文件")
    ap.add_argument("--check", action="store_true", help="只做连通性/结构检查，不批量下载")
    args = ap.parse_args()

    setup_logging()
    years = parse_spec(args.years, 1979, 2100, "years")
    months = parse_spec(args.months, 1, 12, "months")
    log.info("source=%s  years=%d-%d  months=%s  units=%s",
             args.source, years[0], years[-1], months, args.units)

    if args.check:
        try:
            if args.source == "oisst":
                check_oisst(args.timeout)
            else:
                check_planette()
        except Exception as e:                                    # noqa: BLE001
            log.error("检查失败：%s", e)
            log.error("若提示 Network is unreachable，说明当前环境连不到该主机；"
                      "请在你自己能上网的终端里重跑本命令确认。")
            return 2
        return 0

    done = skipped = failed = 0
    if args.source == "oisst":
        for y in years:
            try:
                if fetch_oisst_year(y, months, args.stride, args.units,
                                    args.timeout, args.force):
                    done += 1
                else:
                    skipped += 1
            except FileNotFoundError as e:
                log.warning("跳过 %d：%s", y, e)
                skipped += 1
            except Exception as e:                                # noqa: BLE001
                log.error("%d 失败：%s", y, e)
                failed += 1
            time.sleep(args.sleep)
    else:
        ds = xr.open_zarr(planette_store(), group=PLANETTE_GROUP,
                          consolidated=False, decode_timedelta=True, chunks={})
        try:
            for y in years:
                try:
                    if fetch_planette_year(ds, y, months, args.units, args.force):
                        done += 1
                    else:
                        skipped += 1
                except Exception as e:                            # noqa: BLE001
                    log.error("%d 失败：%s", y, e)
                    failed += 1
        finally:
            ds.close()

    log.info("完成：新下载 %d 年，跳过 %d 年，失败 %d 年", done, skipped, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())