"""从 data/raw/mslp-json/ 提取 5-8 月 MSLP，写成与 tmax 一致的 NetCDF。

输入：data/raw/mslp-json/YYYY-MM.json
    Open-Meteo 响应，每个文件是一份"325 个点 × 该月逐日"的列表。
    只取 daily.pressure_msl_mean（hPa）；同文件里的 sea_surface_temperature_mean
    实测全为 null，本脚本不使用。

输出：data/raw/mslp/mslp_YYYY.nc
    变量 pressure_msl，dims=(time, latitude, longitude)，
    经纬度轴与 data/raw/tmax 的 13×25 / 0.5° 网格完全对齐。

⚠️ 必须知道的网格问题
    请求的 325 个点是 13×25 的 0.5° 规则网格，但 Open-Meteo 把其中 6 个点
    吸附到了它自己的细网格上（例如请求 121.0°E 返回 120.75°E，
    请求 30.5°N 返回 30.75°N），因此返回的点云有轻微不规则。
    本脚本把这 325 个点重新插值回目标规则网格（先线性、退化为最近邻），
    以便和 tmax/z500 等资料直接对齐。少数边缘点差值很小，影响可忽略，
    但数据字典里应记录该事实。

用法：
    python scripts/extract_mslp_json.py

产出：
    data/raw/mslp/mslp_YYYY.nc
    docs/data_dictionary_mslp.md
"""

from pathlib import Path

import numpy as np
import xarray as xr
from scipy.interpolate import griddata

SRC_DIR = Path("data/raw/mslp-json")
OUT_DIR = Path("data/raw/mslp")

MONTHS = (5, 6, 7, 8)
JSON_KEY = "pressure_msl_mean"   # JSON 里的字段名
VAR = "pressure_msl"             # 输出 nc 里的变量名

# 目标网格：与 data/raw/tmax 完全一致
TARGET_LAT = np.arange(28.0, 34.0 + 1e-9, 0.5)
TARGET_LON = np.arange(110.0, 122.0 + 1e-9, 0.5)


def read_month(path):
    """读一个 JSON，返回 (时间数组, 经度轴, 纬度轴, 值) ，值形状 (time, nlat, nlon)。"""
    import json

    with open(path, encoding="utf-8") as f:
        recs = json.load(f)

    if not recs:
        return None
    times = recs[0]["daily"]["time"]
    nlat, nlon = 13, 25          # 原始请求是 13 行 × 25 列，按行主序返回

    # 每个点一条时间序列；按请求顺序（纬度外循环、经度内循环）摆回网格
    lat_axis, lon_axis = [], []
    flat = np.full((len(times), len(recs)), np.nan)
    for i, r in enumerate(recs):
        vals = r["daily"].get(JSON_KEY)
        if vals is not None:
            flat[:, i] = [np.nan if v is None else float(v) for v in vals]
        if i < nlat * nlon:
            if i % nlon == 0:
                lat_axis.append(r["latitude"])
            if i < nlon:
                lon_axis.append(r["longitude"])
    if len(lat_axis) != nlat or len(lon_axis) != nlon:
        raise ValueError(f"{path.name}: 点排列不符合 {nlat}×{nlon} 规则网格")
    if not np.isfinite(flat).any():
        raise ValueError(
            f"{path.name}: 字段 {JSON_KEY!r} 一个有效值都没有。"
            f"实际字段为 {list(recs[0]['daily'])}，请检查 JSON_KEY 是否写对。")

    return np.array(times, dtype="datetime64[D]"), lat_axis, lon_axis, flat


def regrid(values, src_lat, src_lon):
    """把不规则点云（src_lat × src_lon，行主序展平）插值到目标规则网格。

    values: (time, nlat*nlon) 展平后的点值。
    先做线性插值；线性无法覆盖的边缘点用最近邻补齐。
    """
    la, lo = np.meshgrid(src_lat, src_lon, indexing="ij")
    pts = np.column_stack([la.ravel(), lo.ravel()])
    tl, to = np.meshgrid(TARGET_LAT, TARGET_LON, indexing="ij")
    tgt = np.column_stack([tl.ravel(), to.ravel()])

    out = np.full((values.shape[0], tgt.shape[0]), np.nan)
    for t in range(values.shape[0]):
        v = values[t]
        ok = np.isfinite(v)
        if ok.sum() < 4:
            continue
        lin = griddata(pts[ok], v[ok], tgt, method="linear")
        near = griddata(pts[ok], v[ok], tgt, method="nearest")
        bad = ~np.isfinite(lin)
        lin[bad] = near[bad]
        out[t] = lin
    return out.reshape(values.shape[0], len(TARGET_LAT), len(TARGET_LON))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_times, all_values = [], []
    yearly = {}

    for year in range(1981, 2026):
        per_year_t, per_year_v = [], []
        for m in MONTHS:
            p = SRC_DIR / f"{year}-{m:02d}.json"
            if not p.exists():
                continue
            got = read_month(p)
            if got is None:
                continue
            t, sla, slo, flat = got
            per_year_t.append(t)
            per_year_v.append(regrid(flat, sla, slo))
        if not per_year_t:
            print(f"  {year}: 无 5-8 月数据，跳过")
            continue

        times = np.concatenate(per_year_t)
        vals = np.concatenate(per_year_v, axis=0)
        order = np.argsort(times)
        times, vals = times[order], vals[order]

        # 同一天重复时取均值（正常不应发生）
        uniq, inv = np.unique(times, return_inverse=True)
        if len(uniq) != len(times):
            vals = np.stack([vals[inv == i].mean(axis=0) for i in range(len(uniq))])
            times = uniq

        dst = OUT_DIR / f"mslp_{year}.nc"
        da = xr.DataArray(
            vals.astype("float32"),
            dims=("time", "latitude", "longitude"),
            coords={"time": times.astype("datetime64[ns]"),
                    "latitude": TARGET_LAT.astype("float32"),
                    "longitude": TARGET_LON.astype("float32")},
            name=VAR,
            attrs={
                "units": "hPa",
                "long_name": "Mean sea level pressure, daily mean",
                "source": "ERA5 via Open-Meteo Archive API (models=era5)",
                "region": "28-34N, 110-122E",
                "notes": "daily statistics; tcc project; regridded from 325 snapped points",
            },
        )
        ds = da.to_dataset()
        ds.attrs["history"] = "extracted from data/raw/mslp-json by extract_mslp_json.py"
        tmp = dst.with_suffix(".nc.tmp")
        ds.to_netcdf(tmp, engine="netcdf4",
                     encoding={VAR: {"zlib": True, "complevel": 4, "dtype": "float32"}})
        tmp.replace(dst)

        yearly[year] = (len(times), float(np.nanmean(vals)))
        all_times.append(times)
        all_values.append(vals)
        print(f"  {year}: {len(times)} 天, 平均 {float(np.nanmean(vals)):.1f} hPa -> {dst.name}")

    if not yearly:
        raise SystemExit("没有提取到任何数据")

    tot = sum(v[0] for v in yearly.values())
    print(f"\n共 {len(yearly)} 年、{tot} 天")
    print("年份范围: %d-%d" % (min(yearly), max(yearly)))
    miss = [y for y in range(1981, 2026) if y not in yearly]
    if miss:
        print("缺 5-8 月数据的年份: %s" % miss)

if __name__ == "__main__":
    main()
