"""摸清 ARCO-ERA5 Zarr 结构：维度、坐标、分块、时间编码。"""
import warnings

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

BUCKET = "gcp-public-data-arco-era5"
PREFIX = "ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
# 走 gs:// 让 gcsfs 处理匿名访问；用 https:// 时必须不传 storage_options
STORE = "gs://" + BUCKET + "/" + PREFIX

print("=== 打开 Zarr（只读元数据）===")
ds = xr.open_zarr(STORE, chunks={}, storage_options={"token": "anon"})
print("  顶层维度:", dict(ds.sizes))
print("  坐标:", list(ds.coords))
print("  变量数:", len(ds.data_vars))

print()
print("=== 2m_temperature 结构 ===")
da = ds["2m_temperature"]
print("  dims  :", da.dims)
print("  shape :", da.shape)
print("  dtype :", da.dtype)
print("  chunks:", da.chunks)
print("  attrs :", {k: str(v)[:50] for k, v in list(da.attrs.items())[:4]})

print()
print("=== 坐标范围 ===")
for c in da.coords:
    v = da[c]
    if v.ndim == 1 and v.size > 1:
        step = abs(float(v.values[1]) - float(v.values[0]))
        print("  %-10s size=%-6d %s -> %s   步长=%s" % (
            c, v.size, v.values[0], v.values[-1], step))

print()
print("=== 时间编码 ===")
t = da["time"]
print("  dtype:", t.dtype, " 值域:", t.values[0], "->", t.values[-1])
print("  attrs:", dict(t.attrs))

print()
print("=== 空间方向 ===")
lat = da["latitude"].values
lon = da["longitude"].values
print("  latitude : %.2f -> %.2f (%s)" % (lat[0], lat[-1], "递减" if lat[0] > lat[-1] else "递增"))
print("  longitude: %.2f -> %.2f" % (lon[0], lon[-1]))

print()
print("=== 试读研究区 3 天 ===")
sub = da.sel(latitude=slice(34, 28), longitude=slice(110, 122),
             time=slice("1981-05-01", "1981-05-03"))
print("  选取结果:", dict(sub.sizes))
arr = np.asarray(sub.values)
print("  内存: %.1f MB" % (arr.nbytes / 1e6))
ok = arr[np.isfinite(arr)]
if ok.size:
    print("  取值: %.2f -> %.2f K  (%.2f -> %.2f C)" % (
        ok.min(), ok.max(), ok.min() - 273.15, ok.max() - 273.15))
print("  NaN 占比: %.1f%%" % (100 * np.isnan(arr).mean()))

print()
print("=== 算一次日最高（验证 UTC->北京时 的位移方式）===")
import pandas as pd  # noqa: E402

loc = sub.assign_coords(time=sub.time + pd.Timedelta(hours=8))
daily = loc.resample(time="1D").max()
print("  日最高结果:", dict(daily.sizes))
print("  各日区域平均(℃):", np.round(
    np.asarray(daily.mean(dim=["latitude", "longitude"]).values) - 273.15, 2))
print()
print("OK 结构確認完成")
