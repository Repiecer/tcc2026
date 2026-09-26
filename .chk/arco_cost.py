"""实测 ARCO 取数成本，并验证"6 小时采样算日最高"的精度损失。

两个问题一起回答：
  1. 从 ARCO 拉研究区到底要多久？（chunk 是全球一层，很贵）
  2. 如果降到 6 小时采样，日最高温会差多少？（关系到 σ(d) 仅 1.2—1.5℃ 的容忍度）
"""
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

STORE = ("gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3")
AREA = dict(latitude=slice(34, 28), longitude=slice(110, 122))

print("=== 打开 Zarr（只要 2m_temperature）===")
t0 = time.time()
ds = xr.open_zarr(STORE, chunks={"time": 120}, storage_options={"token": "anon"})
da = ds["2m_temperature"]
print("  打开耗时 %.1fs" % (time.time() - t0))

# 取 10 天逐小时
t0 = time.time()
sub = da.sel(time=slice("2022-07-01", "2022-07-11"), **AREA)
arr = sub.compute()
elapsed = time.time() - t0
nbytes = arr.nbytes
print()
print("=== 抓取 10 天逐小时（研究区子集）===")
print("  结果形状 :", dict(arr.sizes))
print("  有效体积 : %.1f MB" % (nbytes / 1e6))
print("  耗时     : %.1f s" % elapsed)
print("  等效速度 : %.2f MB/s" % (nbytes / 1e6 / elapsed))
print("  推算全年 : 45年×123天，逐小时 → %.0f 小时" % (
    (45 * 123 / 10) * elapsed / 3600))
print("  推算全年 : 6小时采样       → %.1f 小时" % (
    (45 * 123 / 10) * elapsed / 6 / 3600))

# ---- 日最高：全小时 vs 降采样 ----
loc = arr.assign_coords(time=arr.time + pd.Timedelta(hours=8))
full = loc.resample(time="1D").max()


def daily_from(stride):
    s = arr.isel(time=slice(0, None, stride))
    s = s.assign_coords(time=s.time + pd.Timedelta(hours=8))
    return s.resample(time="1D").max()


print()
print("=== 日最高温精度：降采样 vs 逐小时 ===")
base = np.asarray(full.mean(dim=["latitude", "longitude"]).values)
print("  %-10s %-12s %-12s %-12s" % ("采样", "平均偏差", "最大偏差", "相对σ(d)"))
for stride, label in ((3, "3小时"), (6, "6小时"), (8, "8小时")):
    d = np.asarray(daily_from(stride).mean(dim=["latitude", "longitude"]).values)
    diff = d - base
    sd = 1.35  # 本项目气温距平年际标准差约 1.2—1.5℃
    print("  %-10s %+8.3f ℃  %8.3f ℃  %8.3f σ" % (
        label, diff.mean(), np.abs(diff).max(), np.abs(diff).max() / sd))

print()
print("=== 逐小时日最高的几个值（℃）===")
print(" ", np.round(base - 273.15, 2))
