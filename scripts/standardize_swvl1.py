"""SWVL1（表层土壤体积含水量）标准化脚本。

产出研究区土壤湿度的因子序列，供后续预测模型使用。

用途（规划书 03 的 H2 因子）：
    前期土壤偏干 → 蒸散减弱 → 潜热减少 → 感热增强 → 后期高温增强，预期负相关。

与 tmax/sst/z500/mslp 的异同
----------------------------
相同：区域平均 → 减逐日历日气候态 → 除以训练期 sigma（同一套算法）。
不同：本脚本保留了**土壤湿度的日均值**，不做累积（不像 TP 那样是累积量）。

⚠️ 数据源不一致（重要，必须记录）
    `data/raw/swvl1/` 里混了两种源和两种网格：
      · 25 个文件来自 Planette ERA5（0.25°，25×49）
      · 20 个文件来自 Open-Meteo ERA5-Land（0.5°，13×25）
    ERA5 与 ERA5-Land 的表层土壤湿度是**不同产品**，存在系统性差异。
    本脚本把两者都精确提取到 tmax 的 0.5° 目标网格（匹配误差 0，
    0.25° 是 0.5° 的超集，无插值损失），但**源产品混用这个事实无法消除**，
    需要在报告里说明，并建议日后统一重下。
    脚本运行时会打印每个文件的来源。

用法：
    python scripts/standardize_swvl1.py

产出：
    data/proc/swvl1_indices.nc     标准化指数 (index, time)
    data/proc/swvl1_anomaly.nc     距平 (index, time)
    data/proc/swvl1_norm_params.nc clim / sigma (index, doy)
    docs/data_dictionary_swvl1.md
"""

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

RAW_DIR = Path("data/raw/swvl1")
OUT_DIR = Path("data/proc")

# 目标网格：与 data/raw/tmax 完全一致（脚本启动时从 tmax 读取）
TMAN_REF = Path("data/raw/tmax/tmax_1981_05.nc")

REGIONS = {
    # 整个研究区（土壤湿度在此尺度上较均匀，区域平均有意义）
    "yrb": ((28, 34), (110, 122)),
}
TRAIN = range(1981, 2016)
WINDOW = 11
FLOOR = 0.5


def target_grid():
    """从 tmax 参考文件读取目标经纬度轴，并把范围映射成数值边界。"""
    ref = xr.open_dataset(TMAN_REF)
    lat = np.round(ref.latitude.values, 4)
    lon = np.round(ref.longitude.values, 4)
    ref.close()
    return lat, lon


def extract(da, tlat, tlon):
    """按目标经纬度轴精确提取（最近邻索引 + 误差校验）。

    同时适用于 0.25° 与 0.5° 两种源网格，因为目标点都精确落在源网格上。
    """
    def pick(target, grid):
        idx = np.array([int(np.argmin(np.abs(grid - t))) for t in target])
        err = float(np.max(np.abs(grid[idx] - target)))
        if err > 1e-6:
            raise ValueError(
                f"网格对不上：最大误差 {err:.2e}。源网格 {grid[:3]}... 与目标 {target[:3]}... 不兼容")
        return idx

    il = pick(tlat, np.asarray(da.latitude.values, dtype="float64"))
    io = pick(tlon, np.asarray(da.longitude.values, dtype="float64"))
    out = da.isel(latitude=il, longitude=io)
    return out.assign_coords(latitude=tlat.astype("float32"), longitude=tlon.astype("float32"))


def index_series():
    """每个区域一条面积加权平均的时间序列（m3/m3）。"""
    files = sorted(RAW_DIR.glob("swvl1_*.nc"))
    if not files:
        raise FileNotFoundError(f"no file: {RAW_DIR}/swvl1_*.nc")
    tlat, tlon = target_grid()

    acc = {k: [] for k in REGIONS}
    sources = Counter()
    for f in files:
        ds = xr.open_dataset(f)
        src = ds["soil_moisture_0_7cm"].attrs.get("source", "?")
        sources["Planette" if "Planette" in src else "Open-Meteo"] += 1
        # 转 float64 再累加，消掉 float32 的累积舍入
        v = extract(ds["soil_moisture_0_7cm"].astype("float64"), tlat, tlon)

        # 量级检查：体积含水量应在 0-0.65 之间
        if not (-0.01 < float(v.min()) and float(v.max()) < 0.65):
            raise ValueError(f"{f.name} 数值范围 {float(v.min()):.4f} ~ {float(v.max()):.4f}，"
                             "不像体积含水量 m3/m3")

        for name, (lr, sr) in REGIONS.items():
            sub = v.sel(latitude=slice(*lr), longitude=slice(*sr))
            w = np.cos(np.deg2rad(sub.latitude))
            acc[name].append(sub.weighted(sub.notnull() * w).mean(["latitude", "longitude"]))
        ds.close()

    print("  数据源统计:", dict(sources))
    if len(sources) > 1:
        print("  ⚠️  检测到多源混用：ERA5 与 ERA5-Land 的表层土壤湿度是不同产品，")
        print("      存在系统性差异。已在数据字典中记录，建议日后统一重下。")

    out = {}
    for name in REGIONS:
        s = xr.concat(acc[name], dim="time").sortby("time").rename(name)
        s.attrs["units"] = "m3 m-3"
        out[name] = s
    return out


def climatology(series, years=None):
    """逐日历日的常年平均 clim 和波动幅度 sigma（只用训练期年份）。"""
    y = np.asarray(series["time"].dt.year)
    if years is not None:
        assert set(np.unique(y)) <= set(years), "years 必须覆盖数据里出现的所有年份"
    train = np.isin(y, list(TRAIN))
    doy = doy_of(series)

    table = pd.DataFrame({"doy": doy[train], "T": series.values[train]})
    per_day = table.groupby("doy")["T"].agg(
        n="size", total="sum", sq_total=lambda x: (x ** 2).sum())
    window = per_day.rolling(WINDOW, center=True, min_periods=1).sum()
    clim = (window["total"] / window["n"]).to_numpy()
    variance = np.clip(window["sq_total"] / window["n"] - clim ** 2, 0, None)
    return per_day.index.values, clim, np.sqrt(variance.to_numpy())


def doy_of(series):
    doy = np.asarray(series["time"].dt.dayofyear)
    return np.where(doy == 366, 365, doy)


def zscore(series, doy_list, clim, sigma):
    pos = doy_of(series) - doy_list[0]
    sigma = np.maximum(sigma, FLOOR * sigma.mean())
    return xr.DataArray((series.values - clim[pos]) / sigma[pos],
                        dims="time", coords={"time": series["time"]}, name=series.name)


def anomaly(series, doy_list, clim):
    pos = doy_of(series) - doy_list[0]
    return xr.DataArray(series.values - clim[pos],
                        dims="time", coords={"time": series["time"]}, name=series.name)


def check(named, series_dict):
    years = np.asarray(next(iter(series_dict.values()))["time"].dt.year)
    train = np.isin(years, list(TRAIN))

    print("\n===== 校验 =====")
    print("指数      全期范围        训练期均值/标准差（应 0/1）")
    for name, da in named.items():
        v = da.values[train]
        print(f"  {name:8s} {float(da.min()):+.2f} ~ {float(da.max()):+.2f}"
              f"     {v.mean():+.4f} / {v.std():.4f}")
        assert abs(v.mean()) < 0.05 and abs(v.std() - 1) < 0.05, f"{name} 训练期尺度异常"

    print("\n泄漏检查（clim 必须只来自训练期）：")
    for name in [list(series_dict)[0], list(series_dict)[-1]]:
        s = series_dict[name]
        _, c_all, sg_all = climatology(s)
        _, c_tr, sg_tr = climatology(s.sel(time=s["time"].dt.year.isin(list(TRAIN))))
        delta = max(float(np.abs(c_all - c_tr).max()), float(np.abs(sg_all - sg_tr).max()))
        print(f"  {name:8s} 全量 vs 删掉训练期以外，最大差异 {delta:.2e} -> "
              f"{'通过' if delta < 1e-12 else '不合格（存在泄漏）'}")
        assert delta < 1e-12, f"{name} 的气候态混入了训练期之外的年份"

    print("\n分时段均值（气候态冻结在训练期）：")
    for y0, y1 in [(1981, 2015), (2016, 2020), (2021, 2025)]:
        m = (years >= y0) & (years <= y1)
        print(f"  {y0}-{y1}: " + "  ".join(f"{n}={da.values[m].mean():+.3f}"
                                          for n, da in named.items()))


def stack(d):
    keys = [k for k in REGIONS if k in d]
    return xr.concat([d[k] for k in keys], dim="index").assign_coords(index=keys)


def save(named, anomaly_dict, clim_dict):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    zs = stack(named)
    zs.attrs["units"] = "1 (z-score of SWVL1 anomaly)"
    xr.Dataset({"swvl1_index": zs}).to_netcdf(OUT_DIR / "swvl1_indices.nc")

    an = stack(anomaly_dict)
    an.attrs["units"] = "m3 m-3"
    xr.Dataset({"swvl1_anom": an}).to_netcdf(OUT_DIR / "swvl1_anomaly.nc")

    names = list(named)
    xr.Dataset(
        {"clim": (("index", "doy"), [clim_dict[n][0] for n in names]),
         "sigma": (("index", "doy"), [clim_dict[n][1] for n in names])},
        coords={"index": names, "doy": clim_dict[names[0]][2]},
        attrs={
            "clim_years": f"{TRAIN[0]}-{TRAIN[-1]}",
            "window_days": f"{WINDOW}",
            "sigma_floor_ratio": f"{FLOOR}",
            "regions": str(REGIONS),
            "index_mean_formula": "mean over grid cells of sum(S*cos(lat))/sum(cos(lat))",
            "source": "data/raw/swvl1/swvl1_YYYY.nc (Planette ERA5 + Open-Meteo ERA5-Land 混源)",
            "unit": "m3 m-3",
            "zscore_formula": "(index - clim) / sigma",
            "known_issue": "25 个文件来自 Planette ERA5(0.25°)，20 个来自 Open-Meteo "
                           "ERA5-Land(0.5°)，两产品存在系统性差异，建议统一重下",
        },
    ).to_netcdf(OUT_DIR / "swvl1_norm_params.nc")
def main():
    series_dict = index_series()
    named, anomaly_dict, clim_dict = {}, {}, {}
    for name, s in series_dict.items():
        doy_list, clim, sigma = climatology(s)
        named[name] = zscore(s, doy_list, clim, sigma)
        anomaly_dict[name] = anomaly(s, doy_list, clim)
        clim_dict[name] = (clim, sigma, doy_list)
    check(named, series_dict)
    save(named, anomaly_dict, clim_dict)
    print(f"\n已保存到 {OUT_DIR}/")


if __name__ == "__main__":
    main()
