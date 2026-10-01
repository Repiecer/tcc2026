"""SWV2（第 2 层土壤体积含水量，7-28 cm）标准化脚本。

用途：**H2 陆面记忆因子**，比 `swv1`（0-7 cm）更深、记忆更长。

为什么深一层更有价值
    ERA5 把土壤分成 4 层，各自的记忆时间尺度差很多：

        swvl1   0-7 cm     1-3 天（一场雨就变）
        swv2    7-28 cm    1-2 周      ← 本脚本
        swv3    28-100 cm  数周-数月（植物根区）
        swv4    100-289 cm 数月

    延伸期预测（1-4 周）最需要的正是**慢变量**。实测（2022 年域平均）：
        swv1  0.2987 m3/m3   日波动标准差 0.0465
        swv2  0.3029 m3/m3   日波动标准差 0.0376   （比 swv1 小 19%）

物理链条（规划书 H2）
    前期土壤偏干 → 蒸散减弱 → 潜热通量减少 → 感热增强 → 高温增强
    预期**负相关**。

产出：
    data/proc/swv2_indices.nc     标准化指数 (index, time)
    data/proc/swv2_anomaly.nc     距平 m3 m-3 (index, time)
    data/proc/swv2_norm_params.nc clim / sigma (index, doy)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

RAW_DIR = Path("data/raw/swv2")
OUT_DIR = Path("data/proc")
VAR = "soil_moisture_7_28cm"
KEY = "swv2"
LAYER = "7-28 cm"

REGIONS = {
    "yrb": ((28, 34), (110, 122)),
}
TRAIN = range(1981, 2016)
WINDOW = 11
FLOOR = 0.5


def index_series():
    """研究区面积加权平均的日序列（m3 m-3）。

    直接用源数据的完整 0.25° 网格（25×49），不下采样到 13×25：
    源文件本身已经是研究区子集，全网格点数更多、区域平均噪声更小。
    """
    files = sorted(RAW_DIR.glob(f"{KEY}_*.nc"))
    if not files:
        raise FileNotFoundError(f"no file: {RAW_DIR}/{KEY}_*.nc")

    acc = {k: [] for k in REGIONS}
    for f in files:
        ds = xr.open_dataset(f)
        # 转 float64 再累加，消掉 float32 的累积舍入
        v = ds[VAR].astype("float64")

        # 量级检查：体积含水量应在 0-0.65 之间
        if not (-0.01 < float(v.min()) and float(v.max()) < 0.65):
            raise ValueError(f"{f.name} 数值范围 {float(v.min()):.4f} ~ "
                             f"{float(v.max()):.4f}，不像体积含水量 m3/m3")

        for name, (lr, sr) in REGIONS.items():
            sub = v.sel(latitude=slice(*lr), longitude=slice(*sr))
            w = np.cos(np.deg2rad(sub.latitude.astype("float64")))
            acc[name].append(sub.weighted(sub.notnull() * w).mean(["latitude", "longitude"]))
        ds.close()

    print(f"  文件 {len(files)} 个，{files[0].stem[-4:]}-{files[-1].stem[-4:]} 年")
    out = {}
    for name in REGIONS:
        s = xr.concat(acc[name], dim="time").sortby("time").rename(name)
        months = np.unique(np.asarray(s["time"].dt.month))
        if list(months) != [5, 6, 7, 8]:
            raise ValueError(f"季节不是 5-8 月，实际月份 {months}")
        s.attrs["units"] = "m3 m-3"
        out[name] = s
    return out


def doy_of(series):
    """季节内第几天：5/1 = 1 … 8/31 = 123。

    不能用 dayofyear：闰年 3/1 之后的 dayofyear 整体 +1（1984-05-01 是 122，
    平年是 121），会让同一日历日在闰年/平年错配到不同的气候态上。
    """
    t = pd.DatetimeIndex(series["time"].values)
    d = ((t - pd.to_datetime(t.year.astype(str) + "-05-01")).days + 1).values
    assert d.min() == 1, "季节日编号必须从 1 开始"
    return d


def climatology(series):
    """逐日历日的常年平均 clim 和波动幅度 sigma（只用训练期年份）。"""
    y = np.asarray(series["time"].dt.year)
    train = np.isin(y, list(TRAIN))
    doy = doy_of(series)

    table = pd.DataFrame({"doy": doy[train], "T": series.values[train]})
    per_day = table.groupby("doy")["T"].agg(
        n="size", total="sum", sq_total=lambda x: (x ** 2).sum())
    window = per_day.rolling(WINDOW, center=True, min_periods=1).sum()
    clim = (window["total"] / window["n"]).to_numpy()
    variance = np.clip(window["sq_total"] / window["n"] - clim ** 2, 0, None)
    return per_day.index.values, clim, np.sqrt(variance.to_numpy())


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
    doy_chk, _, _ = climatology(next(iter(series_dict.values())))
    print(f"日历日编号 {doy_chk[0]}..{doy_chk[-1]}（{len(doy_chk)} 个，应为 123）")
    assert len(doy_chk) == 123, "日历日个数不对，检查 doy_of()"

    print("指数      全期范围        训练期均值/标准差（应 0/1）")
    for name, da in named.items():
        v = da.values[train]
        print(f"  {name:8s} {float(da.min()):+.2f} ~ {float(da.max()):+.2f}"
              f"     {v.mean():+.4f} / {v.std():.4f}")
        assert abs(v.mean()) < 0.05 and abs(v.std() - 1) < 0.05, f"{name} 训练期尺度异常"

    print("\n泄漏检查（clim 必须只来自训练期）：")
    for name, s in series_dict.items():
        _, c_all, g_all = climatology(s)
        _, c_tr, g_tr = climatology(s.sel(time=s["time"].dt.year.isin(list(TRAIN))))
        delta = max(float(np.abs(c_all - c_tr).max()), float(np.abs(g_all - g_tr).max()))
        print(f"  {name:8s} 最大差异 {delta:.2e} -> "
              f"{'通过' if delta < 1e-12 else '不合格（存在泄漏）'}")
        assert delta < 1e-12, f"{name} 的气候态混入了训练期之外的年份"

    print("\n分时段均值（气候态冻结在训练期）：")
    for y0, y1 in [(1981, 2015), (2016, 2020), (2021, 2025)]:
        m = (years >= y0) & (years <= y1)
        print(f"  {y0}-{y1}: " + "  ".join(f"{n}={da.values[m].mean():+.3f}"
                                          for n, da in named.items()))


def stack(d):
    keys = list(d)
    return xr.concat([d[k] for k in keys], dim="index").assign_coords(index=keys)


def save(named, anomaly_dict, clim_dict):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    names = list(named)

    zs = stack(named)
    zs.attrs["units"] = f"1 (z-score of SWV2 anomaly)"
    xr.Dataset({f"{KEY}_index": zs}).to_netcdf(OUT_DIR / f"{KEY}_indices.nc")

    an = stack(anomaly_dict)
    an.attrs["units"] = "m3 m-3"
    xr.Dataset({f"{KEY}_anom": an}).to_netcdf(OUT_DIR / f"{KEY}_anomaly.nc")

    xr.Dataset(
        {"clim": (("index", "doy"), [clim_dict[n][0] for n in names]),
         "sigma": (("index", "doy"), [clim_dict[n][1] for n in names])},
        coords={"index": names, "doy": clim_dict[names[0]][2]},
        attrs={
            "clim_years": f"{TRAIN[0]}-{TRAIN[-1]}",
            "window_days": f"{WINDOW}",
            "sigma_floor_ratio": f"{FLOOR}",
            "regions": str(REGIONS),
            "layer_depth": LAYER,
            "index_mean_formula": "mean over grid cells of sum(SWV*cos(lat))/sum(cos(lat))",
            "source": f"data/raw/{KEY}/{KEY}_YYYY.nc (Planette ERA5, 0.25 deg, 1981-2025)",
            "unit": "m3 m-3",
            "physical_meaning": "土壤偏干 -> 蒸散弱 -> 潜热少 -> 感热多 -> 高温（预期负相关）",
            "zscore_formula": "(index - clim) / sigma",
        },
    ).to_netcdf(OUT_DIR / f"{KEY}_norm_params.nc")


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
    print("数据字典请运行: python scripts/write_dictionaries.py")


if __name__ == "__main__":
    main()
