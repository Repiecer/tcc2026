"""V850（850 hPa 经向风）标准化脚本。

单位 m s-1，**北向为正**（ERA5 原生约定）。

在长江中下游，v850 异常的物理含义很直接：

    v850 > 0（南风异常）= 来自海洋的**暖湿**气流 → 西太副高西侧/南侧的
                          西南风，通常伴随副高西伸
    v850 < 0（北风异常）= 冷空气南下，通常对应降温

所以 v850 是规划书 H9 的关键变量之一（"西南风输送水汽"）。

与 U850 脚本的差别：本脚本**不加派生指数**。U850 的南北差有"反气旋性"
这个明确物理意义，而 v850 的南北差（经向风切变）含义不清晰，加进来只会
制造噪声，因此只保留区域平均。

产出：
    data/proc/v850_indices.nc     标准化指数 (index, time)
    data/proc/v850_anomaly.nc     距平 m s-1 (index, time)
    data/proc/v850_norm_params.nc clim / sigma (index, doy)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

RAW_DIR = Path("data/raw/v850")
OUT_DIR = Path("data/proc")
VAR = "v_component_of_wind_850"

REGIONS = {
    "yrb":   ((28, 34), (110, 122)),
    "north": ((31, 34), (110, 122)),
    "south": ((28, 31), (110, 122)),
}
TRAIN = range(1981, 2016)
WINDOW = 11
FLOOR = 0.5


def index_series():
    """每个区域一条面积加权平均的时间序列（m s-1）。"""
    files = sorted(RAW_DIR.glob("v850_*.nc"))
    if not files:
        raise FileNotFoundError(f"no file: {RAW_DIR}/v850_*.nc")

    acc = {k: [] for k in REGIONS}
    for f in files:
        ds = xr.open_dataset(f)
        # 转 float64 再累加，消掉 float32 的累积舍入
        v = ds[VAR].astype("float64")

        # 量级检查：850 hPa 经向风不会超过 ±120 m/s
        if not (float(v.min()) > -120 and float(v.max()) < 120):
            raise ValueError(
                f"{f.name} 数值 {float(v.min()):.2f} ~ {float(v.max()):.2f}，"
                "不像 m/s 的风速")

        for name, (lr, sr) in REGIONS.items():
            sub = v.sel(latitude=slice(*lr), longitude=slice(*sr))
            w = np.cos(np.deg2rad(sub.latitude.astype("float64")))
            acc[name].append(sub.weighted(sub.notnull() * w).mean(["latitude", "longitude"]))
        ds.close()

    out = {n: xr.concat(acc[n], dim="time").sortby("time").rename(n) for n in REGIONS}
    for n in out:
        out[n].attrs["units"] = "m s-1"
        out[n].attrs["sign_convention"] = "northward positive"
    print(f"  文件 {len(files)} 个，{files[0].stem[-4:]}-{files[-1].stem[-4:]} 年"
          f"，指数 {list(out)}")
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
    print("指数        全期范围           训练期均值/标准差（应 0/1）")
    for name, da in named.items():
        v = da.values[train]
        print(f"  {name:10s} {float(da.min()):+7.2f} ~ {float(da.max()):+7.2f}"
              f"      {v.mean():+.4f} / {v.std():.4f}")
        assert abs(v.mean()) < 0.05 and abs(v.std() - 1) < 0.05, f"{name} 训练期尺度异常"

    print("\n泄漏检查（clim 必须只来自训练期）：")
    for name, s in series_dict.items():
        _, c_all, g_all = climatology(s)
        _, c_tr, g_tr = climatology(s.sel(time=s["time"].dt.year.isin(list(TRAIN))))
        delta = max(float(np.abs(c_all - c_tr).max()), float(np.abs(g_all - g_tr).max()))
        print(f"  {name:10s} 最大差异 {delta:.2e} -> "
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
    zs.attrs["units"] = "1 (z-score of V850 anomaly)"
    xr.Dataset({"v850_index": zs}).to_netcdf(OUT_DIR / "v850_indices.nc")

    an = stack(anomaly_dict)
    an.attrs["units"] = "m s-1"
    xr.Dataset({"v850_anom": an}).to_netcdf(OUT_DIR / "v850_anomaly.nc")

    xr.Dataset(
        {"clim": (("index", "doy"), [clim_dict[n][0] for n in names]),
         "sigma": (("index", "doy"), [clim_dict[n][1] for n in names])},
        coords={"index": names, "doy": clim_dict[names[0]][2]},
        attrs={
            "clim_years": f"{TRAIN[0]}-{TRAIN[-1]}",
            "window_days": f"{WINDOW}",
            "sigma_floor_ratio": f"{FLOOR}",
            "regions": str(REGIONS),
            "index_mean_formula": "mean over grid cells of sum(v*cos(lat))/sum(cos(lat))",
            "source": "data/raw/v850/v850_YYYY.nc (Planette ERA5, 850 hPa, 0.25 deg)",
            "unit": "m s-1",
            "sign_convention": "northward positive (era5)",
            "physical_meaning": "v850 > 0 南风，有暖湿气流",
            "zscore_formula": "(index - clim) / sigma",
        },
    ).to_netcdf(OUT_DIR / "v850_norm_params.nc")


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
