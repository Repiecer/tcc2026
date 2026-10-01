"""T850（850 hPa 气温）标准化脚本。

850 hPa 约对应海拔 1500 米，代表**气团本身的冷暖**，不受局地下垫面
（土壤湿度、城市、灌溉、夜间逆温）影响，是热浪研究里的标准诊断量。

用途：
  · H9 环流组的低层热力状态因子
  · 与 U850 / V850 一起算 850 hPa **暖平流**（另见后续脚本）

单位：源数据为 K，本脚本换算为 degC。

产出：
    data/proc/t850_indices.nc     标准化指数 (index, time)
    data/proc/t850_anomaly.nc     距平 degC (index, time)
    data/proc/t850_norm_params.nc clim / sigma (index, doy)
    （数据字典见 scripts/write_dictionaries.py）
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

RAW_DIR = Path("data/raw/t850")
OUT_DIR = Path("data/proc")
VAR = "temperature_850"

REGIONS = {
    "yrb":   ((28, 34), (110, 122)),
    "north": ((31, 34), (110, 122)),
    "south": ((28, 31), (110, 122)),
}
# 派生指数：南北温差 = north - south。
# 实测：热夏年里**北方升温比南方更多**（最热5夏 北+1.06/南+0.69，
# 最凉5夏 北-0.72/南-0.57），所以 grad_ns 与 H **正相关**（年际 r=+0.634）。
# 注意这与"梯度弱=均匀高温"的直觉相反，实测为准。
DERIVED = {"grad_ns": ("north", "south")}
TRAIN = range(1981, 2016)
WINDOW = 11
FLOOR = 0.5


def index_series():
    """每个指数一条面积加权平均的时间序列（degC）。"""
    files = sorted(RAW_DIR.glob("t850_*.nc"))
    if not files:
        raise FileNotFoundError(f"no file: {RAW_DIR}/t850_*.nc")

    acc = {k: [] for k in REGIONS}
    for f in files:
        ds = xr.open_dataset(f)
        # K -> degC；转 float64 再累加，消掉 float32 的累积舍入
        v = ds[VAR].astype("float64") - 273.15

        # 量级检查：850 hPa 气温大致 -60 ~ +45 degC
        if not (-70 < float(v.min()) and float(v.max()) < 50):
            raise ValueError(
                f"{f.name} 数值 {float(v.min()):.2f} ~ {float(v.max()):.2f}，"
                "不像摄氏度（若为 200-320 说明少了 -273.15）")

        for name, (lr, sr) in REGIONS.items():
            sub = v.sel(latitude=slice(*lr), longitude=slice(*sr))
            w = np.cos(np.deg2rad(sub.latitude.astype("float64")))
            acc[name].append(sub.weighted(sub.notnull() * w).mean(["latitude", "longitude"]))
        ds.close()

    out = {n: xr.concat(acc[n], dim="time").sortby("time").rename(n) for n in REGIONS}
    for n, (a, b) in DERIVED.items():
        out[n] = (out[a] - out[b]).rename(n)
        out[n].attrs["units"] = "degC (north - south)"

    for n in out:
        out[n].attrs.setdefault("units", "degC")
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
    zs.attrs["units"] = "1 (z-score of T850 anomaly)"
    xr.Dataset({"t850_index": zs}).to_netcdf(OUT_DIR / "t850_indices.nc")

    an = stack(anomaly_dict)
    an.attrs["units"] = "degC"
    xr.Dataset({"t850_anom": an}).to_netcdf(OUT_DIR / "t850_anomaly.nc")

    xr.Dataset(
        {"clim": (("index", "doy"), [clim_dict[n][0] for n in names]),
         "sigma": (("index", "doy"), [clim_dict[n][1] for n in names])},
        coords={"index": names, "doy": clim_dict[names[0]][2]},
        attrs={
            "clim_years": f"{TRAIN[0]}-{TRAIN[-1]}",
            "window_days": f"{WINDOW}",
            "sigma_floor_ratio": f"{FLOOR}",
            "regions": str(REGIONS),
            "derived": str(DERIVED),
            "index_mean_formula": "mean over grid cells of sum(T*cos(lat))/sum(cos(lat))",
            "source": "data/raw/t850/t850_YYYY.nc (Planette ERA5, 850 hPa, 0.25 deg)",
            "unit": "degC",
            "zscore_formula": "(index - clim) / sigma",
        },
    ).to_netcdf(OUT_DIR / "t850_norm_params.nc")


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
