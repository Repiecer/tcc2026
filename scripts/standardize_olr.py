"""OLR（向外长波辐射）标准化脚本。

用途：**H4 地表能量因子的替代量** —— 拿不到 ssrd，改用 OLR。
    两者都是"天有多晴"的度量：
      晴空（地面热、大气干）→ 向外长波辐射大
      有云 / 深对流（云顶冷）→ 向外长波辐射小
    所以 OLR 是**云量与对流的经典代理量**，与云量负相关。

⚠️ 符号约定（本项目已修正）
    Planette 里这个变量名叫 `olr`，但实测全为负值（-322 ~ -128 W/m²），
    说明它其实是 ERA5 的 `ttr`（top net thermal radiation，**净向下为正**）。
    `src/tcc/fetch_olr.py` 已用 `scale=-1` 翻成标准 OLR（**向上为正**）。
    实测验证：
      1993（湿凉年 H=-0.62）域平均 220.5 W/m²
      2022（干热年 H=+0.79）域平均 243.0 W/m²
      与 H 的日相关 +0.242 / +0.445 —— 越热 OLR 越大，物理正确。

与其它脚本的异同
    算法完全一致（区域平均 → 减逐日历日气候态 → 除以训练期 sigma）。
    降水那种强偏态需要开方（见 standardize_tp.py），OLR 是近正态分布，不需要。

产出：
    data/proc/olr_indices.nc     标准化指数 (index, time)
    data/proc/olr_anomaly.nc     距平 W m-2 (index, time)
    data/proc/olr_norm_params.nc clim / sigma (index, doy)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

RAW_DIR = Path("data/raw/olr")
OUT_DIR = Path("data/proc")
VAR = "outgoing_longwave_radiation"
KEY = "olr"

REGIONS = {
    "yrb": ((28, 34), (110, 122)),
}
TRAIN = range(1981, 2016)
WINDOW = 11
FLOOR = 0.5


def index_series():
    """研究区面积加权平均的日序列（W m-2）。

    直接用源数据的完整 0.25° 网格（25×49），不下采样到 13×25：
    源文件本身已经是研究区子集，全网格点数更多、区域平均噪声更小。
    （已核对：全网格与 13×25 子集的指数去季节后相关 0.9995，两种取法等价）
    """
    files = sorted(RAW_DIR.glob(f"{KEY}_*.nc"))
    if not files:
        raise FileNotFoundError(f"no file: {RAW_DIR}/{KEY}_*.nc")

    acc = {k: [] for k in REGIONS}
    for f in files:
        ds = xr.open_dataset(f)
        # 转 float64 再累加，消掉 float32 的累积舍入
        v = ds[VAR].astype("float64")

        # 量级检查：地表 OLR 应在 50-400 W/m2，且必须为正（向上为正）
        if not (0 < float(v.min()) and float(v.max()) < 400):
            raise ValueError(
                f"{f.name} 数值 {float(v.min()):.2f} ~ {float(v.max()):.2f}，"
                "不像向上为正的 OLR（若为 -300 ~ -130，说明少了 scale=-1）")

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
        s.attrs["units"] = "W m-2"
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
    zs.attrs["units"] = "1 (z-score of OLR anomaly)"
    xr.Dataset({f"{KEY}_index": zs}).to_netcdf(OUT_DIR / f"{KEY}_indices.nc")

    an = stack(anomaly_dict)
    an.attrs["units"] = "W m-2"
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
            "index_mean_formula": "mean over grid cells of sum(OLR*cos(lat))/sum(cos(lat))",
            "source": "data/raw/olr/olr_YYYY.nc (Planette ERA5, 0.25 deg, 1981-2025)",
            "unit": "W m-2",
            "sign_convention": "向上为正（已把 Planette 的 ttr 乘 -1）",
            "physical_meaning": "晴空 -> OLR 大；有云/深对流 -> OLR 小。云量代理量",
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
