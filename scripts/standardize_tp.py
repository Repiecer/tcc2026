"""TP（总降水）标准化脚本。

产出研究区降水的因子序列，供后续预测模型使用。

用途（规划书 03 的 H3 因子）：
    前期少雨 → 土壤干化、云量少、蒸发冷却弱 → 高温易发展，预期负相关。

为什么降水要单独处理：必须开方
------------------------------
降水是**强右偏**分布：少数暴雨日贡献大部分方差。实测日区域平均降水：

    | 统计量 | 原始 P | sqrt(P) |
    |---|---|---|
    | 偏度 | +1.42 | +0.34 |
    | 标准化后偏度 | +1.36 | — |
    | 标准化后最大 z | **+4.94** | — |

若直接算 z，一天 44 mm 的暴雨就产生 z = +4.9，把方差和相关系数全拉到
极端日上，线性模型会被几天暴雨主导。**开方（sqrt）是降水最经典的
方差稳定变换**，变换后分布接近正态。

因此本脚本对同一条序列输出两个尺度：

    z 指数（模型输入） = ( sqrt(P) - clim_sqrt ) / sigma_sqrt
    距平（物理量）     = P - clim                    单位 mm/day

日历日编号用"季节内第几天"（5/1=1 … 8/31=123），**不用 dayofyear**：
闰年 5-8 月的 dayofyear 整体 +1，会让同一日历日在闰年/平年错配气候态。

用法：
    python scripts/standardize_tp.py

产出：
    data/proc/tp_indices.nc      sqrt 尺度的标准化指数 (index, time)
    data/proc/tp_anomaly.nc      降水距平 mm/day (index, time)
    data/proc/tp_norm_params.nc  两套 clim/sigma (index, doy)
    （数据字典见 scripts/write_dictionaries.py）
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

RAW_DIR = Path("data/raw/tp")
OUT_DIR = Path("data/proc")

# 目标网格：与 data/raw/tmax 完全一致
TMAN_REF = Path("data/raw/tmax/tmax_1981_05.nc")

REGIONS = {
    "yrb": ((28, 34), (110, 122)),
}
TRAIN = range(1981, 2016)
WINDOW = 11
FLOOR = 0.5


def target_grid():
    """从 tmax 参考文件读取目标经纬度轴。"""
    ref = xr.open_dataset(TMAN_REF)
    lat = np.round(ref.latitude.values, 4)
    lon = np.round(ref.longitude.values, 4)
    ref.close()
    return lat, lon


def extract(da, tlat, tlon):
    """按目标经纬度轴精确提取（最近邻索引 + 误差校验）。"""
    def pick(target, grid):
        idx = np.array([int(np.argmin(np.abs(grid - t))) for t in target])
        err = float(np.max(np.abs(grid[idx] - target)))
        if err > 1e-6:
            raise ValueError(
                f"网格对不上：最大误差 {err:.2e}。"
                f"源网格 {grid[:3]}... 与目标 {target[:3]}... 不兼容")
        return idx

    il = pick(tlat, np.asarray(da.latitude.values, dtype="float64"))
    io = pick(tlon, np.asarray(da.longitude.values, dtype="float64"))
    return da.isel(latitude=il, longitude=io).assign_coords(
        # 坐标保持 float64：若转成 float32，np.cos 会退化为 float32 权重，
        # 给区域平均引入约 1e-7 的精度损失
        latitude=tlat.astype("float64"), longitude=tlon.astype("float64"))


def index_series():
    """每个区域一条面积加权平均的日降水序列（mm/day）。"""
    files = sorted(RAW_DIR.glob("tp_*.nc"))
    if not files:
        raise FileNotFoundError(f"no file: {RAW_DIR}/tp_*.nc")
    tlat, tlon = target_grid()

    acc = {k: [] for k in REGIONS}
    for f in files:
        ds = xr.open_dataset(f)
        # 转 float64 再累加，消掉 float32 的累积舍入
        v = extract(ds["precipitation_sum"].astype("float64"), tlat, tlon)

        # 量级检查：日累计降水不应为负，也不应超过几百 mm
        if float(v.min()) < -0.1 or float(v.max()) > 500:
            raise ValueError(
                f"{f.name} 数值 {float(v.min()):.2f} ~ {float(v.max()):.2f}，"
                "不像 mm/day 的日累计降水（若为 m 或 kg m-2 s-1 说明少了换算）")

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
        s.attrs["units"] = "mm/day"
        out[name] = s
    return out


def doy_of(series):
    """季节内第几天：5/1 = 1 … 8/31 = 123。

    ⚠️ 不能用 dayofyear：闰年 5-8 月的 dayofyear 整体 +1
    （1984-05-01 是 122，平年是 121），会让同一个日历日在闰年/平年
    落到不同的气候态上，还会凭空多出一个只有闰年样本的"日历日"。
    实测这样会让目标 H 产生最大 0.13 sigma 的偏差。
    """
    t = pd.DatetimeIndex(series["time"].values)
    return ((t - pd.to_datetime(t.year.astype(str) + "-05-01")).days + 1).values


def climatology(series):
    """逐日历日的常年平均 clim 和波动幅度 sigma（只用训练期年份）。"""
    y = np.asarray(series["time"].dt.year)
    train = np.isin(y, list(TRAIN))
    doy = doy_of(series)

    table = pd.DataFrame({"doy": doy[train], "P": series.values[train]})
    per_day = table.groupby("doy")["P"].agg(
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
    print(f"日历日编号 {doy_chk[0]}..{doy_chk[-1]}（{len(doy_chk)} 个，应为 123 个 = 5/1..8/31）")
    assert len(doy_chk) == 123, "日历日个数不对，检查 doy_of() 的季节定义"

    print("指数      全期范围        训练期均值/标准差（应 0/1）")
    for name, da in named.items():
        v = da.values[train]
        print(f"  {name:8s} {float(da.min()):+.2f} ~ {float(da.max()):+.2f}"
              f"     {v.mean():+.4f} / {v.std():.4f}")
        assert abs(v.mean()) < 0.05 and abs(v.std() - 1) < 0.05, f"{name} 训练期尺度异常"

    print("\n开方前后对比（训练期 z 的偏度，越接近 0 越好）：")
    for name, s in series_dict.items():
        doy_list, clim, sigma = climatology(s)
        z_raw = (s.values - clim[doy_of(s) - doy_list[0]]) / sigma[doy_of(s) - doy_list[0]]
        print(f"  {name:8s} 原始 z 偏度 {pd.Series(z_raw[train]).skew():+.3f}"
              f"  ->  开方 z 偏度 {pd.Series(named[name].values[train]).skew():+.3f}"
              f"   (最大 |z| 从 {np.abs(z_raw[train]).max():.2f} 降到 "
              f"{float(np.abs(named[name].values).max()):.2f})")

    print("\n泄漏检查（clim 必须只来自训练期）：")
    for name, s in series_dict.items():
        for scale, arr in [("原始", s), ("开方", np.sqrt(s))]:
            _, c_all, g_all = climatology(arr)
            _, c_tr, g_tr = climatology(arr.sel(time=arr["time"].dt.year.isin(list(TRAIN))))
            delta = max(float(np.abs(c_all - c_tr).max()), float(np.abs(g_all - g_tr).max()))
            print(f"  {name:8s} {scale}  最大差异 {delta:.2e} -> "
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
    names = list(named)

    zs = stack(named)
    zs.attrs["units"] = "1 (z-score of sqrt(TP) anomaly)"
    xr.Dataset({"tp_index": zs}).to_netcdf(OUT_DIR / "tp_indices.nc")

    an = stack(anomaly_dict)
    an.attrs["units"] = "mm/day"
    xr.Dataset({"tp_anom": an}).to_netcdf(OUT_DIR / "tp_anomaly.nc")

    xr.Dataset(
        {"clim": (("index", "doy"), [clim_dict[n]["clim"] for n in names]),
         "sigma": (("index", "doy"), [clim_dict[n]["sigma"] for n in names]),
         "clim_sqrt": (("index", "doy"), [clim_dict[n]["clim_sqrt"] for n in names]),
         "sigma_sqrt": (("index", "doy"), [clim_dict[n]["sigma_sqrt"] for n in names])},
        coords={"index": names, "doy": clim_dict[names[0]]["doy"]},
        attrs={
            "clim_years": f"{TRAIN[0]}-{TRAIN[-1]}",
            "window_days": f"{WINDOW}",
            "sigma_floor_ratio": f"{FLOOR}",
            "regions": str(REGIONS),
            "index_mean_formula": "mean over grid cells of sum(P*cos(lat))/sum(cos(lat))",
            "source": "data/raw/tp/tp_YYYY.nc (Planette ERA5, 0.25 deg, 1981-2025)",
            "unit_raw": "mm/day",
            "anomaly_formula": "P - clim          (单位 mm/day)",
            "zscore_formula": "(sqrt(P) - clim_sqrt) / sigma_sqrt",
            "transform_reason": "这里开方了，降sigma",
        },
    ).to_netcdf(OUT_DIR / "tp_norm_params.nc")


def main():
    series_dict = index_series()
    named, anomaly_dict, clim_dict = {}, {}, {}
    for name, s in series_dict.items():
        doy_list, clim, sigma = climatology(s)
        # 开方尺度：把 sqrt(P) 当作新序列，走完全相同的流程
        root = np.sqrt(s)
        doy_s, clim_s, sigma_s = climatology(root)
        named[name] = zscore(root, doy_s, clim_s, sigma_s)
        anomaly_dict[name] = anomaly(s, doy_list, clim)
        clim_dict[name] = {"clim": clim, "sigma": sigma,
                           "clim_sqrt": clim_s, "sigma_sqrt": sigma_s, "doy": doy_list}
    check(named, series_dict)
    save(named, anomaly_dict, clim_dict)
    print(f"\n已保存到 {OUT_DIR}/")
    print("数据字典请运行: python scripts/write_dictionaries.py")


if __name__ == "__main__":
    main()
