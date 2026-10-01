"""D2m（2 米露点温度）标准化脚本。

产出研究区露点的因子序列，供后续预测模型使用。

用途（规划书 03 的 H6 因子）：
    前期偏干（露点低）→ 蒸发需求大、土壤失水快 → 高温易发展，预期负相关。

与 tmax/sst/z500/mslp/swvl1 的异同
----------------------------------
相同：区域平均 → 减逐日历日气候态 → 除以训练期 sigma（同一套算法）。
不同：历史上有两种源网格，故保留"任意网格精确提取"的写法（见 extract）。

数据源（已统一）
    `data/raw/d2m/` 45 个文件全部来自 Planette ERA5（0.25°，25×49），1981-2025。
    历史遗留：1982-1992 那 11 年曾用 Open-Meteo ERA5-Land 下载，两源区域平均
    相差约 0.51 degC；若与其余年份混用，会在训练期内部造成台阶。现已全部
    改用 Planette 重下，混源已消除。脚本仍保留来源统计，一旦再现混源会自动告警。

用法：
    python scripts/standardize_d2m.py

产出：
    data/proc/d2m_indices.nc      标准化指数 (index, time)
    data/proc/d2m_anomaly.nc      距平 (index, time)
    data/proc/d2m_norm_params.nc  clim / sigma (index, doy)
    （数据字典见 scripts/write_dictionaries.py）
"""

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

RAW_DIR = Path("data/raw/d2m")
OUT_DIR = Path("data/proc")

TMAN_REF = Path("data/raw/tmax/tmax_1981_05.nc")

REGIONS = {
    "yrb": ((28, 34), (110, 122)),
}
TRAIN = range(1981, 2016)
WINDOW = 11
FLOOR = 0.5


def target_grid():
    ref = xr.open_dataset(TMAN_REF)
    lat = np.round(ref.latitude.values, 4)
    lon = np.round(ref.longitude.values, 4)
    ref.close()
    return lat, lon


def extract(da, tlat, tlon):
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
    """每个区域一条面积加权平均的时间序列（degC）。"""
    files = sorted(RAW_DIR.glob("d2m_*.nc"))
    if not files:
        raise FileNotFoundError(f"no file: {RAW_DIR}/d2m_*.nc")
    tlat, tlon = target_grid()

    acc = {k: [] for k in REGIONS}
    sources = Counter()
    years = []
    for f in files:
        ds = xr.open_dataset(f)
        v0 = ds["dew_point_2m"]
        s = v0.attrs.get("source", "")
        src = "Planette" if "Planette" in s else ("Open-Meteo" if "Open-Meteo" in s else s or "未知")
        sources[src] += 1
        years.append(int(f.stem.split("_")[-1]))
        # 转 float64 再累加，消掉 float32 的累积舍入
        v = extract(v0.astype("float64"), tlat, tlon)

        # 量级检查：研究区 5-8 月露点大致 -15 ~ 35 degC
        if not (-25 < float(v.min()) and float(v.max()) < 45):
            raise ValueError(
                f"{f.name} 数值范围 {float(v.min()):.2f} ~ {float(v.max()):.2f}，"
                "不像摄氏度露点。若源数据是 K（约 260-300），说明少了 shift=-273.15")

        for name, (lr, sr) in REGIONS.items():
            sub = v.sel(latitude=slice(*lr), longitude=slice(*sr))
            w = np.cos(np.deg2rad(sub.latitude.astype("float64")))
            acc[name].append(sub.weighted(sub.notnull() * w).mean(["latitude", "longitude"]))
        ds.close()

    print("  数据源统计:", dict(sources), f" 年份 {min(years)}-{max(years)} 共 {len(years)} 年")
    out = {}
    for name in REGIONS:
        s = xr.concat(acc[name], dim="time").sortby("time").rename(name)
        s.attrs["units"] = "degC"
        out[name] = s
    return out


def climatology(series, years=None):
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
    """季节内第几天：5/1 = 1 … 8/31 = 123。

    不能用 dayofyear：闰年 3/1 之后的 dayofyear 整体 +1（1984-05-01 是 122，
    平年是 121），会让同一日历日在闰年/平年错配到不同的气候态上，还会凭空
    多出一个只有闰年样本的假"日历日"。实测使目标 H 偏差最大 0.13 sigma。
    """
    t = pd.DatetimeIndex(series["time"].values)
    d = ((t - pd.to_datetime(t.year.astype(str) + "-05-01")).days + 1).values
    assert d.min() == 1, "季节日编号必须从 1 开始"
    return d


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
    zs.attrs["units"] = "1 (z-score of D2m anomaly)"
    xr.Dataset({"d2m_index": zs}).to_netcdf(OUT_DIR / "d2m_indices.nc")

    an = stack(anomaly_dict)
    an.attrs["units"] = "degC"
    xr.Dataset({"d2m_anom": an}).to_netcdf(OUT_DIR / "d2m_anomaly.nc")

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
            "index_mean_formula": "mean over grid cells of sum(D*cos(lat))/sum(cos(lat))",
            "source": "data/raw/d2m/d2m_YYYY.nc (Planette ERA5, 0.25 deg, 1981-2025)",
            "unit": "degC",
            "zscore_formula": "(index - clim) / sigma",
        },
    ).to_netcdf(OUT_DIR / "d2m_norm_params.nc")


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
