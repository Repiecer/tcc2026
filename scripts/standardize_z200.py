from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

RAW_DIR = Path("data/raw/z200")
OUT_DIR = Path("data/proc")
REGIONS = {
    # 南亚高压（青藏高压）主体：夏季对流层上部最强大的系统
    "sah":     ((25, 40), (60, 105)),
    # 东亚高层：覆盖副热带西风急流与东亚上空脊区
    "ea":      ((20, 45), (100, 140)),
    # 研究区正上方（与 z500 的 yrb 对应，便于对比中层/高层）
    "yrb":     ((25, 35), (110, 125)),
    # 中高纬：西风带扰动与波列
    "midlat":  ((40, 55), (100, 130)),
}
TRAIN = range(1981, 2016)
WINDOW = 11
FLOOR = 0.5

def index_series():
    files = sorted(RAW_DIR.glob("z200_*.nc"))
    if not files:
        raise FileNotFoundError(f"no file: {RAW_DIR}/z200_*.nc")

    acc = {k: [] for k in REGIONS}
    for f in files:
        ds = xr.open_dataset(f)
        v = ds["geopotential_height_200"].astype("float64")

        for name, (lr, sr) in REGIONS.items():
            sub = v.sel(latitude=slice(*lr), longitude=slice(*sr))
            w = np.cos(np.deg2rad(sub.latitude.astype("float64")))
            acc[name].append(sub.weighted(sub.notnull() * w).mean(["latitude", "longitude"]))
        ds.close()

    out = {}
    for name in REGIONS:
        s = xr.concat(acc[name], dim="time").sortby("time").rename(name)
        s.attrs["units"] = "gpm"
        out[name] = s
    return out


def climatology(series, years=None):
    y = np.asarray(series["time"].dt.year)
    train = np.isin(y, list(TRAIN))
    doy = doy_of(series)

    table = pd.DataFrame({"doy": doy[train], "T": series.values[train]})
    per_day = table.groupby("doy")["T"].agg(n="size", total="sum", sq_total=lambda x: (x ** 2).sum())
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
    return xr.DataArray((series.values - clim[pos]) / sigma[pos],dims="time", coords={"time": series["time"]}, name=series.name)


def anomaly(series, doy_list, clim):
    pos = doy_of(series) - doy_list[0]
    return xr.DataArray(series.values - clim[pos],dims="time", coords={"time": series["time"]}, name=series.name)

def stack(d):
    keys = [k for k in REGIONS if k in d]
    return xr.concat([d[k] for k in keys], dim="index").assign_coords(index=keys)


def save(named, anomaly_dict, clim_dict):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    zs = stack(named)
    zs.attrs["units"] = "1 (z-score of Z200 anomaly)"
    xr.Dataset({"z200_index": zs}).to_netcdf(OUT_DIR / "z200_indices.nc")

    an = stack(anomaly_dict)
    an.attrs["units"] = "gpm"
    xr.Dataset({"z200_anom": an}).to_netcdf(OUT_DIR / "z200_anomaly.nc")

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
            "index_mean_formula": "mean over grid cells of sum(H*cos(lat))/sum(cos(lat))",
            "source": "data/raw/z200/z200_YYYY.nc (ERA5 200hPa geopotential height, gpm)",
            "unit": "gpm",
            "zscore_formula": "(index - clim) / sigma",
            "note": "Z200 南北梯度约 2700 gpm，年际信号仅约 19 gpm，故必须按子区域成指数",
        },
    ).to_netcdf(OUT_DIR / "z200_norm_params.nc")
def main():
    series_dict = index_series()
    named, anomaly_dict, clim_dict = {}, {}, {}
    for name, s in series_dict.items():
        doy_list, clim, sigma = climatology(s)
        named[name] = zscore(s, doy_list, clim, sigma)
        anomaly_dict[name] = anomaly(s, doy_list, clim)
        clim_dict[name] = (clim, sigma, doy_list)
    save(named, anomaly_dict, clim_dict)
    print(f"\n已保存到 {OUT_DIR}/")


if __name__ == "__main__":
    main()
