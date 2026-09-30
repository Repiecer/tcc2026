from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

RAW_DIR = Path("data/raw/sst")
OUT_DIR = Path("data/proc")

REGIONS = {
    "nino34": ((-5, 5), (190, 240)),
    "wpwp":   ((-5, 5), (120, 160)),
    "iod_w":  ((-10, 10), (50, 70)),
    "iod_e":  ((-10, 0), (90, 110)),
    "scs":    ((5, 20), (105, 120)),
}
TRAIN = range(1982, 2016)
WINDOW = 11
FLOOR = 0.5


def index_series():
    files = sorted(RAW_DIR.glob("sst_daily_*.nc"))
    if not files:
        raise FileNotFoundError(f"no file: {RAW_DIR}/sst_daily_*.nc")

    acc = {k: [] for k in REGIONS}
    for f in files:
        ds = xr.open_dataset(f)
        v = ds["sea_surface_temperature_mean"].astype("float64")
        lat, lon = np.asarray(ds.latitude), np.asarray(ds.longitude)
        ocean = ((v > 100) & (v < 320)).any("time")

        for name, (lr, sr) in REGIONS.items():
            sub = v.sel(latitude=slice(*lr), longitude=slice(*sr))
            w = np.cos(np.deg2rad(sub.latitude))
            acc[name].append(sub.weighted(sub.notnull() * w).mean(["latitude", "longitude"]))

    out = {}
    for name in REGIONS:
        s = xr.concat(acc[name], dim="time").sortby("time").rename(name)
        s.attrs["units"] = "K"
        out[name] = s
    return out


def climatology(series, years=None):
    y = np.asarray(series["time"].dt.year)
    train = np.isin(y, list(TRAIN))
    doy = doy_of(series)

    table = pd.DataFrame({"doy": doy[train], "T": series.values[train]})
    per_day = table.groupby("doy")["T"].agg(
        n="size",
        total="sum",
        sq_total=lambda x: (x ** 2).sum(),
    )
    window = per_day.rolling(WINDOW, center=True, min_periods=1).sum()
    clim = (window["total"] / window["n"]).to_numpy()
    variance = np.clip(window["sq_total"] / window["n"] - clim ** 2, 0, None)
    sigma = np.sqrt(variance.to_numpy())
    return per_day.index.values, clim, sigma


def doy_of(series):
    doy = np.asarray(series["time"].dt.dayofyear)
    return np.where(doy == 366, 365, doy)


def zscore(series, doy_list, clim, sigma):
    pos = doy_of(series) - doy_list[0]
    sigma = np.maximum(sigma, FLOOR * sigma.mean())
    return xr.DataArray((series.values - clim[pos]) / sigma[pos], dims="time", coords={"time": series["time"]}, name=series.name)


def anomaly(series, doy_list, clim):
    pos = doy_of(series) - doy_list[0]
    return xr.DataArray(series.values - clim[pos], dims="time", coords={"time": series["time"]}, name=series.name)

def stack(d):
    out = xr.concat([d[k] for k in REGIONS if k in d], dim="index")
    return out.assign_coords(index=[k for k in REGIONS if k in d])

def save(named, anomaly_dict, clim_dict):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    zs = stack(named)
    zs.attrs["units"] = "1 (z-score of SST anomaly)"
    xr.Dataset({"sst_index": zs}).to_netcdf(OUT_DIR / "sst_indices.nc")

    an = stack(anomaly_dict)
    an.attrs["units"] = "K"
    xr.Dataset({"sst_anom": an}).to_netcdf(OUT_DIR / "sst_anomaly.nc")

    names = list(named)
    clim_ds = xr.Dataset(
        {"clim": (("index", "doy"), [clim_dict[n][0] for n in names]),
         "sigma": (("index", "doy"), [clim_dict[n][1] for n in names])},
        coords={"index": names, "doy": clim_dict[names[0]][2]},
    )
    clim_ds.attrs = {
        "clim_years": f"{TRAIN[0]}-{TRAIN[-1]}",
        "window_days": f"{WINDOW}",
        "sigma_floor_ratio": f"{FLOOR}",
        "regions": str({k: v for k, v in REGIONS.items()}),
        "index_mean_formula": "mean over sea cells of sum(T*cos(lat))/sum(cos(lat))",
        "source": "data/raw/sst/sst_daily_YYYY.nc NOAA OISST v2.1 1deg subset",
        "unit": "K",
        "zscore_formula": "(index - clim) / sigma",
        "note": "start from 1982",
    }
    clim_ds.to_netcdf(OUT_DIR / "sst_norm_params.nc")


def main():
    series_dict = index_series()
    named, anomaly_dict, clim_dict = {}, {}, {}
    for name, s in series_dict.items():
        doy_list, clim, sigma = climatology(s)
        named[name] = zscore(s, doy_list, clim, sigma)
        anomaly_dict[name] = anomaly(s, doy_list, clim)
        clim_dict[name] = (clim, sigma, doy_list)
    save(named, anomaly_dict, clim_dict)
    print(f"\nsave at {OUT_DIR}/")


if __name__ == "__main__":
    main()
