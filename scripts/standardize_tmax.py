from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
RAW_DIR = Path("data/raw/tmax")
OUT_DIR = Path("data/proc")

LAT = slice(28, 34)
LON = slice(110, 122)
TRAIN = range(1981, 2016)
WINDOW = 11
FLOOR = 0.5


def region_mean():
    files = sorted(RAW_DIR.glob("tmax_*.nc"))
    if not files:
        raise FileNotFoundError(f"no file: {RAW_DIR}/tmax_*.nc")
    ds = xr.open_mfdataset(files, combine="by_coords")
    grid = ds["t2m"].sel(latitude=LAT, longitude=LON) - 273.15

    lat_weight = np.cos(np.deg2rad(grid["latitude"]))
    series = (grid.weighted(grid.notnull() * lat_weight).mean(["latitude", "longitude"]).dropna("time"))

    series = series.rename("tmax_region")
    series.attrs["units"] = "degC"
    return series

def climatology(series, years=None):
    
    years = TRAIN if years is None else years
    is_train = np.isin(np.asarray(series["time"].dt.year), list(years))
    doy = doy_of(series)

    table = pd.DataFrame({"doy": doy[is_train], "T": series.values[is_train]})
    per_day = table.groupby("doy")["T"].agg(
        n="size",
        total="sum",
        sq_total=lambda x: (x ** 2).sum(),
    )
    print(per_day)
    window = per_day.rolling(WINDOW, center=True, min_periods=1).sum()
    clim = (window["total"] / window["n"]).to_numpy()
    variance = np.clip(window["sq_total"] / window["n"] - clim ** 2, 0, None)
    sigma = np.sqrt(variance.to_numpy())
    return per_day.index.values, clim, sigma


def doy_of(series):
    doy = np.asarray(series["time"].dt.dayofyear)
    return np.where(doy == 366, 365, doy)

def heat_index(series, doy_list, clim, sigma):
    pos = doy_of(series) - doy_list[0]
    sigma = np.maximum(sigma, FLOOR * sigma.mean())
    H = (series.values - clim[pos]) / sigma[pos]
    return xr.DataArray(H, dims="time", coords={"time": series["time"]}, name="H")

def save(series, H, doy_list, clim, sigma):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    xr.Dataset({"tmax_region": series}).to_netcdf(OUT_DIR / "tmax_region.nc")
    xr.Dataset({"H": H}).to_netcdf(OUT_DIR / "tmax_H.nc")
    xr.Dataset(
        {"clim": ("doy", clim), "sigma": ("doy", sigma)},
        coords={"doy": doy_list},
        attrs={
            "clim_years": f"{TRAIN[0]}-{TRAIN[-1]}",
            "window_days": f"{WINDOW}",
            "sigma_floor_ratio": f"{FLOOR}",
            "region": "28-34N, 110-122E",
            "region_mean_formula": "sum(T*cos(lat)*valid) / sum(cos(lat)*valid)",
            "source": "data/raw/tmax/tmax_YYYY_MM.nc ERA5",
            "unit": "degC",
            "H_formula": "H = (tmax_region - clim) / sigma",
        },
    ).to_netcdf(OUT_DIR / "tmax_norm_params.nc")

def main():
    series = region_mean()
    print(series.dims)
    doy_list, clim, sigma = climatology(series)
    H = heat_index(series, doy_list, clim, sigma)
    save(series, H, doy_list, clim, sigma)
if __name__ == "__main__":
    main()
