from common import Factor, run

FACTOR = Factor(
    key="ssrd",
    out_var="surface_solar_radiation_downwards",
    outdir="ssrd",
    source="openmeteo",
    units="W m-2",
    box="yrb",
    long_name="Surface solar radiation downwards, daily mean",
    om_var="shortwave_radiation",
    om_agg="mean",
    om_models="era5_land",
    extra_attrs={
        "origin_variable": "ERA5-Land ssrd",
        "role": "H4 地表能量输入因子",
        "agg": "hourly mean -> daily mean (W m-2)",
    },
)

if __name__ == "__main__":
    raise SystemExit(run(FACTOR))
