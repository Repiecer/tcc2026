from common import Factor, run
FACTOR = Factor(
    key="d2m",
    out_var="dew_point_2m",
    outdir="d2m",
    source="planette",
    units="degC",
    box="yrb",
    long_name="2 m dew point temperature, daily mean",
    group="single",
    pm_var="td2m",
    shift=-273.15,
    extra_attrs={
        "origin_variable": "ERA5 td2m (2 metre dewpoint temperature)",
        "role": "H6 干燥度因子",
        "raw_units": "K",
        "data_source": "Planette ERA5 Archive (AWS us-east-2, anonymous)",
    },
)

if __name__ == "__main__":
    raise SystemExit(run(FACTOR))
