from common import Factor, run

FACTOR = Factor(
    key="tp",
    out_var="precipitation_sum",
    outdir="tp",
    source="planette",
    units="mm",
    box="yrb",
    long_name="Daily accumulated precipitation",
    group="single",
    pm_var="pr",
    scale=86400.0,                 # kg m-2 s-1 -> mm/day
    extra_attrs={
        "origin_variable": "ERA5 pr (mean total precipitation rate)",
        "role": "H3 陆面湿度因子",
        "raw_units": "kg m-2 s-1",
        "agg": "daily rate x 86400 = 24h accumulation (mm)",
        "data_source": "Planette ERA5 Archive (AWS us-east-2, anonymous)",
    },
)

if __name__ == "__main__":
    raise SystemExit(run(FACTOR))
