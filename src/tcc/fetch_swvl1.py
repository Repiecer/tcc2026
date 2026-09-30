from common import Factor, run

FACTOR = Factor(
    key="swvl1",
    out_var="soil_moisture_0_7cm",
    outdir="swvl1",
    source="planette",
    units="m3 m-3",
    box="yrb",
    long_name="Volumetric soil water, layer 1 (0-7 cm), daily mean",
    group="single",
    pm_var="swv_1",
    extra_attrs={
        "origin_variable": "ERA5 swvl1 (swv_1)",
        "role": "H2 陆面记忆核心因子",
        "data_source": "Planette ERA5 Archive (AWS us-east-2, anonymous)",
    },
)

if __name__ == "__main__":
    raise SystemExit(run(FACTOR))
