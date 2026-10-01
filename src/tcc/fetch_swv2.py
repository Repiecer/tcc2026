from common import Factor, run

FACTOR = Factor(
    key="swv2",
    out_var="soil_moisture_7_28cm",
    outdir="swv2",
    source="planette",
    units="m3 m-3",
    box="yrb",
    long_name="Volumetric soil water, layer 2 (7-28 cm), daily mean",
    group="single",
    pm_var="swv_2",
    extra_attrs={
        "origin_variable": "ERA5 swvl2 (swv_2)",
        "role": "H2 陆面记忆因子（比 swv_1 更深、记忆更长）",
        "layer_depth": "7-28 cm",
        "memory": "约 1-2 周（swv_1 只有 1-3 天）",
        "data_source": "Planette ERA5 Archive (AWS us-east-2, anonymous)",
    },
)

if __name__ == "__main__":
    raise SystemExit(run(FACTOR))
