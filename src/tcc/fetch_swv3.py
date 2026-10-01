from common import Factor, run

FACTOR = Factor(
    key="swv3",
    out_var="soil_moisture_28_100cm",
    outdir="swv3",
    source="planette",
    units="m3 m-3",
    box="yrb",
    long_name="Volumetric soil water, layer 3 (28-100 cm), daily mean",
    group="single",
    pm_var="swv_3",
    extra_attrs={
        "origin_variable": "ERA5 swvl3 (swv_3)",
        "role": "H2 陆面记忆因子（根区，延伸期预测最看重的慢变量）",
        "layer_depth": "28-100 cm（植物根区）",
        "memory": "数周至数月",
        "data_source": "Planette ERA5 Archive (AWS us-east-2, anonymous)",
    },
)

if __name__ == "__main__":
    raise SystemExit(run(FACTOR))
