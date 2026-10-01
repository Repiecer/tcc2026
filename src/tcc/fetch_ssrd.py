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
    # ⚠️ 必须是 era5，不能用 era5_land。
    # 实测（2026-09-30，archive-api.open-meteo.com）：
    #   models=era5_land + shortwave_radiation -> 48/48 全为 null（静默失败）
    #   models=era5      + shortwave_radiation -> 48/48 有值（288、439 W/m2）
    # 旧版本误用 era5_land，导致 14 个文件全是 NaN。
    om_models="era5",
    extra_attrs={
        "origin_variable": "ERA5 ssrd",
        "role": "H4 地表能量输入因子",
        "agg": "hourly mean -> daily mean (W m-2)",
        "note": "Open-Meteo 的 ERA5-Land 不提供辐射变量，必须用 models=era5",
    },
)

if __name__ == "__main__":
    raise SystemExit(run(FACTOR))
