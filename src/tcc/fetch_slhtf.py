"""⚠️ 本脚本目前**无法使用**：Open-Meteo 不提供潜热通量。

实测结论（2026-09-30 与 2026-10-01，archive-api.open-meteo.com）：
    models=era5_land + latent_heat_flux  -> 48/48 全为 null
    models=era5      + latent_heat_flux  -> 48/48 全为 null
两个模型都不提供。Planette 的单层变量清单里也没有（只有 olr / tswv 等）。

保留本文件的目的
    1. 记录"这条路走不通"的结论，避免后人重复踩坑；
    2. 一旦换成 CDS（ERA5 官方，变量名 `slhf`），可直接改 source 复用。

H5 假设（潜热偏少 → 更多能量转为感热 → 升温）现在怎么检验？
    改用两个**替代量**，机制覆盖同一件事：
      · `swv1/2/3`（土壤湿度）：潜热通量的**主控状态量**——土壤越干，蒸发越弱；
      · `tp`（降水）：水分的**输入量**。
    实测结果见 `docs/data_dictionary_swv3.md` 与 `docs/data_dictionary_tp.md`：
    深层土壤湿度（swv3）在提前 2-4 周的增量 R² 显著为正，是全项目唯一在
    延伸期有效的慢变量 —— 这条链的机制证据实际上已经从替代量拿到了。

若要真正下载，请改用 CDS：
    pip install cdsapi
    变量：reanalysis-era5-single-levels / surface_latent_heat_flux (slhf)
"""

from common import Factor, run

FACTOR = Factor(
    key="slhtf",
    out_var="surface_latent_heat_flux",
    outdir="slhtf",
    source="openmeteo",
    units="W m-2",
    box="yrb",
    long_name="Surface latent heat flux, daily mean (upward positive)",
    om_var="latent_heat_flux",
    om_agg="mean",
    om_models="era5_land",
    extra_attrs={
        "origin_variable": "ERA5-Land slhf",
        "role": "H5 陆气相互作用因子",
        "sign_convention": "upward positive (ERA5 原生约定)",
        "agg": "hourly mean -> daily mean (W m-2)",
        "status": "❌ 不可用：Open-Meteo 的 era5 与 era5_land 都不提供 latent_heat_flux",
    },
)

if __name__ == "__main__":
    raise SystemExit(run(FACTOR))
