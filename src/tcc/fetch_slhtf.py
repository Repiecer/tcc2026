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
    },
)

if __name__ == "__main__":
    raise SystemExit(run(FACTOR))
