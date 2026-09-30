from common import Factor, run

FACTOR = Factor(
    key="u850",
    out_var="u_component_of_wind_850",
    outdir="u850",
    source="planette",
    units="m s-1",
    box="asia",
    long_name="850 hPa u-component of wind (eastward positive)",
    group="pressure",
    pm_var="u",
    level=850,
    extra_attrs={
        "level_hPa": 850,
        "origin_variable": "u (u_component_of_wind)",
        "role": "H9 低层风场因子",
        "sign_convention": "eastward positive (ERA5 原生约定)",
    },
)

if __name__ == "__main__":
    raise SystemExit(run(FACTOR))
