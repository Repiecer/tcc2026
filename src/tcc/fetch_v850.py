from common import Factor, run

FACTOR = Factor(
    key="v850",
    out_var="v_component_of_wind_850",
    outdir="v850",
    source="planette",
    units="m s-1",
    box="asia",
    long_name="850 hPa v-component of wind (northward positive)",
    group="pressure",
    pm_var="v",
    level=850,
    extra_attrs={
        "level_hPa": 850,
        "origin_variable": "v (v_component_of_wind)",
        "role": "H9 低层风场因子",
        "sign_convention": "northward positive (ERA5 原生约定)",
    },
)

if __name__ == "__main__":
    raise SystemExit(run(FACTOR))
