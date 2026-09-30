from common import Factor, run

FACTOR = Factor(
    key="t850",
    out_var="temperature_850",
    outdir="t850",
    source="planette",
    units="K",
    box="asia",
    long_name="850 hPa air temperature",
    group="pressure",
    pm_var="t",
    level=850,
    extra_attrs={"level_hPa": 850, "origin_variable": "t (temperature)"},
)

if __name__ == "__main__":
    raise SystemExit(run(FACTOR))