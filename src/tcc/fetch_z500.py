from common import GPM_PER_M2S2, Factor, run

FACTOR = Factor(
    key="z500",
    out_var="geopotential_height_500",
    outdir="z500",
    source="planette",
    units="gpm",
    box="asia",
    long_name="500 hPa geopotential height",
    group="pressure",
    pm_var="z",
    level=500,
    scale=GPM_PER_M2S2,
    extra_attrs={"level_hPa": 500, "origin_variable": "z (geopotential)"},
)

if __name__ == "__main__":
    raise SystemExit(run(FACTOR))