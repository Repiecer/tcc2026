from common import GPM_PER_M2S2, Factor, run

FACTOR = Factor(
    key="z200",
    out_var="geopotential_height_200",
    outdir="z200",
    source="planette",
    units="gpm",
    box="asia",
    long_name="200 hPa geopotential height",
    group="pressure",
    pm_var="z",
    level=200,
    scale=GPM_PER_M2S2,
    extra_attrs={"level_hPa": 200, "origin_variable": "z (geopotential)"},
)

if __name__ == "__main__":
    raise SystemExit(run(FACTOR))