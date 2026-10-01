from common import Factor, run

FACTOR = Factor(
    key="olr",
    out_var="outgoing_longwave_radiation",
    outdir="olr",
    source="planette",
    units="W m-2",
    box="yrb",
    long_name="Outgoing longwave radiation at top of atmosphere, daily mean (upward positive)",
    group="single",
    pm_var="olr",
    # Planette 的 `olr` 实测为负值（-322 ~ -128 W/m2），说明它存的是
    # ERA5 ttr（Top net thermal radiation，**净向下为正**），
    # 而不是标准定义的 OLR（向上为正）。乘 -1 换成真正的 OLR。
    scale=-1.0,
    extra_attrs={
        "origin_variable": "ERA5 ttr (top net thermal radiation)",
        "role": "H4 地表能量因子的替代量（替代拿不到的 ssrd）",
        "rationale": "晴空 -> 短波辐射 ssrd 高、向外长波辐射 OLR 也高；"
                     "有云/对流 -> 反之。OLR 是云量与对流的经典代理量",
        "sign_convention": "**向上为正**（已把 Planette 的 ttr 乘 -1）",
        "verified": "实测 1993（湿凉年 H=-0.62）域平均 220.5 W/m2；"
                    "2022（干热年 H=+0.79）242.9 W/m2。越热 -> OLR 越大 -> 天越晴。"
                    "与 H 的日相关 -0.242 / -0.445（对 ttr）已翻为正值",
        "data_source": "Planette ERA5 Archive (AWS us-east-2, anonymous)",
    },
)

if __name__ == "__main__":
    raise SystemExit(run(FACTOR))
