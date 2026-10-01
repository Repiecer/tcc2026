"""生成《样本表因子列说明》文档。

把 `data/proc/samples_lead{1..4}.csv` 里的 34 个因子列**逐个解释**，
并把实测预测力（与 Y 的相关、叠在持续性基线之上的增量 R²、bootstrap 置信区间）
直接写进文档，方便建模时对照。

为什么做成"生成"而不是手写
    因子列会随样本表变化。手写的说明文档一定会慢慢过期。
    本脚本从 CSV **实际读出列名**，若出现未登记的列会直接报错，
    保证文档与数据永远一致。

用法：
    MPLCONFIGDIR=logs .venv/bin/python scripts/write_factor_guide.py

产出：
    docs/样本表因子列说明.md
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

PROC = Path("data/proc")
OUT = Path("docs/样本表因子列说明.md")
LEADS = (1, 2, 3, 4)
N_BOOT = 200
SEED = 0

META = ["issue_date", "feature_start", "feature_end", "target_start", "target_end",
        "lead_week", "year", "issue_block", "split", "Y", "Y_raw"]

# ---------------------------------------------------------------- 区域说明

REGIONS = {
    "yrb": "长江中下游研究区（28–34°N, 110–122°E；气压层变量为 25–35°N）",
    "north": "研究区北半区（31–34°N, 110–122°E）",
    "south": "研究区南半区（28–31°N, 110–122°E）",
    "wpsh": "西太副高主体（20–35°N, 120–150°E）",
    "scs": "副高西脊点附近（15–30°N, 105–125°E）",
    "midlat": "中高纬（40–55°N, 100–130°E）—— 阻塞高压/西风带扰动",
    "sah": "南亚高压（25–40°N, 60–105°E）—— 夏季对流层上部最强系统",
    "ea": "东亚高层（20–45°N, 100–140°E）—— 副热带西风急流与脊区",
    "nino34": "中东太平洋（5°S–5°N, 190–240°E）—— ENSO",
    "wpwp": "西太平洋暖池（5°S–5°N, 120–160°E）",
    "iod_w": "印度洋偶极子西极（10°S–10°N, 50–70°E）",
    "iod_e": "印度洋偶极子东极（10°S–0°, 90–110°E）",
    "grad_ns": "南北温差（= north − south，派生）",
    "shear_ns": "南北 U 差（= north − south，低层反气旋性指数）",
}

# ---------------------------------------------------------------- 逐列解释

COL = {
    "H_pre7": ("高温指数", "—", "发布日前 7 天的平均 H。**这就是持续性基线**——「上周有多热」。"
                              "本身已扣除季节循环，代表偏热程度而非绝对温度。"),
    # 陆面干湿
    "swvl1_yrb_pre7": ("表层土壤湿度", "yrb", "0–7 cm 体积含水量距平。**负 = 偏干**。记忆仅 1–3 天（一场雨就变）。"),
    "swv2_yrb_pre7": ("第 2 层土壤湿度", "yrb", "7–28 cm。记忆 1–2 周，介于 swv1 与 swv3 之间。"),
    "swv3_yrb_pre7": ("第 3 层土壤湿度", "yrb", "28–100 cm（**植物根区**）。记忆数周–数月。"
                                             "⭐ 全项目唯一在提前 2/4 周显著的陆面因子。"),
    "tp_yrb_sum7": ("降水", "yrb", "前 7 天**累计降水**（开方后的 z 分数之和）。**负 = 干旱**。"
                                  "唯一用 `_sum7` 聚合的列。"),
    # 热力与湿度
    "d2m_yrb_pre7": ("2 米露点", "yrb", "近地面水汽条件。**正 = 潮湿**。"
                                       "实测与高温是**正相关**（暖湿同源），不是规划书预期的负相关。"),
    "t850_yrb_pre7": ("850 hPa 气温", "yrb", "约 1500 m 高度的气团温度。代表**气团本身的冷暖**，"
                                            "不受局地下垫面影响。与地面气温年际相关 0.93，冗余度高。"),
    "t850_north_pre7": ("850 hPa 气温", "north", "北半区气团温度。"),
    "t850_south_pre7": ("850 hPa 气温", "south", "南半区气团温度。"),
    "t850_grad_ns_pre7": ("850 hPa 南北温差", "grad_ns", "北−南。实测热夏年**北方升温更多**，"
                                                       "故与 H 是**正**相关（与「梯度弱=均匀高温」的直觉相反）。"),
    # 地表能量
    "olr_yrb_pre7": ("向外长波辐射", "yrb", "**晴空 → OLR 大；有云/深对流 → OLR 小**。"
                                           "是云量与对流的经典代理量，替代拿不到的 ssrd。"),
    # 中层/高层环流
    "z500_wpsh_pre7": ("500 hPa 位势高度", "wpsh", "副高主体强度。偏高 = 副高偏强。"),
    "z500_yrb_pre7": ("500 hPa 位势高度", "yrb", "研究区正上方的中层高度。"),
    "z500_scs_pre7": ("500 hPa 位势高度", "scs", "副高西脊点附近，反映西伸程度。"),
    "z500_midlat_pre7": ("500 hPa 位势高度", "midlat", "中高纬，反映阻塞高压/西风带扰动。"),
    "z200_sah_pre7": ("200 hPa 位势高度", "sah", "南亚高压（青藏高压）强度。"),
    "z200_ea_pre7": ("200 hPa 位势高度", "ea", "东亚高层环流。"),
    "z200_yrb_pre7": ("200 hPa 位势高度", "yrb", "研究区正上方的高层高度。"),
    "z200_midlat_pre7": ("200 hPa 位势高度", "midlat", "中高纬高层，西风带扰动与波列。"),
    # 气压与低层风
    "mslp_yrb_pre7": ("海平面气压", "yrb", "研究区气压距平。⚠️ 同期与 H 是**负相关**（反向因果："
                                          "高温造成热低压），单因子相关≈0，但**增量显著**——信息是非线性/空间型的。"),
    "mslp_north_pre7": ("海平面气压", "north", "北半区气压。"),
    "mslp_south_pre7": ("海平面气压", "south", "南半区气压。"),
    "u850_yrb_pre7": ("850 hPa 纬向风", "yrb", "区域平均东西向风（东向为正）。裸平均**几乎没用**。"),
    "u850_north_pre7": ("850 hPa 纬向风", "north", "北半区纬向风。"),
    "u850_south_pre7": ("850 hPa 纬向风", "south", "南半区纬向风。"),
    "u850_shear_ns_pre7": ("850 hPa 经向切变", "shear_ns", "**低层反气旋性指数**：`u_north − u_south > 0` "
                                                          "⇔ 研究区受反气旋控制。比裸区域平均 u 的增量大 **60 倍**。"),
    "v850_yrb_pre7": ("850 hPa 经向风", "yrb", "区域平均南北向风（北向为正）。**正 = 南风异常 = 暖湿输送**。"),
    "v850_north_pre7": ("850 hPa 经向风", "north", "北半区经向风。"),
    "v850_south_pre7": ("850 hPa 经向风", "south", "南半区经向风。"),
    # 海温
    "sst_nino34_pre7": ("海表温度", "nino34", "中东太平洋（ENSO）。⚠️ 单独与长江高温**不显著**（年际 p=0.64）；"
                                              "要反映 ENSO 应改用 `nino34 − wpwp` 的东西梯度。"),
    "sst_wpwp_pre7": ("海表温度", "wpwp", "西太平洋暖池。⭐ **四个提前期全部显著**，"
                                          "且增量随提前期**递增**（lead 4 达 +0.046）—— 慢变量。"),
    "sst_iod_w_pre7": ("海表温度", "iod_w", "印度洋偶极子西极。"),
    "sst_iod_e_pre7": ("海表温度", "iod_e", "印度洋偶极子东极。⚠️ 两极大都正相关，**相减成 IOD 反而抵消信号**。"),
    "sst_scs_pre7": ("海表温度", "scs", "南海海温。离研究区最近的海区。"),
}

# ---------------------------------------------------------------- 分组

FAMILIES = [
    ("持续性与陆面（消融 A、B 组）",
     "这条链回答规划书的 **H1—H3**：热的持续性、以及「前期偏干 → 后期偏热」。"
     "实测里降水与土壤湿度是全项目最强的两个物理量。",
     ["H_pre7", "swvl1_yrb_pre7", "swv2_yrb_pre7", "swv3_yrb_pre7", "tp_yrb_sum7"]),

    ("热力与近地面湿度（A 组）",
     "T850 代表**约 1500 m 高度上的气团温度**，不受局地下垫面影响；"
     "D2m 代表近地面水汽。两者与地面气温高度共线，主要价值在机制诊断。",
     ["d2m_yrb_pre7", "t850_yrb_pre7", "t850_north_pre7", "t850_south_pre7",
      "t850_grad_ns_pre7"]),

    ("地表能量（C 组）",
     "原计划的 SSRD（短波辐射）与 SLHTF（潜热通量）**都拿不到**，"
     "改用 OLR 代替。短波与长波回答的是同一个问题：**「天有多晴」**。",
     ["olr_yrb_pre7"]),

    ("中层与高层环流（D 组）",
     "500 hPa 看副高与中高纬波列；200 hPa 看南亚高压与高层急流。"
     "**注意：这些区域平均量的同期相关很强（7–8 月 z500_yrb 达 +0.52），"
     "但超前 1 周就衰减到 +0.09 以下。**",
     ["z500_wpsh_pre7", "z500_yrb_pre7", "z500_scs_pre7", "z500_midlat_pre7",
      "z200_sah_pre7", "z200_ea_pre7", "z200_yrb_pre7", "z200_midlat_pre7"]),

    ("气压与低层风（D 组）",
     "MSLP 与 H 是**反向因果**（高温 → 热低压），但增量显著，必须单独评估。"
     "风是矢量，**裸区域平均几乎没用**，必须构造有物理意义的标量（切变）。",
     ["mslp_yrb_pre7", "mslp_north_pre7", "mslp_south_pre7",
      "u850_yrb_pre7", "u850_north_pre7", "u850_south_pre7", "u850_shear_ns_pre7",
      "v850_yrb_pre7", "v850_north_pre7", "v850_south_pre7"]),

    ("海温（E 组）",
     "海温是**远程强迫因子**（研究区内全是陆地，没有 SST）。"
     "实测只有**西太暖池**真正有效，且它在四个提前期全部显著。",
     ["sst_nino34_pre7", "sst_wpwp_pre7", "sst_iod_w_pre7", "sst_iod_e_pre7",
      "sst_scs_pre7"]),
]


def compute_stats():
    """每个 (列, 提前期) 的实测预测力。"""
    rng = np.random.default_rng(SEED)
    stats, base = {}, {}
    for L in LEADS:
        d = pd.read_csv(PROC / f"samples_lead{L}.csv")
        d = d[d.split.isin(["train", "val"])]
        feats = [c for c in d.columns if c not in META]
        y = d.Y.values
        h = d.H_pre7.values.reshape(-1, 1)
        r2p = LinearRegression().fit(h, y).score(h, y)
        base[L] = r2p
        yrs = d.year.values
        uy = np.unique(yrs)
        pos = {u: np.where(yrs == u)[0] for u in uy}
        for c in feats:
            x = d[c].values
            X = np.column_stack([d.H_pre7.values, x])
            inc = LinearRegression().fit(X, y).score(X, y) - r2p
            draws = []
            for _ in range(N_BOOT):
                i = np.concatenate([pos[u] for u in rng.choice(uy, len(uy), replace=True)])
                draws.append(LinearRegression().fit(X[i], y[i]).score(X[i], y[i])
                             - LinearRegression().fit(d.H_pre7.values[i].reshape(-1, 1),
                                                      y[i]).score(
                                 d.H_pre7.values[i].reshape(-1, 1), y[i]))
            stats[(c, L)] = dict(r=float(np.corrcoef(x, y)[0, 1]), inc=float(inc),
                                 lo=float(np.percentile(draws, 2.5)))
    return stats, base


def mark(stats, c, L):
    s = stats[(c, L)]
    return "%+.4f%s" % (s["inc"], " ✅" if s["lo"] > 0.001 else ("  ~" if s["lo"] > 0 else ""))


def main():
    stats, base = compute_stats()
    cols = list(COL)

    # 一致性检查：CSV 里出现的因子列必须都在 COL 里登记
    lead1 = pd.read_csv(PROC / "samples_lead1.csv")
    actual = [c for c in lead1.columns if c not in META]
    missing = [c for c in actual if c not in COL]
    extra = [c for c in COL if c not in actual]
    if missing or extra:
        raise SystemExit(f"因子列与说明表不一致！未登记: {missing}；多余: {extra}")

    doc = []
    A = doc.append
    A("# 样本表因子列说明\n")
    A("> 本文档由 `scripts/write_factor_guide.py` **自动生成**（从 CSV 实读列名，")
    A("> 若出现未登记的因子列会直接报错，保证文档与数据一致）。\n")
    A("> 对象：`data/proc/samples_lead{1,2,3,4}.csv` 的 **34 个因子列**。")
    A("> 数据字典见 [`data_dictionary_samples.md`](data_dictionary_samples.md)。\n")
    A("---\n")

    A("## 一、怎么读列名\n")
    A("```")
    A("{变量}_{指数}_{聚合方式}")
    A("```\n")
    A("| 后缀 | 含义 |")
    A("|---|---|")
    A("| `_pre7` | 发布日**前 7 天**取**平均**（`[issue_date−6, issue_date]`）|")
    A("| `_sum7` | 发布日**前 7 天**取**累计**（只有 `tp_yrb_sum7`，降水是累积量）|\n")
    A("⚠️ **所有因子列都是标准化指数（z 分数），不是物理量。**")
    A("因为 K、hPa、m/s、m³/m³、gpm 五种量纲没法直接比系数大小。\n")
    A("⚠️ **z 分数不能相减**：`z(north) − z(south) ≠ z(north − south)`，")
    A("因为各自的 σ 不同。派生指数（`grad_ns`、`shear_ns`）是**单独标准化**出来的。\n")

    A("---\n")
    A("## 二、区域名称对照\n")
    A("| 代码 | 区域 |")
    A("|---|---|")
    for k in ["yrb", "north", "south", "wpsh", "scs", "midlat", "sah", "ea",
              "nino34", "wpwp", "iod_w", "iod_e", "grad_ns", "shear_ns"]:
        A(f"| `{k}` | {REGIONS[k]} |")
    A("")
    A("> ⚠️ `scs` 在 `z500`（15–30°N, 105–125°E）与 `sst`（5–20°N, 105–120°E）里")
    A("> 是**两个不同的区域**，靠列名前缀区分。\n")

    A("---\n")
    A("## 三、34 个因子列逐个说明\n")
    A("表头含义：`r` = 与目标 Y 的相关系数；`增量 L1/L2/L4` = 叠在持续性基线之上")
    A("对 R² 的增量（提前 1/2/4 周，train+val）；`✅` = bootstrap 95% 置信区间下界 > 0.001。\n")
    for title, intro, names in FAMILIES:
        A(f"### {title}\n")
        A(intro + "\n")
        A("| 列名 | 变量 | 区域 | 物理含义 | r | 增量 L1 | 增量 L2 | 增量 L4 |")
        A("|---|---|---|---|---|---|---|---|")
        for c in names:
            var, reg, desc = COL[c]
            A(f"| `{c}` | {var} | `{reg}` | {desc} | {stats[(c,1)]['r']:+.3f} | "
              f"{mark(stats,c,1)} | {mark(stats,c,2)} | {mark(stats,c,4)} |")
        A("")

    A("---\n")
    A("## 四、实测预测力总表（按提前 1 周增量排序）\n")
    A(f"持续性基线 R²：提前 1 周 **{base[1]:.4f}**、2 周 {base[2]:.4f}、"
      f"3 周 {base[3]:.4f}、4 周 {base[4]:.4f}。\n")
    A("| 排名 | 列名 | r(与 Y) | 增量 L1 | 增量 L2 | 增量 L3 | 增量 L4 |")
    A("|---|---|---|---|---|---|---|")
    order = sorted(actual, key=lambda c: -stats[(c, 1)]["inc"])
    for i, c in enumerate(order, 1):
        A(f"| {i} | `{c}` | {stats[(c,1)]['r']:+.3f} | {mark(stats,c,1)} | "
          f"{mark(stats,c,2)} | {mark(stats,c,3)} | {mark(stats,c,4)} |")
    A("")
    A("> `✅` = 95% CI 下界 > 0.001（明确显著）；`~` = 下界贴在 0（勉强）。\n")

    A("---\n")
    A("## 五、⭐ 核心规律：快变量与慢变量的时间尺度分工\n")
    A("把各提前期**显著**（`✅`）的因子列出来：\n")
    for L in LEADS:
        sig = [(c, stats[(c, L)]) for c in actual if stats[(c, L)]["lo"] > 0.001]
        sig.sort(key=lambda t: -t[1]["inc"])
        if not sig:
            A(f"**提前 {L} 周**（基线 R² = {base[L]:.4f}）：无显著因子。\n")
            continue
        A(f"**提前 {L} 周**（基线 R² = {base[L]:.4f}）：\n")
        A("| 列名 | 增量 R² | CI 下界 |")
        A("|---|---|---|")
        for c, s in sig:
            A(f"| `{c}` | {s['inc']:+.4f} | {s['lo']:+.4f} |")
        A("")
    A("### 结论\n")
    A("| 类型 | 代表因子 | 有效提前期 | 特点 |")
    A("|---|---|---|---|")
    A("| **天气快变量** | `tp_yrb_sum7`、`olr_yrb_pre7`、`u850_shear_ns_pre7` | **仅提前 1 周** | "
      "增量大（+0.041 ~ +0.049），但提前 2 周就归零 |")
    A("| **陆面慢变量** | `swv3_yrb_pre7` | **提前 2、4 周** | 提前 1 周不起眼，延伸期才发力 |")
    A("| **海洋慢变量** | `sst_wpwp_pre7` | **四个提前期全部** | ⭐ 增量随提前期**递增**，4 周达 +0.046 |")
    A("")
    A("> **物理原因**：大气的热惯性只有几天，**土壤记忆以周计、海洋记忆以月计**。")
    A("> 延伸期预测能靠的只有慢变量。")
    A("> 这解释了为什么单纯堆环流因子做不出延伸期技巧 —— ")
    A("> 环流因子虽然同期相关强（z500_yrb 在 7–8 月达 +0.52），但提前 1 周就衰减到 +0.09 以下。\n")
    A("> **建模启示**：四个提前期**不应共用同一套因子**。")
    A("> 提前 1 周用快变量组；提前 2–4 周以慢变量组（`swv3` + `sst_wpwp`）为核心。\n")

    A("---\n")
    A("## 六、读表时的四个陷阱\n")
    A("| 陷阱 | 说明 | 例子 |")
    A("|---|---|---|")
    A("| **单因子相关 ≠ 有用** | 两个方向都有反例 | `mslp_yrb` 的 r≈0 但增量 +0.020 ✅；"
      "`t850_north` 的 r=+0.30 但增量只有 +0.001 |")
    A("| **矢量的区域平均丢信息** | 必须先构造有物理意义的标量 | `u850_yrb` 增量 +0.0007，"
      "同样数据换成 `u850_shear_ns` 后 **+0.0413** |")
    A("| **高度共线的列不要全塞** | 会互相抢贡献度 | `swvl1/swv2/swv3`、`t850_yrb/north/south`、"
      "`sst_*` 五列 |")
    A("| **增量 R² 都很小** | 最好的也只把 R² 从 0.11 提到 0.16 | 这是延伸期预测的客观难度，不是模型不行 |")
    A("")
    A("---\n")
    A("## 七、相关文件\n")
    A("| 文件 | 内容 |")
    A("|---|---|")
    A("| [`data_dictionary_samples.md`](data_dictionary_samples.md) | 样本表结构、P0 冻结项、防泄漏校验 |")
    A("| [`data_dictionary_swv3.md`](data_dictionary_swv3.md) | 深层土壤湿度（延伸期关键因子）|")
    A("| [`data_dictionary_olr.md`](data_dictionary_olr.md) | 向外长波辐射（替代 ssrd）|")
    A("| [`项目规划书/11_实测结论与方案修订.md`](项目规划书/11_实测结论与方案修订.md) | 假设裁决与方案修订 |")

    OUT.write_text("\n".join(doc) + "\n", encoding="utf-8")
    print(f"已保存 {OUT}（{len(actual)} 个因子列）")


if __name__ == "__main__":
    main()
