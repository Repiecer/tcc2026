"""L2 线性模型 + 分组消融实验（A—E）。

消融实验（ablation）= **把因子分组逐层加进去，看每一层多解释了多少方差**。
分组的依据是**物理过程**，且**事先定好**，不是看了结果再挑的 —— 所以结论无偏。

为什么不能一次把 34 个因子全塞进去
    1. 训练样本只有 442-544 个，因子 34 个 → 严重过拟合；
    2. 因子之间高度共线（swvl1/2/3、t850 的四列、sst 的五列），
       系数互相打架，说不清谁在起作用。

分组（逐层累加，都在持续性基线之上）
---------------------------------------------------------------
    0 持续性      H_pre7
    A 热力与湿度   + D2m、T850（4 列）
    B 陆面干湿     + swvl1 / swv2 / swv3 / TP
    C 地表能量     + OLR
    D 环流         + Z500(4) / Z200(4) / MSLP(3) / U850(4) / V850(3)
    E 海温         + SST(5 海区)

另外两组按**物理时间尺度**预定义（不是数据驱动）：
    快变量组  TP / OLR / U850 切变   —— 天气尺度，几十小时
    慢变量组  SWV3 / SST 暖池        —— 陆面记忆以周计、海洋以月计

模型（规划书的 L2）
    OLS   普通最小二乘，无超参数，作基准
    Ridge 岭回归，专治共线性；alpha 由**训练集内部** 5 折 CV 选，不碰验证集

⚠️ 纪律：默认**不评估测试集**（铁律 L5）。要看得加 --include-test。

用法：
    MPLCONFIGDIR=logs .venv/bin/python scripts/run_ablation.py
    MPLCONFIGDIR=logs .venv/bin/python scripts/run_ablation.py --boot 500

产出：
    data/proc/ablation_results.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV

import metrics

PROC = Path("data/proc")
LEADS = (1, 2, 3, 4)
META = ["issue_date", "feature_start", "feature_end", "target_start", "target_end",
        "lead_week", "year", "issue_block", "split", "Y", "Y_raw"]

# ---------------------------------------------------------------- 因子分组
# 每个因子列**只能属于一个组**，脚本会断言全部 34 列都被覆盖且不重复。

BASE = ["H_pre7"]
G_A = ["d2m_yrb_pre7",
       "t850_yrb_pre7", "t850_north_pre7", "t850_south_pre7", "t850_grad_ns_pre7"]
G_B = ["swvl1_yrb_pre7", "swv2_yrb_pre7", "swv3_yrb_pre7", "tp_yrb_sum7"]
G_C = ["olr_yrb_pre7"]
G_D = ["z500_wpsh_pre7", "z500_yrb_pre7", "z500_scs_pre7", "z500_midlat_pre7",
       "z200_sah_pre7", "z200_ea_pre7", "z200_yrb_pre7", "z200_midlat_pre7",
       "mslp_yrb_pre7", "mslp_north_pre7", "mslp_south_pre7",
       "u850_yrb_pre7", "u850_north_pre7", "u850_south_pre7", "u850_shear_ns_pre7",
       "v850_yrb_pre7", "v850_north_pre7", "v850_south_pre7"]
G_E = ["sst_nino34_pre7", "sst_wpwp_pre7", "sst_iod_w_pre7", "sst_iod_e_pre7",
       "sst_scs_pre7"]

# 逐层累加
CUMULATIVE = [
    ("0 持续性(基线)", BASE),
    ("A 热力与湿度", BASE + G_A),
    ("B +陆面干湿", BASE + G_A + G_B),
    ("C +地表能量", BASE + G_A + G_B + G_C),
    ("D +环流", BASE + G_A + G_B + G_C + G_D),
    ("E +海温", BASE + G_A + G_B + G_C + G_D + G_E),
]

# 按物理时间尺度预定义（不是数据驱动）
PHYSICAL = [
    ("快变量组(天气尺度)", BASE + ["tp_yrb_sum7", "olr_yrb_pre7", "u850_shear_ns_pre7"]),
    ("慢变量组(陆面+海洋)", BASE + ["swv3_yrb_pre7", "sst_wpwp_pre7"]),
    ("快+慢组合", BASE + ["tp_yrb_sum7", "olr_yrb_pre7", "u850_shear_ns_pre7",
                          "swv3_yrb_pre7", "sst_wpwp_pre7"]),
]

MODELS = {
    "OLS": lambda: LinearRegression(),
    "Ridge": lambda: RidgeCV(alphas=np.logspace(-2, 3, 24), cv=5),
}


def load(lead):
    d = pd.read_csv(PROC / f"samples_lead{lead}.csv")
    return d


def check_groups(feats):
    """断言分组完整覆盖所有因子列，且不重复。"""
    all_grouped = set(BASE)
    for _, cols in [("A", G_A), ("B", G_B), ("C", G_C), ("D", G_D), ("E", G_E)]:
        all_grouped |= set(cols)
    missing = sorted(set(feats) - all_grouped)
    extra = sorted(all_grouped - set(feats))
    if missing or extra:
        raise SystemExit(f"分组与样本表不一致！未分组: {missing}；多余: {extra}")


def fit_predict(model_name, Xtr, ytr, Xte):
    m = MODELS[model_name]()
    m.fit(Xtr, ytr)
    return m.predict(Xte), m


def run_one(df, cols, lead, split_ev, model_name, n_boot, seed):
    """拟合 + 在 split_ev 上评估 + 年分块 bootstrap 置信区间。"""
    tr = df[df.split == "train"]
    ev = df[df.split == split_ev]
    Xtr, ytr = tr[cols].values, tr.Y.values
    Xev, yev = ev[cols].values, ev.Y.values

    pred, fitted = fit_predict(model_name, Xtr, ytr, Xev)
    point = metrics.evaluate(yev, pred)
    if sl := getattr(fitted, "alpha_", None):
        point["alpha"] = float(sl)

    if len(ev.year.unique()) < 3:
        return point, {k: (np.nan, np.nan) for k in metrics.CI_KEYS}

    # 双层 bootstrap：训练年份重采样拟合，验证年份重采样评估
    rng = np.random.default_rng(seed)
    ty, vy = tr.year.values, ev.year.values
    tpos = {u: np.where(ty == u)[0] for u in np.unique(ty)}
    vpos = {u: np.where(vy == u)[0] for u in np.unique(vy)}
    ut, uv = np.unique(ty), np.unique(vy)
    draws = {k: [] for k in metrics.CI_KEYS}
    for _ in range(n_boot):
        it = np.concatenate([tpos[u] for u in rng.choice(ut, len(ut), replace=True)])
        iv = np.concatenate([vpos[u] for u in rng.choice(uv, len(uv), replace=True)])
        p, _ = fit_predict(model_name, Xtr[it], ytr[it], Xev[iv])
        mm = metrics.evaluate(yev[iv], p)
        for k in metrics.CI_KEYS:
            draws[k].append(mm[k])
    lo, hi = 2.5, 97.5
    ci = {k: (float(np.nanpercentile(v, lo)), float(np.nanpercentile(v, hi)))
          for k, v in draws.items()}
    return point, ci


def main():
    ap = argparse.ArgumentParser(description="L2 线性 + 分组消融")
    ap.add_argument("--include-test", action="store_true",
                    help="同时评估测试集（默认关闭，遵守铁律 L5）")
    ap.add_argument("--boot", type=int, default=300, help="bootstrap 次数")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not args.include_test:
        print("⚠️  未评估测试集（2021-2025）。这是刻意的：铁律 L5 要求")
        print("    测试集在最终模型冻结后只用一次。\n")

    df1 = load(1)
    feats = [c for c in df1.columns if c not in META]
    check_groups(feats)
    print(f"因子列 {len(feats)} 个，分组完整覆盖 ✅\n")

    splits = ["val"] + (["test"] if args.include_test else [])
    rows = []

    for lead in LEADS:
        df = load(lead)
        ntr = int((df.split == "train").sum())
        nev = int((df.split == "val").sum())
        print("=" * 108)
        print(f"提前 {lead} 周   train n={ntr}   val n={nev}")
        print("=" * 108)

        for model_name in MODELS:
            print(f"\n--- {model_name} ---")
            print("%-22s %4s %8s %8s %8s %8s %8s %9s %9s" % (
                "因子组", "n列", "R", "R2", "RMSE", "MAE", "sigma比",
                "SS_vs_L0", "SS_vs_持续"))
            print("-" * 108)

            # L0 气候态（预测 0）的 MSE 就是 mean(Y²)
            clim_mse = {s: float(np.mean(df[df.split == s].Y.values ** 2))
                        for s in splits}
            ref = {}
            for group_name, cols in CUMULATIVE + PHYSICAL:
                for split_ev in splits:
                    m, ci = run_one(df, cols, lead, split_ev, model_name,
                                    args.boot, args.seed)
                    if group_name == CUMULATIVE[0][0]:
                        ref[split_ev] = m
                    ss0 = 1 - m["RMSE"] ** 2 / clim_mse[split_ev]
                    ssp = (1 - m["RMSE"] ** 2 / ref[split_ev]["RMSE"] ** 2
                           if split_ev in ref else float("nan"))
                    tag = "" if split_ev == "val" else " [TEST]"
                    print("%-22s %4d %8.3f %8.4f %8.4f %8.4f %8.3f %+9.4f %+9.4f" % (
                        group_name + tag, len(cols),
                        m["R"] if np.isfinite(m["R"]) else float("nan"),
                        m["R2"], m["RMSE"], m["MAE"], m["sigma_ratio"], ss0, ssp))
                    rows.append(dict(lead_week=lead, split=split_ev, model=model_name,
                                     group=group_name, n_features=len(cols), **m,
                                     SS_vs_L0=ss0, SS_vs_persist=ssp,
                                     R_lo=ci["R"][0], R_hi=ci["R"][1],
                                     R2_lo=ci["R2"][0], R2_hi=ci["R2"][1],
                                     RMSE_lo=ci["RMSE"][0], RMSE_hi=ci["RMSE"][1]))
            print()

    out = pd.DataFrame(rows)
    out.to_csv(PROC / "ablation_results.csv", index=False)

    # 汇总：每层相对上一层的增量
    print("=" * 108)
    print("消融增量（每一层相对上一层多解释的方差）")
    print("=" * 108)
    print("%-22s" % "因子组" + "".join("%14s" % f"lead {L} 周" for L in LEADS))
    print("-" * 108)
    chain = ["0 持续性(基线)"] + [n for n, _ in CUMULATIVE[1:]]
    for i, name in enumerate(chain):
        line = "%-22s" % name
        for L in LEADS:
            r = out[(out.lead_week == L) & (out.model == "OLS") & (out.split == "val")]
            cur = r[r.group == name]["R2"]
            if i == 0 or cur.empty:
                line += "%14s" % ("—" if i == 0 else "n/a")
            else:
                prev = r[r.group == chain[i - 1]]["R2"]
                line += "%+14.4f" % (float(cur.iloc[0]) - float(prev.iloc[0]))
        print(line)
    print()
    print("已保存 data/proc/ablation_results.csv")
    print("\n怎么用：")
    print("  1. 增量最大的一层 = 该类物理过程信息量最大；增量为负说明该组只是噪声；")
    print("  2. 对比 [TEST] 行需要 --include-test，且只能看一次（铁律 L5）；")
    print("  3. 后面 L3 随机森林 / L4 XGBoost 复用同一套分组与指标，结果可直接对比。")


if __name__ == "__main__":
    main()
