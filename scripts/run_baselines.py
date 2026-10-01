"""L0 气候态 / L1 持续性 基线评估。

基线的作用是**给出参照系**：后面的 L2 线性、L3 随机森林、L4 XGBoost
必须跨过这条线才有意义。

两个基线
--------
L0 气候态   yhat = 0
            含义："我什么都不知道，就赌它和常年同期一样"。
            意义：给出 **0 分线**（R² 在训练期恰好为 0）。

L1 持续性   yhat = H_pre7（发布前一周的平均高温指数）
            含义："上周热，这周大概也热"。
            意义：**真正的门槛**。跨不过它就说明模型没学到东西。

⚠️ L1 必须报两个版本
--------------------
未拟合：yhat = H_pre7                    —— 教科书定义的"持续性预报"
拟合  ：yhat = a + b * H_pre7            —— 让线性回归自己定斜率和截距

实测提前 1 周的斜率 b ≈ 0.47，远小于 1（回归稀释：上周偏热 1σ
并不代表这周也偏热 1σ，只有 0.47σ）。因此未拟合版本的 R² 是**负数**
（比直接报平均值还差），而后续所有模型都会做拟合。

=> **模型要跨过的线是"拟合持续性"**，不是未拟合版本。
   只报未拟合基线会制造"模型轻松大胜"的假象。

用法：
    MPLCONFIGDIR=logs .venv/bin/python scripts/run_baselines.py
    MPLCONFIGDIR=logs .venv/bin/python scripts/run_baselines.py --include-test
        （默认**不含测试集**，遵守铁律 L5：测试集只用一次）

产出：
    data/proc/baseline_results.csv    长格式（每行 = 提前期 × 时段 × 方案），19 列
    data/proc/baseline_summary.csv    可读版汇总（每行 = 提前期 × 时段），一行看全
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import metrics

PROC = Path("data/proc")
LEADS = (1, 2, 3, 4)


def make_predictions(sub):
    """返回 {方案名: (真实值, 预测值)}。"""
    y = sub.Y.values
    x = sub.H_pre7.values

    out = {}
    out["L0 气候态"] = (y, np.zeros_like(y))
    out["L1 持续性(未拟合)"] = (y, x)

    lr = LinearRegression().fit(x.reshape(-1, 1), y)
    out["L1 持续性(拟合)"] = (y, lr.predict(x.reshape(-1, 1)))
    return out, lr


def main():
    ap = argparse.ArgumentParser(description="L0 / L1 基线评估")
    ap.add_argument("--include-test", action="store_true",
                    help="同时评估测试集（默认关闭，遵守铁律 L5）")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap 次数")
    args = ap.parse_args()

    splits = ["train", "val"] + (["test"] if args.include_test else [])
    if not args.include_test:
        print("⚠️  未评估测试集（2021-2025）。这是刻意的：铁律 L5 要求")
        print("    测试集在最终模型冻结后只用一次。要现在看请加 --include-test。\n")

    all_rows = []
    slopes = {}
    for lead in LEADS:
        df = pd.read_csv(PROC / f"samples_lead{lead}.csv")
        print("=" * 92)
        print(f"提前 {lead} 周   目标 = 未来第 {lead} 周的周平均 H")
        print("=" * 92)
        print("%-18s %-6s %5s %8s %9s %7s %7s %7s %8s %8s" % (
            "方案", "时段", "n", "R", "R2", "RMSE", "MAE", "Bias", "sigma比", "SS_vs_L0"))
        print("-" * 92)

        for split in splits:
            sub = df[df.split == split]
            preds, lr = make_predictions(sub)
            ref_mse = None
            for name, (y, yhat) in preds.items():
                m = metrics.evaluate(y, yhat)
                if name.startswith("L0"):
                    ref_mse = m["RMSE"] ** 2
                ss = metrics.skill_score(m["RMSE"] ** 2, ref_mse)
                ci = metrics.bootstrap_ci(y, yhat, sub.year.values, n_boot=args.boot)
                print("%-18s %-6s %5d %8s %9.4f %7.4f %7.4f %+7.4f %8.4f %+8.4f" % (
                    name, split, m["n"],
                    "  --  " if not np.isfinite(m["R"]) else "%+.3f" % m["R"],
                    m["R2"], m["RMSE"], m["MAE"], m["Bias"], m["sigma_ratio"], ss))
                if split in ("val", "test"):
                    print("%-18s %-6s %5s %8s %9s %7s %7s %7s %8s %8s" % (
                        "", "└ 95%CI(R)", "", metrics.format_ci(ci["R"]), "", "", "", "", "", ""))
                all_rows.append((name, split, lead, m, ci, ss))

            if split == "train":
                print("%-18s %-6s %5s  拟合斜率 b = %.3f（< 1 说明持续性信号被稀释）"
                      % ("", "", "", lr.coef_[0]))
            slopes[(lead, split)] = float(lr.coef_[0])
        print()

    out = metrics.results_frame(all_rows)
    out.to_csv(PROC / "baseline_results.csv", index=False)

    # ---- 可读版汇总表：一行 = 一个「提前期 × 时段」，一眼看全 ----
    lk = {(lead, split, name): (m, ci, ss)
          for name, split, lead, m, ci, ss in all_rows}
    rows2 = []
    for lead in LEADS:
        for split in splits:
            m0 = lk[(lead, split, "L0 气候态")][0]
            mu, ciu, ssu = lk[(lead, split, "L1 持续性(未拟合)")]
            mf, cif, ssf = lk[(lead, split, "L1 持续性(拟合)")]
            rows2.append({
                "lead_week": lead, "split": split, "n": m0["n"],
                "L0_R2": round(m0["R2"], 4), "L0_RMSE": round(m0["RMSE"], 4),
                "L0_MAE": round(m0["MAE"], 4),
                "持续性_R_未拟合": round(mu["R"], 4),
                "持续性_R_拟合": round(mf["R"], 4),
                "R_95CI_下": round(cif["R"][0], 4), "R_95CI_上": round(cif["R"][1], 4),
                "持续性_R2_未拟合": round(mu["R2"], 4),
                "持续性_R2_拟合": round(mf["R2"], 4),
                "持续性_RMSE": round(mf["RMSE"], 4),
                "持续性_MAE": round(mf["MAE"], 4),
                "持续性_Bias": round(mf["Bias"], 4),
                "sigma比": round(mf["sigma_ratio"], 4),
                "SS_vs_L0": round(ssf, 4),
                "SS_vs_L0_未拟合": round(ssu, 4),
                "拟合斜率b": round(slopes[(lead, split)], 4),
            })
    summ = pd.DataFrame(rows2)
    summ.to_csv(PROC / "baseline_summary.csv", index=False)

    print("=" * 92)
    print("可读版汇总（每行 = 一个「提前期 × 时段」）：")
    print("=" * 92)
    cols = ["lead_week", "split", "n", "L0_R2", "持续性_R_拟合", "R_95CI_下", "R_95CI_上",
            "持续性_R2_未拟合", "持续性_R2_拟合", "持续性_RMSE", "SS_vs_L0", "拟合斜率b"]
    print(summ[cols].to_string(index=False))
    print()
    print("已保存 data/proc/baseline_results.csv     （长格式，19 列，24 行）")
    print("已保存 data/proc/baseline_summary.csv     （可读版，一行看全）")
    print("\n怎么用这张表：")
    print("  1. L0 的 R² 在训练期=0、在验证期<0（验证期偏暖，报常年平均有冷偏差）；")
    print("  2. L1 的 R 随提前期迅速衰减 —— 这是延伸期预测的客观难度；")
    print("  3. 后面每个模型都必须和【L1 持续性(拟合)】比，而不是和 L0 或未拟合版本比。")


if __name__ == "__main__":
    main()
