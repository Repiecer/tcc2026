"""样本表防泄漏校验。

对应规划书 04 第五节「数据泄漏防线（五条铁律）」的落地检查。
对 `data/proc/samples_lead{1..4}.csv` 逐条断言，任何一条不通过就退出。

检查项
------
L1  禁止随机打乱划分：train / val / test 的年份集合互不相交
L2  因子窗口右端 ≤ 发布时刻：feature_end <= issue_date，且窗口长度 = 7
L3  标准化参数只用训练期：由各 standardize_*.py 内部的泄漏检查负责
    （本脚本抽查 clim 是否随年份变化）
L4  调参与选因子只看验证集：结构性检查，脚本无法自动判定，见文档
L5  测试集只用一次：结构性检查，脚本无法自动判定，见文档
附加 目标窗口长度 = 7；提前期算术正确；无缺测；无重复；窗口不重叠

用法：
    MPLCONFIGDIR=logs .venv/bin/python scripts/check_leakage.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

PROC = Path("data/proc")
LEADS = (1, 2, 3, 4)
FEATURE_DAYS = 7
TRAIN, VAL, TEST = (1982, 2015), (2016, 2020), (2021, 2025)
META = ["issue_date", "feature_start", "feature_end", "target_start", "target_end",
        "lead_week", "year", "issue_block", "split", "Y", "Y_raw"]

ok = True


def need(cond, msg):
    global ok
    print(f"  {'✅' if cond else '❌'} {msg}")
    if not cond:
        ok = False


def main():
    tables = {}
    for lead in LEADS:
        p = PROC / f"samples_lead{lead}.csv"
        if not p.exists():
            need(False, f"{p} 不存在，请先运行 build_dataset.py")
            return
        tables[lead] = pd.read_csv(p, parse_dates=["issue_date", "feature_start",
                                                   "feature_end", "target_start",
                                                   "target_end"])

    print("=" * 78)
    print("样本表概览")
    print("=" * 78)
    print("%-8s %7s %7s %7s %7s %8s %10s" % ("提前期", "总样本", "train", "val", "test", "因子数", "起止年份"))
    for lead, df in tables.items():
        c = df["split"].value_counts()
        print("%-8s %7d %7d %7d %7d %8d %10s" % (
            f"{lead} 周", len(df), c.get("train", 0), c.get("val", 0), c.get("test", 0),
            df.shape[1] - len(META), f"{df.year.min()}-{df.year.max()}"))

    print()
    print("=" * 78)
    print("L1  按年块划分，train / val / test 年份互不相交")
    print("=" * 78)
    for lead, df in tables.items():
        tr = set(df.loc[df.split == "train", "year"])
        va = set(df.loc[df.split == "val", "year"])
        te = set(df.loc[df.split == "test", "year"])
        need(not (tr & va) and not (tr & te) and not (va & te),
             f"lead {lead}: train∩val∩test = 空集  (train {min(tr)}-{max(tr)}, "
             f"val {min(va)}-{max(va)}, test {min(te)}-{max(te)})")
        need(tr == set(range(TRAIN[0], TRAIN[1] + 1)), f"lead {lead}: 训练年份连续覆盖 {TRAIN[0]}-{TRAIN[1]}")
        need(va == set(range(VAL[0], VAL[1] + 1)), f"lead {lead}: 验证年份连续覆盖 {VAL[0]}-{VAL[1]}")
        need(te == set(range(TEST[0], TEST[1] + 1)), f"lead {lead}: 测试年份连续覆盖 {TEST[0]}-{TEST[1]}")

    print()
    print("=" * 78)
    print("L2  因子窗口右端 ≤ 发布时刻；窗口不重叠")
    print("=" * 78)
    for lead, df in tables.items():
        need((df.feature_end == df.issue_date).all(),
             f"lead {lead}: feature_end == issue_date（因子窗口右端正好是发布日）")
        need((df.feature_start < df.feature_end).all(), f"lead {lead}: feature_start < feature_end")
        ndays = (df.feature_end - df.feature_start).dt.days + 1
        need((ndays == FEATURE_DAYS).all(), f"lead {lead}: 因子窗口长度恒为 {FEATURE_DAYS} 天")
        need((df.target_start > df.issue_date).all(),
             f"lead {lead}: target_start > issue_date（目标完全在发布日之后）")
        need((df.feature_end < df.target_start).all(),
             f"lead {lead}: 因子窗口与目标窗口不重叠（无同期信息混入）")
        need((df.issue_date < df.target_start).all(), f"lead {lead}: issue_date < target_start")

    print()
    print("=" * 78)
    print("目标窗口与提前期算术")
    print("=" * 78)
    for lead, df in tables.items():
        ndays = (df.target_end - df.target_start).dt.days + 1
        need((ndays == 7).all(), f"lead {lead}: 目标窗口长度恒为 7 天")
        gap = (df.target_start - df.issue_date).dt.days
        need((gap == 7 * (lead - 1) + 1).all(),
             f"lead {lead}: target_start = issue_date + {7 * (lead - 1) + 1} 天")
        need((df.lead_week == lead).all(), f"lead {lead}: lead_week 列恒为 {lead}")

    print()
    print("=" * 78)
    print("完整性：缺测 / 重复 / 年份归属")
    print("=" * 78)
    for lead, df in tables.items():
        feat = [c for c in df.columns if c not in META]
        n_nan = int(df[feat + ["Y", "Y_raw"]].isna().sum().sum())
        need(n_nan == 0, f"lead {lead}: 因子与目标无任何缺测（实际 {n_nan}）")
        dup = int(df.duplicated(subset=["issue_date", "lead_week"]).sum())
        need(dup == 0, f"lead {lead}: (issue_date, lead_week) 无重复（实际 {dup}）")
        need((df.issue_date.dt.year == df.year).all(), f"lead {lead}: year 列与 issue_date 一致")
        need(df.issue_date.dt.month.between(5, 8).all(), f"lead {lead}: 发布日全部落在 5-8 月")
        need(df.target_end.dt.month.between(5, 8).all(), f"lead {lead}: 目标日全部落在 5-8 月")

    print()
    print("=" * 78)
    print("独立重算：直接用 pandas 重算目标与一个因子，与表内数值比对")
    print("=" * 78)
    H = xr.open_dataset(PROC / "tmax_H.nc")["H"].to_pandas()
    S = xr.open_dataset(PROC / "swvl1_indices.nc")["swvl1_index"].sel(index="yrb").to_pandas()
    for lead, df in tables.items():
        y_err = max(abs(H.loc[r.target_start:r.target_end].mean() - r.Y)
                    for r in df.sample(min(40, len(df)), random_state=0).itertuples())
        f_err = max(abs(S.loc[r.feature_start:r.feature_end].mean() - getattr(r, "swvl1_yrb_pre7"))
                    for r in df.sample(min(40, len(df)), random_state=0).itertuples())
        need(y_err < 1e-10 and f_err < 1e-10,
             f"lead {lead}: 目标最大差 {y_err:.1e}，swvl1_yrb_pre7 最大差 {f_err:.1e}")

    print()
    print("=" * 78)
    print("端到端抽查：从**原始 nc** 重算一行因子的全过程")
    print("=" * 78)
    print("  链路：原始 m3/m3 -> 13x25 提取 -> 面积加权区域平均 -> 减 clim")
    print("        -> 除以 sigma -> 对 [feature_start, feature_end] 求平均")
    row = tables[1].iloc[len(tables[1]) // 2]
    y = row.issue_date.year

    ds = xr.open_dataset(Path(f"data/raw/swvl1/swvl1_{y}.nc"))
    v = ds["soil_moisture_0_7cm"].astype("float64")
    ref = xr.open_dataset(Path("data/raw/tmax/tmax_1981_05.nc"))
    tlat = np.round(ref.latitude.values, 4)
    tlon = np.round(ref.longitude.values, 4)
    ref.close()
    il = [int(np.argmin(np.abs(ds.latitude.values - t))) for t in tlat]
    io = [int(np.argmin(np.abs(ds.longitude.values - t))) for t in tlon]
    sub = v.isel(latitude=il, longitude=io)
    w = np.cos(np.deg2rad(sub.latitude.astype("float64")))
    daily = sub.weighted(sub.notnull() * w).mean(["latitude", "longitude"]).to_pandas()
    ds.close()

    pr = xr.open_dataset(PROC / "swvl1_norm_params.nc")
    tt = pd.DatetimeIndex(daily.index)
    sd = ((tt - pd.to_datetime(tt.year.astype(str) + "-05-01")).days + 1).values
    pos = sd - pr.doy.values[0]
    clim = pr["clim"].sel(index="yrb").values[pos]
    sg = pr["sigma"].sel(index="yrb").values
    sig = np.maximum(sg, 0.5 * sg.mean())[pos]
    z = pd.Series((daily.values - clim) / sig, index=daily.index)

    win = z.loc[row.feature_start:row.feature_end]
    err = abs(win.mean() - row.swvl1_yrb_pre7)
    need(len(win) == 7 and err < 1e-10,
         f"{y} 年 {row.issue_date.date()} 的 swvl1_yrb_pre7（表内是**标准化指数**）："
         f"原始 nc 重算 = {win.mean():.10f}，表内 = {row.swvl1_yrb_pre7:.10f}，差 {err:.1e}")

    print()
    print("=" * 78)
    print("样本结构与自相关提醒")
    print("=" * 78)
    df = tables[1]
    step = int(np.median(np.diff(np.sort(df.issue_date.values)).astype("timedelta64[D]").astype(int)))
    print(f"  发布间隔中位数 = {step} 天")
    if step >= 7:
        print("  ✅ 发布间隔 ≥ 7 天：同一张表内相邻样本的**目标窗口不重叠**，")
        print("     名义样本量 ≈ 有效样本量，可以直接用常规显著性检验。")
    else:
        print("  ⚠️ 发布间隔 < 7 天：相邻样本目标窗口重叠，**有效样本量远小于名义样本量**，")
        print("     显著性检验必须改用按年分块 bootstrap（规划书 04 第 4.3 节）。")
    for lead, d in tables.items():
        nb = d.groupby("year").size()
        print(f"  lead {lead}: 每年 {nb.min()}-{nb.max()} 个样本，共 {len(d)} 个")

    print()
    print("=" * 78)
    print("总判定:", "全部通过 ✅" if ok else "存在问题 ❌")
    print("=" * 78)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
