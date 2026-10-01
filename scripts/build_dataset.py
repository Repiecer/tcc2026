"""构建机器学习样本表（提前期 1-4 周，四张表）。

每一行 = 一个「发布日 + 提前期」样本：

    issue_date            发布时刻 t0（只能用这一天及以前的信息）
    feature_start/end     因子聚合窗口 = [t0-6, t0]，右端严格 ≤ t0
    target_start/end      目标窗口 = [t0+7(k-1)+1, t0+7k]
    Y                     目标窗口的平均高温指数 H
    Y_raw                 目标窗口的平均区域 Tmax（degC，供物理解释）
    其余列                各因子在 [t0-6, t0] 内的平均（TP 用累计）

================================ P0 冻结项 ================================
1. **目标定义 T2：周平均 H**（规划书 04 第三节推荐的主目标），窗口固定 7 天。
2. **提前期 k = 1,2,3,4 周**，四张独立表。
3. **发布间隔 7 天**（`ISSUE_STEP`），即不重叠的周。
   这样相邻样本不会共享目标窗口，避免"有效样本量远小于名义样本量"
   （规划书 04 第 4.3 节的提醒）。改成 1 即得"每天发布"的操作版本，
   但那样必须用分块 bootstrap 做显著性检验。
4. **因子窗口固定 7 天，右端 = issue_date**（防泄漏铁律 L2）。
5. **样本年份 1982-2025**：SST 从 1982 年才开始，为了让所有列都不留缺测，
   1981 年不参与建表。
6. **按年块划分**（铁律 L1，禁止随机打乱）：
   训练 1982-2015 / 验证 2016-2020 / 测试 2021-2025。
   测试集只用一次（铁律 L5）。

用法：
    MPLCONFIGDIR=logs .venv/bin/python scripts/build_dataset.py
    MPLCONFIGDIR=logs .venv/bin/python scripts/build_dataset.py --step 1   # 每天发布

产出：
    data/proc/samples_lead{1,2,3,4}.csv
    校验：MPLCONFIGDIR=logs .venv/bin/python scripts/check_leakage.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

PROC = Path("data/proc")

LEADS = (1, 2, 3, 4)
FEATURE_DAYS = 7          # 因子聚合窗口长度
FIRST_YEAR = 1982         # SST 从 1982 起
LAST_YEAR = 2025
SEASON_MONTH, SEASON_DAY = 5, 1
SEASON_END_MONTH, SEASON_END_DAY = 8, 31

TRAIN = (1982, 2015)
VAL = (2016, 2020)
TEST = (2021, 2025)

# 需要标准化的因子指数文件（key -> data/proc/{key}_indices.nc，变量 {key}_index）
FACTOR_KEYS = ["swvl1", "swv2", "swv3", "tp", "d2m", "olr", "mslp",
               "z500", "z200", "sst", "t850", "u850", "v850"]
# 累积量用「求和」，其余用「平均」
SUM_COLS = {"tp_yrb"}


def season_start(t):
    return pd.Timestamp(year=t.year, month=SEASON_MONTH, day=SEASON_DAY)


def season_end(t):
    return pd.Timestamp(year=t.year, month=SEASON_END_MONTH, day=SEASON_END_DAY)


def split_of(y):
    if TRAIN[0] <= y <= TRAIN[1]:
        return "train"
    if VAL[0] <= y <= VAL[1]:
        return "val"
    if TEST[0] <= y <= TEST[1]:
        return "test"
    return "out_of_range"


def load_axis():
    """目标时间轴 = 5-8 月、1982-2025（5412 天）。"""
    H = xr.open_dataset(PROC / "tmax_H.nc")["H"].to_pandas()
    R = xr.open_dataset(PROC / "tmax_region.nc")["tmax_region"].to_pandas()
    axis = H.index[(H.index.year >= FIRST_YEAR) & (H.index.year <= LAST_YEAR)]
    return H.reindex(axis), R.reindex(axis), axis


def load_features(axis):
    """把所有因子指数铺成一张 (time, 因子) 的表。"""
    feats = {}
    for key in FACTOR_KEYS:
        da = xr.open_dataset(PROC / f"{key}_indices.nc")[f"{key}_index"]
        for idx in da.index.values:
            name = f"{key}_{idx}"
            s = da.sel(index=idx).to_pandas().reindex(axis)
            if s.isna().any():
                raise ValueError(f"{name} 在样本时间轴上存在缺测")
            suffix = "sum7" if name in SUM_COLS else "pre7"
            feats[f"{name}_{suffix}"] = s
    wide = pd.DataFrame(feats, index=axis)
    return wide


def build_one(lead, wide, H, R, step):
    """构建一张提前 lead 周的样本表。"""
    rows = []
    for t0 in wide.index:
        off = (t0 - season_start(t0)).days
        if off % step != FEATURE_DAYS - 1:
            continue

        fs = t0 - pd.Timedelta(days=FEATURE_DAYS - 1)
        if fs < season_start(t0):
            continue
        seg = wide.loc[fs:t0]
        if len(seg) != FEATURE_DAYS:
            continue

        ts = t0 + pd.Timedelta(days=7 * (lead - 1) + 1)
        te = t0 + pd.Timedelta(days=7 * lead)
        if te > season_end(t0):
            continue
        tgt = H.loc[ts:te]
        if len(tgt) != 7:
            continue

        row = {
            "issue_date": t0,
            "feature_start": fs,
            "feature_end": t0,
            "target_start": ts,
            "target_end": te,
            "lead_week": lead,
            "year": t0.year,
            "issue_block": off // 7,        # 季节内第几周，供分块交叉验证用
            "split": split_of(t0.year),
            "Y": float(tgt.mean()),
            "Y_raw": float(R.loc[ts:te].mean()),
        }
        for c in wide.columns:
            row[c] = float(seg[c].sum() if c.endswith("_sum7") else seg[c].mean())
        rows.append(row)

    df = pd.DataFrame(rows)
    # 列顺序：元数据 -> 目标 -> 因子
    meta = ["issue_date", "feature_start", "feature_end", "target_start", "target_end",
            "lead_week", "year", "issue_block", "split", "Y", "Y_raw"]
    feats = [c for c in df.columns if c not in meta]
    return df[meta + sorted(feats)]


def main():
    ap = argparse.ArgumentParser(description="构建样本表")
    ap.add_argument("--step", type=int, default=7,
                    help="发布间隔天数（默认 7 = 不重叠周；1 = 每天发布）")
    args = ap.parse_args()

    H, R, axis = load_axis()
    wide = load_features(axis)
    wide["H_pre7"] = H                       # 持续性基线因子
    print(f"因子表: {wide.shape[0]} 天 × {wide.shape[1]} 列（含 H_pre7）")
    print(f"发布间隔: {args.step} 天\n")

    print("%-6s %6s %8s %8s %8s   %s" % ("提前期", "样本", "训练", "验证", "测试", "目标窗口示例"))
    for lead in LEADS:
        df = build_one(lead, wide, H, R, args.step)
        out = PROC / f"samples_lead{lead}.csv"
        df.to_csv(out, index=False)
        c = df["split"].value_counts()
        ex = df.iloc[0]
        print("%-6s %6d %8d %8d %8d   %s -> %s ~ %s" % (
            f"{lead} 周", len(df), c.get("train", 0), c.get("val", 0), c.get("test", 0),
            ex.issue_date.date(), ex.target_start.date(), ex.target_end.date()))
        assert (df["split"] != "out_of_range").all(), "有年份落在划分之外"

    print(f"\n已保存 {len(LEADS)} 张表到 {PROC}/")
    print("下一步：MPLCONFIGDIR=logs .venv/bin/python scripts/check_leakage.py")


if __name__ == "__main__":
    main()
