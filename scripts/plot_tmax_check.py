# -*- coding: utf-8 -*-
"""
tmax 标准化结果检查图
=====================

读取 `scripts/standardize_tmax.py` 的产出，画 4 张核对图：
  (1) 5-8 月气候态与波动幅度
  (2) H 时间序列（含各时段均值）
  (3) 7 月平均 H 逐年（看变暖趋势）
  (4) H 分布（看超过 H=2 的天数）

用法：
    python scripts/plot_tmax_check.py

产出：
    docs/tmax_standardization_check.png
"""

from pathlib import Path

import numpy as np
import xarray as xr

import matplotlib
matplotlib.use("Agg")          # 不弹窗，直接存文件
import matplotlib.pyplot as plt

PROC_DIR = Path("data/proc")
OUT_PNG = Path("docs/tmax_standardization_check.png")

TRAIN_END = 2015               # 训练期最后一年


def main():
    region = xr.open_dataset(PROC_DIR / "tmax_region.nc")["tmax_region"]
    H = xr.open_dataset(PROC_DIR / "tmax_H.nc")["H"]
    params = xr.open_dataset(PROC_DIR / "tmax_norm_params.nc")

    # 用月份判断，不用 dayofyear —— 闰年 5-8 月的 dayofyear 整体 +1，
    # 会把闰年的 6/30 混进 7 月、并漏掉闰年的 7/31
    month = np.asarray(region["time"].dt.month)
    year = np.asarray(region["time"].dt.year)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

    # ---- (1) 气候态与波动幅度 ----
    ax = axes[0, 0]
    ax.plot(params.doy, params.clim, "b-", lw=2, label="climatology (clim)")
    ax.fill_between(params.doy, params.clim - params.sigma, params.clim + params.sigma,
                    color="b", alpha=0.15, label="+/- 1 sigma")
    ax.set_xlabel("day of year")
    ax.set_ylabel("Tmax (degC)")
    ax.set_title("(1) Climatology & variability, May-Aug (1981-2015 only)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    twin = ax.twinx()
    twin.plot(params.doy, params.sigma, "r--", lw=1.5)
    twin.set_ylabel("sigma (degC)", color="r")

    # ---- (2) H 时间序列 ----
    ax = axes[0, 1]
    ax.plot(region["time"], H, "-", lw=0.5, color="gray", alpha=0.7, label="daily H")
    periods = [(1981, TRAIN_END, "tab:blue"),
               (TRAIN_END + 1, 2020, "tab:orange"),
               (2021, 2025, "tab:red")]
    for y0, y1, color in periods:
        m = (year >= y0) & (year <= y1)
        ax.hlines(float(H.values[m].mean()),
                  region["time"].values[m][0], region["time"].values[m][-1],
                  color=color, lw=3, label=f"{y0}-{y1}")
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(2, color="r", ls=":", lw=1, label="H=2 (extreme)")
    ax.set_ylabel("H (standardized heat index)")
    ax.set_title("(2) H time series (thick = period mean)")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    ax.grid(alpha=0.3)

    # ---- (3) 7 月平均 H 逐年 ----
    ax = axes[1, 0]
    july = month == 7
    years = list(np.unique(year))
    july_mean = [H.values[(year == y) & july].mean() for y in years]
    colors = ["tab:blue" if y <= TRAIN_END else "tab:red" for y in years]
    ax.bar(years, july_mean, color=colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("year")
    ax.set_ylabel("July mean H")
    ax.set_title("(3) July mean H per year (blue=train, red=test)")
    ax.grid(alpha=0.3, axis="y")

    # ---- (4) H 分布 ----
    ax = axes[1, 1]
    n_extreme = int((H.values > 2).sum())
    ax.hist(H.values, bins=60, color="steelblue", alpha=0.85)
    ax.axvline(2, color="r", ls="--", lw=2, label="H=2 threshold")
    ax.set_xlabel("H")
    ax.set_ylabel("days")
    ax.set_title(f"(4) H distribution: {n_extreme} days with H>2")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, bbox_inches="tight")
    print(f"已保存 {OUT_PNG}")
    print(f"H>2 天数: {n_extreme} / {H.size} ({100 * n_extreme / H.size:.2f}%)")


if __name__ == "__main__":
    main()
