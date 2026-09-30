from pathlib import Path

import numpy as np
import xarray as xr

RAW_DIR = Path("data/raw/tmax")
OUT_DIR = Path("data/proc")

LAT_SLICE = slice(28, 34)
LON_SLICE = slice(110, 122)

CLIM_YEARS = list(range(1981, 2016))
WINDOW = 11
SIGMA_FLOOR_RATIO = 0.5

DOY_START = 121
DOY_END = 244

def fold_doy(doy):
    return np.where(doy == 366, 365, doy)


def slide_weighted_mean(values, weights, window):
    n = len(values)
    pad = window // 2
    out = np.full(n, np.nan)

    for i in range(n):
        lo = max(0, i - pad)
        hi = min(n, i + pad + 1)
        w = weights[lo:hi]
        v = values[lo:hi]
        keep = w > 0
        total = w[keep].sum()
        if total > 0:
            out[i] = float((v[keep] * w[keep]).sum() / total)
    return out


# ============================================================
# 三、读数据 + 区域平均
# ============================================================

def load_region_mean():
    files = sorted(RAW_DIR.glob("tmax_*.nc"))
    if not files:
        raise FileNotFoundError(f"no file: {RAW_DIR}/tmax_*.nc")

    ds = xr.open_mfdataset(files, combine="by_coords")
    sub = ds["t2m"].sel(latitude=LAT_SLICE, longitude=LON_SLICE) - 273.15
    vmin, vmax = float(sub.min()), float(sub.max())
    if not (-40 < vmin < 55 and -40 < vmax < 55):
        raise ValueError(
            f"Temprature Error {vmin:.1f} - {vmax:.1f}。"
        )
    coslat = np.cos(np.deg2rad(sub["latitude"]))
    weight = sub.notnull().astype(float) * coslat
    numerator = (sub.fillna(0.0) * weight).sum(["latitude", "longitude"])
    denominator = weight.sum(["latitude", "longitude"])
    region = numerator / denominator


    # --- 检查 2：时间轴有没有重复日期 ---
    # 注意：先做完整性检查、后丢弃空值。否则整季第一天全空时会被静默丢掉。
    dates = np.asarray(region["time"].dt.floor("D"))
    n_unique = len(np.unique(dates))
    if n_unique != len(dates):
        raise ValueError(f"时间轴有重复日期：{len(dates)} 行但只有 {n_unique} 个日期。")

    # --- 检查 3：季节内部不能有缺日 ---
    # 注意：数据只含每年 5-8 月，跨年处必然有"跳跃"，这是正常的，不算错误。
    # 只需检查同一年内部是否连续。
    years_found = np.asarray(region["time"].dt.year)
    for y in np.unique(years_found):
        d = np.sort(dates[years_found == y])
        gaps = np.diff(d).astype("timedelta64[D]").astype(int)
        if gaps.size and gaps.max() != 1:
            raise ValueError(f"{y} 年内部时间轴不连续。")

    # --- 检查 4：数据必须落在支持的季节范围内（本脚本只支持 5-8 月）---
    # 用"月份"判断，而不是日历日号码：闰年会让 doy 整体错位一天
    # （闰年 4/30 的 doy 是 121，与平年 5/1 相同），只用 doy 判断会漏掉。
    months = np.unique(np.asarray(region["time"].dt.month))
    bad_months = sorted(int(m) for m in months if m not in (5, 6, 7, 8))
    if bad_months:
        raise ValueError(
            f"数据包含 5-8 月之外的月份 {bad_months}。本脚本只支持 5-8 月，"
            "请扩充 DOY_START/DOY_END 并重跑。"
        )

    # --- 检查 5：完全没有数据的日期（放在所有检查之后才丢弃）---
    n_empty = int(region.isnull().sum())
    if n_empty:
        print(f"  提示：有 {n_empty} 天区域平均为空，将被丢弃。")

    region = region.dropna("time")

    ds.close()
    region = region.rename("tmax_region")
    region.attrs["units"] = "degC"
    region.attrs["long_name"] = "研究区面积加权平均日最高气温"
    return region


# ============================================================
# 四、算气候态 clim 和标准差 sigma
# ============================================================

def calendar_clim_std(tmax, clim_years=None):
    """算每个日历日的常年平均 clim 和波动幅度 sigma。

    只用训练期年份（默认 CLIM_YEARS），绝不使用验证/测试年份，
    否则测试期信息会通过 clim/sigma 泄漏进训练过程（数据泄漏）。

    做法（三步）：
      1. 先按日历日分组，算训练期内的均值、平方的均值、样本数；
      2. 用"样本数加权"的滑动窗口平滑均值与平方均值
         （加权 = 把窗口内所有样本合并统计，等价于池化；
           样本少的日历日不会被过度放大）；
      3. 方差 = E[X^2] - (E[X])^2。

    clim_years 参数存在的唯一目的，是让校验环节能用"换一组年份"来
    检验气候态是否真的只依赖训练期（见 validate 里的泄漏检查）。

    返回 (doy 数组, clim 数组, sigma 数组)。
    """
    if clim_years is None:
        clim_years = CLIM_YEARS

    doy_list = np.arange(DOY_START, DOY_END + 1)

    years = np.asarray(tmax["time"].dt.year)
    doy = fold_doy(np.asarray(tmax["time"].dt.dayofyear))
    is_train = np.isin(years, list(clim_years))
    values = tmax.values

    # --- 第 1 步：逐日历日统计 ---
    mean_per_doy = np.empty(len(doy_list))
    meansq_per_doy = np.empty(len(doy_list))
    count_per_doy = np.empty(len(doy_list))

    for i, d in enumerate(doy_list):
        x = values[is_train & (doy == d)]
        count_per_doy[i] = x.size
        if x.size:
            mean_per_doy[i] = x.mean()
            meansq_per_doy[i] = (x ** 2).mean()
        else:
            # 该日历日在给定年份里不存在（例如平年没有 8/31），记为 NaN
            mean_per_doy[i] = np.nan
            meansq_per_doy[i] = np.nan

    few = count_per_doy < 10
    if few.any() and list(clim_years) == list(CLIM_YEARS):
        # 只在正式估计时提示一次，避免校验环节重复打印
        print(f"  提示：{int(few.sum())} 个日历日的训练期样本少于 10 个 "
              f"(doy: {doy_list[few].tolist()})，属闰年边界正常现象。")

    # --- 第 2 步：样本数加权的滑动平滑 ---
    # 权重 = 该日历日在训练期的样本数，等价于把窗口内所有样本合并池化。
    # 窗口在季节两端自然截断（不借用 5-8 月以外的气候信息）。
    clim = slide_weighted_mean(mean_per_doy, count_per_doy, WINDOW)
    meansq_smooth = slide_weighted_mean(meansq_per_doy, count_per_doy, WINDOW)

    # 权重全为 0 的日历日（该日在该年份集合里不存在）会得到 NaN
    n_nan = int(np.isnan(clim).sum())
    if n_nan:
        print(f"  提示：{n_nan} 个日历日在给定年份集合内没有样本，clim 为空。")

    # --- 第 3 步：方差与标准差 ---
    variance = np.clip(meansq_smooth - clim ** 2, 0, None)   # clip 防浮点负数
    sigma = np.sqrt(variance)

    return doy_list, clim, sigma


# ============================================================
# 五、算 H
# ============================================================

def compute_H(tmax, doy_list, clim, sigma):
    """H = (T - clim) / sigma，逐日元算。"""
    years = np.asarray(tmax["time"].dt.year)
    folded = fold_doy(np.asarray(tmax["time"].dt.dayofyear))

    # 把每个日期的日历日号码换算成 clim/sigma 数组的下标，并检查越界
    index = folded - doy_list[0]
    if index.min() < 0 or index.max() >= len(doy_list):
        raise IndexError(
            f"日历日超出 clim/sigma 的范围：doy {folded.min()}-{folded.max()}，"
            f"而数组只覆盖 {doy_list[0]}-{doy_list[-1]}。"
        )

    # 保险丝：防止某个日历日 sigma 过小导致 H 爆炸
    sigma_floor = SIGMA_FLOOR_RATIO * sigma.mean()
    sigma_use = np.maximum(sigma, sigma_floor)
    n_floor = int((sigma < sigma_floor).sum())
    if n_floor:
        print(f"  提示：有 {n_floor} 个日历日触发了 sigma 下限。")

    H = (tmax.values - clim[index]) / sigma_use[index]
    H = xr.DataArray(H, dims="time", coords={"time": tmax["time"]}, name="H")
    H.attrs["long_name"] = "标准化高温指数 (z-score of Tmax anomaly)"
    H.attrs["formula"] = "(Tmax_region - clim) / sigma"
    return H, years


# ============================================================
# 六、校验
# ============================================================

def validate(H, years, tmax):
    """打印一套校验结果。标准化很容易写错，这一步不能省。"""
    doy = fold_doy(np.asarray(tmax["time"].dt.dayofyear))
    is_train = np.isin(years, CLIM_YEARS)
    h_train = H.values[is_train]

    print("\n===== 校验 =====")

    # 检查 1：训练期 H 的均值和标准差必须是 0 和 1
    print(f"训练期 H 均值 {h_train.mean():+.4f}（应约等于 0）")
    print(f"训练期 H 标准差 {h_train.std():.4f}（应约等于 1）")
    assert abs(h_train.mean()) < 0.05, "训练期 H 均值偏离 0 太多，检查代码"
    assert abs(h_train.std() - 1) < 0.05, "训练期 H 标准差偏离 1 太多，检查代码"

    # 检查 2：气候态必须只依赖 CLIM_YEARS（真正的数据泄漏检查）
    # 做法：用只含单个训练年份的子集去算，气候态应当与全量输入时完全一致。
    # 如果实现里漏了年份过滤（把全部年份都算进去），两者会明显不同。
    print("\n泄漏检查（气候态必须只依赖 CLIM_YEARS）：")
    for only_year in [CLIM_YEARS[0], CLIM_YEARS[-1]]:
        # 让函数在"全量数据"上只挑这一年；它必须与"只喂这一年数据"结果完全一致。
        # 若年份掩膜失效（把其他年份也算进来），两者就会不同。
        _, clim_full, sigma_full = calendar_clim_std(tmax, [only_year])
        subset = tmax.sel(time=tmax["time"].dt.year == only_year)
        _, clim_sub, sigma_sub = calendar_clim_std(subset, [only_year])
        delta = max(float(np.abs(clim_full - clim_sub).max()),
                    float(np.abs(sigma_full - sigma_sub).max()))
        ok = delta < 1e-12
        print(f"  只用 {only_year} 年：全量输入 vs 单年输入，最大差异 {delta:.2e}"
              f" -> {'通过（年份过滤正确）' if ok else '不合格（年份过滤有问题）'}")
        assert ok, (
            f"只用 {only_year} 年时，全量输入与单年输入的气候态不一致（{delta:.2e}），"
            "说明 clim/sigma 没有严格按 clim_years 过滤年份，存在数据泄漏。"
        )

    # 检查 3：变暖趋势必须被保留（气候态冻结在训练期，近年应系统性偏正）
    print("\n分时段统计（气候态冻结在 1981-2015，近年应偏正）：")
    for y0, y1 in [(1981, 2015), (2016, 2020), (2021, 2025)]:
        m = (years >= y0) & (years <= y1)
        s = H.values[m]
        print(f"  {y0}-{y1}: 均值 {s.mean():+.3f}  标准差 {s.std():.3f}  "
              f"最大 {s.max():+.2f}  H>2 占比 {100 * (s > 2).mean():.2f}%")

    recent = H.values[years >= 2021].mean()
    print(f"  参考：2021-2025 整体均值 {recent:+.3f}"
          "（若误用全期气候态，此值会显著变小）")

    # 检查 4：H 里不应残留强季节信号
    # 这是弱检查：残留振幅主要来自各日历日的采样噪声，这里只做提示不作断言。
    print(f"\n残留季节信号振幅 {_seasonal_amplitude(H.values, doy):.4f}"
          "（越接近 0 越好；该指标受采样噪声影响，仅作参考）")


def _seasonal_amplitude(H_values, doy):
    """量度 H 中残留的季节信号：各日历日均值的极差。"""
    inner = np.arange(DOY_START + 5, DOY_END - 4)
    means = np.array([H_values[doy == d].mean() for d in inner])
    return means.max() - means.min()


# ============================================================
# 七、保存结果 + 写数据字典
# ============================================================

def save_results(tmax, H, doy_list, clim, sigma):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    xr.Dataset({"tmax_region": tmax}).to_netcdf(OUT_DIR / "tmax_region.nc")
    xr.Dataset({"H": H}).to_netcdf(OUT_DIR / "tmax_H.nc")

    # 参数文件：一定要存，否则半年后无法复现
    params = xr.Dataset(
        {"clim": ("doy", clim), "sigma": ("doy", sigma)},
        coords={"doy": doy_list},
        attrs={
            "clim_years": f"{CLIM_YEARS[0]}-{CLIM_YEARS[-1]}",
            "window_days": WINDOW,
            "sigma_floor_ratio": SIGMA_FLOOR_RATIO,
            "region": "28-34N, 110-122E",
            "region_mean_formula": "sum(T * cos(lat) * valid) / sum(cos(lat) * valid)",
            "area_weight": "cos(latitude)，海面空格点权重置 0",
            "window_smoothing": "样本数加权的滑动平均（等价于窗口内样本池化）",
            "source": "data/raw/tmax/tmax_YYYY_MM.nc (日最高气温, ERA5)",
            "unit": "degC (源数据为 K，已减 273.15)",
        },
    )
    params.to_netcdf(OUT_DIR / "tmax_norm_params.nc")

    # 数据字典
    doc = Path("docs/data_dictionary_tmax.md")
    doc.write_text(
        "# t2m（日最高气温 Tmax）数据字典\n\n"
        "由 `scripts/standardize_tmax.py` 生成。\n\n"
        "## 输出文件\n\n"
        "| 文件 | 变量 | 含义 | 单位 |\n"
        "|---|---|---|---|\n"
        "| `data/proc/tmax_region.nc` | `tmax_region` | 研究区面积加权平均日最高气温 | degC |\n"
        "| `data/proc/tmax_H.nc` | `H` | 标准化高温指数 | 无量纲 |\n"
        "| `data/proc/tmax_norm_params.nc` | `clim`, `sigma` | 逐日历日气候态与标准差 | degC |\n\n"
        "## 处理参数\n\n"
        "| 项目 | 取值 |\n"
        "|---|---|\n"
        "| 源数据 | `data/raw/tmax/tmax_YYYY_MM.nc`（ERA5 日最高气温）|\n"
        "| 研究区 | 28-34°N, 110-122°E |\n"
        "| 时间范围 | 1981-2025 年，仅 5-8 月（5535 天）|\n"
        "| 单位换算 | K → degC（减 273.15）|\n"
        "| 区域平均公式 | `sum(T · cosφ · valid) / sum(cosφ · valid)`，φ 为纬度 |\n"
        "| 空缺值处理 | 海面空格点权重置 0，不做插值 |\n"
        f"| 气候态窗口 | 以日历日为中心 ±5 天（共 {WINDOW} 天）滑动窗口 |\n"
        "| 窗口平滑方式 | 样本数加权的滑动平均（等价于窗口内样本池化）|\n"
        f"| 气候态基准期 | {CLIM_YEARS[0]}-{CLIM_YEARS[-1]}（**只用训练期，冻结使用**）|\n"
        f"| sigma 下限 | 全季平均 sigma 的 {SIGMA_FLOOR_RATIO} 倍 |\n"
        "| H 公式 | `H = (Tmax_region - clim) / sigma` |\n\n"
        "## 已知事项\n\n"
        "- 研究区内 17 个格点（110-122°E 内的海面格点）整列为空，"
        "区域平均时权重置 0（相当于按有效面积归一化），未做插值。\n"
        "- 源数据变量名虽为 `t2m`，实测数值与 Open-Meteo `temperature_2m_max` "
        "完全一致（相关系数 1.0000），确认为日最高气温，输出统一命名为 `tmax`。\n"
        "- 气候态与 sigma 仅用 1981-2015 估计，验证/测试期（2016-2025）"
        "冻结套用，以避免数据泄漏。脚本会自动做结构性泄漏检查"
        "（额外加入测试年份，气候态若有变化就报错）。\n"
        "- 闰年 8/31（doy=244）在训练期只有 8 个样本；窗口平滑采用样本数加权，"
        "因此该日不会被过度代表。\n"
        "- 本脚本只支持 5-8 月，用**月份**判断（不用日历日号码，避免闰年错位）。"
        "若数据扩展到其他月份，脚本会直接报错退出。\n"
        f"- **sigma 下限在本数据集上未被触发**：下限值 0.5×均值低于全季最小 sigma，"
        "该保险丝仅对未来数据或其他区域起保护作用，目前未经实际检验。\n",
        encoding="utf-8",
    )

    print("\n已保存：")
    print(f"  {OUT_DIR / 'tmax_region.nc'}")
    print(f"  {OUT_DIR / 'tmax_H.nc'}")
    print(f"  {OUT_DIR / 'tmax_norm_params.nc'}")
    print(f"  {doc}")


# ============================================================
# 八、主流程
# ============================================================

def main():
    print("第 1 步：读取数据并做区域平均 ...")
    tmax = load_region_mean()
    years = np.asarray(tmax["time"].dt.year)
    print(f"  共 {tmax.size} 天，{years.min()}-{years.max()}，"
          f"区域平均 {float(tmax.mean()):.2f} degC")

    print(f"第 2 步：算气候态与 sigma（只用 {CLIM_YEARS[0]}-{CLIM_YEARS[-1]}）...")
    doy_list, clim, sigma = calendar_clim_std(tmax)
    print(f"  clim : {clim.min():.2f} ~ {clim.max():.2f} degC")
    print(f"  sigma: {sigma.min():.3f} ~ {sigma.max():.3f} degC")

    print("第 3 步：算标准化高温指数 H ...")
    H, years = compute_H(tmax, doy_list, clim, sigma)

    validate(H, years, tmax)

    print("\n第 4 步：保存结果 ...")
    save_results(tmax, H, doy_list, clim, sigma)

    print("\n完成。")


if __name__ == "__main__":
    main()
