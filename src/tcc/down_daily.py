"""ERA5 批量下载（日统计版）。

最初的版本是「逐小时 + 全变量 + 一刀切大区」：540 个任务、约 99 GB。
这一版把体积压到约 3 GB（约 30 倍），靠三件事：

  1. 换成 CDS 服务端算好的日统计数据集（derived-*-daily-statistics）。
     预报因子本来就是周平均/候平均，小时数据在算完日均那一刻就丢掉了 ——
     直接下逐小时，等于下了 24 倍冗余。
  2. 区域按物理需要分别设定。area 不参与 CDS 的 cost limit，改这里零成本：
     原来的 [55, 70, 15, 140] 在 140°E 截断，把西太暖池和副高核心区切掉了。
  3. 请求按 CDS 的 cost limit 自动打包，任务数 540 → 约 140，排队次数大幅减少。

用法：
    uv run python src/tcc/down_daily.py --dry-run      # 只打印计划和预估体积
    uv run python src/tcc/down_daily.py                # 开始下载（全部 45 年）
    uv run python src/tcc/down_daily.py --years 2024   # 先下一年试点
    uv run python src/tcc/down_daily.py --group circ   # 只下某一组（分组名：tmax_land / sst_mslp / circ）

产物：data/raw/<组名>/<组名>_<起年>-<止年>_<月>.nc
"""

from __future__ import annotations

import argparse
import calendar
import logging
import os
from pathlib import Path

import cdsswarm

# CDS 排队配额满时，底层库（ecmwf/requests）会打一大段 traceback。那不是错误，
# 是「稍后重试」的正常信号，这里压掉，只在最后统一汇报。
for _name in ("cdsswarm", "ecmwf", "urllib3", "requests"):
    logging.getLogger(_name).setLevel(logging.CRITICAL)

QUOTA_HINTS = ("temporarily limited", "queued requests")

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("TCC_DATA_DIR") or ROOT / "data") / "raw"

YEARS = list(range(1981, 2026))
MONTHS = ["05", "06", "07", "08"]

TIME_ZONE = "utc+08:00"      # 按北京时间切「日」，与目标变量 Tmax 的定义保持一致
FREQUENCY = "1_hourly"       # 日最高温的精度靠它：降到 3_hourly 会带来约 1℃ 误差

# CDS 硬上限（实测）：一个请求的「天数 × 变量数 × 层数」超过约 400 就报 request is too large。
#   单变量：372 天通过，465 天被拒。下面的打包完全由这个数字推出来。
COST_LIMIT = 400
MAX_MONTH_DAYS = 31          # 打包时按最长月份保守估算

SINGLE = "derived-era5-single-levels-daily-statistics"
PRESSURE = "derived-era5-pressure-levels-daily-statistics"
LAND = "derived-era5-land-daily-statistics"

# 分组定义。area = [北, 西, 南, 东]。
# 想改区域直接改这里一行：area 只影响文件体积，不影响 CDS 的 cost limit。
GROUPS = {
    # ① 目标变量：长江中下游（28—34°N, 110—122°E）日最高气温。
    # 组名刻意叫 tmax_land 而不是 tmax：download_workers.py 那份走 Open-Meteo 的
    # tmax 是 325 个采样点（13×25），这里是 ERA5-Land 完整 0.1° 网格（61×121），
    # 放进同一个目录会让下游 glob 出两种形状的数组。
    "tmax_land": {
        "dataset": LAND,
        "variables": ["2m_temperature"],
        "statistic": "daily_maximum",
        "area": [34, 110, 28, 122],
        "resolution": 0.1,           # ERA5-Land 是 0.1°
        "bytes_per_value": 0.645,    # 实测：data/daily_tmax_202407.nc
    },
    # ② 海温 + 海平面气压：往东扩到 160°E，把西太暖池和副高核心区包进来
    "sst_mslp": {
        "dataset": SINGLE,
        "variables": ["sea_surface_temperature", "mean_sea_level_pressure"],
        "statistic": "daily_mean",
        "area": [50, 95, 0, 160],
        "resolution": 0.25,
        "bytes_per_value": 1.2,      # 日场实测区间 0.645—2.0 的中值
    },
    # ③ 环流：500/850 hPa 位势与风场，含热带对流区（延伸期可预报性来源）
    "circ": {
        "dataset": PRESSURE,
        "variables": ["geopotential", "u_component_of_wind", "v_component_of_wind"],
        "pressure_level": ["500", "850"],
        "statistic": "daily_mean",
        "area": [55, 95, 0, 160],
        "resolution": 0.25,
        "bytes_per_value": 1.2,
    },
}


# ---------------------------------------------------------------- 计算

def grid_points(area, resolution):
    """区域内的格点数（CDS 的 area 含端点）。"""
    north, west, south, east = area
    return (round((north - south) / resolution) + 1) * (round((east - west) / resolution) + 1)


def fields_of(g):
    """一个请求每时刻的场数 = 变量数 × 层数。"""
    return len(g["variables"]) * len(g.get("pressure_level", [None]))


def years_per_request(g):
    """按 cost limit 自动决定「同一个月」一个请求能装几年。"""
    return max(1, (COST_LIMIT // fields_of(g)) // MAX_MONTH_DAYS)


def chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def days_in(month, year):
    """该月天数。5—8 月与年份无关（只有 2 月受闰年影响）。"""
    return [f"{d:02d}" for d in range(1, calendar.monthrange(year, int(month))[1] + 1)]


def build_tasks(names, years):
    tasks = []
    for name in names:
        g = GROUPS[name]
        out_dir = DATA_DIR / name
        for block in chunks(years, years_per_request(g)):
            for month in MONTHS:
                request = {
                    "variable": list(g["variables"]),
                    "year": [str(y) for y in block],
                    "month": [month],
                    "day": days_in(month, block[0]),
                    "daily_statistic": g["statistic"],
                    "time_zone": TIME_ZONE,
                    "frequency": FREQUENCY,
                    "area": g["area"],
                    "data_format": "netcdf",
                }
                # derived-era5-land-daily-statistics 没有 product_type 这个输入项
                if g["dataset"] != LAND:
                    request["product_type"] = ["reanalysis"]
                if "pressure_level" in g:
                    request["pressure_level"] = g["pressure_level"]

                tasks.append(
                    cdsswarm.Task(
                        dataset=g["dataset"],
                        request=request,
                        target=str(out_dir / f"{name}_{block[0]}-{block[-1]}_{month}.nc"),
                    )
                )
    return tasks


def show_plan(names, years):
    print(f"年份 {years[0]}—{years[-1]}（{len(years)} 年）  月份 {MONTHS}")
    print(f"{'分组':10s}{'网格':>12s}{'场/时刻':>8s}{'年/请求':>8s}{'任务数':>7s}{'预估体积':>11s}")
    total_tasks = total_bytes = 0
    for name in names:
        g = GROUPS[name]
        pts = grid_points(g["area"], g["resolution"])
        vals = pts * sum(len(days_in(m, years[0])) for m in MONTHS) * fields_of(g) * len(years)
        nbytes = vals * g["bytes_per_value"]
        ntasks = (len(years) + years_per_request(g) - 1) // years_per_request(g) * len(MONTHS)
        total_tasks += ntasks
        total_bytes += nbytes
        print(f"{name:10s}{pts:>12,}{fields_of(g):>8d}{years_per_request(g):>8d}{ntasks:>7d}{nbytes / 1e9:>10.2f}G")
    print(f"{'合计':10s}{'':>12s}{'':>8s}{'':>8s}{total_tasks:>7d}{total_bytes / 1e9:>10.2f}G")
    print("体积按实测压缩率估算（ERA5-Land 日最高 0.645 B/值；ERA5 日场 0.645—2.0，取中值 1.2）")


# ---------------------------------------------------------------- 入口

def is_quota_rejection(error):
    """判断失败原因是不是「CDS 排队配额已满」——那不是真错误。"""
    text = (error or "").lower()
    return any(hint in text for hint in QUOTA_HINTS)


def parse_years(text):
    """'2024' 或 '2010-2020' → 年份列表。"""
    if not text:
        return YEARS
    if "-" in text:
        start, end = text.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(text)]


def main():
    parser = argparse.ArgumentParser(description="ERA5 日统计批量下载")
    parser.add_argument("--group", action="append", choices=list(GROUPS),
                        help="只下指定分组（可重复），默认全部")
    parser.add_argument("--years", help="只下指定年份，如 2024 或 2010-2020（默认全部 45 年）")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划和预估体积")
    args = parser.parse_args()

    names = args.group or list(GROUPS)
    years = parse_years(args.years)
    tasks = build_tasks(names, years)
    show_plan(names, years)

    if args.dry_run:
        print("--dry-run：未发起任何请求")
        return

    # CDS 的排队配额是按数据集算的，本地并发开大只会换来更多 rejected
    workers = 1# min(6, max(2, len(names)))
    results = cdsswarm.download(tasks, num_workers=workers)

    ok = sum(1 for r in results if r.success)
    quota = [r for r in results if not r.success and is_quota_rejection(r.error)]
    failed = [r for r in results if not r.success and not is_quota_rejection(r.error)]

    print(f"成功 {ok} / {len(tasks)}")
    if quota:
        print(f"  CDS 排队配额已满，{len(quota)} 个任务没排上队 —— 这不是错误。"
              f"过一会儿重跑本脚本即可：已完成的文件会自动跳过，已提交的 job 会复用，不会重复排队。")
    for r in failed[:5]:
        print(f"  ✗ {r.task.label}: {(r.error or '').splitlines()[0][:110]}")
    if len(failed) > 5:
        print(f"  ... 另有 {len(failed) - 5} 个失败")


if __name__ == "__main__":
    main()
