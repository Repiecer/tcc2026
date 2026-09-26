"""ERA5 数据下载（本地运行版）。

数据源分工：
  * tmax（目标变量）  → Open-Meteo Archive API：免认证、免排队，产品就是 ERA5-Land 日最高温
  * sst_mslp / circ   → CDS（cdsswarm）：派生数据集已算好日统计量，但 CDS 会排队

CDS 在 2026 年长期过载（全局排队 8000+），所以：
  1) 请求按 year_batch 打成小块（cost limit ≈ 天数 × 变量数 × 层数）；
  2) 支持断点续传：已下载的文件跳过，已提交的 job 复用，不重新排队。

用法：
    uv run python -m tcc.download_workers --dry-run      # 只看计划
    uv run python -m tcc.download_workers                # 全量下载
    uv run python -m tcc.download_workers --group tmax   # 只下目标变量
    uv run python -m tcc.download_workers --verify-only  # 只校验并生成校验和

产物：$TCC_DATA_DIR/raw/<分组>/，全部完成后生成 CHECKSUMS.sha256。
"""

from __future__ import annotations

import argparse
import calendar
import hashlib
import os
import sys
import time
from datetime import datetime
from itertools import zip_longest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

try:
    import cdsswarm
except ImportError:            # 允许没装 cdsswarm 时跑 --dry-run
    cdsswarm = None            # type: ignore[assignment]

DEFAULT_RESOLUTION = 0.25      # ERA5 单层 / 气压层
LAND_RESOLUTION = 0.1          # ERA5-Land
MIN_VALID_BYTES = 1024
CDS_RETRY_BASE = 300           # CDS 轮次间隔起始（秒）
CDS_RETRY_CAP = 3600           # 间隔上限（秒）


# ---------------------------------------------------------------- 基础

def log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    data_dir = cfg.get("data_dir") or os.environ.get("TCC_DATA_DIR") or "./data"
    cfg["data_dir"] = Path(data_dir).expanduser().resolve()
    cfg["_years"] = list(range(int(cfg["years"][0]), int(cfg["years"][1]) + 1))
    for name, g in cfg["groups"].items():
        g["name"] = name
        if "dataset" in g:     # 只有 CDS 分组需要 resolution
            g.setdefault("resolution",
                         LAND_RESOLUTION if "land" in g["dataset"] else DEFAULT_RESOLUTION)
    return cfg


def days_in(year: int, month: int) -> list[str]:
    """该年月的合法日期（零填充）：避免给 CDS 送 6 月 31 日这种不存在的日期。"""
    return [f"{d:02d}" for d in range(1, calendar.monthrange(year, month)[1] + 1)]


def is_valid(path: Path) -> bool:
    """能否当作「已完成」跳过。

    便宜的字节日志永远做（CDS 出错时会把 HTML 错误页写成 .nc）；
    装了 xarray 再做一次深度检查。
    """
    if not path.exists() or path.stat().st_size < MIN_VALID_BYTES:
        return False
    with path.open("rb") as fh:
        head = fh.read(8)
    if head[:3] != b"CDF" and head[:4] != b"\x89HDF" and head[:4] != b"GRIB":
        return False
    try:
        import xarray as xr
    except ImportError:
        return True
    try:
        with xr.open_dataset(path) as ds:
            return len(ds.data_vars) > 0 and ds.sizes.get("time", 0) > 0
    except Exception:          # noqa: BLE001
        return False


# ---------------------------------------------------------------- CDS

def build_cds_tasks(cfg: dict, only: list[str] | None) -> list[Any]:
    """按 (年块, 月) × 分组 生成 cdsswarm 任务，并把各分组**交错**排列。

    交错的原因：CDS 的排队配额按数据集算（报错原文 "for this dataset"）。
    若按分组顺序发（先把 circ 发完），其他数据集的配额就闲置了。

    year_batch 的取值来自实测：CDS 的 cost limit ≈ 天数 × 变量数 × 层数，
    单变量 372 天可通过、465 天报 "request is too large"。
    """
    make = cdsswarm.Task if cdsswarm is not None else (lambda **kw: SimpleNamespace(**kw))

    years = cfg["_years"]
    per_group: list[list[Any]] = []
    for name, g in cfg["groups"].items():
        if (only and name not in only) or "dataset" not in g:
            continue
        batch = max(1, int(g.get("year_batch", 1)))
        blocks = [years[i:i + batch] for i in range(0, len(years), batch)]
        tasks = []
        for block in blocks:
            for month in (int(m) for m in cfg["months"]):
                days = sorted({d for y in block for d in days_in(y, month)})
                req: dict[str, Any] = {
                    "variable": list(g["variables"]),
                    "year": [str(y) for y in block],
                    "month": [f"{month:02d}"],      # 零填充：ERA5-Land 只认 '05'
                    "day": days,
                    "daily_statistic": g["statistic"],
                    "time_zone": cfg["time_zone"],
                    "frequency": cfg["frequency"],
                    "area": list(g["area"]),
                    "data_format": "netcdf",
                }
                if g.get("pressure_level"):
                    req["pressure_level"] = list(g["pressure_level"])
                tag = f"{block[0]}-{block[-1]}" if len(block) > 1 else f"{block[0]}"
                tasks.append(make(
                    dataset=g["dataset"], request=req,
                    target=str(cfg["data_dir"] / "raw" / name / f"{name}_{tag}_{month:02d}.nc"),
                ))
        per_group.append(tasks)

    out: list[Any] = []
    for combo in zip_longest(*per_group):
        out.extend(t for t in combo if t is not None)
    return out


def check_cds_credentials(cfg: dict) -> bool:
    """提交前先确认凭据可用——否则会白提交几百个请求再全部失败。"""
    raw = cfg.get("cdsapirc") or os.environ.get("CDSAPI_RC") or "~/.cdsapirc"
    p = Path(raw).expanduser()
    if not p.exists():
        log(f"❌ 找不到 CDS 凭据: {p}")
        log("   三种修法（任选其一）：")
        log(f"     A) cp <你的凭据> {p}  &&  chmod 600 {p}")
        log("     B) 在 configs/download.yaml 里写：cdsapirc: /绝对路径/.cdsapirc")
        log("     C) export CDSAPI_RC=/绝对路径/.cdsapirc")
        return False
    text = p.read_text(encoding="utf-8", errors="ignore")
    if "url" not in text or "key" not in text:
        log(f"❌ 凭据文件不完整（需要 url 和 key 两行）: {p}")
        return False
    os.environ["CDSAPI_RC"] = str(p)   # 让 cdsswarm 内部创建的 client 也读得到
    log(f"✓ CDS 凭据: {p}")
    return True


def run_cds(cfg: dict, tasks: list[Any]) -> int:
    """跑 CDS 任务（轮次重试 + 退避），返回仍未完成的数量。"""
    if not tasks:
        return 0
    if cdsswarm is None:
        log("❌ 未安装 cdsswarm：uv add cdsswarm")
        return len(tasks)
    if not check_cds_credentials(cfg):
        return len(tasks)

    backoff = CDS_RETRY_BASE
    for rnd in range(1, int(cfg.get("max_rounds", 20)) + 1):
        pending = [t for t in tasks if not is_valid(Path(t.target))]
        if not pending:
            break
        log(f"--- CDS 第 {rnd} 轮：{len(pending)} 个任务待下载 ---")
        try:
            results = cdsswarm.download(
                pending,
                num_workers=int(cfg.get("workers", 4)),
                skip_existing=True,   # 续传第 1 层：跳过已完成
                reuse_jobs=True,      # 续传第 2 层：复用 CDS 上已提交的 job（不重新排队）
                max_retries=int(cfg.get("max_retries", 3)),
                on_message=lambda m: log(str(m)),
            )
        except Exception as exc:  # noqa: BLE001
            log(f"本轮异常: {type(exc).__name__}: {exc}")
            results = []

        ok = 0
        for r in results:
            if r.success:
                ok += 1
            else:
                log(f"  ✗ {Path(r.task.target).name}: {r.error}")
        log(f"本轮成功 {ok}/{len(results)}")

        left = []
        for t in pending:                       # 清掉无效文件，下一轮重下
            if is_valid(Path(t.target)):
                continue
            p = Path(t.target)
            if p.exists():
                p.unlink()
            left.append(t)
        if not left:
            break

        made_progress = ok > 0
        if made_progress:
            backoff = CDS_RETRY_BASE            # 有文件落盘 → 配额在放行，退避重置
        log(f"仍有 {len(left)} 个未完成，{backoff}s 后继续")
        time.sleep(backoff)
        if not made_progress:
            backoff = min(backoff * 2, CDS_RETRY_CAP)

    return len([t for t in tasks if not is_valid(Path(t.target))])


# ---------------------------------------------------------------- 校验和

def write_manifest(cfg: dict) -> int:
    """生成 sha256 清单，供团队分发时核对用的是不是同一份数据。"""
    raw = cfg["data_dir"] / "raw"
    lines = []
    for p in sorted(raw.rglob("*.nc")):
        if not is_valid(p):
            continue
        h = hashlib.sha256()
        with p.open("rb") as fh:
            while block := fh.read(1 << 20):
                h.update(block)
        lines.append(f"{h.hexdigest()}  {p.relative_to(raw)}")
    out = cfg["data_dir"] / "CHECKSUMS.sha256"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"✓ 校验和清单: {out}（{len(lines)} 个文件）")
    return len(lines)


# ---------------------------------------------------------------- 主流程

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ERA5 数据下载")
    ap.add_argument("--config", default="configs/download.yaml")
    ap.add_argument("--group", action="append", help="只处理指定分组，可重复")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划，不发请求")
    ap.add_argument("--verify-only", action="store_true", help="只校验并生成清单")
    args = ap.parse_args(argv)

    cfg = load_config(Path(args.config).resolve())
    only = args.group
    cfg["data_dir"].mkdir(parents=True, exist_ok=True)

    # ---- 非 CDS 数据源（Open-Meteo）：增量落盘，可随时中断续跑
    if not args.verify_only and not args.dry_run:
        other = [n for n, g in cfg["groups"].items()
                 if g.get("source") == "openmeteo" and (not only or n in only)]
        if other:
            try:
                from tcc.openmeteo_fetch import fetch_group
            except ImportError:
                sys.path.insert(0, str(Path(__file__).resolve().parent))
                from openmeteo_fetch import fetch_group  # type: ignore
            for n in other:
                fetch_group(cfg, cfg["groups"][n])

    # ---- CDS 任务
    tasks = build_cds_tasks(cfg, only)
    counts: dict[str, int] = {}
    for t in tasks:
        k = Path(t.target).parent.name
        counts[k] = counts.get(k, 0) + 1
    log(f"CDS 任务 {len(tasks)} 个  " + "  ".join(f"{k}={v}" for k, v in counts.items()))

    if args.dry_run:
        log("--dry-run 结束，未发起任何请求")
        return 0
    if args.verify_only:
        write_manifest(cfg)
        return 0

    left = run_cds(cfg, tasks)
    write_manifest(cfg)
    if left:
        log(f"❌ 仍有 {left} 个 CDS 任务未完成；再跑一次会继续（不会重下已完成部分）")
        return 1
    log("🎉 全部完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
