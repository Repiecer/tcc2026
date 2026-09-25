"""ERA5 批量下载 worker。

基于 ``cdsswarm`` 编排 CDS 检索任务，提供：

* **重试**：任务级（cdsswarm ``max_retries``）+ 轮次级（外层退避重试）双重保障；
* **断点续传**：三层保障
    1. ``skip_existing`` —— 目标文件已存在则跳过；
    2. ``reuse_jobs``   —— 复用 CDS 上已提交/已完成的 job，**不重新排队**；
    3. ``resume``       —— cdsswarm 把每个任务的 ``cds_request_id`` 持久化在
       ``~/.cache/cdsswarm/sessions/*.json``，进程被杀后可续；
* **校验**：下载后逐个检查文件是否为有效 NetCDF（防止把 CDN 的 HTML 错误页写成 .nc）；
* **清单**：全部完成后生成 sha256 清单，供团队分发时核对一致性。

用法::

    uv run python -m tcc.download_workers --config configs/download.yaml --dry-run
    uv run python -m tcc.download_workers --config configs/download.yaml
    uv run python -m tcc.download_workers --config configs/download.yaml --group tmax
    uv run python -m tcc.download_workers --config configs/download.yaml --verify-only

退出码：0 = 全部任务成功；1 = 仍有任务未完成（供 systemd 判断是否重启）。
"""

from __future__ import annotations

import argparse
import calendar
import hashlib
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

try:  # 允许在没装 cdsswarm 的机器上做 --dry-run
    import cdsswarm
except ImportError:  # pragma: no cover
    cdsswarm = None  # type: ignore[assignment]

# ---------------------------------------------------------------- 常量

DEFAULT_RESOLUTION = 0.25  # ERA5 单层/气压层
LAND_RESOLUTION = 0.1      # ERA5-Land
MIN_VALID_BYTES = 1024     # 小于这个体积一定是坏文件
BACKOFF_BASE = 60          # 轮次间隔起始秒数
BACKOFF_CAP = 1800         # 轮次间隔上限（30 分钟）

_stop = False


def _handle_signal(signum: int, _frame: Any) -> None:
    """收到 SIGTERM/SIGINT 时优雅退出，让 systemd 重启后再续。"""
    global _stop
    _stop = True
    log(f"收到信号 {signum}，本轮结束后退出（已完成的文件不会重下）")


def log(msg: str, logfile: Path | None = None) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    if logfile:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        with logfile.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


# ---------------------------------------------------------------- 配置


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    # 数据根目录：配置 > 环境变量 > 仓库默认
    data_dir = cfg.get("data_dir") or os.environ.get("TCC_DATA_DIR") or "./data"
    cfg["data_dir"] = Path(data_dir).expanduser().resolve()

    years = cfg["years"]
    cfg["_years"] = list(range(int(years[0]), int(years[1]) + 1))

    for name, g in cfg["groups"].items():
        g.setdefault("resolution", LAND_RESOLUTION if "land" in g["dataset"] else DEFAULT_RESOLUTION)
        g["name"] = name

    return cfg


def _days_in(year: int, month: int) -> list[str]:
    """该年月的合法日期列表（零填充）。避免给 CDS 送不存在的日期。"""
    return [f"{d:02d}" for d in range(1, calendar.monthrange(year, month)[1] + 1)]


def build_tasks(cfg: dict, only: list[str] | None = None) -> list[Any]:
    """按 (年, 月) × 分组 生成 cdsswarm 任务列表。"""
    if cdsswarm is None:
        raise RuntimeError("未安装 cdsswarm：uv add cdsswarm")

    tasks = []
    out_root = cfg["data_dir"] / "raw"

    for name, g in cfg["groups"].items():
        if only and name not in only:
            continue
        for year in cfg["_years"]:
            for month in cfg["months"]:
                month = int(month)
                request: dict[str, Any] = {
                    "variable": list(g["variables"]),
                    "year": [str(year)],
                    "month": [f"{month:02d}"],          # 零填充：ERA5-Land 只认 '05'
                    "day": _days_in(year, month),
                    "daily_statistic": g["statistic"],
                    "time_zone": cfg["time_zone"],
                    "frequency": cfg["frequency"],
                    "area": list(g["area"]),
                    "data_format": "netcdf",
                }
                if g.get("pressure_level"):
                    request["pressure_level"] = list(g["pressure_level"])
                # 单层数据集要求 product_type；派生数据集不接受该字段
                if g["dataset"].startswith("reanalysis-"):
                    request["product_type"] = ["reanalysis"]

                target = out_root / name / f"{name}_{year}{month:02d}.nc"
                tasks.append(cdsswarm.Task(dataset=g["dataset"], request=request, target=str(target)))
    return tasks


def estimate_gb(cfg: dict, only: list[str] | None = None) -> float:
    """粗略估算下载体积（未压缩 float32），用于 --dry-run。"""
    total = 0.0
    n_days = sum(calendar.monthrange(y, int(m))[1] for y in cfg["_years"] for m in cfg["months"])
    for name, g in cfg["groups"].items():
        if only and name not in only:
            continue
        n, w, s, e = g["area"]
        res = g["resolution"]
        pts = (int((n - s) / res) + 1) * (int((e - w) / res) + 1)
        fields = len(g["variables"]) * len(g.get("pressure_level", [1]))
        total += pts * n_days * fields * 4 / 1e9
    return total


# ---------------------------------------------------------------- 校验


def is_valid(path: Path) -> bool:
    """检查文件是否为可用的 NetCDF。

    分两层：便宜的字节检查永远做；xarray 深度检查只在装了 xarray 时做。
    """
    if not path.exists() or path.stat().st_size < MIN_VALID_BYTES:
        return False
    with path.open("rb") as fh:
        head = fh.read(8)
    # CDS 出错时会把 HTML 错误页写成目标文件
    if head.lstrip()[:1] in (b"<",):
        return False
    if head[:3] != b"CDF" and head[:4] != b"\x89HDF" and head[:4] != b"GRIB":
        return False

    try:
        import xarray as xr
    except ImportError:
        return True  # 没装 xarray 就只做字节检查

    try:
        with xr.open_dataset(path) as ds:
            return len(ds.data_vars) > 0 and ds.sizes.get("time", 1) > 0
    except Exception:  # noqa: BLE001 —— 打不开就是坏文件
        return False


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def write_manifest(raw_root: Path, manifest: Path) -> int:
    files = sorted(p for p in raw_root.rglob("*.nc") if p.is_file())
    lines = []
    for p in files:
        if not is_valid(p):
            log(f"  ⚠ 跳过无效文件: {p.name}")
            continue
        lines.append(f"{sha256_of(p)}  {p.relative_to(raw_root)}")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


# ---------------------------------------------------------------- 主流程


def run(cfg: dict, only: list[str] | None, dry_run: bool, verify_only: bool) -> int:
    raw_root = cfg["data_dir"] / "raw"
    logfile = cfg["data_dir"] / "logs" / "download.log"
    raw_root.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(cfg, only)
    by_group: dict[str, int] = {}
    for t in tasks:
        g = Path(t.target).parent.name
        by_group[g] = by_group.get(g, 0) + 1

    log(f"配置: {cfg.get('_config_path', '')}")
    log(f"数据根目录: {cfg['data_dir']}")
    log(f"年份: {cfg['_years'][0]}—{cfg['_years'][-1]}  月份: {cfg['months']}  "
        f"时区: {cfg['time_zone']}  频率: {cfg['frequency']}")
    for g, n in by_group.items():
        log(f"  分组 {g}: {n} 个任务")
    log(f"任务总数: {len(tasks)}  预估体积: {estimate_gb(cfg, only):.2f} GB（未压缩）")

    if dry_run:
        log("--dry-run 结束，未发起任何请求")
        return 0

    if cdsswarm is None:
        log("❌ 未安装 cdsswarm，无法下载。服务器上执行: uv sync --extra download")
        return 1

    # ---- 断点续传第 1 层：跳过已存在且有效的文件
    pending = [t for t in tasks if not is_valid(Path(t.target))]
    log(f"已完成 {len(tasks) - len(pending)} / {len(tasks)}，待下载 {len(pending)}")
    if verify_only:
        n = write_manifest(raw_root, cfg["data_dir"] / "CHECKSUMS.sha256")
        log(f"--verify-only: 清单已写入，共 {n} 个有效文件")
        return 0 if not pending else 1

    # ---- 轮次循环：轮次级重试 + 退避
    backoff = BACKOFF_BASE
    for rnd in range(1, int(cfg["max_rounds"]) + 1):
        if _stop:
            break
        pending = [t for t in tasks if not is_valid(Path(t.target))]
        if not pending:
            break

        log(f"=== 第 {rnd}/{cfg['max_rounds']} 轮：{len(pending)} 个任务待下载 ===")
        try:
            results = cdsswarm.download(
                pending,
                num_workers=int(cfg["workers"]),
                skip_existing=True,   # 断点续传第 2 层
                reuse_jobs=True,      # 断点续传第 3 层：复用 CDS 上已提交的 job
                max_retries=int(cfg["max_retries"]),
                on_message=lambda m: log(str(m), logfile),
            )
        except Exception as exc:  # noqa: BLE001 —— 网络/服务端异常都不该让进程死掉
            log(f"本轮异常: {type(exc).__name__}: {exc}", logfile)
            results = []

        if not results:
            log("本轮未返回结果（被中断或全部失败），退避后重试", logfile)
        else:
            ok = sum(1 for r in results if r.success)
            log(f"本轮成功 {ok} / {len(results)}")
            for r in results:
                if not r.success:
                    log(f"  ✗ {Path(r.task.target).name}: {r.error}", logfile)

        # ---- 校验并清理坏文件，坏文件会在下一轮重下
        bad = [t for t in pending if not is_valid(Path(t.target))]
        for t in bad:
            p = Path(t.target)
            if p.exists():
                log(f"  🗑 删除无效文件以便重下: {p.name}", logfile)
                p.unlink()
        if not bad:
            log("全部任务完成 ✅")
            break

        log(f"仍有 {len(bad)} 个未完成，{backoff}s 后进入下一轮", logfile)
        for _ in range(backoff):
            if _stop:
                break
            time.sleep(1)
        backoff = min(backoff * 2, BACKOFF_CAP)

    # ---- 收尾：清点 + 生成校验和清单
    remaining = [t for t in tasks if not is_valid(Path(t.target))]
    n = write_manifest(raw_root, cfg["data_dir"] / "CHECKSUMS.sha256")
    log(f"有效文件 {n} 个，清单: {cfg['data_dir'] / 'CHECKSUMS.sha256'}")
    if remaining:
        log(f"❌ 仍有 {len(remaining)} 个任务未完成（可再次运行本脚本继续，不会重下已完成部分）")
        return 1
    log("🎉 全部完成")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ERA5 批量下载（带重试/断点续传/校验）")
    ap.add_argument("--config", default="configs/download.yaml", help="配置文件路径")
    ap.add_argument("--group", action="append", help="只处理指定分组，可重复")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划与体积估算")
    ap.add_argument("--verify-only", action="store_true", help="只校验并生成清单，不下载")
    args = ap.parse_args(argv)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)
    cfg["_config_path"] = str(cfg_path)
    return run(cfg, args.group, args.dry_run, args.verify_only)


if __name__ == "__main__":
    sys.exit(main())
