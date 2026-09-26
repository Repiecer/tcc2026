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
from itertools import zip_longest
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
# ⚠️ CDS 对单账号"排队中请求数"有硬上限（实测约 6 个/数据集）。
# 撞上后新提交会被拒："Number queued requests for this dataset is temporarily limited"。
# 因此轮次间隔必须够长，等队列消化掉再继续，否则只是反复撞墙。
BACKOFF_BASE = 300         # 轮次间隔起始秒数（5 分钟）
BACKOFF_CAP = 3600         # 轮次间隔上限（1 小时）

_stop = False


def _import_sibling(module_name: str):
    """导入同目录模块。

    本脚本既可能以 ``python -m tcc.download_workers`` 运行（有包上下文），
    也可能以 ``python /opt/tcc/src/tcc/download_workers.py`` 直接运行
    （部署时为了避免装整个包，用的就是这种），后者没有包上下文，
    所以这里做一次回退。
    """
    import importlib
    try:
        return importlib.import_module(f"tcc.{module_name}")
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        return importlib.import_module(module_name)


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
        g["name"] = name
        # 只有走 CDS 的分组有 dataset / resolution；走其他源的分组由各自模块处理
        if "dataset" in g:
            g.setdefault(
                "resolution",
                LAND_RESOLUTION if "land" in g["dataset"] else DEFAULT_RESOLUTION,
            )

    return cfg


def _days_in(year: int, month: int) -> list[str]:
    """该年月的合法日期列表（零填充）。避免给 CDS 送不存在的日期。"""
    return [f"{d:02d}" for d in range(1, calendar.monthrange(year, month)[1] + 1)]


def build_tasks(cfg: dict, only: list[str] | None = None) -> list[Any]:
    """按 (年, 月) × 分组 生成 cdsswarm 任务列表。

    两个设计都是被 CDS 的配额实况逼出来的：

    1. **年块打包** —— CDS 对单个请求有硬上限（实测 372 天通过、465 天报
       "Your request is too large"）。按 ``year_batch`` 把连续若干年塞进一个请求，
       把请求数压到最少，因为**成功的提交次数才是瓶颈**，不是数据量。
    2. **分组交错** —— CDS 的排队配额按**数据集**算（报错原文 "for this dataset"）。
       若按分组顺序发（先发完所有 tmax），另外两个数据集的配额就白白闲置。
       这里把三组轮转交错，让 N 个 worker 分散到不同数据集，同时用满三份配额。

    未安装 cdsswarm 时退化为轻量占位对象，这样 ``--dry-run`` 在任何环境都能跑。
    """
    if cdsswarm is None:
        from types import SimpleNamespace
        make_task = lambda **kw: SimpleNamespace(**kw)  # noqa: E731
    else:
        make_task = cdsswarm.Task

    out_root = cfg["data_dir"] / "raw"
    per_group: list[list[Any]] = []
    batch = max(1, int(cfg.get("year_batch", 1)))
    years = cfg["_years"]
    year_blocks = [years[i:i + batch] for i in range(0, len(years), batch)]

    for name, g in cfg["groups"].items():
        if only and name not in only:
            continue
        if "dataset" not in g:          # 走 openmeteo / arco 的分组不产生 CDS 任务
            continue
        group_tasks: list[Any] = []
        for block in year_blocks:
            for month in cfg["months"]:
                month = int(month)
                # 同一年块内各年该月的合法日期取并集（只有闰年 2 月才有差异，本项目 5—8 月不受影响）
                days = sorted({d for y in block for d in _days_in(y, month)})
                request: dict[str, Any] = {
                    "variable": list(g["variables"]),
                    "year": [str(y) for y in block],
                    "month": [f"{month:02d}"],          # 零填充：ERA5-Land 只认 '05'
                    "day": days,
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

                tag = f"{block[0]}-{block[-1]}" if len(block) > 1 else f"{block[0]}"
                target = out_root / name / f"{name}_{tag}_{month:02d}.nc"
                group_tasks.append(
                    make_task(dataset=g["dataset"], request=request, target=str(target))
                )
        per_group.append(group_tasks)

    # ---- 轮转交错：tmax[0], sst_mslp[0], circ[0], tmax[1], ...
    # 这样 N 个 worker 会同时压在三份「按数据集」的配额上，而不是先把 tmax 跑完。
    interleaved: list[Any] = []
    for combo in zip_longest(*per_group):
        interleaved.extend(t for t in combo if t is not None)
    return interleaved


def estimate_gb(cfg: dict, only: list[str] | None = None) -> float:
    """粗略估算下载体积（未压缩 float32），用于 --dry-run。"""
    total = 0.0
    n_days = sum(calendar.monthrange(y, int(m))[1] for y in cfg["_years"] for m in cfg["months"])
    for name, g in cfg["groups"].items():
        if only and name not in only:
            continue
        if "resolution" not in g:       # 非 CDS 分组体积另行估算
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


# ---------------------------------------------------------------- 凭据预检


def preflight_credentials(cfg: dict, logfile: Path | None = None) -> bool:
    """在提交任何请求之前确认 CDS 凭据可用。

    为什么必须做这一步：cdsapi 的凭据查找顺序是
    ``$CDSAPI_RC`` → ``~/.cdsapirc``。手动在 shell 里跑脚本时，
    systemd unit 里的 ``Environment=CDSAPI_RC=`` 是不生效的，
    于是会去读 ``/root/.cdsapirc`` 而报
    "Missing/incomplete configuration file"。

    如果不预检，脚本会照样提交全部任务、全部失败，再按 30 轮退避白等几小时。
    这里提前拦住，并显式设置 ``CDSAPI_RC``，让 cdsswarm 内部创建的 client 也能读到。
    """
    configured = cfg.get("cdsapirc") or os.environ.get("CDSAPI_RC")
    path = Path(configured).expanduser() if configured else Path("~/.cdsapirc").expanduser()

    if not path.exists():
        log(f"❌ 找不到 CDS 凭据文件: {path}", logfile)
        log("   当前用户 HOME = " + os.path.expanduser("~"), logfile)
        log("   三种修法（任选其一）:", logfile)
        log("     A) 放到 cdsapi 的默认位置（最省事）:", logfile)
        log(f"        cp /opt/tcc/.cdsapirc {path}  &&  chmod 600 {path}", logfile)
        log("     B) 在 configs/download.yaml 里加一行:", logfile)
        log("        cdsapirc: /opt/tcc/.cdsapirc", logfile)
        log("     C) 在当前 shell 里 export:", logfile)
        log("        export CDSAPI_RC=/opt/tcc/.cdsapirc", logfile)
        return False

    text = path.read_text(encoding="utf-8", errors="ignore")
    if "url" not in text or "key" not in text:
        log(f"❌ 凭据文件内容不完整，需要 url 和 key 两行: {path}", logfile)
        log("   正确格式:", logfile)
        log("     url: https://cds.climate.copernicus.eu/api", logfile)
        log("     key: <你的 API Key>", logfile)
        return False

    # 显式落到环境变量里，保证 cdsswarm 内部 new 出来的 Client 也读得到
    os.environ["CDSAPI_RC"] = str(path)
    mode = path.stat().st_mode & 0o777
    warn = "  ⚠ 权限过宽，建议 chmod 600" if mode & 0o077 else ""
    log(f"✓ CDS 凭据: {path}{warn}", logfile)
    return True


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

    # ---- 非 CDS 数据源：直接取数，不受 CDS 排队与配额影响
    for src, module_name in (("openmeteo", "openmeteo_fetch"), ("arco", "arco_fetch")):
        names = [n for n, g in cfg["groups"].items()
                 if g.get("source") == src and (not only or n in only)]
        if not names:
            continue
        mod = _import_sibling(module_name)
        for n in names:
            try:
                mod.fetch_group(cfg, cfg["groups"][n], logfile)
            except Exception as exc:  # noqa: BLE001
                log(f"❌ [{src}] 分组 {n} 取数失败: {type(exc).__name__}: {exc}", logfile)

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

    # ---- 凭据预检：仅在确实有 CDS 任务时才需要
    if tasks and not preflight_credentials(cfg, logfile):
        return 2

    # ---- 轮次循环：轮次级重试 + 退避
    backoff = BACKOFF_BASE
    prev_errors: frozenset[str] | None = None
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

        ok = 0
        errors: frozenset[str] = frozenset()
        if not results:
            log("本轮未返回结果（被中断或全部失败），退避后重试", logfile)
        else:
            ok = sum(1 for r in results if r.success)
            errors = frozenset(r.error for r in results if not r.success and r.error)
            log(f"本轮成功 {ok} / {len(results)}")
            for r in results:
                if not r.success:
                    log(f"  ✗ {Path(r.task.target).name}: {r.error}", logfile)

            # 连续两轮「零成功 + 完全相同」的错误 → 判定为不可重试的问题
            # （凭据错误、参数非法、变量名不存在等），提前退出而不是白等几小时
            if ok == 0 and errors and errors == prev_errors:
                log("❌ 连续两轮出现完全相同的失败，判定为不可重试错误，提前退出", logfile)
                for e in sorted(errors)[:3]:
                    log(f"   原因: {e}", logfile)
                break
        prev_errors = errors or prev_errors

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
        # 本轮有文件真正落盘 == 配额正在放行 → 把退避重置回起点。
        # 否则退避会一路翻倍到 1 小时封顶再也不降，白等很久。
        if len(bad) < len(pending):
            if backoff != BACKOFF_BASE:
                log(f"  ↺ 本轮有 {len(pending) - len(bad)} 个文件落盘，退避重置为 {BACKOFF_BASE}s", logfile)
            backoff = BACKOFF_BASE
        else:
            backoff = min(backoff * 2, BACKOFF_CAP)
        for _ in range(backoff):
            if _stop:
                break
            time.sleep(1)

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
