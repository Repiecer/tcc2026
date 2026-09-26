#!/usr/bin/env python3
"""挑战杯官方"人才项目库"关键词扫描工具。

用途：抓取项目库列表页，抽取「项目名称 / 学校 / 奖项 / 年份 / 届次」，
      再按关键词筛选，用于选题对标（见 ../cases/09_项目库扫描_同类选题清单.md）。

用法：
    python examples/tools/scan_tiaozhanbei.py --pages 238 --out scan.json
    python examples/tools/scan_tiaozhanbei.py --report      # 只打印关键词命中统计

说明：
  * 该站点是第三方镜像，官方库有反爬保护；请控制并发与频率，避免给站点造成压力。
  * 抓取的原始 HTML 缓存在脚本同级的 .cache/ 目录，可随时删除。
  * 扫描结果只用于选题参考，奖项级别请以官方公示为准。
"""
from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://www.z-legko.com/xmk/project/list/project/"
CACHE = Path(__file__).parent / ".cache"
UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120 Safari/537.36"
)

# 与本课题直接相关的关键词（用于命中统计）
KEYWORDS = [
    "热岛", "城市热", "微气候", "热浪", "高温", "气温",
    "延伸期", "次季节", "季节内", "MJO",
    "土壤湿度", "陆气", "蒸散", "副高", "位势高度", "环流",
    "气象", "气候", "台风", "内涝", "空气质量", "雾霾", "赤潮",
]


def fetch_page(page: int, retries: int = 2) -> str:
    """抓取单页（带磁盘缓存）。"""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{page}.html"
    if path.exists() and path.stat().st_size > 5000:
        return path.read_text(encoding="utf-8", errors="ignore")

    for attempt in range(retries + 1):
        proc = subprocess.run(
            ["curl", "-sL", "-m", "30", "-A", UA, f"{BASE}?page={page}"],
            capture_output=True,
        )
        text = proc.stdout.decode("utf-8", errors="ignore")
        if len(text) > 5000:
            path.write_text(text, encoding="utf-8")
            time.sleep(0.3)
            return text
        time.sleep(1.0 * (attempt + 1))
    return ""


def parse_page(text: str) -> list[dict]:
    """从列表页 HTML 中抽取项目卡片。"""
    rows = []
    blocks = re.split(r'href="//www\.z-legko\.com/xmk/project/(\d+)/"', text)
    for i in range(1, len(blocks) - 1, 2):
        pid, body = blocks[i], blocks[i + 1]
        flat = html.unescape(re.sub(r"(?s)<[^>]+>", "|", body))
        parts = [x.strip() for x in flat.split("|") if x.strip() and x.strip() != ">详情"]
        if len(parts) >= 4:
            rows.append(
                {
                    "id": pid,
                    "title": re.sub(r"\s+", " ", parts[0]),
                    "school": parts[1],
                    "award": parts[2],
                    "year": parts[3],
                    "edition": parts[4] if len(parts) > 4 else "",
                }
            )
    return rows


def crawl(pages: int, workers: int = 8) -> list[dict]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        texts = list(pool.map(fetch_page, range(1, pages + 1)))

    projects, seen = [], set()
    for text in texts:
        for row in parse_page(text):
            if row["id"] not in seen:
                seen.add(row["id"])
                projects.append(row)
    return projects


def report(projects: list[dict]) -> None:
    blob = lambda p: " ".join(p.values())  # noqa: E731
    print(f"项目总数：{len(projects)}")
    print(f"届次分布：{dict(Counter(p['edition'] for p in projects))}")
    print("\n关键词命中：")
    for kw in KEYWORDS:
        hits = [p for p in projects if kw in blob(p)]
        print(f"  {kw:<8} {len(hits):>4}")
        if kw in ("高温", "气象", "气候") and hits:
            for h in hits[:8]:
                print(f"        - {h['title'][:60]}（{h['school']}）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=238, help="抓取页数")
    ap.add_argument("--workers", type=int, default=8, help="并发数（请勿过大）")
    ap.add_argument("--out", default="scan.json", help="结果输出文件")
    ap.add_argument("--report", action="store_true", help="只打印关键词命中统计")
    args = ap.parse_args()

    projects = crawl(args.pages, args.workers)
    Path(args.out).write_text(
        json.dumps(projects, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    report(projects)
    print(f"\n已写入 {args.out}")


if __name__ == "__main__":
    main()
