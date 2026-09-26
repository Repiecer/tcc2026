"""测试 Open-Meteo archive API 的能力上限（点数、年份跨度、响应体积）。

它返回的正是 ERA5-Land 的日最高温，且支持 timezone=Asia/Shanghai（当地日界），
与 CDS 那条路的产品完全一致，但不需要排队也不需要认证。
"""
import json
import time
import urllib.parse
import urllib.request


def fetch(lats, lons, start, end, tz="Asia/Shanghai", models="era5_land"):
    params = {
        "latitude": ",".join(str(x) for x in lats),
        "longitude": ",".join(str(x) for x in lons),
        "start_date": start,
        "end_date": end,
        "daily": "temperature_2m_max",
        "models": models,
        "timezone": tz,
    }
    url = "https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(params)
    t0 = time.time()
    try:
        with urllib.request.urlopen(url, timeout=180) as resp:
            raw = resp.read()
    except Exception as exc:  # noqa: BLE001
        return None, time.time() - t0, str(exc)[:160]
    return raw, time.time() - t0, None


def grid(dlat, dlon, nlat, nlon):
    lats, lons = [], []
    for i in range(nlat):
        for j in range(nlon):
            lats.append(round(28 + i * dlat, 3))
            lons.append(round(110 + j * dlon, 3))
    return lats, lons


print("=== 测试 1：1 点 × 45 年 ===")
raw, el, err = fetch([30.0], [118.0], "1981-01-01", "2025-12-31")
if err:
    print("  失败:", err)
else:
    d = json.loads(raw)["daily"]
    print("  %.0f KB  %.1fs  天数=%d  %s → %s" % (
        len(raw) / 1024, el, len(d["time"]), d["time"][0], d["time"][-1]))

print()
print("=== 测试 2：25 点 × 45 年 ===")
la, lo = grid(0.5, 1.0, 5, 5)
raw, el, err = fetch(la, lo, "1981-01-01", "2025-12-31")
print("  " + ("失败: " + err if err else "%.0f KB  %.1fs" % (len(raw) / 1024, el)))

print()
print("=== 测试 3：325 点（0.5° 网格铺满区域）× 45 年 ===")
la, lo = grid(0.5, 0.5, 13, 25)
print("  点数:", len(la))
raw, el, err = fetch(la, lo, "1981-01-01", "2025-12-31")
print("  " + ("失败: " + err if err else "%.0f KB  %.1fs" % (len(raw) / 1024, el)))

print()
print("=== 测试 4：仅暖季（5—8 月）× 45 年，1 点 ===")
# Open-Meteo 支持 start/end 单区间，暖季需要分年取；这里测单年确认可行
raw, el, err = fetch([30.0], [118.0], "2022-05-01", "2022-08-31")
print("  " + ("失败: " + err if err else "%.0f KB  %.1fs" % (len(raw) / 1024, el)))

print()
print("=== 测试 5：一次能否覆盖 45 个暖季（用全年代替验证吞吐）===")
la, lo = grid(0.5, 0.5, 13, 25)
t0 = time.time()
raw, el, err = fetch(la, lo, "1981-01-01", "1990-12-31")
print("  " + ("失败: " + err if err else "%.0f KB  %.1fs  (10年×325点)" % (len(raw) / 1024, el)))
