"""找出 Open-Meteo archive API 的单次请求点数上限（并给足重试间隔）。"""
import json
import time
import urllib.error
import urllib.parse
import urllib.request


def fetch(nlat, nlon, dlat=0.5, dlon=0.5, start="1981-01-01", end="2025-12-31"):
    lats, lons = [], []
    for i in range(nlat):
        for j in range(nlon):
            lats.append(round(28 + i * dlat, 3))
            lons.append(round(110 + j * dlon, 3))
    params = {
        "latitude": ",".join(str(x) for x in lats),
        "longitude": ",".join(str(x) for x in lons),
        "start_date": start, "end_date": end,
        "daily": "temperature_2m_max", "models": "era5_land",
        "timezone": "Asia/Shanghai",
    }
    url = "https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(params)
    t0 = time.time()
    try:
        with urllib.request.urlopen(url, timeout=180) as resp:
            raw = resp.read()
        return len(lats), len(raw), time.time() - t0, None
    except urllib.error.HTTPError as exc:
        body = exc.read().decode()[:120]
        return len(lats), 0, time.time() - t0, "HTTP %s %s" % (exc.code, body)
    except Exception as exc:  # noqa: BLE001
        return len(lats), 0, time.time() - t0, str(exc)[:100]


print("=== 单次请求点数上限（每次间隔 12s 避开限流）===")
for nlat, nlon in ((5, 5), (5, 10), (8, 10), (10, 13), (13, 25)):
    n, size, el, err = fetch(nlat, nlon)
    flag = "失败 " + err if err else "%.0f KB  %.1fs" % (size / 1024, el)
    print("  %3d 点  %s" % (n, flag))
    if err and "429" in err:
        print("      → 触发限流，等待 45s 再试")
        time.sleep(45)
    else:
        time.sleep(12)
