import calendar
import json
import os
import time

import requests

start = (1987, 5)
API = "https://archive-api.open-meteo.com/v1/archive"
LATS = [28 + 0.5 * i for i in range(13)]
LONS = [110 + 0.5 * j for j in range(25)]
POINTS = [(la, lo) for la in LATS for lo in LONS]
OUT_DIR = "out"
SLEEP = 10.0
def download_month(year, month):
    path = os.path.join(OUT_DIR, "%d-%02d.json" % (year, month))
    if os.path.exists(path):
        print("exist:", path)
        return
    time.sleep(SLEEP)
    last_day = calendar.monthrange(year, month)[1]
    params = {
        "latitude": ",".join("%g" % la for la, _ in POINTS),
        "longitude": ",".join("%g" % lo for _, lo in POINTS),
        "start_date": "%d-%02d-01" % (year, month),
        "end_date": "%d-%02d-%d" % (year, month, last_day),
        "daily": "pressure_msl_mean,sea_surface_temperature_mean",
        "timezone": "Asia/Shanghai",
        "models": "era5",
    }

    while True:
        r = requests.get(API, params=params, timeout=120)
        if r.status_code == 429:
            print("limited")
            time.sleep(60)
            continue
        r.raise_for_status()
        break

    data = r.json()
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)
    print(path, f'do {len(data)}')


if __name__ == "__main__":
    for year in range(start[0], 2026):
        for month in range(start[1], 9):
            try:
                download_month(year, month)
            except KeyboardInterrupt:
                os._exit(1)
            
            
