# MSLP（海平面气压）数据字典

由 `scripts/extract_mslp_json.py` 从 `data/raw/mslp-json/` 提取生成。

## 输出文件

| 文件 | 变量 | 含义 | 单位 |
|---|---|---|---|
| `data/raw/mslp/mslp_YYYY.nc` | `pressure_msl` | 日平均海平面气压 | hPa |

维度 `(time, latitude, longitude)`，经纬度轴与 `data/raw/tmax` 的
13×25 / 0.5° 网格完全一致（28-34°N, 110-122°E）。

## 处理参数

| 项目 | 取值 |
|---|---|
| 源数据 | `data/raw/mslp-json/YYYY-MM.json`（Open-Meteo Archive，models=era5）|
| 取值字段 | `daily.pressure_msl_mean`（hPa）|
| 月份 | [5, 6, 7, 8]（5-8 月暖季）|
| 年份 | 1981-2025，其中实际有 5-8 月数据的有 45 年 |
| 总天数 | 5535 |
| 网格 | 从 325 个点插值回 13×25 / 0.5° 规则网格 |
| 插值方法 | 先 `scipy.griddata` 线性，边缘无法覆盖处退化为最近邻 |

## 已知事项

- **源数据网格有轻微不规则**：请求的是 13×25 的 0.5° 规则网格，
  但 Open-Meteo 把其中 **6 个点吸附到了它自己的细网格**上：
  请求 121.0°E → 返回 120.75°E（28.0/33.0°N 两处）、
  请求 122.0°E → 返回 121.75°E（29.5/31.0°N 两处）、
  请求 30.5°N → 返回 30.75°N、请求 120.5°E → 返回 120.25°E。
  本脚本把这 325 个点重新插值到目标规则网格，以便与其他资料对齐；
  这 6 个点带来的误差很小，但属于**已知的源数据缺陷**，应予记录。
- 同文件里的 `sea_surface_temperature_mean` **实测 325/325 个点全为 null**，
  因此本数据源对 SST **没有任何可用信息**；SST 仍应使用 `data/raw/sst/`。
- 若日后 `data/raw/mslp-json/` 补齐了更多年份，重跑本脚本即可覆盖更新。
