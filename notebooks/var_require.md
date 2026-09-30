## 一、需要下载的因子

| 梯队 | 变量 | CDS 变量名 | 层次 | 日统计口径 | 用途 |
|---|---|---|---|---|---|
| **一** | 2 m 气温 | `2m_temperature` | 单层 | **逐小时→日最高 = Tmax（目标）**；另存日均 | 目标变量 Y + 热状态记忆 |
| **一** | 海平面气压 | `mean_sea_level_pressure` | 单层 | 日均 | 副高/大尺度环流背景 |
| **一** | 海表温度 | `sea_surface_temperature` | 单层 | 日均 | 海洋慢变强迫（3—4 周关键） |
| **一** | 500 hPa 位势高度 | `geopotential` | 500 | 日均 | 副高形势（hgt500） |
| **一** | 850 hPa 风场 | `u/v_component_of_wind` | 850 | 日均 | 低层暖平流（u850/v850） |
| **二** | 总降水 | `total_precipitation` | 单层 | 日累计 | 干湿状态 → 土壤湿度 |
| **二** | 表层土壤湿度 | `volumetric_soil_water_layer_1` | 单层 | 日均（建议加 layer 2–4） | 陆面记忆（本工程证明很关键） |
| **二** | 地表短波辐射 | `surface_solar_radiation_downwards` | 单层 | 日累计 | 地表受热条件 |
| **二** | 地表潜热通量 | `surface_latent_heat_flux` | 单层 | 日均 | 陆—气反馈中介变量 |
| **二** | 2 m 露点 | `2m_dewpoint_temperature` | 单层 | 日均 | 近地面湿度 |
| **二** | 感热通量 | `surface_sensible_heat_flux` | 单层 | 日均 | 感热增强（可选） |
| **三** | 200 hPa 高度/温度/风 | `geopotential`/`temperature`/`u`/`v` | 200 | 日均 | 高层环流增量实验 |
| **三** | 总云量 | `total_cloud_cover` | 单层 | 日均 | 辐射调制 |
| **三** | 850 hPa 比湿 | `specific_humidity` | 850 | 日均 | 水汽背景（备选） |

**一个省 6 倍的工程要点**：只有 **Tmax 需要逐小时**（要求日最大值）。其余前兆因子最终都被聚合为 7 天窗口均值/累计，**用 00/06/12/18 四个时次完全够**，体积直接降到 1/6。气压层数据在 ARCO 里本来就是 6 小时分辨率，天然对齐。

---

## 二、区域与采样密度

| 项目 | 研究区（本工程口径） | **建议下载域** | 密度 | 理由 |
|---|---|---|---|---|
| 目标区 | 25—35°N, 105—125°E | **20—38°N, 100—128°E** | **0.25°**（ERA5 原生，~28 km） | 四周各留 2—3° 余量，供空间合成/环流诊断，避免插值边界失真 |
| 高空层 | 同上 | 同上 | 0.25°，**仅取 200/500/850** | 不要下全 37 层 |
| SST | — | **40°E—280°E(80°W), 10°S—25°N** | **1°×1°（不要 0.25°）** | 关键海区只做面积平均；按 0.25° 下 SST 体积会爆炸（十几倍） |
| 时间 | 1981—2025 | 同 | — | **扩展暖季 3/1—9/30**（214 天/年），为提前 4 周窗口留余量；分析窗口 5—8 月 |

**体积估算**（float32 未压缩；NetCDF 压缩后通常再小 2—4 倍）：

| 变量组 | 时次 | 45 年全时段 | 仅 2015—2025 |
|---|---|---|---|
| `t2m`（出 Tmax） | 逐小时 | ~7.6 GB | ~1.9 GB |
| 其余 6 个地表变量 | 4 次/天 | ~7.6 GB | ~1.9 GB |
| 高空 4—8 场 | 4 次/天 | ~5—10 GB | ~1.2—2.5 GB |
| SST @1° | 4 次/天 | ~1.3 GB | ~0.3 GB |
| **合计** | | **~22—27 GB** | **~5 GB** |

> 建议：**先下 2015—2025 的 ~5 GB 把链路跑通**，再补 1981—2014。

---

## 三、CDS 慢/排队怎么办

### A. 首选：改用 ARCO Zarr 直读，根本不进下载队列 ⭐

ECMWF 已把 ERA5 做成 Analysis-Ready、Cloud-Optimised 的 Zarr，用你的 **CDS API key 当 Bearer token** 直接 `xr.open_zarr` 流式读取子集，**不产生 CDS 请求作业、不排队**（见 [ARCO 数据总览](https://confluence.ecmwf.int/spaces/CKB/pages/639215345/Analysis+Ready+Cloud+Optimised+ARCO+Data)、[气压层 PUG](https://confluence.ecmwf.int/pages/viewpage.action?pageId=699689223)）：

```python
import os, xarray as xr
key = os.environ["CDSAPI_KEY"]

# 单层：逐小时，1940—今
sfc_geo  = "https://arco.datastores.ecmwf.int/cadl-arco-geo-002/arco/reanalysis_era5_single_levels/sfc/geoChunked.zarr"
# 气压层：6 小时，0.25°，1940—今
pl_geo   = "https://arco.datastores.ecmwf.int/cadl-arco-geo-048/arco/reanalysis_era5_pressure_levels/pl/geoChunked.zarr"

ds = xr.open_zarr(sfc_geo, consolidated=True,
                  storage_options={"headers": {"Authorization": f"Bearer {key}"}})
sub = ds["t2m"].sel(time=slice("1981","2025"),
                    latitude=slice(38,20), longitude=slice(100,128))
```

- 需要 `xarray zarr httpio fsspec`；长任务建议加 `obstore` 或 `aiohttp-retry` 做重试（官方给了两段示例代码）。
- **chunk 选型反直觉但很重要**：长时段小区域 → `geoChunked`；大区域短时段 → `timeChunked`。
- 这是 **Beta 服务 + 公平使用限流**，所以务必只取 `(time, lat, lon, level)` 子集，别全量拉。
- ⚠️ ARCO 做过"去累积"等预处理，**用前先 `print(ds)` 核对变量名与单位**，特别是 `tp`/`ssrd` 是否已去累积（决定你是要 `sum` 还是直接取）。

### B. 仍走 `cdsapi` 时的工程化策略

1. **切小**：按 `年 × 变量组 × 层次` 拆成小请求，小请求排队时间远短于大请求；
2. **只取需要的时次与层**：4 次/天 + 200/500/850，体积先降 6—12 倍；
3. **`area` 裁切** + `data_format: netcdf` + `download_format: unarchived`；
4. **提交—轮询—断点续传**：不要阻塞式等一个大请求，维护 request ID 清单，失败续跑；
5. **并发 3—6 条**（一次提几十条会一起排队）；同一请求**不要重复提交**；
6. **避开欧洲白天高峰**（北京 15:00—24:00 最慢），凌晨提交。

### C. 换源（通常最快）

- **[Google ARCO-ERA5](https://console.cloud.google.com/storage/browser/gcp-public-data-arco-era5)**：`gs://gcp-public-data-arco-era5`，ECMWF 官方论坛对"批量下载"的推荐答案就是它（见[论坛帖](https://prod.ecmwf-forum-prod.compute.cci2.ecmwf.int/t/recommended-strategy-for-bulk-downloads/14888/3)），GCS/S3 直读无队列。
- **[WeatherBench 2](https://weatherbench2.readthedocs.io/en/latest/data-guide.html)**：`gs://weatherbench2/datasets/era5/`，含 1959—2023 逐小时 0.25° 全 37 层，也有 6 小时 / 1.5° / 64×32 版本——**做区域平均用粗网格版就够，体积小一个量级**。
- **Google Earth Engine**：`ECMWF/ERA5/HOURLY`、`ECMWF/ERA5/DAILY`、`ECMWF/ERA5_LAND/DAILY_AGGR`——云端裁剪后导出 Drive，几乎不排队，特别适合"中国区 + 暖季"这种子集需求。
- **国内镜像**：国家青藏高原科学数据中心（data.tpdc.ac.cn）有 ERA5 中国区数据集；部分高校/超算有 THREDDS。国内速度快，但变量/时段常不全，需核对。
- **SST 单独换源**：NOAA OISST v2.1（0.25° 日）或 ERSSTv5（2° 月），比从 ERA5 下 SST 轻 1—2 个数量级，且是气候指数研究的标准做法。
- ERA5-Land 也有 ARCO，只做陆面时用它（0.1°、土壤层更细）。

### D. 我建议的实际打法

先用 **ARCO 直读或 GEE** 在几小时内拿到 2015—2025 的 MVP（~5 GB），把整条链路跑通；**同时**用 cdsapi 分年提交 1981—2014 的补录请求慢慢排队。两条通道产出统一落到同一 NetCDF 结构，下游零改动。
