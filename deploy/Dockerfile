# 可选的 Docker 方案。
#
# ⚠️ 结论先说：这个任务用 systemd + uv 更简单，Docker 不是必需的。
#    见 docs/服务器部署指南.md 的「要不要用 Docker」一节。
#    如果你仍想用容器（例如要和其他服务一起编排），这个镜像可以直接用。
#
# 构建：
#   docker build -f deploy/Dockerfile -t tcc-download .
#
# 运行（注意两个挂载：数据 + 断点续传状态）：
#   docker run -d --name tcc-dl --restart unless-stopped \
#     -e TCC_DATA_DIR=/data -e XDG_CACHE_HOME=/cache \
#     -v /srv/tcc-data:/data -v /srv/tcc-cache:/cache \
#     tcc-download

FROM python:3.13-slim

# 只装运行时真正需要的系统库（xarray 读 NetCDF 要 libhdf5/libnetcdf）
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libhdf5-103-1 \
        libnetcdf19 \
        tini \
    && rm -rf /var/lib/apt/lists/*

# uv：用于按 lock 安装依赖
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# 先只拷依赖描述文件，最大化利用 Docker 层缓存
COPY pyproject.toml uv.lock ./
COPY src ./src
COPY configs ./configs

# 只装"下载"所需的最小依赖集。
# 注意：不要 `uv sync` 全量 —— 那会把 torch / cartopy / metpy 一起拖进来（>2 GB）。
RUN uv venv /app/.venv --python 3.13 \
    && uv pip install --python /app/.venv \
        "cdsapi>=0.7.7" \
        "cdsswarm>=0.3.0" \
        "pyyaml>=6.0" \
        "xarray>=2026.7.0" \
        "netcdf4>=1.7.4"

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    TERM=dumb \
    TCC_DATA_DIR=/data \
    XDG_CACHE_HOME=/cache

# 用 tini 当 PID 1，保证 SIGTERM 能正确传到 python
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "tcc.download_workers", "--config", "/app/configs/download.yaml"]
