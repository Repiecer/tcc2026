#!/usr/bin/env bash
# TCC ERA5 下载服务安装脚本
#
# 作用：自动探测「代码在哪、venv 在哪、数据放哪」，据此生成并安装 systemd unit，
#       避免手工改 unit 时路径写错导致 203/EXEC（Unable to locate executable）。
#
# 用法：
#   sudo bash deploy/install.sh                 # 数据目录默认 /srv/tcc-data
#   sudo bash deploy/install.sh /data/tcc       # 自定义数据目录
#   bash deploy/install.sh --print-unit         # 只打印将要写入的 unit，不安装（不需 root）
#   sudo bash deploy/install.sh --uninstall     # 卸载
#
set -euo pipefail

SERVICE_NAME="tcc-download"
UNIT_DIR="${TCC_UNIT_DIR:-/etc/systemd/system}"

# ---------- 定位仓库根目录（本脚本在 <repo>/deploy/ 下）----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------- 解析参数 ----------
UNINSTALL=0
PRINT_UNIT=0
DATA_DIR=""
for arg in "$@"; do
    case "$arg" in
        --uninstall)  UNINSTALL=1 ;;
        --print-unit) PRINT_UNIT=1 ;;
        -h|--help)    sed -n '2,12p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *)            DATA_DIR="$arg" ;;
    esac
done
DATA_DIR="${DATA_DIR:-/srv/tcc-data}"

# ---------- 卸载 ----------
if [[ "$UNINSTALL" == "1" ]]; then
    [[ "$(id -u)" == "0" ]] || { echo "❌ 卸载需要 root" >&2; exit 1; }
    systemctl disable --now "${SERVICE_NAME}.timer" 2>/dev/null || true
    systemctl stop "${SERVICE_NAME}.service" 2>/dev/null || true
    rm -f "${UNIT_DIR}/${SERVICE_NAME}.service" "${UNIT_DIR}/${SERVICE_NAME}.timer"
    systemctl daemon-reload
    echo "✅ 已卸载（数据目录 ${DATA_DIR} 未删除）"
    exit 0
fi

if [[ "$PRINT_UNIT" == "0" && "$(id -u)" != "0" ]]; then
    echo "❌ 请用 root 运行：sudo bash deploy/install.sh" >&2
    echo "   （只想看生成的 unit 可以跑：bash deploy/install.sh --print-unit）" >&2
    exit 1
fi

echo "仓库目录 : ${REPO_DIR}"
echo "数据目录 : ${DATA_DIR}"
echo

# ---------- 1. 找 Python 解释器 ----------
PYTHON=""
for cand in \
    "${REPO_DIR}/.venv/bin/python" \
    "${REPO_DIR}/venv/bin/python" \
    "${REPO_DIR}/.venv-dl/bin/python" \
    "${HOME}/.venv/bin/python"
do
    if [[ -x "$cand" ]]; then PYTHON="$cand"; break; fi
done

if [[ -z "$PYTHON" ]]; then
    echo "❌ 找不到可用的 Python 解释器（虚拟环境不存在）" >&2
    echo "   已查找：" >&2
    echo "     ${REPO_DIR}/.venv/bin/python" >&2
    echo "     ${REPO_DIR}/venv/bin/python" >&2
    echo "     ${REPO_DIR}/.venv-dl/bin/python" >&2
    echo >&2
    echo "   请先创建最小环境（不要 uv sync，那会装 6.8 GB）：" >&2
    echo "     uv venv ${REPO_DIR}/.venv --python 3.13" >&2
    echo "     uv pip install --python ${REPO_DIR}/.venv \\" >&2
    echo "         'cdsapi>=0.7.7' 'cdsswarm>=0.3.0' 'pyyaml>=6.0' 'xarray>=2026.7.0' 'netcdf4>=1.7.4'" >&2
    exit 1
fi
echo "✓ Python  : ${PYTHON}"
# 关键：uv 建的 venv，bin/python 是指向 uv 托管 Python 的符号链接，
# 而托管目录默认在 ~/.local/share/uv/python/ 下（root 就是 /root/.local/...）。
# 若开了 ProtectHome=true，systemd 就看不到链接目标，
# 会报 "Unable to locate executable" 和 203/EXEC —— 极难排查。
PYTHON_REAL="$(readlink -f "${PYTHON}" 2>/dev/null || echo "${PYTHON}")"
echo "  真实解释器: ${PYTHON_REAL}"

# 验证解释器真的能跑，且下载所需的依赖齐全
if ! "$PYTHON" -c "
import importlib.util as u, sys
missing = [m for m in ('cdsapi','cdsswarm','yaml') if u.find_spec(m) is None]
print('缺失依赖: ' + ', '.join(missing), file=sys.stderr) if missing else None
sys.exit(1 if missing else 0)
" ; then
    echo "❌ ${PYTHON} 里缺少依赖，请补装：" >&2
    echo "   uv pip install --python ${PYTHON} 'cdsapi>=0.7.7' 'cdsswarm>=0.3.0' 'pyyaml>=6.0' 'xarray>=2026.7.0' 'netcdf4>=1.7.4'" >&2
    exit 1
fi
echo "✓ 依赖    : cdsapi / cdsswarm / pyyaml 齐全"

# ---------- 2. 找入口脚本与配置 ----------
ENTRY="${REPO_DIR}/src/tcc/download_workers.py"
CONFIG="${REPO_DIR}/configs/download.yaml"
for f in "$ENTRY" "$CONFIG"; do
    if [[ ! -f "$f" ]]; then echo "❌ 找不到：${f}" >&2; exit 1; fi
done
echo "✓ 入口    : ${ENTRY}"
echo "✓ 配置    : ${CONFIG}"

# ---------- 3. CDS 凭据 ----------
RC=""
for cand in "${REPO_DIR}/.cdsapirc" "${HOME}/.cdsapirc"; do
    [[ -f "$cand" ]] && { RC="$cand"; break; }
done
if [[ -z "$RC" ]]; then
    echo "⚠️  未找到 CDS 凭据（${REPO_DIR}/.cdsapirc 或 ~/.cdsapirc）"
    echo "   请创建后再启动，否则下载会全部失败："
    echo "     cat > ${REPO_DIR}/.cdsapirc <<'EOF'"
    echo "     url: https://cds.climate.copernicus.eu/api"
    echo "     key: <你的 API Key>"
    echo "     EOF"
    echo "     chmod 600 ${REPO_DIR}/.cdsapirc"
else
    echo "✓ 凭据    : ${RC}"
fi

# ---------- 4. 数据目录 ----------
if [[ "$PRINT_UNIT" == "0" ]]; then
    mkdir -p "${DATA_DIR}"/{raw,logs}
    echo "✓ 数据    : ${DATA_DIR}（已创建）"
fi
echo

# ---------- 5. ProtectHome 条件判断 ----------
# ProtectHome=true 会让 /home、/root、/run/user 不可读。
# 判定必须覆盖三处：代码目录、数据目录、以及**解释器解析后的真实路径**。
# 最容易漏的就是第三个 —— uv 托管的 Python 恰好就在 ~/.local/share/uv/python/ 下。
HARDEN_HOME="ProtectHome=true"
for pair in "代码目录:${REPO_DIR}" "数据目录:${DATA_DIR}" "解释器:${PYTHON_REAL}"; do
    label="${pair%%:*}"
    path="${pair#*:}"
    for prefix in /home /root /run/user; do
        case "${path}" in
            "${prefix}"|"${prefix}"/*)
                HARDEN_HOME="# ProtectHome 已省略：${label} ${path} 位于 ${prefix} 下，启用后服务将读不到它"
                ;;
        esac
    done
done
if [[ "${HARDEN_HOME}" == "ProtectHome=true" ]]; then
    echo "✓ 加固    : ProtectHome=true 可用"
else
    echo "⚠ 加固    : 已关闭 ProtectHome（有路径位于家目录下）"
fi

RC_ENV="# 未找到 CDS 凭据文件，请按上面的提示创建"
[[ -n "$RC" ]] && RC_ENV="Environment=CDSAPI_RC=${RC}"

# ---------- 6. 渲染 unit ----------
render_service() {
    cat <<EOF
# 本文件由 deploy/install.sh 自动生成；要改配置请改脚本后重跑，不要手改这里
[Unit]
Description=TCC ERA5 bulk download (retry + resume)
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
WorkingDirectory=${REPO_DIR}

Environment=TCC_DATA_DIR=${DATA_DIR}
# cdsswarm 的断点续传状态存这里；不设会落到 ~/.cache → 重启后丢失 → 重新排队
Environment=XDG_CACHE_HOME=${DATA_DIR}/.cache
${RC_ENV}
Environment=TERM=dumb

ExecStart=${PYTHON} ${ENTRY} --config ${CONFIG}

TimeoutStartSec=infinity
KillSignal=SIGTERM
# cdsswarm 不一定会及时响应 SIGTERM（它会先跑完手上的重试），
# 所以给 60s 收尾，之后由 systemd 发 SIGKILL。设太长会让 systemctl stop 卡住。
TimeoutStopSec=60

StandardOutput=journal
StandardError=journal
SyslogIdentifier=${SERVICE_NAME}

NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=full
${HARDEN_HOME}
ReadWritePaths=${DATA_DIR}

[Install]
WantedBy=multi-user.target
EOF
}

render_timer() {
    cat <<EOF
# 本文件由 deploy/install.sh 自动生成
[Unit]
Description=Periodically re-run the TCC ERA5 downloader

[Timer]
OnBootSec=2min
OnUnitActiveSec=30min
Unit=${SERVICE_NAME}.service

[Install]
WantedBy=timers.target
EOF
}

# ---------- 7. --print-unit：只输出不安装 ----------
if [[ "$PRINT_UNIT" == "1" ]]; then
    echo "=================== ${SERVICE_NAME}.service ==================="
    render_service
    echo "=================== ${SERVICE_NAME}.timer ==================="
    render_timer
    exit 0
fi

# ---------- 8. 安装前自检：ExecStart 能不能真的跑起来 ----------
# 这一步就是专门用来提前抓住 203/EXEC 的
if ! "$PYTHON" "$ENTRY" --config "$CONFIG" --dry-run >/dev/null 2>&1; then
    echo "❌ dry-run 自检失败，请手动运行确认问题：" >&2
    echo "     ${PYTHON} ${ENTRY} --config ${CONFIG} --dry-run" >&2
    echo "   未安装 unit，请修好后重跑本脚本。" >&2
    exit 1
fi
echo "✓ dry-run 自检通过"

# ---------- 9. 写入并启用 ----------
mkdir -p "${UNIT_DIR}"
render_service > "${UNIT_DIR}/${SERVICE_NAME}.service"
render_timer   > "${UNIT_DIR}/${SERVICE_NAME}.timer"
[[ -n "$RC" ]] && chmod 600 "$RC" 2>/dev/null || true
echo "✓ 已写入 ${UNIT_DIR}/${SERVICE_NAME}.{service,timer}"

systemctl daemon-reload
systemctl enable --now "${SERVICE_NAME}.timer"

echo
echo "🎉 安装完成"
echo
echo "  看日志： journalctl -u ${SERVICE_NAME} -f"
echo "  看状态： systemctl status ${SERVICE_NAME}.timer"
echo "  立即跑： systemctl start ${SERVICE_NAME}.service"
echo "  看进度： find ${DATA_DIR}/raw -name '*.nc' | wc -l; du -sh ${DATA_DIR}/raw"
echo "  卸载：   sudo bash ${REPO_DIR}/deploy/install.sh --uninstall"
