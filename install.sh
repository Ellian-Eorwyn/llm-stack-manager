#!/usr/bin/env bash
# Install the git-friendly core LLM stack without touching any older stack tree.
set -euo pipefail

STACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/cross-platform.sh
source "${STACK_DIR}/scripts/cross-platform.sh"

CONFIG_DIR="${STACK_DIR}/config"
CONFIG_FILE="${CONFIG_DIR}/llm-stack.env"
EXAMPLE_CONFIG="${CONFIG_DIR}/llm-stack.env.example"
HONCHO_ENV_TEMPLATE="${CONFIG_DIR}/honcho.env.example"
HONCHO_ENV_FILE="${CONFIG_DIR}/honcho.env"
SERVICE_USER="$(cp_stat_user "${STACK_DIR}")"
SERVICE_GROUP="$(cp_stat_group "${STACK_DIR}")"

if [[ "${EUID}" -ne 0 ]]; then
    echo "Run with sudo: sudo bash ${STACK_DIR}/install.sh" >&2
    exit 1
fi

echo "=== LLM Stack Core Installer ==="
echo "Stack directory: ${STACK_DIR}"
echo "Service user:    ${SERVICE_USER}:${SERVICE_GROUP}"

mkdir -p "${STACK_DIR}/models" "${STACK_DIR}/logs" "${STACK_DIR}/deps" "${CONFIG_DIR}" "${CONFIG_DIR}/saved" "${CONFIG_DIR}/chat-templates"
chmod 755 "${STACK_DIR}/scripts"/*.sh "${STACK_DIR}/validate.sh" "${STACK_DIR}/scripts/install-dependencies.py"
chmod 755 "${STACK_DIR}/playwright"/*.sh 2>/dev/null || true

if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "Creating local config: ${CONFIG_FILE}"
    sed -e "s|@STACK_DIR@|${STACK_DIR}|g" -e "s|@SERVICE_USER@|${SERVICE_USER}|g" "${EXAMPLE_CONFIG}" > "${CONFIG_FILE}"
else
    echo "Keeping existing local config: ${CONFIG_FILE}"
fi

merge_honcho_config_defaults() {
    python3 - "${EXAMPLE_CONFIG}" "${CONFIG_FILE}" "${STACK_DIR}" "${SERVICE_USER}" <<'PYMERGEHONCHO'
import re
import sys
from pathlib import Path

example = Path(sys.argv[1])
config = Path(sys.argv[2])
stack_dir = sys.argv[3]
service_user = sys.argv[4]
content = config.read_text(encoding="utf-8")
existing = set(re.findall(r"^([A-Za-z_][A-Za-z0-9_]*)=", content, re.MULTILINE))
missing = []
for line in example.read_text(encoding="utf-8").splitlines():
    if not line.startswith("HONCHO_") or "=" not in line:
        continue
    key = line.split("=", 1)[0]
    if key in existing:
        continue
    rendered = line.replace("@STACK_DIR@", stack_dir).replace("@SERVICE_USER@", service_user)
    missing.append(rendered)
if missing:
    if content and not content.endswith("\n"):
        content += "\n"
    content += "\n# Local Honcho memory service defaults added by install.sh\n"
    content += "\n".join(missing) + "\n"
    config.write_text(content, encoding="utf-8")
PYMERGEHONCHO
}
merge_honcho_config_defaults

merge_config_defaults() {
    python3 - "${EXAMPLE_CONFIG}" "${CONFIG_FILE}" "${STACK_DIR}" "${SERVICE_USER}" <<'PYMERGEDEFAULTS'
import re
import sys
from pathlib import Path

example = Path(sys.argv[1])
config = Path(sys.argv[2])
stack_dir = sys.argv[3]
service_user = sys.argv[4]
content = config.read_text(encoding="utf-8")
existing = set(re.findall(r"^([A-Za-z_][A-Za-z0-9_]*)=", content, re.MULTILINE))
missing = []
for line in example.read_text(encoding="utf-8").splitlines():
    if not line or line.startswith("#") or "=" not in line:
        continue
    key = line.split("=", 1)[0]
    if key in existing:
        continue
    rendered = line.replace("@STACK_DIR@", stack_dir).replace("@SERVICE_USER@", service_user)
    missing.append(rendered)
if missing:
    if content and not content.endswith("\n"):
        content += "\n"
    content += "\n# Missing defaults added by install.sh\n"
    content += "\n".join(missing) + "\n"
    config.write_text(content, encoding="utf-8")
PYMERGEDEFAULTS
}
merge_config_defaults

repair_glmocr_sdk_config() {
    python3 - "${CONFIG_FILE}" <<'PYREPAIRGLMOCR'
import re
import sys
from pathlib import Path

config = Path(sys.argv[1])
content = config.read_text(encoding="utf-8")

def set_env(content: str, key: str, value: str) -> str:
    rendered = '""' if value == "" else value
    pattern = re.compile(r"^" + re.escape(key) + r"=.*$", re.MULTILINE)
    if pattern.search(content):
        return pattern.sub(f"{key}={rendered}", content, count=1)
    if content and not content.endswith("\n"):
        content += "\n"
    return content + f"{key}={rendered}\n"

def env_value(content: str, key: str) -> str | None:
    match = re.search(r"^" + re.escape(key) + r"=(.*)$", content, re.MULTILINE)
    if not match:
        return None
    value = match.group(1).strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        value = value[1:-1]
    return value

layout_gpus = env_value(content, "GLMOCR_LAYOUT_CUDA_VISIBLE_DEVICES")
if layout_gpus is None:
    content = set_env(content, "GLMOCR_LAYOUT_CUDA_VISIBLE_DEVICES", "")
elif "," in layout_gpus:
    content = set_env(content, "GLMOCR_LAYOUT_CUDA_VISIBLE_DEVICES", (layout_gpus.split(",", 1)[0].strip() or "0"))

layout_device = env_value(content, "GLMOCR_LAYOUT_DEVICE")
if layout_device and layout_device.startswith("cuda:") and "," in layout_device:
    content = set_env(content, "GLMOCR_LAYOUT_DEVICE", "cuda:" + (layout_device.removeprefix("cuda:").split(",", 1)[0].strip() or "0"))

config.write_text(content, encoding="utf-8")
PYREPAIRGLMOCR
}
repair_glmocr_sdk_config

# shellcheck source=/dev/null
source "${CONFIG_FILE}"

create_honcho_env() {
    if [[ -f "${HONCHO_ENV_FILE}" ]]; then
        echo "Keeping existing local Honcho env: ${HONCHO_ENV_FILE}"
        chmod 600 "${HONCHO_ENV_FILE}"
        return
    fi
    if [[ ! -f "${HONCHO_ENV_TEMPLATE}" ]]; then
        echo "Missing Honcho env template: ${HONCHO_ENV_TEMPLATE}" >&2
        exit 1
    fi
    local db_password
    db_password="$(python3 - <<'PYHONCHOPASS'
import secrets
print(secrets.token_hex(24))
PYHONCHOPASS
)"
    echo "Creating local Honcho env: ${HONCHO_ENV_FILE}"
    sed \
      -e "s|@HONCHO_DB_PASSWORD@|${db_password}|g" \
      -e "s|@HONCHO_LLM_MODEL@|${HONCHO_LLM_MODEL}|g" \
      -e "s|@HONCHO_LLM_BASE_URL@|${HONCHO_LLM_BASE_URL}|g" \
      -e "s|@HONCHO_EMBED_MODEL@|${HONCHO_EMBED_MODEL}|g" \
      -e "s|@HONCHO_EMBED_BASE_URL@|${HONCHO_EMBED_BASE_URL}|g" \
      -e "s|@HONCHO_EMBED_VECTOR_DIMENSIONS@|${HONCHO_EMBED_VECTOR_DIMENSIONS}|g" \
      "${HONCHO_ENV_TEMPLATE}" > "${HONCHO_ENV_FILE}"
    chmod 600 "${HONCHO_ENV_FILE}"
}

if [[ "${HONCHO_ENABLED:-off}" == "on" ]]; then
    create_honcho_env
fi

chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "${STACK_DIR}"

if [[ "${HONCHO_ENABLED:-off}" == "on" && "${HONCHO_INSTALL_DATASTORES:-on}" == "on" ]]; then
    echo "Installing/configuring local Honcho PostgreSQL/pgvector and Redis..."
    SERVICE_USER="${SERVICE_USER}" bash "${STACK_DIR}/scripts/install-honcho-system-deps.sh"
fi

if [[ "${LLM_STACK_SKIP_DEP_UPDATE:-0}" == "1" ]]; then
    echo "Skipping dependency update because LLM_STACK_SKIP_DEP_UPDATE=1."
else
    echo "Installing/updating dependencies from dependencies.json..."
    if ! sudo -u "${SERVICE_USER}" env HONCHO_ENABLED="${HONCHO_ENABLED:-off}" "${STACK_DIR}/scripts/install-dependencies.py" --update; then
        if [[ "${HONCHO_ENABLED:-off}" == "on" ]]; then
            echo "Dependency update failed while Honcho is enabled." >&2
            exit 1
        fi
        if [[ -x "${LLAMA_SERVER_BIN:-${STACK_DIR}/deps/llama.cpp/build/bin/llama-server}" ]]; then
            echo "Dependency update failed, but an existing llama-server binary is present; continuing with systemd unit installation." >&2
        else
            echo "Dependency update failed and no llama-server binary is available." >&2
            exit 1
        fi
    fi
fi

if [[ "${SEARXNG_ENABLED:-on}" == "on" ]]; then
    echo "Installing/configuring local SearXNG..."
    bash "${STACK_DIR}/scripts/install-searxng.sh"
fi

if [[ "${PLAYWRIGHT_ENABLED:-on}" == "on" ]]; then
    echo "Installing/configuring local Playwright server..."
    sudo -u "${SERVICE_USER}" env PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-${STACK_DIR}/playwright/browsers}" bash "${STACK_DIR}/scripts/install-playwright.sh"
fi

if is_linux; then
    install_playwright_nginx_conf() {
        local url_path="${PLAYWRIGHT_URL_PATH:-/playwright}"
        local nginx_conf="${PLAYWRIGHT_NGINX_CONF:-/etc/nginx/default.apps-available/playwright.conf}"
        local port="${PLAYWRIGHT_PORT:-3001}"
        [[ "${url_path}" == /* ]] || url_path="/${url_path}"
        local url_path_slash="${url_path%/}/"
        mkdir -p "$(dirname "${nginx_conf}")" /etc/nginx/default.d
        cat > "${nginx_conf}" <<NGINX
location = ${url_path} {
    return 308 ${url_path_slash};
}

location ${url_path_slash} {
    proxy_pass http://127.0.0.1:${port}/;
    proxy_http_version 1.1;
    proxy_set_header Host \$host;
    proxy_set_header X-Real-IP \$remote_addr;
    proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto \$scheme;
    proxy_set_header X-Forwarded-Prefix ${url_path};
    proxy_set_header X-Script-Name ${url_path};
    proxy_set_header Upgrade \$http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_read_timeout 3600s;
    proxy_send_timeout 3600s;
}
NGINX
        chmod 644 "${nginx_conf}"
        ln -sfn "${nginx_conf}" /etc/nginx/default.d/playwright.conf
        echo "  installed: nginx playwright location ${url_path}"
    }

    install_unit() {
        local unit_name="$1"
        local description="$2"
        local script="$3"
        local timeout="${4:-300}"
        cat > "/etc/systemd/system/${unit_name}.service" <<UNIT
[Unit]
Description=${description}
After=network.target

[Service]
Type=simple
User=${SERVICE_USER}
Group=${SERVICE_GROUP}
WorkingDirectory=${STACK_DIR}
EnvironmentFile=${CONFIG_FILE}
ExecStart=${STACK_DIR}/scripts/${script}
Restart=always
RestartSec=5
TimeoutStartSec=${timeout}
StandardOutput=journal
StandardError=journal
SyslogIdentifier=${unit_name}
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
UNIT
        chmod 644 "/etc/systemd/system/${unit_name}.service"
        echo "  installed: ${unit_name}.service"
    }

    cat > /etc/systemd/system/llm-manager.service <<UNIT
[Unit]
Description=LLM Stack Manager - web UI
After=network.target

[Service]
Type=simple
User=root
Group=root
WorkingDirectory=${STACK_DIR}
EnvironmentFile=${CONFIG_FILE}
ExecStart=${STACK_DIR}/scripts/start-llm-manager.sh
Restart=always
RestartSec=5
TimeoutStartSec=60
StandardOutput=journal
StandardError=journal
SyslogIdentifier=llm-manager
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
UNIT
    chmod 644 /etc/systemd/system/llm-manager.service
    echo "  installed: llm-manager.service"

    cat << 'UNIT' > /etc/systemd/system/llm-stack-restore.service
[Unit]
Description=Restore LLM Stack active settings
After=network.target llm-manager.service

[Service]
Type=oneshot
User=root
Group=root
WorkingDirectory=@STACK_DIR@
ExecStart=/bin/bash @STACK_DIR@/scripts/restore-active-stack.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
UNIT
    sed -i "s|@STACK_DIR@|${STACK_DIR}|g" /etc/systemd/system/llm-stack-restore.service
    chmod 644 /etc/systemd/system/llm-stack-restore.service
    echo "  installed: llm-stack-restore.service"

    install_unit "think"             "LLM Chat Thinking Legacy - llama-server"          "start-think.sh"             300
    install_unit "nothink"           "LLM Chat Nothink Legacy - llama-server"          "start-nothink.sh"           300
    install_unit "chat-backend"      "LLM Chat Custom Shared Backend - llama-server"   "start-chat-backend.sh"      300
    install_unit "chat-backend-dense"  "LLM Chat Dense Shared Backend - llama-server"    "start-chat-backend-dense.sh"  300
    install_unit "chat-backend-moe"  "LLM Chat MoE Shared Backend - llama-server"      "start-chat-backend-moe.sh"  300
    install_unit "chat-proxy"        "LLM Chat Proxy - think/chat/code ports"          "start-chat-proxy.sh"        30
    install_unit "chat-backend2"     "LLM Chat Custom Shared Backend 2 - llama-server" "start-chat-backend2.sh"     300
    install_unit "chat-proxy2"       "LLM Chat Proxy 2 - think/chat/code ports"        "start-chat-proxy2.sh"       30
    install_unit "embed"         "LLM Embedding Model - llama-server"              "start-embed.sh"         120
    install_unit "embed2"        "LLM Embedding 2 Model - llama-server"            "start-embed2.sh"        120
    install_unit "rerank"          "LLM Reranker Model - llama-server"               "start-rerank.sh"          120
    install_unit "task"              "LLM Task Model - llama-server"                   "start-task.sh"              120
    install_unit "ocr"               "LLM OCR GLM-OCR Backend - llama-server"          "start-ocr.sh"               120
    install_unit "glmocr-sdk"        "LLM OCR GLM-OCR SDK Parser"                      "start-glmocr-sdk.sh"        300
    if [[ "${PLAYWRIGHT_ENABLED:-on}" == "on" ]]; then
        install_playwright_nginx_conf
        cat > /etc/systemd/system/playwright-server.service <<UNIT
[Unit]
Description=Playwright WebSocket Server
After=network.target

[Service]
Type=simple
User=${SERVICE_USER}
Group=${SERVICE_GROUP}
WorkingDirectory=${STACK_DIR}/playwright
EnvironmentFile=${CONFIG_FILE}
Environment=NODE_ENV=${PLAYWRIGHT_NODE_ENV:-production}
ExecStart=${STACK_DIR}/playwright/start.sh
Restart=on-failure
RestartSec=5
TimeoutStartSec=60
StandardOutput=journal
StandardError=journal
SyslogIdentifier=playwright-server

[Install]
WantedBy=multi-user.target
UNIT
        chmod 644 /etc/systemd/system/playwright-server.service
        echo "  installed: playwright-server.service"
    fi
    if [[ "${HONCHO_ENABLED:-off}" == "on" ]]; then
        install_unit "honcho-api"     "Local Honcho Memory API"                         "start-honcho-api.sh"        120
        install_unit "honcho-deriver" "Local Honcho Memory Deriver"                     "start-honcho-deriver.sh"    120
    fi

    cp_sed_inplace "s|^After=network.target$|After=network.target chat-backend.service chat-backend-dense.service chat-backend-moe.service|" /etc/systemd/system/chat-proxy.service
    cp_sed_inplace "s|^After=network.target$|After=network.target chat-backend2.service|" /etc/systemd/system/chat-proxy2.service
    cp_sed_inplace "s|^After=network.target$|After=network.target ocr.service|" /etc/systemd/system/glmocr-sdk.service
    cp_sed_inplace "/^After=/a Wants=ocr.service" /etc/systemd/system/glmocr-sdk.service
    cp_sed_inplace "s|^Restart=always$|Restart=on-failure|" /etc/systemd/system/glmocr-sdk.service
    if [[ "${HONCHO_ENABLED:-off}" == "on" ]]; then
        cp_sed_inplace "s|^After=network.target$|After=network.target postgresql.service redis-server.service chat-proxy.service embed.service embed2.service|" /etc/systemd/system/honcho-api.service
        cp_sed_inplace "/^After=/a Wants=postgresql.service redis-server.service chat-proxy.service embed.service embed2.service" /etc/systemd/system/honcho-api.service
        cp_sed_inplace "s|^After=network.target$|After=network.target honcho-api.service chat-proxy.service embed.service embed2.service|" /etc/systemd/system/honcho-deriver.service
        cp_sed_inplace "/^After=/a Wants=honcho-api.service chat-proxy.service embed.service embed2.service" /etc/systemd/system/honcho-deriver.service
    fi
    cp_sed_inplace "/^After=network.target/a Conflicts=chat-backend-moe.service chat-backend.service" /etc/systemd/system/chat-backend-dense.service
    cp_sed_inplace "/^After=network.target/a Conflicts=chat-backend-dense.service chat-backend.service" /etc/systemd/system/chat-backend-moe.service
    cp_sed_inplace "/^After=network.target/a Conflicts=chat-backend-dense.service chat-backend-moe.service" /etc/systemd/system/chat-backend.service

    systemctl daemon-reload

    DEFAULT_BOOT_SERVICES=(llm-manager llm-stack-restore)
    if [[ "${HONCHO_ENABLED:-off}" == "on" ]]; then
        DEFAULT_BOOT_SERVICES+=(honcho-api honcho-deriver)
    fi
    if [[ "${PLAYWRIGHT_ENABLED:-on}" == "on" ]]; then
        DEFAULT_BOOT_SERVICES+=(playwright-server)
    fi
    NON_DEFAULT_SERVICES=(think nothink chat-backend chat-backend-dense chat-backend-moe chat-backend2 chat-proxy chat-proxy2 embed embed2 rerank task ocr glmocr-sdk)
    LEGACY_SERVICES=(
        qwen-think
        qwen-nothink
        qwen-chat-backend
        qwen-chat-backend-27b
        qwen-chat-backend-35b
        qwen-chat-proxy
        qwen-embedding
        qwen-reranker
        qwen-task
    )
    for svc in "${NON_DEFAULT_SERVICES[@]}" "${LEGACY_SERVICES[@]}"; do
        systemctl disable "${svc}" 2>/dev/null || true
    done
    for svc in "${DEFAULT_BOOT_SERVICES[@]}"; do
        systemctl enable "${svc}"
    done
elif is_mac; then
    # --- macOS launchd installation -------------------------------------------
    install_mac_service() {
        local name="$1"
        local description="$2"
        local script="$3"
        local _launched_wait_for="${4:-}"
        local _launched_conflicts="${5:-}"

        LAUNCHD_WAIT_FOR="${_launched_wait_for}"
        LAUNCHD_CONFLICTS="${_launched_conflicts}"

        # llm-manager runs as root; everything else runs as SERVICE_USER
        if [[ "${name}" == "llm-manager" ]]; then
            local _saved_user="${SERVICE_USER}"
            local _saved_group="${SERVICE_GROUP}"
            SERVICE_USER="root"
            SERVICE_GROUP="wheel"
            generate_launchd_plist "${name}" "${description}" "${script}"
            SERVICE_USER="${_saved_user}"
            SERVICE_GROUP="${_saved_group}"
        else
            generate_launchd_plist "${name}" "${description}" "${script}"
        fi

        _launchd_reset
    }

    echo "Installing launchd services..."

    install_mac_service "llm-manager"        "LLM Stack Manager - web UI"                          "start-llm-manager.sh"
    install_mac_service "think"              "LLM Chat Thinking Legacy - llama-server"             "start-think.sh"
    install_mac_service "nothink"            "LLM Chat Nothink Legacy - llama-server"              "start-nothink.sh"
    install_mac_service "chat-backend"       "LLM Chat Custom Shared Backend - llama-server"       "start-chat-backend.sh"
    install_mac_service "chat-backend-dense" "LLM Chat Dense Shared Backend - llama-server"        "start-chat-backend-dense.sh" \
        "" "chat-backend-moe chat-backend"
    install_mac_service "chat-backend-moe"   "LLM Chat MoE Shared Backend - llama-server"          "start-chat-backend-moe.sh" \
        "" "chat-backend-dense chat-backend"
    install_mac_service "chat-proxy"         "LLM Chat Proxy - think/chat/code ports"              "start-chat-proxy.sh" \
        "chat-backend chat-backend-dense chat-backend-moe"
    install_mac_service "chat-backend2"      "LLM Chat Custom Shared Backend 2 - llama-server"     "start-chat-backend2.sh"
    install_mac_service "chat-proxy2"        "LLM Chat Proxy 2 - think/chat/code ports"            "start-chat-proxy2.sh" \
        "chat-backend2"
    install_mac_service "embed"              "LLM Embedding Model - llama-server"                  "start-embed.sh"
    install_mac_service "embed2"             "LLM Embedding 2 Model - llama-server"                "start-embed2.sh"
    install_mac_service "rerank"             "LLM Reranker Model - llama-server"                   "start-rerank.sh"
    install_mac_service "task"               "LLM Task Model - llama-server"                       "start-task.sh"
    install_mac_service "ocr"                "LLM OCR GLM-OCR Backend - llama-server"              "start-ocr.sh"
    install_mac_service "glmocr-sdk"         "LLM OCR GLM-OCR SDK Parser"                          "start-glmocr-sdk.sh" \
        "ocr"
    if [[ "${HONCHO_ENABLED:-off}" == "on" ]]; then
        install_mac_service "honcho-api"     "Local Honcho Memory API"                             "start-honcho-api.sh" \
            "chat-proxy embed embed2"
        install_mac_service "honcho-deriver" "Local Honcho Memory Deriver"                         "start-honcho-deriver.sh" \
            "honcho-api chat-proxy embed embed2"
    fi

    # Fix glmocr-sdk plist for on-failure restart
    _glmocr_plist="$(svc_plist_path "glmocr-sdk")"
    if [[ -f "${_glmocr_plist}" ]]; then
        cp_sed_inplace 's|<key>KeepAlive</key>|<key>KeepAlive</key>\n    <dict>\n        <key>SuccessfulExit</key>\n        <false/>\n    </dict>|' "${_glmocr_plist}"
        cp_sed_inplace 's|<true/>|<dict>\n        <key>SuccessfulExit</key>\n        <true/>\n    </dict>|' "${_glmocr_plist}"
    fi

    # Own wrapper scripts and plists
    chown -R root:wheel /Library/LaunchDaemons/com.llmstack.*.plist 2>/dev/null || true
    chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "${STACK_DIR}/scripts/launchd-wrapper-"*.sh 2>/dev/null || true

    # Enable default services, disable non-default
    DEFAULT_BOOT_SERVICES=(llm-manager chat-backend-dense chat-proxy embed embed2 rerank task)
    if [[ "${HONCHO_ENABLED:-off}" == "on" ]]; then
        DEFAULT_BOOT_SERVICES+=(honcho-api honcho-deriver)
    fi
    NON_DEFAULT_SERVICES=(think nothink chat-backend chat-backend-moe ocr glmocr-sdk)
    for svc in "${NON_DEFAULT_SERVICES[@]}"; do
        svc_disable "${svc}" 2>/dev/null || true
    done
    for svc in "${DEFAULT_BOOT_SERVICES[@]}"; do
        svc_enable "${svc}"
    done
fi

if [[ "${HONCHO_ENABLED:-off}" == "on" && "${HONCHO_CONFIGURE_HERMES:-on}" == "on" && -d "${STACK_DIR}/hermes" ]]; then
    SERVICE_USER="${SERVICE_USER}" SERVICE_GROUP="${SERVICE_GROUP}" bash "${STACK_DIR}/scripts/configure-hermes-honcho.sh" || true
fi

echo "Install complete. The active stack will automatically be restored on reboot."
echo "You can manually restore your saved settings at any time with:"
echo "  sudo bash ${STACK_DIR}/scripts/restore-active-stack.sh"
echo ""
echo "Or access the web UI at http://localhost:5001"
echo ""
echo "Useful Commands:"
echo "  - Restart manager: sudo systemctl restart llm-manager"
echo "  - Start/stop: sudo bash ${STACK_DIR}/scripts/restore-active-stack.sh"

if is_mac; then
    echo ""
    echo "macOS notes:"
    echo "  - Services are managed via launchd (plist files in /Library/LaunchDaemons/)"
    echo "  - View logs: tail -f ${STACK_DIR}/logs/<service>.stdout.log"
    echo "  - Start/stop: sudo bash ${STACK_DIR}/scripts/default-mode.sh"
fi
