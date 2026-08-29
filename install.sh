#!/usr/bin/env bash
# Install the git-friendly core LLM stack without touching any older stack tree.
set -euo pipefail

INSTALL_MODE="${1:---full}"
case "${INSTALL_MODE}" in
    --full|--manager-only|--configure-services) ;;
    *) echo "Usage: sudo bash install.sh [--full|--manager-only|--configure-services]" >&2; exit 2 ;;
esac

STACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/cross-platform.sh
source "${STACK_DIR}/scripts/cross-platform.sh"

CONFIG_DIR="${STACK_DIR}/config"
CONFIG_FILE="${CONFIG_DIR}/llm-stack.env"
EXAMPLE_CONFIG="${CONFIG_DIR}/llm-stack.env.example"

# Feature flags reach this script through the environment. That works when the
# manager runs it, because systemd gives the manager `EnvironmentFile=` and
# `setup_engine` hands its own environment down — but `update.sh` runs it from a
# plain root shell, where none of the config is set. Any flag that decides how
# units are wired has to come from the file in that case, or "change it in the
# UI, re-run the installer" quietly regenerates the units for the old setting.
#
# Read explicitly and by name rather than sourcing the file: this runs as root
# against a config the service user can write, and sourcing it would execute
# whatever it contains. The caller's environment still wins.
config_flag() {
    local key="$1" fallback="$2" line
    if [[ -n "${!key+x}" ]]; then
        printf '%s' "${!key}"
        return
    fi
    if [[ -f "${CONFIG_FILE}" ]]; then
        line="$(grep -aE "^[[:space:]]*${key}=" "${CONFIG_FILE}" | tail -n 1 || true)"
        if [[ -n "${line}" ]]; then
            line="${line#*=}"
            line="${line%\"}"; line="${line#\"}"
            line="${line%\'}"; line="${line#\'}"
            printf '%s' "${line}"
            return
        fi
    fi
    printf '%s' "${fallback}"
}

MODEL_ROUTER_ENABLED="$(config_flag MODEL_ROUTER_ENABLED off)"
# Derived, so the installer and the router cannot disagree about which
# models are pooled. `router-members.py` reads the `<MEMBER>_POOLED`
# switches first and the string behind them.
MODEL_ROUTER_MEMBERS="$(STACK_DIR="${STACK_DIR}" python3 "${STACK_DIR}/scripts/lib/router-members.py" 2>/dev/null || config_flag MODEL_ROUTER_MEMBERS "")"
TRANSCRIPT_ENABLED="$(config_flag TRANSCRIPT_ENABLED off)"

# Which process serves a slot. The launcher name is already a parameter of both
# the systemd and launchd install paths, so selecting an engine is choosing a
# different script rather than branching anywhere else. `mlx` is Apple silicon
# only; asking for it elsewhere is a configuration error worth failing on rather
# than silently serving from llama.cpp and leaving the operator to wonder why
# the Metal path is not being used.
EMBED_ENGINE="$(config_flag EMBED_ENGINE llamacpp)"
TRANSCRIPT_ENGINE="$(config_flag TRANSCRIPT_ENGINE sidecar)"

resolve_engine_script() {
    local slot="$1" engine="$2" default_script="$3" mlx_script="$4"
    case "${engine}" in
        llamacpp|sidecar) echo "${default_script}" ;;
        mlx|parakeet-mlx)
            if ! is_mac; then
                echo "  ERROR: ${slot} engine '${engine}' needs Apple silicon" >&2
                exit 1
            fi
            echo "${mlx_script}" ;;
        *)
            echo "  ERROR: unknown ${slot} engine '${engine}'" >&2
            exit 1 ;;
    esac
}

EMBED_SCRIPT="$(resolve_engine_script embed "${EMBED_ENGINE}" "start-embed.sh" "start-embed-mlx.sh")"
TRANSCRIPT_SCRIPT="$(resolve_engine_script transcription "${TRANSCRIPT_ENGINE}" "start-transcribe.sh" "start-parakeet-mlx.sh")"
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


merge_config_defaults() {
    python3 - "${EXAMPLE_CONFIG}" "${CONFIG_FILE}" "${STACK_DIR}" "${SERVICE_USER}" <<'PYMERGEDEFAULTS'
import re
import sys
from pathlib import Path

example = Path(sys.argv[1])
config = Path(sys.argv[2])
stack_dir = sys.argv[3]
service_user = sys.argv[4]

# Backfilling on literal key names alone is what kept resurrecting the legacy
# CHAT_DENSE_*/CHAT_MOE_* spellings: a config that had been migrated to the
# canonical names looked, to this script, like a config missing the legacy ones.
# They are no longer in the example, but a config written by an older install
# still is, so honour the map rather than relying on the example alone.
#
# Both directions, and the second one is the dangerous one. When the *example*
# holds the canonical name and the config still holds the legacy one -- which is
# every host on the far side of a rename -- appending the example's default does
# not fill a gap, it *shadows a live value*: `normalize_env_keys` backfills the
# canonical key from its legacy twin only when the canonical is absent, so
# writing a placeholder there silently replaces the operator's setting.
#
# That happened on the llm-a/llm-b rename: 65 keys were appended with example
# defaults over a working config, and the primary backend came back pointed at a
# model file that does not exist on the host.
sys.path.insert(0, str(Path(stack_dir) / "web"))
try:
    from config_fields import LEGACY_ENV_KEY_MAP, legacy_names_for
except Exception:
    LEGACY_ENV_KEY_MAP = {}
    def legacy_names_for(_key):
        return ()

content = config.read_text(encoding="utf-8")
existing = set(re.findall(r"^([A-Za-z_][A-Za-z0-9_]*)=", content, re.MULTILINE))
missing = []
for line in example.read_text(encoding="utf-8").splitlines():
    if not line or line.startswith("#") or "=" not in line:
        continue
    key = line.split("=", 1)[0]
    if key in existing:
        continue
    if LEGACY_ENV_KEY_MAP.get(key) in existing:
        continue
    if any(alias in existing for alias in legacy_names_for(key)):
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

if [[ "${INSTALL_MODE}" == "--manager-only" ]]; then
    if ! is_linux; then
        echo "The fresh-machine manager-only bootstrap currently supports Ubuntu Linux only." >&2
        exit 1
    fi
    cat > /etc/systemd/system/llm-manager.service <<UNIT
[Unit]
Description=LLM Stack Manager - trusted-LAN setup UI
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
Group=root
WorkingDirectory=${STACK_DIR}
EnvironmentFile=${CONFIG_FILE}
ExecStart=${STACK_DIR}/scripts/start-llm-manager.sh
Restart=always
RestartSec=5
TimeoutStartSec=180
StandardOutput=journal
StandardError=journal
SyslogIdentifier=llm-manager
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
UNIT
    chmod 644 /etc/systemd/system/llm-manager.service
    systemctl daemon-reload
    systemctl enable --now llm-manager
    echo "Manager-only bootstrap complete: http://127.0.0.1:${LLM_MANAGER_PORT:-8077}"
    exit 0
fi

if [[ "${LLM_STACK_SKIP_DEP_UPDATE:-0}" == "1" ]]; then
    echo "Skipping dependency update because LLM_STACK_SKIP_DEP_UPDATE=1."
else
    echo "Installing/updating dependencies from dependencies.json..."
    if ! sudo -u "${SERVICE_USER}" "${STACK_DIR}/scripts/install-dependencies.py" --update; then
        if [[ -x "${LLAMA_SERVER_BIN:-${STACK_DIR}/deps/llama.cpp/build/bin/llama-server}" ]]; then
            echo "Dependency update failed, but an existing llama-server binary is present; continuing with systemd unit installation." >&2
        else
            echo "Dependency update failed and no llama-server binary is available." >&2
            exit 1
        fi
    fi
fi

if [[ "${SEARXNG_ENABLED:-on}" == "on" && "${LLM_STACK_SKIP_EXTERNAL_INSTALL:-0}" != "1" ]]; then
    echo "Installing/configuring local SearXNG..."
    bash "${STACK_DIR}/scripts/install-searxng.sh"
fi

if [[ "${PLAYWRIGHT_ENABLED:-on}" == "on" && "${LLM_STACK_SKIP_EXTERNAL_INSTALL:-0}" != "1" ]]; then
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

    setup_has_component() {
        [[ -z "${LLM_STACK_SETUP_COMPONENTS:-}" || ",${LLM_STACK_SETUP_COMPONENTS}," == *",$1,"* ]]
    }
    remove_unselected_units() {
        local component="$1"
        shift
        setup_has_component "${component}" && return 0
        local unit
        for unit in "$@"; do
            systemctl disable --now "${unit}" 2>/dev/null || true
            [[ -f "/etc/systemd/system/${unit}.service" ]] && unlink "/etc/systemd/system/${unit}.service"
        done
    }
    if [[ -n "${LLM_STACK_SETUP_COMPONENTS:-}" ]]; then
        remove_unselected_units llm-a llm-a llm-a-proxy
        remove_unselected_units llm-b llm-b llm-b-proxy
        remove_unselected_units embedding embed
        remove_unselected_units reranker rerank
        remove_unselected_units task task
        remove_unselected_units ocr ocr
        remove_unselected_units glmocr-sdk glmocr-sdk
        remove_unselected_units playwright playwright-server
        remove_unselected_units transcribe transcript-backend
    fi

    # Retired units, removed on every install rather than only when component
    # selection is in play: a host that predates the wizard would otherwise keep
    # a unit whose launcher no longer exists, and find out at the next restart.
    #
    # think/nothink were labelled Legacy and were never in the UI's service
    # table. embed2 was a second embedding slot nothing used. The two honcho
    # units ran a local memory service this stack no longer ships -- and they
    # were `enabled`, so systemd would start them at boot against launchers that
    # are now deleted.
    #
    # chat-backend-dense, chat-backend2, chat-proxy and chat-proxy2 are the two
    # chat slots and their proxies under their old names. Their launchers were
    # renamed with them, so the old unit files point at scripts that no longer
    # exist -- which is why they must be removed on the same run that installs
    # llm-a, llm-b and their proxies, not left for a later one.
    for unit in think nothink embed2 chat-backend chat-backend-moe \
             honcho-api honcho-deriver \
             chat-backend-dense chat-backend2 chat-proxy chat-proxy2; do
        systemctl disable --now "${unit}" 2>/dev/null || true
        [[ -f "/etc/systemd/system/${unit}.service" ]] && unlink "/etc/systemd/system/${unit}.service"
    done

    if setup_has_component llm-a; then
        install_unit "llm-a" "LLM A - llama-server"                   "start-llm-a.sh" 300
        install_unit "llm-a-proxy" "LLM A Proxy - think/chat/code ports"   "start-llm-a-proxy.sh" 30
    fi
    if setup_has_component llm-b; then
        install_unit "llm-b" "LLM B - llama-server"                   "start-llm-b.sh" 300
        install_unit "llm-b-proxy" "LLM B Proxy - think/chat/code ports"     "start-llm-b-proxy.sh" 30
    fi
    setup_has_component embedding && install_unit "embed" "LLM Embedding Model - ${EMBED_ENGINE}" "${EMBED_SCRIPT}" 120
    setup_has_component reranker && install_unit "rerank" "LLM Reranker Model - llama-server" "start-rerank.sh" 120
    setup_has_component task && install_unit "task" "LLM Task Model - llama-server" "start-task.sh" 120
    setup_has_component ocr && install_unit "ocr" "LLM OCR GLM-OCR Backend - llama-server" "start-ocr.sh" 120
    setup_has_component glmocr-sdk && install_unit "glmocr-sdk" "LLM OCR GLM-OCR SDK Parser" "start-glmocr-sdk.sh" 300
    # Holds no VRAM until its first request — the model is loaded on demand and
    # released when idle — so installing the unit costs nothing while it is off.
    setup_has_component transcribe && install_unit "transcript-backend" "LLM Transcription - ${TRANSCRIPT_ENGINE}" "${TRANSCRIPT_SCRIPT}" 120
    # One llama-server owning embed/ocr/rank/task on demand. The member units
    # above stay installed but stopped, so turning the flag off and starting
    # them is the whole rollback.
    install_unit "llama-router" "LLM Model Router - on-demand auxiliary models" "start-model-router.sh" 60
    if [[ "${MODEL_ROUTER_ENABLED:-off}" == "on" ]]; then
        bash "${STACK_DIR}/scripts/install-model-router-nginx.sh" || \
            echo "  WARNING: model router nginx shims failed; the per-model ports will not answer" >&2
    else
        bash "${STACK_DIR}/scripts/install-model-router-nginx.sh" --remove >/dev/null 2>&1 || true
    fi
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

    [[ -f /etc/systemd/system/llm-a-proxy.service ]] && cp_sed_inplace "s|^After=network.target$|After=network.target llm-a.service|" /etc/systemd/system/llm-a-proxy.service
    [[ -f /etc/systemd/system/llm-b-proxy.service ]] && cp_sed_inplace "s|^After=network.target$|After=network.target llm-b.service|" /etc/systemd/system/llm-b-proxy.service
    # In router mode the OCR model is not a unit any more, so the SDK's upstream
    # is the router. Keeping Wants=ocr.service here is what pulled the OCR model
    # onto a full GPU and bounced it 32 times — see docs/service-health.md.
    if [[ "${MODEL_ROUTER_ENABLED:-off}" == "on" ]]; then
        OCR_UPSTREAM_UNIT="llama-router.service"
        EMBED_UPSTREAM_UNITS="llama-router.service"
    else
        OCR_UPSTREAM_UNIT="ocr.service"
        EMBED_UPSTREAM_UNITS="embed.service"
    fi
    if [[ -f /etc/systemd/system/glmocr-sdk.service ]]; then
        cp_sed_inplace "s|^After=network.target$|After=network.target ${OCR_UPSTREAM_UNIT}|" /etc/systemd/system/glmocr-sdk.service
        cp_sed_inplace "/^After=/a Wants=${OCR_UPSTREAM_UNIT}" /etc/systemd/system/glmocr-sdk.service
        cp_sed_inplace "s|^Restart=always$|Restart=on-failure|" /etc/systemd/system/glmocr-sdk.service
    fi
    # Same reason as the SDK above, and it is not optional here: the launcher
    # exits 0 when TRANSCRIPT_ENABLED is off, which is the contract
    # `health.ENABLED_FLAGS` relies on to call a switched-off service "not a
    # fault". With Restart=always systemd reads that clean exit as something to
    # retry, and a disabled sidecar bounces every RestartSec forever.
    if [[ -f /etc/systemd/system/transcript-backend.service ]]; then
        cp_sed_inplace "s|^Restart=always$|Restart=on-failure|" /etc/systemd/system/transcript-backend.service
    fi
    # The router's launcher has the same clean exit when MODEL_ROUTER_ENABLED is
    # off, and so had the same latent bounce loop — dormant only while the
    # router happens to be on.
    if [[ -f /etc/systemd/system/llama-router.service ]]; then
        cp_sed_inplace "s|^Restart=always$|Restart=on-failure|" /etc/systemd/system/llama-router.service
    fi

    systemctl daemon-reload

    DEFAULT_BOOT_SERVICES=(llm-manager llm-stack-restore)
    if [[ "${PLAYWRIGHT_ENABLED:-on}" == "on" ]]; then
        DEFAULT_BOOT_SERVICES+=(playwright-server)
    fi
    NON_DEFAULT_SERVICES=(llm-a llm-b llm-a-proxy llm-b-proxy embed rerank task ocr glmocr-sdk)
    if [[ "${MODEL_ROUTER_ENABLED:-off}" == "on" ]]; then
        # The router has to be up at boot: it is what the per-model ports point
        # at, and it is the only thing that can bring those models back.
        DEFAULT_BOOT_SERVICES+=(llama-router)
    else
        NON_DEFAULT_SERVICES+=(llama-router)
    fi
    if [[ "${TRANSCRIPT_ENABLED:-off}" == "on" ]]; then
        # Safe to boot: it imports no ASR runtime and loads no model until the
        # first request, so an idle sidecar costs a socket and nothing else.
        DEFAULT_BOOT_SERVICES+=(transcript-backend)
    else
        NON_DEFAULT_SERVICES+=(transcript-backend)
    fi
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

        # In the system domain llm-manager runs as root, because installing
        # packages and controlling services needs it. A LaunchAgent cannot: the
        # GUI domain runs everything as the session's own user, and UserName is
        # not honoured there. So the root special-case applies to the system
        # domain only -- in the user domain the manager runs as SERVICE_USER
        # like everything else, and the operations that need privilege prompt
        # for it rather than already having it.
        if [[ "${name}" == "llm-manager" && "$(svc_domain)" == "system" ]]; then
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
    install_mac_service "llm-a" "LLM A - llama-server"                      "start-llm-a.sh"
    install_mac_service "llm-a-proxy"         "LLM A Proxy - think/chat/code ports"                 "start-llm-a-proxy.sh" \
        "llm-a"
    install_mac_service "llm-b"      "LLM B - llama-server"                               "start-llm-b.sh"
    install_mac_service "llm-b-proxy"        "LLM B Proxy - think/chat/code ports"                 "start-llm-b-proxy.sh" \
        "llm-b"
    install_mac_service "embed"              "LLM Embedding Model - ${EMBED_ENGINE}"               "${EMBED_SCRIPT}"
    install_mac_service "rerank"             "LLM Reranker Model - llama-server"                   "start-rerank.sh"
    install_mac_service "task"               "LLM Task Model - llama-server"                       "start-task.sh"
    install_mac_service "ocr"                "LLM OCR GLM-OCR Backend - llama-server"              "start-ocr.sh"
    install_mac_service "llama-router"       "LLM Model Router - on-demand auxiliary models"       "start-model-router.sh"
    # Same reasoning as the systemd path: in router mode the SDK's upstream is
    # the router, and waiting on `ocr` would summon a model nothing manages.
    if [[ "${MODEL_ROUTER_ENABLED:-off}" == "on" ]]; then
        _ocr_upstream="llama-router"
        _embed_upstream="llama-router"
    else
        _ocr_upstream="ocr"
        _embed_upstream="embed"
    fi
    install_mac_service "glmocr-sdk"         "LLM OCR GLM-OCR SDK Parser"                          "start-glmocr-sdk.sh" \
        "${_ocr_upstream}"
    # No upstream: only the optional `router` engine talks to llama-router, and
    # the local runtimes need nothing at all. See the note in web/health.py.
    install_mac_service "transcript-backend" "LLM Transcription - ${TRANSCRIPT_ENGINE}"             "${TRANSCRIPT_SCRIPT}"

    # Fix glmocr-sdk plist for on-failure restart
    _glmocr_plist="$(svc_plist_path "glmocr-sdk")"
    if [[ -f "${_glmocr_plist}" ]]; then
        cp_sed_inplace 's|<key>KeepAlive</key>|<key>KeepAlive</key>\n    <dict>\n        <key>SuccessfulExit</key>\n        <false/>\n    </dict>|' "${_glmocr_plist}"
        cp_sed_inplace 's|<true/>|<dict>\n        <key>SuccessfulExit</key>\n        <true/>\n    </dict>|' "${_glmocr_plist}"
    fi

    # Own wrapper scripts and plists
    # A system daemon's plist must be root-owned; a per-user agent's must be
    # owned by the user whose domain loads it, or launchctl refuses to bootstrap
    # it. Ownership follows the domain rather than being applied unconditionally.
    if [[ "${LLM_LAUNCHD_DOMAIN:-user}" == "system" ]]; then
        chown -R root:wheel /Library/LaunchDaemons/com.llmstack.*.plist 2>/dev/null || true
    else
        chown "${SERVICE_USER}:${SERVICE_GROUP}" "$(svc_plist_dir)"/com.llmstack.*.plist 2>/dev/null || true
    fi
    chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "${STACK_DIR}/scripts/launchd-wrapper-"*.sh 2>/dev/null || true

    # Enable default services, disable non-default
    DEFAULT_BOOT_SERVICES=(llm-manager llm-a llm-a-proxy embed rerank task)
    NON_DEFAULT_SERVICES=(ocr glmocr-sdk)
    for svc in "${NON_DEFAULT_SERVICES[@]}"; do
        svc_disable "${svc}" 2>/dev/null || true
    done
    for svc in "${DEFAULT_BOOT_SERVICES[@]}"; do
        svc_enable "${svc}"
    done
fi

if [[ "${EUID}" -eq 0 && -d /usr/local/bin ]]; then
    ln -sfn "${STACK_DIR}/scripts/llm-stack-manager" /usr/local/bin/llm-stack-manager
    echo "Installed CLI: /usr/local/bin/llm-stack-manager"
fi

echo "Install complete. The active stack will automatically be restored on reboot."
echo "You can manually restore your saved settings at any time with:"
echo "  sudo bash ${STACK_DIR}/scripts/restore-active-stack.sh"
echo ""
echo "Or access the web UI at http://localhost:$(config_flag LLM_MANAGER_PORT 8077)"
echo ""
echo "Useful Commands:"
echo "  - Update (fast, no llama.cpp rebuild): sudo llm-stack-manager update"
echo "  - Stack overview: llm-stack-manager status"
if is_mac; then
    echo "  - Restart manager: sudo launchctl kickstart -k system/com.llmstack.llm-manager"
else
    echo "  - Restart manager: sudo systemctl restart llm-manager"
fi
echo "  - Start/stop: sudo bash ${STACK_DIR}/scripts/restore-active-stack.sh"

if is_mac; then
    echo ""
    echo "macOS notes:"
    echo "  - macOS support is INCOMPLETE and under active development."
    echo "    GPU, memory and swap reporting are not yet implemented on this"
    echo "    platform and will read as zero. Do not rely on the health model here."
    echo "  - Services are managed via launchd in the ${LLM_LAUNCHD_DOMAIN:-user} domain"
    echo "    (plists in $(svc_plist_dir))"
    echo "  - View logs: tail -f ${STACK_DIR}/logs/<service>.stdout.log"
    echo "  - Start/stop: sudo bash ${STACK_DIR}/scripts/restore-active-stack.sh"
fi
