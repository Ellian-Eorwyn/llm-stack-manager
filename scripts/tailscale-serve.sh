#!/usr/bin/env bash
# =============================================================================
# tailscale-serve.sh [on|off|status]
# Publish this host's model endpoints, and the manager, on the tailnet.
#
# Every service here binds 127.0.0.1 by default, which keeps it off the LAN --
# the manager has no login and llama-server no key -- and also off the tailnet.
# `tailscale serve` proxies <this host's MagicDNS name>:<port> on the tailnet to
# 127.0.0.1:<port>, so http://studio:8004/v1 works from any tailnet device and
# from nowhere else, with nothing rebound.
#
# Which ports: those of the services installed here (plus the pooled models'
# ports when the model router is on). A port a service already listens on
# beyond loopback is left alone -- it is reachable as it is, which is how the
# Linux proxies on `llms` are set up.
#
# Not published: the state and control APIs (8078/8079), which have their own
# bearer-token listeners for remote use.
#
# Persistent (`tailscale serve --bg`): survives restarts and reboots. `off`
# removes exactly the ports this script would publish.
# On Linux, tailscale serve needs root or a `tailscale set --operator=$USER`.
# =============================================================================
set -euo pipefail

STACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${STACK_DIR}/scripts/cross-platform.sh"
_stack_dir="${STACK_DIR}"
set -a
# shellcheck disable=SC1091
source "${STACK_DIR}/config/llm-stack.env"
set +a
STACK_DIR="${_stack_dir}"

MODE="${1:-on}"

tailscale_cli() {
    if command -v tailscale >/dev/null 2>&1; then
        command -v tailscale
    elif [[ -x /Applications/Tailscale.app/Contents/MacOS/Tailscale ]]; then
        echo /Applications/Tailscale.app/Contents/MacOS/Tailscale
    else
        return 1
    fi
}

TS="$(tailscale_cli)" || { echo "tailscale is not installed." >&2; exit 1; }

service_ports() {
    case "$1" in
        llm-manager)        echo "${LLM_MANAGER_PORT:-8077}" ;;
        llm-a)              echo "${CHAT_BACKEND_PORT:-8010}" ;;
        llm-a-proxy)        echo "${THINK_PORT:-8003} ${NOTHINK_PORT:-8004} ${CODE_PORT:-8008} ${AGGREGATE_PORT:-8012}" ;;
        llm-b)              echo "${CHAT2_BACKEND_PORT:-8020}" ;;
        llm-b-proxy)        echo "${THINK2_PORT:-8103} ${NOTHINK2_PORT:-8104} ${CODE2_PORT:-8108} ${AGGREGATE2_PORT:-8112}" ;;
        embed)              echo "${EMBED_PORT:-8005}" ;;
        rerank)             echo "${RERANK_PORT:-8006}" ;;
        task)               echo "${TASK_PORT:-8007}" ;;
        ocr)                echo "${OCR_PORT:-8009}" ;;
        transcript-backend) echo "${TRANSCRIPT_PORT:-8014}" ;;
        llama-router)       echo "${MODEL_ROUTER_PORT:-8013}" ;;
    esac
}

SERVICES=(llm-manager llm-a llm-a-proxy llm-b llm-b-proxy embed rerank task ocr
          transcript-backend llama-router)

# The router's members keep their own ports (fronted for it), though their own
# units are not installed.
router_member_services() {
    [[ "${MODEL_ROUTER_ENABLED:-off}" == "on" ]] || return 0
    local member members="${MODEL_ROUTER_MEMBERS:-}"
    for member in ${members//,/ }; do
        case "${member}" in
            EMBED) echo embed ;; RERANK) echo rerank ;; TASK) echo task ;; OCR) echo ocr ;;
        esac
    done
}

# True when something listens on the port on an address other than loopback.
listening_beyond_loopback() {
    local port="$1"
    if is_mac; then
        lsof -nP -iTCP:"${port}" -sTCP:LISTEN -Fn 2>/dev/null \
            | sed -n 's/^n//p' | grep -vqE '^(127\.0\.0\.1|\[::1\]|localhost):'
    else
        ss -Hltn "sport = :${port}" 2>/dev/null \
            | awk '{print $4}' | grep -vqE '^(127\.[0-9.]+|\[::1\]):'
    fi
}

# Installed, not enabled: on Linux the model backends are often started by
# restore-active-stack.sh rather than enabled at boot, and `is-enabled` missed
# llm-a on llms. On launchd, installed and enabled are both "the plist exists".
svc_is_installed() {
    if is_linux; then
        [[ "$(systemctl show -p LoadState --value "$1" 2>/dev/null)" == "loaded" ]]
    else
        svc_is_enabled "$1"
    fi
}

wanted_ports() {
    local svc ports=()
    for svc in "${SERVICES[@]}"; do
        svc_is_installed "${svc}" && ports+=($(service_ports "${svc}"))
    done
    if svc_is_installed llama-router; then
        for svc in $(router_member_services); do
            ports+=($(service_ports "${svc}"))
        done
    fi
    # bash 3.2 treats an empty array as unset under `set -u`.
    printf '%s\n' ${ports[@]+"${ports[@]}"} | awk 'NF && !seen[$0]++'
}

host="$("${TS}" status --json 2>/dev/null | python3 -c '
import json, sys
try:
    print((json.load(sys.stdin)["Self"]["DNSName"] or "").split(".")[0])
except Exception:
    pass')"
host="${host:-$(hostname -s)}"

case "${MODE}" in
    on)
        for port in $(wanted_ports); do
            if listening_beyond_loopback "${port}"; then
                echo "  ${port}: already reachable beyond localhost; left as it is"
                continue
            fi
            "${TS}" serve --bg --http="${port}" "http://127.0.0.1:${port}" >/dev/null
            echo "  http://${host}:${port}"
        done
        ;;
    off)
        for port in $(wanted_ports); do
            "${TS}" serve --http="${port}" off >/dev/null 2>&1 && echo "  ${port}: off" || true
        done
        ;;
    status)
        "${TS}" serve status
        ;;
    *)
        echo "Usage: $0 [on|off|status]" >&2
        exit 2 ;;
esac
