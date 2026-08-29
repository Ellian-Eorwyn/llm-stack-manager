#!/usr/bin/env bash
# =============================================================================
# start-proxy.sh <proxy>
#
# One script for every think/chat/code trio. `web/backends/proxies.py` holds
# what each proxy overrides -- its backend, its three ports, its three aliases,
# its aggregate endpoint, and whether it runs the memory gateway -- and
# `scripts/lib/proxy-env.py` resolves that into the environment
# `llm-chat-proxy.py` already reads.
#
# There were two of these, and the second was a hand copy. They had drifted:
# only one exported the three model aliases, and *neither* exported
# THINK_REASONING_EFFORT or CODE_REASONING_EFFORT, so running either by hand
# dropped both settings while systemd's EnvironmentFile quietly supplied them.
#
# Per-slot persona keys resolve in front of the shared ones -- LLM_B_THINK_TEMP
# before THINK_TEMP -- so the two slots can finally be tuned apart, and a host
# that has only ever set the shared key sees no change at all.
# =============================================================================
set -euo pipefail

PROXY="${1:-}"
if [[ -z "${PROXY}" ]]; then
    echo "usage: ${0##*/} <proxy>; see web/backends/proxies.py" >&2
    exit 2
fi

STACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export STACK_DIR

# `set -a` so ${VAR} references inside the file expand, the same reason
# start-model-router.sh does it. systemd's EnvironmentFile does not expand them.
set -a
# shellcheck source=/dev/null
source "${STACK_DIR}/config/llm-stack.env"
set +a

PYTHON="${PYTHON:-python3}"
if ! eval "$("${PYTHON}" "${STACK_DIR}/scripts/lib/proxy-env.py" "${PROXY}")"; then
    echo "[${PROXY}] could not resolve its environment; see web/backends/proxies.py" >&2
    exit 2
fi

echo "[${PROXY}] Starting proxy"
echo "[${PROXY}] Think port:   ${THINK_PORT}   -> ${CHAT_BACKEND_HOST}:${CHAT_BACKEND_PORT} (${THINK_MODEL_NAME})"
echo "[${PROXY}] Nothink port: ${NOTHINK_PORT} -> ${CHAT_BACKEND_HOST}:${CHAT_BACKEND_PORT} (${NOTHINK_MODEL_NAME}, +enable_thinking=false)"
echo "[${PROXY}] Code port:    ${CODE_PORT}    -> ${CHAT_BACKEND_HOST}:${CHAT_BACKEND_PORT} (${CODE_MODEL_NAME}, temp=${CODE_TEMP})"
echo "[${PROXY}] Aggregate:    ${AGGREGATE_ENABLED} on ${AGGREGATE_PORT} (model-routed ${THINK_MODEL_NAME}/${NOTHINK_MODEL_NAME}/${CODE_MODEL_NAME})"
echo "[${PROXY}] Memory gateway: ${MEMORY_GATEWAY_ENABLED} (graphiti=${MEMORY_GRAPHITI_BASE_URL:-auto}, mode=${MEMORY_INJECTION_MODE:-off})"

exec "${PYTHON}" "${STACK_DIR}/scripts/llm-chat-proxy.py"
