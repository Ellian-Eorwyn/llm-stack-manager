#!/usr/bin/env bash
# =============================================================================
# start-llm-manager.sh
# Starts the LLM Stack Manager web UI.
# Creates a Python venv on first run and installs Flask into it.
# =============================================================================
set -euo pipefail

STACK_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WEB_DIR="${STACK_DIR}/web"
VENV_DIR="${WEB_DIR}/.venv"

# Source config for LLM_MANAGER_PORT / LLM_MANAGER_HOST
source "${STACK_DIR}/config/llm-stack.env"

export LLM_MANAGER_PORT="${LLM_MANAGER_PORT:-8077}"
export LLM_MANAGER_HOST="${LLM_MANAGER_HOST:-0.0.0.0}"

# Create venv and install dependencies on first run
if [[ ! -f "${VENV_DIR}/bin/flask" ]]; then
    echo "[llm-manager] Creating Python venv at ${VENV_DIR}..."
    python3 -m venv "${VENV_DIR}"
    "${VENV_DIR}/bin/pip" install --quiet -r "${WEB_DIR}/requirements.txt"
    echo "[llm-manager] Dependencies installed."
fi

echo "[llm-manager] Starting on http://${LLM_MANAGER_HOST}:${LLM_MANAGER_PORT}"
exec "${VENV_DIR}/bin/python" "${WEB_DIR}/app.py"
