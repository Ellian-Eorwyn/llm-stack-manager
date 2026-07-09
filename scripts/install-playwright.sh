#!/usr/bin/env bash
# Install or update the local Playwright WebSocket server used by LLM Stack Manager.
set -euo pipefail

STACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${STACK_DIR}/config/llm-stack.env"
PLAYWRIGHT_DIR="${STACK_DIR}/playwright"

if [[ -f "${CONFIG_FILE}" ]]; then
    # shellcheck source=/dev/null
    source "${CONFIG_FILE}"
fi

PLAYWRIGHT_ENABLED="${PLAYWRIGHT_ENABLED:-on}"
PLAYWRIGHT_BROWSER="${PLAYWRIGHT_BROWSER:-chromium}"
PLAYWRIGHT_INSTALL_BROWSERS="${PLAYWRIGHT_INSTALL_BROWSERS:-on}"
PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-${PLAYWRIGHT_DIR}/browsers}"

if [[ "${PLAYWRIGHT_ENABLED}" != "on" ]]; then
    echo "Playwright is disabled in ${CONFIG_FILE}; skipping install."
    exit 0
fi

if ! command -v node >/dev/null 2>&1; then
    echo "Node.js is required for Playwright. Install Node.js, then rerun this script." >&2
    exit 1
fi

cd "${PLAYWRIGHT_DIR}"
if [[ -f package-lock.json ]]; then
    npm ci
else
    npm install
fi

if [[ "${PLAYWRIGHT_INSTALL_BROWSERS}" == "on" ]]; then
    mkdir -p "${PLAYWRIGHT_BROWSERS_PATH}"
    PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH}" npx playwright install "${PLAYWRIGHT_BROWSER}"
fi

echo "Playwright dependencies installed in ${PLAYWRIGHT_DIR}"
