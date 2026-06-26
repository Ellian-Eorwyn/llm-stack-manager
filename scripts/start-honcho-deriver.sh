#!/usr/bin/env bash
# Start the local Honcho background deriver worker.
set -euo pipefail

STACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${STACK_DIR}/config/llm-stack.env"

HONCHO_ENV_FILE="${HONCHO_ENV_FILE:-${STACK_DIR}/config/honcho.env}"
if [[ -f "${HONCHO_ENV_FILE}" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "${HONCHO_ENV_FILE}"
    set +a
else
    echo "[honcho-deriver] Missing ${HONCHO_ENV_FILE}; run sudo bash ${STACK_DIR}/install.sh" >&2
    exit 1
fi

cd "${HONCHO_DIR}"
export PYTHONUNBUFFERED=1
export UV_CACHE_DIR="${STACK_DIR}/deps/.uv-cache"

echo "[honcho-deriver] Starting local Honcho deriver"
echo "[honcho-deriver] LLM: ${DERIVER_MODEL_CONFIG__MODEL:-${HONCHO_LLM_MODEL:-chat}} @ ${DERIVER_MODEL_CONFIG__OVERRIDES__BASE_URL:-${HONCHO_LLM_BASE_URL:-}}"

echo "[honcho-deriver] Waiting for Honcho API health..."
for _ in $(seq 1 60); do
    if "${HONCHO_DIR}/.venv/bin/python" - <<'PYHEALTH'
import os
import sys
from urllib.request import urlopen
url = os.environ.get("HONCHO_URL") or f"http://{os.environ.get('HONCHO_HOST', '127.0.0.1')}:{os.environ.get('HONCHO_PORT', '8090')}"
try:
    with urlopen(url.rstrip('/') + "/health", timeout=2) as resp:
        sys.exit(0 if resp.status == 200 else 1)
except Exception:
    sys.exit(1)
PYHEALTH
    then
        break
    fi
    sleep 2
done

exec "${HONCHO_DIR}/.venv/bin/python" -m src.deriver
