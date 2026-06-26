#!/usr/bin/env bash
# Start the local Honcho FastAPI server.
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
    echo "[honcho-api] Missing ${HONCHO_ENV_FILE}; run sudo bash ${STACK_DIR}/install.sh" >&2
    exit 1
fi

cd "${HONCHO_DIR}"
export PYTHONUNBUFFERED=1
export UV_CACHE_DIR="${STACK_DIR}/deps/.uv-cache"

echo "[honcho-api] Starting local Honcho API"
echo "[honcho-api] URL: ${HONCHO_URL:-http://${HONCHO_HOST}:${HONCHO_PORT}}"
echo "[honcho-api] LLM: ${DERIVER_MODEL_CONFIG__MODEL:-${HONCHO_LLM_MODEL:-chat}} @ ${DERIVER_MODEL_CONFIG__OVERRIDES__BASE_URL:-${HONCHO_LLM_BASE_URL:-}}"
echo "[honcho-api] Embed: ${EMBEDDING_MODEL_CONFIG__MODEL:-${HONCHO_EMBED_MODEL:-embed}} @ ${EMBEDDING_MODEL_CONFIG__OVERRIDES__BASE_URL:-${HONCHO_EMBED_BASE_URL:-}}"

"${HONCHO_DIR}/.venv/bin/python" scripts/provision_db.py
"${HONCHO_DIR}/.venv/bin/python" scripts/configure_embeddings.py --yes
exec "${HONCHO_DIR}/.venv/bin/fastapi" run --host "${HONCHO_HOST}" --port "${HONCHO_PORT}" src/main.py
