#!/usr/bin/env bash
set -uo pipefail

STACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${STACK_DIR}/config/llm-stack.env"
EXPECTATIONS="${STACK_DIR}/config/service-expectations.json"

if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: config file not found: ${CONFIG}"
    exit 1
fi

source "${CONFIG}"
BASE=http://localhost
PASS=0
FAIL=0
SKIP=0

check() {
    local label="$1"
    local result="$2"
    local expected="$3"
    if echo "${result}" | grep -q "${expected}"; then
        echo "  [PASS] ${label}"
        ((PASS++))
    else
        echo "  [FAIL] ${label}"
        echo "         Response: ${result:0:200}"
        ((FAIL++))
    fi
}

# Services the operator stopped on purpose, as recorded by the manager when a
# service is started or stopped from the UI. Nothing else on the box carries
# that intent: `systemctl is-enabled` reads `disabled` for units that are
# running, and the `*_ENABLED` flags read `on` for services deliberately down.
EXPECTED_OFF="$(python3 - "${EXPECTATIONS}" <<'PY' 2>/dev/null || true
import json, pathlib, sys
try:
    data = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
except Exception:
    data = {}
print(" ".join(name for name, entry in (data or {}).items()
                if isinstance(entry, dict) and entry.get("expected") == "off"))
PY
)"
EXPECTED_ON="$(python3 - "${EXPECTATIONS}" <<'PY' 2>/dev/null || true
import json, pathlib, sys
try:
    data = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
except Exception:
    data = {}
print(" ".join(name for name, entry in (data or {}).items()
                if isinstance(entry, dict) and entry.get("expected") == "on"))
PY
)"

# Validate a service when it is meant to be up or is up. A service that is off
# on purpose used to fail this script, which made a clean run mean "everything
# is installed" rather than "everything that should be running works".
should_check() {
    local service="$1"
    case " ${EXPECTED_OFF} " in *" ${service} "*) return 1 ;; esac
    case " ${EXPECTED_ON} "  in *" ${service} "*) return 0 ;; esac
    [[ "$(systemctl is-active "${service}" 2>/dev/null)" == "active" ]]
}

skip() {
    echo "  [SKIP] $1 — not running and not expected to be"
    ((SKIP++))
}

echo "============================================================"
echo " LLM Stack Core Endpoint Validation"
echo " Config: ${CONFIG}"
echo " Ports: think=${THINK_PORT} nothink=${NOTHINK_PORT} code=${CODE_PORT} embed=${EMBED_PORT} rerank=${RERANK_PORT} task=${TASK_PORT} honcho=${HONCHO_PORT:-off}"
echo "============================================================"
echo ""

echo "--- Primary backend (port ${CHAT_BACKEND_PORT:-8010}) ---"
if should_check chat-backend-dense || should_check chat-backend-moe || should_check chat-backend; then
    # The proxy ports below all fan into this one process. Checking only those
    # meant a backend that had died behind a live proxy read as a proxy fault.
    PROPS=$(curl -sf "${BASE}:${CHAT_BACKEND_PORT:-8010}/props" 2>&1 || true)
    check "GET :${CHAT_BACKEND_PORT:-8010}/props returns slot geometry" "${PROPS}" '"total_slots"'
else
    skip "Primary backend"
fi

echo ""
echo "--- /v1/models endpoints ---"
if should_check chat-proxy; then
    for port in "${THINK_PORT}" "${NOTHINK_PORT}" "${CODE_PORT}"; do
        r=$(curl -sf "${BASE}:${port}/v1/models" 2>&1 || true)
        check "GET :${port}/v1/models returns JSON" "${r}" '"object"'
    done
else
    skip "Primary proxy"
fi
for entry in "embed:${EMBED_PORT}" "rerank:${RERANK_PORT}" "task:${TASK_PORT}" "embed2:${EMBED2_PORT:-}"; do
    service="${entry%%:*}"
    port="${entry##*:}"
    [[ -z "${port}" ]] && continue
    if should_check "${service}"; then
        r=$(curl -sf "${BASE}:${port}/v1/models" 2>&1 || true)
        check "GET :${port}/v1/models returns JSON" "${r}" '"object"'
    else
        skip "${service} /v1/models"
    fi
done

echo ""
echo "--- Chat endpoint (port ${NOTHINK_PORT}) ---"
if should_check chat-proxy; then
    CHAT_RESP=$(curl -sf "${BASE}:${NOTHINK_PORT}/v1/chat/completions"     -H "Content-Type: application/json"     -d '{"model":"chat","messages":[{"role":"user","content":"Reply with exactly: CHAT_OK"}],"max_tokens":50,"temperature":0}' 2>&1 || true)
    check "Chat returns a message" "${CHAT_RESP}" '"content"'

    echo ""
    echo "--- Code endpoint (port ${CODE_PORT}) ---"
    CODE_RESP=$(curl -sf "${BASE}:${CODE_PORT}/v1/chat/completions"     -H "Content-Type: application/json"     -d '{"model":"code","messages":[{"role":"user","content":"Reply with exactly: CODE_OK"}],"max_tokens":50,"temperature":0}' 2>&1 || true)
    check "Code returns a message" "${CODE_RESP}" '"content"'
else
    skip "Chat and code endpoints"
fi

echo ""
echo "--- Embedding endpoint (port ${EMBED_PORT}) ---"
if should_check embed; then
    EMBED_RESP=$(curl -sf "${BASE}:${EMBED_PORT}/v1/embeddings"     -H "Content-Type: application/json"     -d '{"model":"embed","input":"Hello world"}' 2>&1 || true)
    check "Embedding returns data array" "${EMBED_RESP}" '"embedding"'
else
    skip "Embedding endpoint"
fi

echo ""
echo "--- Reranker endpoint (port ${RERANK_PORT}) ---"
if should_check rerank; then
    RERANK_RESP=$(curl -sf "${BASE}:${RERANK_PORT}/v1/rerank"     -H "Content-Type: application/json"     -d '{"model":"rank","query":"capital of France","documents":["Paris is the capital of France.","Berlin is in Germany."]}' 2>&1 || true)
    check "Reranker returns results" "${RERANK_RESP}" '"relevance_score"'
else
    skip "Reranker endpoint"
fi

echo ""
echo "--- Task endpoint (port ${TASK_PORT}) ---"
if should_check task; then
    TASK_RESP=$(curl -sf "${BASE}:${TASK_PORT}/v1/chat/completions"     -H "Content-Type: application/json"     -d '{"model":"task","messages":[{"role":"user","content":"Reply with exactly: TASK_OK"}],"max_tokens":50,"temperature":0}' 2>&1 || true)
    check "Task chat returns a message" "${TASK_RESP}" '"content"'
else
    skip "Task endpoint"
fi

echo ""
echo "--- OCR (port ${OCR_PORT:-8009}) and OCR SDK (port ${GLMOCR_SDK_PORT:-5002}) ---"
if should_check ocr; then
    OCR_RESP=$(curl -sf "${BASE}:${OCR_PORT:-8009}/v1/models" 2>&1 || true)
    check "GET :${OCR_PORT:-8009}/v1/models returns JSON" "${OCR_RESP}" '"object"'
else
    skip "OCR backend"
fi
if should_check glmocr-sdk; then
    SDK_RESP=$(curl -sf "${BASE}:${GLMOCR_SDK_PORT:-5002}/health" 2>&1 || true)
    check "OCR SDK health endpoint responds" "${SDK_RESP}" '"ok"'
    # The SDK answers /health from its own process, so a healthy SDK says
    # nothing about the backend it hands every document to.
    if ! should_check ocr; then
        echo "  [WARN] OCR SDK is running but its OCR backend is not — documents will fail"
    fi
else
    skip "OCR SDK"
fi

if [[ "${HONCHO_ENABLED:-off}" == "on" ]] && should_check honcho-api; then
    echo ""
    echo "--- Honcho endpoint (port ${HONCHO_PORT}) ---"
    HONCHO_RESP=$(curl -sf "${HONCHO_URL:-${BASE}:${HONCHO_PORT}}/health" 2>&1 || true)
    check "Honcho health endpoint responds" "${HONCHO_RESP}" '"status"'

    if [[ -f "${HONCHO_ENV_FILE:-${STACK_DIR}/config/honcho.env}" ]]; then
        check "Honcho env file exists" "ok" "ok"
    else
        check "Honcho env file exists" "missing" "ok"
    fi

    if [[ -f "${HOME}/.hermes/honcho.json" ]]; then
        check "Hermes Honcho config points local" "$(sed -n '1,80p' "${HOME}/.hermes/honcho.json" 2>/dev/null || true)" "127.0.0.1"
    fi
fi

echo ""
echo "============================================================"
echo " Results: ${PASS} passed, ${FAIL} failed, ${SKIP} skipped"
[[ "${FAIL}" -eq 0 ]]
