#!/usr/bin/env bash
set -uo pipefail

STACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${STACK_DIR}/config/llm-stack.env"

if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: config file not found: ${CONFIG}"
    exit 1
fi

source "${CONFIG}"
BASE=http://localhost
PASS=0
FAIL=0

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

echo "============================================================"
echo " LLM Stack Core Endpoint Validation"
echo " Config: ${CONFIG}"
echo " Ports: think=${THINK_PORT} nothink=${NOTHINK_PORT} code=${CODE_PORT} embed=${EMBED_PORT} rerank=${RERANK_PORT} task=${TASK_PORT} honcho=${HONCHO_PORT:-off}"
echo "============================================================"
echo ""

echo "--- /v1/models endpoints ---"
for port in "${THINK_PORT}" "${NOTHINK_PORT}" "${CODE_PORT}" "${EMBED_PORT}" "${RERANK_PORT}" "${TASK_PORT}"; do
    r=$(curl -sf "${BASE}:${port}/v1/models" 2>&1 || true)
    check "GET :${port}/v1/models returns JSON" "${r}" '"object"'
done

echo ""
echo "--- Chat endpoint (port ${NOTHINK_PORT}) ---"
CHAT_RESP=$(curl -sf "${BASE}:${NOTHINK_PORT}/v1/chat/completions"     -H "Content-Type: application/json"     -d '{"model":"chat","messages":[{"role":"user","content":"Reply with exactly: CHAT_OK"}],"max_tokens":50,"temperature":0}' 2>&1 || true)
check "Chat returns a message" "${CHAT_RESP}" '"content"'

echo ""
echo "--- Code endpoint (port ${CODE_PORT}) ---"
CODE_RESP=$(curl -sf "${BASE}:${CODE_PORT}/v1/chat/completions"     -H "Content-Type: application/json"     -d '{"model":"code","messages":[{"role":"user","content":"Reply with exactly: CODE_OK"}],"max_tokens":50,"temperature":0}' 2>&1 || true)
check "Code returns a message" "${CODE_RESP}" '"content"'

echo ""
echo "--- Embedding endpoint (port ${EMBED_PORT}) ---"
EMBED_RESP=$(curl -sf "${BASE}:${EMBED_PORT}/v1/embeddings"     -H "Content-Type: application/json"     -d '{"model":"embed","input":"Hello world"}' 2>&1 || true)
check "Embedding returns data array" "${EMBED_RESP}" '"embedding"'

echo ""
echo "--- Reranker endpoint (port ${RERANK_PORT}) ---"
RERANK_RESP=$(curl -sf "${BASE}:${RERANK_PORT}/v1/rerank"     -H "Content-Type: application/json"     -d '{"model":"rank","query":"capital of France","documents":["Paris is the capital of France.","Berlin is in Germany."]}' 2>&1 || true)
check "Reranker returns results" "${RERANK_RESP}" '"relevance_score"'

echo ""
echo "--- Task endpoint (port ${TASK_PORT}) ---"
TASK_RESP=$(curl -sf "${BASE}:${TASK_PORT}/v1/chat/completions"     -H "Content-Type: application/json"     -d '{"model":"task","messages":[{"role":"user","content":"Reply with exactly: TASK_OK"}],"max_tokens":50,"temperature":0}' 2>&1 || true)
check "Task chat returns a message" "${TASK_RESP}" '"content"'


if [[ "${HONCHO_ENABLED:-off}" == "on" ]]; then
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
echo " Results: ${PASS} passed, ${FAIL} failed"
[[ "${FAIL}" -eq 0 ]]
