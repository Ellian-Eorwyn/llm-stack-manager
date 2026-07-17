#!/usr/bin/env bash
# =============================================================================
# start-chat-proxy.sh
# Starts the Python proxy/memory gateway that exposes the shared llama-server
# backend on three public ports:
#   THINK_PORT   (default 8003) — thinking enabled (pass-through)
#   NOTHINK_PORT (default 8004) — thinking disabled (injects kwarg per request)
#   CODE_PORT    (default 8008) — coding overrides
#
# Requires chat-backend.service to already be running.
# =============================================================================
set -euo pipefail

STACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${STACK_DIR}/config/llm-stack.env"

: "${EMBED_BACKEND_HOST:=127.0.0.1}"
: "${AGGREGATE_ENABLED:=on}"
: "${AGGREGATE_PORT:=8012}"

# Export all relevant env vars so the Python script can read them
export CHAT_BACKEND_HOST
export CHAT_BACKEND_PORT
export BACKEND_CONNECT_TIMEOUT_SEC
export BACKEND_READ_TIMEOUT_SEC
export THINK_PORT
export NOTHINK_PORT
export CODE_PORT
export AGGREGATE_PORT
export AGGREGATE_ENABLED
export LISTEN_HOST
export EMBED_PORT
export EMBED_MODEL_NAME
export EMBED_BACKEND_HOST
export THINK_MODEL_NAME
export NOTHINK_MODEL_NAME
export CODE_MODEL_NAME
export THINK_PRESERVE_THINKING
export THINK_JINJA
export THINK_TEMP
export THINK_TOP_P
export THINK_TOP_K
export THINK_MIN_P
export THINK_PRESENCE_PENALTY
export THINK_REPEAT_PENALTY
export THINK_REASONING_FORMAT
export THINK_MAX_TOKENS
export THINK_REASONING_STREAM_MODE
export NOTHINK_PRESERVE_THINKING
export NOTHINK_JINJA
export NOTHINK_TEMP
export NOTHINK_TOP_P
export NOTHINK_TOP_K
export NOTHINK_MIN_P
export NOTHINK_PRESENCE_PENALTY
export NOTHINK_REPEAT_PENALTY
export NOTHINK_REASONING_FORMAT
export NOTHINK_MAX_TOKENS
export NOTHINK_REASONING_STREAM_MODE
export CODE_THINKING
export CODE_PRESERVE_THINKING
export CODE_JINJA
export CODE_TEMP
export CODE_TOP_P
export CODE_TOP_K
export CODE_MIN_P
export CODE_PRESENCE_PENALTY
export CODE_REPEAT_PENALTY
export CODE_REASONING_FORMAT
export CODE_MAX_TOKENS
export CODE_REASONING_STREAM_MODE
export PROXY_STREAM_PASSTHROUGH

export GRAPHITI_PORT
export GRAPHITI_PUBLIC_URL
export MEMORY_GATEWAY_ENABLED
export MEMORY_ENABLE_THINK
export MEMORY_ENABLE_NOTHINK
export MEMORY_ENABLE_CODE
export MEMORY_GRAPHITI_BASE_URL
export MEMORY_GRAPHITI_TIMEOUT_SEC
export MEMORY_GRAPHITI_COOLDOWN_SEC
export MEMORY_INJECTION_MODE
export MEMORY_MAX_FACTS
export MEMORY_MAX_FACT_CHARS
export MEMORY_MAX_BLOCK_CHARS
export MEMORY_MAX_QUERY_MESSAGES
export MEMORY_INCLUDE_SYSTEM_IN_QUERY
export MEMORY_MAX_INGEST_CHARS
export MEMORY_GROUP_HEADER_PRIORITY
export MEMORY_GROUP_FALLBACK_SALT
export MEMORY_FAIL_OPEN

echo "[chat-proxy] Starting proxy"
echo "[chat-proxy] Think port:   ${THINK_PORT}   -> ${CHAT_BACKEND_HOST}:${CHAT_BACKEND_PORT}"
echo "[chat-proxy] Nothink port: ${NOTHINK_PORT} -> ${CHAT_BACKEND_HOST}:${CHAT_BACKEND_PORT} (+enable_thinking=false)"
echo "[chat-proxy] Code port:    ${CODE_PORT}    -> ${CHAT_BACKEND_HOST}:${CHAT_BACKEND_PORT} (temp=${CODE_TEMP})"
echo "[chat-proxy] Aggregate:    ${AGGREGATE_ENABLED:-on} on ${AGGREGATE_PORT:-8012} (model-routed think/chat/code)"
echo "[chat-proxy] Memory gateway: ${MEMORY_GATEWAY_ENABLED} (graphiti=${MEMORY_GRAPHITI_BASE_URL:-auto}, mode=${MEMORY_INJECTION_MODE})"

exec python3 "$(dirname "$0")/llm-chat-proxy.py"
