#!/usr/bin/env bash
# =============================================================================
# start-chat-proxy2.sh
# Starts the second Python proxy/memory gateway that exposes the second shared llama-server
# backend on three public ports:
#   THINK2_PORT   (default 8103) — thinking enabled (pass-through)
#   NOTHINK2_PORT (default 8104) — thinking disabled (injects kwarg per request)
#   CODE2_PORT    (default 8108) — coding overrides
#
# Requires chat-backend2.service to already be running.
# =============================================================================
set -euo pipefail

STACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${STACK_DIR}/config/llm-stack.env"

: "${EMBED_BACKEND_HOST:=127.0.0.1}"

# Override the ports and endpoints for proxy 2
export CHAT_BACKEND_PORT="${CHAT2_BACKEND_PORT:-8020}"
export THINK_PORT="${THINK2_PORT:-8103}"
export NOTHINK_PORT="${NOTHINK2_PORT:-8104}"
export CODE_PORT="${CODE2_PORT:-8108}"

# We use the same backend host
export CHAT_BACKEND_HOST

# Export all other relevant env vars just like proxy 1
export BACKEND_CONNECT_TIMEOUT_SEC
export BACKEND_READ_TIMEOUT_SEC
export LISTEN_HOST
export EMBED_PORT
export EMBED_MODEL_NAME
export EMBED_BACKEND_HOST
export THINK_MODEL_NAME="${THINK2_MODEL_NAME:-think2}"
export NOTHINK_MODEL_NAME="${NOTHINK2_MODEL_NAME:-nothink2}"
export CODE_MODEL_NAME="${CODE2_MODEL_NAME:-code2}"
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

# Disable memory gateway for proxy 2 to avoid port collision
export MEMORY_GATEWAY_ENABLED="off"

echo "[chat-proxy2] Starting proxy 2"
echo "[chat-proxy2] Think port:   ${THINK_PORT}   -> ${CHAT_BACKEND_HOST}:${CHAT_BACKEND_PORT}"
echo "[chat-proxy2] Nothink port: ${NOTHINK_PORT} -> ${CHAT_BACKEND_HOST}:${CHAT_BACKEND_PORT} (+enable_thinking=false)"
echo "[chat-proxy2] Code port:    ${CODE_PORT}    -> ${CHAT_BACKEND_HOST}:${CHAT_BACKEND_PORT} (temp=${CODE_TEMP})"

exec python3 "$(dirname "$0")/llm-chat-proxy.py"
