#!/usr/bin/env bash
# =============================================================================
# start-embed.sh
# Launches llama-server in embedding mode for the Qwen embedding model.
# Port: EMBED_PORT (default 8005)
# GPU:  EMBED_GPU_VISIBLE_DEVICES (default GPU 0, single device)
# =============================================================================
set -euo pipefail

# Load configuration
STACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${STACK_DIR}/config/llm-stack.env"
LLAMA_SERVER_DIR="${LLAMA_SERVER_BIN%/*}"
export LD_LIBRARY_PATH="${LLAMA_SERVER_DIR}:${LD_LIBRARY_PATH:-}"
export DYLD_LIBRARY_PATH="${LLAMA_SERVER_DIR}:${DYLD_LIBRARY_PATH:-}"

# Pin to a single GPU for the embedding model
export CUDA_VISIBLE_DEVICES="${EMBED_GPU_VISIBLE_DEVICES}"

echo "[embed] Starting llama-server in embedding mode"
echo "[embed] Model:            ${EMBEDDING_MODEL_PATH}"
echo "[embed] Port:             ${EMBED_PORT}"
echo "[embed] GPU:              ${CUDA_VISIBLE_DEVICES}"
echo "[embed] Context:          ${EMBED_CTX_SIZE}"
echo "[embed] CPU threads:      ${EMBED_THREADS:--1} (batch=${EMBED_THREADS_BATCH:--1})"
echo "[embed] KV cache:         K=${EMBED_CACHE_TYPE_K:-q8_0} V=${EMBED_CACHE_TYPE_V:-q8_0}"
echo "[embed] Reasoning format: ${EMBED_REASONING_FORMAT:-none}"
echo "[embed] Fit to VRAM:      ${EMBED_FIT:-on}"

# Build optional boolean flags from config
OPTS=()
[[ "${EMBED_LOG_PREFIX:-true}" == "true" ]] && OPTS+=(--log-prefix)
[[ "${EMBED_NO_MMAP:-false}" == "true" ]] && OPTS+=(--no-mmap)
[[ "${EMBED_MLOCK:-false}" == "true" ]] && OPTS+=(--mlock)
[[ "${EMBED_JINJA:-off}" == "on" ]] && OPTS+=(--jinja)

exec "${LLAMA_SERVER_BIN}" \
    --model "${EMBEDDING_MODEL_PATH}" \
    --alias "${EMBED_MODEL_NAME:-embed}" \
    --host "${LISTEN_HOST}" \
    --port "${EMBED_PORT}" \
    --ctx-size "${EMBED_CTX_SIZE}" \
    --n-gpu-layers "${EMBED_N_GPU_LAYERS:-${CHAT_N_GPU_LAYERS:--1}}" \
    --split-mode "${EMBED_SPLIT_MODE:-layer}" \
    --tensor-split "${EMBED_TENSOR_SPLIT:-1}" \
    --batch-size "${EMBED_BATCH_SIZE}" \
    --ubatch-size "${EMBED_UBATCH_SIZE:-${CHAT_UBATCH_SIZE:-512}}" \
    --parallel "${EMBED_N_PARALLEL:-1}" \
    --threads "${EMBED_THREADS:--1}" \
    --threads-batch "${EMBED_THREADS_BATCH:--1}" \
    --cache-type-k "${EMBED_CACHE_TYPE_K:-q8_0}" \
    --cache-type-v "${EMBED_CACHE_TYPE_V:-q8_0}" \
    --flash-attn "${EMBED_FLASH_ATTN:-on}" \
    --temp "${EMBED_TEMP:-1.0}" \
    --top-p "${EMBED_TOP_P:-0.95}" \
    --top-k "${EMBED_TOP_K:-20}" \
    --min-p "${EMBED_MIN_P:-0.00}" \
    --reasoning-format "${EMBED_REASONING_FORMAT:-none}" \
    --fit "${EMBED_FIT:-on}" \
    --embedding \
    --pooling mean \
    "${OPTS[@]}" \
    "$@"
