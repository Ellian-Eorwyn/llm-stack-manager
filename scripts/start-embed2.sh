#!/usr/bin/env bash
# =============================================================================
# start-embed2.sh
# Launches llama-server in embedding mode for the second embedding model.
# Port: EMBED2_PORT (default 8011)
# GPU:  EMBED2_GPU_VISIBLE_DEVICES
# =============================================================================
set -euo pipefail

# Load configuration
STACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${STACK_DIR}/config/llm-stack.env"
LLAMA_SERVER_DIR="${LLAMA_SERVER_BIN%/*}"
export LD_LIBRARY_PATH="${LLAMA_SERVER_DIR}:${LD_LIBRARY_PATH:-}"
export DYLD_LIBRARY_PATH="${LLAMA_SERVER_DIR}:${DYLD_LIBRARY_PATH:-}"

# Pin to a single GPU for the embedding model
export CUDA_VISIBLE_DEVICES="${EMBED2_GPU_VISIBLE_DEVICES}"

echo "[embed2] Starting llama-server in embedding mode"
echo "[embed2] Model:            ${EMBED2_MODEL_PATH}"
echo "[embed2] Port:             ${EMBED2_PORT}"
echo "[embed2] GPU:              ${CUDA_VISIBLE_DEVICES}"
echo "[embed2] Context:          ${EMBED2_CTX_SIZE}"
echo "[embed2] CPU threads:      ${EMBED2_THREADS:--1} (batch=${EMBED2_THREADS_BATCH:--1})"
echo "[embed2] KV cache:         K=${EMBED2_CACHE_TYPE_K:-q8_0} V=${EMBED2_CACHE_TYPE_V:-q8_0}"
echo "[embed2] Reasoning format: ${EMBED2_REASONING_FORMAT:-none}"
echo "[embed2] Fit to VRAM:      ${EMBED2_FIT:-on}"

# Build optional boolean flags from config
OPTS=()
[[ "${EMBED2_LOG_PREFIX:-true}" == "true" ]] && OPTS+=(--log-prefix)
[[ "${EMBED2_NO_MMAP:-false}" == "true" ]] && OPTS+=(--no-mmap)
[[ "${EMBED2_MLOCK:-false}" == "true" ]] && OPTS+=(--mlock)
[[ "${EMBED2_JINJA:-off}" == "on" ]] && OPTS+=(--jinja)

exec "${LLAMA_SERVER_BIN}" \
    --model "${EMBED2_MODEL_PATH}" \
    --alias "${EMBED2_MODEL_NAME:-embed2}" \
    --host "${LISTEN_HOST}" \
    --port "${EMBED2_PORT}" \
    --ctx-size "${EMBED2_CTX_SIZE}" \
    --n-gpu-layers "${EMBED2_N_GPU_LAYERS:-${CHAT_N_GPU_LAYERS:--1}}" \
    --split-mode "${EMBED2_SPLIT_MODE:-layer}" \
    --tensor-split "${EMBED2_TENSOR_SPLIT:-1}" \
    --batch-size "${EMBED2_BATCH_SIZE}" \
    --ubatch-size "${EMBED2_UBATCH_SIZE:-${CHAT_UBATCH_SIZE:-512}}" \
    --parallel "${EMBED2_N_PARALLEL:-1}" \
    --threads "${EMBED2_THREADS:--1}" \
    --threads-batch "${EMBED2_THREADS_BATCH:--1}" \
    --cache-type-k "${EMBED2_CACHE_TYPE_K:-q8_0}" \
    --cache-type-v "${EMBED2_CACHE_TYPE_V:-q8_0}" \
    --flash-attn "${EMBED2_FLASH_ATTN:-on}" \
    --temp "${EMBED2_TEMP:-1.0}" \
    --top-p "${EMBED2_TOP_P:-0.95}" \
    --top-k "${EMBED2_TOP_K:-20}" \
    --min-p "${EMBED2_MIN_P:-0.00}" \
    --reasoning-format "${EMBED2_REASONING_FORMAT:-none}" \
    --fit "${EMBED2_FIT:-on}" \
    --embedding \
    --pooling mean \
    "${OPTS[@]}" \
    "$@"
