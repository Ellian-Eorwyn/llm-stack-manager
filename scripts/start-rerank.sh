#!/usr/bin/env bash
# =============================================================================
# start-rerank.sh
# Launches llama-server in reranking mode for the Qwen reranker model.
# Port: RERANK_PORT (default 8006)
# GPU:  RERANK_GPU_VISIBLE_DEVICES (default GPU 1, single device)
#
# llama-server exposes reranking via POST /v1/rerank (OpenAI-compatible format).
# =============================================================================
set -euo pipefail

# Load configuration
STACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${STACK_DIR}/config/llm-stack.env"
LLAMA_SERVER_DIR="${LLAMA_SERVER_BIN%/*}"
export LD_LIBRARY_PATH="${LLAMA_SERVER_DIR}:${LD_LIBRARY_PATH:-}"

# Pin to a single GPU for the reranker model (separate from embedding GPU)
export CUDA_VISIBLE_DEVICES="${RERANK_GPU_VISIBLE_DEVICES}"

echo "[rerank] Starting llama-server in reranking mode"
echo "[rerank] Model:            ${RERANKER_MODEL_PATH}"
echo "[rerank] Port:             ${RERANK_PORT}"
echo "[rerank] GPU:              ${CUDA_VISIBLE_DEVICES}"
echo "[rerank] Context:          ${RERANK_CTX_SIZE}"
echo "[rerank] CPU threads:      ${RERANK_THREADS:--1} (batch=${RERANK_THREADS_BATCH:--1})"
echo "[rerank] KV cache:         K=${RERANK_CACHE_TYPE_K:-q8_0} V=${RERANK_CACHE_TYPE_V:-q8_0}"
echo "[rerank] Reasoning format: ${RERANK_REASONING_FORMAT:-none}"
echo "[rerank] Fit to VRAM:      ${RERANK_FIT:-on}"

# Build optional boolean flags from config
OPTS=()
[[ "${RERANK_LOG_PREFIX:-true}" == "true" ]] && OPTS+=(--log-prefix)
[[ "${RERANK_NO_MMAP:-false}" == "true" ]] && OPTS+=(--no-mmap)
[[ "${RERANK_MLOCK:-false}" == "true" ]] && OPTS+=(--mlock)
[[ "${RERANK_JINJA:-off}" == "on" ]] && OPTS+=(--jinja)

exec "${LLAMA_SERVER_BIN}" \
    --model "${RERANKER_MODEL_PATH}" \
    --alias "${RERANK_MODEL_NAME:-rank}" \
    --host "${LISTEN_HOST}" \
    --port "${RERANK_PORT}" \
    --ctx-size "${RERANK_CTX_SIZE}" \
    --n-gpu-layers "${RERANK_N_GPU_LAYERS:-${CHAT_N_GPU_LAYERS:--1}}" \
    --split-mode "${RERANK_SPLIT_MODE:-layer}" \
    --tensor-split "${RERANK_TENSOR_SPLIT:-1}" \
    --batch-size "${RERANK_BATCH_SIZE}" \
    --ubatch-size "${RERANK_UBATCH_SIZE:-${CHAT_UBATCH_SIZE:-512}}" \
    --parallel "${RERANK_N_PARALLEL:-1}" \
    --threads "${RERANK_THREADS:--1}" \
    --threads-batch "${RERANK_THREADS_BATCH:--1}" \
    --cache-type-k "${RERANK_CACHE_TYPE_K:-q8_0}" \
    --cache-type-v "${RERANK_CACHE_TYPE_V:-q8_0}" \
    --flash-attn "${RERANK_FLASH_ATTN:-on}" \
    --temp "${RERANK_TEMP:-1.0}" \
    --top-p "${RERANK_TOP_P:-0.95}" \
    --top-k "${RERANK_TOP_K:-20}" \
    --min-p "${RERANK_MIN_P:-0.00}" \
    --reasoning-format "${RERANK_REASONING_FORMAT:-none}" \
    --fit "${RERANK_FIT:-on}" \
    --reranking \
    "${OPTS[@]}" \
    "$@"
