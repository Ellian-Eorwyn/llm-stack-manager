#!/usr/bin/env bash
# =============================================================================
# start-think.sh
# Launches llama-server for the Qwen chat model WITH thinking/reasoning enabled.
# Port: THINK_PORT (default 8003)
# Model: same GGUF as nothink endpoint
# Split: across both GPUs defined in CHAT_GPU_VISIBLE_DEVICES
# =============================================================================
set -euo pipefail

# Load configuration
STACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${STACK_DIR}/config/llm-stack.env"
LLAMA_SERVER_DIR="${LLAMA_SERVER_BIN%/*}"
export LD_LIBRARY_PATH="${LLAMA_SERVER_DIR}:${LD_LIBRARY_PATH:-}"
export DYLD_LIBRARY_PATH="${LLAMA_SERVER_DIR}:${DYLD_LIBRARY_PATH:-}"

# Tell CUDA which GPUs to use for this service
export CUDA_VISIBLE_DEVICES="${CHAT_GPU_VISIBLE_DEVICES}"

echo "[think] Starting llama-server (thinking enabled)"
echo "[think] Model:   ${CHAT_MODEL_PATH}"
echo "[think] MMProj:  ${CHAT_MMPROJ_PATH}"
echo "[think] Port:    ${THINK_PORT}"
echo "[think] Context: ${CHAT_CTX_SIZE}"
echo "[think] Main GPU:${CHAT_MAIN_GPU}"
echo "[think] GPUs:    ${CUDA_VISIBLE_DEVICES} (split ${CHAT_TENSOR_SPLIT})"
echo "[think] KV cache: K=${CHAT_CACHE_TYPE_K} V=${CHAT_CACHE_TYPE_V}"

# Build optional boolean flags from config
OPTS=()
[[ "${CHAT_LOG_PREFIX:-true}" == "true" ]] && OPTS+=(--log-prefix)
[[ "${CHAT_NO_MMAP:-false}" == "true" ]] && OPTS+=(--no-mmap)
[[ "${CHAT_MLOCK:-false}" == "true" ]] && OPTS+=(--mlock)
[[ "${CHAT_JINJA:-off}" == "on" ]] && OPTS+=(--jinja)

exec "${LLAMA_SERVER_BIN}" \
    --model "${CHAT_MODEL_PATH}" \
    --mmproj "${CHAT_MMPROJ_PATH}" \
    --alias "qwen-chat-think" \
    --host "${LISTEN_HOST}" \
    --port "${THINK_PORT}" \
    --ctx-size "${CHAT_CTX_SIZE}" \
    --main-gpu "${CHAT_MAIN_GPU}" \
    --n-gpu-layers "${CHAT_N_GPU_LAYERS}" \
    --split-mode "${CHAT_SPLIT_MODE}" \
    --tensor-split "${CHAT_TENSOR_SPLIT}" \
    --batch-size "${CHAT_BATCH_SIZE}" \
    --ubatch-size "${CHAT_UBATCH_SIZE}" \
    --parallel "${CHAT_N_PARALLEL}" \
    --cache-type-k "${CHAT_CACHE_TYPE_K}" \
    --cache-type-v "${CHAT_CACHE_TYPE_V}" \
    --flash-attn "${CHAT_FLASH_ATTN}" \
    --temp "${CHAT_TEMP}" \
    --top-p "${CHAT_TOP_P}" \
    --top-k "${CHAT_TOP_K}" \
    --min-p "${CHAT_MIN_P}" \
    "${OPTS[@]}" \
    "$@"
# NOTE: No --chat-template-kwargs here — thinking is ON by default for Qwen3.
