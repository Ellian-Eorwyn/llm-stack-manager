#!/usr/bin/env bash
# =============================================================================
# start-ocr.sh
# Launches llama-server for GLM-OCR multimodal extraction.
# Port: OCR_PORT (default 8009)
# =============================================================================
set -euo pipefail

STACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${STACK_DIR}/config/llm-stack.env"
LLAMA_SERVER_DIR="${LLAMA_SERVER_BIN%/*}"
export LD_LIBRARY_PATH="${LLAMA_SERVER_DIR}:${LD_LIBRARY_PATH:-}"
export DYLD_LIBRARY_PATH="${LLAMA_SERVER_DIR}:${DYLD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${OCR_GPU_VISIBLE_DEVICES:-${TASK_GPU_VISIBLE_DEVICES:-0}}"

ocr_equal_tensor_split() {
    local devices="${1//[[:space:]]/}"
    local count=0
    local part
    IFS=',' read -ra parts <<< "${devices}"
    for part in "${parts[@]}"; do
        [[ -n "${part}" ]] && ((count += 1))
    done
    ((count < 1)) && count=1
    local split="1"
    local i
    for ((i = 1; i < count; i++)); do
        split+=",1"
    done
    printf '%s\n' "${split}"
}

OCR_EFFECTIVE_TENSOR_SPLIT="${OCR_TENSOR_SPLIT:-auto}"
if [[ -z "${OCR_EFFECTIVE_TENSOR_SPLIT}" || "${OCR_EFFECTIVE_TENSOR_SPLIT}" == "auto" ]]; then
    OCR_EFFECTIVE_TENSOR_SPLIT="$(ocr_equal_tensor_split "${CUDA_VISIBLE_DEVICES}")"
fi

echo "[ocr] Starting llama-server (GLM-OCR backend)"
echo "[ocr] Model:            ${OCR_MODEL_PATH:-${STACK_DIR}/models/GLM-OCR-F16.gguf}"
echo "[ocr] MMProj:           ${OCR_MMPROJ_PATH:-(none)}"
echo "[ocr] Host:             ${OCR_HOST:-${LISTEN_HOST}}"
echo "[ocr] Port:             ${OCR_PORT:-8009}"
echo "[ocr] Context:          ${OCR_CTX_SIZE:-8192}"
echo "[ocr] Batch:            ${OCR_BATCH_SIZE:-2048} (uBatch=${OCR_UBATCH_SIZE:-512})"
echo "[ocr] Main GPU:         ${OCR_MAIN_GPU:-0}"
echo "[ocr] GPUs:             ${CUDA_VISIBLE_DEVICES}"
echo "[ocr] Tensor split:     ${OCR_EFFECTIVE_TENSOR_SPLIT}"
echo "[ocr] Device override:  ${OCR_DEVICE:-auto}"
echo "[ocr] Placement:        split=${OCR_SPLIT_MODE:-layer} kv-offload=${OCR_KV_OFFLOAD:-on} op-offload=${OCR_OP_OFFLOAD:-on} mmproj-offload=${OCR_MMPROJ_OFFLOAD:-on}"
echo "[ocr] CPU threads:      ${OCR_THREADS:--1} (batch=${OCR_THREADS_BATCH:--1})"
echo "[ocr] KV cache:         K=${OCR_CACHE_TYPE_K:-f16} V=${OCR_CACHE_TYPE_V:-f16}"
echo "[ocr] Fit to VRAM:      ${OCR_FIT:-off}"

OPTS=()
[[ "${OCR_LOG_PREFIX:-true}" == "true" ]] && OPTS+=(--log-prefix)
[[ "${OCR_NO_MMAP:-false}" == "true" ]] && OPTS+=(--no-mmap)
[[ "${OCR_MLOCK:-false}" == "true" ]] && OPTS+=(--mlock)
[[ -n "${OCR_DEVICE:-}" ]] && OPTS+=(--device "${OCR_DEVICE}")
[[ "${OCR_KV_OFFLOAD:-on}" == "on" ]] && OPTS+=(--kv-offload) || OPTS+=(--no-kv-offload)
[[ "${OCR_OP_OFFLOAD:-on}" == "on" ]] && OPTS+=(--op-offload) || OPTS+=(--no-op-offload)
[[ "${OCR_MMPROJ_OFFLOAD:-on}" == "on" ]] && OPTS+=(--mmproj-offload) || OPTS+=(--no-mmproj-offload)
[[ -n "${OCR_MMPROJ_PATH:-}" && -f "${OCR_MMPROJ_PATH}" ]] && OPTS+=(--mmproj "${OCR_MMPROJ_PATH}")

CUSTOM_ARGS=()
if [[ -n "${OCR_CUSTOM_ARGS_JSON:-}" && "${OCR_CUSTOM_ARGS_JSON}" != "[]" ]]; then
    while IFS= read -r -d '' token; do
        CUSTOM_ARGS+=("${token}")
    done < <(
        OCR_CUSTOM_ARGS_JSON="${OCR_CUSTOM_ARGS_JSON}" python3 - <<'PYCUSTOM'
import json
import os
import shlex
import sys
try:
    values = json.loads(os.environ.get("OCR_CUSTOM_ARGS_JSON", "[]"))
except Exception:
    values = []
for value in values:
    if not isinstance(value, str):
        continue
    for token in shlex.split(value):
        sys.stdout.buffer.write(token.encode("utf-8"))
        sys.stdout.buffer.write(b"\0")
PYCUSTOM
    )
fi

exec "${LLAMA_SERVER_BIN}" \
    --model "${OCR_MODEL_PATH:-${STACK_DIR}/models/GLM-OCR-F16.gguf}" \
    --alias "${OCR_MODEL_NAME:-ocr}" \
    --host "${OCR_HOST:-${LISTEN_HOST}}" \
    --port "${OCR_PORT:-8009}" \
    --ctx-size "${OCR_CTX_SIZE:-8192}" \
    --main-gpu "${OCR_MAIN_GPU:-0}" \
    --n-gpu-layers "${OCR_N_GPU_LAYERS:--1}" \
    --split-mode "${OCR_SPLIT_MODE:-layer}" \
    --tensor-split "${OCR_EFFECTIVE_TENSOR_SPLIT}" \
    --batch-size "${OCR_BATCH_SIZE:-2048}" \
    --ubatch-size "${OCR_UBATCH_SIZE:-512}" \
    --parallel "${OCR_N_PARALLEL:-1}" \
    --threads "${OCR_THREADS:--1}" \
    --threads-batch "${OCR_THREADS_BATCH:--1}" \
    --cache-type-k "${OCR_CACHE_TYPE_K:-f16}" \
    --cache-type-v "${OCR_CACHE_TYPE_V:-f16}" \
    --flash-attn "${OCR_FLASH_ATTN:-on}" \
    --temp "${OCR_TEMP:-0.1}" \
    --top-p "${OCR_TOP_P:-0.95}" \
    --top-k "${OCR_TOP_K:-1}" \
    --min-p "${OCR_MIN_P:-0.00}" \
    --fit "${OCR_FIT:-off}" \
    "${OPTS[@]}" \
    "${CUSTOM_ARGS[@]}" \
    "$@"
