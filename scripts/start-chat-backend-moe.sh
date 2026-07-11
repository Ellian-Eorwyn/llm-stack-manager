#!/usr/bin/env bash
# =============================================================================
# start-chat-backend-moe.sh
# Shared backend using the configurable MoE preset.
# Binds to 127.0.0.1:CHAT_BACKEND_PORT — not publicly accessible.
# chat-proxy.service exposes this on THINK_PORT and NOTHINK_PORT.
#
# Switch from Dense:  sudo bash scripts/switch-chat-model.sh moe
# =============================================================================
set -euo pipefail

STACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${STACK_DIR}/config/llm-stack.env"

CHAT_MOE_LABEL="${CHAT_SECONDARY_LABEL:-${CHAT_MOE_LABEL:-Secondary Backend}}"
CHAT_MOE_MODEL_NAME="${CHAT_SECONDARY_MODEL_NAME:-${CHAT_MOE_MODEL_NAME:-chat-moe}}"
CHAT_MOE_MODEL_PATH="${CHAT_SECONDARY_MODEL_PATH:-${CHAT_MOE_MODEL_PATH:-${CHAT_MODEL_PATH:-}}}"
CHAT_MOE_MMPROJ_PATH="${CHAT_SECONDARY_MMPROJ_PATH:-${CHAT_MOE_MMPROJ_PATH:-${CHAT_MMPROJ_PATH:-}}}"
CHAT_MOE_CTX_SIZE="${CHAT_SECONDARY_CTX_SIZE:-${CHAT_MOE_CTX_SIZE:-${CHAT_CTX_SIZE:-32768}}}"
CHAT_N_PARALLEL="${CHAT_SECONDARY_N_PARALLEL:-${CHAT_N_PARALLEL:-1}}"
CHAT_THREADS="${CHAT_SECONDARY_THREADS:-${CHAT_THREADS:--1}}"
CHAT_THREADS_BATCH="${CHAT_SECONDARY_THREADS_BATCH:-${CHAT_THREADS_BATCH:--1}}"
CHAT_N_GPU_LAYERS="${CHAT_SECONDARY_N_GPU_LAYERS:-${CHAT_N_GPU_LAYERS:--1}}"
CHAT_MAIN_GPU="${CHAT_SECONDARY_MAIN_GPU:-${CHAT_MAIN_GPU:-0}}"
CHAT_DEVICE="${CHAT_SECONDARY_DEVICE:-${CHAT_DEVICE:-}}"
CHAT_TENSOR_SPLIT="${CHAT_SECONDARY_TENSOR_SPLIT:-${CHAT_TENSOR_SPLIT:-1}}"
CHAT_SPLIT_MODE="${CHAT_SECONDARY_SPLIT_MODE:-${CHAT_SPLIT_MODE:-layer}}"
CHAT_KV_OFFLOAD="${CHAT_SECONDARY_KV_OFFLOAD:-${CHAT_KV_OFFLOAD:-on}}"
CHAT_OP_OFFLOAD="${CHAT_SECONDARY_OP_OFFLOAD:-${CHAT_OP_OFFLOAD:-on}}"
CHAT_MMPROJ_OFFLOAD="${CHAT_SECONDARY_MMPROJ_OFFLOAD:-${CHAT_MMPROJ_OFFLOAD:-on}}"
CHAT_FLASH_ATTN="${CHAT_SECONDARY_FLASH_ATTN:-${CHAT_FLASH_ATTN:-auto}}"
CHAT_CACHE_TYPE_K="${CHAT_SECONDARY_CACHE_TYPE_K:-${CHAT_CACHE_TYPE_K:-q8_0}}"
CHAT_CACHE_TYPE_V="${CHAT_SECONDARY_CACHE_TYPE_V:-${CHAT_CACHE_TYPE_V:-q8_0}}"
CHAT_CACHE_RAM="${CHAT_SECONDARY_CACHE_RAM:-${CHAT_CACHE_RAM:-8192}}"
CHAT_CTX_CHECKPOINTS="${CHAT_SECONDARY_CTX_CHECKPOINTS:-${CHAT_CTX_CHECKPOINTS:-32}}"
CHAT_SWA_FULL="${CHAT_SECONDARY_SWA_FULL:-${CHAT_SWA_FULL:-off}}"
CHAT_BATCH_SIZE="${CHAT_SECONDARY_BATCH_SIZE:-${CHAT_BATCH_SIZE:-2048}}"
CHAT_UBATCH_SIZE="${CHAT_SECONDARY_UBATCH_SIZE:-${CHAT_UBATCH_SIZE:-512}}"
CHAT_NO_MMAP="${CHAT_SECONDARY_NO_MMAP:-${CHAT_NO_MMAP:-false}}"
CHAT_MLOCK="${CHAT_SECONDARY_MLOCK:-${CHAT_MLOCK:-false}}"
CHAT_GPU_VISIBLE_DEVICES="${CHAT_SECONDARY_GPU_VISIBLE_DEVICES:-${CHAT_GPU_VISIBLE_DEVICES:-0}}"
CHAT_JINJA="${CHAT_SECONDARY_JINJA:-${CHAT_JINJA:-off}}"
CHAT_TEMPLATE_ID="${CHAT_SECONDARY_TEMPLATE_ID:-${CHAT_TEMPLATE_ID:-}}"
CHAT_FIT="${CHAT_SECONDARY_FIT:-${CHAT_FIT:-on}}"
CHAT_FIT_TARGET="${CHAT_SECONDARY_FIT_TARGET:-${CHAT_FIT_TARGET:-}}"
CHAT_FIT_CTX="${CHAT_SECONDARY_FIT_CTX:-${CHAT_FIT_CTX:-}}"
CHAT_CACHE_IDLE_SLOTS="${CHAT_SECONDARY_CACHE_IDLE_SLOTS:-${CHAT_CACHE_IDLE_SLOTS:-on}}"
CHAT_CACHE_REUSE="${CHAT_SECONDARY_CACHE_REUSE:-${CHAT_CACHE_REUSE:-0}}"
CHAT_TEMP="${CHAT_SECONDARY_TEMP:-${CHAT_TEMP:-1.0}}"
CHAT_TOP_P="${CHAT_SECONDARY_TOP_P:-${CHAT_TOP_P:-0.95}}"
CHAT_TOP_K="${CHAT_SECONDARY_TOP_K:-${CHAT_TOP_K:-20}}"
CHAT_MIN_P="${CHAT_SECONDARY_MIN_P:-${CHAT_MIN_P:-0.00}}"
CHAT_REASONING_FORMAT="${CHAT_SECONDARY_REASONING_FORMAT:-${CHAT_REASONING_FORMAT:-deepseek}}"
CHAT_CUSTOM_ARGS_JSON="${CHAT_SECONDARY_CUSTOM_ARGS_JSON:-${CHAT_CUSTOM_ARGS_JSON:-[]}}"
CHAT_PRESERVE_THINKING="${CHAT_SECONDARY_PRESERVE_THINKING:-${CHAT_PRESERVE_THINKING:-on}}"
CHAT_SPEC_METHOD="${CHAT_SECONDARY_SPEC_METHOD:-${CHAT_SPEC_METHOD:-off}}"
CHAT_SPEC_NGRAM_MOD="${CHAT_SECONDARY_SPEC_NGRAM_MOD:-${CHAT_SPEC_NGRAM_MOD:-off}}"
CHAT_SPEC_DRAFT_MODEL_PATH="${CHAT_SECONDARY_SPEC_DRAFT_MODEL_PATH:-${CHAT_SPEC_DRAFT_MODEL_PATH:-}}"
CHAT_SPEC_DRAFT_N_GPU_LAYERS="${CHAT_SECONDARY_SPEC_DRAFT_N_GPU_LAYERS:-${CHAT_SPEC_DRAFT_N_GPU_LAYERS:-auto}}"
CHAT_SPEC_DRAFT_DEVICES="${CHAT_SECONDARY_SPEC_DRAFT_DEVICES:-${CHAT_SPEC_DRAFT_DEVICES:-}}"
CHAT_SPEC_DRAFT_TYPE_K="${CHAT_SECONDARY_SPEC_DRAFT_TYPE_K:-${CHAT_SPEC_DRAFT_TYPE_K:-f16}}"
CHAT_SPEC_DRAFT_TYPE_V="${CHAT_SECONDARY_SPEC_DRAFT_TYPE_V:-${CHAT_SPEC_DRAFT_TYPE_V:-f16}}"
CHAT_SPEC_DRAFT_N_MAX="${CHAT_SECONDARY_SPEC_DRAFT_N_MAX:-${CHAT_SPEC_DRAFT_N_MAX:-6}}"
CHAT_SPEC_DRAFT_N_MIN="${CHAT_SECONDARY_SPEC_DRAFT_N_MIN:-${CHAT_SPEC_DRAFT_N_MIN:-0}}"
CHAT_SPEC_DRAFT_P_MIN="${CHAT_SECONDARY_SPEC_DRAFT_P_MIN:-${CHAT_SPEC_DRAFT_P_MIN:-0.75}}"
CHAT_SPEC_DRAFT_P_SPLIT="${CHAT_SECONDARY_SPEC_DRAFT_P_SPLIT:-${CHAT_SPEC_DRAFT_P_SPLIT:-0.10}}"
CHAT_SPEC_NGRAM_MOD_N_MATCH="${CHAT_SECONDARY_SPEC_NGRAM_MOD_N_MATCH:-${CHAT_SPEC_NGRAM_MOD_N_MATCH:-24}}"
CHAT_SPEC_NGRAM_MOD_N_MIN="${CHAT_SECONDARY_SPEC_NGRAM_MOD_N_MIN:-${CHAT_SPEC_NGRAM_MOD_N_MIN:-48}}"
CHAT_SPEC_NGRAM_MOD_N_MAX="${CHAT_SECONDARY_SPEC_NGRAM_MOD_N_MAX:-${CHAT_SPEC_NGRAM_MOD_N_MAX:-64}}"
CHAT_SPEC_NGRAM_SIZE_N="${CHAT_SECONDARY_SPEC_NGRAM_SIZE_N:-${CHAT_SPEC_NGRAM_SIZE_N:-12}}"
CHAT_SPEC_NGRAM_SIZE_M="${CHAT_SECONDARY_SPEC_NGRAM_SIZE_M:-${CHAT_SPEC_NGRAM_SIZE_M:-48}}"
CHAT_SPEC_NGRAM_MIN_HITS="${CHAT_SECONDARY_SPEC_NGRAM_MIN_HITS:-${CHAT_SPEC_NGRAM_MIN_HITS:-1}}"

LLAMA_SERVER_DIR="${LLAMA_SERVER_BIN%/*}"
export LD_LIBRARY_PATH="${LLAMA_SERVER_DIR}:${LD_LIBRARY_PATH:-}"
export DYLD_LIBRARY_PATH="${LLAMA_SERVER_DIR}:${DYLD_LIBRARY_PATH:-}"

export CUDA_VISIBLE_DEVICES="${CHAT_GPU_VISIBLE_DEVICES}"

echo "[chat-backend-moe] Starting shared llama-server backend (${CHAT_MOE_LABEL:-Backend MoE})"
echo "[chat-backend-moe] Model:            ${CHAT_MOE_MODEL_PATH}"
echo "[chat-backend-moe] MMProj:           ${CHAT_MOE_MMPROJ_PATH:-(none)}"
echo "[chat-backend-moe] Port:             ${CHAT_BACKEND_PORT} (localhost only)"
echo "[chat-backend-moe] Context:          ${CHAT_MOE_CTX_SIZE}"
echo "[chat-backend-moe] Main GPU:         ${CHAT_MAIN_GPU}"
echo "[chat-backend-moe] GPUs:             ${CUDA_VISIBLE_DEVICES} (split ${CHAT_TENSOR_SPLIT})"
echo "[chat-backend-moe] Device override:  ${CHAT_DEVICE:-auto}"
echo "[chat-backend-moe] Placement:        split=${CHAT_SPLIT_MODE} kv-offload=${CHAT_KV_OFFLOAD:-on} op-offload=${CHAT_OP_OFFLOAD:-on} mmproj-offload=${CHAT_MMPROJ_OFFLOAD:-on}"
echo "[chat-backend-moe] CPU threads:      ${CHAT_THREADS:--1} (batch=${CHAT_THREADS_BATCH:--1})"
echo "[chat-backend-moe] KV cache:         K=${CHAT_CACHE_TYPE_K} V=${CHAT_CACHE_TYPE_V}"
echo "[chat-backend-moe] Prompt cache:     ram=${CHAT_CACHE_RAM:-8192} MiB ctx-checkpoints=${CHAT_CTX_CHECKPOINTS:-32}"
echo "[chat-backend-moe] SWA full cache:   ${CHAT_SWA_FULL:-off}"
echo "[chat-backend-moe] Reasoning format: ${CHAT_REASONING_FORMAT:-deepseek}"
echo "[chat-backend-moe] Fit to VRAM:      ${CHAT_FIT:-on}"
echo "[chat-backend-moe] Speculative:      ${CHAT_SPEC_METHOD:-off}"
echo "[chat-backend-moe] Chat template:    ${CHAT_TEMPLATE_ID:-model default}"

OPTS=()
[[ "${CHAT_LOG_PREFIX:-true}" == "true" ]] && OPTS+=(--log-prefix)
[[ "${CHAT_NO_MMAP:-false}" == "true" ]] && OPTS+=(--no-mmap)
[[ "${CHAT_MLOCK:-false}" == "true" ]] && OPTS+=(--mlock)
[[ -n "${CHAT_DEVICE:-}" ]] && OPTS+=(--device "${CHAT_DEVICE}")
[[ "${CHAT_KV_OFFLOAD:-on}" == "on" ]] && OPTS+=(--kv-offload) || OPTS+=(--no-kv-offload)
[[ "${CHAT_OP_OFFLOAD:-on}" == "on" ]] && OPTS+=(--op-offload) || OPTS+=(--no-op-offload)
[[ "${CHAT_MMPROJ_OFFLOAD:-on}" == "on" ]] && OPTS+=(--mmproj-offload) || OPTS+=(--no-mmproj-offload)
[[ "${CHAT_SWA_FULL:-off}" == "on" ]] && OPTS+=(--swa-full)
[[ -n "${CHAT_FIT_TARGET:-}" ]] && OPTS+=(--fit-target "${CHAT_FIT_TARGET}")
[[ -n "${CHAT_FIT_CTX:-}" && "${CHAT_FIT_CTX}" != "0" ]] && OPTS+=(--fit-ctx "${CHAT_FIT_CTX}")
[[ "${CHAT_CACHE_IDLE_SLOTS:-on}" == "on" ]] && OPTS+=(--cache-idle-slots) || OPTS+=(--no-cache-idle-slots)
[[ -n "${CHAT_CACHE_REUSE:-}" && "${CHAT_CACHE_REUSE}" != "0" ]] && OPTS+=(--cache-reuse "${CHAT_CACHE_REUSE}")

CUSTOM_ARGS=()
if [[ -n "${CHAT_CUSTOM_ARGS_JSON:-}" && "${CHAT_CUSTOM_ARGS_JSON}" != "[]" ]]; then
    while IFS= read -r -d '' token; do
        CUSTOM_ARGS+=("${token}")
    done < <(
        python3 - <<'PY'
import json
import os
import shlex
import sys

try:
    values = json.loads(os.environ.get("CHAT_CUSTOM_ARGS_JSON", "[]"))
except Exception:
    values = []

for value in values:
    if not isinstance(value, str):
        continue
    for token in shlex.split(value):
        sys.stdout.buffer.write(token.encode("utf-8"))
        sys.stdout.buffer.write(b"\0")
PY
    )
fi

HAS_CUSTOM_JINJA=0
HAS_TEMPLATE_KWARGS=0
HAS_CUSTOM_CHAT_TEMPLATE=0
for token in "${CUSTOM_ARGS[@]:-}"; do
    if [[ "${token}" == "--jinja" ]]; then
        HAS_CUSTOM_JINJA=1
    fi
    if [[ "${token}" == "--chat-template-kwargs" ]]; then
        HAS_TEMPLATE_KWARGS=1
    fi
    if [[ "${token}" == "--chat-template" || "${token}" == "--chat-template-file" ]]; then
        HAS_CUSTOM_CHAT_TEMPLATE=1
    fi
done
if [[ "${CHAT_JINJA:-off}" == "on" && "${HAS_CUSTOM_JINJA}" -eq 0 ]]; then
    OPTS+=(--jinja)
fi
if [[ "${CHAT_PRESERVE_THINKING:-on}" == "on" && "${HAS_TEMPLATE_KWARGS}" -eq 0 ]]; then
    OPTS+=(--chat-template-kwargs '{"preserve_thinking": true}')
fi

if [[ -n "${CHAT_TEMPLATE_ID:-}" && "${HAS_CUSTOM_CHAT_TEMPLATE}" -eq 0 ]]; then
    if [[ ! "${CHAT_TEMPLATE_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "[chat-backend-moe] Invalid CHAT_TEMPLATE_ID: ${CHAT_TEMPLATE_ID}" >&2
        exit 1
    fi
    CHAT_TEMPLATE_FILE="${STACK_DIR}/config/chat-templates/${CHAT_TEMPLATE_ID}.jinja"
    if [[ ! -f "${CHAT_TEMPLATE_FILE}" ]]; then
        echo "[chat-backend-moe] Chat template not found: ${CHAT_TEMPLATE_FILE}" >&2
        exit 1
    fi
    OPTS+=(--chat-template-file "${CHAT_TEMPLATE_FILE}")
fi

SPEC_ARGS=()
SPEC_METHOD="${CHAT_SPEC_METHOD:-off}"
[[ "${SPEC_METHOD}" == "mtp" ]] && SPEC_METHOD="draft-mtp"
COMMON_SPEC_ARGS=(
    --spec-draft-n-max "${CHAT_SPEC_DRAFT_N_MAX:-6}"
    --spec-draft-n-min "${CHAT_SPEC_DRAFT_N_MIN:-0}"
    --spec-draft-p-min "${CHAT_SPEC_DRAFT_P_MIN:-0.75}"
    --spec-draft-p-split "${CHAT_SPEC_DRAFT_P_SPLIT:-0.10}"
)
DRAFT_CACHE_SPEC_ARGS=(
    --spec-draft-type-k "${CHAT_SPEC_DRAFT_TYPE_K:-f16}"
    --spec-draft-type-v "${CHAT_SPEC_DRAFT_TYPE_V:-f16}"
)
NGRAM_MOD_SPEC_ARGS=(
    --spec-ngram-mod-n-match "${CHAT_SPEC_NGRAM_MOD_N_MATCH:-24}"
    --spec-ngram-mod-n-min "${CHAT_SPEC_NGRAM_MOD_N_MIN:-48}"
    --spec-ngram-mod-n-max "${CHAT_SPEC_NGRAM_MOD_N_MAX:-64}"
)
NGRAM_SIMPLE_SPEC_ARGS=(
    --spec-ngram-simple-size-n "${CHAT_SPEC_NGRAM_SIZE_N:-12}"
    --spec-ngram-simple-size-m "${CHAT_SPEC_NGRAM_SIZE_M:-48}"
    --spec-ngram-simple-min-hits "${CHAT_SPEC_NGRAM_MIN_HITS:-1}"
)
NGRAM_MAP_K_SPEC_ARGS=(
    --spec-ngram-map-k-size-n "${CHAT_SPEC_NGRAM_SIZE_N:-12}"
    --spec-ngram-map-k-size-m "${CHAT_SPEC_NGRAM_SIZE_M:-48}"
    --spec-ngram-map-k-min-hits "${CHAT_SPEC_NGRAM_MIN_HITS:-1}"
)
NGRAM_MAP_K4V_SPEC_ARGS=(
    --spec-ngram-map-k4v-size-n "${CHAT_SPEC_NGRAM_SIZE_N:-12}"
    --spec-ngram-map-k4v-size-m "${CHAT_SPEC_NGRAM_SIZE_M:-48}"
    --spec-ngram-map-k4v-min-hits "${CHAT_SPEC_NGRAM_MIN_HITS:-1}"
)
if [[ "${SPEC_METHOD}" == "draft-model" ]]; then
    if [[ -z "${CHAT_SPEC_DRAFT_MODEL_PATH:-}" ]]; then
        echo "[chat-backend-moe] Speculative decoding is enabled, but CHAT_SPEC_DRAFT_MODEL_PATH is empty." >&2
        exit 1
    fi
    if [[ ! -f "${CHAT_SPEC_DRAFT_MODEL_PATH}" ]]; then
        echo "[chat-backend-moe] Draft model not found: ${CHAT_SPEC_DRAFT_MODEL_PATH}" >&2
        exit 1
    fi

    echo "[chat-backend-moe] Draft model:      ${CHAT_SPEC_DRAFT_MODEL_PATH}"
    echo "[chat-backend-moe] Draft GPU layers: ${CHAT_SPEC_DRAFT_N_GPU_LAYERS:-auto}"
    echo "[chat-backend-moe] Draft devices:    ${CHAT_SPEC_DRAFT_DEVICES:-auto}"
    echo "[chat-backend-moe] Draft n-max/min:  ${CHAT_SPEC_DRAFT_N_MAX:-6}/${CHAT_SPEC_DRAFT_N_MIN:-0}"
    echo "[chat-backend-moe] Draft p-min/split:${CHAT_SPEC_DRAFT_P_MIN:-0.75}/${CHAT_SPEC_DRAFT_P_SPLIT:-0.10}"

    SPEC_ARGS+=(
        --spec-draft-model "${CHAT_SPEC_DRAFT_MODEL_PATH}"
        --spec-draft-ngl "${CHAT_SPEC_DRAFT_N_GPU_LAYERS:-auto}"
        "${DRAFT_CACHE_SPEC_ARGS[@]}"
        "${COMMON_SPEC_ARGS[@]}"
    )
    [[ -n "${CHAT_SPEC_DRAFT_DEVICES:-}" ]] && SPEC_ARGS+=(--spec-draft-device "${CHAT_SPEC_DRAFT_DEVICES}")
elif [[ "${SPEC_METHOD}" != "off" ]]; then
    SPEC_ARGS+=(--spec-type "${SPEC_METHOD}" "${DRAFT_CACHE_SPEC_ARGS[@]}" "${COMMON_SPEC_ARGS[@]}")
    if [[ "${SPEC_METHOD}" == "draft-simple" || "${SPEC_METHOD}" == "draft-eagle3" || "${SPEC_METHOD}" == "draft-dflash" ]]; then
        if [[ -z "${CHAT_SPEC_DRAFT_MODEL_PATH:-}" ]]; then
            echo "[chat-backend-moe] ${SPEC_METHOD} is enabled, but CHAT_SPEC_DRAFT_MODEL_PATH is empty." >&2
            exit 1
        fi
        if [[ ! -f "${CHAT_SPEC_DRAFT_MODEL_PATH}" ]]; then
            echo "[chat-backend-moe] Draft model not found: ${CHAT_SPEC_DRAFT_MODEL_PATH}" >&2
            exit 1
        fi
        SPEC_ARGS+=(--spec-draft-model "${CHAT_SPEC_DRAFT_MODEL_PATH}" --spec-draft-ngl "${CHAT_SPEC_DRAFT_N_GPU_LAYERS:-auto}")
        [[ -n "${CHAT_SPEC_DRAFT_DEVICES:-}" ]] && SPEC_ARGS+=(--spec-draft-device "${CHAT_SPEC_DRAFT_DEVICES}")
    fi
    if [[ ",${SPEC_METHOD}," == *,ngram-mod,* ]]; then
        echo "[chat-backend-moe] N-gram mod:       match=${CHAT_SPEC_NGRAM_MOD_N_MATCH:-24} min=${CHAT_SPEC_NGRAM_MOD_N_MIN:-48} max=${CHAT_SPEC_NGRAM_MOD_N_MAX:-64}"
        SPEC_ARGS+=("${NGRAM_MOD_SPEC_ARGS[@]}")
    fi
    if [[ ",${SPEC_METHOD}," == *,ngram-simple,* ]]; then
        SPEC_ARGS+=("${NGRAM_SIMPLE_SPEC_ARGS[@]}")
    fi
    if [[ ",${SPEC_METHOD}," == *,ngram-map-k,* ]]; then
        SPEC_ARGS+=("${NGRAM_MAP_K_SPEC_ARGS[@]}")
    fi
    if [[ ",${SPEC_METHOD}," == *,ngram-map-k4v,* ]]; then
        SPEC_ARGS+=("${NGRAM_MAP_K4V_SPEC_ARGS[@]}")
    fi
    if [[ ",${SPEC_METHOD}," != *,ngram-mod,* && "${CHAT_SPEC_NGRAM_MOD:-off}" == "on" ]]; then
        echo "[chat-backend-moe] N-gram mod assist requested, but this llama-server build only accepts ngram-mod as a standalone --spec-type; leaving --spec-type=${SPEC_METHOD}."
    fi
fi
[[ -n "${CHAT_MOE_MMPROJ_PATH:-}" && -f "${CHAT_MOE_MMPROJ_PATH}" ]] && OPTS+=(--mmproj "${CHAT_MOE_MMPROJ_PATH}")

exec "${LLAMA_SERVER_BIN}" \
    --model "${CHAT_MOE_MODEL_PATH}" \
    --alias "${CHAT_MOE_MODEL_NAME:-chat-moe}" \
    --host "${CHAT_BACKEND_HOST}" \
    --port "${CHAT_BACKEND_PORT}" \
    --ctx-size "${CHAT_MOE_CTX_SIZE}" \
    --main-gpu "${CHAT_MAIN_GPU}" \
    --n-gpu-layers "${CHAT_N_GPU_LAYERS}" \
    --split-mode "${CHAT_SPLIT_MODE}" \
    --tensor-split "${CHAT_TENSOR_SPLIT}" \
    --batch-size "${CHAT_BATCH_SIZE}" \
    --ubatch-size "${CHAT_UBATCH_SIZE}" \
    --parallel "${CHAT_N_PARALLEL}" \
    --threads "${CHAT_THREADS:--1}" \
    --threads-batch "${CHAT_THREADS_BATCH:--1}" \
    --cache-type-k "${CHAT_CACHE_TYPE_K}" \
    --cache-type-v "${CHAT_CACHE_TYPE_V}" \
    --cache-ram "${CHAT_CACHE_RAM:-8192}" \
    --ctx-checkpoints "${CHAT_CTX_CHECKPOINTS:-32}" \
    --flash-attn "${CHAT_FLASH_ATTN}" \
    --temp "${CHAT_TEMP}" \
    --top-p "${CHAT_TOP_P}" \
    --top-k "${CHAT_TOP_K}" \
    --min-p "${CHAT_MIN_P}" \
    --reasoning-format "${CHAT_REASONING_FORMAT:-deepseek}" \
    --fit "${CHAT_FIT:-on}" \
    "${OPTS[@]}" \
    "${SPEC_ARGS[@]}" \
    "${CUSTOM_ARGS[@]}" \
    "$@"
