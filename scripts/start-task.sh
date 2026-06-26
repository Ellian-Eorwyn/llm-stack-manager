#!/usr/bin/env bash
# =============================================================================
# start-task.sh
# Launches llama-server for the small Qwen task model.
# Thinking is disabled — intended for fast, structured, low-latency tasks.
# Port: TASK_PORT (default 8007)
# GPU:  TASK_GPU_VISIBLE_DEVICES (default GPU 0, single device)
# =============================================================================
set -euo pipefail

# Load configuration
STACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${STACK_DIR}/config/llm-stack.env"
LLAMA_SERVER_DIR="${LLAMA_SERVER_BIN%/*}"
export LD_LIBRARY_PATH="${LLAMA_SERVER_DIR}:${LD_LIBRARY_PATH:-}"

# Pin to a single GPU for this small model
export CUDA_VISIBLE_DEVICES="${TASK_GPU_VISIBLE_DEVICES}"

echo "[task] Starting llama-server (small task model)"
echo "[task] Model:            ${TASK_MODEL_PATH}"
echo "[task] MMProj:           ${TASK_MMPROJ_PATH}"
echo "[task] Port:             ${TASK_PORT}"
echo "[task] Context:          ${TASK_CTX_SIZE}"
echo "[task] Batch:            ${TASK_BATCH_SIZE} (uBatch=${TASK_UBATCH_SIZE})"
echo "[task] Main GPU:         ${TASK_MAIN_GPU}"
echo "[task] GPU:              ${CUDA_VISIBLE_DEVICES}"
echo "[task] Device override:  ${TASK_DEVICE:-auto}"
echo "[task] Placement:        split=${TASK_SPLIT_MODE} kv-offload=${TASK_KV_OFFLOAD:-on} op-offload=${TASK_OP_OFFLOAD:-on} mmproj-offload=${TASK_MMPROJ_OFFLOAD:-on}"
echo "[task] CPU threads:      ${TASK_THREADS:--1} (batch=${TASK_THREADS_BATCH:--1})"
echo "[task] KV cache:         K=${TASK_CACHE_TYPE_K} V=${TASK_CACHE_TYPE_V}"
echo "[task] Jinja:            ${TASK_JINJA:-off}"
echo "[task] Thinking:         ${TASK_THINKING:-off}"
echo "[task] Reasoning format: ${TASK_REASONING_FORMAT:-none}"
echo "[task] Fit to VRAM:      ${TASK_FIT:-on}"
echo "[task] Speculative:      ${TASK_SPEC_METHOD:-off}"
echo "[task] Chat template:    ${TASK_CHAT_TEMPLATE_ID:-model default}"

# Build optional boolean flags from config
OPTS=()
[[ "${TASK_LOG_PREFIX:-true}" == "true" ]] && OPTS+=(--log-prefix)
[[ "${TASK_NO_MMAP:-false}" == "true" ]] && OPTS+=(--no-mmap)
[[ "${TASK_MLOCK:-false}" == "true" ]] && OPTS+=(--mlock)
[[ -n "${TASK_DEVICE:-}" ]] && OPTS+=(--device "${TASK_DEVICE}")
[[ "${TASK_KV_OFFLOAD:-on}" == "on" ]] && OPTS+=(--kv-offload) || OPTS+=(--no-kv-offload)
[[ "${TASK_OP_OFFLOAD:-on}" == "on" ]] && OPTS+=(--op-offload) || OPTS+=(--no-op-offload)
[[ "${TASK_MMPROJ_OFFLOAD:-on}" == "on" ]] && OPTS+=(--mmproj-offload) || OPTS+=(--no-mmproj-offload)
[[ -n "${TASK_MMPROJ_PATH:-}" && -f "${TASK_MMPROJ_PATH}" ]] && OPTS+=(--mmproj "${TASK_MMPROJ_PATH}")

CUSTOM_ARGS=()
if [[ -n "${TASK_CUSTOM_ARGS_JSON:-}" && "${TASK_CUSTOM_ARGS_JSON}" != "[]" ]]; then
    while IFS= read -r -d '' token; do
        CUSTOM_ARGS+=("${token}")
    done < <(
        python3 - <<'PY'
import json
import os
import shlex
import sys

try:
    values = json.loads(os.environ.get("TASK_CUSTOM_ARGS_JSON", "[]"))
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

# Build chat-template-kwargs based on thinking setting
if [[ "${TASK_THINKING:-off}" == "on" ]]; then
    TEMPLATE_KWARGS='{"enable_thinking":true}'
else
    TEMPLATE_KWARGS='{"enable_thinking":false}'
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
if [[ "${TASK_JINJA:-off}" == "on" && "${HAS_CUSTOM_JINJA}" -eq 0 ]]; then
    OPTS+=(--jinja)
fi
if [[ "${HAS_TEMPLATE_KWARGS}" -eq 0 ]]; then
    OPTS+=(--chat-template-kwargs "${TEMPLATE_KWARGS}")
fi

if [[ -n "${TASK_CHAT_TEMPLATE_ID:-}" && "${HAS_CUSTOM_CHAT_TEMPLATE}" -eq 0 ]]; then
    if [[ ! "${TASK_CHAT_TEMPLATE_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "[task] Invalid TASK_CHAT_TEMPLATE_ID: ${TASK_CHAT_TEMPLATE_ID}" >&2
        exit 1
    fi
    TASK_CHAT_TEMPLATE_FILE="${STACK_DIR}/config/chat-templates/${TASK_CHAT_TEMPLATE_ID}.jinja"
    if [[ ! -f "${TASK_CHAT_TEMPLATE_FILE}" ]]; then
        echo "[task] Chat template not found: ${TASK_CHAT_TEMPLATE_FILE}" >&2
        exit 1
    fi
    OPTS+=(--chat-template-file "${TASK_CHAT_TEMPLATE_FILE}")
fi

# Speculative decoding (mirrors chat-backend logic)
SPEC_ARGS=()
SPEC_METHOD="${TASK_SPEC_METHOD:-off}"
[[ "${SPEC_METHOD}" == "mtp" ]] && SPEC_METHOD="draft-mtp"
COMMON_SPEC_ARGS=(
    --spec-draft-n-max "${TASK_SPEC_DRAFT_N_MAX:-6}"
    --spec-draft-n-min "${TASK_SPEC_DRAFT_N_MIN:-0}"
    --spec-draft-p-min "${TASK_SPEC_DRAFT_P_MIN:-0.75}"
    --spec-draft-p-split "${TASK_SPEC_DRAFT_P_SPLIT:-0.10}"
)
NGRAM_MOD_SPEC_ARGS=(
    --spec-ngram-mod-n-match "${TASK_SPEC_NGRAM_MOD_N_MATCH:-24}"
    --spec-ngram-mod-n-min "${TASK_SPEC_NGRAM_MOD_N_MIN:-48}"
    --spec-ngram-mod-n-max "${TASK_SPEC_NGRAM_MOD_N_MAX:-64}"
)
if [[ "${SPEC_METHOD}" == "draft-model" ]]; then
    if [[ -z "${TASK_SPEC_DRAFT_MODEL_PATH:-}" ]]; then
        echo "[task] Speculative decoding is enabled, but TASK_SPEC_DRAFT_MODEL_PATH is empty." >&2
        exit 1
    fi
    if [[ ! -f "${TASK_SPEC_DRAFT_MODEL_PATH}" ]]; then
        echo "[task] Draft model not found: ${TASK_SPEC_DRAFT_MODEL_PATH}" >&2
        exit 1
    fi

    echo "[task] Draft model:      ${TASK_SPEC_DRAFT_MODEL_PATH}"
    echo "[task] Draft GPU layers: ${TASK_SPEC_DRAFT_N_GPU_LAYERS:-auto}"
    echo "[task] Draft devices:    ${TASK_SPEC_DRAFT_DEVICES:-auto}"
    echo "[task] Draft ctx:        ${TASK_SPEC_DRAFT_CTX_SIZE:-0}"
    echo "[task] Draft n-max/min:  ${TASK_SPEC_DRAFT_N_MAX:-6}/${TASK_SPEC_DRAFT_N_MIN:-0}"
    echo "[task] Draft p-min/split:${TASK_SPEC_DRAFT_P_MIN:-0.75}/${TASK_SPEC_DRAFT_P_SPLIT:-0.10}"

    SPEC_ARGS+=(
        --spec-draft-model "${TASK_SPEC_DRAFT_MODEL_PATH}"
        --spec-draft-ngl "${TASK_SPEC_DRAFT_N_GPU_LAYERS:-auto}"
        "${COMMON_SPEC_ARGS[@]}"
    )
    [[ -n "${TASK_SPEC_DRAFT_DEVICES:-}" ]] && SPEC_ARGS+=(--spec-draft-device "${TASK_SPEC_DRAFT_DEVICES}")
    [[ -n "${TASK_SPEC_DRAFT_CTX_SIZE:-}" ]] && SPEC_ARGS+=(--spec-draft-ctx-size "${TASK_SPEC_DRAFT_CTX_SIZE}")
elif [[ "${SPEC_METHOD}" != "off" ]]; then
    SPEC_ARGS+=(--spec-type "${SPEC_METHOD}" "${COMMON_SPEC_ARGS[@]}")
    if [[ ",${SPEC_METHOD}," == *,ngram-mod,* ]]; then
        echo "[task] N-gram mod:       match=${TASK_SPEC_NGRAM_MOD_N_MATCH:-24} min=${TASK_SPEC_NGRAM_MOD_N_MIN:-48} max=${TASK_SPEC_NGRAM_MOD_N_MAX:-64}"
        SPEC_ARGS+=("${NGRAM_MOD_SPEC_ARGS[@]}")
    elif [[ "${TASK_SPEC_NGRAM_MOD:-off}" == "on" ]]; then
        echo "[task] N-gram mod assist requested, but this llama-server build only accepts ngram-mod as a standalone --spec-type; leaving --spec-type=${SPEC_METHOD}."
    fi
fi

exec "${LLAMA_SERVER_BIN}" \
    --model "${TASK_MODEL_PATH}" \
    --alias "${TASK_MODEL_NAME:-task}" \
    --host "${LISTEN_HOST}" \
    --port "${TASK_PORT}" \
    --ctx-size "${TASK_CTX_SIZE}" \
    --main-gpu "${TASK_MAIN_GPU}" \
    --n-gpu-layers "${TASK_N_GPU_LAYERS}" \
    --split-mode "${TASK_SPLIT_MODE}" \
    --tensor-split "${TASK_TENSOR_SPLIT}" \
    --batch-size "${TASK_BATCH_SIZE}" \
    --ubatch-size "${TASK_UBATCH_SIZE}" \
    --parallel "${TASK_N_PARALLEL}" \
    --threads "${TASK_THREADS:--1}" \
    --threads-batch "${TASK_THREADS_BATCH:--1}" \
    --cache-type-k "${TASK_CACHE_TYPE_K}" \
    --cache-type-v "${TASK_CACHE_TYPE_V}" \
    --flash-attn "${TASK_FLASH_ATTN}" \
    --temp "${TASK_TEMP}" \
    --top-p "${TASK_TOP_P}" \
    --top-k "${TASK_TOP_K}" \
    --min-p "${TASK_MIN_P}" \
    --presence-penalty "${TASK_PRESENCE_PENALTY:-0.00}" \
    --repeat-penalty "${TASK_REPEAT_PENALTY:-1.00}" \
    --reasoning-format "${TASK_REASONING_FORMAT:-none}" \
    --fit "${TASK_FIT:-on}" \
    "${OPTS[@]}" \
    "${SPEC_ARGS[@]}" \
    "${CUSTOM_ARGS[@]}" \
    "$@"
