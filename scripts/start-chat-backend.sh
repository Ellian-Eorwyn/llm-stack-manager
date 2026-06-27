#!/usr/bin/env bash
# =============================================================================
# start-chat-backend.sh
# Launches the shared llama-server backend that powers both the think and
# nothink endpoints via llm-chat-proxy.py.
#
# Binds to 127.0.0.1:CHAT_BACKEND_PORT (localhost only — not public).
# The proxy (chat-proxy.service) is the public-facing interface.
#
# IMPORTANT: think and nothink services must be stopped before
# enabling this backend+proxy pair, otherwise both will try to load the
# same model and you'll run out of VRAM.
# =============================================================================
set -euo pipefail

STACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${STACK_DIR}/config/llm-stack.env"
LLAMA_SERVER_DIR="${LLAMA_SERVER_BIN%/*}"
export LD_LIBRARY_PATH="${LLAMA_SERVER_DIR}:${LD_LIBRARY_PATH:-}"
export DYLD_LIBRARY_PATH="${LLAMA_SERVER_DIR}:${DYLD_LIBRARY_PATH:-}"

export CUDA_VISIBLE_DEVICES="${CHAT_GPU_VISIBLE_DEVICES}"

echo "[chat-backend] Starting shared llama-server backend"
echo "[chat-backend] Model:   ${CHAT_MODEL_PATH}"
echo "[chat-backend] MMProj:  ${CHAT_MMPROJ_PATH:-(none)}"
echo "[chat-backend] Port:    ${CHAT_BACKEND_PORT} (localhost only)"
echo "[chat-backend] Context: ${CHAT_CTX_SIZE}"
echo "[chat-backend] Main GPU: ${CHAT_MAIN_GPU}"
echo "[chat-backend] GPUs:    ${CUDA_VISIBLE_DEVICES} (split ${CHAT_TENSOR_SPLIT})"
echo "[chat-backend] Device override: ${CHAT_DEVICE:-auto}"
echo "[chat-backend] Placement: split=${CHAT_SPLIT_MODE} kv-offload=${CHAT_KV_OFFLOAD:-on} op-offload=${CHAT_OP_OFFLOAD:-on} mmproj-offload=${CHAT_MMPROJ_OFFLOAD:-on}"
echo "[chat-backend] CPU threads:      ${CHAT_THREADS:--1} (batch=${CHAT_THREADS_BATCH:--1})"
echo "[chat-backend] KV cache: K=${CHAT_CACHE_TYPE_K} V=${CHAT_CACHE_TYPE_V}"
echo "[chat-backend] Reasoning format: ${CHAT_REASONING_FORMAT:-deepseek}"
echo "[chat-backend] Fit to VRAM:      ${CHAT_FIT:-on}"
echo "[chat-backend] Public ports: ${THINK_PORT} (think) ${NOTHINK_PORT} (nothink) via proxy"
echo "[chat-backend] Speculative:      ${CHAT_SPEC_METHOD:-off}"
echo "[chat-backend] Chat template:    ${CHAT_TEMPLATE_ID:-model default}"

OPTS=()
[[ "${CHAT_LOG_PREFIX:-true}" == "true" ]] && OPTS+=(--log-prefix)
[[ "${CHAT_NO_MMAP:-false}" == "true" ]] && OPTS+=(--no-mmap)
[[ "${CHAT_MLOCK:-false}" == "true" ]] && OPTS+=(--mlock)
[[ -n "${CHAT_DEVICE:-}" ]] && OPTS+=(--device "${CHAT_DEVICE}")
[[ "${CHAT_KV_OFFLOAD:-on}" == "on" ]] && OPTS+=(--kv-offload) || OPTS+=(--no-kv-offload)
[[ "${CHAT_OP_OFFLOAD:-on}" == "on" ]] && OPTS+=(--op-offload) || OPTS+=(--no-op-offload)
[[ "${CHAT_MMPROJ_OFFLOAD:-on}" == "on" ]] && OPTS+=(--mmproj-offload) || OPTS+=(--no-mmproj-offload)

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
        echo "[chat-backend] Invalid CHAT_TEMPLATE_ID: ${CHAT_TEMPLATE_ID}" >&2
        exit 1
    fi
    CHAT_TEMPLATE_FILE="${STACK_DIR}/config/chat-templates/${CHAT_TEMPLATE_ID}.jinja"
    if [[ ! -f "${CHAT_TEMPLATE_FILE}" ]]; then
        echo "[chat-backend] Chat template not found: ${CHAT_TEMPLATE_FILE}" >&2
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
NGRAM_MOD_SPEC_ARGS=(
    --spec-ngram-mod-n-match "${CHAT_SPEC_NGRAM_MOD_N_MATCH:-24}"
    --spec-ngram-mod-n-min "${CHAT_SPEC_NGRAM_MOD_N_MIN:-48}"
    --spec-ngram-mod-n-max "${CHAT_SPEC_NGRAM_MOD_N_MAX:-64}"
)
if [[ "${SPEC_METHOD}" == "draft-model" ]]; then
    if [[ -z "${CHAT_SPEC_DRAFT_MODEL_PATH:-}" ]]; then
        echo "[chat-backend] Speculative decoding is enabled, but CHAT_SPEC_DRAFT_MODEL_PATH is empty." >&2
        exit 1
    fi
    if [[ ! -f "${CHAT_SPEC_DRAFT_MODEL_PATH}" ]]; then
        echo "[chat-backend] Draft model not found: ${CHAT_SPEC_DRAFT_MODEL_PATH}" >&2
        exit 1
    fi

    echo "[chat-backend] Draft model:      ${CHAT_SPEC_DRAFT_MODEL_PATH}"
    echo "[chat-backend] Draft GPU layers: ${CHAT_SPEC_DRAFT_N_GPU_LAYERS:-auto}"
    echo "[chat-backend] Draft devices:    ${CHAT_SPEC_DRAFT_DEVICES:-auto}"
    echo "[chat-backend] Draft ctx:        ${CHAT_SPEC_DRAFT_CTX_SIZE:-0}"
    echo "[chat-backend] Draft n-max/min:  ${CHAT_SPEC_DRAFT_N_MAX:-6}/${CHAT_SPEC_DRAFT_N_MIN:-0}"
    echo "[chat-backend] Draft p-min/split:${CHAT_SPEC_DRAFT_P_MIN:-0.75}/${CHAT_SPEC_DRAFT_P_SPLIT:-0.10}"

    SPEC_ARGS+=(
        --spec-draft-model "${CHAT_SPEC_DRAFT_MODEL_PATH}"
        --spec-draft-ngl "${CHAT_SPEC_DRAFT_N_GPU_LAYERS:-auto}"
        "${COMMON_SPEC_ARGS[@]}"
    )
    [[ -n "${CHAT_SPEC_DRAFT_DEVICES:-}" ]] && SPEC_ARGS+=(--spec-draft-device "${CHAT_SPEC_DRAFT_DEVICES}")
    [[ -n "${CHAT_SPEC_DRAFT_CTX_SIZE:-}" ]] && SPEC_ARGS+=(--spec-draft-ctx-size "${CHAT_SPEC_DRAFT_CTX_SIZE}")
elif [[ "${SPEC_METHOD}" != "off" ]]; then
    SPEC_ARGS+=(--spec-type "${SPEC_METHOD}" "${COMMON_SPEC_ARGS[@]}")
    if [[ ",${SPEC_METHOD}," == *,ngram-mod,* ]]; then
        echo "[chat-backend] N-gram mod:       match=${CHAT_SPEC_NGRAM_MOD_N_MATCH:-24} min=${CHAT_SPEC_NGRAM_MOD_N_MIN:-48} max=${CHAT_SPEC_NGRAM_MOD_N_MAX:-64}"
        SPEC_ARGS+=("${NGRAM_MOD_SPEC_ARGS[@]}")
    elif [[ "${CHAT_SPEC_NGRAM_MOD:-off}" == "on" ]]; then
        echo "[chat-backend] N-gram mod assist requested, but this llama-server build only accepts ngram-mod as a standalone --spec-type; leaving --spec-type=${SPEC_METHOD}."
    fi
fi

# Only add --mmproj if the path is non-empty and the file exists
[[ -n "${CHAT_MMPROJ_PATH:-}" && -f "${CHAT_MMPROJ_PATH}" ]] && OPTS+=(--mmproj "${CHAT_MMPROJ_PATH}")

exec "${LLAMA_SERVER_BIN}" \
    --model "${CHAT_MODEL_PATH}" \
    --alias "${CHAT_MODEL_NAME:-chat-custom}" \
    --host "${CHAT_BACKEND_HOST}" \
    --port "${CHAT_BACKEND_PORT}" \
    --ctx-size "${CHAT_CTX_SIZE}" \
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
# NOTE: No --chat-template-kwargs here. Thinking is controlled per-request
# by the proxy, so this backend can be reused across model families.
