#!/usr/bin/env bash
# =============================================================================
# start-chat-backend2.sh
# Launches the shared llama-server backend that powers both the think and
# nothink endpoints via llm-chat-proxy.py.
#
# Binds to 127.0.0.1:CHAT2_BACKEND_PORT (localhost only — not public).
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

: "${CHAT2_BACKEND_HOST:=${CHAT_BACKEND_HOST:-127.0.0.1}}"
: "${CHAT2_BACKEND_PORT:=8020}"
: "${CHAT2_CTX_SIZE:=0}"
: "${CHAT2_MAIN_GPU:=0}"
: "${CHAT2_N_GPU_LAYERS:=-1}"
: "${CHAT2_SPLIT_MODE:=layer}"
: "${CHAT2_TENSOR_SPLIT:=1}"
: "${CHAT2_BATCH_SIZE:=2048}"
: "${CHAT2_UBATCH_SIZE:=512}"
: "${CHAT2_N_PARALLEL:=1}"
: "${CHAT2_THREADS:=-1}"
: "${CHAT2_THREADS_BATCH:=-1}"
: "${CHAT2_CACHE_RAM:=8192}"
: "${CHAT2_CTX_CHECKPOINTS:=32}"
: "${CHAT2_FLASH_ATTN:=auto}"
: "${CHAT2_TEMP:=1.0}"
: "${CHAT2_TOP_P:=0.95}"
: "${CHAT2_TOP_K:=20}"
: "${CHAT2_MIN_P:=0.00}"
: "${CHAT2_GPU_VISIBLE_DEVICES:=${CHAT_GPU_VISIBLE_DEVICES:-0,1}}"

export CUDA_VISIBLE_DEVICES="${CHAT2_GPU_VISIBLE_DEVICES}"

echo "[chat-backend2] Starting shared llama-server backend"
echo "[chat-backend2] Model:   ${CHAT2_MODEL_PATH}"
echo "[chat-backend2] MMProj:  ${CHAT2_MMPROJ_PATH:-(none)}"
echo "[chat-backend2] Port:    ${CHAT2_BACKEND_PORT} (localhost only)"
echo "[chat-backend2] Context: ${CHAT2_CTX_SIZE}"
echo "[chat-backend2] Main GPU: ${CHAT2_MAIN_GPU}"
echo "[chat-backend2] GPUs:    ${CUDA_VISIBLE_DEVICES} (split ${CHAT2_TENSOR_SPLIT})"
echo "[chat-backend2] Device override: ${CHAT2_DEVICE:-auto}"
echo "[chat-backend2] Placement: split=${CHAT2_SPLIT_MODE} kv-offload=${CHAT2_KV_OFFLOAD:-on} op-offload=${CHAT2_OP_OFFLOAD:-on} mmproj-offload=${CHAT2_MMPROJ_OFFLOAD:-on}"
echo "[chat-backend2] CPU threads:      ${CHAT2_THREADS:--1} (batch=${CHAT2_THREADS_BATCH:--1})"
echo "[chat-backend2] KV cache: K=${CHAT2_CACHE_TYPE_K} V=${CHAT2_CACHE_TYPE_V}"
echo "[chat-backend2] SWA full cache:    ${CHAT2_SWA_FULL:-off}"
echo "[chat-backend2] Reasoning format: ${CHAT2_REASONING_FORMAT:-deepseek}"
echo "[chat-backend2] Fit to VRAM:      ${CHAT2_FIT:-on}"
echo "[chat-backend2] Public ports: ${THINK_PORT} (think) ${NOTHINK_PORT} (nothink) via proxy"
echo "[chat-backend2] Speculative:      ${CHAT2_SPEC_METHOD:-off}"
echo "[chat-backend2] Chat template:    ${CHAT2_TEMPLATE_ID:-model default}"

OPTS=()
[[ "${CHAT2_LOG_PREFIX:-true}" == "true" ]] && OPTS+=(--log-prefix)
[[ "${CHAT2_NO_MMAP:-false}" == "true" ]] && OPTS+=(--no-mmap)
[[ "${CHAT2_MLOCK:-false}" == "true" ]] && OPTS+=(--mlock)
[[ -n "${CHAT2_DEVICE:-}" ]] && OPTS+=(--device "${CHAT2_DEVICE}")
[[ "${CHAT2_KV_OFFLOAD:-on}" == "on" ]] && OPTS+=(--kv-offload) || OPTS+=(--no-kv-offload)
[[ "${CHAT2_OP_OFFLOAD:-on}" == "on" ]] && OPTS+=(--op-offload) || OPTS+=(--no-op-offload)
[[ "${CHAT2_MMPROJ_OFFLOAD:-on}" == "on" ]] && OPTS+=(--mmproj-offload) || OPTS+=(--no-mmproj-offload)
[[ "${CHAT2_SWA_FULL:-off}" == "on" ]] && OPTS+=(--swa-full)
[[ -n "${CHAT2_FIT_TARGET:-}" ]] && OPTS+=(--fit-target "${CHAT2_FIT_TARGET}")
[[ -n "${CHAT2_FIT_CTX:-}" && "${CHAT2_FIT_CTX}" != "0" ]] && OPTS+=(--fit-ctx "${CHAT2_FIT_CTX}")
[[ "${CHAT2_CACHE_IDLE_SLOTS:-on}" == "on" ]] && OPTS+=(--cache-idle-slots) || OPTS+=(--no-cache-idle-slots)
[[ -n "${CHAT2_CACHE_REUSE:-}" && "${CHAT2_CACHE_REUSE}" != "0" ]] && OPTS+=(--cache-reuse "${CHAT2_CACHE_REUSE}")

CUSTOM_ARGS=()
if [[ -n "${CHAT2_CUSTOM_ARGS_JSON:-}" && "${CHAT2_CUSTOM_ARGS_JSON}" != "[]" ]]; then
    while IFS= read -r -d '' token; do
        CUSTOM_ARGS+=("${token}")
    done < <(
        python3 - <<'PY'
import json
import os
import shlex
import sys

try:
    values = json.loads(os.environ.get("CHAT2_CUSTOM_ARGS_JSON", "[]"))
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
if [[ "${CHAT2_JINJA:-off}" == "on" && "${HAS_CUSTOM_JINJA}" -eq 0 ]]; then
    OPTS+=(--jinja)
fi
if [[ "${CHAT2_PRESERVE_THINKING:-on}" == "on" && "${HAS_TEMPLATE_KWARGS}" -eq 0 ]]; then
    OPTS+=(--chat-template-kwargs '{"preserve_thinking": true}')
fi

if [[ -n "${CHAT2_TEMPLATE_ID:-}" && "${HAS_CUSTOM_CHAT_TEMPLATE}" -eq 0 ]]; then
    if [[ ! "${CHAT2_TEMPLATE_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "[chat-backend2] Invalid CHAT2_TEMPLATE_ID: ${CHAT2_TEMPLATE_ID}" >&2
        exit 1
    fi
    CHAT2_TEMPLATE_FILE="${STACK_DIR}/config/chat-templates/${CHAT2_TEMPLATE_ID}.jinja"
    if [[ ! -f "${CHAT2_TEMPLATE_FILE}" ]]; then
        echo "[chat-backend2] Chat template not found: ${CHAT2_TEMPLATE_FILE}" >&2
        exit 1
    fi
    OPTS+=(--chat-template-file "${CHAT2_TEMPLATE_FILE}")
fi

SPEC_ARGS=()
SPEC_METHOD="${CHAT2_SPEC_METHOD:-off}"
[[ "${SPEC_METHOD}" == "mtp" ]] && SPEC_METHOD="draft-mtp"
COMMON_SPEC_ARGS=(
    --spec-draft-n-max "${CHAT2_SPEC_DRAFT_N_MAX:-6}"
    --spec-draft-n-min "${CHAT2_SPEC_DRAFT_N_MIN:-0}"
    --spec-draft-p-min "${CHAT2_SPEC_DRAFT_P_MIN:-0.75}"
    --spec-draft-p-split "${CHAT2_SPEC_DRAFT_P_SPLIT:-0.10}"
)
DRAFT_CACHE_SPEC_ARGS=(
    --spec-draft-type-k "${CHAT2_SPEC_DRAFT_TYPE_K:-f16}"
    --spec-draft-type-v "${CHAT2_SPEC_DRAFT_TYPE_V:-f16}"
)
NGRAM_MOD_SPEC_ARGS=(
    --spec-ngram-mod-n-match "${CHAT2_SPEC_NGRAM_MOD_N_MATCH:-24}"
    --spec-ngram-mod-n-min "${CHAT2_SPEC_NGRAM_MOD_N_MIN:-48}"
    --spec-ngram-mod-n-max "${CHAT2_SPEC_NGRAM_MOD_N_MAX:-64}"
)
NGRAM_SIMPLE_SPEC_ARGS=(
    --spec-ngram-simple-size-n "${CHAT2_SPEC_NGRAM_SIZE_N:-12}"
    --spec-ngram-simple-size-m "${CHAT2_SPEC_NGRAM_SIZE_M:-48}"
    --spec-ngram-simple-min-hits "${CHAT2_SPEC_NGRAM_MIN_HITS:-1}"
)
NGRAM_MAP_K_SPEC_ARGS=(
    --spec-ngram-map-k-size-n "${CHAT2_SPEC_NGRAM_SIZE_N:-12}"
    --spec-ngram-map-k-size-m "${CHAT2_SPEC_NGRAM_SIZE_M:-48}"
    --spec-ngram-map-k-min-hits "${CHAT2_SPEC_NGRAM_MIN_HITS:-1}"
)
NGRAM_MAP_K4V_SPEC_ARGS=(
    --spec-ngram-map-k4v-size-n "${CHAT2_SPEC_NGRAM_SIZE_N:-12}"
    --spec-ngram-map-k4v-size-m "${CHAT2_SPEC_NGRAM_SIZE_M:-48}"
    --spec-ngram-map-k4v-min-hits "${CHAT2_SPEC_NGRAM_MIN_HITS:-1}"
)
if [[ "${SPEC_METHOD}" == "draft-model" ]]; then
    if [[ -z "${CHAT2_SPEC_DRAFT_MODEL_PATH:-}" ]]; then
        echo "[chat-backend2] Speculative decoding is enabled, but CHAT2_SPEC_DRAFT_MODEL_PATH is empty." >&2
        exit 1
    fi
    if [[ ! -f "${CHAT2_SPEC_DRAFT_MODEL_PATH}" ]]; then
        echo "[chat-backend2] Draft model not found: ${CHAT2_SPEC_DRAFT_MODEL_PATH}" >&2
        exit 1
    fi

    echo "[chat-backend2] Draft model:      ${CHAT2_SPEC_DRAFT_MODEL_PATH}"
    echo "[chat-backend2] Draft GPU layers: ${CHAT2_SPEC_DRAFT_N_GPU_LAYERS:-auto}"
    echo "[chat-backend2] Draft devices:    ${CHAT2_SPEC_DRAFT_DEVICES:-auto}"
    echo "[chat-backend2] Draft n-max/min:  ${CHAT2_SPEC_DRAFT_N_MAX:-6}/${CHAT2_SPEC_DRAFT_N_MIN:-0}"
    echo "[chat-backend2] Draft p-min/split:${CHAT2_SPEC_DRAFT_P_MIN:-0.75}/${CHAT2_SPEC_DRAFT_P_SPLIT:-0.10}"

    SPEC_ARGS+=(
        --spec-draft-model "${CHAT2_SPEC_DRAFT_MODEL_PATH}"
        --spec-draft-ngl "${CHAT2_SPEC_DRAFT_N_GPU_LAYERS:-auto}"
        "${DRAFT_CACHE_SPEC_ARGS[@]}"
        "${COMMON_SPEC_ARGS[@]}"
    )
    [[ -n "${CHAT2_SPEC_DRAFT_DEVICES:-}" ]] && SPEC_ARGS+=(--spec-draft-device "${CHAT2_SPEC_DRAFT_DEVICES}")
elif [[ "${SPEC_METHOD}" != "off" ]]; then
    SPEC_ARGS+=(--spec-type "${SPEC_METHOD}" "${DRAFT_CACHE_SPEC_ARGS[@]}" "${COMMON_SPEC_ARGS[@]}")
    if [[ "${SPEC_METHOD}" == "draft-simple" || "${SPEC_METHOD}" == "draft-eagle3" || "${SPEC_METHOD}" == "draft-dflash" ]]; then
        if [[ -z "${CHAT2_SPEC_DRAFT_MODEL_PATH:-}" ]]; then
            echo "[chat-backend2] ${SPEC_METHOD} is enabled, but CHAT2_SPEC_DRAFT_MODEL_PATH is empty." >&2
            exit 1
        fi
        if [[ ! -f "${CHAT2_SPEC_DRAFT_MODEL_PATH}" ]]; then
            echo "[chat-backend2] Draft model not found: ${CHAT2_SPEC_DRAFT_MODEL_PATH}" >&2
            exit 1
        fi
        SPEC_ARGS+=(--spec-draft-model "${CHAT2_SPEC_DRAFT_MODEL_PATH}" --spec-draft-ngl "${CHAT2_SPEC_DRAFT_N_GPU_LAYERS:-auto}")
        [[ -n "${CHAT2_SPEC_DRAFT_DEVICES:-}" ]] && SPEC_ARGS+=(--spec-draft-device "${CHAT2_SPEC_DRAFT_DEVICES}")
    fi
    if [[ ",${SPEC_METHOD}," == *,ngram-mod,* ]]; then
        echo "[chat-backend2] N-gram mod:       match=${CHAT2_SPEC_NGRAM_MOD_N_MATCH:-24} min=${CHAT2_SPEC_NGRAM_MOD_N_MIN:-48} max=${CHAT2_SPEC_NGRAM_MOD_N_MAX:-64}"
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
    if [[ ",${SPEC_METHOD}," != *,ngram-mod,* && "${CHAT2_SPEC_NGRAM_MOD:-off}" == "on" ]]; then
        echo "[chat-backend2] N-gram mod assist requested, but this llama-server build only accepts ngram-mod as a standalone --spec-type; leaving --spec-type=${SPEC_METHOD}."
    fi
fi

# Only add --mmproj if the path is non-empty and the file exists
[[ -n "${CHAT2_MMPROJ_PATH:-}" && -f "${CHAT2_MMPROJ_PATH}" ]] && OPTS+=(--mmproj "${CHAT2_MMPROJ_PATH}")

exec "${LLAMA_SERVER_BIN}" \
    --model "${CHAT2_MODEL_PATH}" \
    --alias "${CHAT2_MODEL_NAME:-chat-custom}" \
    --host "${CHAT2_BACKEND_HOST}" \
    --port "${CHAT2_BACKEND_PORT}" \
    --ctx-size "${CHAT2_CTX_SIZE}" \
    --main-gpu "${CHAT2_MAIN_GPU}" \
    --n-gpu-layers "${CHAT2_N_GPU_LAYERS}" \
    --split-mode "${CHAT2_SPLIT_MODE}" \
    --tensor-split "${CHAT2_TENSOR_SPLIT}" \
    --batch-size "${CHAT2_BATCH_SIZE}" \
    --ubatch-size "${CHAT2_UBATCH_SIZE}" \
    --parallel "${CHAT2_N_PARALLEL}" \
    --threads "${CHAT2_THREADS:--1}" \
    --threads-batch "${CHAT2_THREADS_BATCH:--1}" \
    --cache-type-k "${CHAT2_CACHE_TYPE_K}" \
    --cache-type-v "${CHAT2_CACHE_TYPE_V}" \
    --cache-ram "${CHAT2_CACHE_RAM:-8192}" \
    --ctx-checkpoints "${CHAT2_CTX_CHECKPOINTS:-32}" \
    --flash-attn "${CHAT2_FLASH_ATTN}" \
    --temp "${CHAT2_TEMP}" \
    --top-p "${CHAT2_TOP_P}" \
    --top-k "${CHAT2_TOP_K}" \
    --min-p "${CHAT2_MIN_P}" \
    --reasoning-format "${CHAT2_REASONING_FORMAT:-deepseek}" \
    --fit "${CHAT2_FIT:-on}" \
    "${OPTS[@]}" \
    "${SPEC_ARGS[@]}" \
    "${CUSTOM_ARGS[@]}" \
    "$@"
# NOTE: No --chat-template-kwargs here. Thinking is controlled per-request
# by the proxy, so this backend can be reused across model families.
