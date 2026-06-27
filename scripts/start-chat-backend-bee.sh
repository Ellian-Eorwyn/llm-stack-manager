#!/usr/bin/env bash
# =============================================================================
# start-chat-backend-bee.sh
# Shared chat backend using Anbeeld/beellama.cpp. It binds to the same internal
# backend port as the llama.cpp chat backends and is exposed through chat-proxy.
# =============================================================================
set -euo pipefail

STACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${STACK_DIR}/config/llm-stack.env"

BEE_SERVER_BIN="${BEELLAMA_SERVER_BIN:-${STACK_DIR}/deps/beellama.cpp/build/bin/llama-server}"
BEE_SERVER_DIR="${BEE_SERVER_BIN%/*}"
if [[ "${BEE_SERVER_BIN}" == *.gguf ]]; then
    echo "[chat-backend-bee] BEELLAMA_SERVER_BIN points to a GGUF model, not the BeeLLaMA llama-server binary: ${BEE_SERVER_BIN}" >&2
    echo "[chat-backend-bee] Set BEELLAMA_SERVER_BIN=${STACK_DIR}/deps/beellama.cpp/build/bin/llama-server and CHAT_BEE_MODEL_PATH to the model GGUF." >&2
    exit 126
fi
if [[ ! -x "${BEE_SERVER_BIN}" ]]; then
    echo "[chat-backend-bee] BeeLLaMA binary is not executable or does not exist: ${BEE_SERVER_BIN}" >&2
    exit 126
fi
export LD_LIBRARY_PATH="${BEE_SERVER_DIR}:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${CHAT_GPU_VISIBLE_DEVICES}"

log_bool_env() {
    local env_name="$1"
    local value="$2"
    case "${value}" in
        on|true|1|yes) export "${env_name}=1" ;;
        off|false|0|no|"") ;;
        *) export "${env_name}=${value}" ;;
    esac
}

log_bool_env GGML_DFLASH_PROFILE "${CHAT_BEE_DFLASH_PROFILE:-}"
log_bool_env GGML_DFLASH_PROFILE_SYNC_SPLIT "${CHAT_BEE_DFLASH_PROFILE_SYNC_SPLIT:-off}"
log_bool_env GGML_DFLASH_DEBUG "${CHAT_BEE_DFLASH_DEBUG:-off}"
log_bool_env GGML_DFLASH_CRASH_TRACE "${CHAT_BEE_DFLASH_CRASH_TRACE:-off}"
log_bool_env GGML_DFLASH_INPUT_DEBUG "${CHAT_BEE_DFLASH_INPUT_DEBUG:-off}"
log_bool_env GGML_DFLASH_VERBOSE_CONTRACT "${CHAT_BEE_DFLASH_VERBOSE_CONTRACT:-off}"
log_bool_env GGML_DFLASH_FORCE_CPU_CROSS "${CHAT_BEE_DFLASH_FORCE_CPU_CROSS:-off}"
log_bool_env GGML_DFLASH_VERIFY_PAD "${CHAT_BEE_DFLASH_VERIFY_PAD:-off}"
[[ "${CHAT_BEE_DFLASH_SHARED_DRAFT_BATCH:-on}" == "off" ]] && export GGML_DFLASH_SHARED_DRAFT_BATCH=0
[[ "${CHAT_BEE_DFLASH_GPU_RING:-on}" == "off" ]] && export GGML_DFLASH_GPU_RING=0
[[ "${CHAT_BEE_DFLASH_MULTI_GPU_TAPE:-on}" == "off" ]] && export GGML_DFLASH_MULTI_GPU_TAPE=0
[[ -n "${CHAT_BEE_DFLASH_MAX_CTX:-}" ]] && export GGML_DFLASH_MAX_CTX="${CHAT_BEE_DFLASH_MAX_CTX}"
[[ -n "${CHAT_BEE_DFLASH_KV_CACHE_MODE:-}" ]] && export GGML_DFLASH_KV_CACHE_MODE="${CHAT_BEE_DFLASH_KV_CACHE_MODE}"

echo "[chat-backend-bee] Starting BeeLLaMA shared backend (${CHAT_BEE_LABEL:-Backend BeeLLaMA})"
echo "[chat-backend-bee] Binary:           ${BEE_SERVER_BIN}"
echo "[chat-backend-bee] Model:            ${CHAT_BEE_MODEL_PATH}"
echo "[chat-backend-bee] MMProj:           ${CHAT_BEE_MMPROJ_PATH:-(none)}"
echo "[chat-backend-bee] Port:             ${CHAT_BACKEND_PORT} (localhost only)"
echo "[chat-backend-bee] Context:          ${CHAT_BEE_CTX_SIZE}"
echo "[chat-backend-bee] Main GPU:         ${CHAT_MAIN_GPU}"
echo "[chat-backend-bee] GPUs:             ${CUDA_VISIBLE_DEVICES} (split ${CHAT_TENSOR_SPLIT})"
echo "[chat-backend-bee] KV cache:         K=${CHAT_BEE_CACHE_TYPE_K:-${CHAT_CACHE_TYPE_K}} V=${CHAT_BEE_CACHE_TYPE_V:-${CHAT_CACHE_TYPE_V}}"
echo "[chat-backend-bee] Speculative:      ${CHAT_BEE_SPEC_METHOD:-off}"
echo "[chat-backend-bee] Reasoning guard:  ${CHAT_BEE_REASONING_LOOP_GUARD:-force-close}"

OPTS=()
[[ "${CHAT_LOG_PREFIX:-true}" == "true" ]] && OPTS+=(--log-prefix)
[[ "${CHAT_BEE_LOG_TIMESTAMPS:-true}" == "true" ]] && OPTS+=(--log-timestamps)
[[ -n "${CHAT_BEE_LOG_COLORS:-}" ]] && OPTS+=(--log-colors "${CHAT_BEE_LOG_COLORS}")
[[ -n "${CHAT_BEE_VERBOSITY:-}" ]] && OPTS+=(--verbosity "${CHAT_BEE_VERBOSITY}")
[[ "${CHAT_NO_MMAP:-false}" == "true" ]] && OPTS+=(--no-mmap)
[[ "${CHAT_MLOCK:-false}" == "true" ]] && OPTS+=(--mlock)
[[ -n "${CHAT_DEVICE:-}" ]] && OPTS+=(--device "${CHAT_DEVICE}")
[[ "${CHAT_KV_OFFLOAD:-on}" == "on" ]] && OPTS+=(--kv-offload) || OPTS+=(--no-kv-offload)
[[ "${CHAT_OP_OFFLOAD:-on}" == "on" ]] && OPTS+=(--op-offload) || OPTS+=(--no-op-offload)
[[ "${CHAT_MMPROJ_OFFLOAD:-on}" == "on" ]] && OPTS+=(--mmproj-offload) || OPTS+=(--no-mmproj-offload)
[[ "${CHAT_BEE_KV_UNIFIED:-on}" == "on" ]] && OPTS+=(--kv-unified)
[[ "${CHAT_BEE_NO_HOST:-off}" == "on" ]] && OPTS+=(--no-host)
[[ -n "${CHAT_BEE_MMPROJ_PATH:-}" && -f "${CHAT_BEE_MMPROJ_PATH}" ]] && OPTS+=(--mmproj "${CHAT_BEE_MMPROJ_PATH}")
[[ -n "${CHAT_BEE_REASONING:-}" ]] && OPTS+=(--reasoning "${CHAT_BEE_REASONING}")
[[ -n "${CHAT_BEE_REASONING_BUDGET:-}" ]] && OPTS+=(--reasoning-budget "${CHAT_BEE_REASONING_BUDGET}")
[[ -n "${CHAT_BEE_REASONING_LOOP_GUARD:-}" ]] && OPTS+=(--reasoning-loop-guard "${CHAT_BEE_REASONING_LOOP_GUARD}")
[[ -n "${CHAT_BEE_REASONING_LOOP_MIN_TOKENS:-}" ]] && OPTS+=(--reasoning-loop-min-tokens "${CHAT_BEE_REASONING_LOOP_MIN_TOKENS}")
[[ -n "${CHAT_BEE_REASONING_LOOP_WINDOW:-}" ]] && OPTS+=(--reasoning-loop-window "${CHAT_BEE_REASONING_LOOP_WINDOW}")
[[ -n "${CHAT_BEE_REASONING_LOOP_MAX_PERIOD:-}" ]] && OPTS+=(--reasoning-loop-max-period "${CHAT_BEE_REASONING_LOOP_MAX_PERIOD}")
[[ -n "${CHAT_BEE_REASONING_LOOP_MIN_COVERAGE:-}" ]] && OPTS+=(--reasoning-loop-min-coverage "${CHAT_BEE_REASONING_LOOP_MIN_COVERAGE}")
[[ -n "${CHAT_BEE_REASONING_LOOP_CHECK_INTERVAL:-}" ]] && OPTS+=(--reasoning-loop-check-interval "${CHAT_BEE_REASONING_LOOP_CHECK_INTERVAL}")
[[ -n "${CHAT_BEE_REASONING_LOOP_INTERVENTIONS:-}" ]] && OPTS+=(--reasoning-loop-interventions "${CHAT_BEE_REASONING_LOOP_INTERVENTIONS}")

CUSTOM_ARGS=()
if [[ -n "${CHAT_BEE_CUSTOM_ARGS_JSON:-}" && "${CHAT_BEE_CUSTOM_ARGS_JSON}" != "[]" ]]; then
    while IFS= read -r -d '' token; do
        CUSTOM_ARGS+=("${token}")
    done < <(
        CHAT_BEE_CUSTOM_ARGS_JSON="${CHAT_BEE_CUSTOM_ARGS_JSON}" python3 - <<'PYCUSTOM'
import json
import os
import shlex
import sys
try:
    values = json.loads(os.environ.get("CHAT_BEE_CUSTOM_ARGS_JSON", "[]"))
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

HAS_CUSTOM_JINJA=0
HAS_TEMPLATE_KWARGS=0
HAS_CUSTOM_CHAT_TEMPLATE=0
for token in "${CUSTOM_ARGS[@]:-}"; do
    [[ "${token}" == "--jinja" ]] && HAS_CUSTOM_JINJA=1
    [[ "${token}" == "--chat-template-kwargs" ]] && HAS_TEMPLATE_KWARGS=1
    [[ "${token}" == "--chat-template" || "${token}" == "--chat-template-file" ]] && HAS_CUSTOM_CHAT_TEMPLATE=1
done
[[ "${CHAT_JINJA:-off}" == "on" && "${HAS_CUSTOM_JINJA}" -eq 0 ]] && OPTS+=(--jinja)
[[ "${CHAT_PRESERVE_THINKING:-on}" == "on" && "${HAS_TEMPLATE_KWARGS}" -eq 0 ]] && OPTS+=(--chat-template-kwargs '{"preserve_thinking": true}')
if [[ -n "${CHAT_TEMPLATE_ID:-}" && "${HAS_CUSTOM_CHAT_TEMPLATE}" -eq 0 ]]; then
    if [[ ! "${CHAT_TEMPLATE_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "[chat-backend-bee] Invalid CHAT_TEMPLATE_ID: ${CHAT_TEMPLATE_ID}" >&2
        exit 1
    fi
    CHAT_TEMPLATE_FILE="${STACK_DIR}/config/chat-templates/${CHAT_TEMPLATE_ID}.jinja"
    if [[ ! -f "${CHAT_TEMPLATE_FILE}" ]]; then
        echo "[chat-backend-bee] Chat template not found: ${CHAT_TEMPLATE_FILE}" >&2
        exit 1
    fi
    OPTS+=(--chat-template-file "${CHAT_TEMPLATE_FILE}")
fi

SPEC_ARGS=()
SPEC_METHOD="${CHAT_BEE_SPEC_METHOD:-off}"
[[ "${SPEC_METHOD}" == "mtp" ]] && SPEC_METHOD="draft-mtp"
if [[ "${SPEC_METHOD}" != "off" && "${SPEC_METHOD}" != "none" ]]; then
    SPEC_ARGS+=(--spec-type "${SPEC_METHOD}")
    if [[ -n "${CHAT_BEE_SPEC_DRAFT_MODEL_PATH:-}" ]]; then
        if [[ ! -f "${CHAT_BEE_SPEC_DRAFT_MODEL_PATH}" ]]; then
            echo "[chat-backend-bee] Draft model not found: ${CHAT_BEE_SPEC_DRAFT_MODEL_PATH}" >&2
            exit 1
        fi
        echo "[chat-backend-bee] Draft model:      ${CHAT_BEE_SPEC_DRAFT_MODEL_PATH}"
        SPEC_ARGS+=(
            --spec-draft-model "${CHAT_BEE_SPEC_DRAFT_MODEL_PATH}"
            --spec-draft-ngl "${CHAT_BEE_SPEC_DRAFT_N_GPU_LAYERS:-all}"
            --spec-draft-type-k "${CHAT_BEE_SPEC_DRAFT_TYPE_K:-f16}"
            --spec-draft-type-v "${CHAT_BEE_SPEC_DRAFT_TYPE_V:-f16}"
        )
        [[ -n "${CHAT_BEE_SPEC_DRAFT_DEVICES:-}" ]] && SPEC_ARGS+=(--spec-draft-device "${CHAT_BEE_SPEC_DRAFT_DEVICES}")
        [[ -n "${CHAT_BEE_SPEC_DRAFT_CTX_SIZE:-}" ]] && SPEC_ARGS+=(--spec-draft-ctx-size "${CHAT_BEE_SPEC_DRAFT_CTX_SIZE}")
    elif [[ "${SPEC_METHOD}" == "dflash" ]]; then
        echo "[chat-backend-bee] DFlash is enabled, but CHAT_BEE_SPEC_DRAFT_MODEL_PATH is empty." >&2
        exit 1
    fi
    SPEC_ARGS+=(
        --spec-draft-n-max "${CHAT_BEE_SPEC_DRAFT_N_MAX:-16}"
        --spec-draft-n-min "${CHAT_BEE_SPEC_DRAFT_N_MIN:-0}"
        --spec-draft-p-min "${CHAT_BEE_SPEC_DRAFT_P_MIN:-0.0}"
        --spec-draft-p-split "${CHAT_BEE_SPEC_DRAFT_P_SPLIT:-0.10}"
    )
fi
if [[ "${SPEC_METHOD}" == "dflash" ]]; then
    SPEC_ARGS+=(
        --spec-branch-budget "${CHAT_BEE_SPEC_BRANCH_BUDGET:-0}"
        --spec-draft-top-k "${CHAT_BEE_SPEC_DRAFT_TOP_K:-1}"
        --spec-draft-temp "${CHAT_BEE_SPEC_DRAFT_TEMP:-0.0}"
        --spec-dflash-cross-ctx "${CHAT_BEE_SPEC_DFLASH_CROSS_CTX:-512}"
    )
    [[ "${CHAT_BEE_SPEC_DFLASH_MAX_SLOTS:-0}" != "0" ]] && SPEC_ARGS+=(--spec-dflash-max-slots "${CHAT_BEE_SPEC_DFLASH_MAX_SLOTS}")
    [[ "${CHAT_BEE_SPEC_DM_ADAPTIVE:-on}" == "on" ]] && SPEC_ARGS+=(--spec-dm-adaptive) || SPEC_ARGS+=(--no-spec-dm-adaptive)
    SPEC_ARGS+=(
        --spec-dm-controller "${CHAT_BEE_SPEC_DM_CONTROLLER:-profit}"
        --spec-dm-probe-interval "${CHAT_BEE_SPEC_DM_PROBE_INTERVAL:-16}"
        --spec-dm-probe-fraction "${CHAT_BEE_SPEC_DM_PROBE_FRACTION:-0.25}"
        --spec-dm-explore-interval "${CHAT_BEE_SPEC_DM_EXPLORE_INTERVAL:-12}"
        --spec-dm-off-dwell "${CHAT_BEE_SPEC_DM_OFF_DWELL:-8}"
        --spec-dm-fringe-min "${CHAT_BEE_SPEC_DM_FRINGE_MIN:-0.30}"
        --spec-dm-fringe-max "${CHAT_BEE_SPEC_DM_FRINGE_MAX:-0.50}"
        --spec-dm-min-reach "${CHAT_BEE_SPEC_DM_MIN_REACH:-3}"
        --spec-dm-profit-min "${CHAT_BEE_SPEC_DM_PROFIT_MIN:-0.05}"
        --spec-dm-profit-raise-margin "${CHAT_BEE_SPEC_DM_PROFIT_RAISE_MARGIN:-0.05}"
        --spec-dm-profit-lower-margin "${CHAT_BEE_SPEC_DM_PROFIT_LOWER_MARGIN:-0.05}"
        --spec-dm-profit-ewma-alpha "${CHAT_BEE_SPEC_DM_PROFIT_EWMA_ALPHA:-0.15}"
        --spec-dm-profit-min-samples "${CHAT_BEE_SPEC_DM_PROFIT_MIN_SAMPLES:-3}"
        --spec-dm-profit-warmup "${CHAT_BEE_SPEC_DM_PROFIT_WARMUP:-0}"
        --spec-dm-profit-baseline-interval "${CHAT_BEE_SPEC_DM_PROFIT_BASELINE_INTERVAL:-1024}"
    )
fi

exec "${BEE_SERVER_BIN}" \
    --model "${CHAT_BEE_MODEL_PATH}" \
    --alias "${CHAT_BEE_MODEL_NAME:-chat-bee}" \
    --host "${CHAT_BACKEND_HOST}" \
    --port "${CHAT_BACKEND_PORT}" \
    --ctx-size "${CHAT_BEE_CTX_SIZE}" \
    --main-gpu "${CHAT_MAIN_GPU}" \
    --n-gpu-layers "${CHAT_N_GPU_LAYERS}" \
    --split-mode "${CHAT_SPLIT_MODE}" \
    --tensor-split "${CHAT_TENSOR_SPLIT}" \
    --batch-size "${CHAT_BATCH_SIZE}" \
    --ubatch-size "${CHAT_UBATCH_SIZE}" \
    --parallel "${CHAT_N_PARALLEL}" \
    --threads "${CHAT_THREADS:--1}" \
    --threads-batch "${CHAT_THREADS_BATCH:--1}" \
    --cache-type-k "${CHAT_BEE_CACHE_TYPE_K:-${CHAT_CACHE_TYPE_K}}" \
    --cache-type-v "${CHAT_BEE_CACHE_TYPE_V:-${CHAT_CACHE_TYPE_V}}" \
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
