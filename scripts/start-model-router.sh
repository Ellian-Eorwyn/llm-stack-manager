#!/usr/bin/env bash
# =============================================================================
# start-model-router.sh
# Launches llama-server in router mode, owning the auxiliary models instead of
# one systemd unit each. Models load on the first request that names them and
# evict each other once MODEL_ROUTER_MAX are resident, so the tier shares one
# VRAM budget rather than holding a static reservation apiece.
#
# Port: MODEL_ROUTER_PORT (default 8013). Callers keep using EMBED_PORT,
#       RERANK_PORT, TASK_PORT and OCR_PORT; nginx fronts those onto this one
#       and the router picks the model from the request body.
# =============================================================================
set -euo pipefail

STACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# `set -a` so bash expands ${LISTEN_HOST} and friends. systemd's EnvironmentFile
# does not expand ${VAR}, which is why OCR_HOST reaches Python as a literal.
set -a
source "${STACK_DIR}/config/llm-stack.env"
set +a

if [[ "${MODEL_ROUTER_ENABLED:-off}" != "on" ]]; then
    echo "[model-router] Disabled by MODEL_ROUTER_ENABLED=${MODEL_ROUTER_ENABLED:-off}" >&2
    exit 0
fi

PRESET_PATH="${MODEL_ROUTER_PRESET_PATH:-${STACK_DIR}/config/models.ini}"
ROUTER_PORT="${MODEL_ROUTER_PORT:-8013}"
ROUTER_MAX="${MODEL_ROUTER_MAX:-2}"
SLEEP_IDLE="${MODEL_ROUTER_SLEEP_IDLE_SECONDS:-600}"
# Loopback, not LISTEN_HOST. nginx owns the public per-model ports and proxies
# here, so binding this to the LAN would only add a second, unauthenticated way
# in on a port nothing is expected to use.
ROUTER_HOST="${MODEL_ROUTER_HOST:-127.0.0.1}"

LLAMA_SERVER_DIR="${LLAMA_SERVER_BIN%/*}"
export LD_LIBRARY_PATH="${LLAMA_SERVER_DIR}:${LD_LIBRARY_PATH:-}"
export DYLD_LIBRARY_PATH="${LLAMA_SERVER_DIR}:${DYLD_LIBRARY_PATH:-}"

# Every child inherits this, so per-model placement uses real device indices in
# each preset's main-gpu / tensor-split rather than per-service renumbering.
export CUDA_VISIBLE_DEVICES="${MODEL_ROUTER_GPU_VISIBLE_DEVICES:-0,1}"

# The router treats the llama.cpp cache as a model source in addition to the
# preset, so anything ever pulled with -hf becomes routable and shows up on
# /v1/models. Point it at an empty directory of our own: the preset is meant to
# be the only thing this router will serve.
export LLAMA_CACHE="${MODEL_ROUTER_CACHE_DIR:-${STACK_DIR}/config/model-router-cache}"
mkdir -p "${LLAMA_CACHE}"

echo "[model-router] Rendering preset: ${PRESET_PATH}"
python3 "${STACK_DIR}/scripts/render-models-ini.py" "${PRESET_PATH}"

echo "[model-router] Port:          ${ROUTER_PORT}"
echo "[model-router] Host:          ${ROUTER_HOST}"
echo "[model-router] Models max:    ${ROUTER_MAX}"
echo "[model-router] Members:       ${MODEL_ROUTER_MEMBERS:-EMBED,OCR,RERANK,TASK}"
echo "[model-router] GPUs:          ${CUDA_VISIBLE_DEVICES}"
echo "[model-router] Idle unload:   ${SLEEP_IDLE}s"

OPTS=()
# -1 disables idle unloading; passing it through is harmless but noise.
if [[ "${SLEEP_IDLE}" != "-1" && -n "${SLEEP_IDLE}" ]]; then
    OPTS+=(--sleep-idle-seconds "${SLEEP_IDLE}")
fi

exec "${LLAMA_SERVER_BIN}" \
    --host "${ROUTER_HOST}" \
    --port "${ROUTER_PORT}" \
    --models-preset "${PRESET_PATH}" \
    --models-max "${ROUTER_MAX}" \
    --models-autoload \
    "${OPTS[@]}" \
    "$@"
