#!/usr/bin/env bash
# =============================================================================
# start-backend.sh <slot>
#
# One launcher for every backend slot, replacing a script per slot.
#
# The ten it replaces were between 66 and 325 lines each and all the same
# shape: source the env file, resolve a chain of fallbacks, assemble an
# `OPTS=()` array, exec. Between them they duplicated the same twenty-odd flags
# with the same defaults under different prefixes, which is why a fix to one --
# an empty `--tensor-split`, a dead `--fit-ctx` -- had to be made nine more
# times to land everywhere.
#
# What each slot *is* now lives in web/backends/slots.py as data, and an engine
# module turns that into a command line. This script keeps the two jobs that
# genuinely belong in the shell:
#
#   1. Sourcing the env file, which is shell syntax and may contain ${VAR}
#      references that only `source` expands.
#   2. Placement vetting. resolve_split_opts refuses modes that would fail
#      after exec -- as a core dump in a restart loop rather than an error --
#      and vets `tensor` against the model's architecture by reading the GGUF.
#      That has to run before the command is built, so its result is handed to
#      the builder rather than recomputed.
#
# The preflight budget report runs here too, for the same reason it always did:
# so the journal says what a configuration was expected to cost, immediately
# before llama-server says what it actually allocated.
# =============================================================================
set -euo pipefail

SLOT="${1:-}"
if [[ -z "${SLOT}" ]]; then
    echo "usage: start-backend.sh <slot>" >&2
    exit 2
fi
shift || true

STACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set -a
source "${STACK_DIR}/config/llm-stack.env"
set +a
source "${STACK_DIR}/scripts/lib/backend-preflight.sh"

PREFIX="[${SLOT}]"
PYTHON="${LLM_STACK_PYTHON:-python3}"

# Ask the registry for this slot's identity. Doing it here keeps the prefix and
# the model path in one place rather than restating them per slot in shell.
slot_field() {
    STACK_DIR="${STACK_DIR}" "${PYTHON}" - "$1" <<'PY' 2>/dev/null || true
import os, sys
sys.path.insert(0, os.path.join(os.environ["STACK_DIR"], "web"))
from backends.slots import SLOTS
slot = SLOTS.get(os.environ.get("LLM_BACKEND_SLOT", ""))
if slot:
    print(getattr(slot, sys.argv[1], "") or "")
PY
}
export LLM_BACKEND_SLOT="${SLOT}"

SLOT_PREFIX="$(slot_field prefix)"
if [[ -z "${SLOT_PREFIX}" ]]; then
    echo "${PREFIX} unknown slot; see web/backends/slots.py" >&2
    exit 2
fi

# The engine decides whether placement vetting applies at all: it is a
# llama.cpp concept, and the MLX servers take neither a split mode nor a device.
# Mirrors Slot.engine(): the key is {PREFIX}_ENGINE unless a slot names its
# own. Derived rather than asked for, so the shell and the registry cannot
# disagree about which setting selects the engine.
ENGINE_KEY="$(slot_field engine_key)"
[[ -z "${ENGINE_KEY}" ]] && ENGINE_KEY="${SLOT_PREFIX}_ENGINE"
ENGINE="$(eval "printf '%s' \"\${${ENGINE_KEY}:-llamacpp}\"")"
[[ -z "${ENGINE}" ]] && ENGINE="llamacpp"

if [[ "${ENGINE}" == "llamacpp" ]]; then
    LLAMA_SERVER_DIR="${LLAMA_SERVER_BIN%/*}"
    export LD_LIBRARY_PATH="${LLAMA_SERVER_DIR}:${LD_LIBRARY_PATH:-}"
    export DYLD_LIBRARY_PATH="${LLAMA_SERVER_DIR}:${DYLD_LIBRARY_PATH:-}"

    # Inert on a Metal build, and left in place for the CUDA hosts that need it.
    visible="$(eval "printf '%s' \"\${${SLOT_PREFIX}_GPU_VISIBLE_DEVICES:-}\"")"
    [[ -n "${visible}" ]] && export CUDA_VISIBLE_DEVICES="${visible}"

    model_path="$(STACK_DIR="${STACK_DIR}" "${PYTHON}" - <<'PY'
import os, sys
sys.path.insert(0, os.path.join(os.environ["STACK_DIR"], "web"))
from backends.llamacpp import _first
from backends.slots import SLOTS
slot = SLOTS[os.environ["LLM_BACKEND_SLOT"]]
print(_first(os.environ, slot.model_keys, slot.prefix))
PY
)"
    split_mode="$(eval "printf '%s' \"\${${SLOT_PREFIX}_SPLIT_MODE:-layer}\"")"
    tensor_split="$(eval "printf '%s' \"\${${SLOT_PREFIX}_TENSOR_SPLIT:-}\"")"
    main_gpu="$(eval "printf '%s' \"\${${SLOT_PREFIX}_MAIN_GPU:-}\"")"
    flash_attn="$(eval "printf '%s' \"\${${SLOT_PREFIX}_FLASH_ATTN:-on}\"")"

    if [[ "${tensor_split}" == "auto" || -z "${tensor_split}" ]]; then
        tensor_split="$(auto_tensor_split "${tensor_split}" "${CUDA_VISIBLE_DEVICES:-}")"
    fi

    resolve_split_opts "${PREFIX}" "${split_mode}" "${model_path}" \
        "${tensor_split}" "${main_gpu}" "${flash_attn}"
    export LLM_BACKEND_PLACEMENT_JSON="$(
        printf '%s\n' ${SPLIT_OPTS[@]+"${SPLIT_OPTS[@]}"} \
        | "${PYTHON}" -c 'import json,sys; print(json.dumps([l.rstrip("\n") for l in sys.stdin if l.strip()]))'
    )"

    # The values come from what this launcher resolved, not from a second
    # reading of the env file: re-deriving settings independently is precisely
    # how --fit-ctx stayed live after it had been cleared in the UI.
    mmproj_path="$(eval "printf '%s' \"\${${SLOT_PREFIX}_MMPROJ_PATH:-}\"")"
    preflight_report "${PREFIX}" "${SLOT}" "${model_path}" "${mmproj_path}" \
        ctx_size="$(eval "printf '%s' \"\${${SLOT_PREFIX}_CTX_SIZE:-}\"")" \
        parallel="$(eval "printf '%s' \"\${${SLOT_PREFIX}_N_PARALLEL:-1}\"")" \
        cache_type_k="$(eval "printf '%s' \"\${${SLOT_PREFIX}_CACHE_TYPE_K:-}\"")" \
        cache_type_v="$(eval "printf '%s' \"\${${SLOT_PREFIX}_CACHE_TYPE_V:-}\"")" \
        tensor_split="${tensor_split}" || true
fi

echo "${PREFIX} engine: ${ENGINE}"

# Build the command, then exec it. A failure here is fatal on purpose: unlike
# the advisory helpers, which must never stop a backend from starting, a
# command that cannot be assembled is a backend that cannot run correctly.
# NUL-separated, because an argument may legitimately contain whitespace -- a
# model path, a chat-template string, a custom argument -- and command
# substitution would split it.
BUILD_PY="${STACK_DIR}/scripts/lib/build-backend-command.py"
ARGV=()
while IFS= read -r -d '' token; do
    ARGV+=("${token}")
done < <(STACK_DIR="${STACK_DIR}" "${PYTHON}" "${BUILD_PY}" "$@")

if [[ ${#ARGV[@]} -eq 0 ]]; then
    echo "${PREFIX} could not build the command line" >&2
    exit 1
fi

exec "${ARGV[@]}"
