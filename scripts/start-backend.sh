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

# Ask the registry for this slot's identity, its placement settings and the
# settings its memory-fit report carries. One reader for all of them: the
# expansions this replaces read a single prefix, and the primary chat slot has
# two -- so they would have resolved placement and the report differently from
# the command they describe.
export LLM_BACKEND_SLOT="${SLOT}"
eval "$(STACK_DIR="${STACK_DIR}" "${PYTHON}" "${STACK_DIR}/scripts/lib/slot-facts.py" \
        "${SLOT}" 2>/dev/null)" || true
if [[ -z "${FACT_PREFIX:-}" ]]; then
    echo "${PREFIX} unknown slot; see web/backends/slots.py" >&2
    exit 2
fi
ENGINE="${FACT_ENGINE}"

if [[ "${ENGINE}" == "llamacpp" ]]; then
    LLAMA_SERVER_DIR="${LLAMA_SERVER_BIN%/*}"
    export LD_LIBRARY_PATH="${LLAMA_SERVER_DIR}:${LD_LIBRARY_PATH:-}"
    export DYLD_LIBRARY_PATH="${LLAMA_SERVER_DIR}:${DYLD_LIBRARY_PATH:-}"

    # Inert on a Metal build, and left in place for the CUDA hosts that need it.
    #
    # Renumbering is what makes MAIN_GPU mean two different cards depending on
    # who started the model: a slot with GPU_VISIBLE_DEVICES=1 and MAIN_GPU=0
    # runs on physical GPU 1, while the same MAIN_GPU=0 under llama-router --
    # which sets its own visible list -- runs on physical GPU 0.
    #
    # LLM_ABSOLUTE_GPU_INDICES=on stops it, and is off by default because the
    # stored values were written in renumbered space: switching the meaning
    # moves models between cards unless the values move with it.
    # `scripts/lib/gpu-indices.py` reports what would move and prints the
    # absolute values that keep everything where it is.
    if [[ "${LLM_ABSOLUTE_GPU_INDICES:-off}" != "on" ]]; then
        [[ -n "${FACT_VISIBLE}" ]] && export CUDA_VISIBLE_DEVICES="${FACT_VISIBLE}"
    fi

    tensor_split="${FACT_TENSOR_SPLIT}"
    if [[ "${tensor_split}" == "auto" || -z "${tensor_split}" ]]; then
        tensor_split="$(auto_tensor_split "${tensor_split}" "${CUDA_VISIBLE_DEVICES:-}")"
    fi

    resolve_split_opts "${PREFIX}" "${FACT_SPLIT_MODE}" "${FACT_MODEL}" \
        "${tensor_split}" "${FACT_MAIN_GPU}" "${FACT_FLASH_ATTN}"
    export LLM_BACKEND_PLACEMENT_JSON="$(
        printf '%s\n' ${SPLIT_OPTS[@]+"${SPLIT_OPTS[@]}"} \
        | "${PYTHON}" -c 'import json,sys; print(json.dumps([l.rstrip("\n") for l in sys.stdin if l.strip()]))'
    )"

    # Whether --swa-full would do anything is a fact about the GGUF, which only
    # budget.py can read, so it is answered here and handed to the builder the
    # same way placement is. Unknown counts as supported: a helper must never
    # stop a backend from starting.
    export LLM_BACKEND_SWA_FULL=off
    if [[ "${FACT_SWA_FULL}" == "on" ]] && model_supports_swa "${FACT_MODEL}"; then
        export LLM_BACKEND_SWA_FULL=on
    fi

    # The report describes the process about to start, so its settings come
    # from what was resolved here rather than from a second reading of the env
    # file -- re-deriving them independently is precisely how --fit-ctx stayed
    # live after it had been cleared in the UI.
    devices="$(awk -F, '{print NF}' <<< "${CUDA_VISIBLE_DEVICES:-}")"
    eval "$(STACK_DIR="${STACK_DIR}" "${PYTHON}" "${STACK_DIR}/scripts/lib/slot-facts.py" \
            "${SLOT}" --tensor-split "${tensor_split}" --devices "${devices}" 2>/dev/null)" || true
    preflight_report "${PREFIX}" "${FACT_BUDGET_NAME}" "${FACT_MODEL}" "${FACT_MMPROJ}" \
        ${FACT_PREFLIGHT[@]+"${FACT_PREFLIGHT[@]}"} || true
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
