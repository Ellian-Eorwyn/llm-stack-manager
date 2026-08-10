#!/usr/bin/env bash
# =============================================================================
# install-transcribe.sh
# Builds the transcription sidecar's venv.
#
# Engines are opt-in because they are not the same size. faster-whisper is
# CTranslate2 and costs a few hundred MB; nemo drags in torch and the whole NeMo
# toolkit for several GB and is version-fragile. Installing everything by
# default would make a stack that only ever wants Whisper pay for Parakeet.
#
#   bash scripts/install-transcribe.sh                       # faster-whisper
#   bash scripts/install-transcribe.sh --engines nemo         # + Parakeet/Canary
#   bash scripts/install-transcribe.sh --engines nemo,hf      # + transformers
# =============================================================================
set -euo pipefail

STACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${STACK_DIR}/config/llm-stack.env" ]]; then
    set -a
    source "${STACK_DIR}/config/llm-stack.env"
    set +a
fi

VENV_DIR="${TRANSCRIPT_VENV_DIR:-${STACK_DIR}/deps/transcribe-venv}"
ENGINES="${TRANSCRIPT_ENGINES:-faster-whisper}"
FASTER_WHISPER_VERSION="${TRANSCRIPT_FASTER_WHISPER_VERSION:-1.1.1}"
NEMO_VERSION="${TRANSCRIPT_NEMO_VERSION:-}"
# Blank means PyPI, whose torch wheels bundle their own CUDA runtime and work
# against any recent driver. Hardcoding a /whl/cuXXX index instead pins a CUDA
# version that goes stale silently: it keeps resolving, just to older and older
# torch. Set this only to force a specific build.
TORCH_INDEX="${TRANSCRIPT_TORCH_INDEX_URL:-}"

RECREATE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --engines) ENGINES="$2"; shift 2 ;;
        --engines=*) ENGINES="${1#*=}"; shift ;;
        --venv) VENV_DIR="$2"; shift 2 ;;
        --recreate) RECREATE=1; shift ;;
        -h|--help) sed -n '2,14p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "[transcribe] Unknown argument: $1" >&2; exit 2 ;;
    esac
done

has_engine() {
    [[ ",${ENGINES}," == *",$1,"* ]]
}

# Which engines need torch, and therefore an interpreter torch has wheels for.
needs_torch() {
    has_engine nemo || has_engine hf
}

# torch publishes wheels for 3.9-3.13. faster-whisper is CTranslate2 and tracks
# new Pythons quickly, so a host whose `python3` is newer than torch supports —
# 3.14 here — can run Whisper happily and then fail to install NeMo with nothing
# more helpful than "No matching distribution found for torch". Pick an
# interpreter that can actually carry the requested engines instead.
TORCH_MAX_MINOR=13

python_minor() {
    "$1" -c 'import sys; print(sys.version_info[1])' 2>/dev/null || echo 99
}

pick_python() {
    if [[ -n "${TRANSCRIPT_PYTHON:-}" ]]; then
        printf '%s' "${TRANSCRIPT_PYTHON}"
        return
    fi
    if ! needs_torch; then
        printf '%s' "python3"
        return
    fi
    local candidate
    for candidate in python3.12 python3.11 python3.13 python3.10 python3; do
        local resolved
        resolved="$(command -v "${candidate}" 2>/dev/null || true)"
        [[ -n "${resolved}" ]] || continue
        if [[ "$(python_minor "${resolved}")" -le "${TORCH_MAX_MINOR}" ]]; then
            printf '%s' "${resolved}"
            return
        fi
    done
    printf '%s' "python3"
}

PYTHON="$(pick_python)"

echo "[transcribe] venv:    ${VENV_DIR}"
echo "[transcribe] engines: ${ENGINES}"
echo "[transcribe] python:  ${PYTHON} ($(${PYTHON} -V 2>&1))"

if needs_torch && [[ "$(python_minor "${PYTHON}")" -gt "${TORCH_MAX_MINOR}" ]]; then
    echo "[transcribe] No interpreter on this host is new enough to run the stack but old" >&2
    echo "[transcribe] enough for torch (needs 3.${TORCH_MAX_MINOR} or lower; found $(${PYTHON} -V 2>&1))." >&2
    echo "[transcribe] Install python3.12, or set TRANSCRIPT_PYTHON=/path/to/python3.12." >&2
    exit 1
fi

# An existing venv built on an interpreter torch cannot use has to be rebuilt,
# not added to. Say so rather than letting pip fail three minutes in.
if [[ -x "${VENV_DIR}/bin/python" ]] && needs_torch; then
    EXISTING_MINOR="$(python_minor "${VENV_DIR}/bin/python")"
    if [[ "${EXISTING_MINOR}" -gt "${TORCH_MAX_MINOR}" && "${RECREATE}" -ne 1 ]]; then
        echo "[transcribe] The existing venv is on Python 3.${EXISTING_MINOR}, which torch has no" >&2
        echo "[transcribe] wheels for, so ${ENGINES} cannot be added to it." >&2
        echo "[transcribe] Re-run with --recreate to rebuild it on ${PYTHON}:" >&2
        echo "[transcribe]   bash scripts/install-transcribe.sh --engines ${ENGINES} --recreate" >&2
        exit 1
    fi
fi

if [[ "${RECREATE}" -eq 1 && -d "${VENV_DIR}" ]]; then
    echo "[transcribe] removing the existing venv"
    rm -rf "${VENV_DIR}"
fi

# Created only when absent. Re-running to add an engine to an existing venv is
# the normal path — `--engines nemo` on top of a faster-whisper install — and
# `python3 -m venv` over a live venv rewrites its activate scripts, which fails
# outright if they are read-only and achieves nothing when they are not.
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    "${PYTHON}" -m venv "${VENV_DIR}"
else
    echo "[transcribe] reusing existing venv ($(${VENV_DIR}/bin/python -V 2>&1))"
fi
"${VENV_DIR}/bin/python" -m pip install --upgrade pip

# Always: the server itself. `requests` is what the router engine and the
# router-yield use; without it both degrade to warnings rather than failing.
"${VENV_DIR}/bin/python" -m pip install "flask>=3.0" "requests>=2.31"

# Installed once even when both nemo and hf are selected.
install_torch() {
    [[ -n "${TORCH_INSTALLED:-}" ]] && return 0
    if [[ -n "${TORCH_INDEX}" ]]; then
        "${VENV_DIR}/bin/python" -m pip install --index-url "${TORCH_INDEX}" torch torchaudio
    else
        "${VENV_DIR}/bin/python" -m pip install torch torchaudio
    fi
    TORCH_INSTALLED=1
}

if has_engine faster-whisper; then
    echo "[transcribe] installing faster-whisper ${FASTER_WHISPER_VERSION}"
    "${VENV_DIR}/bin/python" -m pip install "faster-whisper==${FASTER_WHISPER_VERSION}"
fi

if has_engine nemo; then
    echo "[transcribe] installing NeMo ASR (this is a multi-GB download)"
    install_torch
    if [[ -n "${NEMO_VERSION}" ]]; then
        "${VENV_DIR}/bin/python" -m pip install "nemo_toolkit[asr]==${NEMO_VERSION}"
    else
        "${VENV_DIR}/bin/python" -m pip install "nemo_toolkit[asr]"
    fi
fi

if has_engine hf; then
    echo "[transcribe] installing transformers ASR"
    install_torch
    "${VENV_DIR}/bin/python" -m pip install transformers accelerate soundfile librosa
fi

# Verify each selected engine separately, so a failure names itself instead of
# reporting "the install is broken" for whichever import happened to run first.
FAILED=0
verify() {
    local label="$1" code="$2"
    if "${VENV_DIR}/bin/python" -c "${code}" >/dev/null 2>&1; then
        echo "[transcribe]   ok: ${label}"
    else
        echo "[transcribe]   FAILED: ${label}" >&2
        FAILED=1
    fi
}

echo "[transcribe] verifying imports"
verify "flask + requests" "import flask, requests"
has_engine faster-whisper && verify "faster-whisper" "import faster_whisper"
has_engine nemo && verify "nemo asr" "import nemo.collections.asr"
has_engine hf && verify "transformers asr" "import transformers, torch"

if [[ "${FAILED}" -ne 0 ]]; then
    echo "[transcribe] One or more engines failed to import. The sidecar will still" >&2
    echo "[transcribe] start and serve the engines that did install; a request naming" >&2
    echo "[transcribe] a broken one answers 503 with the install hint." >&2
    exit 1
fi

echo "[transcribe] Done. Set TRANSCRIPT_ENABLED=on and TRANSCRIPT_ENGINES=${ENGINES} in"
echo "[transcribe] config/llm-stack.env, then run install.sh to (re)write the unit."
