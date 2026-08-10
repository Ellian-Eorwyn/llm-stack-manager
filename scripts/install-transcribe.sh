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
TORCH_INDEX="${TRANSCRIPT_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --engines) ENGINES="$2"; shift 2 ;;
        --engines=*) ENGINES="${1#*=}"; shift ;;
        --venv) VENV_DIR="$2"; shift 2 ;;
        -h|--help) sed -n '2,14p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "[transcribe] Unknown argument: $1" >&2; exit 2 ;;
    esac
done

has_engine() {
    [[ ",${ENGINES}," == *",$1,"* ]]
}

echo "[transcribe] venv:    ${VENV_DIR}"
echo "[transcribe] engines: ${ENGINES}"

python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip

# Always: the server itself. `requests` is what the router engine and the
# router-yield use; without it both degrade to warnings rather than failing.
"${VENV_DIR}/bin/python" -m pip install "flask>=3.0" "requests>=2.31"

if has_engine faster-whisper; then
    echo "[transcribe] installing faster-whisper ${FASTER_WHISPER_VERSION}"
    "${VENV_DIR}/bin/python" -m pip install "faster-whisper==${FASTER_WHISPER_VERSION}"
fi

if has_engine nemo; then
    echo "[transcribe] installing NeMo ASR (this is a multi-GB download)"
    "${VENV_DIR}/bin/python" -m pip install --index-url "${TORCH_INDEX}" torch torchaudio
    if [[ -n "${NEMO_VERSION}" ]]; then
        "${VENV_DIR}/bin/python" -m pip install "nemo_toolkit[asr]==${NEMO_VERSION}"
    else
        "${VENV_DIR}/bin/python" -m pip install "nemo_toolkit[asr]"
    fi
fi

if has_engine hf; then
    echo "[transcribe] installing transformers ASR"
    "${VENV_DIR}/bin/python" -m pip install --index-url "${TORCH_INDEX}" torch torchaudio
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
