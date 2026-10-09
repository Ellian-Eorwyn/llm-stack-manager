#!/usr/bin/env bash
# =============================================================================
# start-voice-tts.sh
# Launches Hermes's voice: Qwen3-TTS behind an OpenAI-style speech API
# (scripts/voice-tts-server.py). See docs/voice-tts.md.
# =============================================================================
set -euo pipefail

STACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set -a
source "${STACK_DIR}/config/llm-stack.env"
set +a

if [[ "${VOICE_TTS_ENABLED:-off}" != "on" ]]; then
    echo "[voice-tts] Disabled by VOICE_TTS_ENABLED=${VOICE_TTS_ENABLED:-off}" >&2
    exit 0
fi

PYTHON="${VOICE_TTS_PYTHON:-${HOME}/AI/voice-tts/venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
    echo "[voice-tts] ${PYTHON} is missing. Build the venv as docs/voice-tts.md describes, or set VOICE_TTS_PYTHON." >&2
    exit 1
fi

# The server reads these itself and treats a set-but-empty value as the value,
# so a field left blank in the config page has to reach it as unset to mean
# "the server's default".
for key in VOICE_TTS_VOICES VOICE_TTS_MODEL VOICE_TTS_CLONE_MODEL VOICE_TTS_LANGUAGE \
           VOICE_TTS_CHUNK VOICE_TTS_FFMPEG; do
    if [[ -z "${!key:-}" ]]; then unset "${key}"; fi
done

export CUDA_VISIBLE_DEVICES="${VOICE_TTS_GPU:-0}"
export PYTHONUNBUFFERED=1
if [[ "${VOICE_TTS_HF_OFFLINE:-on}" == "on" ]]; then
    export HF_HUB_OFFLINE=1
fi

HOST="${VOICE_TTS_HOST:-127.0.0.1}"
PORT="${VOICE_TTS_PORT:-8016}"
echo "[voice-tts] Starting on http://${HOST}:${PORT} (GPU ${CUDA_VISIBLE_DEVICES})"
exec "${PYTHON}" "${STACK_DIR}/scripts/voice-tts-server.py" --host "${HOST}" --port "${PORT}"
