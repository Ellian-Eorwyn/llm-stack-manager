#!/usr/bin/env bash
# =============================================================================
# start-transcribe.sh
# Launches the speech-to-text sidecar (transcript-backend).
# =============================================================================
set -euo pipefail

STACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set -a
source "${STACK_DIR}/config/llm-stack.env"
set +a

VENV_DIR="${TRANSCRIPT_VENV_DIR:-${STACK_DIR}/deps/transcribe-venv}"
CONFIG_PATH="${TRANSCRIPT_CONFIG_PATH:-${STACK_DIR}/config/transcribe.json}"

# Exiting 0 rather than failing is the contract `health.ENABLED_FLAGS` relies on:
# a service switched off on purpose must not read as a fault.
if [[ "${TRANSCRIPT_ENABLED:-off}" != "on" ]]; then
    echo "[transcribe] Disabled by TRANSCRIPT_ENABLED=${TRANSCRIPT_ENABLED:-off}" >&2
    exit 0
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]] || ! "${VENV_DIR}/bin/python" - <<'PYCHECK' >/dev/null 2>&1
import flask
PYCHECK
then
    echo "[transcribe] Sidecar environment is missing or incomplete." >&2
    echo "[transcribe] Run: bash scripts/install-transcribe.sh --engines ${TRANSCRIPT_ENGINES:-faster-whisper}" >&2
    exit 1
fi

mkdir -p "$(dirname "${CONFIG_PATH}")" "${TRANSCRIPT_WORK_DIR:-${STACK_DIR}/logs/transcript/work}"

echo "[transcribe] Host:           ${TRANSCRIPT_HOST:-127.0.0.1}:${TRANSCRIPT_PORT:-8014}"
echo "[transcribe] Active engine:  ${TRANSCRIPT_ACTIVE_ENGINE:-faster-whisper}"
echo "[transcribe] Installed:      ${TRANSCRIPT_ENGINES:-faster-whisper}"
echo "[transcribe] Device:         ${TRANSCRIPT_LOCAL_DEVICE:-cuda} / ${TRANSCRIPT_LOCAL_COMPUTE_TYPE:-float16}"
echo "[transcribe] Idle unload:    ${TRANSCRIPT_IDLE_UNLOAD_SECONDS:-300}s"
echo "[transcribe] Router yield:   ${TRANSCRIPT_ROUTER_YIELD:-asr}"

# Only the local engines see this; the router engine has no weights of its own.
export CUDA_VISIBLE_DEVICES="${TRANSCRIPT_GPU_VISIBLE_DEVICES:-${ASR_GPU_VISIBLE_DEVICES:-0}}"
export HF_HOME="${TRANSCRIPT_HF_HOME:-${STACK_DIR}/models/transcription/.cache}"

python3 - "${CONFIG_PATH}" "${STACK_DIR}" <<'PYCONFIG'
import json
import os
import sys

config_path, stack_dir = sys.argv[1], sys.argv[2]


def getenv(name, default=""):
    return os.environ.get(name, default)


def as_int(name, default):
    try:
        return int(float(getenv(name, str(default))))
    except Exception:
        return default


def as_float(name, default):
    try:
        return float(getenv(name, str(default)))
    except Exception:
        return default


def deep_merge(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


# systemd's EnvironmentFile= does not expand ${VAR}, so a config written as
# TRANSCRIPT_HOST=${LISTEN_HOST} arrives here as that literal string. Same guard
# as start-glmocr-sdk.sh, for the same reason.
def clean_host(value, fallback="127.0.0.1"):
    value = (value or "").strip()
    if not value or value.startswith("$") or value in {"${LISTEN_HOST}", "$LISTEN_HOST"}:
        return fallback
    return value


# The manager already knows how to read `preset:` / `local:` refs. Reuse it so
# there is one implementation of the format, but never let a broken manager tree
# stop the sidecar from starting.
def parse_model_value(value):
    raw = (value or "").strip()
    if raw.startswith(("preset:", "local:")):
        return raw
    return f"preset:{raw}" if raw else ""


try:
    sys.path.insert(0, os.path.join(stack_dir, "web"))
    import models as _manager_models

    def parse_model_value(value):  # noqa: F811 - deliberate upgrade when available
        parsed = _manager_models.parse_transcription_model_value(value)
        kind, val = parsed.get("kind"), parsed.get("value")
        if not val:
            return ""
        return f"{kind}:{val}" if kind in ("preset", "local") else f"preset:{val}"
except Exception:
    pass

ENGINE_PREFIXES = {
    "faster-whisper": "FASTER_WHISPER",
    "parakeet-v3": "PARAKEET_V3",
    "canary-qwen": "CANARY_QWEN",
    "hf-asr": "HF_ASR",
    "router": "ROUTER_ASR",
}

engines = {}
for engine_id, prefix in ENGINE_PREFIXES.items():
    engines[engine_id] = {
        "model": parse_model_value(getenv(f"{prefix}_LOCAL_MODEL", "")),
        "backend_type": getenv(f"{prefix}_BACKEND_TYPE", "local"),
        "upstream_url": getenv(f"{prefix}_UPSTREAM_URL", ""),
        "api_key": getenv(f"{prefix}_API_KEY", ""),
        "transcribe_path": getenv(f"{prefix}_TRANSCRIBE_PATH", "/v1/audio/transcriptions"),
        "models_dir": os.path.join(stack_dir, "models", "transcription", engine_id),
    }

config = {
    "server": {
        "host": clean_host(getenv("TRANSCRIPT_HOST"), clean_host(getenv("LISTEN_HOST"))),
        "port": as_int("TRANSCRIPT_PORT", 8014),
        "token": getenv("TRANSCRIPT_API_TOKEN", ""),
        "log_level": getenv("TRANSCRIPT_LOG_LEVEL", "INFO"),
    },
    "active_engine": getenv("TRANSCRIPT_ACTIVE_ENGINE", "faster-whisper"),
    "runtime": {
        "device": getenv("TRANSCRIPT_LOCAL_DEVICE", "cuda"),
        "compute_type": getenv("TRANSCRIPT_LOCAL_COMPUTE_TYPE", "float16"),
    },
    "router": {
        # Always loopback: the router binds 127.0.0.1 on purpose.
        "host": clean_host(getenv("MODEL_ROUTER_HOST"), "127.0.0.1"),
        "port": as_int("MODEL_ROUTER_PORT", 8013),
        "model": getenv("ASR_MODEL_NAME", "asr"),
        "yield_mode": getenv("TRANSCRIPT_ROUTER_YIELD", "asr"),
        "allow_degraded": getenv("TRANSCRIPT_ROUTER_ALLOW_DEGRADED", "off"),
        "enabled": getenv("MODEL_ROUTER_ENABLED", "off"),
    },
    "limits": {
        "idle_unload_seconds": as_float("TRANSCRIPT_IDLE_UNLOAD_SECONDS", 300),
        "max_concurrency": as_int("TRANSCRIPT_MAX_CONCURRENCY", 1),
        "max_upload_mb": as_int("TRANSCRIPT_MAX_UPLOAD_MB", 512),
        "async_threshold_seconds": as_float("TRANSCRIPT_ASYNC_THRESHOLD_SECONDS", 900),
        "job_ttl_seconds": as_float("TRANSCRIPT_JOB_TTL_SECONDS", 3600),
        "timeout_seconds": as_float("TRANSCRIPT_TIMEOUT_SECONDS", 600),
        "default_format": getenv("TRANSCRIPT_DEFAULT_FORMAT", "json"),
        "url_allow_hosts": getenv("TRANSCRIPT_URL_ALLOW_HOSTS", ""),
        "oai_allow_long": getenv("TRANSCRIPT_OAI_ALLOW_LONG", "off"),
        "work_dir": getenv("TRANSCRIPT_WORK_DIR", os.path.join(stack_dir, "logs", "transcript", "work")),
    },
    "engines": engines,
}

advanced = getenv("TRANSCRIPT_ADVANCED_CONFIG_JSON", "").strip()
if advanced:
    try:
        deep_merge(config, json.loads(advanced))
    except Exception as exc:
        print(f"[transcribe] Ignoring TRANSCRIPT_ADVANCED_CONFIG_JSON: {exc}", file=sys.stderr)

with open(config_path, "w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2)
    handle.write("\n")
print(f"[transcribe] Wrote {config_path}")
PYCONFIG

exec "${VENV_DIR}/bin/python" "${STACK_DIR}/scripts/transcribe-server.py" --config "${CONFIG_PATH}"
