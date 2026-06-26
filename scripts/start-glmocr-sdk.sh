#!/usr/bin/env bash
# =============================================================================
# start-glmocr-sdk.sh
# Launches the local GLM-OCR SDK layout/document parsing server.
# =============================================================================
set -euo pipefail

STACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set -a
source "${STACK_DIR}/config/llm-stack.env"
set +a

VENV_DIR="${GLMOCR_SDK_VENV_DIR:-${STACK_DIR}/deps/glmocr-sdk-venv}"
CONFIG_PATH="${GLMOCR_SDK_CONFIG_PATH:-${STACK_DIR}/config/glmocr-sdk.json}"

if [[ "${GLMOCR_SDK_ENABLED:-on}" != "on" ]]; then
    echo "[glmocr-sdk] Disabled by GLMOCR_SDK_ENABLED=${GLMOCR_SDK_ENABLED}" >&2
    exit 0
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "[glmocr-sdk] Creating Python venv at ${VENV_DIR}..."
    python3 -m venv "${VENV_DIR}"
fi

if ! "${VENV_DIR}/bin/python" - <<'PYCHECK' >/dev/null 2>&1
import glmocr
import flask
PYCHECK
then
    echo "[glmocr-sdk] Installing glmocr self-hosted server dependencies..."
    "${VENV_DIR}/bin/python" -m pip install --upgrade pip
    "${VENV_DIR}/bin/python" -m pip install 'glmocr[selfhosted,server]'
fi

mkdir -p "$(dirname "${CONFIG_PATH}")"

python3 - "${CONFIG_PATH}" <<'PYCONFIG'
import json
import os
import sys
from copy import deepcopy

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

def as_bool(name, default=False):
    value = getenv(name, "on" if default else "off").strip().lower()
    return value in {"1", "true", "yes", "on"}

def first_csv_value(value, default="0"):
    value = str(value or "").strip()
    if not value:
        return default
    first = value.split(",", 1)[0].strip()
    return first or default

def layout_cuda_devices():
    explicit = getenv("GLMOCR_LAYOUT_CUDA_VISIBLE_DEVICES", "").strip()
    if explicit:
        return first_csv_value(explicit)
    return first_csv_value(getenv("OCR_GPU_VISIBLE_DEVICES", "0"))

def layout_device():
    value = getenv("GLMOCR_LAYOUT_DEVICE", "").strip()
    if not value:
        return None
    if value.startswith("cuda:") and "," in value:
        return f"cuda:{first_csv_value(value.removeprefix('cuda:'))}"
    return value

def deep_merge(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base

config_path = sys.argv[1]
ocr_host = getenv("GLMOCR_OCR_API_HOST") or getenv("OCR_HOST") or getenv("LISTEN_HOST", "127.0.0.1")
if ocr_host in {"0.0.0.0", "::", "${LISTEN_HOST}", "$LISTEN_HOST"}:
    ocr_host = "127.0.0.1"

config = {
    "server": {
        "host": getenv("GLMOCR_SDK_HOST", "0.0.0.0"),
        "port": as_int("GLMOCR_SDK_PORT", 5002),
        "debug": as_bool("GLMOCR_SDK_DEBUG", False),
    },
    "logging": {
        "level": getenv("GLMOCR_SDK_LOG_LEVEL", "INFO"),
    },
    "pipeline": {
        "maas": {
            "enabled": False,
            "api_url": "http://127.0.0.1/disabled-local-only",
            "model": "glm-ocr",
            "api_key": None,
            "verify_ssl": False,
        },
        "ocr_api": {
            "api_host": ocr_host,
            "api_port": as_int("GLMOCR_OCR_API_PORT", as_int("OCR_PORT", 8009)),
            "api_scheme": getenv("GLMOCR_OCR_API_SCHEME", "http"),
            "api_path": getenv("GLMOCR_OCR_API_PATH", "/v1/chat/completions"),
            "api_url": getenv("GLMOCR_OCR_API_URL", None) or None,
            "api_key": getenv("GLMOCR_OCR_API_KEY", None) or None,
            "model": getenv("GLMOCR_OCR_MODEL", getenv("OCR_MODEL_NAME", "ocr")),
            "headers": {},
            "verify_ssl": as_bool("GLMOCR_OCR_VERIFY_SSL", False),
            "api_mode": getenv("GLMOCR_OCR_API_MODE", "openai"),
            "connect_timeout": as_int("GLMOCR_OCR_CONNECT_TIMEOUT", 30),
            "request_timeout": as_int("GLMOCR_OCR_REQUEST_TIMEOUT", 120),
            "retry_max_attempts": as_int("GLMOCR_OCR_RETRY_MAX_ATTEMPTS", 2),
            "retry_backoff_base_seconds": as_float("GLMOCR_OCR_RETRY_BACKOFF_BASE_SECONDS", 0.5),
            "retry_backoff_max_seconds": as_float("GLMOCR_OCR_RETRY_BACKOFF_MAX_SECONDS", 8.0),
            "retry_jitter_ratio": as_float("GLMOCR_OCR_RETRY_JITTER_RATIO", 0.2),
            "retry_status_codes": [429, 500, 502, 503, 504],
            "connection_pool_size": as_int("GLMOCR_OCR_CONNECTION_POOL_SIZE", 128),
        },
        "max_workers": as_int("GLMOCR_MAX_WORKERS", 16),
        "page_maxsize": as_int("GLMOCR_PAGE_MAXSIZE", 100),
        "region_maxsize": as_int("GLMOCR_REGION_MAXSIZE", 800),
        "page_loader": {
            "max_tokens": as_int("GLMOCR_PAGE_MAX_TOKENS", 8192),
            "temperature": as_float("GLMOCR_PAGE_TEMPERATURE", 0.0),
            "top_p": as_float("GLMOCR_PAGE_TOP_P", 0.00001),
            "top_k": as_int("GLMOCR_PAGE_TOP_K", 1),
            "repetition_penalty": as_float("GLMOCR_PAGE_REPETITION_PENALTY", 1.1),
            "image_format": getenv("GLMOCR_IMAGE_FORMAT", "JPEG"),
            "min_pixels": as_int("GLMOCR_MIN_PIXELS", 12544),
            "max_pixels": as_int("GLMOCR_MAX_PIXELS", 71372800),
            "pdf_dpi": as_int("GLMOCR_PDF_DPI", 200),
            "pdf_max_pages": None if getenv("GLMOCR_PDF_MAX_PAGES", "").strip() == "" else as_int("GLMOCR_PDF_MAX_PAGES", 0),
            "pdf_verbose": as_bool("GLMOCR_PDF_VERBOSE", False),
            "task_prompt_mapping": {
                "text": getenv("GLMOCR_PROMPT_TEXT", "Text Recognition:"),
                "table": getenv("GLMOCR_PROMPT_TABLE", "Table Recognition:"),
                "formula": getenv("GLMOCR_PROMPT_FORMULA", "Formula Recognition:"),
            },
        },
        "result_formatter": {
            "output_format": getenv("GLMOCR_OUTPUT_FORMAT", "both"),
            "enable_merge_formula_numbers": as_bool("GLMOCR_MERGE_FORMULA_NUMBERS", True),
            "enable_merge_text_blocks": as_bool("GLMOCR_MERGE_TEXT_BLOCKS", True),
            "enable_format_bullet_points": as_bool("GLMOCR_FORMAT_BULLET_POINTS", True),
        },
        "layout": {
            "model_dir": getenv("GLMOCR_LAYOUT_MODEL_DIR", "PaddlePaddle/PP-DocLayoutV3_safetensors"),
            "threshold": as_float("GLMOCR_LAYOUT_THRESHOLD", 0.3),
            "batch_size": as_int("GLMOCR_LAYOUT_BATCH_SIZE", 1),
            "workers": as_int("GLMOCR_LAYOUT_WORKERS", 1),
            "cuda_visible_devices": layout_cuda_devices(),
            "device": layout_device(),
            "layout_nms": as_bool("GLMOCR_LAYOUT_NMS", True),
            "use_polygon": as_bool("GLMOCR_LAYOUT_USE_POLYGON", False),
        },
    },
}

advanced_raw = getenv("GLMOCR_ADVANCED_CONFIG_JSON", "").strip()
if advanced_raw:
    try:
        advanced = json.loads(advanced_raw)
        if isinstance(advanced, dict):
            config = deep_merge(config, deepcopy(advanced))
    except Exception as exc:
        raise SystemExit(f"Invalid GLMOCR_ADVANCED_CONFIG_JSON: {exc}") from exc

with open(config_path, "w", encoding="utf-8") as fh:
    json.dump(config, fh, indent=2)
    fh.write("\n")
PYCONFIG

GLMOCR_EFFECTIVE_OCR_URL="${GLMOCR_OCR_API_URL:-}"
if [[ -z "${GLMOCR_EFFECTIVE_OCR_URL}" ]]; then
    GLMOCR_EFFECTIVE_OCR_HOST="${GLMOCR_OCR_API_HOST:-${OCR_HOST:-127.0.0.1}}"
    [[ "${GLMOCR_EFFECTIVE_OCR_HOST}" == "0.0.0.0" || "${GLMOCR_EFFECTIVE_OCR_HOST}" == "::" ]] && GLMOCR_EFFECTIVE_OCR_HOST="127.0.0.1"
    GLMOCR_EFFECTIVE_OCR_URL="http://${GLMOCR_EFFECTIVE_OCR_HOST}:${GLMOCR_OCR_API_PORT:-${OCR_PORT:-8009}}${GLMOCR_OCR_API_PATH:-/v1/chat/completions}"
fi

echo "[glmocr-sdk] Starting on http://${GLMOCR_SDK_HOST:-0.0.0.0}:${GLMOCR_SDK_PORT:-5002}"
echo "[glmocr-sdk] OCR backend: ${GLMOCR_EFFECTIVE_OCR_URL}"
exec "${VENV_DIR}/bin/python" "${STACK_DIR}/scripts/glmocr-sdk-server.py" --config "${CONFIG_PATH}"
