#!/usr/bin/env python3
"""
The env file, and the key names inside it.

Every configuration surface the manager has — the config form, saved profiles,
the setup wizard, the pre-flight check — funnels through the handful of
functions here, and they exist because the same setting has been spelled several
different ways over the life of this stack.

Reading and writing are deliberately asymmetric:

  * **Reading** is generous. `normalize_env_keys` backfills a canonical key from
    whichever legacy name is present and fills in defaults, so a profile written
    when the primary backend was called `CHAT_DENSE_*` still loads today.
    `read_env_raw` is the escape hatch for the one caller that needs to know
    what the file literally says rather than what it resolves to — the
    deprecation report, which cannot recommend removing a key it was handed a
    backfilled copy of.

  * **Writing** is strict and canonical. `normalize_config_updates` rewrites
    legacy names before anything reaches disk, `allowed_config_keys` bounds what
    may be written at all, and `update_env_values` collapses a key's aliases
    onto a single line. Nothing writes a legacy name any more.

That asymmetry is the whole migration strategy: legacy names are readable
forever and writable never, so they drain out of the file as settings are
re-saved without any flag day. `config_fields.LEGACY_ENV_KEY_MAP` carries the
note on the staged path to removing them entirely.

`allowed_config_keys` admits keys that are already in the env file even when no
UI field declares them. That is not laxness — settings have repeatedly been
added to `llm-stack.env` before they were given a control, and dropping them on
save would silently reconfigure a working stack.
"""

from __future__ import annotations

import re

import core
from config_fields import (
    BUILTIN_CHAT_VARIANT_BY_ID,
    CODE_TO_CHAT_MIRRORS,
    CONFIG_FIELDS,
    LEGACY_ENV_KEY_MAP,
    NEW_ENV_KEY_LEGACY_ALIASES,
    RESTART_HINTS,
)


def apply_code_chat_mirrors(updates: dict) -> dict:
    """Mirror code backend-level settings onto shared chat backend keys.

    Full saved configs contain both CODE_* and CHAT_* values. In that case the
    explicit CHAT_* value must win, otherwise legacy CODE_BATCH_SIZE defaults
    overwrite saved shared-backend batch settings during load/startup.
    """
    expanded = dict(updates)
    for code_key, chat_key in CODE_TO_CHAT_MIRRORS.items():
        if code_key in updates:
            chat_keys = chat_key if isinstance(chat_key, list) else [chat_key]
            for key in chat_keys:
                if key not in updates:
                    expanded[key] = updates[code_key]
    return expanded


def normalize_env_keys(env: dict) -> dict:
    normalized = dict(env)
    for legacy_key, new_key in LEGACY_ENV_KEY_MAP.items():
        if new_key not in normalized and legacy_key in normalized:
            value = normalized[legacy_key]
            if legacy_key == "CHAT_DENSE_LABEL" and value.strip() == "Backend Dense":
                value = BUILTIN_CHAT_VARIANT_BY_ID["dense"]["default_label"]
            elif legacy_key == "CHAT_MOE_LABEL" and value.strip() == "Backend MoE":
                value = BUILTIN_CHAT_VARIANT_BY_ID["moe"]["default_label"]
            normalized[new_key] = value
    backend_defaults = {
        "CHAT_PRIMARY_LABEL": BUILTIN_CHAT_VARIANT_BY_ID["dense"]["default_label"],
        "CHAT_PRIMARY_MODEL_NAME": "chat-dense",
        "CHAT_PRIMARY_MODEL_PATH": normalized.get("CHAT_MODEL_PATH", ""),
        "CHAT_PRIMARY_MMPROJ_PATH": normalized.get("CHAT_MMPROJ_PATH", ""),
        "CHAT_PRIMARY_CTX_SIZE": normalized.get("CHAT_CTX_SIZE", "32768"),
        "CHAT_SECONDARY_LABEL": BUILTIN_CHAT_VARIANT_BY_ID["moe"]["default_label"],
        "CHAT_SECONDARY_MODEL_NAME": "chat-moe",
        "CHAT_SECONDARY_MODEL_PATH": normalized.get("CHAT_MODEL_PATH", ""),
        "CHAT_SECONDARY_MMPROJ_PATH": normalized.get("CHAT_MMPROJ_PATH", ""),
        "CHAT_SECONDARY_CTX_SIZE": normalized.get("CHAT_CTX_SIZE", "32768"),
    }
    for key, value in backend_defaults.items():
        normalized.setdefault(key, value)
    for field in CONFIG_FIELDS:
        key = field.get("key", "")
        if key.startswith("CHAT_PRIMARY_") and key not in normalized:
            legacy_key = "CHAT_" + key[len("CHAT_PRIMARY_"):]
            if legacy_key in normalized:
                normalized[key] = normalized[legacy_key]
        elif key.startswith("CHAT_SECONDARY_") and key not in normalized:
            legacy_key = "CHAT_" + key[len("CHAT_SECONDARY_"):]
            if legacy_key in normalized:
                normalized[key] = normalized[legacy_key]
    normalized.setdefault("CHAT_MODEL_NAME", "chat-custom")
    normalized.setdefault("CHAT_CUSTOM_ARGS_JSON", "[]")
    normalized.setdefault("CHAT_TEMPLATE_ID", "")
    normalized.setdefault("CHAT_THREADS", "-1")
    normalized.setdefault("CHAT_THREADS_BATCH", "-1")
    normalized.setdefault("CHAT_CACHE_RAM", "8192")
    normalized.setdefault("CHAT_CTX_CHECKPOINTS", "8")
    normalized.setdefault("CHAT_CACHE_IDLE_SLOTS", "on")
    normalized.setdefault("CHAT_CACHE_REUSE", "256")
    normalized.setdefault("CHAT_SWA_FULL", "off")
    normalized.setdefault("CHAT_FIT_TARGET", "")
    normalized.setdefault("CHAT_FIT_CTX", "4096")
    normalized.setdefault("CHAT2_CACHE_RAM", "8192")
    normalized.setdefault("CHAT2_CTX_CHECKPOINTS", "8")
    normalized.setdefault("CHAT2_CACHE_IDLE_SLOTS", "on")
    normalized.setdefault("CHAT2_CACHE_REUSE", "256")
    normalized.setdefault("CHAT2_SWA_FULL", "off")
    normalized.setdefault("CHAT2_LABEL", "Secondary Backend")
    normalized.setdefault("CHAT2_FIT_TARGET", "")
    normalized.setdefault("CHAT2_FIT_CTX", "4096")
    normalized.setdefault("CHAT2_CUSTOM_ARGS_JSON", "[]")
    normalized.setdefault("CHAT_SPEC_METHOD", "off")
    normalized.setdefault("CHAT_SPEC_NGRAM_MOD", "off")
    normalized.setdefault("CHAT_SPEC_DRAFT_MODEL_PATH", "")
    normalized.setdefault("CHAT_SPEC_DRAFT_N_GPU_LAYERS", "auto")
    normalized.setdefault("CHAT_SPEC_DRAFT_DEVICES", "")
    normalized.setdefault("CHAT_SPEC_DRAFT_TYPE_K", "f16")
    normalized.setdefault("CHAT_SPEC_DRAFT_TYPE_V", "f16")
    normalized.setdefault("CHAT_SPEC_DRAFT_N_MAX", "6")
    normalized.setdefault("CHAT_SPEC_DRAFT_N_MIN", "0")
    normalized.setdefault("CHAT_SPEC_DRAFT_P_MIN", "0.75")
    normalized.setdefault("CHAT_SPEC_DRAFT_P_SPLIT", "0.10")
    normalized.setdefault("CHAT_SPEC_NGRAM_MOD_N_MATCH", "24")
    normalized.setdefault("CHAT_SPEC_NGRAM_MOD_N_MIN", "48")
    normalized.setdefault("CHAT_SPEC_NGRAM_MOD_N_MAX", "64")
    normalized.setdefault("CHAT_SPEC_NGRAM_SIZE_N", "12")
    normalized.setdefault("CHAT_SPEC_NGRAM_SIZE_M", "48")
    normalized.setdefault("CHAT_SPEC_NGRAM_MIN_HITS", "1")
    normalized.setdefault("CHAT2_SPEC_METHOD", "off")
    normalized.setdefault("CHAT2_SPEC_NGRAM_MOD", "off")
    normalized.setdefault("CHAT2_SPEC_DRAFT_MODEL_PATH", "")
    normalized.setdefault("CHAT2_SPEC_DRAFT_N_GPU_LAYERS", "auto")
    normalized.setdefault("CHAT2_SPEC_DRAFT_DEVICES", "")
    normalized.setdefault("CHAT2_SPEC_DRAFT_TYPE_K", "f16")
    normalized.setdefault("CHAT2_SPEC_DRAFT_TYPE_V", "f16")
    normalized.setdefault("CHAT2_SPEC_DRAFT_N_MAX", "6")
    normalized.setdefault("CHAT2_SPEC_DRAFT_N_MIN", "0")
    normalized.setdefault("CHAT2_SPEC_DRAFT_P_MIN", "0.75")
    normalized.setdefault("CHAT2_SPEC_DRAFT_P_SPLIT", "0.10")
    normalized.setdefault("CHAT2_SPEC_NGRAM_MOD_N_MATCH", "24")
    normalized.setdefault("CHAT2_SPEC_NGRAM_MOD_N_MIN", "48")
    normalized.setdefault("CHAT2_SPEC_NGRAM_MOD_N_MAX", "64")
    normalized.setdefault("CHAT2_SPEC_NGRAM_SIZE_N", "12")
    normalized.setdefault("CHAT2_SPEC_NGRAM_SIZE_M", "48")
    normalized.setdefault("CHAT2_SPEC_NGRAM_MIN_HITS", "1")
    normalized.setdefault("THINK_MODEL_NAME", "think")
    normalized.setdefault("NOTHINK_MODEL_NAME", "chat")
    normalized.setdefault("CODE_MODEL_NAME", "code")
    normalized.setdefault("PROXY_STREAM_PASSTHROUGH", "off")
    normalized.setdefault("UPSTREAM_400_CAPTURE_ENABLED", "off")
    normalized.setdefault("THINK_TEMP", normalized.get("CHAT_TEMP", "0.7"))
    normalized.setdefault("THINK_MAX_TOKENS", "0")
    normalized.setdefault("THINK_TOP_P", normalized.get("CHAT_TOP_P", "0.95"))
    normalized.setdefault("THINK_TOP_K", normalized.get("CHAT_TOP_K", "20"))
    normalized.setdefault("THINK_MIN_P", normalized.get("CHAT_MIN_P", "0.00"))
    normalized.setdefault("THINK_PRESENCE_PENALTY", "0.00")
    normalized.setdefault("THINK_REPEAT_PENALTY", "1.00")
    normalized.setdefault("THINK_REASONING_FORMAT", normalized.get("CHAT_REASONING_FORMAT", "deepseek"))
    normalized.setdefault("THINK_JINJA", "on")
    normalized.setdefault("THINK_PRESERVE_THINKING", "on")
    normalized.setdefault("THINK_REASONING_STREAM_MODE", "hidden")
    normalized.setdefault("NOTHINK_TEMP", normalized.get("CHAT_TEMP", "0.7"))
    normalized.setdefault("NOTHINK_MAX_TOKENS", "0")
    normalized.setdefault("NOTHINK_TOP_P", normalized.get("CHAT_TOP_P", "0.95"))
    normalized.setdefault("NOTHINK_TOP_K", normalized.get("CHAT_TOP_K", "20"))
    normalized.setdefault("NOTHINK_MIN_P", normalized.get("CHAT_MIN_P", "0.00"))
    normalized.setdefault("NOTHINK_PRESENCE_PENALTY", "0.00")
    normalized.setdefault("NOTHINK_REPEAT_PENALTY", "1.00")
    normalized.setdefault("NOTHINK_REASONING_FORMAT", normalized.get("CHAT_REASONING_FORMAT", "deepseek"))
    normalized.setdefault("NOTHINK_JINJA", "on")
    normalized.setdefault("NOTHINK_PRESERVE_THINKING", "off")
    normalized.setdefault("NOTHINK_REASONING_STREAM_MODE", "hidden")
    normalized.setdefault("CODE_PRESERVE_THINKING", "on")
    normalized.setdefault("CODE_REASONING_STREAM_MODE", "hidden")
    normalized.setdefault("CODE_MAX_TOKENS", "0")
    normalized.setdefault("CODE_PRESENCE_PENALTY", "0.00")
    normalized.setdefault("CODE_REPEAT_PENALTY", "1.00")
    normalized.setdefault("TASK_MODEL_NAME", "task")
    normalized.setdefault("TASK_PRESENCE_PENALTY", "0.00")
    normalized.setdefault("TASK_REPEAT_PENALTY", "1.00")
    normalized.setdefault("TASK_CUSTOM_ARGS_JSON", "[]")
    normalized.setdefault("TASK_CHAT_TEMPLATE_ID", "")
    normalized.setdefault("TASK_THREADS", "-1")
    normalized.setdefault("TASK_THREADS_BATCH", "-1")
    normalized.setdefault("TASK_CACHE_RAM", "8192")
    normalized.setdefault("TASK_CTX_CHECKPOINTS", "32")
    normalized.setdefault("TASK_CACHE_IDLE_SLOTS", "on")
    normalized.setdefault("TASK_CACHE_REUSE", "0")
    normalized.setdefault("TASK_SWA_FULL", "off")
    normalized.setdefault("TASK_FIT_TARGET", "")
    normalized.setdefault("TASK_FIT_CTX", "4096")
    normalized.setdefault("TASK_SPEC_METHOD", "off")
    normalized.setdefault("TASK_SPEC_NGRAM_MOD", "off")
    normalized.setdefault("TASK_SPEC_DRAFT_MODEL_PATH", "")
    normalized.setdefault("TASK_SPEC_DRAFT_N_GPU_LAYERS", "auto")
    normalized.setdefault("TASK_SPEC_DRAFT_DEVICES", "")
    normalized.setdefault("TASK_SPEC_DRAFT_TYPE_K", "f16")
    normalized.setdefault("TASK_SPEC_DRAFT_TYPE_V", "f16")
    normalized.setdefault("TASK_SPEC_DRAFT_N_MAX", "6")
    normalized.setdefault("TASK_SPEC_DRAFT_N_MIN", "0")
    normalized.setdefault("TASK_SPEC_DRAFT_P_MIN", "0.75")
    normalized.setdefault("TASK_SPEC_DRAFT_P_SPLIT", "0.10")
    normalized.setdefault("TASK_SPEC_NGRAM_MOD_N_MATCH", "24")
    normalized.setdefault("TASK_SPEC_NGRAM_MOD_N_MIN", "48")
    normalized.setdefault("TASK_SPEC_NGRAM_MOD_N_MAX", "64")
    normalized.setdefault("TASK_SPEC_NGRAM_SIZE_N", "12")
    normalized.setdefault("TASK_SPEC_NGRAM_SIZE_M", "48")
    normalized.setdefault("TASK_SPEC_NGRAM_MIN_HITS", "1")
    normalized.setdefault("EMBED_MODEL_NAME", "embed")
    normalized.setdefault("EMBED2_MODEL_NAME", "embed2")
    normalized.setdefault("EMBED_THREADS", "-1")
    normalized.setdefault("EMBED_THREADS_BATCH", "-1")
    normalized.setdefault("RERANK_MODEL_NAME", "rank")
    normalized.setdefault("RERANK_THREADS", "-1")
    normalized.setdefault("RERANK_THREADS_BATCH", "-1")
    normalized.setdefault("OCR_MODEL_NAME", "ocr")
    normalized.setdefault("OCR_MODEL_PATH", str(core.STACK_DIR / "models" / "GLM-OCR-F16.gguf"))
    normalized.setdefault("OCR_MMPROJ_PATH", "")
    normalized.setdefault("OCR_HOST", normalized.get("LISTEN_HOST", "0.0.0.0"))
    normalized.setdefault("OCR_PORT", "8009")
    normalized.setdefault("OCR_CTX_SIZE", "8192")
    normalized.setdefault("OCR_N_PARALLEL", "1")
    normalized.setdefault("OCR_THREADS", "-1")
    normalized.setdefault("OCR_THREADS_BATCH", "-1")
    normalized.setdefault("OCR_N_GPU_LAYERS", "-1")
    normalized.setdefault("OCR_MAIN_GPU", "0")
    normalized.setdefault("OCR_DEVICE", "")
    normalized.setdefault("OCR_TENSOR_SPLIT", "auto")
    normalized.setdefault("OCR_SPLIT_MODE", "layer")
    normalized.setdefault("OCR_KV_OFFLOAD", "on")
    normalized.setdefault("OCR_OP_OFFLOAD", "on")
    normalized.setdefault("OCR_MMPROJ_OFFLOAD", "on")
    normalized.setdefault("OCR_BATCH_SIZE", "2048")
    normalized.setdefault("OCR_UBATCH_SIZE", "512")
    normalized.setdefault("OCR_FLASH_ATTN", "on")
    normalized.setdefault("OCR_CACHE_TYPE_K", "f16")
    normalized.setdefault("OCR_CACHE_TYPE_V", "f16")
    normalized.setdefault("OCR_NO_MMAP", "false")
    normalized.setdefault("OCR_MLOCK", "false")
    normalized.setdefault(
        "OCR_GPU_VISIBLE_DEVICES",
        normalized.get("CHAT_GPU_VISIBLE_DEVICES", normalized.get("TASK_GPU_VISIBLE_DEVICES", "0")),
    )
    normalized.setdefault("OCR_PROMPT", "OCR")
    normalized.setdefault("OCR_TEMP", "0.1")
    normalized.setdefault("OCR_TOP_P", "0.95")
    normalized.setdefault("OCR_TOP_K", "1")
    normalized.setdefault("OCR_MIN_P", "0.00")
    normalized.setdefault("OCR_FIT", "off")
    normalized.setdefault("OCR_CUSTOM_ARGS_JSON", "[]")
    # Read by /api/ocr/extract. Generous because a router-mode request can wait
    # out a cold model load before any inference starts.
    normalized.setdefault("OCR_TIMEOUT_SECONDS", "600")
    normalized.setdefault("GLMOCR_SDK_ENABLED", "on")
    normalized.setdefault("GLMOCR_SDK_HOST", normalized.get("LISTEN_HOST", "0.0.0.0"))
    normalized.setdefault("GLMOCR_SDK_PORT", "5002")
    normalized.setdefault("GLMOCR_PUBLIC_URL", f"http://127.0.0.1:{normalized.get('GLMOCR_SDK_PORT', '5002')}/glmocr/parse")
    normalized.setdefault("GLMOCR_SDK_LOG_LEVEL", "INFO")
    normalized.setdefault("GLMOCR_OCR_API_MODE", "openai")
    normalized.setdefault("GLMOCR_OCR_API_URL", "")
    # A read timeout, and the binding constraint on OCR: it has to cover a cold
    # model load as well as the inference. Connecting is never the problem —
    # something is always listening — so the connect timeout stays short.
    normalized.setdefault("GLMOCR_OCR_REQUEST_TIMEOUT", "600")
    # Not a socket timeout: the SDK's startup probe is a real inference POST,
    # retried until this expires. Under the router that first POST is what loads
    # the OCR model, so a value shorter than a cold load makes the SDK exit and
    # `Restart=on-failure` flap it — the failure this stack already has history
    # with. Each individual attempt is capped at 30s inside the SDK, and the
    # load continues server-side after one gives up, so this only needs to be
    # long enough for a later retry to find the model ready.
    normalized.setdefault("GLMOCR_OCR_CONNECT_TIMEOUT", "300")
    normalized.setdefault("GLMOCR_OCR_RETRY_MAX_ATTEMPTS", "4")
    normalized.setdefault("GLMOCR_OCR_RETRY_BACKOFF_BASE_SECONDS", "0.5")
    normalized.setdefault("GLMOCR_OCR_RETRY_BACKOFF_MAX_SECONDS", "30")
    normalized.setdefault("GLMOCR_OCR_CONNECTION_POOL_SIZE", "128")
    normalized.setdefault("GLMOCR_MAX_WORKERS", "16")
    normalized.setdefault("GLMOCR_PAGE_MAXSIZE", "100")
    normalized.setdefault("GLMOCR_REGION_MAXSIZE", "800")
    normalized.setdefault("GLMOCR_PAGE_MAX_TOKENS", "8192")
    normalized.setdefault("GLMOCR_PAGE_TEMPERATURE", "0.0")
    normalized.setdefault("GLMOCR_PAGE_TOP_P", "0.00001")
    normalized.setdefault("GLMOCR_PAGE_TOP_K", "1")
    normalized.setdefault("GLMOCR_PAGE_REPETITION_PENALTY", "1.1")
    normalized.setdefault("GLMOCR_IMAGE_FORMAT", "JPEG")
    normalized.setdefault("GLMOCR_MIN_PIXELS", "12544")
    normalized.setdefault("GLMOCR_MAX_PIXELS", "71372800")
    normalized.setdefault("GLMOCR_PDF_DPI", "200")
    normalized.setdefault("GLMOCR_PDF_MAX_PAGES", "")
    normalized.setdefault("GLMOCR_LAYOUT_MODEL_DIR", "PaddlePaddle/PP-DocLayoutV3_safetensors")
    normalized.setdefault("GLMOCR_LAYOUT_DEVICE", "")
    normalized.setdefault("GLMOCR_LAYOUT_CUDA_VISIBLE_DEVICES", "")
    layout_gpus = str(normalized.get("GLMOCR_LAYOUT_CUDA_VISIBLE_DEVICES") or "").strip()
    if "," in layout_gpus:
        normalized["GLMOCR_LAYOUT_CUDA_VISIBLE_DEVICES"] = layout_gpus.split(",", 1)[0].strip() or "0"
    layout_device = str(normalized.get("GLMOCR_LAYOUT_DEVICE") or "").strip()
    if layout_device.startswith("cuda:") and "," in layout_device:
        normalized["GLMOCR_LAYOUT_DEVICE"] = f"cuda:{layout_device.removeprefix('cuda:').split(',', 1)[0].strip() or '0'}"
    normalized.setdefault("GLMOCR_LAYOUT_THRESHOLD", "0.3")
    normalized.setdefault("GLMOCR_LAYOUT_BATCH_SIZE", "1")
    normalized.setdefault("GLMOCR_LAYOUT_WORKERS", "1")
    normalized.setdefault("GLMOCR_LAYOUT_USE_POLYGON", "off")
    normalized.setdefault("GLMOCR_OUTPUT_FORMAT", "both")
    normalized.setdefault("GLMOCR_MERGE_FORMULA_NUMBERS", "on")
    normalized.setdefault("GLMOCR_MERGE_TEXT_BLOCKS", "on")
    normalized.setdefault("GLMOCR_FORMAT_BULLET_POINTS", "on")
    normalized.setdefault("GLMOCR_PROMPT_TEXT", "Text Recognition:")
    normalized.setdefault("GLMOCR_PROMPT_TABLE", "Table Recognition:")
    normalized.setdefault("GLMOCR_PROMPT_FORMULA", "Formula Recognition:")
    normalized.setdefault("GLMOCR_ADVANCED_CONFIG_JSON", "{}")
    # Model router. Off by default: turning it on replaces four systemd units
    # with one, so it is a deliberate act rather than something an upgrade does
    # to a working stack.
    normalized.setdefault("MODEL_ROUTER_ENABLED", "off")
    normalized.setdefault("MODEL_ROUTER_PORT", "8013")
    # Loopback: nginx fronts the per-model ports and proxies here, so exposing
    # the router itself would only add an unauthenticated way in.
    normalized.setdefault("MODEL_ROUTER_HOST", "127.0.0.1")
    normalized.setdefault("MODEL_ROUTER_MAX", "2")
    normalized.setdefault("MODEL_ROUTER_MEMBERS", "EMBED,OCR,RERANK,TASK")
    normalized.setdefault("MODEL_ROUTER_SLEEP_IDLE_SECONDS", "600")
    normalized.setdefault("MODEL_ROUTER_GPU_VISIBLE_DEVICES", "0,1")
    normalized.setdefault("SEARXNG_ENABLED", "on")
    normalized.setdefault("SEARXNG_URL_PATH", "/searxng")
    normalized.setdefault("SEARXNG_BASE_URL", "http://127.0.0.1/searxng/")
    normalized.setdefault("SEARXNG_PUBLIC_URL", normalized.get("SEARXNG_BASE_URL", "http://127.0.0.1/searxng/"))
    normalized.setdefault("SEARXNG_INSTANCE_NAME", "SearXNG")
    normalized.setdefault("SEARXNG_SAFE_SEARCH", "2")
    normalized.setdefault("SEARXNG_AUTOCOMPLETE", "duckduckgo")
    normalized.setdefault("SEARXNG_FORMATS", "html,json")
    normalized.setdefault("SEARXNG_LIMITER", "false")
    normalized.setdefault("SEARXNG_IMAGE_PROXY", "true")
    normalized.setdefault("SEARXNG_SECRET", "")
    normalized.setdefault("SEARXNG_VALKEY_URL", "valkey://localhost:6379/0")
    normalized.setdefault("SEARXNG_HOME", "/usr/local/searxng")
    normalized.setdefault("SEARXNG_SETTINGS_PATH", "/etc/searxng/settings.yml")
    normalized.setdefault("SEARXNG_UWSGI_INI", "/etc/uwsgi/apps-available/searxng.ini")
    normalized.setdefault("SEARXNG_UWSGI_SOCKET", "/usr/local/searxng/run/socket")
    normalized.setdefault("SEARXNG_NGINX_CONF", "/etc/nginx/default.apps-available/searxng.conf")
    normalized.setdefault("PLAYWRIGHT_ENABLED", "on")
    normalized.setdefault("PLAYWRIGHT_HOST", "0.0.0.0")
    normalized.setdefault("PLAYWRIGHT_PORT", "3001")
    normalized.setdefault("PLAYWRIGHT_UPSTREAM_PORT", "13001")
    normalized.setdefault("PLAYWRIGHT_URL_PATH", "/playwright")
    normalized.setdefault("PLAYWRIGHT_PUBLIC_WS_URL", "ws://127.0.0.1/playwright/")
    normalized.setdefault("PLAYWRIGHT_PUBLIC_HTTP_URL", "http://127.0.0.1/playwright/")
    normalized.setdefault("PLAYWRIGHT_BROWSER", "chromium")
    normalized.setdefault("PLAYWRIGHT_INSTALL_BROWSERS", "on")
    normalized.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(core.STACK_DIR / "playwright" / "browsers"))
    normalized.setdefault("PLAYWRIGHT_NODE_ENV", "production")
    normalized.setdefault("PLAYWRIGHT_NGINX_CONF", "/etc/nginx/default.apps-available/playwright.conf")
    shared_transcript_model = normalized.get("TRANSCRIPT_LOCAL_MODEL")
    shared_transcript_legacy = normalized.get("TRANSCRIPT_LOCAL_MODEL_SIZE", "large-v3")
    normalized.setdefault("PARAKEET_V3_BACKEND_TYPE", "upstream")
    normalized.setdefault("WHISPERKIT_LARGE_V3_BACKEND_TYPE", "upstream")
    normalized.setdefault(
        "PARAKEET_V3_LOCAL_MODEL",
        shared_transcript_model
        or f"preset:{normalized.get('PARAKEET_V3_LOCAL_MODEL_SIZE', shared_transcript_legacy)}",
    )
    normalized.setdefault(
        "WHISPERKIT_LARGE_V3_LOCAL_MODEL",
        shared_transcript_model
        or f"preset:{normalized.get('WHISPERKIT_LARGE_V3_LOCAL_MODEL_SIZE', shared_transcript_legacy)}",
    )
    normalized.setdefault("PARAKEET_V3_STREAM_OUTPUT_ENABLED", "off")
    normalized.setdefault("PARAKEET_V3_STREAM_OUTPUT_TARGET", "")
    normalized.setdefault("PARAKEET_V3_STREAM_OUTPUT_FORMAT", "webhook")
    normalized.setdefault("PARAKEET_V3_SPEAKER_DETECTION", "off")
    normalized.setdefault("PARAKEET_V3_SPEAKER_MODE", "auto")
    normalized.setdefault("PARAKEET_V3_SPEAKER_COUNT", "2")
    normalized.setdefault("WHISPERKIT_LARGE_V3_STREAM_OUTPUT_ENABLED", "off")
    normalized.setdefault("WHISPERKIT_LARGE_V3_STREAM_OUTPUT_TARGET", "")
    normalized.setdefault("WHISPERKIT_LARGE_V3_STREAM_OUTPUT_FORMAT", "webhook")
    normalized.setdefault("WHISPERKIT_LARGE_V3_SPEAKER_DETECTION", "off")
    normalized.setdefault("WHISPERKIT_LARGE_V3_SPEAKER_MODE", "auto")
    normalized.setdefault("WHISPERKIT_LARGE_V3_SPEAKER_COUNT", "2")
    return normalized

def read_env_raw() -> dict:
    """Exactly the keys the env file carries, with no backfill or defaults.

    `read_env` answers "what is the configuration"; this answers "what is
    written down". Only the deprecation report needs the difference, which is
    precisely the set of legacy keys still on disk.
    """
    env = {}
    try:
        with open(core.CONFIG_FILE) as f:
            for line in f:
                stripped = line.strip()
                if stripped and not stripped.startswith('#') and '=' in stripped:
                    key, _, value = stripped.partition('=')
                    value = value.strip()
                    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                        value = value[1:-1]
                    env[key.strip()] = value
    except FileNotFoundError:
        pass
    return env


def read_env() -> dict:
    """Parse env file and return non-commented key=value pairs."""
    return normalize_env_keys(read_env_raw())


def _quote_env_value(value) -> str:
    text = "" if value is None else str(value)
    if text == "":
        return '""'
    if re.fullmatch(r'[A-Za-z0-9_./,:@%+-]+', text):
        return text
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def update_env_values(updates: dict):
    """Update key=value lines in the env file, preserving all comments."""
    updates = normalize_config_updates(updates)
    with open(core.CONFIG_FILE, 'r') as f:
        content = f.read()
    for key, value in updates.items():
        rendered = _quote_env_value(value)
        aliases = [key] + NEW_ENV_KEY_LEGACY_ALIASES.get(key, [])
        replaced = False
        for alias in aliases:
            pattern = re.compile(r'^' + re.escape(alias) + r'=.*$', re.MULTILINE)
            if pattern.search(content):
                if not replaced:
                    content = pattern.sub(f'{key}={rendered}', content, count=1)
                    replaced = True
                else:
                    content = pattern.sub('', content)
        if not replaced:
            content += f'\n{key}={rendered}\n'
    content = re.sub(r'\n{3,}', '\n\n', content)
    with open(core.CONFIG_FILE, 'w') as f:
        f.write(content)


def normalize_config_updates(updates: dict) -> dict:
    normalized = {}
    for key, value in updates.items():
        target = LEGACY_ENV_KEY_MAP.get(key, key)
        if target in normalized and key in LEGACY_ENV_KEY_MAP:
            continue
        normalized[target] = value
    return normalized


def allowed_config_keys(env: dict | None = None) -> set[str]:
    """Keys that may be changed by the config APIs.

    CONFIG_FIELDS is the UI registry, but saved configs can contain settings
    that were added to llm-stack.env before they were given explicit UI controls.
    Allowing current env keys keeps saved config apply from silently dropping
    those values.
    """
    keys = {f["key"] for f in CONFIG_FIELDS}
    keys.update(RESTART_HINTS.keys())
    keys.update(LEGACY_ENV_KEY_MAP.keys())
    keys.update(LEGACY_ENV_KEY_MAP.values())
    keys.update((env or read_env()).keys())
    return keys


def filter_config_updates(updates: dict, env: dict | None = None) -> dict:
    if not isinstance(updates, dict):
        return {}
    allowed = allowed_config_keys(env)
    normalized = normalize_config_updates(updates)
    return {
        key: "" if value is None else str(value)
        for key, value in normalized.items()
        if key in allowed and (value is None or isinstance(value, (str, int, float, bool)))
    }


def config_form_snapshot(values: dict, env: dict | None = None) -> dict:
    """Exact UI form values from a saved profile, filtered to valid config keys."""
    return filter_config_updates(values, env)
