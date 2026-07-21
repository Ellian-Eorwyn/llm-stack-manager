#!/usr/bin/env python3
"""
LLM Stack Manager — Flask web UI for managing llama.cpp services.

Runs as root (via systemd) so it can call systemctl and scripts directly.
All paths are resolved relative to this file's location, so the stack can
live anywhere on disk.
"""

import json
import os
import re
import shlex
import socket
import subprocess
import threading
import time
import traceback
import uuid
import base64
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from urllib import error as urlerror
from urllib import request as urlrequest
from urllib.parse import quote, unquote, urlparse

from flask import Flask, Response, jsonify, render_template, request, send_file, stream_with_context
from flask import has_request_context
import sys
try:
    import grp
    import pwd
except ImportError:
    grp = None
    pwd = None

class ServiceManager:
    IS_MAC = sys.platform == 'darwin'

    @classmethod
    def run_cmd(cls, cmd, timeout=30):
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    @classmethod
    def is_active(cls, name: str) -> bool:
        if cls.IS_MAC:
            r = cls.run_cmd(["launchctl", "list", f"com.llmstack.{name}"], timeout=5)
            if r.returncode != 0:
                return False
            try:
                import json
                data = json.loads(r.stdout.strip())
                return int(data.get("PID", 0)) > 0
            except Exception:
                return False
        else:
            r = cls.run_cmd(["systemctl", "is-active", name], timeout=5)
            return r.stdout.strip() == "active"

    @classmethod
    def start(cls, name: str, timeout=30) -> subprocess.CompletedProcess:
        if cls.IS_MAC:
            label = f"com.llmstack.{name}"
            plist = f"/Library/LaunchDaemons/{label}.plist"
            cls.run_cmd(["launchctl", "bootout", f"system/{label}"])
            return cls.run_cmd(["launchctl", "bootstrap", "system", plist], timeout=timeout)
        else:
            return cls.run_cmd(["systemctl", "start", name], timeout=timeout)

    @classmethod
    def stop(cls, name: str, timeout=30) -> subprocess.CompletedProcess:
        if cls.IS_MAC:
            label = f"com.llmstack.{name}"
            return cls.run_cmd(["launchctl", "bootout", f"system/{label}"], timeout=timeout)
        else:
            return cls.run_cmd(["systemctl", "stop", name], timeout=timeout)

    @classmethod
    def restart(cls, name: str, timeout=120) -> tuple[int, str]:
        if cls.IS_MAC:
            cls.stop(name, timeout=timeout)
            r = cls.start(name, timeout=timeout)
            return r.returncode, (r.stdout + r.stderr).strip()
        else:
            r = cls.run_cmd(["systemctl", "restart", name], timeout=timeout)
            return r.returncode, (r.stdout + r.stderr).strip()

    @classmethod
    def action(cls, act: str, name: str, timeout=30) -> tuple[int, str]:
        if act == "start":
            r = cls.start(name, timeout=timeout)
            return r.returncode, (r.stdout + r.stderr).strip()
        elif act == "stop":
            r = cls.stop(name, timeout=timeout)
            return r.returncode, (r.stdout + r.stderr).strip()
        elif act == "restart":
            return cls.restart(name, timeout=timeout)
        else:
            return 1, "unsupported action"

    @classmethod
    def get_pid(cls, name: str) -> int:
        if cls.IS_MAC:
            label = f"com.llmstack.{name}"
            r = cls.run_cmd(["launchctl", "list", label], timeout=2)
            if r.returncode == 0:
                try:
                    import json
                    return int(json.loads(r.stdout.strip()).get("PID", 0))
                except Exception:
                    return 0
            return 0
        else:
            r = cls.run_cmd(["systemctl", "show", name, "--property=MainPID", "--value"], timeout=2)
            try:
                return int((r.stdout or "0").strip() or "0")
            except Exception:
                return 0

    @classmethod
    def is_installed(cls, name: str) -> bool:
        if cls.IS_MAC:
            from pathlib import Path
            plist = Path(f"/Library/LaunchDaemons/com.llmstack.{name}.plist")
            return plist.exists()
        else:
            r = cls.run_cmd(["systemctl", "show", name, "--property=LoadState", "--value"], timeout=5)
            return r.returncode == 0 and r.stdout.strip() != "not-found"

app = Flask(__name__)

STACK_DIR   = Path(__file__).resolve().parent.parent
CONFIG_FILE = STACK_DIR / "config" / "llm-stack.env"
SCRIPTS_DIR = STACK_DIR / "scripts"
MODELS_DIR  = STACK_DIR / "models"
TRANSCRIPTION_MODELS_DIR = MODELS_DIR / "transcription"
CUSTOM_MODELS_FILE = STACK_DIR / "config" / "custom-models.json"
CUSTOM_MODEL_ARG_PRESETS_FILE = STACK_DIR / "config" / "custom-model-arg-presets.json"
SAVED_CONFIGS_DIR  = STACK_DIR / "config" / "saved"
DEFAULT_SAVED_CONFIG_FILE = STACK_DIR / "config" / "default-saved-config"
CHAT_TEMPLATES_DIR = STACK_DIR / "config" / "chat-templates"
CHAT_TEMPLATES_META_FILE = CHAT_TEMPLATES_DIR / "templates.json"
TTS_CONFIG_FILE    = STACK_DIR / "config" / "tts-backends.json"
TTS_STATE_FILE     = STACK_DIR / "config" / "tts-state.json"
LOGS_DIR           = STACK_DIR / "logs"
GRAPHITI_EXPORTS_DIR = STACK_DIR / "exports" / "graphiti"
HF_ALLOWED_HOSTS = {"huggingface.co", "www.huggingface.co", "hf.co"}
HF_DOWNLOAD_JOBS: dict[str, dict] = {}
HF_DOWNLOAD_JOBS_LOCK = threading.Lock()
TRANSCRIPTION_MODEL_PRESETS = [
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large-v1",
    "large-v2",
    "large-v3",
    "distil-large-v2",
    "distil-large-v3",
    "turbo",
]
TRANSCRIPTION_ENGINES = [
    {"id": "parakeet-v3", "label": "Parakeet v3", "env_prefix": "PARAKEET_V3"},
    {"id": "whisperkit-large-v3", "label": "WhisperKit Large v3", "env_prefix": "WHISPERKIT_LARGE_V3"},
]
TRANSCRIPTION_ENGINE_BY_ID = {item["id"]: item for item in TRANSCRIPTION_ENGINES}
LLAMA_KV_CACHE_OPTIONS = ["q8_0", "f16", "f32", "bf16", "q5_0", "q5_1", "q4_0", "q4_1", "iq4_nl"]
BEE_KV_CACHE_OPTIONS = [
    {"value": "f32", "label": "f32 - unquantized 32-bit float"},
    {"value": "f16", "label": "f16 - unquantized 16-bit float"},
    {"value": "bf16", "label": "bf16 - unquantized bfloat16"},
    {"value": "q8_0", "label": "q8_0 - regular 8-bit KV quant"},
    {"value": "q6_0", "label": "q6_0 - regular 6-bit KV quant"},
    {"value": "q5_0", "label": "q5_0 - regular 5-bit KV quant"},
    {"value": "q5_1", "label": "q5_1 - regular 5-bit KV quant"},
    {"value": "q4_0", "label": "q4_0 - regular 4-bit KV quant"},
    {"value": "q4_1", "label": "q4_1 - regular 4-bit KV quant"},
    {"value": "iq4_nl", "label": "iq4_nl - regular importance 4-bit KV quant"},
    {"value": "turbo2", "label": "turbo2 - TurboQuant 2-bit"},
    {"value": "turbo3", "label": "turbo3 - TurboQuant 3-bit"},
    {"value": "turbo4", "label": "turbo4 - TurboQuant 4-bit"},
    {"value": "turbo2_tcq", "label": "turbo2_tcq - TurboQuant TCQ 2-bit"},
    {"value": "turbo3_tcq", "label": "turbo3_tcq - TurboQuant TCQ 3-bit"},
]

BUILTIN_CUSTOM_MODEL_ARG_PRESETS = {
    "qwen3.6": [
        "--chat-template-kwargs '{\"preserve_thinking\": true}'",
        "--jinja",
    ],
}

BUILTIN_CHAT_VARIANTS = [
    {
        "id": "dense",
        "service": "chat-backend-dense",
        "default_label": "Primary Backend",
        "default_desc": "Primary model preset · shared proxy",
        "label_key": "CHAT_PRIMARY_LABEL",
    },
    {
        "id": "moe",
        "service": "chat-backend-moe",
        "default_label": "Secondary Backend",
        "default_desc": "Secondary model preset · shared proxy",
        "label_key": "CHAT_SECONDARY_LABEL",
    },
]
BUILTIN_CHAT_VARIANT_IDS = {item["id"] for item in BUILTIN_CHAT_VARIANTS}
BUILTIN_CHAT_VARIANT_BY_ID = {item["id"]: item for item in BUILTIN_CHAT_VARIANTS}
BUILTIN_CHAT_VARIANT_BY_SERVICE = {item["service"]: item for item in BUILTIN_CHAT_VARIANTS}
LEGACY_ENV_KEY_MAP = {
    "CHAT_MODEL_27B_PATH": "CHAT_PRIMARY_MODEL_PATH",
    "CHAT_MMPROJ_27B_PATH": "CHAT_PRIMARY_MMPROJ_PATH",
    "CHAT_27B_CTX_SIZE": "CHAT_PRIMARY_CTX_SIZE",
    "CHAT_MODEL_35B_PATH": "CHAT_SECONDARY_MODEL_PATH",
    "CHAT_MMPROJ_35B_PATH": "CHAT_SECONDARY_MMPROJ_PATH",
    "CHAT_35B_CTX_SIZE": "CHAT_SECONDARY_CTX_SIZE",
    "CHAT_DENSE_LABEL": "CHAT_PRIMARY_LABEL",
    "CHAT_DENSE_MODEL_NAME": "CHAT_PRIMARY_MODEL_NAME",
    "CHAT_DENSE_MODEL_PATH": "CHAT_PRIMARY_MODEL_PATH",
    "CHAT_DENSE_MMPROJ_PATH": "CHAT_PRIMARY_MMPROJ_PATH",
    "CHAT_DENSE_CTX_SIZE": "CHAT_PRIMARY_CTX_SIZE",
    "CHAT_MOE_LABEL": "CHAT_SECONDARY_LABEL",
    "CHAT_MOE_MODEL_NAME": "CHAT_SECONDARY_MODEL_NAME",
    "CHAT_MOE_MODEL_PATH": "CHAT_SECONDARY_MODEL_PATH",
    "CHAT_MOE_MMPROJ_PATH": "CHAT_SECONDARY_MMPROJ_PATH",
    "CHAT_MOE_CTX_SIZE": "CHAT_SECONDARY_CTX_SIZE",
}
NEW_ENV_KEY_LEGACY_ALIASES = defaultdict(list)
for legacy_key, new_key in LEGACY_ENV_KEY_MAP.items():
    NEW_ENV_KEY_LEGACY_ALIASES[new_key].append(legacy_key)

SHARED_CHAT_BACKEND_RESTART = ["chat-backend-dense", "chat-backend-moe", "chat-backend", "chat-backend2"]
TTS_BACKEND_SERVICES = []
TTS_MANAGED_SERVICES = []
TRANSCRIPT_MANAGED_SERVICE = ""
SERVICES = [
    {"group": "chat",      "name": "chat-backend-dense", "label": "Primary Backend", "desc": "Primary model backend", "ports": "8010 internal / llms:8010", "config_section": "Primary Backend"},
    {"group": "chat",      "name": "chat-proxy",       "label": "Primary Proxy",   "desc": "Routes primary think/chat/code", "ports": "8003 / 8004 / 8008 / 8012"},
    {"group": "chat",      "name": "chat-backend2",    "label": "Secondary Backend", "desc": "Secondary model backend",  "ports": "8020 internal / llms:8020", "config_section": "Secondary Backend"},
    {"group": "chat",      "name": "chat-proxy2",      "label": "Secondary Proxy", "desc": "Routes secondary think/chat/code", "ports": "8103 / 8104 / 8108 / 8112"},
    {"group": "auxiliary", "name": "embed",        "label": "Embedding",    "desc": "Embedding model",                   "ports": "8005", "config_section": "Embedding"},
    {"group": "auxiliary", "name": "embed2",       "label": "Embedding 2",  "desc": "Second embedding backend",          "ports": "8011", "config_section": "Embedding 2"},
    {"group": "auxiliary", "name": "rerank",         "label": "Reranker",     "desc": "Reranker model",                    "ports": "8006", "config_section": "Reranker"},
    {"group": "auxiliary", "name": "task",             "label": "Task",         "desc": "Small fast task model",             "ports": "8007", "config_section": "Task Model"},
    {"group": "auxiliary", "name": "ocr",              "label": "OCR Model",    "desc": "GLM-OCR llama.cpp model backend",      "ports": "8009", "config_section": "OCR"},
    {"group": "auxiliary", "name": "glmocr-sdk",       "label": "OCR SDK",      "desc": "Local GLM-OCR layout/PDF parser",       "ports": "5002", "config_section": "GLM-OCR SDK"},
    {"group": "auxiliary", "name": "honcho-api",       "label": "Honcho API",   "desc": "Local Honcho memory API",           "ports": "8090"},
    {"group": "auxiliary", "name": "honcho-deriver",   "label": "Honcho Worker", "desc": "Local Honcho background deriver",   "ports": "worker"},
    {"group": "auxiliary", "name": "searxng",          "label": "SearXNG",      "desc": "Local metasearch engine via uWSGI/nginx", "ports": "/searxng", "config_section": "SearXNG"},
    {"group": "auxiliary", "name": "playwright-server", "label": "Playwright",   "desc": "Remote browser automation WebSocket server", "ports": "3001", "config_section": "Playwright"},
]

LLAMACPP_MODEL_SERVICES = [
    "chat-backend",
    "chat-backend2",
    "chat-backend-dense",
    "chat-backend-moe",
    "embed",
    "embed2",
    "rerank",
    "task",
    "ocr",
]
LLAMACPP_PROXY_SERVICE = "chat-proxy"
CORE_CONFIG_SECTIONS = {
    "Chat Templates",
    "Primary Backend",
    "Secondary Backend",
    "Shared Backend",
    "Task Model",
    "Thinking Endpoint",
    "Instruct Endpoint",
    "Coding Endpoint",
    "Embedding",
    "Embedding 2",
    "Reranker",
    "OCR",
    "GLM-OCR SDK",
    "SearXNG",
    "Playwright",
    "Ports",
}

CODE_TO_CHAT_MIRRORS = {
    "CODE_CTX_SIZE":            ["CHAT_CTX_SIZE", "CHAT_DENSE_CTX_SIZE", "CHAT_MOE_CTX_SIZE"],
    "CODE_N_PARALLEL":          "CHAT_N_PARALLEL",
    "CODE_THREADS":             "CHAT_THREADS",
    "CODE_THREADS_BATCH":       "CHAT_THREADS_BATCH",
    "CODE_N_GPU_LAYERS":        "CHAT_N_GPU_LAYERS",
    "CODE_TENSOR_SPLIT":        "CHAT_TENSOR_SPLIT",
    "CODE_SPLIT_MODE":          "CHAT_SPLIT_MODE",
    "CODE_FLASH_ATTN":          "CHAT_FLASH_ATTN",
    "CODE_CACHE_TYPE_K":        "CHAT_CACHE_TYPE_K",
    "CODE_CACHE_TYPE_V":        "CHAT_CACHE_TYPE_V",
    "CODE_BATCH_SIZE":          "CHAT_BATCH_SIZE",
    "CODE_UBATCH_SIZE":         "CHAT_UBATCH_SIZE",
    "CODE_NO_MMAP":             "CHAT_NO_MMAP",
    "CODE_MLOCK":               "CHAT_MLOCK",
    "CODE_GPU_VISIBLE_DEVICES": "CHAT_GPU_VISIBLE_DEVICES",
    "CODE_REASONING_FORMAT":    "CHAT_REASONING_FORMAT",
    "CODE_FIT":                 "CHAT_FIT",
}

# ---------------------------------------------------------------------------
# Config fields exposed in the UI
# ---------------------------------------------------------------------------
LLAMA_SPEC_METHOD_OPTIONS = ["off", "draft-model", "draft-simple", "draft-eagle3", "draft-mtp", "draft-dflash", "ngram-cache", "ngram-simple", "ngram-map-k", "ngram-map-k4v", "ngram-mod"]
LLAMA_CACHE_IDLE_OPTIONS = ["on", "off"]

CONFIG_FIELDS = [
    {"section": "Chat Templates", "key": "CHAT_TEMPLATE_MANAGER", "label": "Template Manager", "type": "template_manager", "hint": "Create and edit reusable llama.cpp Jinja chat templates"},

    # Secondary Backend
    {"section": "Secondary Backend", "key": "CHAT2_LABEL",                 "label": "Backend Label",           "type": "text",   "hint": "UI label for the secondary backend slot"},
    {"section": "Secondary Backend", "key": "CHAT2_MODEL_NAME",            "label": "Model Alias",             "type": "text",   "hint": "llama.cpp alias for the secondary backend"},
    {"section": "Secondary Backend", "key": "CHAT2_MODEL_PATH",            "label": "Model Path",              "type": "path"},
    {"section": "Secondary Backend", "key": "CHAT2_MMPROJ_PATH",           "label": "MMProj Path",             "type": "path"},
    {"section": "Secondary Backend", "key": "CHAT2_CTX_SIZE",              "label": "Context Size",            "type": "number"},
    {"section": "Secondary Backend", "key": "CHAT2_BACKEND_PORT",          "label": "Backend Port",            "type": "number"},
    {"section": "Secondary Backend", "key": "THINK2_PORT",                 "label": "Think Port",              "type": "number"},
    {"section": "Secondary Backend", "key": "NOTHINK2_PORT",               "label": "Chat Port",               "type": "number"},
    {"section": "Secondary Backend", "key": "CODE2_PORT",                  "label": "Code Port",               "type": "number"},
    {"section": "Secondary Backend", "key": "AGGREGATE2_ENABLED",          "label": "Aggregate Proxy",         "type": "select", "options": ["on", "off"], "hint": "Single model-routed endpoint exposing think, chat, and code for the secondary backend"},
    {"section": "Secondary Backend", "key": "AGGREGATE2_PORT",             "label": "Aggregate Port",          "type": "number"},
    {"section": "Secondary Backend", "key": "THINK2_MODEL_NAME",           "label": "Think Alias",             "type": "text", "hint": "Advertised model id on the secondary aggregate and think port"},
    {"section": "Secondary Backend", "key": "NOTHINK2_MODEL_NAME",         "label": "Chat Alias",              "type": "text", "hint": "Advertised model id on the secondary aggregate and chat port"},
    {"section": "Secondary Backend", "key": "CODE2_MODEL_NAME",            "label": "Code Alias",              "type": "text", "hint": "Advertised model id on the secondary aggregate and code port"},
    {"section": "Secondary Backend", "key": "CHAT2_N_PARALLEL",            "label": "Parallel Slots",          "type": "number"},
    {"section": "Secondary Backend", "key": "CHAT2_THREADS",               "label": "CPU Threads",             "type": "number", "hint": "llama.cpp --threads for generation; -1 lets llama.cpp choose"},
    {"section": "Secondary Backend", "key": "CHAT2_THREADS_BATCH",         "label": "CPU Batch Threads",       "type": "number", "hint": "llama.cpp --threads-batch for prompt/batch processing; -1 follows --threads"},
    {"section": "Secondary Backend", "key": "CHAT2_N_GPU_LAYERS",          "label": "GPU Layers (−1=all)",     "type": "number"},
    {"section": "Secondary Backend", "key": "CHAT2_MAIN_GPU",              "label": "Main GPU Index",          "type": "number", "hint": "GPU index (within visible devices) for split-mode=none, or KV/intermediate buffers with row split"},
    {"section": "Secondary Backend", "key": "CHAT2_DEVICE",                "label": "Main/Draft Offload Devices", "type": "text", "hint": "Optional llama.cpp --device override; use --list-devices names like CUDA0,CUDA1 or none"},
    {"section": "Secondary Backend", "key": "CHAT2_TENSOR_SPLIT",          "label": "Tensor Split",            "type": "text",   "hint": "e.g. 1,1"},
    {"section": "Secondary Backend", "key": "CHAT2_SPLIT_MODE",            "label": "Split Mode",              "type": "select", "options": ["none", "layer", "row", "tensor"], "hint": "none=model on one GPU, layer=layer sharding, row=row sharding, tensor=parallel tensor+KV sharding"},
    {"section": "Secondary Backend", "key": "CHAT2_KV_OFFLOAD",            "label": "KV Offload",              "type": "select", "options": ["on", "off"], "hint": "Controls --kv-offload / --no-kv-offload"},
    {"section": "Secondary Backend", "key": "CHAT2_OP_OFFLOAD",            "label": "Host Op Offload",         "type": "select", "options": ["on", "off"], "hint": "Controls --op-offload / --no-op-offload for host tensor ops"},
    {"section": "Secondary Backend", "key": "CHAT2_MMPROJ_OFFLOAD",        "label": "MMProj Offload",          "type": "select", "options": ["on", "off"], "hint": "Controls --mmproj-offload / --no-mmproj-offload when an MMProj is loaded"},
    {"section": "Secondary Backend", "key": "CHAT2_FLASH_ATTN",            "label": "Flash Attention",         "type": "select", "options": ["on", "off", "auto"]},
    {"section": "Secondary Backend", "key": "CHAT2_CACHE_TYPE_K",          "label": "KV Cache Key Type",       "type": "select", "options": LLAMA_KV_CACHE_OPTIONS},
    {"section": "Secondary Backend", "key": "CHAT2_CACHE_TYPE_V",          "label": "KV Cache Value Type",     "type": "select", "options": LLAMA_KV_CACHE_OPTIONS},
    {"section": "Secondary Backend", "key": "CHAT2_CACHE_RAM",             "label": "Prompt Cache RAM",        "type": "number", "hint": "llama.cpp --cache-ram in MiB; 0 disables server prompt-cache storage"},
    {"section": "Secondary Backend", "key": "CHAT2_CTX_CHECKPOINTS",       "label": "Context Checkpoints",     "type": "number", "hint": "llama.cpp --ctx-checkpoints; 0 disables context checkpoint creation"},
    {"section": "Secondary Backend", "key": "CHAT2_SWA_FULL",              "label": "Full SWA KV Cache",       "type": "select", "options": ["off", "on"], "hint": "Adds llama.cpp --swa-full for SWA models; uses more KV VRAM but improves prompt-cache reuse"},
    {"section": "Secondary Backend", "key": "CHAT2_BATCH_SIZE",            "label": "Batch Size",              "type": "number", "hint": "Prefill batch (default 2048)"},
    {"section": "Secondary Backend", "key": "CHAT2_UBATCH_SIZE",           "label": "Micro-Batch Size",        "type": "number", "hint": "Physical sub-batch (default 512)"},
    {"section": "Secondary Backend", "key": "CHAT2_NO_MMAP",               "label": "Disable mmap",            "type": "select", "options": ["false", "true"]},
    {"section": "Secondary Backend", "key": "CHAT2_MLOCK",                 "label": "Lock Memory",             "type": "select", "options": ["false", "true"]},
    {"section": "Secondary Backend", "key": "CHAT2_GPU_VISIBLE_DEVICES",   "label": "GPU Devices",             "type": "text",   "hint": "e.g. 0,1"},
    {"section": "Secondary Backend", "key": "CHAT2_JINJA",                 "label": "Backend Jinja Support",   "type": "select", "options": ["off", "on"], "hint": "Enables --jinja on the secondary backend so proxy ports can expose tool calling"},
    {"section": "Secondary Backend", "key": "CHAT2_TEMPLATE_ID",           "label": "Effective Chat Template", "type": "chat_template", "hint": "Custom Jinja template file passed to the secondary backend; model default leaves GGUF metadata unchanged"},
    {"section": "Secondary Backend", "key": "CHAT2_FIT",                   "label": "Auto-Fit to VRAM",        "type": "select", "options": ["on", "off"], "hint": "When on, may reduce context size to fit in VRAM"},
    {"section": "Secondary Backend", "key": "CHAT2_FIT_TARGET",            "label": "Fit Target MiB",          "type": "text",   "hint": "llama.cpp --fit-target per-device margin, e.g. 1024 or 1024,2048; empty uses llama.cpp default"},
    {"section": "Secondary Backend", "key": "CHAT2_FIT_CTX",               "label": "Minimum Fit Context",     "type": "number", "hint": "llama.cpp --fit-ctx minimum context when auto-fit adjusts settings"},
    {"section": "Secondary Backend", "key": "CHAT2_CACHE_IDLE_SLOTS",      "label": "Cache Idle Slots",        "type": "select", "options": LLAMA_CACHE_IDLE_OPTIONS, "hint": "Controls --cache-idle-slots / --no-cache-idle-slots"},
    {"section": "Secondary Backend", "key": "CHAT2_CACHE_REUSE",           "label": "Cache Reuse Chunk",       "type": "number", "hint": "llama.cpp --cache-reuse minimum chunk size; 0 leaves llama.cpp default"},
    {"section": "Secondary Backend", "key": "CHAT2_SPEC_METHOD",           "label": "Speculative Method",      "type": "select", "options": LLAMA_SPEC_METHOD_OPTIONS, "hint": "Base llama.cpp mode. draft-dflash requires an upstream DFlash draft GGUF with general.architecture=dflash;"},
    {"section": "Secondary Backend", "key": "CHAT2_SPEC_NGRAM_MOD",        "label": "N-Gram Mod Assist",       "type": "select", "options": ["off", "on"], "hint": "When on, appends ngram-mod to MTP-style spec types, e.g. draft-mtp,ngram-mod"},
    {"section": "Secondary Backend", "key": "CHAT2_SPEC_DRAFT_MODEL_PATH", "label": "Draft Model Path",        "type": "path",   "hint": "Smaller GGUF used as the speculative draft model"},
    {"section": "Secondary Backend", "key": "CHAT2_SPEC_DRAFT_N_GPU_LAYERS", "label": "Draft GPU Layers",      "type": "text",   "hint": "Draft-model --spec-draft-ngl value: auto, all, or an exact layer count"},
    {"section": "Secondary Backend", "key": "CHAT2_SPEC_DRAFT_DEVICES",    "label": "Draft Devices",           "type": "text",   "hint": "Optional --spec-draft-device override, e.g. 0,1 or none"},
    {"section": "Secondary Backend", "key": "CHAT2_SPEC_DRAFT_TYPE_K",     "label": "Draft KV Key Type",       "type": "select", "options": LLAMA_KV_CACHE_OPTIONS, "hint": "llama.cpp --spec-draft-type-k"},
    {"section": "Secondary Backend", "key": "CHAT2_SPEC_DRAFT_TYPE_V",     "label": "Draft KV Value Type",     "type": "select", "options": LLAMA_KV_CACHE_OPTIONS, "hint": "llama.cpp --spec-draft-type-v"},
    {"section": "Secondary Backend", "key": "CHAT2_SPEC_DRAFT_N_MAX",      "label": "Draft Max Tokens",        "type": "number", "hint": "llama.cpp --spec-draft-n-max (recommended 6 for MTP)"},
    {"section": "Secondary Backend", "key": "CHAT2_SPEC_DRAFT_N_MIN",      "label": "Draft Min Tokens",        "type": "number", "hint": "llama.cpp --spec-draft-n-min (default 0)"},
    {"section": "Secondary Backend", "key": "CHAT2_SPEC_DRAFT_P_MIN",      "label": "Draft Min Probability",   "type": "text",   "hint": "llama.cpp --spec-draft-p-min (default 0.75)"},
    {"section": "Secondary Backend", "key": "CHAT2_SPEC_DRAFT_P_SPLIT",    "label": "Draft Split Probability", "type": "text",   "hint": "llama.cpp --spec-draft-p-split (default 0.10)"},
    {"section": "Secondary Backend", "key": "CHAT2_SPEC_NGRAM_MOD_N_MATCH","label": "N-Gram Match Tokens",     "type": "number", "hint": "llama.cpp --spec-ngram-mod-n-match (default 24)"},
    {"section": "Secondary Backend", "key": "CHAT2_SPEC_NGRAM_MOD_N_MIN",  "label": "N-Gram Min Tokens",       "type": "number", "hint": "llama.cpp --spec-ngram-mod-n-min (default 48)"},
    {"section": "Secondary Backend", "key": "CHAT2_SPEC_NGRAM_MOD_N_MAX",  "label": "N-Gram Max Tokens",       "type": "number", "hint": "llama.cpp --spec-ngram-mod-n-max (default 64)"},
    {"section": "Secondary Backend", "key": "CHAT2_SPEC_NGRAM_SIZE_N",     "label": "N-Gram Lookup Size",      "type": "number", "hint": "llama.cpp --spec-ngram-*-size-n for ngram-simple/map modes"},
    {"section": "Secondary Backend", "key": "CHAT2_SPEC_NGRAM_SIZE_M",     "label": "N-Gram Draft Size",       "type": "number", "hint": "llama.cpp --spec-ngram-*-size-m for ngram-simple/map modes"},
    {"section": "Secondary Backend", "key": "CHAT2_SPEC_NGRAM_MIN_HITS",   "label": "N-Gram Min Hits",         "type": "number", "hint": "llama.cpp --spec-ngram-*-min-hits for ngram-simple/map modes"},
    {"section": "Secondary Backend", "key": "CHAT2_CUSTOM_ARGS_JSON",      "label": "Custom Arguments",        "type": "custom_args", "hint": "Extra llama.cpp flags applied to the secondary backend"},
    # Shared Backend
    {"section": "Shared Backend", "key": "CHAT_DENSE_LABEL",           "label": "Dense Slot Label",       "type": "text",   "hint": "UI label for the dense preset button/card"},
    {"section": "Shared Backend", "key": "CHAT_DENSE_MODEL_NAME",      "label": "Dense Model Alias",      "type": "text",   "hint": "llama.cpp alias for the dense preset"},
    {"section": "Shared Backend", "key": "CHAT_DENSE_MODEL_PATH",      "label": "Dense Model Path",       "type": "path"},
    {"section": "Shared Backend", "key": "CHAT_DENSE_MMPROJ_PATH",     "label": "Dense MMProj Path",      "type": "path"},
    {"section": "Shared Backend", "key": "CHAT_DENSE_CTX_SIZE",        "label": "Dense Context Size",     "type": "number"},
    {"section": "Shared Backend", "key": "CHAT_MOE_LABEL",             "label": "MoE Slot Label",         "type": "text",   "hint": "UI label for the MoE preset button/card"},
    {"section": "Shared Backend", "key": "CHAT_MOE_MODEL_NAME",        "label": "MoE Model Alias",        "type": "text",   "hint": "llama.cpp alias for the MoE preset"},
    {"section": "Shared Backend", "key": "CHAT_MOE_MODEL_PATH",        "label": "MoE Model Path",         "type": "path"},
    {"section": "Shared Backend", "key": "CHAT_MOE_MMPROJ_PATH",       "label": "MoE MMProj Path",        "type": "path"},
    {"section": "Shared Backend", "key": "CHAT_MOE_CTX_SIZE",          "label": "MoE Context Size",       "type": "number"},
    {"section": "Shared Backend", "key": "CHAT_MODEL_NAME",            "label": "Custom Backend Alias",   "type": "text",   "hint": "llama.cpp alias for the generic custom backend"},
    {"section": "Shared Backend", "key": "CHAT_N_PARALLEL",            "label": "Parallel Slots",         "type": "number"},
    {"section": "Shared Backend", "key": "CHAT_THREADS",               "label": "CPU Threads",            "type": "number", "hint": "llama.cpp --threads for generation; -1 lets llama.cpp choose"},
    {"section": "Shared Backend", "key": "CHAT_THREADS_BATCH",         "label": "CPU Batch Threads",      "type": "number", "hint": "llama.cpp --threads-batch for prompt/batch processing; -1 follows --threads"},
    {"section": "Shared Backend", "key": "CHAT_N_GPU_LAYERS",          "label": "GPU Layers (−1=all)",    "type": "number"},
    {"section": "Shared Backend", "key": "CHAT_MAIN_GPU",              "label": "Main GPU Index",         "type": "number", "hint": "GPU index (within visible devices) for split-mode=none, or KV/intermediate buffers with row split"},
    {"section": "Shared Backend", "key": "CHAT_DEVICE",                "label": "Main/Draft Offload Devices", "type": "text", "hint": "Optional llama.cpp --device override for shared backends; use --list-devices names like CUDA0,CUDA1 or none"},
    {"section": "Shared Backend", "key": "CHAT_TENSOR_SPLIT",          "label": "Tensor Split",           "type": "text",   "hint": "e.g. 1,1"},
    {"section": "Shared Backend", "key": "CHAT_SPLIT_MODE",            "label": "Split Mode",             "type": "select", "options": ["none", "layer", "row", "tensor"], "hint": "none=model on one GPU, layer=layer sharding, row=row sharding, tensor=parallel tensor+KV sharding"},
    {"section": "Shared Backend", "key": "CHAT_KV_OFFLOAD",            "label": "KV Offload",             "type": "select", "options": ["on", "off"], "hint": "Controls --kv-offload / --no-kv-offload"},
    {"section": "Shared Backend", "key": "CHAT_OP_OFFLOAD",            "label": "Host Op Offload",        "type": "select", "options": ["on", "off"], "hint": "Controls --op-offload / --no-op-offload for host tensor ops"},
    {"section": "Shared Backend", "key": "CHAT_MMPROJ_OFFLOAD",        "label": "MMProj Offload",         "type": "select", "options": ["on", "off"], "hint": "Controls --mmproj-offload / --no-mmproj-offload when an MMProj is loaded"},
    {"section": "Shared Backend", "key": "CHAT_FLASH_ATTN",            "label": "Flash Attention",        "type": "select", "options": ["on", "off", "auto"]},
    {"section": "Shared Backend", "key": "CHAT_CACHE_TYPE_K",          "label": "KV Cache Key Type",      "type": "select", "options": LLAMA_KV_CACHE_OPTIONS},
    {"section": "Shared Backend", "key": "CHAT_CACHE_TYPE_V",          "label": "KV Cache Value Type",    "type": "select", "options": LLAMA_KV_CACHE_OPTIONS},
    {"section": "Shared Backend", "key": "CHAT_CACHE_RAM",             "label": "Prompt Cache RAM",      "type": "number", "hint": "llama.cpp --cache-ram in MiB; 0 disables server prompt-cache storage"},
    {"section": "Shared Backend", "key": "CHAT_CTX_CHECKPOINTS",       "label": "Context Checkpoints",   "type": "number", "hint": "llama.cpp --ctx-checkpoints; 0 disables context checkpoint creation"},
    {"section": "Shared Backend", "key": "CHAT_SWA_FULL",              "label": "Full SWA KV Cache",     "type": "select", "options": ["off", "on"], "hint": "Adds llama.cpp --swa-full for SWA models; uses more KV VRAM but improves prompt-cache reuse"},
    {"section": "Shared Backend", "key": "CHAT_BATCH_SIZE",            "label": "Batch Size",             "type": "number", "hint": "Prefill batch (default 2048)"},
    {"section": "Shared Backend", "key": "CHAT_UBATCH_SIZE",           "label": "Micro-Batch Size",       "type": "number", "hint": "Physical sub-batch (default 512)"},
    {"section": "Shared Backend", "key": "CHAT_NO_MMAP",               "label": "Disable mmap",           "type": "select", "options": ["false", "true"]},
    {"section": "Shared Backend", "key": "CHAT_MLOCK",                 "label": "Lock Memory",            "type": "select", "options": ["false", "true"]},
    {"section": "Shared Backend", "key": "CHAT_GPU_VISIBLE_DEVICES",   "label": "GPU Devices",            "type": "text",   "hint": "e.g. 0,1"},
    {"section": "Shared Backend", "key": "CHAT_JINJA",                 "label": "Backend Jinja Support",  "type": "select", "options": ["off", "on"], "hint": "Enables --jinja on the shared backend so proxy ports can expose tool calling"},
    {"section": "Shared Backend", "key": "CHAT_TEMPLATE_ID",           "label": "Effective Chat Template", "type": "chat_template", "hint": "Custom Jinja template file passed to the shared backend; model default leaves GGUF metadata unchanged"},
    {"section": "Shared Backend", "key": "CHAT_FIT",                   "label": "Auto-Fit to VRAM",       "type": "select", "options": ["on", "off"], "hint": "When on, may reduce context size to fit in VRAM"},
    {"section": "Shared Backend", "key": "CHAT_FIT_TARGET",            "label": "Fit Target MiB",         "type": "text",   "hint": "llama.cpp --fit-target per-device margin, e.g. 1024 or 1024,2048; empty uses llama.cpp default"},
    {"section": "Shared Backend", "key": "CHAT_FIT_CTX",               "label": "Minimum Fit Context",    "type": "number", "hint": "llama.cpp --fit-ctx minimum context when auto-fit adjusts settings"},
    {"section": "Shared Backend", "key": "CHAT_CACHE_IDLE_SLOTS",      "label": "Cache Idle Slots",       "type": "select", "options": LLAMA_CACHE_IDLE_OPTIONS, "hint": "Controls --cache-idle-slots / --no-cache-idle-slots"},
    {"section": "Shared Backend", "key": "CHAT_CACHE_REUSE",           "label": "Cache Reuse Chunk",      "type": "number", "hint": "llama.cpp --cache-reuse minimum chunk size; 0 leaves llama.cpp default"},
    {"section": "Shared Backend", "key": "CHAT_SPEC_METHOD",           "label": "Speculative Method",     "type": "select", "options": LLAMA_SPEC_METHOD_OPTIONS, "hint": "Base llama.cpp mode. draft-dflash requires an upstream DFlash draft GGUF with general.architecture=dflash;"},
    {"section": "Shared Backend", "key": "CHAT_SPEC_NGRAM_MOD",        "label": "N-Gram Mod Assist",      "type": "select", "options": ["off", "on"], "hint": "When on, appends ngram-mod to MTP-style spec types, e.g. draft-mtp,ngram-mod"},
    {"section": "Shared Backend", "key": "CHAT_SPEC_DRAFT_MODEL_PATH", "label": "Draft Model Path",       "type": "path",   "hint": "Smaller GGUF used as the speculative draft model"},
    {"section": "Shared Backend", "key": "CHAT_SPEC_DRAFT_N_GPU_LAYERS", "label": "Draft GPU Layers",     "type": "text",   "hint": "Draft-model --spec-draft-ngl value: auto, all, or an exact layer count"},
    {"section": "Shared Backend", "key": "CHAT_SPEC_DRAFT_DEVICES",    "label": "Draft Devices",          "type": "text",   "hint": "Optional --spec-draft-device override, e.g. 0,1 or none"},
    {"section": "Shared Backend", "key": "CHAT_SPEC_DRAFT_TYPE_K",     "label": "Draft KV Key Type",      "type": "select", "options": LLAMA_KV_CACHE_OPTIONS, "hint": "llama.cpp --spec-draft-type-k"},
    {"section": "Shared Backend", "key": "CHAT_SPEC_DRAFT_TYPE_V",     "label": "Draft KV Value Type",    "type": "select", "options": LLAMA_KV_CACHE_OPTIONS, "hint": "llama.cpp --spec-draft-type-v"},
    {"section": "Shared Backend", "key": "CHAT_SPEC_DRAFT_N_MAX",      "label": "Draft Max Tokens",       "type": "number", "hint": "llama.cpp --spec-draft-n-max (recommended 6 for MTP)"},
    {"section": "Shared Backend", "key": "CHAT_SPEC_DRAFT_N_MIN",      "label": "Draft Min Tokens",       "type": "number", "hint": "llama.cpp --spec-draft-n-min (default 0)"},
    {"section": "Shared Backend", "key": "CHAT_SPEC_DRAFT_P_MIN",      "label": "Draft Min Probability",  "type": "text",   "hint": "llama.cpp --spec-draft-p-min (default 0.75)"},
    {"section": "Shared Backend", "key": "CHAT_SPEC_DRAFT_P_SPLIT",    "label": "Draft Split Probability","type": "text",   "hint": "llama.cpp --spec-draft-p-split (default 0.10)"},
    {"section": "Shared Backend", "key": "CHAT_SPEC_NGRAM_MOD_N_MATCH","label": "N-Gram Match Tokens",    "type": "number", "hint": "llama.cpp --spec-ngram-mod-n-match (default 24)"},
    {"section": "Shared Backend", "key": "CHAT_SPEC_NGRAM_MOD_N_MIN",  "label": "N-Gram Min Tokens",      "type": "number", "hint": "llama.cpp --spec-ngram-mod-n-min (default 48)"},
    {"section": "Shared Backend", "key": "CHAT_SPEC_NGRAM_MOD_N_MAX",  "label": "N-Gram Max Tokens",      "type": "number", "hint": "llama.cpp --spec-ngram-mod-n-max (default 64)"},
    {"section": "Shared Backend", "key": "CHAT_SPEC_NGRAM_SIZE_N",     "label": "N-Gram Lookup Size",     "type": "number", "hint": "llama.cpp --spec-ngram-*-size-n for ngram-simple/map modes"},
    {"section": "Shared Backend", "key": "CHAT_SPEC_NGRAM_SIZE_M",     "label": "N-Gram Draft Size",      "type": "number", "hint": "llama.cpp --spec-ngram-*-size-m for ngram-simple/map modes"},
    {"section": "Shared Backend", "key": "CHAT_SPEC_NGRAM_MIN_HITS",   "label": "N-Gram Min Hits",        "type": "number", "hint": "llama.cpp --spec-ngram-*-min-hits for ngram-simple/map modes"},
    {"section": "Shared Backend", "key": "CHAT_CUSTOM_ARGS_JSON",      "label": "Custom Arguments",       "type": "custom_args", "hint": "Extra llama.cpp flags applied to all shared chat backends"},
    # Task Model
    {"section": "Task Model",  "key": "TASK_MODEL_NAME",            "label": "Model Name",           "type": "text",   "hint": "Advertised on /v1/models for the task endpoint"},
    {"section": "Task Model",  "key": "TASK_MODEL_PATH",            "label": "Task Model Path",      "type": "path"},
    {"section": "Task Model",  "key": "TASK_MMPROJ_PATH",           "label": "MMProj Path",          "type": "path"},
    {"section": "Task Model",  "key": "TASK_CTX_SIZE",              "label": "Context Size",         "type": "number"},
    {"section": "Task Model",  "key": "TASK_N_PARALLEL",            "label": "Parallel Slots",       "type": "number"},
    {"section": "Task Model",  "key": "TASK_THREADS",               "label": "CPU Threads",          "type": "number", "hint": "llama.cpp --threads for generation; -1 lets llama.cpp choose"},
    {"section": "Task Model",  "key": "TASK_THREADS_BATCH",         "label": "CPU Batch Threads",    "type": "number", "hint": "llama.cpp --threads-batch for prompt/batch processing; -1 follows --threads"},
    {"section": "Task Model",  "key": "TASK_N_GPU_LAYERS",          "label": "GPU Layers (−1=all)",  "type": "number"},
    {"section": "Task Model",  "key": "TASK_MAIN_GPU",              "label": "Main GPU Index",       "type": "number", "hint": "GPU index (within visible devices) for split-mode=none, or KV/intermediate buffers with row split"},
    {"section": "Task Model",  "key": "TASK_DEVICE",                "label": "Offload Devices",      "type": "text",   "hint": "Optional llama.cpp --device override, e.g. 0,1 or none"},
    {"section": "Task Model",  "key": "TASK_TENSOR_SPLIT",          "label": "Tensor Split",         "type": "text",   "hint": "e.g. 1,1"},
    {"section": "Task Model",  "key": "TASK_SPLIT_MODE",            "label": "Split Mode",           "type": "select", "options": ["none", "layer", "row", "tensor"], "hint": "none=model on one GPU, layer=layer sharding, row=row sharding, tensor=parallel tensor+KV sharding"},
    {"section": "Task Model",  "key": "TASK_KV_OFFLOAD",            "label": "KV Offload",           "type": "select", "options": ["on", "off"], "hint": "Controls --kv-offload / --no-kv-offload"},
    {"section": "Task Model",  "key": "TASK_OP_OFFLOAD",            "label": "Host Op Offload",      "type": "select", "options": ["on", "off"], "hint": "Controls --op-offload / --no-op-offload for host tensor ops"},
    {"section": "Task Model",  "key": "TASK_MMPROJ_OFFLOAD",        "label": "MMProj Offload",       "type": "select", "options": ["on", "off"], "hint": "Controls --mmproj-offload / --no-mmproj-offload when an MMProj is loaded"},
    {"section": "Task Model",  "key": "TASK_BATCH_SIZE",             "label": "Batch Size",           "type": "number", "hint": "Prefill batch (default 2048)"},
    {"section": "Task Model",  "key": "TASK_UBATCH_SIZE",            "label": "Micro-Batch Size",     "type": "number", "hint": "Physical sub-batch (default 512)"},
    {"section": "Task Model",  "key": "TASK_NO_MMAP",                "label": "Disable mmap",         "type": "select", "options": ["false", "true"]},
    {"section": "Task Model",  "key": "TASK_MLOCK",                  "label": "Lock Memory",          "type": "select", "options": ["false", "true"]},
    {"section": "Task Model",  "key": "TASK_GPU_VISIBLE_DEVICES",   "label": "GPU Devices",          "type": "text"},
    {"section": "Task Model",  "key": "TASK_FLASH_ATTN",            "label": "Flash Attention",      "type": "select", "options": ["on", "off", "auto"]},
    {"section": "Task Model",  "key": "TASK_CACHE_TYPE_K",          "label": "KV Cache Key Type",    "type": "select", "options": LLAMA_KV_CACHE_OPTIONS},
    {"section": "Task Model",  "key": "TASK_CACHE_TYPE_V",          "label": "KV Cache Value Type",  "type": "select", "options": LLAMA_KV_CACHE_OPTIONS},
    {"section": "Task Model",  "key": "TASK_CACHE_RAM",             "label": "Prompt Cache RAM",     "type": "number", "hint": "llama.cpp --cache-ram in MiB; 0 disables server prompt-cache storage"},
    {"section": "Task Model",  "key": "TASK_CTX_CHECKPOINTS",       "label": "Context Checkpoints",  "type": "number", "hint": "llama.cpp --ctx-checkpoints; 0 disables context checkpoint creation"},
    {"section": "Task Model",  "key": "TASK_SWA_FULL",              "label": "Full SWA KV Cache",    "type": "select", "options": ["off", "on"], "hint": "Adds llama.cpp --swa-full for SWA models; uses more KV VRAM but improves prompt-cache reuse"},
    {"section": "Task Model",  "key": "TASK_TEMP",                  "label": "Temperature",          "type": "text",   "hint": "e.g. 1.0"},
    {"section": "Task Model",  "key": "TASK_TOP_P",                 "label": "Top-P",                "type": "text",   "hint": "e.g. 0.95"},
    {"section": "Task Model",  "key": "TASK_TOP_K",                 "label": "Top-K",                "type": "number"},
    {"section": "Task Model",  "key": "TASK_MIN_P",                 "label": "Min-P",                "type": "text",   "hint": "e.g. 0.00"},
    {"section": "Task Model",  "key": "TASK_PRESENCE_PENALTY",      "label": "Presence Penalty",     "type": "text",   "hint": "e.g. 0.00"},
    {"section": "Task Model",  "key": "TASK_REPEAT_PENALTY",        "label": "Repeat Penalty",       "type": "text",   "hint": "e.g. 1.10"},
    {"section": "Task Model",  "key": "TASK_JINJA",                 "label": "Native Tool Calling",  "type": "select", "options": ["off", "on"], "hint": "Enables --jinja for OpenAI-compatible tool/function calling"},
    {"section": "Task Model",  "key": "TASK_CHAT_TEMPLATE_ID",        "label": "Chat Template",        "type": "chat_template", "hint": "Custom Jinja template file passed to the standalone task model"},
    {"section": "Task Model",  "key": "TASK_THINKING",              "label": "Thinking",             "type": "select", "options": ["off", "on"], "hint": "Enable/disable thinking/reasoning for the task model"},
    {"section": "Task Model",  "key": "TASK_REASONING_FORMAT",      "label": "Reasoning Format",     "type": "select", "options": ["none", "deepseek", "deepseek-legacy"], "hint": "How thinking content appears in API responses"},
    {"section": "Task Model",  "key": "TASK_FIT",                   "label": "Auto-Fit to VRAM",     "type": "select", "options": ["on", "off"], "hint": "When on, may reduce context size to fit in VRAM"},
    {"section": "Task Model",  "key": "TASK_FIT_TARGET",            "label": "Fit Target MiB",       "type": "text",   "hint": "llama.cpp --fit-target per-device margin, e.g. 1024 or 1024,2048; empty uses llama.cpp default"},
    {"section": "Task Model",  "key": "TASK_FIT_CTX",               "label": "Minimum Fit Context",  "type": "number", "hint": "llama.cpp --fit-ctx minimum context when auto-fit adjusts settings"},
    {"section": "Task Model",  "key": "TASK_CACHE_IDLE_SLOTS",      "label": "Cache Idle Slots",     "type": "select", "options": LLAMA_CACHE_IDLE_OPTIONS, "hint": "Controls --cache-idle-slots / --no-cache-idle-slots"},
    {"section": "Task Model",  "key": "TASK_CACHE_REUSE",           "label": "Cache Reuse Chunk",    "type": "number", "hint": "llama.cpp --cache-reuse minimum chunk size; 0 leaves llama.cpp default"},
    {"section": "Task Model",  "key": "TASK_CUSTOM_ARGS_JSON",      "label": "Custom Arguments",     "type": "custom_args", "hint": "Extra llama.cpp flags applied to the task model launcher"},
    {"section": "Task Model",  "key": "TASK_SPEC_METHOD",           "label": "Speculative Method",     "type": "select", "options": LLAMA_SPEC_METHOD_OPTIONS, "hint": "Base llama.cpp mode. draft-dflash requires an upstream DFlash draft GGUF with general.architecture=dflash;"},
    {"section": "Task Model",  "key": "TASK_SPEC_NGRAM_MOD",        "label": "N-Gram Mod Assist",      "type": "select", "options": ["off", "on"], "hint": "When on, appends ngram-mod to MTP-style spec types"},
    {"section": "Task Model",  "key": "TASK_SPEC_DRAFT_MODEL_PATH", "label": "Draft Model Path",       "type": "path",   "hint": "Smaller GGUF used as the speculative draft model"},
    {"section": "Task Model",  "key": "TASK_SPEC_DRAFT_N_GPU_LAYERS", "label": "Draft GPU Layers",     "type": "text",   "hint": "Draft-model --spec-draft-ngl value: auto, all, or an exact layer count"},
    {"section": "Task Model",  "key": "TASK_SPEC_DRAFT_DEVICES",    "label": "Draft Devices",          "type": "text",   "hint": "Optional --spec-draft-device override, e.g. 0,1 or none"},
    {"section": "Task Model",  "key": "TASK_SPEC_DRAFT_TYPE_K",     "label": "Draft KV Key Type",      "type": "select", "options": LLAMA_KV_CACHE_OPTIONS, "hint": "llama.cpp --spec-draft-type-k"},
    {"section": "Task Model",  "key": "TASK_SPEC_DRAFT_TYPE_V",     "label": "Draft KV Value Type",    "type": "select", "options": LLAMA_KV_CACHE_OPTIONS, "hint": "llama.cpp --spec-draft-type-v"},
    {"section": "Task Model",  "key": "TASK_SPEC_DRAFT_N_MAX",      "label": "Draft Max Tokens",       "type": "number", "hint": "llama.cpp --spec-draft-n-max (recommended 6 for MTP)"},
    {"section": "Task Model",  "key": "TASK_SPEC_DRAFT_N_MIN",      "label": "Draft Min Tokens",       "type": "number", "hint": "llama.cpp --spec-draft-n-min (default 0)"},
    {"section": "Task Model",  "key": "TASK_SPEC_DRAFT_P_MIN",      "label": "Draft Min Probability",  "type": "text",   "hint": "llama.cpp --spec-draft-p-min (default 0.75)"},
    {"section": "Task Model",  "key": "TASK_SPEC_DRAFT_P_SPLIT",    "label": "Draft Split Probability","type": "text",   "hint": "llama.cpp --spec-draft-p-split (default 0.10)"},
    {"section": "Task Model",  "key": "TASK_SPEC_NGRAM_MOD_N_MATCH","label": "N-Gram Match Tokens",    "type": "number", "hint": "llama.cpp --spec-ngram-mod-n-match (default 24)"},
    {"section": "Task Model",  "key": "TASK_SPEC_NGRAM_MOD_N_MIN",  "label": "N-Gram Min Tokens",      "type": "number", "hint": "llama.cpp --spec-ngram-mod-n-min (default 48)"},
    {"section": "Task Model",  "key": "TASK_SPEC_NGRAM_MOD_N_MAX",  "label": "N-Gram Max Tokens",      "type": "number", "hint": "llama.cpp --spec-ngram-mod-n-max (default 64)"},
    {"section": "Task Model",  "key": "TASK_SPEC_NGRAM_SIZE_N",     "label": "N-Gram Lookup Size",     "type": "number", "hint": "llama.cpp --spec-ngram-*-size-n for ngram-simple/map modes"},
    {"section": "Task Model",  "key": "TASK_SPEC_NGRAM_SIZE_M",     "label": "N-Gram Draft Size",      "type": "number", "hint": "llama.cpp --spec-ngram-*-size-m for ngram-simple/map modes"},
    {"section": "Task Model",  "key": "TASK_SPEC_NGRAM_MIN_HITS",   "label": "N-Gram Min Hits",        "type": "number", "hint": "llama.cpp --spec-ngram-*-min-hits for ngram-simple/map modes"},
    # Thinking Endpoint (proxied request-time overrides)
    {"section": "Thinking Endpoint", "key": "THINK_MODEL_NAME",          "label": "Thinking Model Name",   "type": "text",   "hint": "Advertised on /v1/models for the thinking endpoint"},
    {"section": "Thinking Endpoint", "key": "PROXY_STREAM_PASSTHROUGH",  "label": "Raw Stream Passthrough", "type": "select", "options": ["off", "on"], "hint": "When on, SSE responses bypass proxy JSON rewriting after request shaping"},
    {"section": "Thinking Endpoint", "key": "THINK_PRESERVE_THINKING",   "label": "Preserve Thinking",     "type": "select", "options": ["on", "off"], "hint": "Injects chat_template_kwargs.preserve_thinking into thinking requests"},
    {"section": "Thinking Endpoint", "key": "THINK_REASONING_STREAM_MODE", "label": "Reasoning Stream", "type": "select", "options": ["content", "hidden", "mirror"], "hint": "content makes thinking tokens visible immediately for clients that ignore reasoning_content"},
    {"section": "Thinking Endpoint", "key": "THINK_JINJA",               "label": "Expose Tool Calling",   "type": "select", "options": ["on", "off"], "hint": "When off, strips tools/tool_choice from thinking requests"},
    {"section": "Thinking Endpoint", "key": "THINK_TEMP",                "label": "Temperature",           "type": "text",   "hint": "e.g. 0.7"},
    {"section": "Thinking Endpoint", "key": "THINK_MAX_TOKENS",          "label": "Max Tokens",            "type": "number", "hint": "Overrides client max_tokens; 0 leaves client setting unchanged"},
    {"section": "Thinking Endpoint", "key": "THINK_TOP_P",               "label": "Top-P",                 "type": "text",   "hint": "e.g. 0.95"},
    {"section": "Thinking Endpoint", "key": "THINK_TOP_K",               "label": "Top-K",                 "type": "number"},
    {"section": "Thinking Endpoint", "key": "THINK_MIN_P",               "label": "Min-P",                 "type": "text",   "hint": "e.g. 0.00"},
    {"section": "Thinking Endpoint", "key": "THINK_PRESENCE_PENALTY",    "label": "Presence Penalty",      "type": "text",   "hint": "e.g. 0.00"},
    {"section": "Thinking Endpoint", "key": "THINK_REPEAT_PENALTY",      "label": "Repeat Penalty",        "type": "text",   "hint": "e.g. 1.10"},
    {"section": "Thinking Endpoint", "key": "THINK_REASONING_FORMAT",    "label": "Reasoning Format",      "type": "select", "options": ["none", "deepseek", "deepseek-legacy"], "hint": "Injected per request for the thinking endpoint"},
    # Instruct Endpoint (proxied request-time overrides)
    {"section": "Instruct Endpoint", "key": "NOTHINK_MODEL_NAME",        "label": "Instruct Model Name",   "type": "text",   "hint": "Advertised on /v1/models for the non-thinking instruct endpoint"},
    {"section": "Instruct Endpoint", "key": "NOTHINK_PRESERVE_THINKING", "label": "Preserve Thinking",     "type": "select", "options": ["on", "off"], "hint": "Injects chat_template_kwargs.preserve_thinking into instruct requests"},
    {"section": "Instruct Endpoint", "key": "NOTHINK_REASONING_STREAM_MODE", "label": "Reasoning Stream", "type": "select", "options": ["hidden", "content", "mirror"], "hint": "Usually hidden because this endpoint disables thinking"},
    {"section": "Instruct Endpoint", "key": "NOTHINK_JINJA",             "label": "Expose Tool Calling",   "type": "select", "options": ["on", "off"], "hint": "When off, strips tools/tool_choice from instruct requests"},
    {"section": "Instruct Endpoint", "key": "NOTHINK_TEMP",              "label": "Temperature",           "type": "text",   "hint": "e.g. 0.7"},
    {"section": "Instruct Endpoint", "key": "NOTHINK_MAX_TOKENS",        "label": "Max Tokens",            "type": "number", "hint": "Overrides client max_tokens; 0 leaves client setting unchanged"},
    {"section": "Instruct Endpoint", "key": "NOTHINK_TOP_P",             "label": "Top-P",                 "type": "text",   "hint": "e.g. 0.95"},
    {"section": "Instruct Endpoint", "key": "NOTHINK_TOP_K",             "label": "Top-K",                 "type": "number"},
    {"section": "Instruct Endpoint", "key": "NOTHINK_MIN_P",             "label": "Min-P",                 "type": "text",   "hint": "e.g. 0.00"},
    {"section": "Instruct Endpoint", "key": "NOTHINK_PRESENCE_PENALTY",  "label": "Presence Penalty",      "type": "text",   "hint": "e.g. 0.00"},
    {"section": "Instruct Endpoint", "key": "NOTHINK_REPEAT_PENALTY",    "label": "Repeat Penalty",        "type": "text",   "hint": "e.g. 1.10"},
    {"section": "Instruct Endpoint", "key": "NOTHINK_REASONING_FORMAT",  "label": "Reasoning Format",      "type": "select", "options": ["none", "deepseek", "deepseek-legacy"], "hint": "Injected per request for the instruct endpoint"},
    # Coding Endpoint (proxied request-time overrides)
    {"section": "Coding Endpoint", "key": "CODE_MODEL_NAME",          "label": "Coding Model Name",     "type": "text",   "hint": "Advertised on /v1/models for the code endpoint"},
    {"section": "Coding Endpoint", "key": "CODE_THINKING",            "label": "Thinking",              "type": "select", "options": ["on", "off"], "hint": "Enable/disable thinking for the code endpoint"},
    {"section": "Coding Endpoint", "key": "CODE_PRESERVE_THINKING",   "label": "Preserve Thinking",     "type": "select", "options": ["on", "off"], "hint": "Injects chat_template_kwargs.preserve_thinking into code requests"},
    {"section": "Coding Endpoint", "key": "CODE_REASONING_STREAM_MODE", "label": "Reasoning Stream", "type": "select", "options": ["content", "hidden", "mirror"], "hint": "content makes thinking tokens visible immediately for clients that ignore reasoning_content"},
    {"section": "Coding Endpoint", "key": "CODE_JINJA",               "label": "Expose Tool Calling",   "type": "select", "options": ["on", "off"], "hint": "When off, strips tools/tool_choice from code requests"},
    {"section": "Coding Endpoint", "key": "CODE_TEMP",                "label": "Temperature",           "type": "text",   "hint": "e.g. 0.7"},
    {"section": "Coding Endpoint", "key": "CODE_MAX_TOKENS",          "label": "Max Tokens",            "type": "number", "hint": "Overrides client max_tokens; 0 leaves client setting unchanged"},
    {"section": "Coding Endpoint", "key": "CODE_TOP_P",               "label": "Top-P",                 "type": "text",   "hint": "e.g. 0.95"},
    {"section": "Coding Endpoint", "key": "CODE_TOP_K",               "label": "Top-K",                 "type": "number"},
    {"section": "Coding Endpoint", "key": "CODE_MIN_P",               "label": "Min-P",                 "type": "text",   "hint": "e.g. 0.00"},
    {"section": "Coding Endpoint", "key": "CODE_PRESENCE_PENALTY",    "label": "Presence Penalty",      "type": "text",   "hint": "e.g. 0.00"},
    {"section": "Coding Endpoint", "key": "CODE_REPEAT_PENALTY",      "label": "Repeat Penalty",        "type": "text",   "hint": "e.g. 1.10"},
    {"section": "Coding Endpoint", "key": "CODE_REASONING_FORMAT",    "label": "Reasoning Format",      "type": "select", "options": ["none", "deepseek", "deepseek-legacy"], "hint": "Injected per request for the coding endpoint"},
    # Embedding
    {"section": "Embedding",   "key": "EMBED_MODEL_NAME",           "label": "Model Name",           "type": "text",   "hint": "Advertised on /v1/models for the embedding endpoint"},
    {"section": "Embedding",   "key": "EMBEDDING_MODEL_PATH",       "label": "Model Path",           "type": "path"},
    {"section": "Embedding",   "key": "EMBED_CTX_SIZE",             "label": "Context Size",         "type": "number"},
    {"section": "Embedding",   "key": "EMBED_N_PARALLEL",           "label": "Parallel Slots",       "type": "number"},
    {"section": "Embedding",   "key": "EMBED_THREADS",              "label": "CPU Threads",          "type": "number", "hint": "llama.cpp --threads for generation; -1 lets llama.cpp choose"},
    {"section": "Embedding",   "key": "EMBED_THREADS_BATCH",        "label": "CPU Batch Threads",    "type": "number", "hint": "llama.cpp --threads-batch for prompt/batch processing; -1 follows --threads"},
    {"section": "Embedding",   "key": "EMBED_N_GPU_LAYERS",         "label": "GPU Layers (−1=all)",  "type": "number"},
    {"section": "Embedding",   "key": "EMBED_TENSOR_SPLIT",         "label": "Tensor Split",         "type": "text",   "hint": "e.g. 1,1"},
    {"section": "Embedding",   "key": "EMBED_SPLIT_MODE",           "label": "Split Mode",           "type": "select", "options": ["layer", "row", "none"]},
    {"section": "Embedding",   "key": "EMBED_FLASH_ATTN",           "label": "Flash Attention",      "type": "select", "options": ["on", "off", "auto"]},
    {"section": "Embedding",   "key": "EMBED_CACHE_TYPE_K",         "label": "KV Cache Key Type",    "type": "select", "options": LLAMA_KV_CACHE_OPTIONS},
    {"section": "Embedding",   "key": "EMBED_CACHE_TYPE_V",         "label": "KV Cache Value Type",  "type": "select", "options": LLAMA_KV_CACHE_OPTIONS},
    {"section": "Embedding",   "key": "EMBED_BATCH_SIZE",           "label": "Batch Size",           "type": "number"},
    {"section": "Embedding",   "key": "EMBED_UBATCH_SIZE",          "label": "Micro-Batch Size",     "type": "number"},
    {"section": "Embedding",   "key": "EMBED_NO_MMAP",              "label": "Disable mmap",         "type": "select", "options": ["false", "true"]},
    {"section": "Embedding",   "key": "EMBED_MLOCK",                "label": "Lock Memory",          "type": "select", "options": ["false", "true"]},
    {"section": "Embedding",   "key": "EMBED_GPU_VISIBLE_DEVICES",  "label": "GPU Devices",          "type": "text"},
    {"section": "Embedding",   "key": "EMBED_TEMP",                 "label": "Temperature",          "type": "text",   "hint": "e.g. 1.0"},
    {"section": "Embedding",   "key": "EMBED_TOP_P",                "label": "Top-P",                "type": "text",   "hint": "e.g. 0.95"},
    {"section": "Embedding",   "key": "EMBED_TOP_K",                "label": "Top-K",                "type": "number"},
    {"section": "Embedding",   "key": "EMBED_MIN_P",                "label": "Min-P",                "type": "text",   "hint": "e.g. 0.00"},
    {"section": "Embedding",   "key": "EMBED_JINJA",                "label": "Native Tool Calling",  "type": "select", "options": ["off", "on"]},
    {"section": "Embedding",   "key": "EMBED_REASONING_FORMAT",     "label": "Reasoning Format",     "type": "select", "options": ["none", "deepseek", "deepseek-legacy"]},
    {"section": "Embedding",   "key": "EMBED_FIT",                  "label": "Auto-Fit to VRAM",     "type": "select", "options": ["on", "off"]},
    # Embedding 2
    {"section": "Embedding 2", "key": "EMBED2_MODEL_NAME",          "label": "Model Name",           "type": "text",   "hint": "Advertised on /v1/models for the embedding 2 endpoint"},
    {"section": "Embedding 2", "key": "EMBED2_MODEL_PATH",          "label": "Model Path",           "type": "path"},
    {"section": "Embedding 2", "key": "EMBED2_CTX_SIZE",            "label": "Context Size",         "type": "number"},
    {"section": "Embedding 2", "key": "EMBED2_N_PARALLEL",          "label": "Parallel Slots",       "type": "number"},
    {"section": "Embedding 2", "key": "EMBED2_THREADS",             "label": "CPU Threads",          "type": "number", "hint": "llama.cpp --threads for generation; -1 lets llama.cpp choose"},
    {"section": "Embedding 2", "key": "EMBED2_THREADS_BATCH",       "label": "CPU Batch Threads",    "type": "number", "hint": "llama.cpp --threads-batch for prompt/batch processing; -1 follows --threads"},
    {"section": "Embedding 2", "key": "EMBED2_N_GPU_LAYERS",        "label": "GPU Layers (−1=all)",  "type": "number"},
    {"section": "Embedding 2", "key": "EMBED2_TENSOR_SPLIT",        "label": "Tensor Split",         "type": "text",   "hint": "e.g. 1,1"},
    {"section": "Embedding 2", "key": "EMBED2_SPLIT_MODE",          "label": "Split Mode",           "type": "select", "options": ["layer", "row", "none"]},
    {"section": "Embedding 2", "key": "EMBED2_FLASH_ATTN",          "label": "Flash Attention",      "type": "select", "options": ["on", "off", "auto"]},
    {"section": "Embedding 2", "key": "EMBED2_CACHE_TYPE_K",        "label": "KV Cache Key Type",    "type": "select", "options": LLAMA_KV_CACHE_OPTIONS},
    {"section": "Embedding 2", "key": "EMBED2_CACHE_TYPE_V",        "label": "KV Cache Value Type",  "type": "select", "options": LLAMA_KV_CACHE_OPTIONS},
    {"section": "Embedding 2", "key": "EMBED2_BATCH_SIZE",          "label": "Batch Size",           "type": "number"},
    {"section": "Embedding 2", "key": "EMBED2_UBATCH_SIZE",         "label": "Micro-Batch Size",     "type": "number"},
    {"section": "Embedding 2", "key": "EMBED2_NO_MMAP",             "label": "Disable mmap",         "type": "select", "options": ["false", "true"]},
    {"section": "Embedding 2", "key": "EMBED2_MLOCK",               "label": "Lock Memory",          "type": "select", "options": ["false", "true"]},
    {"section": "Embedding 2", "key": "EMBED2_GPU_VISIBLE_DEVICES", "label": "GPU Devices",          "type": "text"},
    {"section": "Embedding 2", "key": "EMBED2_TEMP",                "label": "Temperature",          "type": "text",   "hint": "e.g. 1.0"},
    {"section": "Embedding 2", "key": "EMBED2_TOP_P",               "label": "Top-P",                "type": "text",   "hint": "e.g. 0.95"},
    {"section": "Embedding 2", "key": "EMBED2_TOP_K",               "label": "Top-K",                "type": "number"},
    {"section": "Embedding 2", "key": "EMBED2_MIN_P",               "label": "Min-P",                "type": "text",   "hint": "e.g. 0.00"},
    {"section": "Embedding 2", "key": "EMBED2_JINJA",               "label": "Native Tool Calling",  "type": "select", "options": ["off", "on"]},
    {"section": "Embedding 2", "key": "EMBED2_REASONING_FORMAT",    "label": "Reasoning Format",     "type": "select", "options": ["none", "deepseek", "deepseek-legacy"]},
    {"section": "Embedding 2", "key": "EMBED2_FIT",                 "label": "Auto-Fit to VRAM",     "type": "select", "options": ["on", "off"]},
    # Reranker
    {"section": "Reranker",    "key": "RERANK_MODEL_NAME",          "label": "Model Name",           "type": "text",   "hint": "Advertised on /v1/models for the reranker endpoint"},
    {"section": "Reranker",    "key": "RERANKER_MODEL_PATH",        "label": "Model Path",           "type": "path"},
    {"section": "Reranker",    "key": "RERANK_CTX_SIZE",            "label": "Context Size",         "type": "number"},
    {"section": "Reranker",    "key": "RERANK_N_PARALLEL",          "label": "Parallel Slots",       "type": "number"},
    {"section": "Reranker",    "key": "RERANK_THREADS",             "label": "CPU Threads",          "type": "number", "hint": "llama.cpp --threads for generation; -1 lets llama.cpp choose"},
    {"section": "Reranker",    "key": "RERANK_THREADS_BATCH",       "label": "CPU Batch Threads",    "type": "number", "hint": "llama.cpp --threads-batch for prompt/batch processing; -1 follows --threads"},
    {"section": "Reranker",    "key": "RERANK_N_GPU_LAYERS",        "label": "GPU Layers (−1=all)",  "type": "number"},
    {"section": "Reranker",    "key": "RERANK_TENSOR_SPLIT",        "label": "Tensor Split",         "type": "text",   "hint": "e.g. 1,1"},
    {"section": "Reranker",    "key": "RERANK_SPLIT_MODE",          "label": "Split Mode",           "type": "select", "options": ["layer", "row", "none"]},
    {"section": "Reranker",    "key": "RERANK_FLASH_ATTN",          "label": "Flash Attention",      "type": "select", "options": ["on", "off", "auto"]},
    {"section": "Reranker",    "key": "RERANK_CACHE_TYPE_K",        "label": "KV Cache Key Type",    "type": "select", "options": LLAMA_KV_CACHE_OPTIONS},
    {"section": "Reranker",    "key": "RERANK_CACHE_TYPE_V",        "label": "KV Cache Value Type",  "type": "select", "options": LLAMA_KV_CACHE_OPTIONS},
    {"section": "Reranker",    "key": "RERANK_BATCH_SIZE",          "label": "Batch Size",           "type": "number"},
    {"section": "Reranker",    "key": "RERANK_UBATCH_SIZE",         "label": "Micro-Batch Size",     "type": "number"},
    {"section": "Reranker",    "key": "RERANK_NO_MMAP",             "label": "Disable mmap",         "type": "select", "options": ["false", "true"]},
    {"section": "Reranker",    "key": "RERANK_MLOCK",               "label": "Lock Memory",          "type": "select", "options": ["false", "true"]},
    {"section": "Reranker",    "key": "RERANK_GPU_VISIBLE_DEVICES", "label": "GPU Devices",          "type": "text"},
    {"section": "Reranker",    "key": "RERANK_TEMP",                "label": "Temperature",          "type": "text",   "hint": "e.g. 1.0"},
    {"section": "Reranker",    "key": "RERANK_TOP_P",               "label": "Top-P",                "type": "text",   "hint": "e.g. 0.95"},
    {"section": "Reranker",    "key": "RERANK_TOP_K",               "label": "Top-K",                "type": "number"},
    {"section": "Reranker",    "key": "RERANK_MIN_P",               "label": "Min-P",                "type": "text",   "hint": "e.g. 0.00"},
    {"section": "Reranker",    "key": "RERANK_JINJA",               "label": "Native Tool Calling",  "type": "select", "options": ["off", "on"]},
    {"section": "Reranker",    "key": "RERANK_REASONING_FORMAT",    "label": "Reasoning Format",     "type": "select", "options": ["none", "deepseek", "deepseek-legacy"]},
    {"section": "Reranker",    "key": "RERANK_FIT",                 "label": "Auto-Fit to VRAM",     "type": "select", "options": ["on", "off"]},
    # OCR
    {"section": "OCR",        "key": "OCR_MODEL_NAME",           "label": "Model Name",           "type": "text",   "hint": "Advertised on /v1/models for the OCR endpoint"},
    {"section": "OCR",        "key": "OCR_MODEL_PATH",           "label": "GLM-OCR Model Path",   "type": "path"},
    {"section": "OCR",        "key": "OCR_MMPROJ_PATH",          "label": "MMProj Path",          "type": "path",   "hint": "Optional multimodal projector if your GGUF build requires a separate file"},
    {"section": "OCR",        "key": "OCR_HOST",                 "label": "Listen Host",          "type": "text"},
    {"section": "OCR",        "key": "OCR_PORT",                 "label": "Port",                 "type": "number"},
    {"section": "OCR",        "key": "OCR_CTX_SIZE",             "label": "Context Size",         "type": "number"},
    {"section": "OCR",        "key": "OCR_N_PARALLEL",           "label": "Parallel Slots",       "type": "number"},
    {"section": "OCR",        "key": "OCR_THREADS",              "label": "CPU Threads",          "type": "number", "hint": "llama.cpp --threads for generation; -1 lets llama.cpp choose"},
    {"section": "OCR",        "key": "OCR_THREADS_BATCH",        "label": "CPU Batch Threads",    "type": "number"},
    {"section": "OCR",        "key": "OCR_N_GPU_LAYERS",         "label": "GPU Layers (-1=all)",  "type": "number"},
    {"section": "OCR",        "key": "OCR_MAIN_GPU",             "label": "Main GPU Index",       "type": "number", "hint": "GPU index within OCR GPU Devices; use 0 for the first visible GPU, 1 for the second"},
    {"section": "OCR",        "key": "OCR_DEVICE",               "label": "Offload Devices",      "type": "text",   "hint": "Optional llama.cpp --device override for OCR, e.g. CUDA0,CUDA1 or none"},
    {"section": "OCR",        "key": "OCR_TENSOR_SPLIT",         "label": "Tensor Split",         "type": "text",   "hint": "auto expands to one weight per visible OCR GPU, e.g. GPU Devices 0,1 -> 1,1; set 2,1 to bias GPU 0"},
    {"section": "OCR",        "key": "OCR_SPLIT_MODE",           "label": "Split Mode",           "type": "select", "options": ["none", "layer", "row", "tensor"], "hint": "none keeps OCR on one GPU; layer/row/tensor split OCR across OCR GPU Devices"},
    {"section": "OCR",        "key": "OCR_KV_OFFLOAD",           "label": "KV Offload",           "type": "select", "options": ["on", "off"]},
    {"section": "OCR",        "key": "OCR_OP_OFFLOAD",           "label": "Host Op Offload",      "type": "select", "options": ["on", "off"]},
    {"section": "OCR",        "key": "OCR_MMPROJ_OFFLOAD",       "label": "MMProj Offload",       "type": "select", "options": ["on", "off"]},
    {"section": "OCR",        "key": "OCR_BATCH_SIZE",           "label": "Batch Size",           "type": "number"},
    {"section": "OCR",        "key": "OCR_UBATCH_SIZE",          "label": "Micro-Batch Size",     "type": "number"},
    {"section": "OCR",        "key": "OCR_FLASH_ATTN",           "label": "Flash Attention",      "type": "select", "options": ["on", "off", "auto"]},
    {"section": "OCR",        "key": "OCR_CACHE_TYPE_K",         "label": "KV Cache Key Type",    "type": "select", "options": LLAMA_KV_CACHE_OPTIONS},
    {"section": "OCR",        "key": "OCR_CACHE_TYPE_V",         "label": "KV Cache Value Type",  "type": "select", "options": LLAMA_KV_CACHE_OPTIONS},
    {"section": "OCR",        "key": "OCR_NO_MMAP",              "label": "Disable mmap",         "type": "select", "options": ["false", "true"]},
    {"section": "OCR",        "key": "OCR_MLOCK",                "label": "Lock Memory",          "type": "select", "options": ["false", "true"]},
    {"section": "OCR",        "key": "OCR_GPU_VISIBLE_DEVICES",  "label": "OCR GPU Devices",      "type": "text",   "hint": "CUDA_VISIBLE_DEVICES for OCR. Use 0 or 1 for one GPU, 0,1 for both GPUs."},
    {"section": "OCR",        "key": "OCR_PROMPT",               "label": "Default OCR Prompt",   "type": "text",   "hint": "Used by /api/ocr/extract when a call does not provide a prompt"},
    {"section": "OCR",        "key": "OCR_TEMP",                 "label": "Temperature",          "type": "text",   "hint": "Low values are best for OCR"},
    {"section": "OCR",        "key": "OCR_TOP_P",                "label": "Top-P",                "type": "text"},
    {"section": "OCR",        "key": "OCR_TOP_K",                "label": "Top-K",                "type": "number"},
    {"section": "OCR",        "key": "OCR_MIN_P",                "label": "Min-P",                "type": "text"},
    {"section": "OCR",        "key": "OCR_FIT",                  "label": "Auto-Fit to VRAM",     "type": "select", "options": ["on", "off"]},
    {"section": "OCR",        "key": "OCR_CUSTOM_ARGS_JSON",     "label": "Custom Arguments",     "type": "custom_args", "hint": "Extra llama.cpp flags applied to the OCR backend"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_SDK_ENABLED",        "label": "SDK Server Enabled",   "type": "select", "options": ["on", "off"], "hint": "Runs the local self-hosted GLM-OCR SDK parser; no MaaS/cloud OCR calls"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_SDK_HOST",           "label": "SDK Listen Host",      "type": "text"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_SDK_PORT",           "label": "SDK Port",             "type": "number"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_PUBLIC_URL",         "label": "Public OCR URL",       "type": "text", "hint": "Stable URL for other apps; defaults to the SDK server"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_SDK_LOG_LEVEL",      "label": "Log Level",            "type": "select", "options": ["DEBUG", "INFO", "WARNING", "ERROR"]},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_OCR_API_MODE",       "label": "OCR API Mode",         "type": "select", "options": ["openai", "ollama_generate"], "hint": "How the SDK calls the local OCR model backend"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_OCR_API_URL",        "label": "OCR API URL Override", "type": "text", "hint": "Optional full local OCR backend URL; leave empty to use OCR_HOST/OCR_PORT"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_OCR_REQUEST_TIMEOUT", "label": "OCR Request Timeout",  "type": "number"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_OCR_CONNECT_TIMEOUT", "label": "OCR Connect Timeout",  "type": "number"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_OCR_RETRY_MAX_ATTEMPTS", "label": "OCR Retry Attempts", "type": "number"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_OCR_CONNECTION_POOL_SIZE", "label": "Connection Pool", "type": "number"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_MAX_WORKERS",        "label": "OCR Workers",          "type": "number", "hint": "Concurrent region OCR requests to the local OCR model"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_PAGE_MAXSIZE",       "label": "Page Queue Size",      "type": "number"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_REGION_MAXSIZE",     "label": "Region Queue Size",    "type": "number"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_PAGE_MAX_TOKENS",    "label": "Max Output Tokens",    "type": "number"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_PAGE_TEMPERATURE",   "label": "Temperature",          "type": "text"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_PAGE_TOP_P",         "label": "Top-P",                "type": "text"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_PAGE_TOP_K",         "label": "Top-K",                "type": "number"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_PAGE_REPETITION_PENALTY", "label": "Repeat Penalty", "type": "text"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_IMAGE_FORMAT",       "label": "Region Image Format",  "type": "select", "options": ["JPEG", "PNG", "WEBP"]},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_MIN_PIXELS",         "label": "Minimum Pixels",       "type": "number"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_MAX_PIXELS",         "label": "Maximum Pixels",       "type": "number"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_PDF_DPI",            "label": "PDF DPI",              "type": "number"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_PDF_MAX_PAGES",      "label": "PDF Max Pages",        "type": "number", "hint": "Empty means no SDK-side page cap"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_LAYOUT_MODEL_DIR",   "label": "Layout Model",         "type": "text"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_LAYOUT_DEVICE",      "label": "Layout Device",        "type": "text", "hint": "cpu, cuda, cuda:0, or empty for auto"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_LAYOUT_CUDA_VISIBLE_DEVICES", "label": "Layout GPUs", "type": "text"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_LAYOUT_THRESHOLD",   "label": "Layout Threshold",     "type": "text"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_LAYOUT_BATCH_SIZE",  "label": "Layout Batch Size",    "type": "number"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_LAYOUT_WORKERS",     "label": "Layout Workers",       "type": "number"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_LAYOUT_USE_POLYGON", "label": "Polygon Crops",        "type": "select", "options": ["off", "on"]},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_OUTPUT_FORMAT",      "label": "Output Format",        "type": "select", "options": ["both", "markdown", "json"]},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_MERGE_FORMULA_NUMBERS", "label": "Merge Formula Numbers", "type": "select", "options": ["on", "off"]},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_MERGE_TEXT_BLOCKS",  "label": "Merge Text Blocks",    "type": "select", "options": ["on", "off"]},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_FORMAT_BULLET_POINTS", "label": "Format Bullets",     "type": "select", "options": ["on", "off"]},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_PROMPT_TEXT",        "label": "Text Prompt",          "type": "text"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_PROMPT_TABLE",       "label": "Table Prompt",         "type": "text"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_PROMPT_FORMULA",     "label": "Formula Prompt",       "type": "text"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_ADVANCED_CONFIG_JSON", "label": "Advanced SDK Config", "type": "text", "hint": "JSON object deep-merged into the generated SDK config"},
    # Honcho
    {"section": "Honcho",     "key": "HONCHO_ENABLED",                 "label": "Enabled",              "type": "select", "options": ["on", "off"]},
    {"section": "Honcho",     "key": "HONCHO_HOST",                    "label": "Listen Host",          "type": "text"},
    {"section": "Honcho",     "key": "HONCHO_PORT",                    "label": "Port",                 "type": "number"},
    {"section": "Honcho",     "key": "HONCHO_URL",                     "label": "Base URL",             "type": "text"},
    {"section": "Honcho",     "key": "HONCHO_WORKSPACE",               "label": "Workspace",            "type": "text"},
    {"section": "Honcho",     "key": "HONCHO_USER_PEER",               "label": "User Peer",            "type": "text"},
    {"section": "Honcho",     "key": "HONCHO_INSTALL_DATASTORES",        "label": "Manage Data Stores",    "type": "select", "options": ["on", "off"]},
    {"section": "Honcho",     "key": "HONCHO_CONFIGURE_HERMES",          "label": "Configure Hermes",     "type": "select", "options": ["on", "off"]},
    {"section": "Honcho",     "key": "HONCHO_AI_PEER",                 "label": "AI Peer",              "type": "text"},
    {"section": "Honcho",     "key": "HONCHO_LLM_BASE_URL",            "label": "LLM Base URL",         "type": "text"},
    {"section": "Honcho",     "key": "HONCHO_LLM_MODEL",               "label": "LLM Model",            "type": "text"},
    {"section": "Honcho",     "key": "HONCHO_EMBED_BASE_URL",          "label": "Embedding Base URL",   "type": "text"},
    {"section": "Honcho",     "key": "HONCHO_EMBED_MODEL",             "label": "Embedding Model",      "type": "text"},
    {"section": "Honcho",     "key": "HONCHO_EMBED_VECTOR_DIMENSIONS", "label": "Embedding Dimensions", "type": "number"},
    # Graphiti
    {"section": "Graphiti",    "key": "GRAPHITI_PUBLIC_URL",        "label": "Public URL",           "type": "text",   "hint": "URL used by external tools (OpenWebUI, OpenClaw, etc.)"},
    {"section": "Graphiti",    "key": "GRAPHITI_HOST",              "label": "Listen Host",          "type": "text"},
    {"section": "Graphiti",    "key": "GRAPHITI_PORT",              "label": "Port",                 "type": "number"},
    {"section": "Graphiti",    "key": "GRAPHITI_LLM_BASE_URL",      "label": "LLM Base URL",         "type": "text"},
    {"section": "Graphiti",    "key": "GRAPHITI_LLM_MODEL",         "label": "LLM Model",            "type": "text"},
    {"section": "Graphiti",    "key": "GRAPHITI_EMBED_BASE_URL",    "label": "Embedding Base URL",   "type": "text"},
    {"section": "Graphiti",    "key": "GRAPHITI_EMBED_MODEL",       "label": "Embedding Model",      "type": "text"},
    {"section": "Graphiti",    "key": "GRAPHITI_RERANKER_PROVIDER", "label": "Reranker Provider",    "type": "select", "options": ["llamacpp", "openai"]},
    {"section": "Graphiti",    "key": "GRAPHITI_RERANKER_BASE_URL", "label": "Reranker Base URL",    "type": "text"},
    {"section": "Graphiti",    "key": "GRAPHITI_RERANKER_MODEL",    "label": "Reranker Model",       "type": "text"},
    {"section": "Graphiti",    "key": "GRAPHITI_MEMORY_MAX_QUERY_CHARS", "label": "Memory Query Max Chars", "type": "number", "hint": "Trim retrieval query text before embedding"},
    {"section": "Graphiti",    "key": "GRAPHITI_MEMORY_MAX_MESSAGES",    "label": "Memory Query Max Messages", "type": "number", "hint": "Number of latest messages used to compose retrieval query"},
    {"section": "Graphiti",    "key": "GRAPHITI_MEMORY_MAX_FACTS",       "label": "Memory Max Facts", "type": "number", "hint": "Server-side cap for /search and /get-memory facts"},
    {"section": "Graphiti",    "key": "GRAPHITI_NEO4J_URI",         "label": "Neo4j URI",            "type": "text"},
    {"section": "Graphiti",    "key": "GRAPHITI_NEO4J_USER",        "label": "Neo4j User",           "type": "text"},
    {"section": "Graphiti",    "key": "GRAPHITI_NEO4J_PASSWORD",    "label": "Neo4j Password",       "type": "text"},
    {"section": "Graphiti",    "key": "GRAPHITI_NEO4J_DATABASE",    "label": "Neo4j Database",       "type": "text"},
    {"section": "Graphiti",    "key": "GRAPHITI_NEO4J_BOLT_PORT",   "label": "Neo4j Bolt Port",      "type": "number"},
    {"section": "Graphiti",    "key": "GRAPHITI_NEO4J_HTTP_PORT",   "label": "Neo4j HTTP Port",      "type": "number"},
    # SearXNG
    {"section": "SearXNG",     "key": "SEARXNG_ENABLED",            "label": "Install On Stack Setup", "type": "select", "options": ["on", "off"]},
    {"section": "SearXNG",     "key": "SEARXNG_PUBLIC_URL",         "label": "Public URL",             "type": "text", "hint": "URL clients and the manager should use"},
    {"section": "SearXNG",     "key": "SEARXNG_BASE_URL",           "label": "Base URL",               "type": "text", "hint": "URL passed through to SearXNG when needed"},
    {"section": "SearXNG",     "key": "SEARXNG_URL_PATH",           "label": "Nginx URL Path",         "type": "text", "hint": "Path mounted into the default nginx server block"},
    {"section": "SearXNG",     "key": "SEARXNG_INSTANCE_NAME",      "label": "Instance Name",          "type": "text"},
    {"section": "SearXNG",     "key": "SEARXNG_SAFE_SEARCH",        "label": "Safe Search",            "type": "select", "options": ["0", "1", "2"]},
    {"section": "SearXNG",     "key": "SEARXNG_AUTOCOMPLETE",       "label": "Autocomplete",           "type": "text"},
    {"section": "SearXNG",     "key": "SEARXNG_FORMATS",            "label": "Search Formats",         "type": "text", "hint": "Comma-separated: html,json,csv,rss"},
    {"section": "SearXNG",     "key": "SEARXNG_LIMITER",            "label": "Limiter",                "type": "select", "options": ["false", "true"]},
    {"section": "SearXNG",     "key": "SEARXNG_IMAGE_PROXY",        "label": "Image Proxy",            "type": "select", "options": ["true", "false"]},
    {"section": "SearXNG",     "key": "SEARXNG_VALKEY_URL",         "label": "Valkey URL",             "type": "text"},
    {"section": "SearXNG",     "key": "SEARXNG_HOME",               "label": "Install Directory",      "type": "path"},
    {"section": "SearXNG",     "key": "SEARXNG_SETTINGS_PATH",      "label": "Settings File",          "type": "path"},
    {"section": "SearXNG",     "key": "SEARXNG_UWSGI_INI",          "label": "uWSGI Config",           "type": "path"},
    {"section": "SearXNG",     "key": "SEARXNG_UWSGI_SOCKET",       "label": "uWSGI Socket",           "type": "path"},
    {"section": "SearXNG",     "key": "SEARXNG_NGINX_CONF",         "label": "Nginx Config",           "type": "path"},
    # Playwright
    {"section": "Playwright",  "key": "PLAYWRIGHT_ENABLED",         "label": "Install On Stack Setup", "type": "select", "options": ["on", "off"]},
    {"section": "Playwright",  "key": "PLAYWRIGHT_PUBLIC_WS_URL",   "label": "Public WS URL",          "type": "text", "hint": "Use with playwright.chromium.connect(...)"},
    {"section": "Playwright",  "key": "PLAYWRIGHT_PUBLIC_HTTP_URL", "label": "Public HTTP URL",        "type": "text", "hint": "Same listener exposed as HTTP/WebSocket"},
    {"section": "Playwright",  "key": "PLAYWRIGHT_URL_PATH",        "label": "Nginx URL Path",         "type": "text", "hint": "Path mounted into the default nginx server block"},
    {"section": "Playwright",  "key": "PLAYWRIGHT_HOST",            "label": "Listen Host",            "type": "text"},
    {"section": "Playwright",  "key": "PLAYWRIGHT_PORT",            "label": "Port",                   "type": "number"},
    {"section": "Playwright",  "key": "PLAYWRIGHT_UPSTREAM_PORT",   "label": "Internal Upstream Port", "type": "number", "hint": "Loopback-only Playwright run-server port used behind the public wrapper"},
    {"section": "Playwright",  "key": "PLAYWRIGHT_BROWSER",         "label": "Browser",                "type": "select", "options": ["chromium", "firefox", "webkit"]},
    {"section": "Playwright",  "key": "PLAYWRIGHT_INSTALL_BROWSERS", "label": "Install Browser Binaries", "type": "select", "options": ["on", "off"]},
    {"section": "Playwright",  "key": "PLAYWRIGHT_BROWSERS_PATH",   "label": "Browser Cache Path",     "type": "path"},
    {"section": "Playwright",  "key": "PLAYWRIGHT_NODE_ENV",        "label": "Node Environment",       "type": "text"},
    {"section": "Playwright",  "key": "PLAYWRIGHT_NGINX_CONF",      "label": "Nginx Config",           "type": "path"},
    # Ports
    {"section": "Ports",       "key": "THINK_PORT",                 "label": "Thinking Port",        "type": "number"},
    {"section": "Ports",       "key": "NOTHINK_PORT",               "label": "Chat Port",            "type": "number"},
    {"section": "Ports",       "key": "CODE_PORT",                  "label": "Code Port",            "type": "number"},
    {"section": "Ports",       "key": "AGGREGATE_ENABLED",          "label": "Aggregate Proxy",      "type": "select", "options": ["on", "off"], "hint": "Single model-routed endpoint exposing think, chat, and code"},
    {"section": "Ports",       "key": "AGGREGATE_PORT",             "label": "Aggregate Port",       "type": "number"},
    {"section": "Ports",       "key": "EMBED_PORT",                 "label": "Embedding Port",       "type": "number"},
    {"section": "Ports",       "key": "EMBED2_PORT",                "label": "Embedding 2 Port",     "type": "number"},
    {"section": "Ports",       "key": "RERANK_PORT",                "label": "Reranker Port",        "type": "number"},
    {"section": "Ports",       "key": "TASK_PORT",                  "label": "Task Port",            "type": "number"},
    {"section": "Ports",       "key": "CHAT_BACKEND_PORT",          "label": "Backend Port",         "type": "number"},
    {"section": "Ports",       "key": "CHAT_BACKEND_HOST",          "label": "Backend Host",         "type": "text"},
    {"section": "Ports",       "key": "LISTEN_HOST",                "label": "Listen Host",          "type": "select", "options": ["0.0.0.0", "127.0.0.1"]},
    # TTS Gateway
    {"section": "TTS Gateway", "key": "TTS_PUBLIC_URL",             "label": "Public TTS URL",       "type": "text",   "hint": "Stable URL clients should use"},
    {"section": "TTS Gateway", "key": "TTS_GATEWAY_HOST",           "label": "Gateway Listen Host",  "type": "text"},
    {"section": "TTS Gateway", "key": "TTS_GATEWAY_PORT",           "label": "Gateway Port",         "type": "number"},
    {"section": "TTS Gateway", "key": "TTS_SINGLE_ACTIVE",          "label": "Single Active Backend","type": "select", "options": ["on", "off"], "hint": "When on, activating one backend stops the others"},
    {"section": "TTS Gateway", "key": "TTS_DEFAULT_FORMAT",         "label": "Default Audio Format", "type": "select", "options": ["mp3", "wav", "flac", "opus", "aac", "pcm"]},
    {"section": "TTS Backends","key": "KOKORO_UPSTREAM_URL",        "label": "Kokoro Upstream URL",  "type": "text",   "hint": "Local HTTP runtime exposing /v1/audio/speech"},
    {"section": "TTS Backends","key": "KOKORO_LAUNCH_CMD",          "label": "Kokoro Launch Command","type": "text",   "hint": "Optional command for a local Kokoro runtime"},
    {"section": "TTS Backends","key": "KOKORO_VOICES",              "label": "Kokoro Voices",        "type": "text",   "hint": "Comma-separated voice ids"},
    {"section": "TTS Backends","key": "CHATTERBOX_UPSTREAM_URL",    "label": "Chatterbox Upstream URL","type": "text", "hint": "Local HTTP runtime exposing /v1/audio/speech"},
    {"section": "TTS Backends","key": "CHATTERBOX_LAUNCH_CMD",      "label": "Chatterbox Launch Command","type": "text","hint": "Optional command for a local Chatterbox runtime"},
    {"section": "TTS Backends","key": "CHATTERBOX_VOICES",          "label": "Chatterbox Voices",    "type": "text",   "hint": "Comma-separated voice ids"},
    {"section": "TTS Backends","key": "VIBEVOICE_UPSTREAM_URL",     "label": "VibeVoice Upstream URL","type": "text",  "hint": "Local HTTP runtime exposing /v1/audio/speech"},
    {"section": "TTS Backends","key": "VIBEVOICE_LAUNCH_CMD",       "label": "VibeVoice Launch Command","type": "text", "hint": "Optional command for a local VibeVoice runtime"},
    {"section": "TTS Backends","key": "VIBEVOICE_VOICES",           "label": "VibeVoice Voices",     "type": "text",   "hint": "Comma-separated voice ids"},
    {"section": "TTS Backends","key": "VIBEVOICE_MODEL_PATH",       "label": "VibeVoice Model Path", "type": "text",   "hint": "HF model id or local model directory"},
    {"section": "TTS Backends","key": "VIBEVOICE_DEVICE",           "label": "VibeVoice Device",     "type": "select", "options": ["cuda", "cpu", "mps"]},
    {"section": "TTS Backends","key": "VIBEVOICE_RUNTIME_HOST",     "label": "VibeVoice Runtime Host","type": "text"},
    {"section": "TTS Backends","key": "VIBEVOICE_RUNTIME_PORT",     "label": "VibeVoice Runtime Port","type": "number"},
    {"section": "TTS Backends","key": "VIBEVOICE_CFG_SCALE",        "label": "VibeVoice CFG Scale",  "type": "text"},
    {"section": "TTS Backends","key": "VIBEVOICE_DDPM_STEPS",       "label": "VibeVoice DDPM Steps", "type": "number"},
    # Transcription
    {"section": "Transcription", "key": "TRANSCRIPT_PUBLIC_URL",        "label": "Public Transcript URL",   "type": "text",   "hint": "Stable URL clients should use"},
    {"section": "Transcription", "key": "TRANSCRIPT_HOST",              "label": "Listen Host",            "type": "text"},
    {"section": "Transcription", "key": "TRANSCRIPT_PORT",              "label": "Port",                   "type": "number"},
    {"section": "Transcription", "key": "TRANSCRIPT_ACTIVE_ENGINE",     "label": "Default Engine",         "type": "select", "options": ["parakeet-v3", "whisperkit-large-v3"]},
    {"section": "Transcription", "key": "TRANSCRIPT_TIMEOUT_SECONDS",   "label": "Request Timeout (sec)",  "type": "number"},
    {"section": "Transcription", "key": "TRANSCRIPT_LOCAL_DEVICE",      "label": "Local Device",           "type": "select", "options": ["cuda", "cpu"]},
    {"section": "Transcription", "key": "TRANSCRIPT_LOCAL_COMPUTE_TYPE","label": "Local Compute Type",     "type": "select", "options": ["float16", "int8", "int8_float16", "float32"]},
    {"section": "Transcription", "key": "PARAKEET_V3_BACKEND_TYPE",     "label": "Parakeet Backend Type",  "type": "select", "options": ["local", "upstream"]},
    {"section": "Transcription", "key": "PARAKEET_V3_LOCAL_MODEL",      "label": "Parakeet Local Model",   "type": "transcript_model", "engine_id": "parakeet-v3", "hint": "Model used when Parakeet backend type is local"},
    {"section": "Transcription", "key": "PARAKEET_V3_UPSTREAM_URL",     "label": "Parakeet v3 Upstream URL","type": "text",  "hint": "Upstream OpenAI-compatible transcription endpoint host"},
    {"section": "Transcription", "key": "PARAKEET_V3_MODEL",            "label": "Parakeet v3 Model Name", "type": "text",   "hint": "Model string sent upstream"},
    {"section": "Transcription", "key": "PARAKEET_V3_API_KEY",          "label": "Parakeet v3 API Key",    "type": "text"},
    {"section": "Transcription", "key": "PARAKEET_V3_TRANSCRIBE_PATH",  "label": "Parakeet v3 Path",       "type": "text",   "hint": "Default: /v1/audio/transcriptions"},
    {"section": "Transcription", "key": "PARAKEET_V3_STREAM_OUTPUT_ENABLED", "label": "Parakeet Streaming Output", "type": "select", "options": ["off", "on"]},
    {"section": "Transcription", "key": "PARAKEET_V3_STREAM_OUTPUT_TARGET",  "label": "Parakeet Stream Target",   "type": "text", "hint": "Future scaffold: webhook/SSE/WebSocket destination"},
    {"section": "Transcription", "key": "PARAKEET_V3_STREAM_OUTPUT_FORMAT",  "label": "Parakeet Stream Format",   "type": "select", "options": ["webhook", "sse", "websocket"]},
    {"section": "Transcription", "key": "PARAKEET_V3_SPEAKER_DETECTION",     "label": "Parakeet Speaker Detection", "type": "select", "options": ["off", "on"]},
    {"section": "Transcription", "key": "PARAKEET_V3_SPEAKER_MODE",          "label": "Parakeet Speaker Mode",      "type": "select", "options": ["auto", "fixed"]},
    {"section": "Transcription", "key": "PARAKEET_V3_SPEAKER_COUNT",         "label": "Parakeet Speaker Count",     "type": "number"},
    {"section": "Transcription", "key": "WHISPERKIT_LARGE_V3_BACKEND_TYPE",  "label": "WhisperKit Backend Type", "type": "select", "options": ["local", "upstream"]},
    {"section": "Transcription", "key": "WHISPERKIT_LARGE_V3_LOCAL_MODEL", "label": "WhisperKit Local Model", "type": "transcript_model", "engine_id": "whisperkit-large-v3", "hint": "Model used when WhisperKit backend type is local"},
    {"section": "Transcription", "key": "WHISPERKIT_LARGE_V3_UPSTREAM_URL",    "label": "WhisperKit Large v3 Upstream URL", "type": "text", "hint": "Upstream OpenAI-compatible transcription endpoint host"},
    {"section": "Transcription", "key": "WHISPERKIT_LARGE_V3_MODEL",           "label": "WhisperKit Large v3 Model Name",   "type": "text", "hint": "Model string sent upstream"},
    {"section": "Transcription", "key": "WHISPERKIT_LARGE_V3_API_KEY",         "label": "WhisperKit Large v3 API Key",      "type": "text"},
    {"section": "Transcription", "key": "WHISPERKIT_LARGE_V3_TRANSCRIBE_PATH", "label": "WhisperKit Large v3 Path",         "type": "text", "hint": "Default: /v1/audio/transcriptions"},
    {"section": "Transcription", "key": "WHISPERKIT_LARGE_V3_STREAM_OUTPUT_ENABLED", "label": "WhisperKit Streaming Output", "type": "select", "options": ["off", "on"]},
    {"section": "Transcription", "key": "WHISPERKIT_LARGE_V3_STREAM_OUTPUT_TARGET",  "label": "WhisperKit Stream Target",   "type": "text", "hint": "Future scaffold: webhook/SSE/WebSocket destination"},
    {"section": "Transcription", "key": "WHISPERKIT_LARGE_V3_STREAM_OUTPUT_FORMAT",  "label": "WhisperKit Stream Format",   "type": "select", "options": ["webhook", "sse", "websocket"]},
    {"section": "Transcription", "key": "WHISPERKIT_LARGE_V3_SPEAKER_DETECTION",     "label": "WhisperKit Speaker Detection", "type": "select", "options": ["off", "on"]},
    {"section": "Transcription", "key": "WHISPERKIT_LARGE_V3_SPEAKER_MODE",          "label": "WhisperKit Speaker Mode",      "type": "select", "options": ["auto", "fixed"]},
    {"section": "Transcription", "key": "WHISPERKIT_LARGE_V3_SPEAKER_COUNT",         "label": "WhisperKit Speaker Count",     "type": "number"},
]

CHAT_BACKEND_IDENTITY_KEYS = {
    "primary": {
        "CHAT_DENSE_LABEL": "CHAT_PRIMARY_LABEL",
        "CHAT_DENSE_MODEL_NAME": "CHAT_PRIMARY_MODEL_NAME",
        "CHAT_DENSE_MODEL_PATH": "CHAT_PRIMARY_MODEL_PATH",
        "CHAT_DENSE_MMPROJ_PATH": "CHAT_PRIMARY_MMPROJ_PATH",
        "CHAT_DENSE_CTX_SIZE": "CHAT_PRIMARY_CTX_SIZE",
    },
    "secondary": {
        "CHAT_MOE_LABEL": "CHAT_SECONDARY_LABEL",
        "CHAT_MOE_MODEL_NAME": "CHAT_SECONDARY_MODEL_NAME",
        "CHAT_MOE_MODEL_PATH": "CHAT_SECONDARY_MODEL_PATH",
        "CHAT_MOE_MMPROJ_PATH": "CHAT_SECONDARY_MMPROJ_PATH",
        "CHAT_MOE_CTX_SIZE": "CHAT_SECONDARY_CTX_SIZE",
    },
}
CHAT_BACKEND_GENERIC_SKIP_KEYS = {
    "CHAT_MODEL_NAME",
    "CHAT_DENSE_LABEL",
    "CHAT_DENSE_MODEL_NAME",
    "CHAT_DENSE_MODEL_PATH",
    "CHAT_DENSE_MMPROJ_PATH",
    "CHAT_DENSE_CTX_SIZE",
    "CHAT_MOE_LABEL",
    "CHAT_MOE_MODEL_NAME",
    "CHAT_MOE_MODEL_PATH",
    "CHAT_MOE_MMPROJ_PATH",
    "CHAT_MOE_CTX_SIZE",
}


def _clone_chat_backend_field(field: dict, variant: str) -> dict | None:
    key = field.get("key", "")
    identity_key = CHAT_BACKEND_IDENTITY_KEYS[variant].get(key)
    if identity_key:
        cloned = dict(field)
        cloned["key"] = identity_key
        if variant == "primary":
            cloned["section"] = "Primary Backend"
            cloned["label"] = cloned.get("label", "").replace("Dense", "Primary").replace("Slot", "Backend")
            cloned["hint"] = cloned.get("hint", "").replace("dense preset", "primary backend")
        else:
            cloned["section"] = "Secondary Backend"
            cloned["label"] = cloned.get("label", "").replace("MoE", "Secondary").replace("Slot", "Backend")
            cloned["hint"] = cloned.get("hint", "").replace("MoE preset", "secondary backend")
        return cloned
    if not key.startswith("CHAT_") or key in CHAT_BACKEND_GENERIC_SKIP_KEYS:
        return None
    cloned = dict(field)
    cloned["section"] = "Primary Backend" if variant == "primary" else "Secondary Backend"
    cloned["key"] = ("CHAT_PRIMARY" if variant == "primary" else "CHAT_SECONDARY") + key[len("CHAT"):]
    return cloned


_shared_backend_fields = [field for field in CONFIG_FIELDS if field.get("section") == "Shared Backend"]
_generated_backend_fields = []
for _variant in ("primary",):
    for _field in _shared_backend_fields:
        _cloned = _clone_chat_backend_field(_field, _variant)
        if _cloned is not None:
            _generated_backend_fields.append(_cloned)

_rebuilt_config_fields = []
_inserted_backend_fields = False
for _field in CONFIG_FIELDS:
    if _field.get("section") == "Shared Backend":
        if not _inserted_backend_fields:
            _rebuilt_config_fields.extend(_generated_backend_fields)
            _inserted_backend_fields = True
    else:
        _rebuilt_config_fields.append(_field)
CONFIG_FIELDS = _rebuilt_config_fields


# Which services should be restarted after changing a given config key
RESTART_HINTS = {
    "CHAT_DENSE_LABEL":          ["chat-backend-dense"],
    "CHAT_DENSE_MODEL_NAME":     ["chat-backend-dense"],
    "CHAT_DENSE_MODEL_PATH":     ["chat-backend-dense"],
    "CHAT_DENSE_MMPROJ_PATH":    ["chat-backend-dense"],
    "CHAT_DENSE_CTX_SIZE":       ["chat-backend-dense"],
    "CHAT_MOE_LABEL":            ["chat-backend-moe"],
    "CHAT_MOE_MODEL_NAME":       ["chat-backend-moe"],
    "CHAT_MOE_MODEL_PATH":       ["chat-backend-moe"],
    "CHAT_MOE_MMPROJ_PATH":      ["chat-backend-moe"],
    "CHAT_MOE_CTX_SIZE":         ["chat-backend-moe"],
    "CHAT_MODEL_NAME":           ["chat-backend"],
    "CHAT_N_PARALLEL":           ["chat-backend-dense", "chat-backend-moe", "chat-backend"],
    "CHAT_THREADS":              ["chat-backend-dense", "chat-backend-moe", "chat-backend"],
    "CHAT_THREADS_BATCH":        ["chat-backend-dense", "chat-backend-moe", "chat-backend"],
    "CHAT_N_GPU_LAYERS":         ["chat-backend-dense", "chat-backend-moe", "chat-backend"],
    "CHAT_MAIN_GPU":             ["chat-backend-dense", "chat-backend-moe", "chat-backend"],
    "CHAT_DEVICE":               ["chat-backend-dense", "chat-backend-moe", "chat-backend"],
    "CHAT_TENSOR_SPLIT":         ["chat-backend-dense", "chat-backend-moe", "chat-backend"],
    "CHAT_SPLIT_MODE":           ["chat-backend-dense", "chat-backend-moe", "chat-backend"],
    "CHAT_KV_OFFLOAD":           ["chat-backend-dense", "chat-backend-moe", "chat-backend"],
    "CHAT_OP_OFFLOAD":           ["chat-backend-dense", "chat-backend-moe", "chat-backend"],
    "CHAT_MMPROJ_OFFLOAD":       ["chat-backend-dense", "chat-backend-moe", "chat-backend"],
    "CHAT_FLASH_ATTN":           ["chat-backend-dense", "chat-backend-moe", "chat-backend"],
    "CHAT_CACHE_TYPE_K":         ["chat-backend-dense", "chat-backend-moe", "chat-backend"],
    "CHAT_CACHE_TYPE_V":         ["chat-backend-dense", "chat-backend-moe", "chat-backend"],
    "CHAT_BATCH_SIZE":           ["chat-backend-dense", "chat-backend-moe", "chat-backend"],
    "CHAT_UBATCH_SIZE":          ["chat-backend-dense", "chat-backend-moe", "chat-backend"],
    "CHAT_NO_MMAP":              ["chat-backend-dense", "chat-backend-moe", "chat-backend"],
    "CHAT_MLOCK":                ["chat-backend-dense", "chat-backend-moe", "chat-backend"],
    "CHAT_GPU_VISIBLE_DEVICES":  ["chat-backend-dense", "chat-backend-moe", "chat-backend"],
    "CHAT_TEMP":                 ["chat-backend-dense", "chat-backend-moe", "chat-backend"],
    "CHAT_TOP_P":                ["chat-backend-dense", "chat-backend-moe", "chat-backend"],
    "CHAT_TOP_K":                ["chat-backend-dense", "chat-backend-moe", "chat-backend"],
    "CHAT_MIN_P":                ["chat-backend-dense", "chat-backend-moe", "chat-backend"],
    "CHAT_PRESERVE_THINKING":    ["chat-backend-dense", "chat-backend-moe", "chat-backend"],
    "CHAT_JINJA":                ["chat-backend-dense", "chat-backend-moe", "chat-backend"],
    "CHAT_REASONING_FORMAT":     ["chat-backend-dense", "chat-backend-moe", "chat-backend"],
    "CHAT_FIT":                  ["chat-backend-dense", "chat-backend-moe", "chat-backend"],
    "CHAT_SPEC_METHOD":          ["chat-backend-dense", "chat-backend-moe", "chat-backend"],
    "CHAT_SPEC_NGRAM_MOD":       ["chat-backend-dense", "chat-backend-moe", "chat-backend"],
    "CHAT_SPEC_DRAFT_MODEL_PATH": ["chat-backend-dense", "chat-backend-moe", "chat-backend"],
    "CHAT_SPEC_DRAFT_N_GPU_LAYERS": ["chat-backend-dense", "chat-backend-moe", "chat-backend"],
    "CHAT_SPEC_DRAFT_DEVICES":   ["chat-backend-dense", "chat-backend-moe", "chat-backend"],
    "CHAT_SPEC_DRAFT_N_MAX":     ["chat-backend-dense", "chat-backend-moe", "chat-backend"],
    "CHAT_SPEC_DRAFT_N_MIN":     ["chat-backend-dense", "chat-backend-moe", "chat-backend"],
    "CHAT_SPEC_DRAFT_P_MIN":     ["chat-backend-dense", "chat-backend-moe", "chat-backend"],
    "CHAT_SPEC_DRAFT_P_SPLIT":   ["chat-backend-dense", "chat-backend-moe", "chat-backend"],
    "CHAT_SPEC_NGRAM_MOD_N_MATCH": ["chat-backend-dense", "chat-backend-moe", "chat-backend"],
    "CHAT_SPEC_NGRAM_MOD_N_MIN": ["chat-backend-dense", "chat-backend-moe", "chat-backend"],
    "CHAT_SPEC_NGRAM_MOD_N_MAX": ["chat-backend-dense", "chat-backend-moe", "chat-backend"],
    "CHAT_CACHE_RAM":            ["chat-backend-dense", "chat-backend-moe", "chat-backend"],
    "CHAT_CTX_CHECKPOINTS":      ["chat-backend-dense", "chat-backend-moe", "chat-backend"],
    "CHAT_SWA_FULL":             ["chat-backend-dense", "chat-backend-moe", "chat-backend"],
    "CHAT_CUSTOM_ARGS_JSON":     ["chat-backend-dense", "chat-backend-moe", "chat-backend"],
    "CHAT_TEMPLATE_ID":           ["chat-backend-dense", "chat-backend-moe", "chat-backend"],
    "CHAT_BACKEND_HOST":         ["chat-proxy"],
    "CHAT_BACKEND_PORT":         ["chat-proxy"],
    "PROXY_STREAM_PASSTHROUGH":  ["chat-proxy"],
    "CHAT2_CACHE_RAM":           ["chat-backend2"],
    "CHAT2_CTX_CHECKPOINTS":     ["chat-backend2"],
    "CHAT2_SWA_FULL":            ["chat-backend2"],
    "CHAT2_CUSTOM_ARGS_JSON":    ["chat-backend2"],
    "CODE_THINKING":             ["chat-proxy"],
    "CODE_PRESERVE_THINKING":    ["chat-proxy"],
    "CODE_REASONING_STREAM_MODE": ["chat-proxy"],
    "CODE_JINJA":                ["chat-proxy"],
    "CODE_CTX_SIZE":             ["chat-proxy"] + SHARED_CHAT_BACKEND_RESTART,
    "CODE_N_PARALLEL":           ["chat-proxy"] + SHARED_CHAT_BACKEND_RESTART,
    "CODE_THREADS":              ["chat-proxy"] + SHARED_CHAT_BACKEND_RESTART,
    "CODE_THREADS_BATCH":        ["chat-proxy"] + SHARED_CHAT_BACKEND_RESTART,
    "CODE_N_GPU_LAYERS":         ["chat-proxy"] + SHARED_CHAT_BACKEND_RESTART,
    "CODE_TENSOR_SPLIT":         ["chat-proxy"] + SHARED_CHAT_BACKEND_RESTART,
    "CODE_SPLIT_MODE":           ["chat-proxy"] + SHARED_CHAT_BACKEND_RESTART,
    "CODE_FLASH_ATTN":           ["chat-proxy"] + SHARED_CHAT_BACKEND_RESTART,
    "CODE_CACHE_TYPE_K":         ["chat-proxy"] + SHARED_CHAT_BACKEND_RESTART,
    "CODE_CACHE_TYPE_V":         ["chat-proxy"] + SHARED_CHAT_BACKEND_RESTART,
    "CODE_BATCH_SIZE":           ["chat-proxy"] + SHARED_CHAT_BACKEND_RESTART,
    "CODE_UBATCH_SIZE":          ["chat-proxy"] + SHARED_CHAT_BACKEND_RESTART,
    "CODE_NO_MMAP":              ["chat-proxy"] + SHARED_CHAT_BACKEND_RESTART,
    "CODE_MLOCK":                ["chat-proxy"] + SHARED_CHAT_BACKEND_RESTART,
    "CODE_GPU_VISIBLE_DEVICES":  ["chat-proxy"] + SHARED_CHAT_BACKEND_RESTART,
    "CODE_TEMP":                 ["chat-proxy"],
    "CODE_MAX_TOKENS":           ["chat-proxy"],
    "CODE_TOP_P":                ["chat-proxy"],
    "CODE_TOP_K":                ["chat-proxy"],
    "CODE_MIN_P":                ["chat-proxy"],
    "CODE_PRESENCE_PENALTY":     ["chat-proxy"],
    "CODE_REPEAT_PENALTY":       ["chat-proxy"],
    "CODE_REASONING_FORMAT":     ["chat-proxy"] + SHARED_CHAT_BACKEND_RESTART,
    "CODE_FIT":                  ["chat-proxy"] + SHARED_CHAT_BACKEND_RESTART,
    "THINK_MODEL_NAME":          ["chat-proxy"],
    "THINK_PRESERVE_THINKING":   ["chat-proxy"],
    "THINK_REASONING_STREAM_MODE": ["chat-proxy"],
    "THINK_JINJA":               ["chat-proxy"],
    "THINK_TEMP":                ["chat-proxy"],
    "THINK_MAX_TOKENS":          ["chat-proxy"],
    "THINK_TOP_P":               ["chat-proxy"],
    "THINK_TOP_K":               ["chat-proxy"],
    "THINK_MIN_P":               ["chat-proxy"],
    "THINK_PRESENCE_PENALTY":    ["chat-proxy"],
    "THINK_REPEAT_PENALTY":      ["chat-proxy"],
    "THINK_REASONING_FORMAT":    ["chat-proxy"],
    "NOTHINK_MODEL_NAME":        ["chat-proxy"],
    "NOTHINK_PRESERVE_THINKING": ["chat-proxy"],
    "NOTHINK_REASONING_STREAM_MODE": ["chat-proxy"],
    "NOTHINK_JINJA":             ["chat-proxy"],
    "NOTHINK_TEMP":              ["chat-proxy"],
    "NOTHINK_TOP_P":             ["chat-proxy"],
    "NOTHINK_TOP_K":             ["chat-proxy"],
    "NOTHINK_MIN_P":             ["chat-proxy"],
    "NOTHINK_PRESENCE_PENALTY":  ["chat-proxy"],
    "NOTHINK_REPEAT_PENALTY":    ["chat-proxy"],
    "NOTHINK_REASONING_FORMAT":  ["chat-proxy"],
    "CODE_MODEL_NAME":           ["chat-proxy"],
    "TASK_MODEL_NAME":           ["task"],
    "TASK_MODEL_PATH":           ["task"],
    "TASK_MMPROJ_PATH":          ["task"],
    "TASK_CTX_SIZE":             ["task"],
    "TASK_N_PARALLEL":           ["task"],
    "TASK_THREADS":              ["task"],
    "TASK_THREADS_BATCH":        ["task"],
    "TASK_N_GPU_LAYERS":         ["task"],
    "TASK_MAIN_GPU":             ["task"],
    "TASK_DEVICE":               ["task"],
    "TASK_TENSOR_SPLIT":         ["task"],
    "TASK_SPLIT_MODE":           ["task"],
    "TASK_KV_OFFLOAD":           ["task"],
    "TASK_OP_OFFLOAD":           ["task"],
    "TASK_MMPROJ_OFFLOAD":       ["task"],
    "TASK_BATCH_SIZE":           ["task"],
    "TASK_UBATCH_SIZE":          ["task"],
    "TASK_NO_MMAP":              ["task"],
    "TASK_MLOCK":                ["task"],
    "TASK_GPU_VISIBLE_DEVICES":  ["task"],
    "TASK_FLASH_ATTN":           ["task"],
    "TASK_CACHE_TYPE_K":         ["task"],
    "TASK_CACHE_TYPE_V":         ["task"],
    "TASK_CACHE_RAM":            ["task"],
    "TASK_CTX_CHECKPOINTS":      ["task"],
    "TASK_SWA_FULL":             ["task"],
    "TASK_TEMP":                 ["task"],
    "TASK_TOP_P":                ["task"],
    "TASK_TOP_K":                ["task"],
    "TASK_MIN_P":                ["task"],
    "TASK_PRESENCE_PENALTY":     ["task"],
    "TASK_REPEAT_PENALTY":       ["task"],
    "TASK_JINJA":                ["task"],
    "TASK_THINKING":             ["task"],
    "TASK_REASONING_FORMAT":     ["task"],
    "TASK_FIT":                  ["task"],
    "TASK_CUSTOM_ARGS_JSON":     ["task"],
    "TASK_CHAT_TEMPLATE_ID":      ["task"],
    "TASK_SPEC_METHOD":             ["task"],
    "TASK_SPEC_NGRAM_MOD":          ["task"],
    "TASK_SPEC_DRAFT_MODEL_PATH":   ["task"],
    "TASK_SPEC_DRAFT_N_GPU_LAYERS": ["task"],
    "TASK_SPEC_DRAFT_DEVICES":      ["task"],
    "TASK_SPEC_DRAFT_N_MAX":        ["task"],
    "TASK_SPEC_DRAFT_N_MIN":        ["task"],
    "TASK_SPEC_DRAFT_P_MIN":        ["task"],
    "TASK_SPEC_DRAFT_P_SPLIT":      ["task"],
    "TASK_SPEC_NGRAM_MOD_N_MATCH":  ["task"],
    "TASK_SPEC_NGRAM_MOD_N_MIN":    ["task"],
    "TASK_SPEC_NGRAM_MOD_N_MAX":    ["task"],
    "EMBED_MODEL_NAME":          ["embed"],
    "EMBEDDING_MODEL_PATH":      ["embed"],
    "EMBED_CTX_SIZE":            ["embed"],
    "EMBED_N_PARALLEL":          ["embed"],
    "EMBED_THREADS":             ["embed"],
    "EMBED_THREADS_BATCH":       ["embed"],
    "EMBED_N_GPU_LAYERS":        ["embed"],
    "EMBED_TENSOR_SPLIT":        ["embed"],
    "EMBED_SPLIT_MODE":          ["embed"],
    "EMBED_FLASH_ATTN":          ["embed"],
    "EMBED_CACHE_TYPE_K":        ["embed"],
    "EMBED_CACHE_TYPE_V":        ["embed"],
    "EMBED_BATCH_SIZE":          ["embed"],
    "EMBED_UBATCH_SIZE":         ["embed"],
    "EMBED_NO_MMAP":             ["embed"],
    "EMBED_MLOCK":               ["embed"],
    "EMBED_GPU_VISIBLE_DEVICES": ["embed"],
    "EMBED_TEMP":                ["embed"],
    "EMBED_TOP_P":               ["embed"],
    "EMBED_TOP_K":               ["embed"],
    "EMBED_MIN_P":               ["embed"],
    "EMBED_JINJA":               ["embed"],
    "EMBED_REASONING_FORMAT":    ["embed"],
    "EMBED_FIT":                 ["embed"],
    "EMBED2_MODEL_NAME":         ["embed2"],
    "EMBED2_MODEL_PATH":         ["embed2"],
    "EMBED2_CTX_SIZE":           ["embed2"],
    "EMBED2_N_PARALLEL":         ["embed2"],
    "EMBED2_THREADS":            ["embed2"],
    "EMBED2_THREADS_BATCH":      ["embed2"],
    "EMBED2_N_GPU_LAYERS":       ["embed2"],
    "EMBED2_TENSOR_SPLIT":       ["embed2"],
    "EMBED2_SPLIT_MODE":         ["embed2"],
    "EMBED2_FLASH_ATTN":         ["embed2"],
    "EMBED2_CACHE_TYPE_K":       ["embed2"],
    "EMBED2_CACHE_TYPE_V":       ["embed2"],
    "EMBED2_BATCH_SIZE":         ["embed2"],
    "EMBED2_UBATCH_SIZE":        ["embed2"],
    "EMBED2_NO_MMAP":            ["embed2"],
    "EMBED2_MLOCK":              ["embed2"],
    "EMBED2_GPU_VISIBLE_DEVICES":["embed2"],
    "EMBED2_TEMP":               ["embed2"],
    "EMBED2_TOP_P":              ["embed2"],
    "EMBED2_TOP_K":              ["embed2"],
    "EMBED2_MIN_P":              ["embed2"],
    "EMBED2_JINJA":              ["embed2"],
    "EMBED2_REASONING_FORMAT":   ["embed2"],
    "EMBED2_FIT":                ["embed2"],
    "RERANK_MODEL_NAME":         ["rerank"],
    "RERANKER_MODEL_PATH":       ["rerank"],
    "RERANK_CTX_SIZE":           ["rerank"],
    "RERANK_N_PARALLEL":         ["rerank"],
    "RERANK_THREADS":            ["rerank"],
    "RERANK_THREADS_BATCH":      ["rerank"],
    "RERANK_N_GPU_LAYERS":       ["rerank"],
    "RERANK_TENSOR_SPLIT":       ["rerank"],
    "RERANK_SPLIT_MODE":         ["rerank"],
    "RERANK_FLASH_ATTN":         ["rerank"],
    "RERANK_CACHE_TYPE_K":       ["rerank"],
    "RERANK_CACHE_TYPE_V":       ["rerank"],
    "RERANK_BATCH_SIZE":         ["rerank"],
    "RERANK_UBATCH_SIZE":        ["rerank"],
    "RERANK_NO_MMAP":            ["rerank"],
    "RERANK_MLOCK":              ["rerank"],
    "RERANK_GPU_VISIBLE_DEVICES":["rerank"],
    "RERANK_TEMP":               ["rerank"],
    "RERANK_TOP_P":              ["rerank"],
    "RERANK_TOP_K":              ["rerank"],
    "RERANK_MIN_P":              ["rerank"],
    "RERANK_JINJA":              ["rerank"],
    "RERANK_REASONING_FORMAT":   ["rerank"],
    "RERANK_FIT":                ["rerank"],
    "HONCHO_ENABLED": ["honcho-api", "honcho-deriver"],
    "HONCHO_INSTALL_DATASTORES": ["honcho-api", "honcho-deriver"],
    "HONCHO_CONFIGURE_HERMES": ["honcho-api", "honcho-deriver"],
    "HONCHO_HOST": ["honcho-api", "honcho-deriver"],
    "HONCHO_PORT": ["honcho-api", "honcho-deriver"],
    "HONCHO_URL": ["honcho-api", "honcho-deriver"],
    "HONCHO_WORKSPACE": ["honcho-api", "honcho-deriver"],
    "HONCHO_USER_PEER": ["honcho-api", "honcho-deriver"],
    "HONCHO_AI_PEER": ["honcho-api", "honcho-deriver"],
    "HONCHO_LLM_BASE_URL": ["honcho-api", "honcho-deriver"],
    "HONCHO_LLM_MODEL": ["honcho-api", "honcho-deriver"],
    "HONCHO_EMBED_BASE_URL": ["honcho-api", "honcho-deriver"],
    "HONCHO_EMBED_MODEL": ["honcho-api", "honcho-deriver"],
    "HONCHO_EMBED_VECTOR_DIMENSIONS": ["honcho-api", "honcho-deriver"],
    "GRAPHITI_PUBLIC_URL":       ["graphiti"],
    "GRAPHITI_HOST":             ["graphiti"],
    "GRAPHITI_PORT":             ["graphiti"],
    "GRAPHITI_LLM_BASE_URL":     ["graphiti"],
    "GRAPHITI_LLM_MODEL":        ["graphiti"],
    "GRAPHITI_EMBED_BASE_URL":   ["graphiti"],
    "GRAPHITI_EMBED_MODEL":      ["graphiti"],
    "GRAPHITI_RERANKER_PROVIDER":["graphiti"],
    "GRAPHITI_RERANKER_BASE_URL":["graphiti"],
    "GRAPHITI_RERANKER_MODEL":   ["graphiti"],
    "GRAPHITI_MEMORY_MAX_QUERY_CHARS": ["graphiti"],
    "GRAPHITI_MEMORY_MAX_MESSAGES":    ["graphiti"],
    "GRAPHITI_MEMORY_MAX_FACTS":       ["graphiti"],
    "GRAPHITI_NEO4J_URI":        ["graphiti"],
    "GRAPHITI_NEO4J_USER":       ["graphiti"],
    "GRAPHITI_NEO4J_PASSWORD":   ["graphiti"],
    "GRAPHITI_NEO4J_DATABASE":   ["graphiti"],
    "GRAPHITI_NEO4J_BOLT_PORT":  ["graphiti"],
    "GRAPHITI_NEO4J_HTTP_PORT":  ["graphiti"],
    "THINK_PORT":                ["chat-proxy"],
    "NOTHINK_PORT":              ["chat-proxy"],
    "CODE_PORT":                 ["chat-proxy"],
    "AGGREGATE_ENABLED":         ["chat-proxy"],
    "AGGREGATE_PORT":            ["chat-proxy"],
    "THINK2_PORT":               ["chat-proxy2"],
    "NOTHINK2_PORT":             ["chat-proxy2"],
    "CODE2_PORT":                ["chat-proxy2"],
    "AGGREGATE2_ENABLED":        ["chat-proxy2"],
    "AGGREGATE2_PORT":           ["chat-proxy2"],
    "THINK2_MODEL_NAME":         ["chat-proxy2"],
    "NOTHINK2_MODEL_NAME":       ["chat-proxy2"],
    "CODE2_MODEL_NAME":          ["chat-proxy2"],
    "EMBED_PORT":                ["embed"],
    "EMBED2_PORT":               ["embed2"],
    "RERANK_PORT":               ["rerank"],
    "TASK_PORT":                 ["task"],
    "LISTEN_HOST":               ["chat-proxy", "embed", "embed2", "rerank", "task"],
    "CHAT_MODEL_PATH":           ["chat-backend"],
    "CHAT_MMPROJ_PATH":          ["chat-backend"],
    "CHAT_CTX_SIZE":             ["chat-backend"],
    "TTS_PUBLIC_URL":            ["tts-gateway"],
    "TTS_GATEWAY_HOST":          ["tts-gateway"],
    "TTS_GATEWAY_PORT":          ["tts-gateway"],
    "TTS_SINGLE_ACTIVE":         ["tts-gateway"],
    "TTS_DEFAULT_FORMAT":        ["tts-gateway"],
    "KOKORO_UPSTREAM_URL":       ["tts-backend-kokoro"],
    "KOKORO_LAUNCH_CMD":         ["tts-backend-kokoro"],
    "KOKORO_VOICES":             ["tts-backend-kokoro", "tts-gateway"],
    "CHATTERBOX_UPSTREAM_URL":   ["tts-backend-chatterbox"],
    "CHATTERBOX_LAUNCH_CMD":     ["tts-backend-chatterbox"],
    "CHATTERBOX_VOICES":         ["tts-backend-chatterbox", "tts-gateway"],
    "VIBEVOICE_UPSTREAM_URL":    ["tts-backend-vibevoice"],
    "VIBEVOICE_LAUNCH_CMD":      ["tts-backend-vibevoice"],
    "VIBEVOICE_VOICES":          ["tts-backend-vibevoice", "tts-gateway"],
    "VIBEVOICE_MODEL_PATH":      ["tts-backend-vibevoice"],
    "VIBEVOICE_DEVICE":          ["tts-backend-vibevoice"],
    "VIBEVOICE_RUNTIME_HOST":    ["tts-backend-vibevoice"],
    "VIBEVOICE_RUNTIME_PORT":    ["tts-backend-vibevoice"],
    "VIBEVOICE_CFG_SCALE":       ["tts-backend-vibevoice"],
    "VIBEVOICE_DDPM_STEPS":      ["tts-backend-vibevoice"],
    "TRANSCRIPT_PUBLIC_URL":     ["transcript-backend"],
    "TRANSCRIPT_HOST":           ["transcript-backend"],
    "TRANSCRIPT_PORT":           ["transcript-backend"],
    "TRANSCRIPT_ACTIVE_ENGINE":  ["transcript-backend"],
    "TRANSCRIPT_TIMEOUT_SECONDS":["transcript-backend"],
    "TRANSCRIPT_LOCAL_MODEL_SIZE":  ["transcript-backend"],
    "TRANSCRIPT_LOCAL_DEVICE":      ["transcript-backend"],
    "TRANSCRIPT_LOCAL_COMPUTE_TYPE":["transcript-backend"],
    "PARAKEET_V3_BACKEND_TYPE":  ["transcript-backend"],
    "PARAKEET_V3_LOCAL_MODEL":   ["transcript-backend"],
    "PARAKEET_V3_UPSTREAM_URL":  ["transcript-backend"],
    "PARAKEET_V3_MODEL":         ["transcript-backend"],
    "PARAKEET_V3_API_KEY":       ["transcript-backend"],
    "PARAKEET_V3_TRANSCRIBE_PATH": ["transcript-backend"],
    "PARAKEET_V3_STREAM_OUTPUT_ENABLED": ["transcript-backend"],
    "PARAKEET_V3_STREAM_OUTPUT_TARGET": ["transcript-backend"],
    "PARAKEET_V3_STREAM_OUTPUT_FORMAT": ["transcript-backend"],
    "PARAKEET_V3_SPEAKER_DETECTION": ["transcript-backend"],
    "PARAKEET_V3_SPEAKER_MODE": ["transcript-backend"],
    "PARAKEET_V3_SPEAKER_COUNT": ["transcript-backend"],
    "WHISPERKIT_LARGE_V3_BACKEND_TYPE": ["transcript-backend"],
    "WHISPERKIT_LARGE_V3_LOCAL_MODEL": ["transcript-backend"],
    "WHISPERKIT_LARGE_V3_UPSTREAM_URL": ["transcript-backend"],
    "WHISPERKIT_LARGE_V3_MODEL": ["transcript-backend"],
    "WHISPERKIT_LARGE_V3_API_KEY": ["transcript-backend"],
    "WHISPERKIT_LARGE_V3_TRANSCRIBE_PATH": ["transcript-backend"],
    "WHISPERKIT_LARGE_V3_STREAM_OUTPUT_ENABLED": ["transcript-backend"],
    "WHISPERKIT_LARGE_V3_STREAM_OUTPUT_TARGET": ["transcript-backend"],
    "WHISPERKIT_LARGE_V3_STREAM_OUTPUT_FORMAT": ["transcript-backend"],
    "WHISPERKIT_LARGE_V3_SPEAKER_DETECTION": ["transcript-backend"],
    "WHISPERKIT_LARGE_V3_SPEAKER_MODE": ["transcript-backend"],
    "WHISPERKIT_LARGE_V3_SPEAKER_COUNT": ["transcript-backend"],
}

for _field in CONFIG_FIELDS:
    _key = _field.get("key", "")
    if _key.startswith("CHAT_PRIMARY_"):
        RESTART_HINTS.setdefault(_key, ["chat-backend-dense"])
    if _key.startswith("CHAT_SECONDARY_"):
        RESTART_HINTS.setdefault(_key, ["chat-backend-moe"])
    if _key.startswith("CHAT2_"):
        RESTART_HINTS.setdefault(_key, ["chat-backend2"])
    if _key.startswith("OCR_"):
        RESTART_HINTS.setdefault(_key, ["ocr"])
    if _key.startswith("GLMOCR_"):
        RESTART_HINTS.setdefault(_key, ["glmocr-sdk"])
    if _key.startswith("SEARXNG_"):
        RESTART_HINTS.setdefault(_key, ["searxng"])
    if _key.startswith("PLAYWRIGHT_"):
        RESTART_HINTS.setdefault(_key, ["playwright-server"])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    normalized.setdefault("CHAT_CTX_CHECKPOINTS", "32")
    normalized.setdefault("CHAT_CACHE_IDLE_SLOTS", "on")
    normalized.setdefault("CHAT_CACHE_REUSE", "0")
    normalized.setdefault("CHAT_SWA_FULL", "off")
    normalized.setdefault("CHAT_FIT_TARGET", "")
    normalized.setdefault("CHAT_FIT_CTX", "4096")
    normalized.setdefault("CHAT2_CACHE_RAM", "8192")
    normalized.setdefault("CHAT2_CTX_CHECKPOINTS", "32")
    normalized.setdefault("CHAT2_CACHE_IDLE_SLOTS", "on")
    normalized.setdefault("CHAT2_CACHE_REUSE", "0")
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
    normalized.setdefault("THINK_REASONING_STREAM_MODE", "content")
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
    normalized.setdefault("CODE_REASONING_STREAM_MODE", "content")
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
    normalized.setdefault("OCR_MODEL_PATH", str(STACK_DIR / "models" / "GLM-OCR-F16.gguf"))
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
    normalized.setdefault("GLMOCR_SDK_ENABLED", "on")
    normalized.setdefault("GLMOCR_SDK_HOST", normalized.get("LISTEN_HOST", "0.0.0.0"))
    normalized.setdefault("GLMOCR_SDK_PORT", "5002")
    normalized.setdefault("GLMOCR_PUBLIC_URL", f"http://127.0.0.1:{normalized.get('GLMOCR_SDK_PORT', '5002')}/glmocr/parse")
    normalized.setdefault("GLMOCR_SDK_LOG_LEVEL", "INFO")
    normalized.setdefault("GLMOCR_OCR_API_MODE", "openai")
    normalized.setdefault("GLMOCR_OCR_API_URL", "")
    normalized.setdefault("GLMOCR_OCR_REQUEST_TIMEOUT", "120")
    normalized.setdefault("GLMOCR_OCR_CONNECT_TIMEOUT", "30")
    normalized.setdefault("GLMOCR_OCR_RETRY_MAX_ATTEMPTS", "2")
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
    normalized.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(STACK_DIR / "playwright" / "browsers"))
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

def read_env() -> dict:
    """Parse env file and return non-commented key=value pairs."""
    env = {}
    try:
        with open(CONFIG_FILE) as f:
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
    return normalize_env_keys(env)


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
    with open(CONFIG_FILE, 'r') as f:
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
    with open(CONFIG_FILE, 'w') as f:
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


def saved_config_apply_updates(config: dict) -> dict:
    updates = {k: v for k, v in config.items()
               if not k.startswith('_') and (v is None or isinstance(v, (str, int, float, bool)))}
    updates = filter_config_updates(updates)
    form_snapshot = config.get("_config_form")
    if isinstance(form_snapshot, dict):
        updates.update(config_form_snapshot(form_snapshot))
    return updates


def builtin_chat_variants(env: dict | None = None) -> list[dict]:
    env = normalize_env_keys(env or read_env())
    items = []
    for item in BUILTIN_CHAT_VARIANTS:
        items.append({
            **item,
            "label": env.get(item["label_key"], item["default_label"]).strip() or item["default_label"],
            "desc": item["default_desc"],
        })
    return items


def patch_service_labels(env: dict | None = None) -> list[dict]:
    env = env or read_env()
    variant_by_service = {item["service"]: item for item in builtin_chat_variants(env)}
    patched = []
    for svc in SERVICES:
        item = variant_by_service.get(svc["name"])
        if item:
            updated = dict(svc)
            updated["label"] = item["label"]
            updated["desc"] = item["desc"]
            patched.append(updated)
        else:
            patched.append(svc)
    return patched


def get_service_status(name: str) -> str:
    if is_searxng_service(name):
        ok, output = run_searxng_manager('status')
        if ok:
            return output.strip() or 'unknown'
        return 'failed'
    if should_use_local_transcript_manager(name):
        ok, output = run_transcript_manager('status')
        if ok:
            return output.strip() or 'unknown'
        return 'failed'
    if should_use_local_tts_manager(name):
        ok, output = run_tts_manager(name, 'status')
        if ok:
            return output.strip() or 'unknown'
        return 'failed'
    try:
        if ServiceManager.is_active(name):
            return 'active'
        elif ServiceManager.is_installed(name):
            return 'inactive'
        else:
            return 'unknown'
    except Exception:
        return 'unknown'


def active_chat_model_snapshot(env: dict | None = None) -> dict:
    """Return the active primary chat backend in a form saved configs can replay."""
    env = env or read_env()
    for item in builtin_chat_variants(env):
        if get_service_status(item['service']) == 'active':
            return {
                "variant": item["id"],
                "service": item["service"],
                "label": item["label"],
                "kind": "builtin",
            }

    if get_service_status('chat-backend') == 'active':
        model_path = env.get('CHAT_MODEL_PATH', '')
        for model in load_custom_models():
            if model.get('model_path') == model_path:
                return {
                    "variant": model.get("id"),
                    "service": "chat-backend",
                    "label": model.get("display_name") or model.get("model_name") or "Custom",
                    "kind": "custom",
                    "model_path": model_path,
                }
        return {
            "variant": "generic",
            "service": "chat-backend",
            "label": "Custom",
            "kind": "generic",
            "model_path": model_path,
        }

    return {"variant": None, "service": None, "label": "", "kind": "none"}


def active_secondary_backend_snapshot(env: dict | None = None) -> dict:
    env = normalize_env_keys(env or read_env())
    if get_service_status('chat-backend2') != 'active':
        return {"variant": None, "service": None, "label": "", "kind": "none"}
    label = env.get("CHAT2_LABEL", "").strip() or "Secondary Backend"
    return {
        "variant": "secondary",
        "service": "chat-backend2",
        "label": label,
        "kind": "secondary",
        "model_name": env.get("CHAT2_MODEL_NAME", ""),
        "model_path": env.get("CHAT2_MODEL_PATH", ""),
    }


def active_backend_slots_snapshot(env: dict | None = None) -> dict:
    env = env or read_env()
    return {
        "primary": active_chat_model_snapshot(env),
        "secondary": active_secondary_backend_snapshot(env),
    }


def saved_config_name(name: str) -> str:
    return re.sub(r'[^\w\-]', '_', name)


def get_default_saved_config_name() -> str:
    try:
        name = DEFAULT_SAVED_CONFIG_FILE.read_text().strip()
    except FileNotFoundError:
        return ""
    return saved_config_name(name) if name else ""


def set_default_saved_config_name(name: str):
    safe_name = saved_config_name(name)
    DEFAULT_SAVED_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_SAVED_CONFIG_FILE.write_text(f"{safe_name}\n")


def clear_default_saved_config_name(name: str | None = None):
    if not DEFAULT_SAVED_CONFIG_FILE.exists():
        return
    if name is not None and get_default_saved_config_name() != saved_config_name(name):
        return
    DEFAULT_SAVED_CONFIG_FILE.unlink()


def launch_chat_backend_for_saved_config(active: dict | None) -> tuple[bool, str, list[str]]:
    active = active or {}
    variant = active.get("variant")
    service = active.get("service")
    if variant in BUILTIN_CHAT_VARIANT_IDS:
        ok, output = run_script('switch-chat-model.sh', variant)
        return ok, output, [BUILTIN_CHAT_VARIANT_BY_ID[variant]["service"], "chat-proxy"]

    if service != "chat-backend":
        ServiceManager.start('chat-proxy')
        return True, "No saved chat backend was active; left chat backend unchanged.", []

    for svc in ('chat-backend-dense', 'chat-backend-moe', 'chat-backend',
                'qwen-chat-backend-27b', 'qwen-chat-backend-35b', 'qwen-chat-backend'):
        ServiceManager.stop(svc)
    r = ServiceManager.start('chat-backend')
    ServiceManager.start('chat-proxy')
    return r.returncode == 0, (r.stdout + r.stderr).strip(), ["chat-backend", "chat-proxy"]


def service_main_pids() -> dict[int, str]:
    mapping = {}
    for svc in SERVICES:
        name = svc.get("name")
        if not name:
            continue
        try:
            pid = ServiceManager.get_pid(name)
            if pid > 0:
                mapping[pid] = name
        except Exception:
            continue
    return mapping


def label_gpu_process(pid: int, process_name: str, service_pids: dict[int, str]) -> str:
    if pid in service_pids:
        return service_pids[pid]
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_text(errors="ignore").replace("\x00", " ").strip()
    except Exception:
        cmdline = ""
    haystack = f"{process_name} {cmdline}"
    for svc in SERVICES:
        name = svc.get("name", "")
        if name and (name in haystack or f"start-{name}.sh" in haystack):
            return name
    if "llama-server" in haystack:
        return "llama-server"
    if "python" in process_name.lower():
        return Path(cmdline.split(" ", 1)[0] if cmdline else process_name).name
    return Path(process_name or "process").name


def get_gpu_processes(uuid_by_index: dict[int, str]) -> dict[int, list[dict]]:
    uuid_to_index = {uuid: index for index, uuid in uuid_by_index.items() if uuid}
    service_pids = service_main_pids()
    processes = {index: [] for index in uuid_by_index}
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5,
        )
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            gpu_uuid, pid_text, process_name, used_text = parts[:4]
            index = uuid_to_index.get(gpu_uuid)
            if index is None:
                continue
            try:
                pid = int(pid_text)
                used = int(float(used_text))
            except ValueError:
                continue
            processes.setdefault(index, []).append({
                "pid": pid,
                "name": label_gpu_process(pid, process_name, service_pids),
                "process_name": Path(process_name).name,
                "used_memory": used,
            })
    except Exception:
        pass
    for items in processes.values():
        items.sort(key=lambda item: item.get("used_memory", 0), reverse=True)
    return processes


def get_gpu_info() -> list:
    try:
        r = subprocess.run(
            ['nvidia-smi',
             '--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5,
        )
        gpus = []
        uuid_by_index = {}
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 7:
                index = int(parts[0])
                mem_used, mem_total = int(parts[3]), int(parts[4])
                uuid_by_index[index] = parts[1]
                gpus.append({
                    'index':     index,
                    'uuid':      parts[1],
                    'name':      parts[2],
                    'mem_used':  mem_used,
                    'mem_total': mem_total,
                    'util':      int(parts[5]),
                    'temp':      int(parts[6]),
                    'mem_pct':   round(100 * mem_used / max(mem_total, 1)),
                    'processes': [],
                })
        processes = get_gpu_processes(uuid_by_index)
        for gpu in gpus:
            gpu["processes"] = processes.get(gpu["index"], [])
        return gpus
    except Exception:
        return []


def run_script(script_name: str, *args) -> tuple:
    script = SCRIPTS_DIR / script_name
    try:
        r = subprocess.run(
            ['bash', str(script)] + list(args),
            capture_output=True, text=True, timeout=60,
        )
        return r.returncode == 0, (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return False, 'Script timed out'
    except Exception as e:
        return False, str(e)


def run_command(cmd: list[str], cwd: Path | None = None, timeout: int = 1200) -> tuple[int, str]:
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(cwd) if cwd else None,
        )
        return r.returncode, (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return 124, f"Command timed out after {timeout}s: {' '.join(cmd)}"
    except Exception as exc:
        return 1, str(exc)


def append_command_log(lines: list[str], cmd: list[str], rc: int, out: str):
    lines.append(f"$ {' '.join(cmd)}")
    lines.append(f"[exit {rc}]")
    if out:
        lines.append(out)


def find_git_repo_root(start: Path) -> Path | None:
    current = start.resolve(strict=False)
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def read_meminfo() -> dict[str, int]:
    data: dict[str, int] = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                key, _, raw_value = line.partition(":")
                parts = raw_value.strip().split()
                if parts and parts[0].isdigit():
                    data[key] = int(parts[0])
    except Exception:
        pass
    return data


def format_kib_as_gib(value_kib: int) -> str:
    return f"{value_kib / 1024 / 1024:.1f} GiB"


def determine_llamacpp_build_parallelism(env: dict) -> tuple[int, list[str]]:
    notes: list[str] = []
    cpu_count = max(os.cpu_count() or 1, 1)
    meminfo = read_meminfo()
    mem_available_kib = meminfo.get("MemAvailable", 0)
    swap_free_kib = meminfo.get("SwapFree", 0)

    notes.append(f"detected CPU cores: {cpu_count}")
    if mem_available_kib:
        notes.append(f"MemAvailable: {format_kib_as_gib(mem_available_kib)}")
    if swap_free_kib or "SwapFree" in meminfo:
        notes.append(f"SwapFree: {format_kib_as_gib(swap_free_kib)}")

    override_raw = (env.get("LLAMACPP_UPDATE_BUILD_JOBS") or "").strip()
    if override_raw:
        try:
            override_jobs = int(override_raw)
            if override_jobs < 1:
                raise ValueError
            notes.append(f"using configured LLAMACPP_UPDATE_BUILD_JOBS={override_jobs}")
            return override_jobs, notes
        except ValueError as exc:
            raise RuntimeError(
                f"Invalid LLAMACPP_UPDATE_BUILD_JOBS value: {override_raw}"
            ) from exc

    min_mem_gib_raw = (env.get("LLAMACPP_UPDATE_MIN_MEM_GB") or "").strip()
    try:
        min_mem_gib = float(min_mem_gib_raw) if min_mem_gib_raw else 4.0
    except ValueError as exc:
        raise RuntimeError(
            f"Invalid LLAMACPP_UPDATE_MIN_MEM_GB value: {min_mem_gib_raw}"
        ) from exc
    min_mem_kib = int(min_mem_gib * 1024 * 1024)

    if mem_available_kib and mem_available_kib < min_mem_kib:
        raise RuntimeError(
            "Refusing to build llama.cpp because available memory is too low "
            f"({format_kib_as_gib(mem_available_kib)} available, need at least {min_mem_gib:.1f} GiB). "
            "Free memory or set LLAMACPP_UPDATE_BUILD_JOBS / LLAMACPP_UPDATE_MIN_MEM_GB explicitly if you want to override this safeguard."
        )

    jobs_by_memory = cpu_count
    if mem_available_kib:
        reserve_kib = 2 * 1024 * 1024
        per_job_kib = int(1.5 * 1024 * 1024)
        jobs_by_memory = max(1, (max(mem_available_kib - reserve_kib, 0) // per_job_kib) or 1)

    jobs = max(1, min(cpu_count, max(1, cpu_count - 1), jobs_by_memory, 4))
    notes.append(f"selected build parallelism: {jobs}")
    return jobs, notes


def resolve_llamacpp_paths(env: dict) -> tuple[Path, Path, Path]:
    raw_bin = (env.get("LLAMA_SERVER_BIN") or "").strip()
    if not raw_bin:
        raise RuntimeError("LLAMA_SERVER_BIN is not set in config/llm-stack.env")
    bin_path = Path(raw_bin).expanduser()
    if not bin_path.is_absolute():
        bin_path = (STACK_DIR / bin_path).resolve(strict=False)
    else:
        bin_path = bin_path.resolve(strict=False)

    source_dir = find_git_repo_root(bin_path.parent)
    if source_dir is None:
        raise RuntimeError(
            "Could not find llama.cpp git repo by walking upward from "
            f"configured LLAMA_SERVER_BIN: {bin_path}"
        )

    build_dir_candidates: list[Path] = []
    try:
        rel = bin_path.relative_to(source_dir)
        if len(rel.parts) >= 3 and rel.parts[-2] == "bin":
            build_dir_candidates.append(source_dir.joinpath(*rel.parts[:-2]))
    except ValueError:
        pass
    build_dir_candidates.append(source_dir / "build")

    build_dir = next((candidate for candidate in build_dir_candidates if candidate.exists()), build_dir_candidates[0])
    return bin_path, build_dir, source_dir


def detect_origin_default_branch(git_cmd: list[str]) -> str:
    rc, out = run_command([*git_cmd, "symbolic-ref", "refs/remotes/origin/HEAD"], timeout=30)
    if rc == 0 and out.strip().startswith("refs/remotes/origin/"):
        return out.strip().rsplit("/", 1)[-1]
    for candidate in ("master", "main"):
        rc2, _ = run_command([*git_cmd, "show-ref", "--verify", f"refs/remotes/origin/{candidate}"], timeout=30)
        if rc2 == 0:
            return candidate
    raise RuntimeError("Unable to determine upstream default branch (origin/HEAD)")


def get_current_git_branch(git_cmd: list[str]) -> str | None:
    rc, out = run_command([*git_cmd, "symbolic-ref", "--quiet", "--short", "HEAD"], timeout=30)
    branch = out.strip()
    return branch if rc == 0 and branch else None


def get_current_git_upstream(git_cmd: list[str]) -> str | None:
    rc, out = run_command(
        [*git_cmd, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"],
        timeout=30,
    )
    upstream = out.strip()
    return upstream if rc == 0 and upstream else None


LLAMACPP_UPDATE_IGNORABLE_DIRTY_FILES = {
    "tools/ui/package-lock.json",
}


def has_uncommitted_git_changes(git_cmd: list[str]) -> tuple[bool, str]:
    rc, out = run_command([*git_cmd, "status", "--porcelain"], timeout=30)
    if rc != 0:
        raise RuntimeError(out or "git status failed")
    return bool(out.strip()), out


def try_restore_ignorable_llamacpp_update_changes(git_cmd: list[str], dirty_output: str, lines: list[str]) -> bool:
    entries = [line for line in dirty_output.splitlines() if line.strip()]
    if not entries:
        return False

    paths: list[str] = []
    for line in entries:
        status = line[:2]
        path = line[3:] if len(line) > 3 else ""
        if status != " M" or path not in LLAMACPP_UPDATE_IGNORABLE_DIRTY_FILES:
            return False
        paths.append(path)

    lines.append(
        "Only generated llama.cpp UI lockfile metadata changed; restoring it before update."
    )
    cmd = [*git_cmd, "restore", "--", *paths]
    rc, out = run_command(cmd, timeout=60)
    append_command_log(lines, cmd, rc, out)
    if rc != 0:
        return False

    dirty_after, dirty_after_output = has_uncommitted_git_changes(git_cmd)
    if dirty_after:
        lines.append("llama.cpp checkout is still dirty after restoring generated files:")
        lines.append(dirty_after_output)
        return False
    return True


def local_branch_exists(git_cmd: list[str], branch: str) -> bool:
    rc, _ = run_command([*git_cmd, "show-ref", "--verify", f"refs/heads/{branch}"], timeout=30)
    return rc == 0


def remote_branch_exists(git_cmd: list[str], branch: str) -> bool:
    rc, _ = run_command([*git_cmd, "show-ref", "--verify", f"refs/remotes/origin/{branch}"], timeout=30)
    return rc == 0


def determine_update_branch(git_cmd: list[str], lines: list[str]) -> tuple[str, str | None]:
    current_branch = get_current_git_branch(git_cmd)
    current_upstream = get_current_git_upstream(git_cmd)
    lines.append(f"current branch: {current_branch or 'detached HEAD'}")
    lines.append(f"current upstream: {current_upstream or 'none'}")

    if current_upstream and current_upstream.startswith("origin/"):
        return current_upstream.split("/", 1)[1], current_branch

    default_branch = detect_origin_default_branch(git_cmd)
    lines.append(f"upstream default branch: {default_branch}")
    return default_branch, current_branch


def ensure_branch_checked_out(git_cmd: list[str], branch: str, current_branch: str | None, lines: list[str]) -> tuple[bool, str]:
    if current_branch == branch:
        return True, current_branch

    switch_cmds: list[list[str]] = []
    if local_branch_exists(git_cmd, branch):
        switch_cmds.extend([
            [*git_cmd, "switch", branch],
            [*git_cmd, "checkout", branch],
        ])
    elif remote_branch_exists(git_cmd, branch):
        switch_cmds.extend([
            [*git_cmd, "switch", "-c", branch, "--track", f"origin/{branch}"],
            [*git_cmd, "checkout", "-b", branch, "--track", f"origin/{branch}"],
        ])
    else:
        raise RuntimeError(f"Remote branch origin/{branch} does not exist after fetch")

    last_output = ""
    for cmd in switch_cmds:
        rc, out = run_command(cmd, timeout=60)
        append_command_log(lines, cmd, rc, out)
        if rc == 0:
            return True, branch
        last_output = out
    return False, last_output or f"Unable to switch to branch {branch}"


def update_llamacpp_and_restart_active_services() -> tuple[bool, str, list[str]]:
    lines: list[str] = []
    restarted: list[str] = []
    try:
        env = read_env()
        bin_path, build_dir, source_dir = resolve_llamacpp_paths(env)
        git_cmd = ["git", "-c", f"safe.directory={source_dir}", "-C", str(source_dir)]

        lines.append(f"LLAMA_SERVER_BIN: {bin_path}")
        lines.append(f"llama.cpp source: {source_dir}")
        lines.append(f"build dir: {build_dir}")
        lines.append(f"configured binary exists: {'yes' if bin_path.exists() else 'no'}")

        build_jobs, build_notes = determine_llamacpp_build_parallelism(env)
        lines.extend(build_notes)

        rc, out = run_command([*git_cmd, "remote", "set-head", "origin", "-a"], timeout=60)
        append_command_log(lines, [*git_cmd, "remote", "set-head", "origin", "-a"], rc, out)
        if rc != 0:
            lines.append("Continuing with cached origin metadata because origin HEAD refresh failed.")

        dirty, dirty_output = has_uncommitted_git_changes(git_cmd)
        if dirty and try_restore_ignorable_llamacpp_update_changes(git_cmd, dirty_output, lines):
            dirty, dirty_output = has_uncommitted_git_changes(git_cmd)
        if dirty:
            lines.append("Refusing to update because the llama.cpp checkout has local modifications:")
            lines.append(dirty_output)
            lines.append("Commit, stash, or discard local changes before using Update llama.cpp.")
            return False, "\n".join(lines), restarted

        update_branch, current_branch = determine_update_branch(git_cmd, lines)

        for cmd in (
            [*git_cmd, "fetch", "origin", "--prune"],
            [*git_cmd, "rev-parse", "--short", "HEAD"],
        ):
            rc, out = run_command(cmd, timeout=3600)
            append_command_log(lines, cmd, rc, out)
            if rc != 0:
                return False, "\n".join(lines), restarted

        ok, branch_result = ensure_branch_checked_out(git_cmd, update_branch, current_branch, lines)
        if not ok:
            lines.append(branch_result)
            return False, "\n".join(lines), restarted

        for cmd in (
            [*git_cmd, "pull", "--ff-only", "origin", update_branch],
            [*git_cmd, "rev-parse", "--short", "HEAD"],
        ):
            rc, out = run_command(cmd, timeout=3600)
            append_command_log(lines, cmd, rc, out)
            if rc != 0:
                return False, "\n".join(lines), restarted

        build_dir.mkdir(parents=True, exist_ok=True)
        for cmd in (
            ["cmake", "-S", str(source_dir), "-B", str(build_dir)],
            ["nice", "-n", "15", "cmake", "--build", str(build_dir), "--target", "llama-server", "--parallel", str(build_jobs)],
        ):
            rc, out = run_command(cmd, timeout=3600)
            append_command_log(lines, cmd, rc, out)
            if rc != 0:
                return False, "\n".join(lines), restarted

        if not bin_path.exists():
            lines.append(f"Build finished but llama-server is still missing at: {bin_path}")
            return False, "\n".join(lines), restarted

        active_model_services = [
            svc for svc in LLAMACPP_MODEL_SERVICES if get_service_status(svc) == "active"
        ]
        restart_failures: list[str] = []

        for svc in active_model_services:
            cmd = ["ServiceManager.restart", svc]
            rc, out = ServiceManager.restart(svc, timeout=120)
            append_command_log(lines, cmd, rc, out)
            if rc == 0:
                restarted.append(svc)
            else:
                restart_failures.append(svc)

        if get_service_status(LLAMACPP_PROXY_SERVICE) == "active":
            cmd = ["ServiceManager.restart", LLAMACPP_PROXY_SERVICE]
            rc, out = ServiceManager.restart(LLAMACPP_PROXY_SERVICE, timeout=120)
            append_command_log(lines, cmd, rc, out)
            if rc == 0:
                restarted.append(LLAMACPP_PROXY_SERVICE)
            else:
                restart_failures.append(LLAMACPP_PROXY_SERVICE)

        if restart_failures:
            lines.append("llama.cpp updated, but some services failed to restart:")
            lines.append(", ".join(restart_failures))
            return False, "\n".join(lines), restarted

        lines.append("llama.cpp update complete.")
        return True, "\n".join(lines), restarted
    except Exception as exc:
        lines.append(f"Unhandled llama.cpp update error: {exc}")
        lines.append(traceback.format_exc())
        return False, "\n".join(lines), restarted


def is_tts_service(name: str) -> bool:
    return name in TTS_MANAGED_SERVICES


def systemd_unit_exists(name: str) -> bool:
    try:
        return ServiceManager.is_installed(name)
    except Exception:
        return False


def should_use_local_tts_manager(name: str) -> bool:
    return is_tts_service(name) and not systemd_unit_exists(name)


def run_tts_manager(name: str, action: str) -> tuple:
    try:
        r = subprocess.run(
            ['bash', str(SCRIPTS_DIR / 'manage-tts-service.sh'), name, action],
            capture_output=True, text=True, timeout=60,
        )
        return r.returncode == 0, (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return False, 'TTS manager action timed out'
    except Exception as e:
        return False, str(e)


def should_use_local_transcript_manager(name: str) -> bool:
    return name == TRANSCRIPT_MANAGED_SERVICE and not systemd_unit_exists(name)


def run_transcript_manager(action: str) -> tuple:
    try:
        r = subprocess.run(
            ['bash', str(SCRIPTS_DIR / 'manage-transcript-service.sh'), action],
            capture_output=True, text=True, timeout=60,
        )
        return r.returncode == 0, (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return False, 'Transcript manager action timed out'
    except Exception as e:
        return False, str(e)


def is_searxng_service(name: str) -> bool:
    return name == "searxng"


def run_searxng_manager(action: str) -> tuple[bool, str]:
    if action == "status":
        uwsgi = get_service_status("uwsgi")
        nginx = get_service_status("nginx")
        socket_path = Path(read_env().get("SEARXNG_UWSGI_SOCKET", "/usr/local/searxng/run/socket"))
        if uwsgi == "active" and nginx == "active" and socket_path.exists():
            return True, "active"
        if uwsgi == "failed" or nginx == "failed":
            return True, "failed"
        return True, "inactive"
    try:
        if action == "start":
            ServiceManager.start("nginx")
            r = ServiceManager.start("uwsgi")
            return r.returncode == 0, (r.stdout + r.stderr).strip()
        if action == "stop":
            r = ServiceManager.stop("uwsgi")
            return r.returncode == 0, (r.stdout + r.stderr).strip()
        if action == "restart":
            rc, output = ServiceManager.restart("uwsgi")
            ServiceManager.run_cmd(["nginx", "-t"], timeout=10)
            ServiceManager.run_cmd(["systemctl", "reload", "nginx"], timeout=10)
            return rc == 0, output
        if action == "install":
            r = subprocess.run(["bash", str(SCRIPTS_DIR / "install-searxng.sh")], capture_output=True, text=True, timeout=900)
            return r.returncode == 0, (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return False, "SearXNG manager action timed out"
    except Exception as exc:
        return False, str(exc)
    return False, "unsupported action"


def tts_log_file(name: str) -> Path:
    return LOGS_DIR / 'tts' / f'{name}.log'


def transcript_log_file() -> Path:
    return LOGS_DIR / 'transcript' / f'{TRANSCRIPT_MANAGED_SERVICE}.log'


def load_json_file(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def tts_gateway_url() -> str:
    env = read_env()
    port = env.get("TTS_GATEWAY_PORT", "8060")
    return f"http://127.0.0.1:{port}"


def http_json(url: str, method: str = "GET", payload=None, timeout: int = 30):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urlrequest.Request(url, data=data, method=method, headers=headers)
    with urlrequest.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
        if not body:
            return {}
        return json.loads(body.decode("utf-8"))


def http_bytes(url: str, method: str = "GET", payload=None, timeout: int = 300):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urlrequest.Request(url, data=data, method=method, headers=headers)
    with urlrequest.urlopen(req, timeout=timeout) as resp:
        return resp.read(), resp.headers.get_content_type()


def parse_int(value, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        out = default
    if minimum is not None:
        out = max(minimum, out)
    if maximum is not None:
        out = min(maximum, out)
    return out


def parse_iso_datetime(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        # Validate only; we keep the original string for query params.
        datetime.fromisoformat(value.replace('Z', '+00:00'))
        return value
    except ValueError:
        return None


def truncate_text(value: str | None, limit: int = 220) -> str:
    if not isinstance(value, str):
        return ''
    if len(value) <= limit:
        return value
    if limit <= 3:
        return value[:limit]
    return value[: limit - 3] + '...'


def graphiti_config(env: dict | None = None) -> dict:
    env = env or read_env()
    graphiti_url = (
        env.get('GRAPHITI_PUBLIC_URL')
        or f"http://127.0.0.1:{env.get('GRAPHITI_PORT', '8070')}"
    ).rstrip('/')
    parsed_bolt = urlparse(env.get('GRAPHITI_NEO4J_URI', 'bolt://127.0.0.1:7687'))
    neo4j_host_default = parsed_bolt.hostname or env.get('GRAPHITI_NEO4J_BOLT_BIND', '127.0.0.1')
    neo4j_http_host = env.get('GRAPHITI_NEO4J_HTTP_BIND', neo4j_host_default)
    neo4j_http_port = env.get('GRAPHITI_NEO4J_HTTP_PORT', '7474')
    return {
        'graphiti_url': graphiti_url,
        'llm_base_url': (env.get('GRAPHITI_LLM_BASE_URL') or '').rstrip('/'),
        'llm_model': env.get('GRAPHITI_LLM_MODEL', ''),
        'embed_base_url': (env.get('GRAPHITI_EMBED_BASE_URL') or '').rstrip('/'),
        'embed_model': env.get('GRAPHITI_EMBED_MODEL', ''),
        'reranker_provider': env.get('GRAPHITI_RERANKER_PROVIDER', ''),
        'reranker_base_url': (env.get('GRAPHITI_RERANKER_BASE_URL') or '').rstrip('/'),
        'reranker_model': env.get('GRAPHITI_RERANKER_MODEL', ''),
        'neo4j_uri': env.get('GRAPHITI_NEO4J_URI', ''),
        'neo4j_user': env.get('GRAPHITI_NEO4J_USER', ''),
        'neo4j_password': env.get('GRAPHITI_NEO4J_PASSWORD', ''),
        'neo4j_database': env.get('GRAPHITI_NEO4J_DATABASE', 'neo4j'),
        'neo4j_http_url': f"http://{neo4j_http_host}:{neo4j_http_port}",
    }


def neo4j_http_query(cypher: str, parameters: dict | None = None, timeout: int = 15) -> dict:
    cfg = graphiti_config()
    statement = {'statement': cypher, 'parameters': parameters or {}}
    payload = {'statements': [statement]}
    auth_bytes = f"{cfg['neo4j_user']}:{cfg['neo4j_password']}".encode('utf-8')
    auth_header = base64.b64encode(auth_bytes).decode('ascii')
    req = urlrequest.Request(
        f"{cfg['neo4j_http_url']}/db/{cfg['neo4j_database']}/tx/commit",
        data=json.dumps(payload).encode('utf-8'),
        method='POST',
        headers={
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': f'Basic {auth_header}',
        },
    )
    with urlrequest.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    data = json.loads(body.decode('utf-8')) if body else {}
    errors = data.get('errors') or []
    if errors:
        raise RuntimeError(errors[0].get('message', 'Neo4j query failed'))
    results = data.get('results') or []
    if not results:
        return {'columns': [], 'rows': []}
    rows = []
    columns = results[0].get('columns', [])
    for item in results[0].get('data', []):
        rows.append(item.get('row', []))
    return {'columns': columns, 'rows': rows}


def neo4j_rows_as_dicts(result: dict) -> list[dict]:
    columns = result.get('columns', [])
    out = []
    for row in result.get('rows', []):
        out.append({columns[i]: row[i] for i in range(min(len(columns), len(row)))})
    return out


def normalize_entity_labels(raw_labels) -> list[str]:
    if isinstance(raw_labels, list):
        return [str(x) for x in raw_labels if str(x) and str(x) != 'Entity']
    return []


def graphiti_recent_episodes(
    page: int = 1,
    page_size: int = 25,
    group_id: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
) -> dict:
    skip = (page - 1) * page_size
    where = []
    params: dict[str, object] = {'skip': skip, 'limit': page_size}
    if group_id:
        where.append('e.group_id = $group_id')
        params['group_id'] = group_id
    if start_time:
        where.append('e.created_at >= datetime($start_time)')
        params['start_time'] = start_time
    if end_time:
        where.append('e.created_at <= datetime($end_time)')
        params['end_time'] = end_time
    where_clause = f"WHERE {' AND '.join(where)}" if where else ''
    count_query = f"MATCH (e:Episodic) {where_clause} RETURN count(e) AS total"
    list_query = f"""
        MATCH (e:Episodic)
        {where_clause}
        RETURN e.uuid AS uuid,
               e.name AS name,
               e.group_id AS group_id,
               toString(e.created_at) AS created_at,
               toString(e.valid_at) AS valid_at,
               e.source AS source,
               e.source_description AS source_description,
               e.content AS content
        ORDER BY e.created_at DESC
        SKIP $skip
        LIMIT $limit
    """
    total_rows = neo4j_rows_as_dicts(neo4j_http_query(count_query, params))
    total = int(total_rows[0]['total']) if total_rows else 0
    rows = neo4j_rows_as_dicts(neo4j_http_query(list_query, params))
    items = []
    for row in rows:
        items.append(
            {
                'uuid': row.get('uuid'),
                'name': row.get('name') or '',
                'group_id': row.get('group_id') or '',
                'created_at': row.get('created_at'),
                'valid_at': row.get('valid_at'),
                'source': row.get('source') or '',
                'source_description': row.get('source_description') or '',
                'content_snippet': truncate_text(row.get('content'), 280),
                'content': row.get('content') or '',
            }
        )
    return {'items': items, 'page': page, 'page_size': page_size, 'total': total}


def graphiti_recent_entities(
    page: int = 1,
    page_size: int = 25,
    group_id: str | None = None,
    name_query: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
) -> dict:
    skip = (page - 1) * page_size
    where = []
    params: dict[str, object] = {'skip': skip, 'limit': page_size}
    if group_id:
        where.append('n.group_id = $group_id')
        params['group_id'] = group_id
    if name_query:
        where.append('toLower(coalesce(n.name, "")) CONTAINS toLower($name_query)')
        params['name_query'] = name_query
    if start_time:
        where.append('n.created_at >= datetime($start_time)')
        params['start_time'] = start_time
    if end_time:
        where.append('n.created_at <= datetime($end_time)')
        params['end_time'] = end_time
    where_clause = f"WHERE {' AND '.join(where)}" if where else ''
    count_query = f"MATCH (n:Entity) {where_clause} RETURN count(n) AS total"
    list_query = f"""
        MATCH (n:Entity)
        {where_clause}
        RETURN n.uuid AS uuid,
               n.name AS name,
               n.group_id AS group_id,
               toString(n.created_at) AS created_at,
               n.summary AS summary,
               n.labels AS prop_labels,
               [x IN labels(n) WHERE x <> 'Entity'] AS node_labels,
               COUNT {{ (n)--() }} AS degree
        ORDER BY n.created_at DESC
        SKIP $skip
        LIMIT $limit
    """
    total_rows = neo4j_rows_as_dicts(neo4j_http_query(count_query, params))
    total = int(total_rows[0]['total']) if total_rows else 0
    rows = neo4j_rows_as_dicts(neo4j_http_query(list_query, params))
    items = []
    for row in rows:
        labels = normalize_entity_labels(row.get('prop_labels')) or normalize_entity_labels(row.get('node_labels'))
        items.append(
            {
                'uuid': row.get('uuid'),
                'name': row.get('name') or '',
                'group_id': row.get('group_id') or '',
                'created_at': row.get('created_at'),
                'summary': row.get('summary') or '',
                'summary_snippet': truncate_text(row.get('summary'), 240),
                'labels': labels,
                'degree': int(row.get('degree') or 0),
            }
        )
    return {'items': items, 'page': page, 'page_size': page_size, 'total': total}


def graphiti_recent_relationships(
    page: int = 1,
    page_size: int = 25,
    group_id: str | None = None,
    relation_query: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
) -> dict:
    skip = (page - 1) * page_size
    where = []
    params: dict[str, object] = {'skip': skip, 'limit': page_size}
    if group_id:
        where.append('r.group_id = $group_id')
        params['group_id'] = group_id
    if relation_query:
        where.append('toLower(coalesce(r.name, "")) CONTAINS toLower($relation_query)')
        params['relation_query'] = relation_query
    if start_time:
        where.append('r.created_at >= datetime($start_time)')
        params['start_time'] = start_time
    if end_time:
        where.append('r.created_at <= datetime($end_time)')
        params['end_time'] = end_time
    where_clause = f"WHERE {' AND '.join(where)}" if where else ''
    count_query = f"MATCH (:Entity)-[r:RELATES_TO]->(:Entity) {where_clause} RETURN count(r) AS total"
    list_query = f"""
        MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity)
        {where_clause}
        RETURN r.uuid AS uuid,
               r.group_id AS group_id,
               r.name AS relation_name,
               r.fact AS fact,
               toString(r.created_at) AS created_at,
               toString(r.valid_at) AS valid_at,
               toString(r.invalid_at) AS invalid_at,
               toString(r.expired_at) AS expired_at,
               s.uuid AS source_uuid,
               s.name AS source_name,
               t.uuid AS target_uuid,
               t.name AS target_name
        ORDER BY r.created_at DESC
        SKIP $skip
        LIMIT $limit
    """
    total_rows = neo4j_rows_as_dicts(neo4j_http_query(count_query, params))
    total = int(total_rows[0]['total']) if total_rows else 0
    rows = neo4j_rows_as_dicts(neo4j_http_query(list_query, params))
    items = []
    for row in rows:
        items.append(
            {
                'uuid': row.get('uuid'),
                'group_id': row.get('group_id') or '',
                'relation_name': row.get('relation_name') or '',
                'fact': row.get('fact') or '',
                'fact_snippet': truncate_text(row.get('fact'), 260),
                'created_at': row.get('created_at'),
                'valid_at': row.get('valid_at'),
                'invalid_at': row.get('invalid_at'),
                'expired_at': row.get('expired_at'),
                'source_uuid': row.get('source_uuid'),
                'source_name': row.get('source_name') or '',
                'target_uuid': row.get('target_uuid'),
                'target_name': row.get('target_name') or '',
            }
        )
    return {'items': items, 'page': page, 'page_size': page_size, 'total': total}


def graphiti_markdown_export(payload: dict) -> str:
    meta = payload.get('metadata', {})
    lines = [
        f"# Graphiti Export: {meta.get('export_type', 'unknown')}",
        '',
        f"- Exported at: `{meta.get('exported_at', '')}`",
        f"- Graphiti URL: `{meta.get('graphiti_url', '')}`",
        f"- Neo4j DB: `{meta.get('neo4j_database', '')}`",
        f"- Item count: `{meta.get('item_count', 0)}`",
        '',
    ]

    if payload.get('episodes'):
        lines.append('## Episodes')
        for ep in payload['episodes']:
            lines.extend(
                [
                    f"### {ep.get('name') or ep.get('uuid')}",
                    f"- UUID: `{ep.get('uuid')}`",
                    f"- Group: `{ep.get('group_id', '')}`",
                    f"- Created: `{ep.get('created_at', '')}`",
                    f"- Source: `{ep.get('source', '')}`",
                    f"- Source Description: `{ep.get('source_description', '')}`",
                    '',
                    truncate_text(ep.get('content', ''), 2000) or '_No content_',
                    '',
                ]
            )

    if payload.get('entities'):
        lines.append('## Entities')
        for ent in payload['entities']:
            labels = ', '.join(ent.get('labels') or [])
            lines.extend(
                [
                    f"### {ent.get('name') or ent.get('uuid')}",
                    f"- UUID: `{ent.get('uuid')}`",
                    f"- Group: `{ent.get('group_id', '')}`",
                    f"- Created: `{ent.get('created_at', '')}`",
                    f"- Labels: `{labels}`",
                    f"- Degree: `{ent.get('degree', 0)}`",
                    f"- Summary: {truncate_text(ent.get('summary', ''), 600)}",
                    '',
                ]
            )

    if payload.get('relationships'):
        lines.append('## Relationships')
        for rel in payload['relationships']:
            lines.extend(
                [
                    f"### {rel.get('relation_name') or rel.get('uuid')}",
                    f"- UUID: `{rel.get('uuid')}`",
                    f"- Group: `{rel.get('group_id', '')}`",
                    f"- Created: `{rel.get('created_at', '')}`",
                    f"- Source: `{rel.get('source_name', '')}` (`{rel.get('source_uuid', '')}`)",
                    f"- Target: `{rel.get('target_name', '')}` (`{rel.get('target_uuid', '')}`)",
                    f"- Fact: {truncate_text(rel.get('fact', ''), 700)}",
                    '',
                ]
            )

    return '\n'.join(lines).strip() + '\n'


def safe_export_filename(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9._-]+', '_', name)


def graphiti_entity_neighborhood(entity_uuid: str, limit: int = 50) -> dict:
    query = """
        MATCH (center:Entity {uuid: $uuid})
        OPTIONAL MATCH (center)-[r1:RELATES_TO]->(n1:Entity)
        WITH center, collect(DISTINCT {
          uuid: r1.uuid,
          relation_name: r1.name,
          fact: r1.fact,
          direction: 'out',
          source_uuid: center.uuid,
          source_name: center.name,
          target_uuid: n1.uuid,
          target_name: n1.name
        }) AS outgoing
        OPTIONAL MATCH (n2:Entity)-[r2:RELATES_TO]->(center)
        WITH center, outgoing, collect(DISTINCT {
          uuid: r2.uuid,
          relation_name: r2.name,
          fact: r2.fact,
          direction: 'in',
          source_uuid: n2.uuid,
          source_name: n2.name,
          target_uuid: center.uuid,
          target_name: center.name
        }) AS incoming
        RETURN center.uuid AS uuid,
               center.name AS name,
               center.group_id AS group_id,
               toString(center.created_at) AS created_at,
               center.summary AS summary,
               outgoing[0..$limit] AS outgoing,
               incoming[0..$limit] AS incoming
    """
    rows = neo4j_rows_as_dicts(neo4j_http_query(query, {'uuid': entity_uuid, 'limit': limit}))
    if not rows:
        raise RuntimeError('Entity not found')
    return rows[0]


def load_tts_backends() -> list:
    return load_json_file(TTS_CONFIG_FILE, {"backends": []}).get("backends", [])


def load_tts_state() -> dict:
    return load_json_file(TTS_STATE_FILE, {"active_backend": None, "updated_at": None})


def wait_for_tts_gateway(timeout: float = 15.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            http_json(f'{tts_gateway_url()}/health', timeout=3)
            return True
        except Exception:
            time.sleep(0.5)
    return False


# ---------------------------------------------------------------------------
# GGUF / Custom models / Saved configs helpers
# ---------------------------------------------------------------------------


def chat_template_id_from_name(name: str) -> str:
    base = re.sub(r"[^A-Za-z0-9._-]+", "-", (name or "").strip()).strip("-._")
    return base[:80] or f"template-{int(time.time())}"


def validate_chat_template_id(template_id: str) -> str:
    template_id = (template_id or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,80}", template_id):
        raise ValueError("Template id may only contain letters, numbers, dot, underscore, and dash")
    return template_id


def load_chat_template_meta() -> dict:
    try:
        data = json.loads(CHAT_TEMPLATES_META_FILE.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_chat_template_meta(meta: dict):
    CHAT_TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    CHAT_TEMPLATES_META_FILE.write_text(json.dumps(meta, indent=2, sort_keys=True))


def chat_template_path(template_id: str) -> Path:
    return CHAT_TEMPLATES_DIR / f"{validate_chat_template_id(template_id)}.jinja"


def list_chat_templates() -> list[dict]:
    meta = load_chat_template_meta()
    templates = [{
        "id": "",
        "name": "Model default",
        "description": "Use the chat template embedded in the GGUF/model metadata.",
        "builtin": True,
        "updated_at": 0,
    }]
    if CHAT_TEMPLATES_DIR.is_dir():
        for path in sorted(CHAT_TEMPLATES_DIR.glob("*.jinja")):
            template_id = path.stem
            item = meta.get(template_id, {}) if isinstance(meta.get(template_id), dict) else {}
            templates.append({
                "id": template_id,
                "name": item.get("name") or template_id,
                "description": item.get("description", ""),
                "builtin": False,
                "updated_at": item.get("updated_at", int(path.stat().st_mtime)),
            })
    return templates


def list_gguf_files() -> list:
    """Return all .gguf files in the models directory."""
    files = []
    if MODELS_DIR.is_dir():
        for f in sorted(MODELS_DIR.rglob("*.gguf")):
            if f.name.startswith('.'):
                continue  # skip macOS resource forks and hidden files
            files.append({
                "path": str(f),
                "name": f.name,
                "size_gb": round(f.stat().st_size / (1024**3), 2),
                "relative": str(f.relative_to(MODELS_DIR)),
                "is_mmproj": is_mmproj_gguf(f.name, f.stat().st_size),
            })
    return files


def is_mmproj_gguf(filename: str, size_bytes: int | None = None) -> bool:
    name = Path(filename).name.lower()
    path_text = str(filename).lower()
    if any(token in path_text for token in ("mmproj", "mm_project", "projector")):
        return True
    if ("clip" in name or "vision" in name) and size_bytes and size_bytes < 2 * 1024**3:
        return True
    return False


def slugify_repo_ref(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-") or "model"


def validate_transcription_engine_id(engine_id: str) -> dict:
    engine = TRANSCRIPTION_ENGINE_BY_ID.get((engine_id or "").strip())
    if not engine:
        raise ValueError("Unknown transcription engine")
    return engine


def transcription_engine_models_dir(engine_id: str) -> Path:
    validate_transcription_engine_id(engine_id)
    return TRANSCRIPTION_MODELS_DIR / engine_id


def transcription_model_storage_dir(engine_id: str, repo_ref: dict) -> Path:
    repo_slug = slugify_repo_ref(repo_ref["repo_id"].replace("/", "--"))
    revision = repo_ref.get("revision") or "main"
    base_dir = transcription_engine_models_dir(engine_id)
    if revision == "main":
        return base_dir / repo_slug
    return base_dir / f"{repo_slug}--{slugify_repo_ref(revision)}"


def transcription_model_dir_info(path: Path) -> dict | None:
    if not path.is_dir():
        return None
    ctranslate2_markers = ("model.bin", "config.json", "tokenizer.json", "preprocessor_config.json")
    if any((path / name).exists() for name in ctranslate2_markers):
        return {
            "runtime": "faster-whisper",
            "format": "ctranslate2",
            "supported_local": True,
        }
    nemo_files = sorted(path.glob("*.nemo"))
    if nemo_files:
        return {
            "runtime": "nemo",
            "format": "nemo",
            "supported_local": True,
            "primary_file": nemo_files[0].name,
        }
    return None


def format_transcription_model_value(kind: str, value: str) -> str:
    return f"{kind}:{value}"


def parse_transcription_model_value(value: str) -> dict[str, str]:
    raw = (value or "").strip()
    if not raw:
        return {"kind": "", "value": ""}
    if raw.startswith("preset:"):
        return {"kind": "preset", "value": raw.split(":", 1)[1]}
    if raw.startswith("local:"):
        return {"kind": "local", "value": raw.split(":", 1)[1]}
    return {"kind": "legacy", "value": raw}


def list_transcription_models(engine_id: str) -> list[dict]:
    engine = validate_transcription_engine_id(engine_id)
    items = []
    seen_values = set()
    base_dir = transcription_engine_models_dir(engine_id)

    for preset in TRANSCRIPTION_MODEL_PRESETS:
        value = format_transcription_model_value("preset", preset)
        items.append({
            "value": value,
            "label": preset,
            "kind": "preset",
            "path": "",
            "relative": "",
            "source": f"{engine['label']} preset",
        })
        seen_values.add(value)

    if base_dir.is_dir():
        for path in sorted(base_dir.rglob("*")):
            info = transcription_model_dir_info(path)
            if not info:
                continue
            value = format_transcription_model_value("local", str(path))
            if value in seen_values:
                continue
            seen_values.add(value)
            try:
                relative = str(path.relative_to(base_dir))
            except ValueError:
                relative = path.name
            items.append({
                "value": value,
                "label": relative,
                "kind": "local",
                "path": str(path),
                "relative": relative,
                "format": info["format"],
                "runtime": info["runtime"],
                "supported_local": info["supported_local"],
                "primary_file": info.get("primary_file", ""),
                "source": f"{engine['label']} local folder",
            })
    return items


def transcript_engine_capabilities(env: dict | None = None) -> dict[str, dict[str, bool]]:
    env = env or read_env()
    result = {}
    for engine in TRANSCRIPTION_ENGINES:
        prefix = engine["env_prefix"]
        backend_type = (env.get(f"{prefix}_BACKEND_TYPE", "upstream") or "upstream").strip().lower()
        local_info = None
        local_value = (env.get(f"{prefix}_LOCAL_MODEL", "") or "").strip()
        if local_value.startswith("local:"):
            local_path = Path(local_value.split(":", 1)[1])
            local_info = transcription_model_dir_info(local_path) if local_path.is_dir() else (
                {"runtime": "nemo", "format": "nemo", "supported_local": True, "primary_file": local_path.name}
                if local_path.is_file() and local_path.suffix.lower() == ".nemo" else None
            )
        result[engine["id"]] = {
            "supports_streaming": (
                env.get(f"{prefix}_SUPPORTS_STREAMING", "").strip().lower() in {"1", "true", "yes", "on"}
                or (engine["id"] == "parakeet-v3" and backend_type == "local" and bool(local_info and local_info.get("runtime") == "nemo"))
            ),
            "supports_speaker_detection": (env.get(f"{prefix}_SUPPORTS_SPEAKER_DETECTION", "off").strip().lower() in {"1", "true", "yes", "on"}),
        }
    return result


def normalize_custom_arg_entries(values) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized = []
    for value in values:
        text = str(value).strip()
        if text:
            normalized.append(text)
    return normalized


def validate_custom_arg_entries(values: list[str]):
    for value in values:
        shlex.split(value)


def infer_model_arg_family(*texts) -> str:
    merged = " ".join(str(text or "") for text in texts).lower()
    normalized = re.sub(r"[^a-z0-9]+", "", merged)
    if re.search(r"qwen[\s._-]*3[\s._-]*6", merged) or "qwen36" in normalized:
        return "qwen3.6"
    if re.search(r"gemma[\s._-]*4\b", merged) or "gemma4" in normalized:
        return "gemma4"
    return ""


def format_model_arg_family_label(family: str) -> str:
    if family == "qwen3.6":
        return "Qwen 3.6"
    if family == "gemma4":
        return "Gemma 4"
    return family or "Custom"


def load_custom_model_arg_presets() -> dict[str, list[str]]:
    if CUSTOM_MODEL_ARG_PRESETS_FILE.exists():
        try:
            data = json.loads(CUSTOM_MODEL_ARG_PRESETS_FILE.read_text())
            if isinstance(data, dict):
                return {
                    str(key): normalize_custom_arg_entries(value)
                    for key, value in data.items()
                }
        except Exception:
            return {}
    return {}


def save_custom_model_arg_presets(presets: dict[str, list[str]]):
    payload = {
        str(key): normalize_custom_arg_entries(value)
        for key, value in presets.items()
    }
    CUSTOM_MODEL_ARG_PRESETS_FILE.write_text(json.dumps(payload, indent=2))


def resolve_custom_args_for_family(family: str) -> tuple[list[str], str]:
    if not family:
        return [], "none"
    presets = load_custom_model_arg_presets()
    if family in presets:
        return presets[family], "family"
    if family in BUILTIN_CUSTOM_MODEL_ARG_PRESETS:
        return BUILTIN_CUSTOM_MODEL_ARG_PRESETS[family], "builtin"
    return [], "none"


def resolve_custom_args_for_model(model: dict) -> tuple[list[str], str, str]:
    family = model.get("arg_family", "")
    if not family:
        family = infer_model_arg_family(
            model.get("display_name", ""),
            model.get("model_name", ""),
            model.get("model_path", ""),
        )
    family_args, source = resolve_custom_args_for_family(family)
    if family_args:
        return family_args, family, source
    model_args = normalize_custom_arg_entries(model.get("custom_args", []))
    if model_args:
        return model_args, family, "model"
    return [], family, "none"


def normalize_custom_model(model: dict) -> dict:
    item = dict(model)
    item["custom_args"] = normalize_custom_arg_entries(item.get("custom_args", []))
    item["arg_family"] = item.get("arg_family") or infer_model_arg_family(
        item.get("display_name", ""),
        item.get("model_name", ""),
        item.get("model_path", ""),
    )
    resolved_args, family, source = resolve_custom_args_for_model(item)
    item["arg_family"] = family
    item["arg_family_label"] = format_model_arg_family_label(family) if family else ""
    item["resolved_custom_args"] = resolved_args
    item["custom_arg_source"] = source
    return item


def display_name_from_model_path(model_path: str) -> str:
    name = Path(model_path or "").name
    if name.lower().endswith(".gguf"):
        name = name[:-5]
    return name or "Custom Model"


def model_name_from_display_name(display_name: str) -> str:
    slug = re.sub(r"[^a-z0-9._-]+", "-", (display_name or "").strip().lower())
    slug = re.sub(r"-+", "-", slug).strip("-._")
    return slug or "custom-model"


def load_custom_models() -> list:
    """Load custom model definitions from JSON file."""
    if CUSTOM_MODELS_FILE.exists():
        try:
            data = json.loads(CUSTOM_MODELS_FILE.read_text())
            if isinstance(data, list):
                return [normalize_custom_model(item) for item in data if isinstance(item, dict)]
        except Exception:
            return []
    return []


def save_custom_models_file(models: list):
    """Write custom model definitions to JSON file."""
    payload = []
    for model in models:
        item = dict(model)
        item["custom_args"] = normalize_custom_arg_entries(item.get("custom_args", []))
        item["arg_family"] = item.get("arg_family") or infer_model_arg_family(
            item.get("display_name", ""),
            item.get("model_name", ""),
            item.get("model_path", ""),
        )
        item.pop("resolved_custom_args", None)
        item.pop("custom_arg_source", None)
        item.pop("arg_family_label", None)
        payload.append(item)
    CUSTOM_MODELS_FILE.write_text(json.dumps(payload, indent=2))


def huggingface_headers() -> dict[str, str]:
    headers = {"User-Agent": "llm-stack-manager/1.0"}
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def parse_huggingface_repo_ref(value: str) -> dict:
    raw = (value or "").strip()
    if not raw:
        raise ValueError("Hugging Face repo URL or repo id is required")

    if re.fullmatch(r"[\w.-]+/[\w.-]+", raw):
        return {
            "repo_id": raw,
            "revision": "main",
            "repo_url": f"https://huggingface.co/{raw}",
        }

    parsed = urlparse(raw)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Hugging Face URL must start with http:// or https://")
    host = (parsed.netloc or "").lower()
    if host not in HF_ALLOWED_HOSTS:
        raise ValueError("URL must point to huggingface.co")

    parts = [unquote(part) for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        raise ValueError("Could not determine repo id from Hugging Face URL")

    repo_id = f"{parts[0]}/{parts[1]}"
    revision = "main"
    if len(parts) >= 4 and parts[2] in {"tree", "blob", "resolve"}:
        revision = parts[3] or "main"

    return {
        "repo_id": repo_id,
        "revision": revision,
        "repo_url": f"https://huggingface.co/{repo_id}",
    }


def list_huggingface_repo_files(repo_ref: dict) -> list[dict]:
    repo_id = repo_ref["repo_id"]
    revision = repo_ref.get("revision") or "main"
    api_url = f"https://huggingface.co/api/models/{quote(repo_id, safe='/')}"
    if revision and revision != "main":
        api_url += f"/revision/{quote(revision, safe='')}"
    req = urlrequest.Request(api_url, headers=huggingface_headers())
    with urlrequest.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    siblings = payload.get("siblings") or []
    files = []
    for item in siblings:
        filename = item.get("rfilename") or item.get("path") or item.get("name")
        if not filename:
            continue
        files.append({
            "path": filename,
            "name": Path(filename).name,
            "size": item.get("size"),
        })
    return files


def score_mmproj_match(model_name: str, mmproj_name: str) -> tuple[int, int]:
    def tokens(text: str) -> set[str]:
        parts = re.split(r"[^a-z0-9]+", text.lower())
        ignored = {
            "gguf", "mmproj", "model", "q2", "q3", "q4", "q5", "q6", "q8",
            "k", "m", "s", "xs", "l", "xl", "xxl", "f16", "bf16",
        }
        return {part for part in parts if part and part not in ignored}

    model_tokens = tokens(Path(model_name).stem)
    mmproj_tokens = tokens(Path(mmproj_name).stem)
    overlap = len(model_tokens & mmproj_tokens)
    return overlap, -len(mmproj_name)


def choose_matching_mmproj_file(model_file: str, repo_files: list[dict]) -> str:
    mmproj_candidates = [
        item["path"]
        for item in repo_files
        if item["name"].lower().endswith(".gguf") and is_mmproj_gguf(item["path"], item.get("size"))
    ]
    if not mmproj_candidates:
        return ""
    if len(mmproj_candidates) == 1:
        return mmproj_candidates[0]
    return max(mmproj_candidates, key=lambda candidate: score_mmproj_match(model_file, candidate))


def derive_mmproj_target_name(model_filename: str) -> str:
    model_path = Path(model_filename)
    return f"{model_path.stem}.mmproj{model_path.suffix}"


def build_huggingface_download_url(repo_ref: dict, repo_file: str) -> str:
    repo_id = quote(repo_ref["repo_id"], safe="/")
    revision = quote(repo_ref.get("revision") or "main", safe="")
    file_path = quote(repo_file, safe="/")
    return f"https://huggingface.co/{repo_id}/resolve/{revision}/{file_path}?download=true"


def update_hf_download_job(job_id: str, **changes):
    with HF_DOWNLOAD_JOBS_LOCK:
        job = HF_DOWNLOAD_JOBS.get(job_id)
        if not job:
            return
        job.update(changes)
        job["updated_at"] = int(time.time())


def stream_download_to_path(url: str, dest_path: Path, job_id: str | None = None, label: str = ""):
    req = urlrequest.Request(url, headers=huggingface_headers())
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_name(dest_path.name + ".part")
    downloaded = 0
    total = None
    with urlrequest.urlopen(req, timeout=300) as resp, tmp_path.open("wb") as fh:
        total_header = resp.headers.get("Content-Length")
        try:
            total = int(total_header) if total_header else None
        except ValueError:
            total = None
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
            downloaded += len(chunk)
            if job_id:
                progress = None
                if total:
                    progress = round((downloaded / total) * 100, 1)
                update_hf_download_job(
                    job_id,
                    current_file=label or dest_path.name,
                    current_bytes=downloaded,
                    total_bytes=total,
                    progress=progress,
                )
    tmp_path.replace(dest_path)


def download_huggingface_model_bundle(repo_ref: dict, model_file: str, mmproj_file: str = "", job_id: str | None = None) -> dict:
    model_target = MODELS_DIR / Path(model_file).name
    stream_download_to_path(
        build_huggingface_download_url(repo_ref, model_file),
        model_target,
        job_id=job_id,
        label=Path(model_file).name,
    )

    mmproj_target = None
    if mmproj_file:
        mmproj_target = MODELS_DIR / derive_mmproj_target_name(Path(model_file).name)
        stream_download_to_path(
            build_huggingface_download_url(repo_ref, mmproj_file),
            mmproj_target,
            job_id=job_id,
            label=Path(mmproj_file).name,
        )

    return {
        "model_path": str(model_target),
        "mmproj_path": str(mmproj_target) if mmproj_target else "",
        "model_name": Path(model_file).name,
        "mmproj_name": mmproj_target.name if mmproj_target else "",
    }


def run_hf_download_job(job_id: str, repo_ref: dict, model_file: str, mmproj_file: str):
    try:
        update_hf_download_job(job_id, status="running", stage="Downloading model bundle")
        result = download_huggingface_model_bundle(repo_ref, model_file, mmproj_file, job_id=job_id)
        update_hf_download_job(
            job_id,
            ok=True,
            status="done",
            stage="Completed",
            progress=100.0,
            current_file="",
            result=result,
        )
    except Exception as exc:
        update_hf_download_job(
            job_id,
            ok=False,
            status="error",
            error=str(exc),
            stage="Failed",
        )


def list_transcription_repo_download_files(repo_ref: dict) -> list[dict]:
    files = []
    for item in list_huggingface_repo_files(repo_ref):
        path = item.get("path") or ""
        name = item.get("name") or ""
        if not path or not name or name.startswith("."):
            continue
        if path.startswith(".") or "/." in path:
            continue
        if name in {".gitattributes", "README.md"}:
            continue
        files.append(item)
    if not files:
        raise ValueError("No downloadable transcription model files were found in this repo")
    return files


def download_huggingface_transcription_model(engine_id: str, repo_ref: dict, job_id: str | None = None) -> dict:
    engine = validate_transcription_engine_id(engine_id)
    target_dir = transcription_model_storage_dir(engine_id, repo_ref)
    repo_files = list_transcription_repo_download_files(repo_ref)
    total_files = len(repo_files)
    for index, item in enumerate(repo_files, start=1):
        relative_path = item["path"]
        update_hf_download_job(
            job_id,
            stage=f"Downloading file {index}/{total_files}",
            current_file=relative_path,
        )
        stream_download_to_path(
            build_huggingface_download_url(repo_ref, relative_path),
            target_dir / relative_path,
            job_id=job_id,
            label=relative_path,
        )
    return {
        "engine_id": engine_id,
        "engine_label": engine["label"],
        "model_dir": str(target_dir),
        "model_value": format_transcription_model_value("local", str(target_dir)),
        "model_label": str(target_dir.relative_to(transcription_engine_models_dir(engine_id))) if target_dir.exists() else target_dir.name,
        "file_count": total_files,
        "repo_id": repo_ref["repo_id"],
        "revision": repo_ref["revision"],
    }


def run_hf_transcription_download_job(job_id: str, engine_id: str, repo_ref: dict):
    try:
        update_hf_download_job(job_id, status="running", stage="Downloading transcription model")
        result = download_huggingface_transcription_model(engine_id, repo_ref, job_id=job_id)
        update_hf_download_job(
            job_id,
            ok=True,
            status="done",
            stage="Completed",
            progress=100.0,
            current_file="",
            result=result,
        )
    except Exception as exc:
        update_hf_download_job(
            job_id,
            ok=False,
            status="error",
            error=str(exc),
            stage="Failed",
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    env = read_env()
    groups = defaultdict(list)
    for svc in patch_service_labels(env):
        groups[svc['group']].append(svc)
    sections = defaultdict(list)
    for f in CONFIG_FIELDS:
        if f.get('section') in CORE_CONFIG_SECTIONS:
            sections[f['section']].append(f)
    return render_template('index.html',
                           service_groups=dict(groups),
                           config_sections=dict(sections),
                           custom_models=load_custom_models(),
                           builtin_chat_variants=builtin_chat_variants(env),
                           models_dir=str(MODELS_DIR))


@app.route('/api/status')
def api_status():
    statuses = {s['name']: get_service_status(s['name']) for s in patch_service_labels()}
    return jsonify(services=statuses, gpus=get_gpu_info())


@app.route('/api/service/<name>/<action>', methods=['POST'])
def api_service_action(name, action):
    if action not in ('start', 'stop', 'restart'):
        return jsonify(ok=False, error='Unknown action'), 400
    if name not in {s['name'] for s in patch_service_labels()}:
        return jsonify(ok=False, error='Unknown service'), 400
    if is_searxng_service(name):
        ok, output = run_searxng_manager(action)
        return jsonify(ok=ok, output=output)
    if should_use_local_transcript_manager(name):
        ok, output = run_transcript_manager(action)
        return jsonify(ok=ok, output=output)
    if should_use_local_tts_manager(name):
        ok, output = run_tts_manager(name, action)
        return jsonify(ok=ok, output=output)
    try:
        rc, output = ServiceManager.action(action, name, timeout=30)
        return jsonify(ok=(rc == 0), output=output)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.route('/api/searxng/status')
def api_searxng_status():
    cfg = searxng_config()
    checks = {
        "uwsgi": {"ok": get_service_status("uwsgi") == "active", "status": get_service_status("uwsgi")},
        "nginx": {"ok": get_service_status("nginx") == "active", "status": get_service_status("nginx")},
        "settings": {"ok": Path(cfg["settings_path"]).exists(), "path": cfg["settings_path"]},
        "uwsgi_ini": {"ok": Path(cfg["uwsgi_ini"]).exists(), "path": cfg["uwsgi_ini"]},
        "socket": {"ok": Path(cfg["uwsgi_socket"]).exists(), "path": cfg["uwsgi_socket"]},
        "nginx_conf": {"ok": Path(cfg["nginx_conf"]).exists(), "path": cfg["nginx_conf"]},
        "search_api": {"ok": False, "error": ""},
    }
    try:
        result = http_json(f"{cfg['local_url']}search?q=llm-stack&format=json", timeout=8)
        checks["search_api"]["ok"] = isinstance(result, dict) and "results" in result
        checks["search_api"]["result_count"] = len(result.get("results", [])) if isinstance(result, dict) else 0
    except Exception as exc:
        checks["search_api"]["error"] = str(exc)
    return jsonify(ok=True, service_status=get_service_status("searxng"), config=cfg, checks=checks, last_refresh=int(time.time()))


@app.route('/api/searxng/install', methods=['POST'])
def api_searxng_install():
    ok, output = run_searxng_manager("install")
    return jsonify(ok=ok, output=output)


@app.route('/api/playwright/status')
def api_playwright_status():
    cfg = playwright_config()
    port_ok, port_error = tcp_port_open(cfg["host"], cfg["port"])
    checks = {
        "service": {"ok": get_service_status("playwright-server") == "active", "status": get_service_status("playwright-server")},
        "unit": {"ok": Path(cfg["service_unit"]).exists(), "path": cfg["service_unit"]},
        "package_json": {"ok": Path(cfg["package_json"]).exists(), "path": cfg["package_json"]},
        "node_modules": {"ok": Path(cfg["node_modules"]).exists(), "path": cfg["node_modules"]},
        "browser_cache": playwright_browser_cache_status(cfg),
        "nginx_conf": {"ok": Path(cfg["nginx_conf"]).exists(), "path": cfg["nginx_conf"]},
        "tcp_listener": {"ok": port_ok, "error": port_error, "endpoint": cfg["public_ws_url"]},
    }
    return jsonify(ok=True, service_status=get_service_status("playwright-server"), config=cfg, checks=checks, last_refresh=int(time.time()))


@app.route('/api/playwright/install', methods=['POST'])
def api_playwright_install():
    ok, output = run_playwright_install()
    return jsonify(ok=ok, output=output)



def _ocr_backend_url(env: dict) -> str:
    host = env.get("OCR_HOST") or env.get("LISTEN_HOST") or "127.0.0.1"
    if host in {"${LISTEN_HOST}", "$LISTEN_HOST"}:
        host = env.get("LISTEN_HOST") or "127.0.0.1"
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1"
    port = env.get("OCR_PORT", "8009")
    return f"http://{host}:{port}/v1/chat/completions"


def _glmocr_backend_url(env: dict) -> str:
    public_url = (env.get("GLMOCR_PUBLIC_URL") or "").strip()
    if public_url:
        return public_url
    host = env.get("GLMOCR_SDK_HOST") or "127.0.0.1"
    if host in {"${LISTEN_HOST}", "$LISTEN_HOST"}:
        host = env.get("LISTEN_HOST") or "127.0.0.1"
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1"
    port = env.get("GLMOCR_SDK_PORT", "5002")
    return f"http://{host}:{port}/glmocr/parse"


def request_origin_for_tool(ws: bool = False) -> str:
    if not has_request_context():
        return "ws://127.0.0.1" if ws else "http://127.0.0.1"
    scheme = request.headers.get("X-Forwarded-Proto", request.scheme)
    host = request.headers.get("X-Forwarded-Host", request.host)
    parsed_host = urlparse(f"//{host}")
    hostname = parsed_host.hostname or host
    port = parsed_host.port
    manager_port = str(read_env().get("LLM_MANAGER_PORT", "8077"))
    if port and str(port) != manager_port:
        host = f"{hostname}:{port}"
    else:
        host = hostname
    if ws:
        scheme = "wss" if scheme == "https" else "ws"
    return f"{scheme}://{host}"


def browser_tool_url(configured: str, url_path: str, ws: bool = False) -> str:
    configured = (configured or "").strip()
    if configured:
        parsed = urlparse(configured)
        if parsed.hostname not in {"127.0.0.1", "localhost", "0.0.0.0"}:
            return configured if configured.endswith("/") else configured + "/"
    path = url_path if url_path.startswith("/") else f"/{url_path}"
    return f"{request_origin_for_tool(ws)}{path}/"


def searxng_config(env: dict | None = None) -> dict:
    env = env or read_env()
    url_path = env.get("SEARXNG_URL_PATH", "/searxng") or "/searxng"
    if not url_path.startswith("/"):
        url_path = "/" + url_path
    configured_public_url = env.get("SEARXNG_PUBLIC_URL") or env.get("SEARXNG_BASE_URL") or f"http://127.0.0.1{url_path}/"
    public_url = browser_tool_url(configured_public_url, url_path)
    if not public_url.endswith("/"):
        public_url += "/"
    endpoints = {
        "html": public_url,
        "json_search": f"{public_url}search?q=<query>&format=json",
        "html_search": f"{public_url}search?q=<query>",
        "opensearch": f"{public_url}opensearch.xml",
        "preferences": f"{public_url}preferences",
    }
    return {
        "enabled": env.get("SEARXNG_ENABLED", "on"),
        "public_url": public_url,
        "base_url": env.get("SEARXNG_BASE_URL", public_url),
        "local_url": f"http://127.0.0.1{url_path}/",
        "url_path": url_path,
        "settings_path": env.get("SEARXNG_SETTINGS_PATH", "/etc/searxng/settings.yml"),
        "uwsgi_ini": env.get("SEARXNG_UWSGI_INI", "/etc/uwsgi/apps-available/searxng.ini"),
        "uwsgi_socket": env.get("SEARXNG_UWSGI_SOCKET", "/usr/local/searxng/run/socket"),
        "nginx_conf": env.get("SEARXNG_NGINX_CONF", "/etc/nginx/default.apps-available/searxng.conf"),
        "home": env.get("SEARXNG_HOME", "/usr/local/searxng"),
        "formats": env.get("SEARXNG_FORMATS", "html,json"),
        "endpoints": endpoints,
    }


def playwright_config(env: dict | None = None) -> dict:
    env = env or read_env()
    port = env.get("PLAYWRIGHT_PORT", "3001")
    upstream_port = env.get("PLAYWRIGHT_UPSTREAM_PORT") or str(int(port) + 10000 if str(port).isdigit() else 13001)
    url_path = env.get("PLAYWRIGHT_URL_PATH", "/playwright") or "/playwright"
    if not url_path.startswith("/"):
        url_path = "/" + url_path
    public_ws = browser_tool_url(env.get("PLAYWRIGHT_PUBLIC_WS_URL") or f"ws://127.0.0.1{url_path}/", url_path, ws=True)
    public_http = browser_tool_url(env.get("PLAYWRIGHT_PUBLIC_HTTP_URL") or f"http://127.0.0.1{url_path}/", url_path)
    if not public_ws.endswith("/"):
        public_ws += "/"
    if not public_http.endswith("/"):
        public_http += "/"
    endpoints = {
        "protocol": "Playwright remote protocol, not Chrome DevTools Protocol",
        "websocket": public_ws,
        "http": public_http,
        "node_connect": f"const browser = await playwright.chromium.connect('{public_ws}');",
        "python_connect": f"browser = playwright.chromium.connect('{public_ws}')",
        "not_cdp": "Do not use chromium.connectOverCDP(...) with this endpoint",
    }
    return {
        "enabled": env.get("PLAYWRIGHT_ENABLED", "on"),
        "host": env.get("PLAYWRIGHT_HOST", "0.0.0.0"),
        "port": port,
        "upstream_port": upstream_port,
        "url_path": url_path,
        "browser": env.get("PLAYWRIGHT_BROWSER", "chromium"),
        "install_browsers": env.get("PLAYWRIGHT_INSTALL_BROWSERS", "on"),
        "browsers_path": env.get("PLAYWRIGHT_BROWSERS_PATH", str(STACK_DIR / "playwright" / "browsers")),
        "node_env": env.get("PLAYWRIGHT_NODE_ENV", "production"),
        "public_ws_url": public_ws,
        "public_http_url": public_http,
        "server_dir": str(STACK_DIR / "playwright"),
        "package_json": str(STACK_DIR / "playwright" / "package.json"),
        "node_modules": str(STACK_DIR / "playwright" / "node_modules"),
        "service_unit": "/etc/systemd/system/playwright-server.service",
        "nginx_conf": env.get("PLAYWRIGHT_NGINX_CONF", "/etc/nginx/default.apps-available/playwright.conf"),
        "endpoints": endpoints,
    }


def tcp_port_open(host: str, port: str | int, timeout: float = 1.5) -> tuple[bool, str]:
    try:
        target_host = "127.0.0.1" if host in {"0.0.0.0", "::", ""} else host
        with socket.create_connection((target_host, int(port)), timeout=timeout):
            return True, ""
    except Exception as exc:
        return False, str(exc)


def playwright_browser_cache_candidates(cfg: dict) -> list[Path]:
    candidates = [Path(cfg["browsers_path"])]
    home = os.environ.get("HOME")
    if home:
        candidates.append(Path(home) / ".cache" / "ms-playwright")
    try:
        user, _ = stack_owner_user_group()
        if pwd is not None:
            candidates.append(Path(pwd.getpwnam(user).pw_dir) / ".cache" / "ms-playwright")
    except Exception:
        pass
    deduped = []
    seen = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            deduped.append(path)
            seen.add(key)
    return deduped


def playwright_browser_cache_status(cfg: dict) -> dict:
    candidates = playwright_browser_cache_candidates(cfg)
    for path in candidates:
        if path.exists() and any(path.iterdir()):
            return {"ok": True, "path": str(path), "configured_path": cfg["browsers_path"]}
    return {
        "ok": False,
        "path": cfg["browsers_path"],
        "candidates": [str(path) for path in candidates],
        "error": "No installed Playwright browser cache found",
    }


def stack_owner_user_group() -> tuple[str, str]:
    if pwd is None or grp is None:
        return os.environ.get("USER", "root"), os.environ.get("USER", "root")
    stat = STACK_DIR.stat()
    return pwd.getpwuid(stat.st_uid).pw_name, grp.getgrgid(stat.st_gid).gr_name


def write_playwright_systemd_unit(cfg: dict | None = None) -> None:
    cfg = cfg or playwright_config()
    user, group = stack_owner_user_group()
    unit = f"""[Unit]
Description=Playwright WebSocket Server
After=network.target

[Service]
Type=simple
User={user}
Group={group}
WorkingDirectory={STACK_DIR / 'playwright'}
EnvironmentFile={CONFIG_FILE}
Environment=NODE_ENV={cfg['node_env']}
ExecStart={STACK_DIR / 'playwright' / 'start.sh'}
Restart=on-failure
RestartSec=5
TimeoutStartSec=60
StandardOutput=journal
StandardError=journal
SyslogIdentifier=playwright-server

[Install]
WantedBy=multi-user.target
"""
    unit_path = Path(cfg["service_unit"])
    unit_path.write_text(unit, encoding="utf-8")
    unit_path.chmod(0o644)
    ServiceManager.run_cmd(["systemctl", "daemon-reload"], timeout=15)


def write_playwright_nginx_conf(cfg: dict | None = None) -> None:
    cfg = cfg or playwright_config()
    conf_path = Path(cfg["nginx_conf"])
    conf_path.parent.mkdir(parents=True, exist_ok=True)
    Path("/etc/nginx/default.d").mkdir(parents=True, exist_ok=True)
    url_path = cfg["url_path"].rstrip("/") or "/playwright"
    url_path_slash = f"{url_path}/"
    content = f"""location = {url_path} {{
    return 308 {url_path_slash};
}}

location {url_path_slash} {{
    proxy_pass http://127.0.0.1:{cfg['port']}/;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_set_header X-Forwarded-Prefix {url_path};
    proxy_set_header X-Script-Name {url_path};
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_read_timeout 3600s;
    proxy_send_timeout 3600s;
}}
"""
    conf_path.write_text(content, encoding="utf-8")
    conf_path.chmod(0o644)
    link_path = Path("/etc/nginx/default.d/playwright.conf")
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    link_path.symlink_to(conf_path)
    nginx_test = ServiceManager.run_cmd(["nginx", "-t"], timeout=15)
    if nginx_test.returncode != 0:
        raise RuntimeError((nginx_test.stdout + nginx_test.stderr).strip())
    ServiceManager.run_cmd(["systemctl", "reload", "nginx"], timeout=15)


def run_playwright_install() -> tuple[bool, str]:
    cfg = playwright_config()
    env = os.environ.copy()
    env.update({
        "PLAYWRIGHT_ENABLED": cfg["enabled"],
        "PLAYWRIGHT_BROWSER": cfg["browser"],
        "PLAYWRIGHT_INSTALL_BROWSERS": cfg["install_browsers"],
        "PLAYWRIGHT_BROWSERS_PATH": cfg["browsers_path"],
    })
    cmd = ["bash", str(SCRIPTS_DIR / "install-playwright.sh")]
    try:
        if os.geteuid() == 0 and pwd is not None:
            user, _ = stack_owner_user_group()
            cmd = [
                "sudo", "-u", user, "env",
                f"PLAYWRIGHT_ENABLED={cfg['enabled']}",
                f"PLAYWRIGHT_BROWSER={cfg['browser']}",
                f"PLAYWRIGHT_INSTALL_BROWSERS={cfg['install_browsers']}",
                f"PLAYWRIGHT_BROWSERS_PATH={cfg['browsers_path']}",
                *cmd,
            ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
        output = (r.stdout + r.stderr).strip()
        if r.returncode != 0:
            return False, output
        if not ServiceManager.IS_MAC and os.geteuid() == 0:
            write_playwright_systemd_unit(cfg)
            write_playwright_nginx_conf(cfg)
        return True, output
    except subprocess.TimeoutExpired:
        return False, "Playwright install timed out"
    except Exception as exc:
        return False, str(exc)


def _normalize_ocr_parse_response(payload: dict) -> dict:
    text = payload.get("markdown_result") or payload.get("md_results") or payload.get("text") or ""
    return {
        "ok": True,
        "text": text,
        "markdown_result": payload.get("markdown_result", text),
        "md_results": payload.get("md_results", text),
        "json_result": payload.get("json_result"),
        "layout_details": payload.get("layout_details"),
        "layout_visualization": payload.get("layout_visualization", []),
        "data_info": payload.get("data_info", {}),
        "usage": payload.get("usage", {}),
        "raw": payload,
    }


def _extract_chat_text(payload: dict) -> str:
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not choices:
        return ""
    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return ""


@app.route('/api/ocr/parse', methods=['POST'])
def api_ocr_parse():
    env = read_env()
    data = request.get_json(silent=True) or {}
    images = data.get("images")
    if isinstance(images, str):
        images = [images]
    elif not isinstance(images, list):
        images = []
    if not images:
        for key in ("file", "image_url"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                images = [value.strip()]
                break
    if not images and isinstance(data.get("image_base64"), str):
        raw = data.get("image_base64", "").strip()
        mime_type = str(data.get("mime_type") or "image/png").strip() or "image/png"
        images = [raw if raw.startswith("data:") else f"data:{mime_type};base64,{raw}"]
    if not images:
        return jsonify(ok=False, error="file, images, image_url, or image_base64 is required"), 400

    payload = {"images": images}
    for key in (
        "file",
        "model",
        "return_crop_images",
        "need_layout_visualization",
        "start_page_id",
        "end_page_id",
        "request_id",
        "user_id",
    ):
        if key in data:
            payload[key] = data[key]
    timeout = int(float(data.get("timeout") or env.get("GLMOCR_OCR_REQUEST_TIMEOUT", "300") or 300))
    req = urlrequest.Request(
        _glmocr_backend_url(env),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and parsed.get("error"):
            return jsonify(ok=False, error=parsed.get("error"), raw=parsed), 502
        return jsonify(_normalize_ocr_parse_response(parsed if isinstance(parsed, dict) else {"result": parsed}))
    except urlerror.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else ""
        return jsonify(ok=False, error=f"GLM-OCR SDK returned HTTP {exc.code}", body=body), 502
    except Exception as exc:
        return jsonify(ok=False, error=str(exc)), 502


@app.route('/api/ocr/extract', methods=['POST'])
def api_ocr_extract():
    env = read_env()
    data = request.get_json(silent=True) or {}
    image_url = str(data.get("image_url") or "").strip()
    image_base64 = str(data.get("image_base64") or "").strip()
    if not image_url and not image_base64:
        return jsonify(ok=False, error="image_url or image_base64 is required"), 400
    if image_base64 and not image_url:
        mime_type = str(data.get("mime_type") or "image/png").strip() or "image/png"
        if image_base64.startswith("data:"):
            image_url = image_base64
        else:
            image_url = f"data:{mime_type};base64,{image_base64}"
    prompt = str(data.get("prompt") or env.get("OCR_PROMPT") or "OCR")
    payload = {
        "model": env.get("OCR_MODEL_NAME", "ocr"),
        "temperature": float(env.get("OCR_TEMP", "0.1") or 0.1),
        "top_p": float(env.get("OCR_TOP_P", "0.95") or 0.95),
        "top_k": int(float(env.get("OCR_TOP_K", "1") or 1)),
        "min_p": float(env.get("OCR_MIN_P", "0.00") or 0.0),
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
    }
    timeout = int(float(data.get("timeout") or env.get("OCR_TIMEOUT_SECONDS", "120") or 120))
    req = urlrequest.Request(
        _ocr_backend_url(env),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        parsed = json.loads(raw)
        return jsonify(ok=True, text=_extract_chat_text(parsed), raw=parsed)
    except urlerror.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else ""
        return jsonify(ok=False, error=f"OCR backend returned HTTP {exc.code}", body=body), 502
    except Exception as exc:
        return jsonify(ok=False, error=str(exc)), 502

@app.route('/api/restore-active-stack', methods=['POST'])
def api_default_mode():
    default_name = get_default_saved_config_name()
    if default_name:
        result = apply_saved_config(default_name, launch=True)
        status = 200 if result.get('ok') else 500
        return jsonify(result), status
    ok, output = run_script('restore-active-stack.sh')
    return jsonify(ok=ok, output=output)


@app.route('/api/app/update', methods=['POST'])
def api_app_update():
    script_path = os.path.join(STACK_DIR, 'update.sh')
    if not os.path.exists(script_path):
        return jsonify(ok=False, error="update.sh not found"), 404

    def run_update():
        import subprocess
        import time
        import os
        import signal
        # Run update synchronously
        subprocess.run(['bash', script_path, '--branch', 'main'])
        # On macOS, update.sh can't restart us directly due to sudo restrictions.
        # But our launchd plist has KeepAlive=true, so we can just cleanly exit
        # and launchd will automatically boot us right back up!
        time.sleep(1)
        os.kill(os.getpid(), signal.SIGTERM)

    import threading
    threading.Thread(target=run_update, daemon=True).start()
    return jsonify(ok=True)

@app.route('/api/llamacpp/update', methods=['POST'])
def api_llamacpp_update():
    ok, output, restarted = update_llamacpp_and_restart_active_services()
    return jsonify(ok=ok, output=output, restarted_services=restarted)


@app.route('/api/switch/<variant>', methods=['POST'])
def api_switch(variant):
    if variant in BUILTIN_CHAT_VARIANT_IDS:
        # Stop generic backend if running, then use existing switch script
        for svc in ('chat-backend', 'qwen-chat-backend', 'qwen-chat-backend-27b', 'qwen-chat-backend-35b'):
            ServiceManager.stop(svc)
        ok, output = run_script('switch-chat-model.sh', variant)
        return jsonify(ok=ok, output=output)

    # Custom model switch
    models = load_custom_models()
    model = next((m for m in models if m['id'] == variant), None)
    if not model:
        return jsonify(ok=False, error='Unknown model variant'), 400

    # Stop all chat backends
    for svc in ('chat-backend-dense', 'chat-backend-moe', 'chat-backend', 'qwen-chat-backend-27b', 'qwen-chat-backend-35b', 'qwen-chat-backend'):
        ServiceManager.stop(svc)

    # Update env with custom model paths
    updates = {
        'CHAT_MODEL_PATH': model['model_path'],
        'CHAT_MODEL_NAME': model.get('model_name', 'chat-custom'),
        'CHAT_CTX_SIZE': model.get('ctx_size', '32768'),
        'CHAT_CUSTOM_ARGS_JSON': json.dumps(resolve_custom_args_for_model(model)[0]),
    }
    if model.get('mmproj_path'):
        updates['CHAT_MMPROJ_PATH'] = model['mmproj_path']
    else:
        updates['CHAT_MMPROJ_PATH'] = ''
    update_env_values(updates)

    # Start generic backend + ensure proxy is running
    try:
        r = subprocess.run(['systemctl', 'start', 'chat-backend'],
                           capture_output=True, text=True, timeout=30)
        subprocess.run(['systemctl', 'start', 'chat-proxy'],
                       capture_output=True, timeout=30)
        return jsonify(ok=(r.returncode == 0),
                       output=(r.stdout + r.stderr).strip())
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.route('/api/gguf-files')
def api_gguf_files():
    """List all .gguf files in the models directory."""
    return jsonify(list_gguf_files())


@app.route('/api/transcription-models/<engine_id>')
def api_transcription_models(engine_id):
    try:
        engine = validate_transcription_engine_id(engine_id)
    except ValueError as exc:
        return jsonify(ok=False, error=str(exc)), 404
    return jsonify({
        "ok": True,
        "engine_id": engine_id,
        "engine_label": engine["label"],
        "directory": str(transcription_engine_models_dir(engine_id)),
        "models": list_transcription_models(engine_id),
    })


@app.route('/api/transcription-capabilities')
def api_transcription_capabilities():
    return jsonify({
        "ok": True,
        "engines": transcript_engine_capabilities(),
    })


@app.route('/api/huggingface/repo-files', methods=['POST'])
def api_huggingface_repo_files():
    data = request.json or {}
    try:
        repo_ref = parse_huggingface_repo_ref(data.get('repo_url', ''))
        files = list_huggingface_repo_files(repo_ref)
    except ValueError as exc:
        return jsonify(ok=False, error=str(exc)), 400
    except urlerror.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore").strip()
        return jsonify(ok=False, error=detail or f"Hugging Face request failed: HTTP {exc.code}"), 502
    except Exception as exc:
        return jsonify(ok=False, error=str(exc)), 500

    gguf_files = [
        file for file in files
        if file["name"].lower().endswith(".gguf")
    ]
    model_files = [
        file for file in gguf_files
        if not is_mmproj_gguf(file["path"], file.get("size"))
    ]
    mmproj_files = [
        file for file in gguf_files
        if is_mmproj_gguf(file["path"], file.get("size"))
    ]
    for file in model_files:
        matched_mmproj = choose_matching_mmproj_file(file["path"], files)
        file["matched_mmproj"] = matched_mmproj
        file["renamed_mmproj"] = derive_mmproj_target_name(file["name"]) if matched_mmproj else ""

    return jsonify({
        "ok": True,
        "repo_id": repo_ref["repo_id"],
        "revision": repo_ref["revision"],
        "repo_url": repo_ref["repo_url"],
        "model_files": model_files,
        "mmproj_files": mmproj_files,
    })


@app.route('/api/huggingface/downloads', methods=['GET'])
def api_huggingface_downloads_list():
    with HF_DOWNLOAD_JOBS_LOCK:
        jobs = sorted(
            (dict(job) for job in HF_DOWNLOAD_JOBS.values()),
            key=lambda item: item.get('created_at', 0),
        )
    return jsonify(ok=True, jobs=jobs)


@app.route('/api/huggingface/downloads', methods=['POST'])
def api_huggingface_download_create():
    data = request.json or {}
    model_file = (data.get('model_file') or '').strip()
    if not model_file:
        return jsonify(ok=False, error='model_file is required'), 400

    try:
        repo_ref = parse_huggingface_repo_ref(data.get('repo_url', ''))
    except ValueError as exc:
        return jsonify(ok=False, error=str(exc)), 400

    job_id = uuid.uuid4().hex[:10]
    job = {
        "id": job_id,
        "ok": False,
        "status": "queued",
        "stage": "Queued",
        "progress": 0.0,
        "repo_id": repo_ref["repo_id"],
        "revision": repo_ref["revision"],
        "model_file": model_file,
        "mmproj_file": (data.get('mmproj_file') or '').strip(),
        "current_file": "",
        "current_bytes": 0,
        "total_bytes": None,
        "created_at": int(time.time()),
        "updated_at": int(time.time()),
        "result": {},
    }
    with HF_DOWNLOAD_JOBS_LOCK:
        HF_DOWNLOAD_JOBS[job_id] = job

    thread = threading.Thread(
        target=run_hf_download_job,
        args=(job_id, repo_ref, job["model_file"], job["mmproj_file"]),
        daemon=True,
    )
    thread.start()
    return jsonify(ok=True, job=job)


@app.route('/api/huggingface/downloads/<job_id>', methods=['GET'])
def api_huggingface_download_status(job_id):
    with HF_DOWNLOAD_JOBS_LOCK:
        job = HF_DOWNLOAD_JOBS.get(job_id)
        if not job:
            return jsonify(ok=False, error='Download job not found'), 404
        return jsonify(ok=True, job=job)


@app.route('/api/huggingface/transcription-repo-files', methods=['POST'])
def api_huggingface_transcription_repo_files():
    data = request.json or {}
    try:
        engine = validate_transcription_engine_id(data.get('engine_id', ''))
        repo_ref = parse_huggingface_repo_ref(data.get('repo_url', ''))
        files = list_transcription_repo_download_files(repo_ref)
    except ValueError as exc:
        return jsonify(ok=False, error=str(exc)), 400
    except urlerror.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore").strip()
        return jsonify(ok=False, error=detail or f"Hugging Face request failed: HTTP {exc.code}"), 502
    except Exception as exc:
        return jsonify(ok=False, error=str(exc)), 500

    return jsonify({
        "ok": True,
        "engine_id": engine["id"],
        "engine_label": engine["label"],
        "repo_id": repo_ref["repo_id"],
        "revision": repo_ref["revision"],
        "repo_url": repo_ref["repo_url"],
        "file_count": len(files),
        "target_dir": str(transcription_model_storage_dir(engine["id"], repo_ref)),
        "sample_files": [item["path"] for item in files[:8]],
    })


@app.route('/api/huggingface/transcription-downloads', methods=['POST'])
def api_huggingface_transcription_download_create():
    data = request.json or {}
    try:
        engine = validate_transcription_engine_id(data.get('engine_id', ''))
        repo_ref = parse_huggingface_repo_ref(data.get('repo_url', ''))
    except ValueError as exc:
        return jsonify(ok=False, error=str(exc)), 400

    job_id = uuid.uuid4().hex[:10]
    job = {
        "id": job_id,
        "ok": False,
        "status": "queued",
        "stage": "Queued",
        "progress": 0.0,
        "repo_id": repo_ref["repo_id"],
        "revision": repo_ref["revision"],
        "current_file": "",
        "current_bytes": 0,
        "total_bytes": None,
        "created_at": int(time.time()),
        "updated_at": int(time.time()),
        "result": {},
        "kind": "transcription-model",
        "engine_id": engine["id"],
    }
    with HF_DOWNLOAD_JOBS_LOCK:
        HF_DOWNLOAD_JOBS[job_id] = job

    thread = threading.Thread(
        target=run_hf_transcription_download_job,
        args=(job_id, engine["id"], repo_ref),
        daemon=True,
    )
    thread.start()
    return jsonify(ok=True, job=job)


@app.route('/api/custom-models', methods=['GET'])
def api_custom_models_list():
    return jsonify(load_custom_models())


@app.route('/api/custom-models', methods=['POST'])
def api_custom_models_add():
    data = request.json or {}
    model_path = (data.get('model_path') or '').strip()
    if not model_path:
        return jsonify(ok=False, error='model_path is required'), 400
    display_name = (data.get('display_name') or '').strip() or display_name_from_model_path(model_path)
    model_name = (data.get('model_name') or '').strip() or model_name_from_display_name(display_name)
    custom_args_supplied = 'custom_args' in data
    custom_args = normalize_custom_arg_entries(data.get('custom_args', []))
    try:
        validate_custom_arg_entries(custom_args)
    except ValueError as exc:
        return jsonify(ok=False, error=f'invalid custom argument: {exc}'), 400
    family = infer_model_arg_family(
        display_name,
        model_name,
        model_path,
    )
    if not custom_args_supplied and family:
        custom_args, _ = resolve_custom_args_for_family(family)
    model = {
        'id': str(uuid.uuid4())[:8],
        'display_name': display_name,
        'model_name': model_name,
        'model_path': model_path,
        'mmproj_path': data.get('mmproj_path', ''),
        'ctx_size': str(data.get('ctx_size', '32768')),
        'custom_args': custom_args,
        'arg_family': family,
        'created': int(time.time()),
    }
    if family and custom_args_supplied:
        presets = load_custom_model_arg_presets()
        presets[family] = custom_args
        save_custom_model_arg_presets(presets)
    models = load_custom_models()
    models.append(model)
    save_custom_models_file(models)
    return jsonify(ok=True, model=normalize_custom_model(model))


@app.route('/api/custom-models/<model_id>', methods=['PUT'])
def api_custom_models_update(model_id):
    data = request.json or {}
    custom_args_supplied = 'custom_args' in (data or {})
    custom_args = normalize_custom_arg_entries((data or {}).get('custom_args', []))
    if custom_args_supplied:
        try:
            validate_custom_arg_entries(custom_args)
        except ValueError as exc:
            return jsonify(ok=False, error=f'invalid custom argument: {exc}'), 400
    models = load_custom_models()
    for m in models:
        if m['id'] == model_id:
            for k in ('display_name', 'model_name', 'model_path',
                       'mmproj_path', 'ctx_size'):
                if k in data:
                    m[k] = data[k]
            family = infer_model_arg_family(
                m.get('display_name', ''),
                m.get('model_name', ''),
                m.get('model_path', ''),
            )
            m['arg_family'] = family
            if custom_args_supplied:
                m['custom_args'] = custom_args
            elif family and not normalize_custom_arg_entries(m.get('custom_args', [])):
                family_args, _ = resolve_custom_args_for_family(family)
                if family_args:
                    m['custom_args'] = family_args
            if family and custom_args_supplied:
                presets = load_custom_model_arg_presets()
                presets[family] = normalize_custom_arg_entries(m.get('custom_args', []))
                save_custom_model_arg_presets(presets)
            save_custom_models_file(models)
            return jsonify(ok=True, model=normalize_custom_model(m))
    return jsonify(ok=False, error='Model not found'), 404


@app.route('/api/custom-model-arg-presets/match', methods=['POST'])
def api_custom_model_arg_preset_match():
    data = request.json or {}
    family = infer_model_arg_family(
        data.get('display_name', ''),
        data.get('model_name', ''),
        data.get('model_path', ''),
    )
    args, source = resolve_custom_args_for_family(family)
    return jsonify({
        'family': family,
        'family_label': format_model_arg_family_label(family) if family else '',
        'args': args,
        'source': source,
    })


@app.route('/api/custom-models/<model_id>', methods=['DELETE'])
def api_custom_models_delete(model_id):
    models = load_custom_models()
    models = [m for m in models if m['id'] != model_id]
    save_custom_models_file(models)
    return jsonify(ok=True)


@app.route('/api/active-chat-model')
def api_active_chat_model():
    """Determine which chat model is currently loaded."""
    active = active_chat_model_snapshot()
    payload = dict(active)
    if active.get('kind') == 'custom':
        for m in load_custom_models():
            if m.get('id') == active.get('variant'):
                payload['custom_model'] = m
                break
    return jsonify(payload)


@app.route('/api/tts/overview')
def api_tts_overview():
    env = read_env()
    state = load_tts_state()
    backends = load_tts_backends()
    gateway_data = {}
    gateway_error = None
    try:
        gateway_data = http_json(f'{tts_gateway_url()}/api/backends', timeout=10)
    except Exception as exc:
        gateway_error = str(exc)

    gateway_backends = {item['id']: item for item in gateway_data.get('backends', [])}
    items = []
    for backend in backends:
        service_name = backend.get('service_name')
        health = gateway_backends.get(backend['id'], {}).get('health', {})
        configured = bool(
            env.get(backend.get('upstream_url_env', ''), '').strip()
            or env.get(backend.get('launch_command_env', ''), '').strip()
        )
        items.append({
            **backend,
            'service_status': get_service_status(service_name) if service_name else 'unknown',
            'active': backend['id'] == state.get('active_backend'),
            'configured': configured,
            'voices': health.get('voices', []),
            'health': health,
        })

    return jsonify({
        'gateway_service_status': get_service_status('tts-gateway'),
        'gateway_error': gateway_error,
        'public_endpoint': env.get('TTS_PUBLIC_URL', 'http://127.0.0.1:8060'),
        'default_format': env.get('TTS_DEFAULT_FORMAT', 'mp3'),
        'single_active': env.get('TTS_SINGLE_ACTIVE', 'on'),
        'active_backend': state.get('active_backend'),
        'updated_at': state.get('updated_at'),
        'backends': items,
    })


@app.route('/api/tts/activate/<backend_id>', methods=['POST'])
def api_tts_activate(backend_id):
    backends = {item['id']: item for item in load_tts_backends()}
    backend = backends.get(backend_id)
    if not backend:
        return jsonify(ok=False, error='Unknown TTS backend'), 404

    outputs = []
    try:
        if should_use_local_tts_manager('tts-gateway'):
            ok, output = run_tts_manager('tts-gateway', 'start')
            outputs.append(output)
            if not ok:
                return jsonify(ok=False, error=output), 500
        else:
            subprocess.run(['systemctl', 'start', 'tts-gateway'], capture_output=True, timeout=30)
        if not wait_for_tts_gateway():
            return jsonify(ok=False, error='TTS gateway did not become ready in time'), 502
        if read_env().get('TTS_SINGLE_ACTIVE', 'on') != 'off':
            for service_name in TTS_BACKEND_SERVICES:
                if service_name != backend.get('service_name'):
                    if should_use_local_tts_manager(service_name):
                        run_tts_manager(service_name, 'stop')
                    else:
                        subprocess.run(['systemctl', 'stop', service_name], capture_output=True, timeout=30)
        if backend.get('service_name'):
            if should_use_local_tts_manager(backend['service_name']):
                ok, output = run_tts_manager(backend['service_name'], 'start')
                outputs.append(output)
                if not ok:
                    return jsonify(ok=False, error=output), 500
            else:
                start_result = subprocess.run(
                    ['systemctl', 'start', backend['service_name']],
                    capture_output=True, text=True, timeout=30,
                )
                outputs.append((start_result.stdout + start_result.stderr).strip())
        gateway_result = http_json(f'{tts_gateway_url()}/api/activate/{backend_id}', method='POST', timeout=15)
        return jsonify(ok=True, backend_id=backend_id, gateway=gateway_result, output='\n'.join(filter(None, outputs)))
    except urlerror.HTTPError as exc:
        return jsonify(ok=False, error=exc.read().decode('utf-8', errors='ignore') or str(exc)), exc.code
    except Exception as exc:
        return jsonify(ok=False, error=str(exc)), 500


@app.route('/api/tts/test', methods=['POST'])
def api_tts_test():
    payload = request.json or {}
    try:
        audio, content_type = http_bytes(f'{tts_gateway_url()}/v1/audio/speech', method='POST', payload=payload, timeout=300)
        return Response(audio, mimetype=content_type)
    except urlerror.HTTPError as exc:
        return jsonify(ok=False, error=exc.read().decode('utf-8', errors='ignore') or str(exc)), exc.code
    except Exception as exc:
        return jsonify(ok=False, error=str(exc)), 500



@app.route('/api/chat-templates', methods=['GET'])
def api_chat_templates_list():
    return jsonify(list_chat_templates())


@app.route('/api/chat-templates', methods=['POST'])
def api_chat_templates_create():
    data = request.get_json(silent=True) or {}
    name = (data.get('name') or '').strip()
    content = data.get('content')
    if not name:
        return jsonify(ok=False, error='Name is required'), 400
    if not isinstance(content, str) or not content.strip():
        return jsonify(ok=False, error='Template content is required'), 400
    template_id = chat_template_id_from_name(data.get('id') or name)
    existing_ids = {item['id'] for item in list_chat_templates() if item.get('id')}
    base_id = template_id
    suffix = 2
    while template_id in existing_ids:
        template_id = f'{base_id}-{suffix}'[:80]
        suffix += 1
    try:
        path = chat_template_path(template_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        meta = load_chat_template_meta()
        meta[template_id] = {
            'name': name,
            'description': (data.get('description') or '').strip(),
            'updated_at': int(time.time()),
        }
        save_chat_template_meta(meta)
    except Exception as exc:
        return jsonify(ok=False, error=f'Could not save chat template: {exc}'), 500
    return jsonify(ok=True, id=template_id)


@app.route('/api/chat-templates/<template_id>', methods=['GET'])
def api_chat_templates_get(template_id):
    try:
        path = chat_template_path(template_id)
    except ValueError as exc:
        return jsonify(ok=False, error=str(exc)), 400
    if not path.exists():
        return jsonify(ok=False, error='Template not found'), 404
    item = load_chat_template_meta().get(template_id, {})
    return jsonify({
        'ok': True,
        'id': template_id,
        'name': item.get('name') or template_id,
        'description': item.get('description', ''),
        'content': path.read_text(),
        'updated_at': item.get('updated_at', int(path.stat().st_mtime)),
    })


@app.route('/api/chat-templates/<template_id>', methods=['PUT'])
def api_chat_templates_update(template_id):
    data = request.get_json(silent=True) or {}
    try:
        path = chat_template_path(template_id)
    except ValueError as exc:
        return jsonify(ok=False, error=str(exc)), 400
    if not path.exists():
        return jsonify(ok=False, error='Template not found'), 404
    content = data.get('content')
    if not isinstance(content, str) or not content.strip():
        return jsonify(ok=False, error='Template content is required'), 400
    try:
        path.write_text(content)
        meta = load_chat_template_meta()
        current = meta.get(template_id, {}) if isinstance(meta.get(template_id), dict) else {}
        current.update({
            'name': (data.get('name') or current.get('name') or template_id).strip(),
            'description': (data.get('description') or '').strip(),
            'updated_at': int(time.time()),
        })
        meta[template_id] = current
        save_chat_template_meta(meta)
    except Exception as exc:
        return jsonify(ok=False, error=f'Could not update chat template: {exc}'), 500
    return jsonify(ok=True, id=template_id)


@app.route('/api/chat-templates/<template_id>', methods=['DELETE'])
def api_chat_templates_delete(template_id):
    try:
        path = chat_template_path(template_id)
    except ValueError as exc:
        return jsonify(ok=False, error=str(exc)), 400
    if path.exists():
        path.unlink()
    meta = load_chat_template_meta()
    if template_id in meta:
        del meta[template_id]
        save_chat_template_meta(meta)
    return jsonify(ok=True)


@app.route('/api/saved-configs', methods=['GET'])
def api_saved_configs_list():
    configs = []
    default_name = get_default_saved_config_name()
    for f in sorted(SAVED_CONFIGS_DIR.glob('*.json')):
        try:
            data = json.loads(f.read_text())
            active = data.get('_active_chat_model') if isinstance(data.get('_active_chat_model'), dict) else {}
            slots = data.get('_active_backend_slots') if isinstance(data.get('_active_backend_slots'), dict) else {}
            configs.append({
                'name': f.stem,
                'display_name': data.get('_name', f.stem),
                'timestamp': data.get('_timestamp', 0),
                'description': data.get('_description', ''),
                'is_default': f.stem == default_name,
                'active_chat_model': active,
                'active_backend_slots': slots,
            })
        except Exception:
            pass
    return jsonify(configs)


@app.route('/api/saved-configs', methods=['POST'])
def api_saved_configs_save():
    data = request.json
    name = (data or {}).get('name', '').strip()
    if not name:
        return jsonify(ok=False, error='Name is required'), 400
    safe_name = re.sub(r'[^\w\-]', '_', name)
    env = read_env()
    config = normalize_env_keys(env)
    form_config = (data or {}).get('config')
    if isinstance(form_config, dict):
        snapshot = config_form_snapshot(form_config, env)
        config.update(snapshot)
        config['_config_form'] = snapshot
    config['_timestamp'] = int(time.time())
    config['_description'] = (data or {}).get('description', '')
    config['_name'] = name
    active = (data or {}).get('active_chat_model')
    slots = (data or {}).get('active_backend_slots')
    config['_active_chat_model'] = active if isinstance(active, dict) else active_chat_model_snapshot(env)
    config['_active_backend_slots'] = slots if isinstance(slots, dict) else active_backend_slots_snapshot(env)
    
    active_services = []
    for svc in SERVICES:
        name = svc.get('name')
        if not name or name in ('chat-backend', 'chat-backend-dense', 'chat-backend-moe', 
                                'qwen-chat-backend-27b', 'qwen-chat-backend-35b', 'qwen-chat-backend', 'chat-proxy'):
            continue
        if get_service_status(name) == 'active':
            active_services.append(name)
    config['_active_services'] = active_services

    try:
        SAVED_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
        (SAVED_CONFIGS_DIR / f'{safe_name}.json').write_text(
            json.dumps(config, indent=2))
    except Exception as exc:
        return jsonify(ok=False, error=f'Could not save config: {exc}'), 500
    return jsonify(ok=True, name=safe_name)


@app.route('/api/saved-configs/<name>', methods=['GET'])
def api_saved_configs_load(name):
    safe_name = re.sub(r'[^\w\-]', '_', name)
    path = SAVED_CONFIGS_DIR / f'{safe_name}.json'
    if not path.exists():
        return jsonify(ok=False, error='Config not found'), 404
    return jsonify(json.loads(path.read_text()))


def apply_saved_config(name: str, launch: bool = False) -> dict:
    safe_name = saved_config_name(name)
    path = SAVED_CONFIGS_DIR / f'{safe_name}.json'
    if not path.exists():
        return {'ok': False, 'error': 'Config not found'}
    config = json.loads(path.read_text())
    updates = saved_config_apply_updates(config)
    updates = apply_code_chat_mirrors(updates)
    try:
        update_env_values(updates)
    except Exception as e:
        return {'ok': False, 'error': str(e)}
    restart_needed = set()
    for key in updates:
        restart_needed.update(RESTART_HINTS.get(key, []))

    active = config.get('_active_chat_model') if isinstance(config.get('_active_chat_model'), dict) else {}
    slots = config.get('_active_backend_slots') if isinstance(config.get('_active_backend_slots'), dict) else {}
    if not active.get("variant") and isinstance(slots.get("primary"), dict):
        active = slots.get("primary") or active
    launched = []
    launch_output = ''
    if launch:
        ok, launch_output, launched = launch_chat_backend_for_saved_config(active)
        if not ok:
            return {
                'ok': False,
                'error': launch_output or 'Failed to launch saved chat backend',
                'restart_needed': sorted(restart_needed),
                'active_chat_model': active,
                'active_backend_slots': slots,
            }
        restart_needed.difference_update(SHARED_CHAT_BACKEND_RESTART)
        restart_needed.discard('chat-proxy')

        active_services = config.get('_active_services')
        if active_services is not None:
            for svc in SERVICES:
                name = svc.get('name')
                if not name or name in ('chat-backend', 'chat-backend-dense', 'chat-backend-moe', 
                                        'qwen-chat-backend-27b', 'qwen-chat-backend-35b', 'qwen-chat-backend', 'chat-proxy'):
                    continue
                is_active = get_service_status(name) == 'active'
                should_be_active = name in active_services
                if should_be_active and not is_active:
                    ServiceManager.start(name)
                    launched.append(name)
                elif not should_be_active and is_active:
                    ServiceManager.stop(name)
        else:
            secondary = slots.get("secondary") if isinstance(slots.get("secondary"), dict) else {}
            if secondary.get("service") == "chat-backend2" and secondary.get("variant"):
                if get_service_status("chat-backend2") != "active":
                    ServiceManager.start("chat-backend2")
                    launched.append("chat-backend2")
                if get_service_status("chat-proxy2") != "active":
                    ServiceManager.start("chat-proxy2")
                    launched.append("chat-proxy2")

    return {
        'ok': True,
        'restart_needed': sorted(restart_needed),
        'active_chat_model': active,
        'active_backend_slots': slots,
        'launched_services': launched,
        'output': launch_output,
    }


def update_saved_config_values(name: str, updates: dict) -> dict:
    safe_name = saved_config_name(name)
    path = SAVED_CONFIGS_DIR / f'{safe_name}.json'
    if not path.exists():
        return {'ok': False, 'error': 'Config not found'}
    if not isinstance(updates, dict):
        return {'ok': False, 'error': 'Expected updates object'}
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            return {'ok': False, 'error': 'Saved config is invalid'}
        filtered = filter_config_updates(updates)
        if not filtered:
            return {'ok': True, 'name': safe_name, 'updated_keys': []}
        data.update(filtered)
        form_snapshot = data.get('_config_form') if isinstance(data.get('_config_form'), dict) else {}
        form_snapshot.update(filtered)
        data['_config_form'] = form_snapshot
        data['_timestamp'] = int(time.time())
        path.write_text(json.dumps(data, indent=2))
        return {'ok': True, 'name': safe_name, 'updated_keys': sorted(filtered.keys())}
    except Exception as exc:
        return {'ok': False, 'error': f'Could not update saved config: {exc}'}


@app.route('/api/saved-configs/<name>/apply', methods=['POST'])
def api_saved_configs_apply(name):
    data = request.get_json(silent=True) or {}
    result = apply_saved_config(name, launch=bool(data.get('launch')))
    status = 200 if result.get('ok') else (404 if result.get('error') == 'Config not found' else 500)
    return jsonify(result), status


@app.route('/api/saved-configs/<name>/patch', methods=['POST'])
def api_saved_configs_patch(name):
    data = request.get_json(silent=True) or {}
    result = update_saved_config_values(name, data.get('updates', data))
    status = 200 if result.get('ok') else (404 if result.get('error') == 'Config not found' else 400)
    return jsonify(result), status


@app.route('/api/saved-configs/<name>/default', methods=['POST'])
def api_saved_configs_set_default(name):
    safe_name = saved_config_name(name)
    path = SAVED_CONFIGS_DIR / f'{safe_name}.json'
    if not path.exists():
        return jsonify(ok=False, error='Config not found'), 404
    set_default_saved_config_name(safe_name)
    return jsonify(ok=True, name=safe_name)


@app.route('/api/saved-configs/<name>/default', methods=['DELETE'])
def api_saved_configs_clear_default(name):
    clear_default_saved_config_name(name)
    return jsonify(ok=True)


@app.route('/api/saved-configs/<name>', methods=['DELETE'])
def api_saved_configs_delete(name):
    safe_name = saved_config_name(name)
    path = SAVED_CONFIGS_DIR / f'{safe_name}.json'
    if path.exists():
        path.unlink()
    clear_default_saved_config_name(safe_name)
    return jsonify(ok=True)


@app.route('/api/config', methods=['GET'])
def api_config_get():
    return jsonify(normalize_env_keys(read_env()))


@app.route('/api/config', methods=['POST'])
def api_config_save():
    updates = request.json
    if not isinstance(updates, dict):
        return jsonify(ok=False, error='Expected JSON object'), 400
    filtered = filter_config_updates(updates)
    filtered = apply_code_chat_mirrors(filtered)
    try:
        update_env_values(filtered)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500
    restart_needed = set()
    for key in filtered:
        restart_needed.update(RESTART_HINTS.get(key, []))
    return jsonify(ok=True, restart_needed=sorted(restart_needed))


@app.route('/api/graphiti/status')
def api_graphiti_status():
    env = read_env()
    cfg = graphiti_config(env)
    status = {
        'ok': True,
        'last_refresh': int(time.time()),
        'config': {
            'graphiti_url': cfg['graphiti_url'],
            'llm_base_url': cfg['llm_base_url'],
            'llm_model': cfg['llm_model'],
            'embed_base_url': cfg['embed_base_url'],
            'embed_model': cfg['embed_model'],
            'reranker_provider': cfg['reranker_provider'],
            'reranker_base_url': cfg['reranker_base_url'],
            'reranker_model': cfg['reranker_model'],
            'neo4j_uri': cfg['neo4j_uri'],
            'neo4j_database': cfg['neo4j_database'],
            'neo4j_http_url': cfg['neo4j_http_url'],
        },
        'checks': {
            'graphiti_api': {'ok': False, 'error': ''},
            'neo4j': {'ok': False, 'error': ''},
            'llm_endpoint': {'ok': False, 'error': ''},
            'embedding_endpoint': {'ok': False, 'error': ''},
            'reranker_endpoint': {'ok': False, 'error': ''},
            'ingestion_worker': {'ok': None, 'error': 'not exposed by current Graphiti API'},
        },
    }

    try:
        graphiti_health = http_json(f"{cfg['graphiti_url']}/healthcheck", timeout=6)
        status['checks']['graphiti_api']['ok'] = graphiti_health.get('status') == 'healthy'
    except Exception as exc:
        status['checks']['graphiti_api']['error'] = str(exc)

    try:
        neo4j_http_query('RETURN 1 AS ok', timeout=8)
        status['checks']['neo4j']['ok'] = True
    except Exception as exc:
        status['checks']['neo4j']['error'] = str(exc)

    endpoint_checks = [
        ('llm_endpoint', cfg['llm_base_url']),
        ('embedding_endpoint', cfg['embed_base_url']),
        ('reranker_endpoint', cfg['reranker_base_url']),
    ]
    for key, base_url in endpoint_checks:
        if not base_url:
            status['checks'][key]['error'] = 'not configured'
            continue
        try:
            http_json(f"{base_url}/models", timeout=6)
            status['checks'][key]['ok'] = True
        except Exception as exc:
            status['checks'][key]['error'] = str(exc)

    status['ok'] = all(v.get('ok') is True for k, v in status['checks'].items() if k != 'ingestion_worker')
    return jsonify(status)


@app.route('/api/graphiti/stats')
def api_graphiti_stats():
    try:
        total_counts_query = """
            CALL {
              MATCH (e:Episodic) RETURN count(e) AS episodes
            }
            CALL {
              MATCH (n:Entity) RETURN count(n) AS entities
            }
            CALL {
              MATCH (:Entity)-[r:RELATES_TO]->(:Entity) RETURN count(r) AS relationships
            }
            RETURN episodes, entities, relationships
        """
        counts_rows = neo4j_rows_as_dicts(neo4j_http_query(total_counts_query))
        counts = counts_rows[0] if counts_rows else {'episodes': 0, 'entities': 0, 'relationships': 0}

        by_day_episodes_query = """
            MATCH (e:Episodic)
            WHERE e.created_at IS NOT NULL
            WITH toString(date(datetime(e.created_at))) AS day, count(e) AS c
            RETURN day, c
            ORDER BY day DESC
            LIMIT 30
        """
        by_day_entities_query = """
            MATCH (n:Entity)
            WHERE n.created_at IS NOT NULL
            WITH toString(date(datetime(n.created_at))) AS day, count(n) AS c
            RETURN day, c
            ORDER BY day DESC
            LIMIT 30
        """
        top_groups_query = """
            MATCH (e:Episodic)
            WHERE coalesce(e.group_id, '') <> ''
            RETURN e.group_id AS group_id, count(e) AS c
            ORDER BY c DESC
            LIMIT 15
        """
        top_entities_query = """
            MATCH (n:Entity)
            WITH n, COUNT { (n)--() } AS degree
            RETURN n.uuid AS uuid, n.name AS name, n.group_id AS group_id, degree
            ORDER BY degree DESC
            LIMIT 15
        """

        return jsonify(
            {
                'ok': True,
                'totals': {
                    'episodes': int(counts.get('episodes', 0) or 0),
                    'entities': int(counts.get('entities', 0) or 0),
                    'relationships': int(counts.get('relationships', 0) or 0),
                },
                'episodes_by_day': neo4j_rows_as_dicts(neo4j_http_query(by_day_episodes_query)),
                'entities_by_day': neo4j_rows_as_dicts(neo4j_http_query(by_day_entities_query)),
                'top_groups': neo4j_rows_as_dicts(neo4j_http_query(top_groups_query)),
                'top_entities': neo4j_rows_as_dicts(neo4j_http_query(top_entities_query)),
                'last_refresh': int(time.time()),
            }
        )
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 500


@app.route('/api/graphiti/recent/episodes')
def api_graphiti_recent_episodes():
    page = parse_int(request.args.get('page'), 1, minimum=1)
    page_size = parse_int(request.args.get('page_size'), 25, minimum=1, maximum=100)
    group_id = (request.args.get('group_id') or '').strip() or None
    start_time = parse_iso_datetime(request.args.get('start_time'))
    end_time = parse_iso_datetime(request.args.get('end_time'))
    try:
        data = graphiti_recent_episodes(
            page=page,
            page_size=page_size,
            group_id=group_id,
            start_time=start_time,
            end_time=end_time,
        )
        return jsonify({'ok': True, **data})
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 500


@app.route('/api/graphiti/recent/entities')
def api_graphiti_recent_entities():
    page = parse_int(request.args.get('page'), 1, minimum=1)
    page_size = parse_int(request.args.get('page_size'), 25, minimum=1, maximum=100)
    group_id = (request.args.get('group_id') or '').strip() or None
    name_query = (request.args.get('q') or '').strip() or None
    start_time = parse_iso_datetime(request.args.get('start_time'))
    end_time = parse_iso_datetime(request.args.get('end_time'))
    try:
        data = graphiti_recent_entities(
            page=page,
            page_size=page_size,
            group_id=group_id,
            name_query=name_query,
            start_time=start_time,
            end_time=end_time,
        )
        return jsonify({'ok': True, **data})
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 500


@app.route('/api/graphiti/recent/relationships')
def api_graphiti_recent_relationships():
    page = parse_int(request.args.get('page'), 1, minimum=1)
    page_size = parse_int(request.args.get('page_size'), 25, minimum=1, maximum=100)
    group_id = (request.args.get('group_id') or '').strip() or None
    relation_query = (request.args.get('q') or '').strip() or None
    start_time = parse_iso_datetime(request.args.get('start_time'))
    end_time = parse_iso_datetime(request.args.get('end_time'))
    try:
        data = graphiti_recent_relationships(
            page=page,
            page_size=page_size,
            group_id=group_id,
            relation_query=relation_query,
            start_time=start_time,
            end_time=end_time,
        )
        return jsonify({'ok': True, **data})
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 500


@app.route('/api/graphiti/detail/episode/<episode_uuid>')
def api_graphiti_episode_detail(episode_uuid):
    query = """
        MATCH (e:Episodic {uuid: $uuid})
        OPTIONAL MATCH (e)-[m:MENTIONS]->(n:Entity)
        RETURN e.uuid AS uuid,
               e.name AS name,
               e.group_id AS group_id,
               toString(e.created_at) AS created_at,
               toString(e.valid_at) AS valid_at,
               e.source AS source,
               e.source_description AS source_description,
               e.content AS content,
               collect(DISTINCT {
                 uuid: n.uuid,
                 name: n.name,
                 group_id: n.group_id,
                 mention_uuid: m.uuid
               }) AS entities
    """
    try:
        rows = neo4j_rows_as_dicts(neo4j_http_query(query, {'uuid': episode_uuid}))
        if not rows:
            return jsonify({'ok': False, 'error': 'Episode not found'}), 404
        return jsonify({'ok': True, 'item': rows[0]})
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 500


@app.route('/api/graphiti/detail/entity/<entity_uuid>')
def api_graphiti_entity_detail(entity_uuid):
    query = """
        MATCH (n:Entity {uuid: $uuid})
        CALL {
          WITH n
          OPTIONAL MATCH (ep:Episodic)-[m:MENTIONS]->(n)
          RETURN collect(DISTINCT {
            uuid: ep.uuid,
            name: ep.name,
            group_id: ep.group_id,
            created_at: toString(ep.created_at),
            mention_uuid: m.uuid
          }) AS episodes
        }
        CALL {
          WITH n
          OPTIONAL MATCH (n)-[r:RELATES_TO]->(t:Entity)
          RETURN collect(DISTINCT {
            uuid: r.uuid,
            relation_name: r.name,
            fact: r.fact,
            group_id: r.group_id,
            created_at: toString(r.created_at),
            target_uuid: t.uuid,
            target_name: t.name,
            direction: 'out'
          }) AS outgoing
        }
        CALL {
          WITH n
          OPTIONAL MATCH (s:Entity)-[r:RELATES_TO]->(n)
          RETURN collect(DISTINCT {
            uuid: r.uuid,
            relation_name: r.name,
            fact: r.fact,
            group_id: r.group_id,
            created_at: toString(r.created_at),
            source_uuid: s.uuid,
            source_name: s.name,
            direction: 'in'
          }) AS incoming
        }
        RETURN n.uuid AS uuid,
               n.name AS name,
               n.group_id AS group_id,
               toString(n.created_at) AS created_at,
               n.summary AS summary,
               n.labels AS prop_labels,
               [x IN labels(n) WHERE x <> 'Entity'] AS node_labels,
               episodes,
               outgoing,
               incoming
    """
    try:
        rows = neo4j_rows_as_dicts(neo4j_http_query(query, {'uuid': entity_uuid}))
        if not rows:
            return jsonify({'ok': False, 'error': 'Entity not found'}), 404
        item = rows[0]
        item['labels'] = normalize_entity_labels(item.get('prop_labels')) or normalize_entity_labels(item.get('node_labels'))
        return jsonify({'ok': True, 'item': item})
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 500


@app.route('/api/graphiti/detail/relationship/<relationship_uuid>')
def api_graphiti_relationship_detail(relationship_uuid):
    query = """
        MATCH (s:Entity)-[r:RELATES_TO {uuid: $uuid}]->(t:Entity)
        OPTIONAL MATCH (ep:Episodic)
        WHERE r.episodes IS NOT NULL AND ep.uuid IN r.episodes
        RETURN r.uuid AS uuid,
               r.group_id AS group_id,
               r.name AS relation_name,
               r.fact AS fact,
               toString(r.created_at) AS created_at,
               toString(r.valid_at) AS valid_at,
               toString(r.invalid_at) AS invalid_at,
               toString(r.expired_at) AS expired_at,
               s.uuid AS source_uuid,
               s.name AS source_name,
               t.uuid AS target_uuid,
               t.name AS target_name,
               collect(DISTINCT {
                 uuid: ep.uuid,
                 name: ep.name,
                 group_id: ep.group_id,
                 created_at: toString(ep.created_at)
               }) AS linked_episodes
    """
    try:
        rows = neo4j_rows_as_dicts(neo4j_http_query(query, {'uuid': relationship_uuid}))
        if not rows:
            return jsonify({'ok': False, 'error': 'Relationship not found'}), 404
        return jsonify({'ok': True, 'item': rows[0]})
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 500


@app.route('/api/graphiti/search/memory', methods=['POST'])
def api_graphiti_search_memory():
    data = request.json or {}
    query = (data.get('query') or '').strip()
    if not query:
        return jsonify({'ok': False, 'error': 'query is required'}), 400
    group_id = (data.get('group_id') or '').strip()
    max_facts = parse_int(data.get('max_facts'), 10, minimum=1, maximum=50)
    cfg = graphiti_config()
    payload = {'query': query, 'max_facts': max_facts}
    if group_id:
        payload['group_ids'] = [group_id]
    try:
        result = http_json(f"{cfg['graphiti_url']}/search", method='POST', payload=payload, timeout=20)
        return jsonify({'ok': True, 'result': result})
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 500


@app.route('/api/graphiti/search/group/<group_id>')
def api_graphiti_group_history(group_id):
    last_n = parse_int(request.args.get('last_n'), 50, minimum=1, maximum=500)
    cfg = graphiti_config()
    try:
        result = http_json(f"{cfg['graphiti_url']}/episodes/{group_id}?last_n={last_n}", timeout=25)
        return jsonify({'ok': True, 'group_id': group_id, 'episodes': result})
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 500


@app.route('/api/graphiti/search/entities')
def api_graphiti_search_entities():
    q = (request.args.get('q') or '').strip()
    page = parse_int(request.args.get('page'), 1, minimum=1)
    page_size = parse_int(request.args.get('page_size'), 25, minimum=1, maximum=100)
    try:
        data = graphiti_recent_entities(page=page, page_size=page_size, name_query=q or None)
        return jsonify({'ok': True, **data})
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 500


@app.route('/api/graphiti/neighborhood/<entity_uuid>')
def api_graphiti_neighborhood(entity_uuid):
    limit = parse_int(request.args.get('limit'), 50, minimum=1, maximum=200)
    try:
        item = graphiti_entity_neighborhood(entity_uuid, limit=limit)
        return jsonify({'ok': True, 'item': item})
    except RuntimeError as exc:
        if str(exc) == 'Entity not found':
            return jsonify({'ok': False, 'error': 'Entity not found'}), 404
        return jsonify({'ok': False, 'error': str(exc)}), 500
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 500


@app.route('/api/graphiti/exports')
def api_graphiti_exports_list():
    files = []
    for path in sorted(GRAPHITI_EXPORTS_DIR.glob('*'), reverse=True):
        if not path.is_file():
            continue
        files.append(
            {
                'filename': path.name,
                'size_bytes': path.stat().st_size,
                'modified_at': int(path.stat().st_mtime),
                'download_url': f"/api/graphiti/exports/{path.name}",
            }
        )
    return jsonify({'ok': True, 'items': files[:200], 'directory': str(GRAPHITI_EXPORTS_DIR)})


@app.route('/api/graphiti/exports/<path:filename>')
def api_graphiti_export_download(filename):
    safe_name = safe_export_filename(filename)
    path = GRAPHITI_EXPORTS_DIR / safe_name
    if not path.exists() or not path.is_file():
        return jsonify({'ok': False, 'error': 'Export file not found'}), 404
    return send_file(path, as_attachment=True, download_name=path.name)


@app.route('/api/graphiti/export', methods=['POST'])
def api_graphiti_export():
    data = request.json or {}
    export_type = (data.get('export_type') or 'recent').strip()
    fmt = (data.get('format') or 'json').strip().lower()
    if fmt not in ('json', 'md'):
        return jsonify({'ok': False, 'error': 'format must be json or md'}), 400

    env = read_env()
    cfg = graphiti_config(env)
    timestamp = int(time.time())
    dt = datetime.utcfromtimestamp(timestamp).strftime('%Y%m%d-%H%M%S')
    payload: dict = {
        'metadata': {
            'export_type': export_type,
            'exported_at': datetime.utcfromtimestamp(timestamp).isoformat() + 'Z',
            'graphiti_url': cfg['graphiti_url'],
            'neo4j_database': cfg['neo4j_database'],
        },
        'episodes': [],
        'entities': [],
        'relationships': [],
    }

    try:
        if export_type == 'group':
            group_id = (data.get('group_id') or '').strip()
            if not group_id:
                return jsonify({'ok': False, 'error': 'group_id is required for export_type=group'}), 400
            limit = parse_int(data.get('limit'), 200, minimum=1, maximum=5000)
            payload['metadata']['group_id'] = group_id
            payload['episodes'] = graphiti_recent_episodes(page=1, page_size=limit, group_id=group_id)['items']
            payload['entities'] = graphiti_recent_entities(page=1, page_size=limit, group_id=group_id)['items']
            payload['relationships'] = graphiti_recent_relationships(page=1, page_size=limit, group_id=group_id)['items']
            base_name = f"graphiti-group-{safe_export_filename(group_id)}-{dt}"
        elif export_type == 'entity':
            entity_uuid = (data.get('entity_uuid') or '').strip()
            if not entity_uuid:
                return jsonify({'ok': False, 'error': 'entity_uuid is required for export_type=entity'}), 400
            entity_rows = neo4j_rows_as_dicts(
                neo4j_http_query(
                    """
                    MATCH (n:Entity {uuid: $uuid})
                    RETURN n.uuid AS uuid, n.name AS name, n.group_id AS group_id,
                           toString(n.created_at) AS created_at, n.summary AS summary,
                           n.labels AS prop_labels, [x IN labels(n) WHERE x <> 'Entity'] AS node_labels,
                           COUNT { (n)--() } AS degree
                    """,
                    {'uuid': entity_uuid},
                )
            )
            if not entity_rows:
                return jsonify({'ok': False, 'error': 'Entity not found'}), 404
            ent = entity_rows[0]
            ent['labels'] = normalize_entity_labels(ent.get('prop_labels')) or normalize_entity_labels(ent.get('node_labels'))
            payload['entities'] = [ent]
            neighborhood = graphiti_entity_neighborhood(entity_uuid, limit=200)
            rels = (neighborhood.get('outgoing') or []) + (neighborhood.get('incoming') or [])
            payload['relationships'] = rels
            payload['metadata']['entity_uuid'] = entity_uuid
            base_name = f"graphiti-entity-{safe_export_filename(entity_uuid)}-{dt}"
        elif export_type == 'recent':
            limit = parse_int(data.get('limit'), 200, minimum=1, maximum=5000)
            payload['episodes'] = graphiti_recent_episodes(page=1, page_size=limit)['items']
            payload['entities'] = graphiti_recent_entities(page=1, page_size=limit)['items']
            payload['relationships'] = graphiti_recent_relationships(page=1, page_size=limit)['items']
            payload['metadata']['limit'] = limit
            base_name = f"graphiti-recent-{limit}-{dt}"
        elif export_type == 'date_range':
            start_time = parse_iso_datetime(data.get('start_time'))
            end_time = parse_iso_datetime(data.get('end_time'))
            if not start_time or not end_time:
                return jsonify({'ok': False, 'error': 'start_time and end_time are required for date_range export'}), 400
            limit = parse_int(data.get('limit'), 1000, minimum=1, maximum=10000)
            payload['episodes'] = graphiti_recent_episodes(page=1, page_size=limit, start_time=start_time, end_time=end_time)['items']
            payload['entities'] = graphiti_recent_entities(page=1, page_size=limit, start_time=start_time, end_time=end_time)['items']
            payload['relationships'] = graphiti_recent_relationships(page=1, page_size=limit, start_time=start_time, end_time=end_time)['items']
            payload['metadata']['start_time'] = start_time
            payload['metadata']['end_time'] = end_time
            payload['metadata']['limit'] = limit
            base_name = f"graphiti-date-range-{dt}"
        else:
            return jsonify({'ok': False, 'error': f'unsupported export_type: {export_type}'}), 400

        payload['metadata']['item_count'] = (
            len(payload.get('episodes', []))
            + len(payload.get('entities', []))
            + len(payload.get('relationships', []))
        )

        if fmt == 'json':
            file_name = f"{base_name}.json"
            file_path = GRAPHITI_EXPORTS_DIR / file_name
            file_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        else:
            file_name = f"{base_name}.md"
            file_path = GRAPHITI_EXPORTS_DIR / file_name
            file_path.write_text(graphiti_markdown_export(payload), encoding='utf-8')

        return jsonify(
            {
                'ok': True,
                'file': {
                    'filename': file_name,
                    'path': str(file_path),
                    'size_bytes': file_path.stat().st_size,
                    'download_url': f"/api/graphiti/exports/{file_name}",
                },
                'directory': str(GRAPHITI_EXPORTS_DIR),
                'summary': payload['metadata'],
            }
        )
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 500


@app.route('/api/logs/<name>')
def api_logs(name):
    if name not in {s['name'] for s in SERVICES}:
        return jsonify(error='Unknown service'), 400

    def generate():
        if should_use_local_transcript_manager(name):
            log_file = transcript_log_file()
            log_file.parent.mkdir(parents=True, exist_ok=True)
            log_file.touch(exist_ok=True)
            proc = subprocess.Popen(
                ['tail', '-n', '100', '-F', str(log_file)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            try:
                for line in iter(proc.stdout.readline, ''):
                    yield f'data: {json.dumps(line.rstrip())}\n\n'
            finally:
                proc.terminate()
                proc.wait()
            return

        if should_use_local_tts_manager(name):
            log_file = tts_log_file(name)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            log_file.touch(exist_ok=True)
            proc = subprocess.Popen(
                ['tail', '-n', '100', '-F', str(log_file)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            try:
                for line in iter(proc.stdout.readline, ''):
                    yield f'data: {json.dumps(line.rstrip())}\n\n'
            finally:
                proc.terminate()
                proc.wait()
            return

        journal_unit = "uwsgi" if is_searxng_service(name) else name
        proc = subprocess.Popen(
            ['journalctl', '-u', journal_unit, '-f', '-n', '100',
             '--no-pager', '--output=short-iso'],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        try:
            for line in iter(proc.stdout.readline, ''):
                yield f'data: {json.dumps(line.rstrip())}\n\n'
        finally:
            proc.terminate()
            proc.wait()

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


def apply_default_saved_config_on_startup():
    default_name = get_default_saved_config_name()
    if default_name:
        print(
            f'[llm-manager] Default saved config is {default_name}; not applying it on manager startup.',
            flush=True,
        )


if __name__ == '__main__':
    apply_default_saved_config_on_startup()
    port = int(os.environ.get('LLM_MANAGER_PORT', 8080))
    host = os.environ.get('LLM_MANAGER_HOST', '0.0.0.0')
    print(f'[llm-manager] Serving on http://{host}:{port}', flush=True)
    app.run(host=host, port=port, debug=False, threaded=True)
