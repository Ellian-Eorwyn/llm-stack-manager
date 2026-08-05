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
import shutil
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import setup_engine

# Explicit rather than relying on the script directory, so importing app.py by
# path (as the tests do) resolves sibling modules the same way systemd does.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import budget
import config_env
import config_fields
import core
import deploy
import health
import models
import public_api
import scheduling
import telemetry

from routes import graphiti as graphiti_routes
from routes import models as model_routes
from routes import public as public_routes
from routes import setup as setup_routes

# Bound rather than accessed through the module, because these are tables that
# are read and never reassigned. Anything a test needs to substitute — the env
# reader, a path, a service call — is reached as `module.name` instead, so the
# substitution is visible to every importer.
from config_fields import (
    BEE_KV_CACHE_OPTIONS,
    BUILTIN_CHAT_VARIANTS,
    BUILTIN_CHAT_VARIANT_BY_ID,
    BUILTIN_CHAT_VARIANT_BY_SERVICE,
    BUILTIN_CHAT_VARIANT_IDS,
    CODE_TO_CHAT_MIRRORS,
    CONFIG_FIELDS,
    CORE_CONFIG_SECTIONS,
    DEFAULT_DEPRECATION_NOTE,
    DEPRECATED_ENV_KEY_NOTES,
    LEGACY_ENV_KEY_MAP,
    LLAMA_CACHE_IDLE_OPTIONS,
    LLAMA_KV_CACHE_OPTIONS,
    LLAMA_METRICS_OPTIONS,
    LLAMA_SPEC_METHOD_OPTIONS,
    NEW_ENV_KEY_LEGACY_ALIASES,
    RESTART_HINTS,
    SHARED_CHAT_BACKEND_RESTART,
)


app = Flask(__name__)
app.register_blueprint(graphiti_routes.bp)
app.register_blueprint(model_routes.bp)
app.register_blueprint(setup_routes.bp)
# Also served here so `/api/v1/*` works on the manager's own port for local use
# and testing. The listener that exists for other machines is built in
# `create_state_api_app()`, and is the one that enforces the token.
app.register_blueprint(public_routes.bp)

# The tree this process is actually running from, which is not the tree anyone
# edits: systemd starts the manager from the installed checkout, and that only
# advances when update.sh is run.
DEPLOY_WATCHER = deploy.DriftWatcher(core.STACK_DIR)


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
    {"group": "auxiliary", "name": "llama-router",     "label": "Model Router", "desc": "Loads the auxiliary models on demand",  "ports": "8013", "config_section": "Model Router"},
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


def router_pooled_units(env: dict) -> set:
    """Units the model router owns, which nothing else may start or stop."""
    return telemetry.pooled_units(env)


def apply_router_restart_hints(restart_needed: set, env: dict) -> set:
    """Point restart hints at the router when it owns the model.

    `RESTART_HINTS` names the unit that serves each setting, which is right
    whenever the models are units. In router mode the setting is rendered into
    the preset file instead, so the thing to restart is the router — and the
    named unit is not even running.
    """
    pooled = router_pooled_units(env)
    if not pooled or not restart_needed & pooled:
        return restart_needed
    return (restart_needed - pooled) | {telemetry.ROUTER_UNIT}




# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------




# Which env prefix each llama.cpp service is launched from. Only the chat
# backends divide their context across slots, but the rest are listed so the
# services panel can show a context for every model service it renders.
SERVICE_ENV_PREFIXES = {
    "chat-backend-dense": "CHAT_PRIMARY",
    "chat-backend": "CHAT_PRIMARY",
    "chat-backend2": "CHAT2",
    "chat-backend-moe": "CHAT2",
    "embed": "EMBED",
    "embed2": "EMBED2",
    "rerank": "RERANK",
    "task": "TASK",
    "ocr": "OCR",
}


def backend_context_summary(env: dict | None = None) -> dict:
    """Configured total and per-slot context for each llama.cpp service.

    `--ctx-size` is a total that llama.cpp divides by `--parallel`; a request is
    measured against the quotient, not the total. Showing only the total is how
    a backend the UI called "262144" came to reject a 155751-token request. This
    is read from the env rather than from the running backend so it is available
    for services that are stopped, which is when the number is being chosen.
    """
    env = config_env.normalize_env_keys(env or config_env.read_env())
    summary = {}
    for service, prefix in SERVICE_ENV_PREFIXES.items():
        try:
            total = int(str(env.get(f"{prefix}_CTX_SIZE") or "").strip() or 0)
            slots = int(str(env.get(f"{prefix}_N_PARALLEL") or "").strip() or 1)
        except ValueError:
            continue
        if total <= 0:
            continue
        slots = max(1, slots)
        summary[service] = {
            "total_context": total,
            "slots": slots,
            "per_slot_context": total // slots,
        }
    return summary


def preflight_config(updates: dict, env: dict | None = None) -> dict:
    """Price a proposed configuration before it is written.

    The config form has always accepted anything, and the cost of that arrived
    later and somewhere else: a backend that fails to allocate on restart, or a
    prompt cache that evicts on most requests because its checkpoints never fit
    the RAM they were given. Both are computable from the model's own geometry,
    so they are computable here, before the write.

    Only backends the update actually touches are priced — reading GGUF
    metadata means touching the model file, and an unrelated port change should
    not pay for it.
    """
    env = config_env.normalize_env_keys(env or config_env.read_env())
    proposed = dict(env)
    proposed.update(updates)

    gpus = get_gpu_info()
    host = telemetry.host_memory(core.read_meminfo())
    backends = []
    for backend, prefix in budget.BACKEND_PREFIXES.items():
        if not any(key.startswith(f"{prefix}_") for key in updates):
            continue
        result = budget.budget_for(proposed, backend, gpus, host)
        verdict = result.get("verdict") or {}
        prediction = result.get("prediction") or {}
        backends.append({
            "backend": backend,
            "prefix": prefix,
            # A model that cannot be read is not a configuration that fails —
            # the operator may be pointing at a file they have not fetched yet.
            "error": result.get("error"),
            "issues": verdict.get("issues", []),
            "per_slot_context": prediction.get("per_slot_context"),
            "total_context": prediction.get("total_context"),
            "slots": prediction.get("slots"),
            "vram_upper_mib": (prediction.get("vram") or {}).get("upper_mib"),
            "cache_ram_shortfall_mib": (prediction.get("host") or {}).get("cache_ram_shortfall_mib"),
        })

    issues = [issue for entry in backends for issue in entry["issues"]]
    return {
        "ok": not any(issue["level"] == "error" for issue in issues),
        "backends": backends,
        "errors": [issue for issue in issues if issue["level"] == "error"],
        "warnings": [issue for issue in issues if issue["level"] == "warn"],
    }


def saved_config_apply_updates(config: dict) -> dict:
    updates = {k: v for k, v in config.items()
               if not k.startswith('_') and (v is None or isinstance(v, (str, int, float, bool)))}
    updates = config_env.filter_config_updates(updates)
    form_snapshot = config.get("_config_form")
    if isinstance(form_snapshot, dict):
        updates.update(config_env.config_form_snapshot(form_snapshot))
    return updates


def builtin_chat_variants(env: dict | None = None) -> list[dict]:
    env = config_env.normalize_env_keys(env or config_env.read_env())
    items = []
    for item in BUILTIN_CHAT_VARIANTS:
        items.append({
            **item,
            "label": env.get(item["label_key"], item["default_label"]).strip() or item["default_label"],
            "desc": item["default_desc"],
        })
    return items


def patch_service_labels(env: dict | None = None) -> list[dict]:
    env = env or config_env.read_env()
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


def status_from_unit_state(state: dict) -> str:
    """The one place a systemd unit state becomes a status string."""
    if state['active']:
        return 'active'
    # A crashed unit used to be indistinguishable from a stopped one, since
    # `is-active` reports neither as active. Callers that compare against
    # 'active' are unaffected; the services panel now has something to say.
    if state['failed']:
        return 'failed'
    # A unit that cannot start is caught here far more often than in `failed`,
    # because `Restart=` puts it back into `activating` within seconds. Its own
    # state, so a service mid-launch is not read as one that is down.
    if state.get('starting'):
        return 'starting'
    return 'inactive' if state['installed'] else 'unknown'


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
        return status_from_unit_state(core.ServiceManager.state(name))
    except Exception:
        return 'unknown'


def record_service_expectation(name: str, action: str, ok: bool, source: str = 'operator') -> None:
    """Remember that a service was deliberately started or stopped.

    Only a successful action counts. A stop that failed leaves the service
    running, so recording it as expected-off would mislabel a card that is
    working; a start that failed generally lands the unit in `failed`, which
    already says more than an expectation would.
    """
    if not ok:
        return
    # systemd zeroes NRestarts on a clean stop, so this is belt and braces —
    # but a service the operator just cycled should start from no history
    # either way, rather than carrying a flap verdict across the action that
    # was meant to resolve it.
    health.RESTARTS.reset(name)
    health.record_expectation(name, 'off' if action == 'stop' else 'on', source=source)


def active_chat_model_snapshot(env: dict | None = None) -> dict:
    """Return the active primary chat backend in a form saved configs can replay."""
    env = env or config_env.read_env()
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
        for model in models.load_custom_models():
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
    env = config_env.normalize_env_keys(env or config_env.read_env())
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
    env = env or config_env.read_env()
    return {
        "primary": active_chat_model_snapshot(env),
        "secondary": active_secondary_backend_snapshot(env),
    }


def saved_config_name(name: str) -> str:
    return re.sub(r'[^\w\-]', '_', name)


def get_default_saved_config_name() -> str:
    try:
        name = core.DEFAULT_SAVED_CONFIG_FILE.read_text().strip()
    except FileNotFoundError:
        return ""
    return saved_config_name(name) if name else ""


def set_default_saved_config_name(name: str):
    safe_name = saved_config_name(name)
    core.DEFAULT_SAVED_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    core.DEFAULT_SAVED_CONFIG_FILE.write_text(f"{safe_name}\n")


def clear_default_saved_config_name(name: str | None = None):
    if not core.DEFAULT_SAVED_CONFIG_FILE.exists():
        return
    if name is not None and get_default_saved_config_name() != saved_config_name(name):
        return
    core.DEFAULT_SAVED_CONFIG_FILE.unlink()


def launch_chat_backend_for_saved_config(active: dict | None) -> tuple[bool, str, list[str]]:
    active = active or {}
    variant = active.get("variant")
    service = active.get("service")
    if variant in BUILTIN_CHAT_VARIANT_IDS:
        ok, output = core.run_script('switch-chat-model.sh', variant)
        return ok, output, [BUILTIN_CHAT_VARIANT_BY_ID[variant]["service"], "chat-proxy"]

    if service != "chat-backend":
        core.ServiceManager.start('chat-proxy')
        return True, "No saved chat backend was active; left chat backend unchanged.", []

    for svc in ('chat-backend-dense', 'chat-backend-moe', 'chat-backend',
                'qwen-chat-backend-27b', 'qwen-chat-backend-35b', 'qwen-chat-backend'):
        core.ServiceManager.stop(svc)
    r = core.ServiceManager.start('chat-backend')
    core.ServiceManager.start('chat-proxy')
    return r.returncode == 0, (r.stdout + r.stderr).strip(), ["chat-backend", "chat-proxy"]


def process_cmdline(pid: int) -> str:
    """How a process was actually launched, which is not always how it is configured."""
    try:
        return Path(f"/proc/{int(pid)}/cmdline").read_text(errors="ignore").replace("\x00", " ").strip()
    except Exception:
        return ""


_CGROUP_UNIT_RE = re.compile(r"/([\w\-.@\\]+)\.service\b")


def process_unit(pid: int) -> str:
    """The systemd unit a PID belongs to, read from its cgroup.

    Matching against each unit's MainPID only ever finds the process systemd
    started, and the interesting ones are often children: `llama-router` forks a
    `llama-server` per resident model, and it is those children that hold the
    VRAM. The cgroup names the unit for every process in it, parent or child,
    from one file read and no subprocess.
    """
    try:
        text = Path(f"/proc/{int(pid)}/cgroup").read_text(errors="ignore")
    except (OSError, ValueError):
        return ""
    match = _CGROUP_UNIT_RE.search(text)
    return match.group(1) if match else ""


def process_model_args(cmdline: str) -> tuple[str, str]:
    """(model path, alias) from a llama-server command line.

    The router's children are distinguished only by these: same binary, same
    unit, different model. Without them every model the router holds would land
    in one indistinguishable block.
    """
    parts = cmdline.split()
    model = alias = ""
    for flag, value in zip(parts, parts[1:]):
        if flag == "--model" and not model:
            model = value
        elif flag == "--alias" and not alias:
            alias = value
    return model, alias


@core.ttl_cache(2.0)
def service_main_pids() -> dict[int, str]:
    """PID -> service name for every managed unit.

    Cached because it costs one `systemctl show` per service and is only ever
    used to label GPU processes, which nothing needs sub-second.
    """
    mapping = {}
    for svc in SERVICES:
        name = svc.get("name")
        if not name:
            continue
        try:
            pid = core.ServiceManager.get_pid(name)
            if pid > 0:
                mapping[pid] = name
        except Exception:
            continue
    return mapping


def label_gpu_process(pid: int, process_name: str, service_pids: dict[int, str]) -> str:
    """Which service a GPU compute process belongs to.

    nvidia-smi reports the executable, which for anything launched through a
    wrapper is the interpreter — every Python service on the box would otherwise
    be labelled `python3` and share one indistinguishable block of VRAM. The
    cmdline is what distinguishes them, so it is read once and used for both the
    service match and the fallback.
    """
    # The cgroup is authoritative and covers children; everything below it is
    # inference from a command line.
    unit = process_unit(pid)
    if unit in {svc.get("name") for svc in SERVICES}:
        return unit
    if pid in service_pids:
        return service_pids[pid]
    cmdline = process_cmdline(pid)
    haystack = f"{process_name} {cmdline}"

    # Longest name first, and never mid-word. A plain substring test made every
    # `glmocr-sdk` process report as `ocr` — "ocr" occurs inside "glmocr" — which
    # silently moved one service's VRAM onto another's row.
    names = sorted((svc.get("name", "") for svc in SERVICES), key=len, reverse=True)
    for name in names:
        if name and f"start-{name}.sh" in haystack:
            return name
    for name in names:
        if name and re.search(rf"(?<![\w-]){re.escape(name)}", haystack):
            return name

    if "llama-server" in haystack:
        return "llama-server"
    if "python" in process_name.lower():
        # argv[0] is the interpreter; the script is what identifies the service.
        script = next((part for part in cmdline.split(" ")[1:] if part.endswith(".py")), "")
        return Path(script or process_name).name
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
            model, alias = process_model_args(process_cmdline(pid))
            processes.setdefault(index, []).append({
                "pid": pid,
                "name": label_gpu_process(pid, process_name, service_pids),
                "process_name": Path(process_name).name,
                "used_memory": used,
                # Which model this particular process is holding. The router runs
                # one child per resident model under a single unit, so the unit
                # alone cannot say what the VRAM is being spent on.
                "model": model,
                "alias": alias,
            })
    # Narrow on purpose: this used to be a bare `except Exception`, and it spent
    # an unknown length of time swallowing a NameError that discarded every
    # attribution here while the payload still looked well-formed.
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        print(f"[llm-manager] GPU process attribution failed: {exc}", flush=True)
    for items in processes.values():
        items.sort(key=lambda item: item.get("used_memory", 0), reverse=True)
    return processes


# Order matters: this is the `--query-gpu` field list and the column order it
# comes back in. The first seven are what the UI has always shown; the rest are
# for API consumers, and every one of them can be `[N/A]` on some card or driver
# — an eGPU reports no fan, a datacentre card no power limit — so they parse to
# None rather than failing the row.
GPU_QUERY_FIELDS = [
    'index', 'uuid', 'name', 'memory.used', 'memory.total',
    'utilization.gpu', 'temperature.gpu',
    'utilization.memory', 'power.draw', 'enforced.power.limit',
    'clocks.current.sm', 'clocks.current.memory', 'fan.speed', 'pstate',
]


def _gpu_number(value: str):
    """A numeric nvidia-smi field, or None for the several ways it says N/A."""
    text = (value or "").strip()
    if not text or text.startswith("[") or text.lower() in {"n/a", "unknown"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return int(number) if number.is_integer() else round(number, 2)


@core.ttl_cache(2.0)
def get_gpu_info() -> list:
    try:
        r = subprocess.run(
            ['nvidia-smi',
             '--query-gpu=' + ','.join(GPU_QUERY_FIELDS),
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

                def field(position: int) -> str:
                    return parts[position] if position < len(parts) else ""

                gpus.append({
                    'index':     index,
                    'uuid':      parts[1],
                    'name':      parts[2],
                    'mem_used':  mem_used,
                    'mem_total': mem_total,
                    'util':      int(parts[5]),
                    'temp':      int(parts[6]),
                    'mem_pct':   round(100 * mem_used / max(mem_total, 1)),
                    'mem_free':  max(0, mem_total - mem_used),
                    'mem_util':      _gpu_number(field(7)),
                    'power_watts':   _gpu_number(field(8)),
                    'power_limit_watts': _gpu_number(field(9)),
                    'clock_sm_mhz':  _gpu_number(field(10)),
                    'clock_mem_mhz': _gpu_number(field(11)),
                    'fan_pct':       _gpu_number(field(12)),
                    'pstate':        field(13) or None,
                    'processes': [],
                })
        processes = get_gpu_processes(uuid_by_index)
        for gpu in gpus:
            gpu["processes"] = processes.get(gpu["index"], [])
        return gpus
    except Exception:
        return []




def determine_llamacpp_build_parallelism(env: dict) -> tuple[int, list[str]]:
    notes: list[str] = []
    cpu_count = max(os.cpu_count() or 1, 1)
    meminfo = core.read_meminfo()
    mem_available_kib = meminfo.get("MemAvailable", 0)
    swap_free_kib = meminfo.get("SwapFree", 0)

    notes.append(f"detected CPU cores: {cpu_count}")
    if mem_available_kib:
        notes.append(f"MemAvailable: {core.format_kib_as_gib(mem_available_kib)}")
    if swap_free_kib or "SwapFree" in meminfo:
        notes.append(f"SwapFree: {core.format_kib_as_gib(swap_free_kib)}")

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
            f"({core.format_kib_as_gib(mem_available_kib)} available, need at least {min_mem_gib:.1f} GiB). "
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
        bin_path = (core.STACK_DIR / bin_path).resolve(strict=False)
    else:
        bin_path = bin_path.resolve(strict=False)

    source_dir = core.find_git_repo_root(bin_path.parent)
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
    rc, out = core.run_command([*git_cmd, "symbolic-ref", "refs/remotes/origin/HEAD"], timeout=30)
    if rc == 0 and out.strip().startswith("refs/remotes/origin/"):
        return out.strip().rsplit("/", 1)[-1]
    for candidate in ("master", "main"):
        rc2, _ = core.run_command([*git_cmd, "show-ref", "--verify", f"refs/remotes/origin/{candidate}"], timeout=30)
        if rc2 == 0:
            return candidate
    raise RuntimeError("Unable to determine upstream default branch (origin/HEAD)")


def get_current_git_branch(git_cmd: list[str]) -> str | None:
    rc, out = core.run_command([*git_cmd, "symbolic-ref", "--quiet", "--short", "HEAD"], timeout=30)
    branch = out.strip()
    return branch if rc == 0 and branch else None


def get_current_git_upstream(git_cmd: list[str]) -> str | None:
    rc, out = core.run_command(
        [*git_cmd, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"],
        timeout=30,
    )
    upstream = out.strip()
    return upstream if rc == 0 and upstream else None


LLAMACPP_UPDATE_IGNORABLE_DIRTY_FILES = {
    "tools/ui/package-lock.json",
}


def llamacpp_patches() -> list[Path]:
    """Local patches this repo re-applies to deps/llama.cpp.

    Declared in dependencies.json so that `install-dependencies.py` and this
    updater work from one list rather than two that can drift apart. See
    docs/llama-cpp-patches.md.
    """
    try:
        manifest = json.loads((core.STACK_DIR / "dependencies.json").read_text())
    except (OSError, json.JSONDecodeError):
        return []
    for dep in manifest.get("dependencies", []):
        if dep.get("path") == "deps/llama.cpp":
            return [core.STACK_DIR / name for name in dep.get("patches", [])]
    return []


def llamacpp_patched_paths() -> set[str]:
    """Files inside the checkout that those patches modify.

    A patched checkout is dirty by design, so these have to read as expected
    rather than as local edits worth protecting — otherwise applying a patch
    permanently disables the Update llama.cpp button.
    """
    paths: set[str] = set()
    for patch in llamacpp_patches():
        try:
            text = patch.read_text()
        except OSError:
            continue
        for line in text.splitlines():
            if line.startswith("+++ b/"):
                paths.add(line[len("+++ b/"):].strip())
    return paths


def apply_llamacpp_patches(git_cmd: list[str], lines: list[str]) -> bool:
    """Re-apply the local patches after the checkout has moved."""
    for patch in llamacpp_patches():
        if not patch.is_file():
            lines.append(f"Patch is declared but missing: {patch}")
            return False
        cmd = [*git_cmd, "apply", str(patch)]
        rc, out = core.run_command(cmd, timeout=60)
        core.append_command_log(lines, cmd, rc, out)
        if rc != 0:
            lines.append(
                f"Could not apply {patch.name} to the updated checkout. "
                "Refresh the patch against the new revision, or drop it if it landed upstream."
            )
            return False
    return True


def has_uncommitted_git_changes(git_cmd: list[str]) -> tuple[bool, str]:
    rc, out = core.run_command([*git_cmd, "status", "--porcelain"], timeout=30)
    if rc != 0:
        raise RuntimeError(out or "git status failed")
    return bool(out.strip()), out


def try_restore_ignorable_llamacpp_update_changes(git_cmd: list[str], dirty_output: str, lines: list[str]) -> bool:
    entries = [line for line in dirty_output.splitlines() if line.strip()]
    if not entries:
        return False

    restorable = LLAMACPP_UPDATE_IGNORABLE_DIRTY_FILES | llamacpp_patched_paths()
    paths: list[str] = []
    for line in entries:
        status = line[:2]
        path = line[3:] if len(line) > 3 else ""
        if status != " M" or path not in restorable:
            return False
        paths.append(path)

    lines.append(
        "Only generated metadata and locally patched files changed; "
        "restoring them before update. Patches are re-applied after the pull."
    )
    cmd = [*git_cmd, "restore", "--", *paths]
    rc, out = core.run_command(cmd, timeout=60)
    core.append_command_log(lines, cmd, rc, out)
    if rc != 0:
        return False

    dirty_after, dirty_after_output = has_uncommitted_git_changes(git_cmd)
    if dirty_after:
        lines.append("llama.cpp checkout is still dirty after restoring generated files:")
        lines.append(dirty_after_output)
        return False
    return True


def local_branch_exists(git_cmd: list[str], branch: str) -> bool:
    rc, _ = core.run_command([*git_cmd, "show-ref", "--verify", f"refs/heads/{branch}"], timeout=30)
    return rc == 0


def remote_branch_exists(git_cmd: list[str], branch: str) -> bool:
    rc, _ = core.run_command([*git_cmd, "show-ref", "--verify", f"refs/remotes/origin/{branch}"], timeout=30)
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
        rc, out = core.run_command(cmd, timeout=60)
        core.append_command_log(lines, cmd, rc, out)
        if rc == 0:
            return True, branch
        last_output = out
    return False, last_output or f"Unable to switch to branch {branch}"


def update_llamacpp_and_restart_active_services() -> tuple[bool, str, list[str]]:
    lines: list[str] = []
    restarted: list[str] = []
    try:
        env = config_env.read_env()
        bin_path, build_dir, source_dir = resolve_llamacpp_paths(env)
        git_cmd = ["git", "-c", f"safe.directory={source_dir}", "-C", str(source_dir)]

        lines.append(f"LLAMA_SERVER_BIN: {bin_path}")
        lines.append(f"llama.cpp source: {source_dir}")
        lines.append(f"build dir: {build_dir}")
        lines.append(f"configured binary exists: {'yes' if bin_path.exists() else 'no'}")

        build_jobs, build_notes = determine_llamacpp_build_parallelism(env)
        lines.extend(build_notes)

        rc, out = core.run_command([*git_cmd, "remote", "set-head", "origin", "-a"], timeout=60)
        core.append_command_log(lines, [*git_cmd, "remote", "set-head", "origin", "-a"], rc, out)
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
            rc, out = core.run_command(cmd, timeout=3600)
            core.append_command_log(lines, cmd, rc, out)
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
            rc, out = core.run_command(cmd, timeout=3600)
            core.append_command_log(lines, cmd, rc, out)
            if rc != 0:
                return False, "\n".join(lines), restarted

        # The pull discarded them along with everything else, and this path builds
        # with cmake directly rather than through install-dependencies.py, so it
        # has to put them back itself or the update silently reverts the fix.
        if not apply_llamacpp_patches(git_cmd, lines):
            return False, "\n".join(lines), restarted

        build_dir.mkdir(parents=True, exist_ok=True)
        for cmd in (
            ["cmake", "-S", str(source_dir), "-B", str(build_dir)],
            ["nice", "-n", "15", "cmake", "--build", str(build_dir), "--target", "llama-server", "--parallel", str(build_jobs)],
        ):
            rc, out = core.run_command(cmd, timeout=3600)
            core.append_command_log(lines, cmd, rc, out)
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
            rc, out = core.ServiceManager.restart(svc, timeout=120)
            core.append_command_log(lines, cmd, rc, out)
            if rc == 0:
                restarted.append(svc)
            else:
                restart_failures.append(svc)

        if get_service_status(LLAMACPP_PROXY_SERVICE) == "active":
            cmd = ["ServiceManager.restart", LLAMACPP_PROXY_SERVICE]
            rc, out = core.ServiceManager.restart(LLAMACPP_PROXY_SERVICE, timeout=120)
            core.append_command_log(lines, cmd, rc, out)
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
        return core.ServiceManager.is_installed(name)
    except Exception:
        return False


def should_use_local_tts_manager(name: str) -> bool:
    return is_tts_service(name) and not systemd_unit_exists(name)


def run_tts_manager(name: str, action: str) -> tuple:
    try:
        r = subprocess.run(
            ['bash', str(core.SCRIPTS_DIR / 'manage-tts-service.sh'), name, action],
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
            ['bash', str(core.SCRIPTS_DIR / 'manage-transcript-service.sh'), action],
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
        socket_path = Path(config_env.read_env().get("SEARXNG_UWSGI_SOCKET", "/usr/local/searxng/run/socket"))
        if uwsgi == "active" and nginx == "active" and socket_path.exists():
            return True, "active"
        if uwsgi == "failed" or nginx == "failed":
            return True, "failed"
        return True, "inactive"
    try:
        if action == "start":
            core.ServiceManager.start("nginx")
            r = core.ServiceManager.start("uwsgi")
            return r.returncode == 0, (r.stdout + r.stderr).strip()
        if action == "stop":
            r = core.ServiceManager.stop("uwsgi")
            return r.returncode == 0, (r.stdout + r.stderr).strip()
        if action == "restart":
            rc, output = core.ServiceManager.restart("uwsgi")
            core.ServiceManager.run_cmd(["nginx", "-t"], timeout=10)
            core.ServiceManager.run_cmd(["systemctl", "reload", "nginx"], timeout=10)
            return rc == 0, output
        if action == "install":
            r = subprocess.run(["bash", str(core.SCRIPTS_DIR / "install-searxng.sh")], capture_output=True, text=True, timeout=900)
            return r.returncode == 0, (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return False, "SearXNG manager action timed out"
    except Exception as exc:
        return False, str(exc)
    return False, "unsupported action"


def tts_log_file(name: str) -> Path:
    return core.LOGS_DIR / 'tts' / f'{name}.log'


def transcript_log_file() -> Path:
    return core.LOGS_DIR / 'transcript' / f'{TRANSCRIPT_MANAGED_SERVICE}.log'




def tts_gateway_url() -> str:
    env = config_env.read_env()
    port = env.get("TTS_GATEWAY_PORT", "8060")
    return f"http://127.0.0.1:{port}"






def load_tts_backends() -> list:
    return core.load_json_file(core.TTS_CONFIG_FILE, {"backends": []}).get("backends", [])


def load_tts_state() -> dict:
    return core.load_json_file(core.TTS_STATE_FILE, {"active_backend": None, "updated_at": None})


def wait_for_tts_gateway(timeout: float = 15.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            core.http_json(f'{tts_gateway_url()}/health', timeout=3)
            return True
        except Exception:
            time.sleep(0.5)
    return False


# ---------------------------------------------------------------------------
# GGUF / Custom models / Saved configs helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    env = config_env.read_env()
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
                           custom_models=models.load_custom_models(),
                           builtin_chat_variants=builtin_chat_variants(env),
                           models_dir=str(core.MODELS_DIR),
                           asset_version=asset_version())


def asset_version() -> str:
    """Cache key for the static scripts, stamped with the deployed commit.

    `url_for('static', ...)` emits no version, so a browser holding the previous
    build's modules would keep them across a deploy — and a page that mixes new
    markup with old handlers fails in ways that look like anything but a caching
    problem. The commit already identifies the tree exactly, and the watcher has
    it cached, so this costs nothing per render.
    """
    return DEPLOY_WATCHER.snapshot().get('head_short') or 'dev'


def service_names(env: dict | None = None) -> set:
    """Every service the panel shows, plus every unit named as an upstream.

    Units that only appear as somebody's upstream are asked about too: the panel
    shows one card for the primary backend, but three mutually exclusive units
    can serve it, and a proxy whose backend is one of the other two is not
    degraded.
    """
    return {s['name'] for s in patch_service_labels(env or config_env.read_env())} | health.dependency_units()


def service_unit_snapshot(env: dict | None = None) -> tuple[dict, dict]:
    """(status per service, restart count per service) from one pass.

    The restart count is only meaningful as a delta between polls, which is what
    `health.RESTARTS` keeps: a service that cannot start spends its life
    bouncing between `activating` and `failed`, so any single sample catches
    whichever phase the poll happened to land in and none of them say "this
    keeps dying".
    """
    statuses, restarts = {}, {}
    for name in service_names(env):
        if (is_searxng_service(name) or should_use_local_transcript_manager(name)
                or should_use_local_tts_manager(name)):
            statuses[name] = get_service_status(name)
            continue
        try:
            state = core.ServiceManager.state(name)
        except Exception:
            statuses[name] = 'unknown'
            continue
        statuses[name] = status_from_unit_state(state)
        restarts[name] = state.get('n_restarts', 0)
    return statuses, restarts


def all_service_statuses(env: dict | None = None) -> dict:
    return service_unit_snapshot(env)[0]


def sweep_pi_forge_leases() -> None:
    """Opt-in orphan reaping, riding the health prober's cadence.

    Off by default: `~/.pi-forge/agent/inference-leases` belongs to pi-forge, and
    the manager should not delete another program's files unless asked. Asking
    is either this flag or the button, and both use the same conservative rule.
    """
    env = config_env.read_env()
    if str(env.get('PI_FORGE_LEASE_REAP', 'off')).strip().lower() != 'on':
        return
    scheduling.reap_leases(scheduling.lease_directory(env))


health.PROBER.add_task(sweep_pi_forge_leases)


def service_health_snapshot(env: dict | None = None) -> tuple[dict, dict]:
    """(unit statuses for the panel, health for every service it can judge)."""
    env = env or config_env.read_env()
    statuses, restarts = service_unit_snapshot(env)
    flapping = health.RESTARTS.observe(restarts)
    health.PROBER.start(config_env.read_env, all_service_statuses)
    DEPLOY_WATCHER.start(config_env.read_env)
    entries = health.collect(env, statuses, health.PROBER.snapshot(), flapping=flapping)
    panel = {s['name'] for s in patch_service_labels(env)}
    return {name: statuses[name] for name in panel}, entries


@app.route('/api/status')
def api_status():
    env = config_env.read_env()
    statuses, service_health = service_health_snapshot(env)
    # Deployment rides this poll rather than adding a timer of its own, and is a
    # cache read: the fetch that fills it happens on the watcher's thread.
    return jsonify(services=statuses, health=service_health, gpus=get_gpu_info(),
                   contexts=backend_context_summary(env),
                   deployment=deployment_report())


@app.route('/api/backend/telemetry')
def api_backend_telemetry():
    """Runtime detail for every active llama.cpp backend.

    Combines /props, /slots and /metrics with parsed journal history, so the UI
    can show throughput, prompt-cache behaviour and slot scheduling rather than
    just "active". Widening `window` re-seeds the journal backfill, which makes
    this endpoint usable for before/after comparisons of backend settings.
    """
    window = telemetry.clamp_window(
        request.args.get('window'), telemetry.DEFAULT_WINDOW_SECONDS)
    return jsonify(telemetry.collect(
        config_env.read_env(),
        get_service_status,
        get_gpu_info(),
        core.read_meminfo(),
        window_seconds=window,
    ))


@app.route('/api/backend/budget')
def api_backend_budget():
    """Predicted memory footprint of a backend's configuration.

    Answers the question the config form has never been able to: will this fit,
    and if not, what is the term that does not. Query parameters override
    individual settings, so the form can price a change before it is saved —
    `?ctx_size=131072&ctx_checkpoints=4` prices that edit against the model
    currently configured.

    Reading GGUF geometry means touching the model file, so this cannot move to
    the browser; it is the single source of truth for anything that needs to
    recommend or validate a backend configuration.
    """
    backend = request.args.get('backend', 'chat-primary')
    if backend not in budget.BACKEND_PREFIXES:
        return jsonify(error=f"unknown backend {backend!r}",
                       backends=sorted(budget.BACKEND_PREFIXES)), 400

    overrides = {key: value for key, value in request.args.items()
                 if key in budget.OVERRIDABLE_SETTINGS and value != ''}
    return jsonify(budget.budget_for(
        config_env.read_env(),
        backend,
        get_gpu_info(),
        telemetry.host_memory(core.read_meminfo()),
        overrides=overrides,
    ))


@app.route('/api/backend/budget/recommend')
def api_backend_budget_recommend():
    """A backend configuration derived from this host, not from constants.

    The recommended preset used to be hardcoded, and the values it recommended
    were the ones measured to thrash this box. This computes them instead, from
    detected VRAM, host RAM and the selected model's own geometry.
    """
    backend = request.args.get('backend', 'chat-primary')
    if backend not in budget.BACKEND_PREFIXES:
        return jsonify(error=f"unknown backend {backend!r}",
                       backends=sorted(budget.BACKEND_PREFIXES)), 400

    env = config_env.read_env()
    model_path = request.args.get('model_path') or env.get(
        f"{budget.BACKEND_PREFIXES[backend]}_MODEL_PATH", '')
    try:
        slots = max(1, int(request.args.get('slots') or 2))
    except ValueError:
        slots = 2

    try:
        geometry = budget.model_geometry(budget.read_gguf_metadata(model_path))
    except (budget.GGUFError, OSError, TypeError) as exc:
        return jsonify(error=f"could not read model metadata: {exc}", model_path=model_path), 400

    # Price candidates against the backend as it will actually launch. The
    # projector and the draft head are worth gigabytes that a bare prediction
    # never sees, and a recommendation blind to them recommends what does not fit.
    base = budget.settings_from_env(env, backend)
    projector = base.get("mmproj_path") or ''
    if projector and Path(projector).is_file():
        base["projector_mib"] = Path(projector).stat().st_size / budget.MIB

    return jsonify(budget.recommend(
        geometry,
        get_gpu_info(),
        telemetry.host_memory(core.read_meminfo()),
        slots=slots,
        base_settings=base,
    ) | {"model_path": model_path, "backend": backend})


def scheduling_verification(window_seconds: int | None = None) -> dict:
    """Assemble everything `scheduling.verify` needs from the running stack."""
    env = config_env.read_env()
    window = telemetry.clamp_window(window_seconds, telemetry.DEFAULT_WINDOW_SECONDS)
    unit = next((name for name in ('chat-backend-dense', 'chat-backend-moe', 'chat-backend')
                 if get_service_status(name) == 'active'), None)
    props = slots = stats = None
    cmdline = ''
    if unit:
        target = next((t for t in telemetry.resolve_targets(env, get_service_status)
                       if t['name'] == 'chat-primary'), None)
        if target:
            props = telemetry.probe_props(target['base_url'])
            slots = telemetry.probe_slots(target['base_url'])
        cmdline = process_cmdline(core.ServiceManager.get_pid(unit))
        collector = telemetry.REGISTRY.collector(unit, window)
        stats = telemetry.summarize(collector.snapshot(), window)
    return scheduling.verify(
        env, unit=unit, unit_active=bool(unit), cmdline=cmdline,
        props=props, slots=slots, stats=stats, window_seconds=window)


@app.route('/api/scheduling/verify')
def api_scheduling_verify():
    """Whether the running backend can honour the pi-forge slot contract.

    Passive: it reads `/props`, `/slots`, the backend's own command line and the
    journal telemetry already tails. Sending nothing matters here — a probe
    pinned to a slot displaces the prefix that slot is holding, which is the
    eviction the rest of this work removed. The active check is the POST below,
    and it is opt-in.
    """
    return jsonify(scheduling_verification(request.args.get('window')))


@app.route('/api/scheduling/verify', methods=['POST'])
def api_scheduling_verify_probe():
    """The passive report, plus an optional live `id_slot` probe.

    `{"probe": true}` sends one minimal request per slot and reads back from the
    journal which slot served it. It costs both slots their cached prefixes.
    """
    body = request.get_json(silent=True) or {}
    result = scheduling_verification(body.get('window'))
    if not body.get('probe'):
        return jsonify(result)
    if not result['unit_active']:
        return jsonify(result | {"probe": {"error": "no primary backend is running"}}), 409
    window = telemetry.clamp_window(body.get('window'), telemetry.DEFAULT_WINDOW_SECONDS)
    collector = telemetry.REGISTRY.collector(result['unit'], window)
    return jsonify(result | {"probe": scheduling.probe_slot_pinning(config_env.read_env(), collector)})


@app.route('/api/scheduling/leases/reap', methods=['POST'])
def api_scheduling_leases_reap():
    """Delete leases whose writer is gone and whose claim is long expired.

    The directory belongs to pi-forge, so this never runs unasked unless
    `PI_FORGE_LEASE_REAP` is turned on, and it never touches a lease that is
    fresh or whose process is still alive.
    """
    body = request.get_json(silent=True) or {}
    env = config_env.read_env()
    return jsonify(scheduling.reap_leases(scheduling.lease_directory(env),
                                          dry_run=bool(body.get('dry_run'))))




@app.route('/api/service/<name>/health')
def api_service_health(name):
    """Everything behind one card's colour: probe, upstreams, expectation."""
    env = config_env.read_env()
    statuses = all_service_statuses(env)
    if name not in statuses:
        return jsonify(ok=False, error='Unknown service'), 400
    entries = health.collect(env, statuses, health.PROBER.snapshot())
    endpoint = health.endpoint_for(name, env)
    return jsonify(ok=True, name=name, health=entries.get(name),
                   endpoint=f"{endpoint[0]}:{endpoint[1]}" if endpoint else None,
                   unit=core.ServiceManager.state(name))


@app.route('/api/service/<name>/expect', methods=['POST'])
def api_service_expect(name):
    """Say whether a service is meant to be running.

    Nothing else on the box carries this: `systemctl is-enabled` reads
    `disabled` for units that are running right now, and the `*_ENABLED` env
    flags read `on` for services that are deliberately stopped.
    """
    if name not in {s['name'] for s in patch_service_labels()}:
        return jsonify(ok=False, error='Unknown service'), 400
    expected = ((request.get_json(silent=True) or {}).get('expected') or '').strip().lower()
    try:
        health.record_expectation(name, expected, source='operator')
    except ValueError as exc:
        return jsonify(ok=False, error=str(exc)), 400
    return jsonify(ok=True, name=name, expected=expected)


@app.route('/api/service/<name>/<action>', methods=['POST'])
def api_service_action(name, action):
    if action not in ('start', 'stop', 'restart'):
        return jsonify(ok=False, error='Unknown action'), 400
    if name not in {s['name'] for s in patch_service_labels()}:
        return jsonify(ok=False, error='Unknown service'), 400
    if name in router_pooled_units(config_env.read_env()):
        # Starting it would fight nginx for the port and put a second copy of
        # the model on the GPU. Say so rather than half-succeeding.
        return jsonify(
            ok=False,
            error=(f"{name} is held by the model router, which loads it on demand. "
                   f"Use the Model Router controls, or turn MODEL_ROUTER_ENABLED off "
                   f"to run it as its own service again."),
        ), 409
    if is_searxng_service(name):
        ok, output = run_searxng_manager(action)
        record_service_expectation(name, action, ok)
        return jsonify(ok=ok, output=output)
    if should_use_local_transcript_manager(name):
        ok, output = run_transcript_manager(action)
        record_service_expectation(name, action, ok)
        return jsonify(ok=ok, output=output)
    if should_use_local_tts_manager(name):
        ok, output = run_tts_manager(name, action)
        record_service_expectation(name, action, ok)
        return jsonify(ok=ok, output=output)
    try:
        rc, output = core.ServiceManager.action(action, name, timeout=30)
        record_service_expectation(name, action, rc == 0)
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
        result = core.http_json(f"{cfg['local_url']}search?q=llm-stack&format=json", timeout=8)
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


def _model_router_url(env: dict, path: str) -> str:
    host = env.get("MODEL_ROUTER_HOST") or "127.0.0.1"
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1"
    port = env.get("MODEL_ROUTER_PORT", "8013")
    return f"http://{host}:{port}{path}"


def _model_router_request(env: dict, path: str, payload=None, timeout=15):
    """Call the router's control API. Returns (ok, parsed-or-error-string)."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urlrequest.Request(
        _model_router_url(env, path),
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data is not None else "GET",
    )
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        return True, (json.loads(raw) if raw.strip() else {})
    except urlerror.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else ""
        return False, f"router returned HTTP {exc.code}: {body[:400]}"
    except Exception as exc:
        return False, str(exc)


def model_router_overview(env: dict) -> dict:
    """Which pooled models exist and which are resident right now.

    `GET /models` is one of the endpoints llama.cpp documents as never loading
    a model, so the services page can poll this without causing the very
    swapping it is reporting on.

    Separate from the route because the state API reports the same thing, and a
    second implementation of "which models are loaded" is a second answer to it.
    """
    if not telemetry.router_enabled(env):
        return {"ok": True, "enabled": False, "models": []}
    ok, result = _model_router_request(env, "/models")
    if not ok:
        return {"ok": False, "enabled": True, "reachable": False, "error": result, "models": []}
    models = []
    for entry in (result.get("data") or []):
        status = entry.get("status") or {}
        models.append({
            "id": entry.get("id"),
            "state": status.get("value") if isinstance(status, dict) else str(status),
            "path": entry.get("path", ""),
        })
    models.sort(key=lambda m: str(m.get("id") or ""))
    return {"ok": True, "enabled": True, "reachable": True, "models": models,
            "max_resident": env.get("MODEL_ROUTER_MAX", "2")}


@app.route('/api/model-router')
def api_model_router():
    return jsonify(model_router_overview(config_env.read_env()))


@app.route('/api/model-router/<action>', methods=['POST'])
def api_model_router_action(action):
    if action not in ('load', 'unload'):
        return jsonify(ok=False, error='Unknown action'), 400
    env = config_env.read_env()
    if not telemetry.router_enabled(env):
        return jsonify(ok=False, error='The model router is not enabled'), 409
    model = str((request.get_json(silent=True) or {}).get('model') or '').strip()
    if not model:
        return jsonify(ok=False, error='model is required'), 400
    # A cold load is the whole point, so allow for one rather than timing out
    # partway through and reporting a failure that is still in progress.
    timeout = int(float(env.get("MODEL_ROUTER_LOAD_TIMEOUT", "600") or 600))
    ok, result = _model_router_request(
        env, f"/models/{action}", {"model": model},
        timeout=timeout if action == 'load' else 60)
    if not ok:
        return jsonify(ok=False, error=result), 502
    return jsonify(ok=True, result=result)


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
    manager_port = str(config_env.read_env().get("LLM_MANAGER_PORT", "8077"))
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
    env = env or config_env.read_env()
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
    env = env or config_env.read_env()
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
        "browsers_path": env.get("PLAYWRIGHT_BROWSERS_PATH", str(core.STACK_DIR / "playwright" / "browsers")),
        "node_env": env.get("PLAYWRIGHT_NODE_ENV", "production"),
        "public_ws_url": public_ws,
        "public_http_url": public_http,
        "server_dir": str(core.STACK_DIR / "playwright"),
        "package_json": str(core.STACK_DIR / "playwright" / "package.json"),
        "node_modules": str(core.STACK_DIR / "playwright" / "node_modules"),
        "service_unit": "/etc/systemd/system/playwright-server.service",
        "nginx_conf": env.get("PLAYWRIGHT_NGINX_CONF", "/etc/nginx/default.apps-available/playwright.conf"),
        "endpoints": endpoints,
    }


# The readiness probes need the same listener check, so it lives with them.
tcp_port_open = health.tcp_port_open


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
    stat = core.STACK_DIR.stat()
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
WorkingDirectory={core.STACK_DIR / 'playwright'}
EnvironmentFile={core.CONFIG_FILE}
Environment=NODE_ENV={cfg['node_env']}
ExecStart={core.STACK_DIR / 'playwright' / 'start.sh'}
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
    core.ServiceManager.run_cmd(["systemctl", "daemon-reload"], timeout=15)


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
    nginx_test = core.ServiceManager.run_cmd(["nginx", "-t"], timeout=15)
    if nginx_test.returncode != 0:
        raise RuntimeError((nginx_test.stdout + nginx_test.stderr).strip())
    core.ServiceManager.run_cmd(["systemctl", "reload", "nginx"], timeout=15)


def run_playwright_install() -> tuple[bool, str]:
    cfg = playwright_config()
    env = os.environ.copy()
    env.update({
        "PLAYWRIGHT_ENABLED": cfg["enabled"],
        "PLAYWRIGHT_BROWSER": cfg["browser"],
        "PLAYWRIGHT_INSTALL_BROWSERS": cfg["install_browsers"],
        "PLAYWRIGHT_BROWSERS_PATH": cfg["browsers_path"],
    })
    cmd = ["bash", str(core.SCRIPTS_DIR / "install-playwright.sh")]
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
        if not core.ServiceManager.IS_MAC and os.geteuid() == 0:
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
    env = config_env.read_env()
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
    env = config_env.read_env()
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
    ok, output = core.run_script('restore-active-stack.sh')
    return jsonify(ok=ok, output=output)


def deployment_report(snapshot: dict | None = None) -> dict:
    """The deployed tree's state plus the one-line verdict the badge renders."""
    snapshot = DEPLOY_WATCHER.snapshot() if snapshot is None else snapshot
    return {**snapshot, "summary": deploy.summarize(snapshot), "remedy": deploy.REMEDY}


@app.route('/api/deploy/status')
def api_deploy_status():
    """What the installed tree is running, and how far behind it has fallen.

    Always a cache read. The fetch that fills the cache runs on the watcher's
    thread, because this endpoint is on the 5s UI poll path and a network round
    trip there would stall every card on the page.
    """
    DEPLOY_WATCHER.start(config_env.read_env)
    return jsonify(deployment_report())


@app.route('/api/deploy/check', methods=['POST'])
def api_deploy_check():
    """Fetch now rather than waiting for the next interval."""
    return jsonify(deployment_report(DEPLOY_WATCHER.check()))


@app.route('/api/app/update', methods=['POST'])
def api_app_update():
    script_path = os.path.join(core.STACK_DIR, 'update.sh')
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
            core.ServiceManager.stop(svc)
        ok, output = core.run_script('switch-chat-model.sh', variant)
        return jsonify(ok=ok, output=output)

    # Custom model switch
    models = models.load_custom_models()
    model = next((m for m in models if m['id'] == variant), None)
    if not model:
        return jsonify(ok=False, error='Unknown model variant'), 400

    # Stop all chat backends
    for svc in ('chat-backend-dense', 'chat-backend-moe', 'chat-backend', 'qwen-chat-backend-27b', 'qwen-chat-backend-35b', 'qwen-chat-backend'):
        core.ServiceManager.stop(svc)

    # Update env with custom model paths
    updates = {
        'CHAT_MODEL_PATH': model['model_path'],
        'CHAT_MODEL_NAME': model.get('model_name', 'chat-custom'),
        'CHAT_CTX_SIZE': model.get('ctx_size', '32768'),
        'CHAT_CUSTOM_ARGS_JSON': json.dumps(models.resolve_custom_args_for_model(model)[0]),
    }
    if model.get('mmproj_path'):
        updates['CHAT_MMPROJ_PATH'] = model['mmproj_path']
    else:
        updates['CHAT_MMPROJ_PATH'] = ''
    config_env.update_env_values(updates)

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




@app.route('/api/active-chat-model')
def api_active_chat_model():
    """Determine which chat model is currently loaded."""
    active = active_chat_model_snapshot()
    payload = dict(active)
    if active.get('kind') == 'custom':
        for m in models.load_custom_models():
            if m.get('id') == active.get('variant'):
                payload['custom_model'] = m
                break
    return jsonify(payload)


@app.route('/api/tts/overview')
def api_tts_overview():
    env = config_env.read_env()
    state = load_tts_state()
    backends = load_tts_backends()
    gateway_data = {}
    gateway_error = None
    try:
        gateway_data = core.http_json(f'{tts_gateway_url()}/api/backends', timeout=10)
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
        if config_env.read_env().get('TTS_SINGLE_ACTIVE', 'on') != 'off':
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
        gateway_result = core.http_json(f'{tts_gateway_url()}/api/activate/{backend_id}', method='POST', timeout=15)
        return jsonify(ok=True, backend_id=backend_id, gateway=gateway_result, output='\n'.join(filter(None, outputs)))
    except urlerror.HTTPError as exc:
        return jsonify(ok=False, error=exc.read().decode('utf-8', errors='ignore') or str(exc)), exc.code
    except Exception as exc:
        return jsonify(ok=False, error=str(exc)), 500


@app.route('/api/tts/test', methods=['POST'])
def api_tts_test():
    payload = request.json or {}
    try:
        audio, content_type = core.http_bytes(f'{tts_gateway_url()}/v1/audio/speech', method='POST', payload=payload, timeout=300)
        return Response(audio, mimetype=content_type)
    except urlerror.HTTPError as exc:
        return jsonify(ok=False, error=exc.read().decode('utf-8', errors='ignore') or str(exc)), exc.code
    except Exception as exc:
        return jsonify(ok=False, error=str(exc)), 500





@app.route('/api/saved-configs', methods=['GET'])
def api_saved_configs_list():
    configs = []
    default_name = get_default_saved_config_name()
    for f in sorted(core.SAVED_CONFIGS_DIR.glob('*.json')):
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
    env = config_env.read_env()
    config = config_env.normalize_env_keys(env)
    form_config = (data or {}).get('config')
    if isinstance(form_config, dict):
        snapshot = config_env.config_form_snapshot(form_config, env)
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
        core.SAVED_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
        (core.SAVED_CONFIGS_DIR / f'{safe_name}.json').write_text(
            json.dumps(config, indent=2))
    except Exception as exc:
        return jsonify(ok=False, error=f'Could not save config: {exc}'), 500
    return jsonify(ok=True, name=safe_name)


@app.route('/api/saved-configs/<name>', methods=['GET'])
def api_saved_configs_load(name):
    safe_name = re.sub(r'[^\w\-]', '_', name)
    path = core.SAVED_CONFIGS_DIR / f'{safe_name}.json'
    if not path.exists():
        return jsonify(ok=False, error='Config not found'), 404
    return jsonify(json.loads(path.read_text()))


def apply_saved_config(name: str, launch: bool = False) -> dict:
    safe_name = saved_config_name(name)
    path = core.SAVED_CONFIGS_DIR / f'{safe_name}.json'
    if not path.exists():
        return {'ok': False, 'error': 'Config not found'}
    config = json.loads(path.read_text())
    updates = saved_config_apply_updates(config)
    updates = config_env.apply_code_chat_mirrors(updates)
    # Reported, not enforced. Applying a saved profile is a deliberate choice
    # of a whole configuration, and this path also runs at startup, where
    # refusing would leave the stack with no configuration at all.
    preflight = preflight_config(updates)
    try:
        config_env.update_env_values(updates)
    except Exception as e:
        return {'ok': False, 'error': str(e)}
    restart_needed = set()
    for key in updates:
        restart_needed.update(RESTART_HINTS.get(key, []))
    restart_needed = apply_router_restart_hints(restart_needed, config_env.read_env())

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
                'preflight': preflight,
            }
        restart_needed.difference_update(SHARED_CHAT_BACKEND_RESTART)
        restart_needed.discard('chat-proxy')

        active_services = config.get('_active_services')
        if active_services is not None:
            # Saved profiles predate the router and still list the models as
            # services. Starting one now would race nginx for its port and
            # duplicate a model the router already has in hand, so the router's
            # members are left alone and it decides what is resident.
            pooled = router_pooled_units(config_env.read_env())
            for svc in SERVICES:
                name = svc.get('name')
                if not name or name in ('chat-backend', 'chat-backend-dense', 'chat-backend-moe',
                                        'qwen-chat-backend-27b', 'qwen-chat-backend-35b', 'qwen-chat-backend', 'chat-proxy'):
                    continue
                if name in pooled:
                    continue
                is_active = get_service_status(name) == 'active'
                should_be_active = name in active_services
                # The profile is a statement about which services belong up, so
                # it is also the expectation the services panel judges against.
                health.record_expectation(name, 'on' if should_be_active else 'off',
                                          source='config-apply')
                if should_be_active and not is_active:
                    core.ServiceManager.start(name)
                    launched.append(name)
                elif not should_be_active and is_active:
                    core.ServiceManager.stop(name)
        else:
            secondary = slots.get("secondary") if isinstance(slots.get("secondary"), dict) else {}
            if secondary.get("service") == "chat-backend2" and secondary.get("variant"):
                if get_service_status("chat-backend2") != "active":
                    core.ServiceManager.start("chat-backend2")
                    launched.append("chat-backend2")
                if get_service_status("chat-proxy2") != "active":
                    core.ServiceManager.start("chat-proxy2")
                    launched.append("chat-proxy2")

    return {
        'ok': True,
        'restart_needed': sorted(restart_needed),
        'active_chat_model': active,
        'active_backend_slots': slots,
        'launched_services': launched,
        'output': launch_output,
        'preflight': preflight,
    }


def update_saved_config_values(name: str, updates: dict) -> dict:
    safe_name = saved_config_name(name)
    path = core.SAVED_CONFIGS_DIR / f'{safe_name}.json'
    if not path.exists():
        return {'ok': False, 'error': 'Config not found'}
    if not isinstance(updates, dict):
        return {'ok': False, 'error': 'Expected updates object'}
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            return {'ok': False, 'error': 'Saved config is invalid'}
        filtered = config_env.filter_config_updates(updates)
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
    path = core.SAVED_CONFIGS_DIR / f'{safe_name}.json'
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
    path = core.SAVED_CONFIGS_DIR / f'{safe_name}.json'
    if path.exists():
        path.unlink()
    clear_default_saved_config_name(safe_name)
    return jsonify(ok=True)


@app.route('/api/config', methods=['GET'])
def api_config_get():
    return jsonify(config_env.normalize_env_keys(config_env.read_env()))


def collect_env_deprecations() -> dict:
    """Every legacy env key still written down, and what replaces it.

    Two places carry them: llm-stack.env, which the migration can rewrite, and
    saved profiles, which it deliberately cannot — those are user data, and the
    read-side backfill keeps them working untouched. See
    DEPRECATED_ENV_KEY_NOTES for the staged path this reports against.
    """
    raw = config_env.read_env_raw()
    env_keys = []
    for legacy_key, canonical in LEGACY_ENV_KEY_MAP.items():
        if legacy_key not in raw:
            continue
        env_keys.append({
            "key": legacy_key,
            "replacement": canonical,
            "value": raw[legacy_key],
            # A legacy key whose canonical twin is absent is the only case where
            # deleting the line would lose a value, so migration must write the
            # canonical key first. Reported so that is visible before it runs.
            "canonical_present": canonical in raw,
            "note": DEPRECATED_ENV_KEY_NOTES.get(legacy_key, DEFAULT_DEPRECATION_NOTE),
        })

    profiles = []
    for path in sorted(core.SAVED_CONFIGS_DIR.glob('*.json')) if core.SAVED_CONFIGS_DIR.is_dir() else []:
        try:
            config = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        found = sorted(key for key in LEGACY_ENV_KEY_MAP if key in config)
        form = config.get("_config_form")
        if isinstance(form, dict):
            found = sorted(set(found) | {key for key in LEGACY_ENV_KEY_MAP if key in form})
        if found:
            profiles.append({"name": path.stem, "keys": found})

    return {
        "ok": True,
        "env_file": str(core.CONFIG_FILE),
        "env_keys": sorted(env_keys, key=lambda item: item["key"]),
        "saved_configs": profiles,
        "migratable": len(env_keys),
        "stage": "canonical on write, legacy readable; migration is opt-in",
    }


@app.route('/api/config/deprecations', methods=['GET'])
def api_config_deprecations():
    return jsonify(collect_env_deprecations())


@app.route('/api/config/deprecations/migrate', methods=['POST'])
def api_config_deprecations_migrate():
    """Rewrite llm-stack.env onto the canonical key names.

    `update_env_values` already collapses a canonical key's legacy aliases onto
    one line when it writes, so migrating is writing each canonical key with the
    value the configuration already resolves to. Saved profiles are untouched
    by design.
    """
    report = collect_env_deprecations()
    if not report["env_keys"]:
        return jsonify(ok=True, migrated=[], report=report)

    env = config_env.read_env()
    updates = {}
    for entry in report["env_keys"]:
        canonical = entry["replacement"]
        # read_env has already resolved which value wins; writing that keeps the
        # migration a rename rather than a change of configuration.
        if canonical in env:
            updates[canonical] = env[canonical]
    try:
        config_env.update_env_values(updates)
    except Exception as exc:
        return jsonify(ok=False, error=str(exc)), 500
    return jsonify(ok=True, migrated=sorted(updates), report=collect_env_deprecations())


@app.route('/api/config/preflight', methods=['POST'])
def api_config_preflight():
    """What the budget model makes of a configuration that has not been saved.

    Same body as the save, no write. The form uses this to show the cost of an
    edit while it is still an edit.
    """
    updates = request.json
    if not isinstance(updates, dict):
        return jsonify(ok=False, error='Expected JSON object'), 400
    return jsonify(preflight_config(config_env.apply_code_chat_mirrors(config_env.filter_config_updates(updates))))


@app.route('/api/config', methods=['POST'])
def api_config_save():
    updates = request.json
    if not isinstance(updates, dict):
        return jsonify(ok=False, error='Expected JSON object'), 400
    filtered = config_env.filter_config_updates(updates)
    filtered = config_env.apply_code_chat_mirrors(filtered)

    # Refuse configurations the budget model says cannot allocate. `?force=1`
    # overrides, because the model is a prediction and the operator is the one
    # holding the hardware — but the refusal is the default so the failure
    # surfaces here rather than in a restart loop.
    preflight = preflight_config(filtered)
    forced = str(request.args.get('force', '')).lower() in {'1', 'true', 'yes', 'on'}
    if not preflight['ok'] and not forced:
        return jsonify(
            ok=False,
            error='; '.join(issue['text'] for issue in preflight['errors']),
            preflight=preflight,
        ), 409

    try:
        config_env.update_env_values(filtered)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500
    restart_needed = set()
    for key in filtered:
        restart_needed.update(RESTART_HINTS.get(key, []))
    restart_needed = apply_router_restart_hints(restart_needed, config_env.read_env())
    return jsonify(ok=True, restart_needed=sorted(restart_needed),
                   preflight=preflight, forced=forced and not preflight['ok'])



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


# ---------------------------------------------------------------------------
# The read-only state API
# ---------------------------------------------------------------------------

# Every entry is a lambda rather than the function itself, so the name is looked
# up on this module at call time. Binding `get_gpu_info` here would capture the
# original object and leave `patch.object(manager, 'get_gpu_info', ...)` with
# nothing to patch — the same trap the module-boundary rule exists to avoid.
STATE_API_PROVIDERS = public_api.Providers(
    read_env=lambda: config_env.read_env(),
    service_status=lambda name: get_service_status(name),
    service_health=lambda env: service_health_snapshot(env),
    gpu_info=lambda: get_gpu_info(),
    context_summary=lambda env: backend_context_summary(env),
    deployment=lambda: deployment_report(),
    router_overview=lambda env: model_router_overview(env),
    services_table=lambda env: patch_service_labels(env),
)

STATE_API_BROADCASTER = public_routes.configure(STATE_API_PROVIDERS)


def create_state_api_app() -> Flask:
    """The second listener: the read-only blueprint and nothing else.

    A separate Flask app rather than a prefix or a filter on the existing one,
    because the property worth having is that the routes which stop services and
    read the env file are *not registered* on the port other machines can reach.
    A check that has to be right on every route is a check that will eventually
    be missed on one; an app that never learned those routes cannot serve them.
    """
    # `static_folder=None` because a default Flask app serves `web/static`, and
    # the UI's scripts have no business being on a port whose entire purpose is
    # that it carries less than the manager does.
    state_app = Flask(__name__, static_folder=None)
    state_app.config['PUBLIC_API_ENFORCE_TOKEN'] = True
    state_app.register_blueprint(public_routes.bp)
    return state_app


def start_state_api(settings: dict) -> None:
    """Serve the state API on its own port, on a daemon thread.

    Same process as the manager on purpose: the journal tailers, the health
    prober and the GPU cache are already running here, so a client costs a
    dict serialisation rather than another `nvidia-smi`.
    """
    if not settings.get('enabled'):
        print('[llm-api] LLM_API_ENABLED is off; the state API is not listening.', flush=True)
        return

    from werkzeug.serving import make_server

    host, port = settings['host'], settings['port']
    try:
        server = make_server(host, port, create_state_api_app(), threaded=True)
    # `SystemExit` is not paranoia: werkzeug prints its own message and calls
    # `sys.exit(1)` when the address is in use, which would take the manager's
    # UI down over a port collision on a secondary feature. This is the more
    # important of the two servers and must survive the other failing to bind.
    except (OSError, SystemExit) as exc:
        detail = exc if isinstance(exc, OSError) else 'address already in use'
        print(f'[llm-api] Could not bind {host}:{port} ({detail}); '
              f'the manager is unaffected and the state API is not listening.', flush=True)
        return

    threading.Thread(target=server.serve_forever, name='llm-api', daemon=True).start()
    print(f'[llm-api] Read-only state API on http://{host}:{port}/api/v1/snapshot', flush=True)
    if settings.get('token'):
        print('[llm-api] A token is set; requests must send Authorization: Bearer <token>.', flush=True)
    if settings.get('bind_warning'):
        print(f'[llm-api] WARNING: {settings["bind_warning"]}', flush=True)
    if settings.get('webhook_url'):
        # The broadcaster only runs while something needs it, and a webhook is
        # a reason to run with no client connected.
        STATE_API_BROADCASTER.ensure_running()


if __name__ == '__main__':
    apply_default_saved_config_on_startup()
    start_state_api(public_routes.api_settings())
    port = int(os.environ.get('LLM_MANAGER_PORT', 8080))
    host = os.environ.get('LLM_MANAGER_HOST', '0.0.0.0')
    print(f'[llm-manager] Serving on http://{host}:{port}', flush=True)
    app.run(host=host, port=port, debug=False, threaded=True)
