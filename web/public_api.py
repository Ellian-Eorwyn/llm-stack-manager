#!/usr/bin/env python3
"""
The read-only state API other applications consume.

The manager already knows everything an external app wants to know — which GPUs
are busy, what each model is holding, how much context the slots have eaten,
which backend is unhappy. None of it was reachable: it lived in `/api/status`
and `/api/backend/telemetry`, whose shapes exist to serve one particular page
and change whenever that page does, on a port that is unauthenticated and can
stop services and read every API key in the env file.

This module assembles the same facts into a versioned payload with a contract,
and `routes/public.py` serves it from a listener that has no mutating routes on
it at all. That separation is the actual safety property here; the optional
token is a convenience on top of it.

**It does not import `app`.** Everything it needs from that layer arrives as a
`Providers` bundle of callables, built in `app.py` and passed in. That keeps the
one-way module dependency the rest of `web/` follows, and it is what lets the
tests drive a whole snapshot without a systemd, a GPU or a running backend.

Three things here are genuinely new rather than a reshuffle of existing data:

  * VRAM attributed **per model**, by joining nvidia-smi's compute processes to
    systemd units to the model path each backend reports. The router breaks that
    join — one process holds several models — so those blocks are marked shared
    rather than split on a guess.
  * Context rolled up **across slots**. `/props` reports per-slot geometry and
    `/slots` reports per-slot occupancy; the sum is what tells you whether a
    backend is nearly full, and until now only the browser computed it.
  * **Coded alerts.** `telemetry.warnings_for` writes prose for a human reading
    a page. A program needs to branch, so these carry stable `code` values and
    the subject they are about.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import platform
import queue
import re
import threading
import time
from dataclasses import dataclass
from typing import Callable

import config_fields
import core
import telemetry

API_VERSION = "1.0"

# Sections a caller can ask for. `logs` is deliberately absent: it is unbounded
# in a way the others are not, and has its own endpoint with its own limits.
SECTIONS = (
    "stack", "gpus", "backends", "services", "host",
    "router", "alerts", "config", "deployment",
)

# --------------------------------------------------------------------------
# thresholds
# --------------------------------------------------------------------------
# These decide what becomes an alert. `telemetry.warnings_for` holds its own
# copies inline for the UI banner; the two are allowed to differ, because a
# banner is tuned for what is worth interrupting someone about and this is tuned
# for what a program should be able to act on. Exposed via `/api/v1/schema` so a
# consumer can see what it is being told.
GPU_BUSY_UTIL_PCT = 5
GPU_VRAM_LOW_MIB = 1024
HOST_MEM_LOW_PCT = 10
CACHE_THRASH_PER_LAUNCH = 0.25
SLOT_DELAY_P90_SECONDS = 1.0
CONTEXT_HIGH_PCT = 85

# Any config key matching this never leaves the process, whatever the allow-list
# below says. The allow-list is the intended control; this is the one that has to
# hold when someone adds a field to it without thinking about who reads it.
SECRET_KEY_RE = re.compile(r"(TOKEN|KEY|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH)", re.IGNORECASE)

# Config sections worth reporting: the settings that decide how a backend runs,
# and therefore the ones that explain why it is behaving as it is.
CONFIG_SECTIONS = (
    "Primary Backend", "Secondary Backend", "Shared Backend", "Task Model",
    "Embedding", "Embedding 2", "Reranker", "OCR", "Model Router", "Ports",
)

# Which env key holds the model each unit was configured to load, in the order
# the launcher falls back through them. `start-chat-backend-dense.sh` tries
# three, so reading only the first would report a mismatch against a backend
# that is loading exactly what it was told to.
CONFIGURED_MODEL_KEYS = {
    "chat-backend-dense": ("CHAT_PRIMARY_MODEL_PATH", "CHAT_DENSE_MODEL_PATH", "CHAT_MODEL_PATH"),
    "chat-backend-moe":   ("CHAT_SECONDARY_MODEL_PATH", "CHAT_MOE_MODEL_PATH", "CHAT_MODEL_PATH"),
    "chat-backend":       ("CHAT_MODEL_PATH",),
    "chat-backend2":      ("CHAT2_MODEL_PATH",),
    "embed":              ("EMBEDDING_MODEL_PATH",),
    "embed2":             ("EMBED2_MODEL_PATH",),
    "rerank":             ("RERANKER_MODEL_PATH",),
    "task":               ("TASK_MODEL_PATH",),
    "ocr":                ("OCR_MODEL_PATH",),
}

# llama.cpp's router reports a per-model state; these mean the weights are in
# memory right now, which is what makes the VRAM on that process theirs.
ROUTER_RESIDENT_STATES = {"loaded", "ready", "active", "resident"}


@dataclass
class Providers:
    """The `app.py` surface this module is allowed to reach.

    Every field is a callable rather than a value so it is resolved per request:
    binding `app.get_gpu_info` once would capture the original function and make
    it unsubstitutable, which is the same trap `ModuleBoundaryTests` guards
    against for imports.
    """

    read_env: Callable[[], dict]
    service_status: Callable[[str], str]
    service_health: Callable[[dict], tuple[dict, dict]]
    gpu_info: Callable[[], list]
    context_summary: Callable[[dict], dict]
    deployment: Callable[[], dict]
    router_overview: Callable[[dict], dict]
    services_table: Callable[[dict], list]


# --------------------------------------------------------------------------
# derived signals
# --------------------------------------------------------------------------

def context_rollup(backend: dict, configured: dict | None = None) -> dict:
    """Context use summed across a backend's slots.

    `/props` gives the geometry and `/slots` gives the occupancy, and neither
    alone answers "how full is this backend". `used_pct` is against the whole
    backend; `max_slot_pct` is the slot closest to overflowing, which is the one
    that will actually reject a request — a backend at 30% overall can still
    refuse the next prompt if it all lands on one slot.
    """
    props = backend.get("props") or {}
    slots = backend.get("slots") or []

    per_slot = props.get("n_ctx_per_slot")
    total = props.get("n_ctx_total")
    if total is None and per_slot and slots:
        total = per_slot * len(slots)

    used = sum(slot.get("n_prompt_tokens") or 0 for slot in slots)
    cached = sum(slot.get("n_prompt_tokens_cache") or 0 for slot in slots)
    percentages = [slot["ctx_pct"] for slot in slots if slot.get("ctx_pct") is not None]

    rollup = {
        "n_ctx_total": total,
        "n_ctx_per_slot": per_slot,
        "slots_total": props.get("total_slots") if props.get("total_slots") is not None else (len(slots) or None),
        "slots_busy": sum(1 for slot in slots if slot.get("is_processing")),
        "used_tokens": used,
        "cached_tokens": cached,
        "free_tokens": max(0, total - used) if total else None,
        "used_pct": round(100 * used / total, 1) if total else None,
        "max_slot_pct": max(percentages) if percentages else None,
    }
    if configured:
        rollup["configured_total"] = configured.get("total_context")
        rollup["configured_per_slot"] = configured.get("per_slot_context")
    return rollup


def configured_model_path(env: dict, unit: str | None) -> str:
    for key in CONFIGURED_MODEL_KEYS.get(unit or "", ()):
        value = str(env.get(key) or "").strip()
        if value:
            return value
    return ""


def backend_drift(backend: dict, env: dict, configured: dict | None) -> list[dict]:
    """Where the running backend disagrees with the saved configuration.

    A backend only reads its config at launch, so an edit that has not been
    followed by a restart leaves the two silently out of step — the panel shows
    the number that was typed and the model serves the number it started with.
    This is the difference, per field.
    """
    props = backend.get("props") or {}
    drift = []

    running_model = props.get("model_path")
    wanted_model = configured_model_path(env, backend.get("unit"))
    if running_model and wanted_model and os.path.basename(running_model) != os.path.basename(wanted_model):
        drift.append({"field": "model_path", "running": running_model, "configured": wanted_model})

    running_ctx = props.get("n_ctx_per_slot")
    wanted_ctx = (configured or {}).get("per_slot_context")
    if running_ctx and wanted_ctx and int(running_ctx) != int(wanted_ctx):
        drift.append({"field": "n_ctx_per_slot", "running": running_ctx, "configured": wanted_ctx})

    return drift


def gpu_model_blocks(gpu: dict, backends: list[dict], router: dict | None) -> list[dict]:
    """VRAM on one GPU, grouped by the model holding it.

    nvidia-smi reports processes, not models. Each process already carries the
    unit it belongs to (from its cgroup) and the `--model` it was launched with,
    which between them name the model without having to ask the backend.

    The router needs both. It runs one `llama-server` child per resident model,
    all in `llama-router.service`, so grouping by unit alone would merge every
    model it holds into one block — but each child names its own `--model`, so
    the split is observable after all. Blocks are keyed on (unit, model) for
    that reason, and `owner` says which unit is holding it.

    A process whose model cannot be determined is reported rather than dropped:
    it is competing for the same VRAM either way.
    """
    by_unit: dict[str, dict] = {}
    for backend in backends:
        unit = backend.get("unit")
        props = backend.get("props") or {}
        if unit and props.get("model_path"):
            by_unit[unit] = {
                "model": props.get("model_path"),
                "model_alias": props.get("model_alias"),
                "backend": backend.get("name"),
            }

    resident = {
        str(entry.get("id")): entry for entry in ((router or {}).get("models") or [])
        if str(entry.get("state") or "").lower() in ROUTER_RESIDENT_STATES
    }

    grouped: dict[tuple, dict] = {}
    for process in gpu.get("processes") or []:
        unit = process.get("name") or "unknown"
        known = by_unit.get(unit, {})
        # The process's own launch arguments first: they are the only thing that
        # distinguishes one router child from another.
        model = process.get("model") or known.get("model") or ""
        alias = process.get("alias") or known.get("model_alias") or ""

        block = grouped.setdefault((unit, model), {
            "unit": unit, "model": model or None, "model_alias": alias or None,
            "backend": known.get("backend"), "vram_mib": 0, "pids": [], "process_names": [],
        })
        block["vram_mib"] += process.get("used_memory") or 0
        block["pids"].append(process.get("pid"))
        name = process.get("process_name")
        if name and name not in block["process_names"]:
            block["process_names"].append(name)

    blocks = []
    for (unit, model), block in grouped.items():
        if not model:
            # Using the GPU, but nothing here can say what it is running.
            block["attribution"] = "unattributed"
        elif unit == telemetry.ROUTER_UNIT:
            block["attribution"] = "router"
            block["router_resident"] = block["model_alias"] in resident
        else:
            block["attribution"] = "exclusive"
        blocks.append(block)

    blocks.sort(key=lambda item: item.get("vram_mib", 0), reverse=True)
    return blocks


def annotate_gpus(gpus: list[dict], backends: list[dict], router: dict | None) -> list[dict]:
    annotated = []
    for gpu in gpus:
        entry = dict(gpu)
        entry["models"] = gpu_model_blocks(gpu, backends, router)
        # "Is this GPU doing work" cannot come from utilisation alone: between
        # tokens it reads 0, so a poll lands in a gap and reports an idle GPU
        # mid-generation. Memory-controller activity fills most of those gaps.
        entry["busy"] = bool(
            (gpu.get("util") or 0) >= GPU_BUSY_UTIL_PCT
            or (gpu.get("mem_util") or 0) >= GPU_BUSY_UTIL_PCT
        )
        entry["mem_free"] = gpu.get("mem_free", max(0, (gpu.get("mem_total") or 0) - (gpu.get("mem_used") or 0)))
        annotated.append(entry)
    return annotated


# --------------------------------------------------------------------------
# alerts
# --------------------------------------------------------------------------

def _alert(level: str, code: str, subject: str, text: str, **detail) -> dict:
    return {"level": level, "code": code, "subject": subject, "text": text, "detail": detail}


def derive_alerts(gpus: list[dict], backends: list[dict], host: dict,
                  services: list[dict], bind_warning: str = "") -> list[dict]:
    """Everything worth acting on, with a stable code per condition.

    Ordered by severity so a consumer showing only the first few shows the worst
    few.
    """
    alerts: list[dict] = []

    if bind_warning:
        alerts.append(_alert("info", "api_unauthenticated", "local-api", bind_warning))

    for gpu in gpus:
        free = gpu.get("mem_free")
        if free is not None and free < GPU_VRAM_LOW_MIB:
            alerts.append(_alert(
                "warn", "gpu_vram_low", f"gpu{gpu.get('index')}",
                f"GPU {gpu.get('index')} has {free} MiB of VRAM free — the next model will not fit.",
                free_mib=free, total_mib=gpu.get("mem_total")))

    swap = (host.get("swap_activity") or {})
    if swap.get("active"):
        alerts.append(_alert(
            "error", "host_swapping", "host",
            f"The host is actively swapping ({host.get('swap_used_mib')} MiB used) — "
            f"generation will stall until it stops.",
            swap_used_mib=host.get("swap_used_mib")))
    elif host.get("swap_used_pct"):
        alerts.append(_alert(
            "warn", "host_swap_used", "host",
            f"Host swap is {host['swap_used_pct']}% used ({host.get('swap_used_mib')} MiB).",
            swap_used_pct=host["swap_used_pct"]))

    available_pct = host.get("mem_available_pct")
    if available_pct is not None and available_pct <= HOST_MEM_LOW_PCT:
        alerts.append(_alert(
            "warn", "host_memory_low", "host",
            f"Only {host.get('mem_available_mib')} MiB of host RAM is available ({available_pct}%).",
            available_pct=available_pct))

    for service in services:
        name = service.get("name")
        state = service.get("state")
        if service.get("restarts"):
            alerts.append(_alert(
                "error", "service_flapping", name,
                f"{service.get('label') or name} has restarted {service['restarts']} time(s) "
                f"while being watched — it is not staying up.",
                restarts=service["restarts"]))
        elif state == "failed":
            alerts.append(_alert("error", "service_failed", name,
                                 f"{service.get('label') or name} has failed.",
                                 reason=service.get("reason")))
        elif state == "degraded":
            alerts.append(_alert("warn", "service_degraded", name,
                                 f"{service.get('label') or name} is up but not answering: "
                                 f"{service.get('reason') or 'probe failed'}.",
                                 reason=service.get("reason")))

    for backend in backends:
        label = backend.get("label") or backend.get("name")
        stats = backend.get("stats") or {}
        cache = stats.get("cache") or {}
        scheduling = stats.get("scheduling") or {}
        context_stats = stats.get("context") or {}
        rollup = backend.get("context") or {}

        for entry in backend.get("drift") or []:
            alerts.append(_alert(
                "warn", "config_runtime_mismatch", backend.get("unit") or backend.get("name"),
                f"{label} is running with {entry['field']}={entry['running']} but is configured "
                f"for {entry['configured']} — the change needs a restart to take effect.",
                **entry))

        if context_stats.get("overflow_count"):
            alerts.append(_alert(
                "error", "ctx_overflow", backend.get("unit") or backend.get("name"),
                f"{label}: {context_stats['overflow_count']} request(s) exceeded the per-slot context.",
                count=context_stats["overflow_count"]))

        used_pct = rollup.get("max_slot_pct")
        if used_pct is not None and used_pct >= CONTEXT_HIGH_PCT:
            alerts.append(_alert(
                "warn", "context_high", backend.get("unit") or backend.get("name"),
                f"{label}: a slot is {used_pct}% through its context.",
                max_slot_pct=used_pct))

        per_launch = cache.get("evictions_per_launch")
        if per_launch is not None and per_launch >= CACHE_THRASH_PER_LAUNCH:
            alerts.append(_alert(
                "warn", "cache_thrash", backend.get("unit") or backend.get("name"),
                f"{label}: {per_launch} prompt-cache evictions per slot launch — "
                f"the cache is too small to hold a working set.",
                evictions_per_launch=per_launch))

        p90 = (scheduling.get("select_to_launch_seconds") or {}).get("p90")
        if p90 is not None and p90 >= SLOT_DELAY_P90_SECONDS:
            alerts.append(_alert(
                "warn", "slot_delay", backend.get("unit") or backend.get("name"),
                f"{label}: p90 slot select-to-launch delay is {p90}s before generation starts.",
                p90_seconds=p90))

        if backend.get("active") and not backend.get("metrics_available"):
            alerts.append(_alert(
                "info", "metrics_disabled", backend.get("unit") or backend.get("name"),
                f"{label}: Prometheus metrics are off, so counter-level detail is unavailable."))

    order = {"error": 0, "warn": 1, "info": 2}
    alerts.sort(key=lambda item: order.get(item["level"], 3))
    return alerts


# --------------------------------------------------------------------------
# config, redacted
# --------------------------------------------------------------------------

def _config_allowlist() -> list[dict]:
    return [
        field for field in config_fields.CONFIG_FIELDS
        if field.get("section") in CONFIG_SECTIONS
        and field.get("type") not in {"template_manager"}
        and not SECRET_KEY_RE.search(field.get("key", ""))
    ]


def redacted_config(env: dict) -> dict:
    """The launch settings that explain a backend's behaviour, and nothing else.

    Two independent gates, because this is the one section where a mistake
    leaks a credential: the field must be in an allow-listed section, *and* its
    key must not look like a secret. `/api/config`'s payload is never reused
    here — it is the whole env file, keys included.
    """
    grouped: dict[str, dict] = {}
    for field in _config_allowlist():
        key = field["key"]
        value = env.get(key)
        if value in (None, ""):
            continue
        grouped.setdefault(field["section"], {})[key] = value
    return grouped


# --------------------------------------------------------------------------
# snapshot assembly
# --------------------------------------------------------------------------

class _Assembly:
    """One request's worth of state, computed once and shared between sections.

    Sections overlap heavily — alerts need the backends, the backends need the
    services to know which unit is active — and each of those costs subprocesses
    or HTTP probes. Memoising per request means asking for everything costs the
    same as asking for the most expensive thing.
    """

    def __init__(self, providers: Providers, window: int, bind_warning: str = ""):
        self.providers = providers
        self.window = window
        self.bind_warning = bind_warning
        self._cache: dict = {}

    def _memo(self, key, build):
        if key not in self._cache:
            self._cache[key] = build()
        return self._cache[key]

    @property
    def env(self) -> dict:
        return self._memo("env", self.providers.read_env)

    @property
    def health(self) -> tuple[dict, dict]:
        return self._memo("health", lambda: self.providers.service_health(self.env))

    @property
    def gpus_raw(self) -> list:
        return self._memo("gpus_raw", self.providers.gpu_info)

    @property
    def router(self) -> dict:
        return self._memo("router", lambda: self.providers.router_overview(self.env))

    @property
    def telemetry_payload(self) -> dict:
        def build():
            return telemetry.collect(
                self.env, self.providers.service_status, self.gpus_raw,
                core.read_meminfo(), window_seconds=self.window,
            )
        return self._memo("telemetry", build)

    @property
    def backends(self) -> list[dict]:
        def build():
            configured = self.providers.context_summary(self.env)
            enriched = []
            for backend in self.telemetry_payload.get("backends") or []:
                entry = dict(backend)
                unit_config = configured.get(backend.get("unit") or "")
                entry["context"] = context_rollup(backend, unit_config)
                entry["drift"] = backend_drift(backend, self.env, unit_config)
                entry["busy"] = any(
                    slot.get("is_processing") for slot in (backend.get("slots") or []))
                enriched.append(entry)
            return enriched
        return self._memo("backends", build)

    @property
    def host(self) -> dict:
        return self.telemetry_payload.get("host") or {}

    @property
    def gpus(self) -> list[dict]:
        return self._memo("gpus", lambda: annotate_gpus(self.gpus_raw, self.backends, self.router))

    @property
    def services(self) -> list[dict]:
        def build():
            _statuses, entries = self.health
            labels = {item["name"]: item for item in self.providers.services_table(self.env)}
            services = []
            for name, entry in entries.items():
                meta = labels.get(name, {})
                services.append({
                    "name": name,
                    "label": meta.get("label") or name,
                    "group": meta.get("group"),
                    "description": meta.get("desc"),
                    "state": entry.get("state"),
                    "unit_state": entry.get("unit"),
                    "expected": entry.get("expected"),
                    "reason": entry.get("reason") or "",
                    "restarts": entry.get("restarts") or 0,
                    "probe": entry.get("probe"),
                    "upstreams": entry.get("upstreams") or [],
                    "checked_at": entry.get("checked_at"),
                })
            services.sort(key=lambda item: item["name"])
            return services
        return self._memo("services", build)

    @property
    def alerts(self) -> list[dict]:
        return self._memo("alerts", lambda: derive_alerts(
            self.gpus, self.backends, self.host, self.services, self.bind_warning))

    @property
    def stack(self) -> dict:
        def build():
            counts = {"error": 0, "warn": 0, "info": 0}
            for alert in self.alerts:
                counts[alert["level"]] = counts.get(alert["level"], 0) + 1
            active = [s for s in self.services if s["state"] == "active"]
            return {
                "hostname": platform.node(),
                "stack_dir": str(core.STACK_DIR),
                "api_version": API_VERSION,
                "busy": any(gpu["busy"] for gpu in self.gpus) or any(b["busy"] for b in self.backends),
                "services_active": len(active),
                "services_total": len(self.services),
                "backends_active": sum(1 for b in self.backends if b.get("active")),
                "alert_counts": counts,
                "router_enabled": telemetry.router_enabled(self.env),
            }
        return self._memo("stack", build)


def resolve_sections(requested: str | None) -> tuple[list[str], list[str]]:
    """(sections to build, names that are not sections).

    An unknown section is reported rather than ignored: a consumer that asks for
    `gpu` when the name is `gpus` should be told, not handed a payload that is
    silently missing the only thing it wanted.
    """
    if not requested:
        return list(SECTIONS), []
    names = [part.strip() for part in requested.split(",") if part.strip()]
    known = [name for name in names if name in SECTIONS]
    unknown = [name for name in names if name not in SECTIONS]
    return known, unknown


def snapshot(providers: Providers, sections: list[str] | None = None,
             window: int = telemetry.DEFAULT_WINDOW_SECONDS,
             bind_warning: str = "") -> dict:
    """The payload `/api/v1/snapshot` serves, and every other section endpoint."""
    wanted = list(SECTIONS) if sections is None else sections
    assembly = _Assembly(providers, telemetry.clamp_window(window), bind_warning)

    builders = {
        "stack": lambda: assembly.stack,
        "gpus": lambda: assembly.gpus,
        "backends": lambda: assembly.backends,
        "services": lambda: assembly.services,
        "host": lambda: assembly.host,
        "router": lambda: assembly.router,
        "alerts": lambda: assembly.alerts,
        "config": lambda: redacted_config(assembly.env),
        "deployment": lambda: providers.deployment(),
    }

    payload = {
        "api_version": API_VERSION,
        "generated_at": time.time(),
        "window_seconds": assembly.window,
    }
    for name in wanted:
        build = builders.get(name)
        if build is not None:
            payload[name] = build()
    return payload


# --------------------------------------------------------------------------
# Prometheus exposition
# --------------------------------------------------------------------------

def _label_value(value) -> str:
    return str(value if value is not None else "").replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def _metric(name: str, value, labels: dict | None = None) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        value = int(value)
    rendered = ",".join(f'{k}="{_label_value(v)}"' for k, v in (labels or {}).items() if v not in (None, ""))
    return f"{name}{{{rendered}}} {value}" if rendered else f"{name} {value}"


MIB = 1024 * 1024

# name -> help text and type, emitted as HELP/TYPE comments so a scrape is
# self-describing in Grafana's metric browser.
_METRIC_META = [
    ("llmstack_gpu_utilization_percent", "gauge", "GPU core utilisation."),
    ("llmstack_gpu_memory_used_bytes", "gauge", "VRAM in use on the GPU."),
    ("llmstack_gpu_memory_total_bytes", "gauge", "VRAM installed on the GPU."),
    ("llmstack_gpu_temperature_celsius", "gauge", "GPU temperature."),
    ("llmstack_gpu_power_watts", "gauge", "GPU power draw."),
    ("llmstack_gpu_busy", "gauge", "1 when the GPU is doing work."),
    ("llmstack_model_vram_bytes", "gauge", "VRAM held by one model, or shared by the router."),
    ("llmstack_service_up", "gauge", "1 when the service is active."),
    ("llmstack_backend_context_used_tokens", "gauge", "Context tokens held across a backend's slots."),
    ("llmstack_backend_context_total_tokens", "gauge", "Context tokens the backend has in total."),
    ("llmstack_backend_context_used_ratio", "gauge", "Context use across the backend, 0..1."),
    ("llmstack_backend_slots_busy", "gauge", "Slots currently processing."),
    ("llmstack_backend_slots_total", "gauge", "Slots the backend was launched with."),
    ("llmstack_slot_context_used_tokens", "gauge", "Context tokens held by one slot."),
    ("llmstack_generation_tokens_per_second", "gauge", "Generation throughput over the window."),
    ("llmstack_host_memory_used_bytes", "gauge", "Host RAM in use."),
    ("llmstack_host_swap_used_bytes", "gauge", "Host swap in use."),
    ("llmstack_alerts", "gauge", "Active alerts by level."),
]


def render_prometheus(payload: dict) -> str:
    """A snapshot as Prometheus exposition text.

    Rendered from the same object the JSON endpoints serve, so the two can never
    disagree about what the stack is doing.
    """
    lines: list[str] = []
    for name, kind, help_text in _METRIC_META:
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} {kind}")

    for gpu in payload.get("gpus") or []:
        labels = {"gpu": gpu.get("index"), "name": gpu.get("name"), "uuid": gpu.get("uuid")}
        lines += [line for line in (
            _metric("llmstack_gpu_utilization_percent", gpu.get("util"), labels),
            _metric("llmstack_gpu_memory_used_bytes", (gpu.get("mem_used") or 0) * MIB, labels),
            _metric("llmstack_gpu_memory_total_bytes", (gpu.get("mem_total") or 0) * MIB, labels),
            _metric("llmstack_gpu_temperature_celsius", gpu.get("temp"), labels),
            _metric("llmstack_gpu_power_watts", gpu.get("power_watts"), labels),
            _metric("llmstack_gpu_busy", gpu.get("busy"), labels),
        ) if line]
        for block in gpu.get("models") or []:
            lines.append(_metric("llmstack_model_vram_bytes", (block.get("vram_mib") or 0) * MIB, {
                "gpu": gpu.get("index"),
                "unit": block.get("unit"),
                "model": os.path.basename(block.get("model") or "") or block.get("unit"),
                "attribution": block.get("attribution"),
            }))

    for service in payload.get("services") or []:
        lines.append(_metric("llmstack_service_up", 1 if service.get("state") == "active" else 0, {
            "name": service.get("name"), "state": service.get("state"),
        }))

    for backend in payload.get("backends") or []:
        labels = {"unit": backend.get("unit") or backend.get("name"), "backend": backend.get("name")}
        context = backend.get("context") or {}
        used, total = context.get("used_tokens"), context.get("n_ctx_total")
        lines += [line for line in (
            _metric("llmstack_backend_context_used_tokens", used, labels),
            _metric("llmstack_backend_context_total_tokens", total, labels),
            _metric("llmstack_backend_context_used_ratio",
                    round(used / total, 5) if used is not None and total else None, labels),
            _metric("llmstack_backend_slots_busy", context.get("slots_busy"), labels),
            _metric("llmstack_backend_slots_total", context.get("slots_total"), labels),
        ) if line]
        for slot in backend.get("slots") or []:
            lines.append(_metric("llmstack_slot_context_used_tokens", slot.get("n_prompt_tokens"),
                                 dict(labels, slot=slot.get("id"))))
        throughput = ((backend.get("stats") or {}).get("throughput") or {})
        for quantile in ("p50", "p90"):
            value = (throughput.get("generation_tps") or {}).get(quantile)
            line = _metric("llmstack_generation_tokens_per_second", value, dict(labels, quantile=quantile))
            if line:
                lines.append(line)

    host = payload.get("host") or {}
    lines += [line for line in (
        _metric("llmstack_host_memory_used_bytes", (host.get("mem_used_mib") or 0) * MIB),
        _metric("llmstack_host_swap_used_bytes", (host.get("swap_used_mib") or 0) * MIB),
    ) if line]

    counts = {"error": 0, "warn": 0, "info": 0}
    for alert in payload.get("alerts") or []:
        counts[alert["level"]] = counts.get(alert["level"], 0) + 1
    for level, count in counts.items():
        lines.append(_metric("llmstack_alerts", count, {"level": level}))

    return "\n".join(line for line in lines if line) + "\n"


# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------

def schema() -> dict:
    """What this API serves, so a consumer can discover it rather than guess."""
    return {
        "api_version": API_VERSION,
        "sections": {
            "stack": "Roll-up: whether anything is busy, how many services are up, alert counts.",
            "gpus": "Per GPU: utilisation, VRAM, temperature, power, and VRAM grouped by model.",
            "backends": "Per llama.cpp backend: model, slots, context rollup, throughput, config drift.",
            "services": "Per systemd service: health state, probe result, upstreams, restart count.",
            "host": "Host RAM and swap, including whether it is actively paging.",
            "router": "Model router: which pooled models exist and which are resident.",
            "alerts": "Conditions worth acting on, with a stable code each.",
            "config": "Launch-relevant backend settings. Secrets are never included.",
            "deployment": "Whether the installed tree is behind the remote.",
        },
        "alert_codes": {
            "api_unauthenticated": "The API is bound to a non-loopback address with no token set.",
            "gpu_vram_low": f"A GPU has under {GPU_VRAM_LOW_MIB} MiB free.",
            "host_swapping": "The host is paging right now.",
            "host_swap_used": "Swap is in use but not actively paging.",
            "host_memory_low": f"Available RAM is at or below {HOST_MEM_LOW_PCT}%.",
            "service_flapping": "A service keeps restarting.",
            "service_failed": "A service is in the failed state.",
            "service_degraded": "A service is running but not answering its probe.",
            "config_runtime_mismatch": "A backend is running with settings that differ from the saved config.",
            "ctx_overflow": "A request exceeded the per-slot context.",
            "context_high": f"A slot is at or above {CONTEXT_HIGH_PCT}% of its context.",
            "cache_thrash": f"At or above {CACHE_THRASH_PER_LAUNCH} prompt-cache evictions per slot launch.",
            "slot_delay": f"p90 slot select-to-launch delay at or above {SLOT_DELAY_P90_SECONDS}s.",
            "metrics_disabled": "A backend was launched without --metrics.",
        },
        "thresholds": {
            "gpu_busy_util_pct": GPU_BUSY_UTIL_PCT,
            "gpu_vram_low_mib": GPU_VRAM_LOW_MIB,
            "host_mem_low_pct": HOST_MEM_LOW_PCT,
            "cache_thrash_per_launch": CACHE_THRASH_PER_LAUNCH,
            "slot_delay_p90_seconds": SLOT_DELAY_P90_SECONDS,
            "context_high_pct": CONTEXT_HIGH_PCT,
        },
        "endpoints": {
            "/api/v1/schema": "This document.",
            "/api/v1/snapshot": "Every section. `?include=` selects, `?window=` sets the stats window.",
            "/api/v1/gpu": "The gpus section alone.",
            "/api/v1/backends": "The backends section alone.",
            "/api/v1/services": "The services section alone.",
            "/api/v1/alerts": "The alerts section alone.",
            "/api/v1/logs": "Parsed backend log events. `?unit=`, `?kind=`, `?level=`, `?since=`, `?limit=`.",
            "/api/v1/logs/raw": "Unparsed journal lines. `?unit=`, `?lines=`.",
            "/api/v1/metrics": "Prometheus exposition of the same snapshot.",
            "/api/v1/events": "SSE stream: snapshot, delta, log, alert. `?include=`, `?interval=`.",
            "/api/v1/health": "Liveness. Never requires a token.",
        },
        "event_types": {
            "snapshot": "A full payload, on the stream interval.",
            "delta": "Service state and alert transitions only.",
            "log": "A parsed backend log event as it happens.",
            "alert": "An alert that has just started.",
        },
        "window_seconds": {
            "default": telemetry.DEFAULT_WINDOW_SECONDS,
            "min": telemetry.MIN_WINDOW_SECONDS,
            "max": telemetry.MAX_WINDOW_SECONDS,
        },
    }


# --------------------------------------------------------------------------
# log access
# --------------------------------------------------------------------------

def log_events(units: list[str], window: int, kinds: set[str] | None = None,
               since: float | None = None, limit: int = 200,
               registry: telemetry.TelemetryRegistry | None = None) -> list[dict]:
    """Parsed journal events, from the tailers telemetry already runs.

    Deliberately not a new `journalctl` invocation: `telemetry.REGISTRY` is
    already tailing every backend unit and parsing every line, so serving this
    from its ring buffer costs a list copy. Spawning a process per API client
    would make a log endpoint the most expensive thing in the manager.
    """
    registry = telemetry.REGISTRY if registry is None else registry
    events: list[dict] = []
    for unit in units:
        collector = registry.collector(unit, window)
        for event in collector.snapshot():
            if kinds and event.get("kind") not in kinds:
                continue
            if since is not None and (event.get("ts") or 0) <= since:
                continue
            events.append(event)
    events.sort(key=lambda item: item.get("ts") or 0)
    return events[-limit:] if limit else events


# --------------------------------------------------------------------------
# webhooks
# --------------------------------------------------------------------------

WEBHOOK_QUEUE_DEPTH = 64
WEBHOOK_TIMEOUT_SECONDS = 5


def sign_webhook(body: bytes, token: str) -> str:
    return "sha256=" + hmac.new(token.encode(), body, hashlib.sha256).hexdigest()


class WebhookSender:
    """Posts state transitions to a configured URL, off the collection thread.

    The collector must never wait on someone else's HTTP server, so delivery is
    a bounded queue and one worker. A full queue drops the oldest event and says
    so: falling behind is worth a log line, and blocking the thread that watches
    the stack is not an acceptable way to avoid it.
    """

    def __init__(self):
        self._queue: queue.Queue = queue.Queue(maxsize=WEBHOOK_QUEUE_DEPTH)
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.delivered = 0
        self.dropped = 0
        self.last_error: str | None = None

    def send(self, url: str, token: str, event: dict) -> None:
        if not url:
            return
        try:
            self._queue.put_nowait((url, token, event))
        except queue.Full:
            self.dropped += 1
            print(f"[llm-api] webhook queue full, dropped {event.get('type')} event", flush=True)
            return
        self._ensure_worker()

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=self._run, name="llm-api-webhook", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        while True:
            try:
                url, token, event = self._queue.get(timeout=30)
            except queue.Empty:
                return
            try:
                self._post(url, token, event)
                self.delivered += 1
            except Exception as exc:
                self.last_error = str(exc)
                print(f"[llm-api] webhook delivery failed: {exc}", flush=True)

    def _post(self, url: str, token: str, event: dict) -> None:
        from urllib import request as urlrequest

        body = json.dumps(event).encode()
        headers = {"Content-Type": "application/json", "User-Agent": "llm-stack-manager"}
        if token:
            headers["X-LLM-Stack-Signature"] = sign_webhook(body, token)
        request = urlrequest.Request(url, data=body, headers=headers, method="POST")
        with urlrequest.urlopen(request, timeout=WEBHOOK_TIMEOUT_SECONDS):
            pass


WEBHOOKS = WebhookSender()


# --------------------------------------------------------------------------
# the live stream
# --------------------------------------------------------------------------

MIN_STREAM_INTERVAL = 1
MAX_STREAM_INTERVAL = 60
DEFAULT_STREAM_INTERVAL = 2
# Enough to ride out a slow reader without letting one client's backlog grow
# without bound. Past this the client is not keeping up and is dropped.
SUBSCRIBER_QUEUE_DEPTH = 128
# Log events are the only unbounded source on the stream; a backend under load
# can emit hundreds a second and no consumer needs all of them live.
MAX_LOG_EVENTS_PER_TICK = 50


def clamp_interval(value, default: int = DEFAULT_STREAM_INTERVAL) -> int:
    try:
        interval = int(value)
    except (TypeError, ValueError):
        return default
    return max(MIN_STREAM_INTERVAL, min(MAX_STREAM_INTERVAL, interval))


class Subscription:
    """One SSE client's view of the stream."""

    def __init__(self, include: list[str] | None, interval: int = DEFAULT_STREAM_INTERVAL):
        self.include = set(include) if include else None
        self.interval = clamp_interval(interval)
        self.queue: queue.Queue = queue.Queue(maxsize=SUBSCRIBER_QUEUE_DEPTH)
        self.dropped = False

    def wants(self, event_type: str) -> bool:
        return self.include is None or event_type in self.include

    def offer(self, event_type: str, data: dict) -> None:
        if not self.wants(event_type):
            return
        try:
            self.queue.put_nowait((event_type, data))
        except queue.Full:
            # Marked rather than raised: the generator serving this client is
            # blocked in `get()` and is the only place that can close it.
            self.dropped = True


class Broadcaster:
    """One collector, many clients.

    The naive shape — a thread per SSE connection, each polling — multiplies
    every subprocess and every backend probe by the number of connected apps,
    which on this box is how a dashboard becomes the reason the GPU is busy.
    So there is exactly one loop: it builds a snapshot on an interval, works out
    what changed, and hands the result to every subscriber.

    It runs only while something needs it — a connected client or a configured
    webhook — and stops on its own once nothing does.
    """

    def __init__(self, providers: Providers, settings: Callable[[], dict]):
        self.providers = providers
        self.settings = settings
        self._subscribers: list[Subscription] = []
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._previous_services: dict[str, str] = {}
        self._previous_alerts: dict[str, dict] = {}
        self._log_marks: dict[str, float] = {}
        self.ticks = 0

    # -- subscription ------------------------------------------------------

    def subscribe(self, include: list[str] | None = None,
                  interval: int = DEFAULT_STREAM_INTERVAL) -> Subscription:
        subscription = Subscription(include, interval)
        with self._lock:
            self._subscribers.append(subscription)
        self.ensure_running()
        return subscription

    def unsubscribe(self, subscription: Subscription) -> None:
        with self._lock:
            if subscription in self._subscribers:
                self._subscribers.remove(subscription)

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)

    def ensure_running(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="llm-api-stream", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    # -- the loop ----------------------------------------------------------

    def _current_interval(self, settings: dict) -> int:
        """The fastest cadence anything currently connected asked for.

        A client wanting one-second updates should get them, but it must not be
        able to make the box collect faster than the slowest thing it is asking
        about can answer — so this is a floor negotiated among subscribers, not
        a per-connection timer.
        """
        configured = clamp_interval(settings.get("interval"))
        with self._lock:
            requested = [s.interval for s in self._subscribers]
        return min([configured] + requested) if requested else configured

    def _run(self) -> None:
        while not self._stop.is_set():
            settings = self.settings()
            if not self._subscribers and not settings.get("webhook_url"):
                return
            try:
                self.tick(settings)
            except Exception as exc:
                print(f"[llm-api] stream tick failed: {exc}", flush=True)
            self._stop.wait(self._current_interval(settings))

    def tick(self, settings: dict | None = None) -> dict:
        """One round: snapshot, diff, fan out. Returns the snapshot it built."""
        settings = self.settings() if settings is None else settings
        payload = snapshot(self.providers, list(SECTIONS),
                           window=settings.get("window") or telemetry.DEFAULT_WINDOW_SECONDS,
                           bind_warning=settings.get("bind_warning", ""))
        self.ticks += 1

        self._publish("snapshot", payload)
        for event in self._service_deltas(payload) + self._alert_deltas(payload):
            self._publish(event["type"], event)
            self._maybe_webhook(settings, event)
        for event in self._log_deltas(payload, settings):
            self._publish("log", event)
        return payload

    def _publish(self, event_type: str, data: dict) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            subscriber.offer(event_type, data)

    # -- diffing -----------------------------------------------------------

    def _service_deltas(self, payload: dict) -> list[dict]:
        current = {s["name"]: s["state"] for s in payload.get("services") or []}
        events = []
        # Skipped on the first tick: everything would read as a transition from
        # nothing, and a webhook consumer would get the whole service table as
        # alarms every time the manager restarts.
        if self._previous_services:
            for name, state in current.items():
                previous = self._previous_services.get(name)
                if previous is not None and previous != state:
                    events.append({
                        "type": "delta", "kind": "service_state", "name": name,
                        "from": previous, "to": state, "at": payload["generated_at"],
                    })
        self._previous_services = current
        return events

    def _alert_deltas(self, payload: dict) -> list[dict]:
        current = {f"{a['code']}:{a['subject']}": a for a in payload.get("alerts") or []}
        events = []
        if self._previous_alerts or self.ticks > 1:
            for key, alert in current.items():
                if key not in self._previous_alerts:
                    events.append({
                        "type": "alert", "kind": "alert_raised", "alert": alert,
                        "at": payload["generated_at"],
                    })
            for key, alert in self._previous_alerts.items():
                if key not in current:
                    events.append({
                        "type": "alert", "kind": "alert_cleared", "alert": alert,
                        "at": payload["generated_at"],
                    })
        self._previous_alerts = current
        return events

    def _log_deltas(self, payload: dict, settings: dict) -> list[dict]:
        units = [b.get("unit") for b in payload.get("backends") or [] if b.get("active") and b.get("unit")]
        window = settings.get("window") or telemetry.DEFAULT_WINDOW_SECONDS
        fresh: list[dict] = []
        for unit in units:
            mark = self._log_marks.get(unit)
            events = log_events([unit], window, since=mark, limit=MAX_LOG_EVENTS_PER_TICK)
            if events:
                self._log_marks[unit] = events[-1].get("ts") or mark
            elif mark is None:
                # First sight of this unit: start from now rather than replaying
                # a whole backfill window into a client that just connected.
                self._log_marks[unit] = time.time()
            fresh.extend(events if mark is not None else [])
        return fresh

    # -- webhooks ----------------------------------------------------------

    def _maybe_webhook(self, settings: dict, event: dict) -> None:
        url = settings.get("webhook_url")
        if not url:
            return
        wanted = settings.get("webhook_events") or set()
        name = "service_state" if event.get("kind") == "service_state" else "alert"
        if name not in wanted:
            return
        WEBHOOKS.send(url, settings.get("token") or "", event)
