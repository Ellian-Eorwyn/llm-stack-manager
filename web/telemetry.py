#!/usr/bin/env python3
"""
Backend telemetry for the LLM Stack Manager.

llama-server exposes plenty of runtime detail, but almost none of it reaches the
manager UI: `/api/status` only reports GPU memory and systemd unit states. This
module collects the rest from three sources and folds them into one snapshot:

  * `/props`   — model, build, per-slot context, slot count.
  * `/slots`   — live per-slot occupancy and cached prompt tokens.
  * `/metrics` — Prometheus counters, when the backend was launched with
                 `--metrics`. Absent metrics degrade to `metrics_available=False`
                 rather than failing the snapshot.

plus the service journal, which is the only place throughput, speculative-decode
acceptance, prompt-cache eviction and slot-scheduling latency are reported at
all. Journal lines are parsed by pure functions (see `parse_line`) so they stay
unit-testable, and a background tailer keeps a bounded window of events per unit.

Note that `/props` reports context *per slot*, which is `--ctx-size` divided by
`--parallel`. That distinction matters: a backend launched with `--ctx-size
262144 --parallel 2` rejects requests above 131072 tokens.
"""

import functools
import json
import re
import subprocess
import threading
import time
from collections import deque
from datetime import datetime
from urllib import error as urlerror
from urllib import request as urlrequest

DEFAULT_WINDOW_SECONDS = 3600
MIN_WINDOW_SECONDS = 60
MAX_WINDOW_SECONDS = 7 * 24 * 3600
# Roughly a week of this box's traffic; a runaway unit cannot grow memory past it.
MAX_EVENTS_PER_UNIT = 20000
PROBE_TIMEOUT_SECONDS = 3

# Backends that speak the llama-server HTTP API, in UI order. `port_key` is the
# env key holding the port; `host_key` falls back to loopback when unset.
BACKEND_TARGETS = [
    {"name": "chat-primary",   "label": "Primary Backend",   "port_key": "CHAT_BACKEND_PORT",  "host_key": "CHAT_BACKEND_HOST",
     "units": ["chat-backend-dense", "chat-backend-moe", "chat-backend"]},
    {"name": "chat-secondary", "label": "Secondary Backend", "port_key": "CHAT_BACKEND2_PORT", "host_key": "CHAT_BACKEND2_HOST",
     "units": ["chat-backend2"]},
    {"name": "embed",          "label": "Embedding",         "port_key": "EMBED_PORT",         "host_key": "EMBED_BACKEND_HOST",
     "units": ["embed"]},
    {"name": "embed2",         "label": "Embedding 2",       "port_key": "EMBED2_PORT",        "host_key": "EMBED2_BACKEND_HOST",
     "units": ["embed2"]},
    {"name": "rerank",         "label": "Reranker",          "port_key": "RERANK_PORT",        "host_key": "RERANK_BACKEND_HOST",
     "units": ["rerank"]},
    {"name": "task",           "label": "Task Model",        "port_key": "TASK_PORT",          "host_key": "TASK_BACKEND_HOST",
     "units": ["task"]},
    {"name": "ocr",            "label": "OCR Model",         "port_key": "OCR_PORT",           "host_key": "OCR_BACKEND_HOST",
     "units": ["ocr"]},
]

DEFAULT_BACKEND_PORTS = {
    "CHAT_BACKEND_PORT": "8010",
    "CHAT_BACKEND2_PORT": "8020",
    "EMBED_PORT": "8005",
    "EMBED2_PORT": "8011",
    "RERANK_PORT": "8006",
    "TASK_PORT": "8007",
    "OCR_PORT": "8009",
}

# Env prefix -> unit, for the models `llama-router` can own. When router mode is
# on these units are deliberately not running: the router holds the models as
# child processes and loads them on demand, so the unit being inactive says
# nothing about whether the model is available.
#
# `scripts/render-models-ini.py` keys the same set by prefix. A member added
# there needs adding here too, or the panel will report it as stopped-on-purpose
# while the router is quietly serving it.
ROUTER_MEMBER_UNITS = {
    "EMBED": "embed",
    "EMBED2": "embed2",
    "RERANK": "rerank",
    "TASK": "task",
    "OCR": "ocr",
}

ROUTER_UNIT = "llama-router"


def router_enabled(env: dict) -> bool:
    return str(env.get("MODEL_ROUTER_ENABLED", "")).strip().lower() == "on"


@functools.lru_cache(maxsize=8)
def _pooled_units(enabled: bool, members: str) -> frozenset:
    if not enabled:
        return frozenset()
    return frozenset(
        unit for unit in
        (ROUTER_MEMBER_UNITS.get(prefix.strip().upper()) for prefix in members.split(","))
        if unit
    )


def pooled_units(env: dict) -> frozenset:
    """Units the router owns right now, or an empty set when it is off.

    Cached on its two inputs because `collect()` asks once per service on every
    five-second poll, and the answer only changes when the config does.
    """
    return _pooled_units(router_enabled(env),
                         str(env.get("MODEL_ROUTER_MEMBERS") or "EMBED,OCR,RERANK,TASK"))


# --------------------------------------------------------------------------
# journal line parsing
# --------------------------------------------------------------------------

# `journalctl -o short-iso-precise` prefix: timestamp, host, unit[pid]: body
_JOURNAL_LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}T[\d:.]+[+-]\d{2}:\d{2})\s+\S+\s+(?P<unit>[^\[\s]+)\[\d+\]:\s?(?P<body>.*)$"
)
# llama.cpp's own `--log-prefix` stamp and level, e.g. "5960.17.293.563 I ".
_LLAMA_PREFIX_RE = re.compile(r"^[\d.]+\s+[IWED]\s+")

_SLOT_SELECT_RE = re.compile(
    r"slot get_availabl:\s+id\s+(?P<slot>\d+)\s+\|.*selected slot by (?P<method>id|LRU|LCP similarity)"
)
_SLOT_LAUNCH_RE = re.compile(r"slot launch_slot_:\s+id\s+(?P<slot>\d+)\s+\|\s+task\s+(?P<task>\d+)")
_SLOT_RELEASE_RE = re.compile(
    r"slot\s+release:\s+id\s+(?P<slot>\d+)\s+\|\s+task\s+(?P<task>\d+)\s+\|\s+stop processing:\s+n_tokens\s*=\s*(?P<n_tokens>\d+)"
)
_TG_RE = re.compile(r"n_decoded\s*=\s*(?P<n>\d+),\s*tg\s*=\s*(?P<tg>[\d.]+)\s*t/s")
_EVAL_TIME_RE = re.compile(
    r"\|\s*(?P<label>prompt eval|eval|total)\s+time\s*=\s*(?P<ms>[\d.]+)\s*ms\s*/\s*(?P<tokens>\d+)\s+tokens"
    r"(?:\s*\(\s*[\d.]+\s*ms per token,\s*(?P<tps>[\d.]+)\s*tokens per second\))?"
)
_DRAFT_RE = re.compile(
    r"draft acceptance\s*=\s*(?P<rate>[\d.]+)\s*\(\s*(?P<accepted>\d+)\s+accepted\s*/\s*(?P<generated>\d+)\s+generated\),"
    r"\s*mean len\s*=\s*(?P<mean_len>[\d.]+)"
)
_EVICT_RE = re.compile(
    r"making room for prompt cache entry, removing oldest entry \(size\s*=\s*(?P<mib>[\d.]+)\s*MiB\)"
)
_CHECKPOINT_RE = re.compile(
    r"erasing old context checkpoint \(.*?n_tokens\s*=\s*(?P<n_tokens>\d+),\s*size\s*=\s*(?P<mib>[\d.]+)\s*MiB\)"
)
_CTX_OVERFLOW_RE = re.compile(
    r"request \((?P<requested>\d+) tokens\) exceeds the available context size \((?P<available>\d+) tokens\)"
)

# llama.cpp reports exactly 1000000.00 tokens/s when elapsed time rounds to
# 0.00 ms (a one-token eval off a warm cache). It is a divide-by-zero sentinel,
# not a measurement, and it wrecks any mean it lands in.
TPS_SENTINEL = 1000000.0


def parse_timestamp(value: str) -> float | None:
    """Epoch seconds from a journal ISO timestamp, or None if unparseable."""
    try:
        return datetime.fromisoformat(value).timestamp()
    except (ValueError, TypeError):
        return None


def split_journal_line(line: str) -> tuple[float | None, str, str] | None:
    """Split a `short-iso-precise` line into (epoch, unit, message body)."""
    match = _JOURNAL_LINE_RE.match(line.rstrip("\n"))
    if not match:
        return None
    body = _LLAMA_PREFIX_RE.sub("", match.group("body"))
    return parse_timestamp(match.group("ts")), match.group("unit"), body


def parse_event(body: str) -> dict | None:
    """Classify one llama-server log message. Returns None for lines we ignore."""
    match = _SLOT_SELECT_RE.search(body)
    if match:
        method = match.group("method")
        return {
            "kind": "slot_select",
            "slot": int(match.group("slot")),
            # "LCP similarity" reads better as "lcp" in aggregate breakdowns.
            "method": "lcp" if method.startswith("LCP") else method.lower(),
        }

    match = _SLOT_LAUNCH_RE.search(body)
    if match:
        return {"kind": "slot_launch", "slot": int(match.group("slot")), "task": int(match.group("task"))}

    match = _SLOT_RELEASE_RE.search(body)
    if match:
        return {
            "kind": "slot_release",
            "slot": int(match.group("slot")),
            "task": int(match.group("task")),
            "n_tokens": int(match.group("n_tokens")),
        }

    match = _TG_RE.search(body)
    if match:
        return {"kind": "generation", "n_decoded": int(match.group("n")), "tg_tps": float(match.group("tg"))}

    match = _EVAL_TIME_RE.search(body)
    if match:
        label = match.group("label")
        tps = float(match.group("tps")) if match.group("tps") else None
        if tps is not None and tps >= TPS_SENTINEL:
            tps = None
        return {
            "kind": "prompt_eval" if label == "prompt eval" else ("eval" if label == "eval" else "total_time"),
            "ms": float(match.group("ms")),
            "tokens": int(match.group("tokens")),
            "tps": tps,
        }

    match = _DRAFT_RE.search(body)
    if match:
        return {
            "kind": "draft",
            "rate": float(match.group("rate")),
            "accepted": int(match.group("accepted")),
            "generated": int(match.group("generated")),
            "mean_len": float(match.group("mean_len")),
        }

    match = _EVICT_RE.search(body)
    if match:
        return {"kind": "cache_evict", "mib": float(match.group("mib"))}

    match = _CHECKPOINT_RE.search(body)
    if match:
        return {
            "kind": "checkpoint_erase",
            "n_tokens": int(match.group("n_tokens")),
            "mib": float(match.group("mib")),
        }

    match = _CTX_OVERFLOW_RE.search(body)
    if match:
        return {
            "kind": "context_overflow",
            "requested": int(match.group("requested")),
            "available": int(match.group("available")),
            "message": body.strip(),
        }

    return None


def parse_line(line: str) -> dict | None:
    """Parse a full journal line into a timestamped event, or None."""
    split = split_journal_line(line)
    if not split:
        return None
    ts, unit, body = split
    event = parse_event(body)
    if event is None:
        return None
    event["ts"] = ts
    event["unit"] = unit
    return event


# --------------------------------------------------------------------------
# Prometheus text format
# --------------------------------------------------------------------------

def parse_prometheus(text: str) -> dict[str, float]:
    """Flatten Prometheus exposition text to {metric_name: value}.

    llama-server emits unlabelled gauges and counters, so labels are folded into
    the key as `name{labels}` on the rare occasion they appear.
    """
    metrics: dict[str, float] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name, _, remainder = line.partition(" ")
        if not remainder:
            continue
        try:
            metrics[name] = float(remainder.split()[0])
        except (ValueError, IndexError):
            continue
    return metrics


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile over an unsorted list. `q` is 0..1."""
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return round(ordered[index], 3)


def _distribution(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "p50": None, "p90": None, "p99": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p99": percentile(values, 0.99),
        "max": round(max(values), 3),
        "mean": round(sum(values) / len(values), 3),
    }


def summarize(events: list[dict], window_seconds: int, now: float | None = None) -> dict:
    """Roll a list of parsed events into the stats the UI and tuning work need.

    Events are expected newest-last. Anything older than `window_seconds` is
    dropped, so the same function serves both the live panel and the wide
    backfill window used for before/after comparisons.
    """
    now = time.time() if now is None else now
    cutoff = now - window_seconds
    scoped = [e for e in events if e.get("ts") is None or e["ts"] >= cutoff]

    # Two different measures of the same thing, kept apart on purpose: `eval
    # time` is a whole-request average, while `tg =` lines are mid-generation
    # samples. Pooling them would double-count long requests.
    tg_values: list[float] = []
    live_values: list[float] = []
    pp_values: list[float] = []
    draft_rates: list[float] = []
    draft_mean_lens: list[float] = []
    evict_sizes: list[float] = []
    checkpoint_sizes: list[float] = []
    release_tokens: list[float] = []
    delays: list[float] = []
    select_methods: dict[str, int] = {}
    select_by_id_slots: dict[str, int] = {}
    overflows: list[dict] = []
    launches = 0
    last_generation_ts = None
    last_tg = None
    last_draft = None

    # A slot is selected, then launched; the gap is time spent making room in the
    # prompt cache. Only the most recent selection per slot can be pending.
    pending_select: dict[int, float] = {}

    for event in scoped:
        kind = event["kind"]
        ts = event.get("ts")

        if kind == "slot_select":
            select_methods[event["method"]] = select_methods.get(event["method"], 0) + 1
            if event["method"] == "id":
                # Which slot, not just how many: the pi-forge contract is that
                # interactive work lands on slot 0 and background work on slot 1,
                # and a total cannot tell those two apart.
                slot = str(event["slot"])
                select_by_id_slots[slot] = select_by_id_slots.get(slot, 0) + 1
            if ts is not None:
                pending_select[event["slot"]] = ts
        elif kind == "slot_launch":
            launches += 1
            started = pending_select.pop(event["slot"], None)
            if started is not None and ts is not None:
                delays.append(max(0.0, ts - started))
        elif kind == "slot_release":
            release_tokens.append(float(event["n_tokens"]))
        elif kind == "generation":
            live_values.append(event["tg_tps"])
            last_tg = event["tg_tps"]
            last_generation_ts = ts
        elif kind == "prompt_eval":
            if event.get("tps"):
                pp_values.append(event["tps"])
        elif kind == "eval":
            if event.get("tps"):
                tg_values.append(event["tps"])
                last_tg = event["tps"]
                last_generation_ts = ts
        elif kind == "draft":
            draft_rates.append(event["rate"])
            draft_mean_lens.append(event["mean_len"])
            last_draft = event["rate"]
        elif kind == "cache_evict":
            evict_sizes.append(event["mib"])
        elif kind == "checkpoint_erase":
            checkpoint_sizes.append(event["mib"])
        elif kind == "context_overflow":
            overflows.append({"ts": ts, "requested": event["requested"], "available": event["available"]})

    over_1s = sum(1 for d in delays if d > 1.0)
    over_2s = sum(1 for d in delays if d > 2.0)

    return {
        "window_seconds": window_seconds,
        "events": len(scoped),
        "throughput": {
            "generation_tps": _distribution(tg_values),
            "live_tps": _distribution(live_values),
            "prompt_tps": _distribution(pp_values),
            "last_generation_tps": last_tg,
            "last_generation_ts": last_generation_ts,
            "draft_acceptance": _distribution(draft_rates),
            "draft_mean_len": _distribution(draft_mean_lens),
            "last_draft_acceptance": last_draft,
        },
        "cache": {
            "launches": launches,
            "evictions": len(evict_sizes),
            "evictions_per_launch": round(len(evict_sizes) / launches, 3) if launches else None,
            "evicted_mib": _distribution(evict_sizes),
            "evicted_mib_total": round(sum(evict_sizes), 1),
            "checkpoint_erasures": len(checkpoint_sizes),
            "checkpoint_mib": _distribution(checkpoint_sizes),
        },
        "scheduling": {
            "select_to_launch_seconds": _distribution(delays),
            "over_1s": over_1s,
            "over_1s_pct": round(100 * over_1s / len(delays), 1) if delays else None,
            "over_2s": over_2s,
            "over_2s_pct": round(100 * over_2s / len(delays), 1) if delays else None,
            # `by_id` proves the pi-forge id_slot contract is reaching the backend.
            "select_methods": select_methods,
            "select_by_id_slots": select_by_id_slots,
        },
        "context": {
            "released_tokens": _distribution(release_tokens),
            "overflow_count": len(overflows),
            "overflows": overflows[-5:],
        },
    }


# --------------------------------------------------------------------------
# journal collection
# --------------------------------------------------------------------------

class UnitCollector:
    """Bounded event window for one systemd unit, seeded then kept live.

    Backfill matters as much as tailing: without it the first useful stats would
    be an hour away, which defeats the point of measuring a config change.
    """

    def __init__(self, unit: str, max_events: int = MAX_EVENTS_PER_UNIT):
        self.unit = unit
        self.events: deque[dict] = deque(maxlen=max_events)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen | None = None
        self._stop = threading.Event()
        self.backfilled_window = 0
        self.error: str | None = None

    def snapshot(self) -> list[dict]:
        with self._lock:
            return list(self.events)

    def _add(self, event: dict):
        with self._lock:
            self.events.append(event)

    def backfill(self, window_seconds: int):
        """Seed history from the journal. Widening the window re-seeds."""
        with self._lock:
            if window_seconds <= self.backfilled_window:
                return
            self.backfilled_window = window_seconds
        try:
            result = subprocess.run(
                ["journalctl", "-u", self.unit, "--since", f"-{window_seconds}s",
                 "--no-pager", "--output=short-iso-precise"],
                capture_output=True, text=True, timeout=60,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            self.error = f"backfill failed: {exc}"
            return

        seeded = [event for event in (parse_line(line) for line in result.stdout.splitlines()) if event]
        with self._lock:
            # Seeded history is strictly older than anything the tailer has
            # collected, so rebuild the deque in timestamp order.
            live = list(self.events)
            merged = seeded + [e for e in live if not seeded or (e.get("ts") or 0) > (seeded[-1].get("ts") or 0)]
            self.events = deque(merged[-self.events.maxlen:], maxlen=self.events.maxlen)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._tail, name=f"telemetry-{self.unit}", daemon=True)
        self._thread.start()

    def _tail(self):
        while not self._stop.is_set():
            try:
                self._process = subprocess.Popen(
                    ["journalctl", "-u", self.unit, "-f", "-n", "0",
                     "--no-pager", "--output=short-iso-precise"],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                )
            except OSError as exc:
                self.error = f"tail failed: {exc}"
                return
            try:
                for line in iter(self._process.stdout.readline, ""):
                    if self._stop.is_set():
                        break
                    event = parse_line(line)
                    if event:
                        self._add(event)
            finally:
                self._terminate_process()
            if self._stop.is_set():
                return
            # journalctl exited on its own (unit restart, log rotation). Back off
            # briefly rather than spinning on a persistent failure.
            self._stop.wait(2.0)

    def _terminate_process(self):
        process, self._process = self._process, None
        if process is None:
            return
        try:
            process.terminate()
            process.wait(timeout=5)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

    def stop(self):
        self._stop.set()
        self._terminate_process()


class TelemetryRegistry:
    """Owns one collector per unit, created on demand and reused across polls."""

    def __init__(self):
        self._collectors: dict[str, UnitCollector] = {}
        self._lock = threading.Lock()

    def collector(self, unit: str, window_seconds: int) -> UnitCollector:
        with self._lock:
            collector = self._collectors.get(unit)
            if collector is None:
                collector = UnitCollector(unit)
                self._collectors[unit] = collector
                collector.start()
        collector.backfill(window_seconds)
        return collector

    def stop_all(self):
        with self._lock:
            collectors = list(self._collectors.values())
            self._collectors.clear()
        for collector in collectors:
            collector.stop()


REGISTRY = TelemetryRegistry()


# --------------------------------------------------------------------------
# HTTP probes
# --------------------------------------------------------------------------

def _http_text(url: str, timeout: int = PROBE_TIMEOUT_SECONDS) -> tuple[str | None, int | None]:
    """Fetch a URL, returning (body, status). Both are None on transport failure."""
    try:
        with urlrequest.urlopen(urlrequest.Request(url), timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace"), response.status
    except urlerror.HTTPError as exc:
        return None, exc.code
    except Exception:
        return None, None


def _http_json(url: str, timeout: int = PROBE_TIMEOUT_SECONDS):
    body, _ = _http_text(url, timeout)
    if body is None:
        return None
    try:
        return json.loads(body)
    except ValueError:
        return None


def probe_props(base_url: str) -> dict | None:
    """Model identity and slot geometry. `n_ctx` here is per slot, not total."""
    data = _http_json(f"{base_url}/props")
    if not isinstance(data, dict):
        return None
    slots = data.get("total_slots")
    per_slot_ctx = (data.get("default_generation_settings") or {}).get("n_ctx")
    total_ctx = per_slot_ctx * slots if isinstance(per_slot_ctx, int) and isinstance(slots, int) else None
    return {
        "model_path": data.get("model_path"),
        "model_alias": data.get("model_alias"),
        "model_ftype": data.get("model_ftype"),
        "build_info": data.get("build_info"),
        "total_slots": slots,
        "n_ctx_per_slot": per_slot_ctx,
        "n_ctx_total": total_ctx,
        "modalities": data.get("modalities"),
        "is_sleeping": data.get("is_sleeping"),
    }


def probe_slots(base_url: str) -> list[dict] | None:
    """Per-slot occupancy, with context use expressed against the slot's own ctx."""
    data = _http_json(f"{base_url}/slots")
    if not isinstance(data, list):
        return None
    slots = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        n_ctx = entry.get("n_ctx") or 0
        used = entry.get("n_prompt_tokens") or 0
        slots.append({
            "id": entry.get("id"),
            "n_ctx": n_ctx,
            "is_processing": bool(entry.get("is_processing")),
            "n_prompt_tokens": used,
            "n_prompt_tokens_cache": entry.get("n_prompt_tokens_cache"),
            "speculative": bool(entry.get("speculative")),
            "ctx_pct": round(100 * used / n_ctx, 1) if n_ctx else None,
        })
    return slots


def probe_metrics(base_url: str) -> tuple[dict | None, bool]:
    """Prometheus metrics, plus whether the backend has `--metrics` enabled.

    A backend started without the flag answers 501 here; that is a config state,
    not an error, so the caller renders a hint instead of a failure.
    """
    body, status = _http_text(f"{base_url}/metrics")
    if body is None:
        return None, False
    if status is not None and status >= 400:
        return None, False
    metrics = parse_prometheus(body)
    return (metrics, True) if metrics else (None, False)


# --------------------------------------------------------------------------
# snapshot assembly
# --------------------------------------------------------------------------

def clamp_window(value, default: int = DEFAULT_WINDOW_SECONDS) -> int:
    try:
        window = int(value)
    except (TypeError, ValueError):
        return default
    return max(MIN_WINDOW_SECONDS, min(MAX_WINDOW_SECONDS, window))


def resolve_targets(env: dict, service_status) -> list[dict]:
    """Backends worth probing, with the systemd unit currently serving each one.

    A slot like `chat-primary` can be served by any of several units (dense, moe,
    custom); whichever is active owns the journal we tail.
    """
    targets = []
    for spec in BACKEND_TARGETS:
        port = (env.get(spec["port_key"]) or DEFAULT_BACKEND_PORTS.get(spec["port_key"]) or "").strip()
        if not port:
            continue
        host = (env.get(spec["host_key"]) or "127.0.0.1").strip() or "127.0.0.1"
        # 0.0.0.0 is a bind address, not a reachable one.
        if host in {"0.0.0.0", "::"}:
            host = "127.0.0.1"
        active_unit = next((unit for unit in spec["units"] if service_status(unit) == "active"), None)
        targets.append({
            "name": spec["name"],
            "label": spec["label"],
            "units": spec["units"],
            "unit": active_unit,
            "host": host,
            "port": port,
            "base_url": f"http://{host}:{port}",
            "active": active_unit is not None,
        })
    return targets


def backend_snapshot(target: dict, window_seconds: int, registry: TelemetryRegistry | None = None) -> dict:
    """One backend's full telemetry. Every section degrades to None on its own."""
    registry = REGISTRY if registry is None else registry
    snapshot = {
        "name": target["name"],
        "label": target["label"],
        "unit": target["unit"],
        "base_url": target["base_url"],
        "active": target["active"],
        "props": None,
        "slots": None,
        "metrics": None,
        "metrics_available": False,
        "stats": None,
        "collector_error": None,
    }
    if not target["active"]:
        return snapshot

    snapshot["props"] = probe_props(target["base_url"])
    snapshot["slots"] = probe_slots(target["base_url"])
    snapshot["metrics"], snapshot["metrics_available"] = probe_metrics(target["base_url"])

    if target["unit"]:
        collector = registry.collector(target["unit"], window_seconds)
        snapshot["stats"] = summarize(collector.snapshot(), window_seconds)
        snapshot["collector_error"] = collector.error
    return snapshot


# Sustained paging above this rate is real pressure. Below it, a few pages
# drifting in as something touches a cold page is not.
SWAP_ACTIVE_PAGES_PER_SECOND = 256
_PAGE_MIB = 4 / 1024


class SwapMonitor:
    """Turns /proc/vmstat's cumulative page counters into a current rate.

    Swap *usage* is a poor pressure signal: pages written during one bad
    configuration stay resident long after it is fixed, because nothing brings
    them back until something reads them. This box sat at 83% swap with 19 GB
    of RAM free and no paging at all. Rate is the signal; usage is history.
    """

    def __init__(self):
        self._last: tuple[float, int, int] | None = None
        self._lock = threading.Lock()

    @staticmethod
    def _read_counters() -> tuple[int, int] | None:
        try:
            with open("/proc/vmstat", "r", encoding="utf-8") as handle:
                counters = {}
                for line in handle:
                    key, _, value = line.partition(" ")
                    if key in ("pswpin", "pswpout"):
                        counters[key] = int(value.strip())
                if len(counters) == 2:
                    return counters["pswpin"], counters["pswpout"]
        except (OSError, ValueError):
            pass
        return None

    def sample(self, now: float | None = None) -> dict:
        """Paging rate since the previous call. The first call has no rate."""
        now = time.time() if now is None else now
        counters = self._read_counters()
        if counters is None:
            return {"available": False, "in_pages_per_second": None,
                    "out_pages_per_second": None, "active": None}

        pages_in, pages_out = counters
        with self._lock:
            previous, self._last = self._last, (now, pages_in, pages_out)

        if previous is None or now <= previous[0]:
            # No baseline yet: report the counters without claiming a rate.
            return {"available": True, "in_pages_per_second": None,
                    "out_pages_per_second": None, "active": None}

        elapsed = now - previous[0]
        rate_in = max(0, pages_in - previous[1]) / elapsed
        rate_out = max(0, pages_out - previous[2]) / elapsed
        return {
            "available": True,
            "in_pages_per_second": round(rate_in, 1),
            "out_pages_per_second": round(rate_out, 1),
            "in_mib_per_second": round(rate_in * _PAGE_MIB, 2),
            "out_mib_per_second": round(rate_out * _PAGE_MIB, 2),
            # Pressure is pages going *out*: that is the kernel choosing to
            # evict rather than allocate. Pages coming back in with nothing
            # going out is recovery — a model reload touching what an earlier
            # configuration had swapped — and a backend restart alone can push
            # that rate near the threshold. Sustained paging in still counts
            # when it comes with any paging out at all, which is thrashing.
            "active": rate_out >= SWAP_ACTIVE_PAGES_PER_SECOND
                      or (rate_in >= SWAP_ACTIVE_PAGES_PER_SECOND and rate_out > 0),
        }


SWAP_MONITOR = SwapMonitor()


def host_memory(meminfo: dict[str, int], swap_activity: dict | None = None) -> dict:
    """Host RAM and swap in MiB, from an already-parsed /proc/meminfo.

    Swap is the reason this exists: the manager has never shown it, and heavy
    prompt-cache RAM budgets push this box deep into it. `swap_activity` adds
    whether it is *currently* paging, which is the part that distinguishes a
    problem from its residue.
    """
    def mib(key: str) -> int:
        return round(meminfo.get(key, 0) / 1024)

    total = mib("MemTotal")
    available = mib("MemAvailable")
    swap_total = mib("SwapTotal")
    swap_free = mib("SwapFree")
    swap_used = max(0, swap_total - swap_free)
    return {
        "mem_total_mib": total,
        "mem_available_mib": available,
        "mem_used_mib": max(0, total - available),
        "mem_used_pct": round(100 * (total - available) / total) if total else None,
        "mem_available_pct": round(100 * available / total) if total else None,
        "swap_total_mib": swap_total,
        "swap_used_mib": swap_used,
        "swap_used_pct": round(100 * swap_used / swap_total) if swap_total else None,
        "swap_activity": swap_activity or {"available": False, "active": None},
    }


def warnings_for(backends: list[dict], host: dict, gpus: list[dict]) -> list[dict]:
    """Conditions worth surfacing above the fold, in severity order."""
    alerts: list[dict] = []

    for gpu in gpus or []:
        free = (gpu.get("mem_total") or 0) - (gpu.get("mem_used") or 0)
        if gpu.get("mem_total") and free < 1024:
            alerts.append({
                "level": "warn",
                "text": f"GPU {gpu.get('index')} has only {free} MiB free — a backend restart may fail to allocate.",
            })

    # Swap: rate first, usage second. A box can sit at 80% swap with plenty of
    # free RAM and no paging, which is residue from an earlier configuration
    # rather than a live problem — warning about it teaches operators to ignore
    # the warning, which is worse than not having one.
    if host.get("swap_used_pct") is not None and host["swap_used_pct"] >= 50:
        activity = host.get("swap_activity") or {}
        if activity.get("active"):
            alerts.append({
                "level": "warn",
                "text": f"Host is actively swapping: {activity.get('out_mib_per_second', 0)} MiB/s out, "
                        f"{activity.get('in_mib_per_second', 0)} MiB/s in, with "
                        f"{host['swap_used_pct']}% of swap used ({host['swap_used_mib']} MiB). "
                        f"Reduce prompt-cache RAM or context checkpoints.",
            })
        elif activity.get("active") is False:
            alerts.append({
                "level": "info",
                "text": f"Swap is {host['swap_used_pct']}% used ({host['swap_used_mib']} MiB) but idle — "
                        f"pages left behind by an earlier configuration, not current pressure.",
            })
        else:
            alerts.append({
                "level": "warn",
                "text": f"Host swap is {host['swap_used_pct']}% used ({host['swap_used_mib']} MiB) — "
                        f"check prompt-cache RAM and context checkpoint budgets.",
            })

    if host.get("mem_available_pct") is not None and host["mem_available_pct"] <= 10:
        alerts.append({
            "level": "warn",
            "text": f"Only {host['mem_available_mib']} MiB of host RAM is available "
                    f"({host['mem_available_pct']}%) — the next allocation is likely to swap.",
        })

    for backend in backends:
        stats = backend.get("stats") or {}
        cache = stats.get("cache") or {}
        scheduling = stats.get("scheduling") or {}
        context = stats.get("context") or {}
        label = backend.get("label") or backend.get("name")

        per_launch = cache.get("evictions_per_launch")
        if per_launch is not None and per_launch >= 0.25:
            alerts.append({
                "level": "warn",
                "text": f"{label}: {per_launch} prompt-cache evictions per slot launch — "
                        f"the cache is too small to hold a working set.",
            })

        p90 = (scheduling.get("select_to_launch_seconds") or {}).get("p90")
        if p90 is not None and p90 >= 1.0:
            alerts.append({
                "level": "warn",
                "text": f"{label}: p90 slot select-to-launch delay is {p90}s before generation starts.",
            })

        if context.get("overflow_count"):
            available = (context.get("overflows") or [{}])[-1].get("available")
            alerts.append({
                "level": "warn",
                "text": f"{label}: {context['overflow_count']} request(s) exceeded the per-slot context"
                        + (f" of {available:,} tokens." if available else "."),
            })

        if backend.get("active") and not backend.get("metrics_available"):
            alerts.append({
                "level": "info",
                "text": f"{label}: Prometheus metrics are off. Enable the backend's Metrics Endpoint "
                        f"setting and restart for counter-level detail.",
            })

    return alerts


def collect(env: dict, service_status, gpus: list[dict], meminfo: dict[str, int],
            window_seconds: int = DEFAULT_WINDOW_SECONDS,
            registry: TelemetryRegistry | None = None) -> dict:
    """Assemble the full telemetry payload served by /api/backend/telemetry."""
    window = clamp_window(window_seconds)
    targets = resolve_targets(env, service_status)
    backends = [backend_snapshot(target, window, registry) for target in targets]
    host = host_memory(meminfo, SWAP_MONITOR.sample())
    return {
        "generated_at": time.time(),
        "window_seconds": window,
        "backends": backends,
        "gpus": gpus,
        "host": host,
        "warnings": warnings_for(backends, host, gpus),
    }
