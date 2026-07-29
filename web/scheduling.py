#!/usr/bin/env python3
"""
The manager<->pi-forge slot-scheduling contract, made checkable.

pi-forge pins interactive turns to `id_slot: 0` and background work to
`id_slot: 1`, so a long-running agent session keeps its prompt prefix in one
slot while background skills churn the other. Three things have to hold for that
to work, and until now none of them could be checked without reading journals by
hand:

  1. The backend has to be *launched* with at least two slots, idle-slot caching
     on and auto-fit off. What the config form holds is not necessarily what the
     running process was launched with, so both are read and compared.
  2. `id_slot` has to survive the proxy. It does, but only implicitly: the proxy
     round-trips unknown JSON keys, so nothing in the code says so and nothing
     asserted it. `tests/test_llm_chat_proxy.py` asserts it now.
  3. Both sides have to agree on the lease protocol in
     `~/.pi-forge/agent/inference-leases`, which is how a background worker
     learns to yield to an interactive turn.

Verification is passive by default. The journal already records which slot was
chosen and how (`selected slot by id (0)`), telemetry already parses it, so the
contract can be shown live without sending a single request. That matters: a
probe pinned to slot 0 would displace whatever prefix that slot is holding,
which is exactly the eviction this stack spent its effort removing. The active
probe exists, but it is opt-in and says so.

Lease files are read here and, on request, reaped. They belong to pi-forge, so
the rules are deliberately conservative: a lease is only ever removed when it is
far past the staleness window *and* the process that wrote it is gone.
"""

import json
import pwd
import re
import time
from pathlib import Path
from urllib import error as urlerror
from urllib import request as urlrequest

STACK_DIR = Path(__file__).resolve().parent.parent

# What pi-forge needs from the backend regardless of model or host. Mirrors
# CONTRACT in web/static/cache-aware-scheduling.js; `tests/test_scheduling.py`
# parses that file and asserts the two still agree.
CONTRACT = {
    "N_PARALLEL": "2",
    "CACHE_IDLE_SLOTS": "on",
    "FIT": "off",
    "FIT_CTX": "",
}
MINIMUM_PER_SLOT_CONTEXT = 32768

INTERACTIVE_SLOT = 0
BACKGROUND_SLOT = 1

# Must match forge_llm.LEASE_STALE_MS and forge-llm.mjs LEASE_STALE_MS. A Python
# worker and a JavaScript one read each other's leases; a disagreement here
# means one ignores the other and both generate at once. The manager only reads,
# so a mismatch would misreport rather than break scheduling — but it would
# misreport the one number the whole protocol turns on.
LEASE_STALE_MS = 15_000

# Reaping threshold, far beyond the staleness window on purpose. A lease that is
# merely stale may belong to a live process between refreshes; only one whose
# writer is gone is safe to remove.
LEASE_ORPHAN_SECONDS = 600

DEFAULT_PREFIX = "CHAT_PRIMARY"


# ---------------------------------------------------------------------------
# the configured side
# ---------------------------------------------------------------------------

def _integer(value) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return 0


def contract_check(env: dict, prefix: str = DEFAULT_PREFIX) -> dict:
    """Judge a backend's configured settings against the contract.

    Mirrors `evaluate()` in cache-aware-scheduling.js, minus the host
    measurement — the recommended profile is `/api/backend/budget/recommend`'s
    job, and this answers only "can pi-forge schedule against this at all".
    """
    values = {suffix: str(env.get(f"{prefix}_{suffix}", "") or "").strip()
              for suffix in ("N_PARALLEL", "CTX_SIZE", "CACHE_RAM", "CTX_CHECKPOINTS",
                             "CACHE_IDLE_SLOTS", "FIT", "FIT_CTX")}
    slots = _integer(values["N_PARALLEL"])
    total_context = _integer(values["CTX_SIZE"])
    # llama.cpp's own default is one slot, so an unset value divides by one
    # rather than reporting zero context per slot on top of the real complaint.
    per_slot = total_context // (slots if slots > 0 else 1)
    issues = []

    if slots < 2:
        issues.append("Configure at least 2 parallel slots, so interactive and "
                      "background work can be pinned apart.")
    if per_slot < MINIMUM_PER_SLOT_CONTEXT:
        issues.append(f"Each slot needs at least {MINIMUM_PER_SLOT_CONTEXT:,} tokens of "
                      f"context; this configuration gives each one {per_slot:,}.")
    if values["CACHE_IDLE_SLOTS"].lower() != "on":
        issues.append("Enable idle-slot caching, or a yielded slot loses its prefix.")
    if values["FIT"].lower() != "off":
        issues.append("Disable auto-fit, so the context a pinned slot was sized "
                      "against cannot be reduced at launch.")

    return {
        "values": values,
        "slots": slots,
        "total_context": total_context,
        "per_slot_context": per_slot,
        "issues": issues,
        "compatible": not issues,
    }


# ---------------------------------------------------------------------------
# the launched side
# ---------------------------------------------------------------------------

# Long and short spellings of the flags the contract cares about, as llama.cpp
# accepts them and as scripts/start-chat-backend-dense.sh emits them.
_VALUE_FLAGS = {
    "--ctx-size": "CTX_SIZE", "-c": "CTX_SIZE",
    "--parallel": "N_PARALLEL", "-np": "N_PARALLEL",
    "--cache-ram": "CACHE_RAM", "-cram": "CACHE_RAM",
    "--ctx-checkpoints": "CTX_CHECKPOINTS", "-ctxcp": "CTX_CHECKPOINTS",
    "--cache-reuse": "CACHE_REUSE",
    "--fit": "FIT",
    "--fit-ctx": "FIT_CTX",
}
_BARE_FLAGS = {
    "--cache-idle-slots": ("CACHE_IDLE_SLOTS", "on"),
    "--no-cache-idle-slots": ("CACHE_IDLE_SLOTS", "off"),
    "--swa-full": ("SWA_FULL", "on"),
}


def launched_settings(cmdline: str) -> dict:
    """The contract-relevant flags a running backend was actually launched with.

    The config form holds what someone intends; this holds what the process is
    doing. They drift whenever a setting is saved and the backend is not
    restarted, and the drift is invisible from both ends.
    """
    tokens = [token for token in re.split(r"[\s\x00]+", cmdline or "") if token]
    values: dict[str, str] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        key, _, inline = token.partition("=")
        if key in _VALUE_FLAGS:
            if inline:
                values[_VALUE_FLAGS[key]] = inline
            elif index + 1 < len(tokens):
                values[_VALUE_FLAGS[key]] = tokens[index + 1]
                index += 1
        elif token in _BARE_FLAGS:
            name, value = _BARE_FLAGS[token]
            values[name] = value
        index += 1
    return values


def launched_check(cmdline: str) -> dict:
    """`contract_check` against a live command line rather than the env."""
    values = launched_settings(cmdline)
    if not values:
        return {"available": False, "values": {}, "issues": [], "compatible": False,
                "slots": 0, "total_context": 0, "per_slot_context": 0}
    # `--cache-idle-slots` defaults on in llama.cpp, and the launcher always
    # passes one of the two spellings, so an absent flag means the default.
    values.setdefault("CACHE_IDLE_SLOTS", "on")
    result = contract_check({f"{DEFAULT_PREFIX}_{key}": value for key, value in values.items()})
    result["available"] = True
    result["values"] = values
    return result


def drift_between(configured: dict, launched: dict) -> list[str]:
    """Settings the running backend does not share with the saved configuration."""
    if not launched.get("available"):
        return []
    notes = []
    labels = {
        "CTX_SIZE": "total context", "N_PARALLEL": "parallel slots",
        "CACHE_RAM": "prompt-cache RAM", "CTX_CHECKPOINTS": "context checkpoints",
        "CACHE_IDLE_SLOTS": "idle-slot caching", "FIT": "auto-fit",
    }
    for key, label in labels.items():
        want = str(configured["values"].get(key, "") or "").strip()
        have = str(launched["values"].get(key, "") or "").strip()
        if not want or not have:
            continue
        if want.lower() != have.lower():
            notes.append(f"{label} is {want} in the configuration but {have} in the "
                         f"running backend — it has not been restarted since that was saved.")
    return notes


# ---------------------------------------------------------------------------
# leases
# ---------------------------------------------------------------------------

def owner_home() -> Path | None:
    """Home directory of whoever owns the stack.

    The manager runs as root, so `Path.home()` is `/root` and the lease
    directory it would look in does not exist.
    """
    try:
        return Path(pwd.getpwuid(STACK_DIR.stat().st_uid).pw_dir)
    except (KeyError, OSError):
        return None


def lease_directory(env: dict | None = None) -> Path | None:
    """Where pi-forge writes its inference leases.

    Resolution follows pi-forge's own: an explicit agent directory wins,
    otherwise `<owner home>/.pi-forge/agent`.
    """
    env = env or {}
    configured = str(env.get("PI_FORGE_AGENT_DIR", "") or "").strip()
    if configured:
        return Path(configured).expanduser() / "inference-leases"
    home = owner_home()
    return (home / ".pi-forge" / "agent" / "inference-leases") if home else None


def _pid_alive(pid) -> bool:
    try:
        return Path(f"/proc/{int(pid)}").exists()
    except (TypeError, ValueError):
        return False


def read_leases(directory: Path | None, now_ms: float | None = None) -> list[dict]:
    """Every lease file, parsed and classified.

    Two writers produce two filename shapes — `<pid>-<uuid>.json` from the
    interactive extension, `background-<pid>-<n>.json` from the workers, where
    that trailing number is a clock in the JavaScript worker and a thread id in
    the Python one. So the age of a lease comes from `updatedAtMs` inside the
    file and never from its name.
    """
    now_ms = time.time() * 1000 if now_ms is None else now_ms
    if not directory or not directory.is_dir():
        return []
    entries = []
    for path in sorted(directory.glob("*.json")):
        entry = {"file": path.name, "path": str(path), "kind": None, "pid": None,
                 "slot": None, "updated_at_ms": None, "age_seconds": None,
                 "pid_alive": False, "classification": "malformed"}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = None
        if isinstance(data, dict):
            # Both runtimes default a missing `kind` to interactive, and an
            # interactive claim is the one that blocks background work.
            entry["kind"] = data.get("kind") or "interactive"
            entry["pid"] = data.get("pid")
            entry["slot"] = data.get("slot")
            updated = data.get("updatedAtMs")
            if isinstance(updated, (int, float)):
                entry["updated_at_ms"] = updated
                entry["age_seconds"] = round((now_ms - updated) / 1000, 1)
        entry["pid_alive"] = _pid_alive(entry["pid"])
        entry["classification"] = _classify_lease(entry)
        entries.append(entry)
    return entries


def _classify_lease(entry: dict) -> str:
    age = entry.get("age_seconds")
    if age is None:
        return "malformed"
    if age * 1000 <= LEASE_STALE_MS:
        return "fresh"
    if age >= LEASE_ORPHAN_SECONDS and not entry["pid_alive"]:
        return "orphan"
    return "stale"


def reapable(entries: list[dict]) -> list[dict]:
    """Leases safe to delete: the writer is gone and the claim is long expired.

    A malformed file counts only once it is older than the orphan threshold by
    file mtime, since it carries no timestamp of its own.
    """
    now = time.time()
    out = []
    for entry in entries:
        if entry["classification"] == "orphan":
            out.append(entry | {"reason": f"process {entry['pid']} is gone and the lease "
                                          f"expired {entry['age_seconds']}s ago"})
        elif entry["classification"] == "malformed":
            try:
                age = now - Path(entry["path"]).stat().st_mtime
            except OSError:
                continue
            if age >= LEASE_ORPHAN_SECONDS:
                out.append(entry | {"reason": f"unreadable lease, last written "
                                              f"{round(age)}s ago"})
    return out


def reap_leases(directory: Path | None, dry_run: bool = False) -> dict:
    """Remove orphaned leases. Fresh and stale-but-live leases are never touched."""
    entries = read_leases(directory)
    candidates = reapable(entries)
    removed, failed = [], []
    if not dry_run:
        for candidate in candidates:
            try:
                Path(candidate["path"]).unlink()
                removed.append(candidate)
            except OSError as exc:
                failed.append(candidate | {"error": str(exc)})
    return {
        "directory": str(directory) if directory else None,
        "dry_run": dry_run,
        "examined": len(entries),
        "candidates": candidates,
        "removed": removed if not dry_run else [],
        "failed": failed,
    }


def lease_summary(directory: Path | None) -> dict:
    entries = read_leases(directory)
    counts = {"fresh": 0, "stale": 0, "orphan": 0, "malformed": 0}
    for entry in entries:
        counts[entry["classification"]] = counts.get(entry["classification"], 0) + 1
    return {
        "directory": str(directory) if directory else None,
        "exists": bool(directory and directory.is_dir()),
        "entries": entries,
        "counts": counts,
        "reapable": len(reapable(entries)),
        "stale_ms": LEASE_STALE_MS,
    }


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------

def verify(env: dict, *, unit: str | None, unit_active: bool, cmdline: str = "",
           props: dict | None = None, slots: list | None = None,
           stats: dict | None = None, prefix: str = DEFAULT_PREFIX,
           window_seconds: int | None = None) -> dict:
    """Whether the running stack can honour the pi-forge scheduling contract.

    Reads only. `stats` is telemetry's journal summary, whose per-slot count of
    selections made by id is the evidence that pi-forge's `id_slot` is not just
    being sent but is being obeyed.
    """
    configured = contract_check(env, prefix)
    launched = launched_check(cmdline) if unit_active else {
        "available": False, "values": {}, "issues": [], "compatible": False,
        "slots": 0, "total_context": 0, "per_slot_context": 0,
    }
    by_id = ((stats or {}).get("scheduling") or {}).get("select_by_id_slots") or {}
    interactive = _integer(by_id.get(str(INTERACTIVE_SLOT), 0))
    background = _integer(by_id.get(str(BACKGROUND_SLOT), 0))
    leases = lease_summary(lease_directory(env))

    runtime_slots = (props or {}).get("total_slots")
    issues = list(configured["issues"])
    if unit_active and launched["available"]:
        issues.extend(issue for issue in launched["issues"] if issue not in issues)
    if unit_active and isinstance(runtime_slots, int) and runtime_slots < 2:
        issues.append(f"The running backend has {runtime_slots} slot(s); pi-forge needs 2.")

    drift = drift_between(configured, launched)
    return {
        "unit": unit,
        "unit_active": unit_active,
        "configured": configured,
        "launched": launched,
        "drift": drift,
        "runtime": {
            "total_slots": runtime_slots,
            "n_ctx_per_slot": (props or {}).get("n_ctx_per_slot"),
            "n_ctx_total": (props or {}).get("n_ctx_total"),
            "slots": slots or [],
        },
        "evidence": {
            "window_seconds": window_seconds,
            "select_methods": ((stats or {}).get("scheduling") or {}).get("select_methods") or {},
            "select_by_id_slots": by_id,
            "interactive_slot": INTERACTIVE_SLOT,
            "background_slot": BACKGROUND_SLOT,
            "interactive_pinned": interactive > 0,
            "background_pinned": background > 0,
            # Both halves seen means pi-forge is scheduling, not just configured
            # to. Neither half is proof of a fault — an idle session pins nothing.
            "observed": interactive > 0 and background > 0,
        },
        "leases": leases,
        "issues": issues,
        "ok": not issues and (not unit_active or launched["compatible"]),
    }


# ---------------------------------------------------------------------------
# the opt-in active probe
# ---------------------------------------------------------------------------

def probe_slot_pinning(env: dict, collector, slots=(INTERACTIVE_SLOT, BACKGROUND_SLOT),
                       timeout: int = 30) -> dict:
    """Send one tiny request per slot and read back which slot served it.

    Disruptive on purpose and never run unasked: a request pinned to slot 0
    replaces whatever prefix that slot is holding, so the next interactive turn
    reprocesses its prompt. The passive check in `verify` costs nothing and is
    what the panel shows by default.
    """
    host = str(env.get("LISTEN_HOST", "127.0.0.1") or "127.0.0.1").strip()
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1"
    port = str(env.get("NOTHINK_PORT", "8004") or "8004").strip()
    model = str(env.get("NOTHINK_MODEL_NAME", "chat") or "chat").strip()
    url = f"http://{host}:{port}/v1/chat/completions"

    started = time.time()
    results = []
    for slot in slots:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": f"slot probe {slot}"}],
            "max_tokens": 1,
            "temperature": 0,
            "id_slot": slot,
            "cache_prompt": True,
        }
        entry = {"requested_slot": slot, "sent": False, "error": "", "served_by": None}
        try:
            request = urlrequest.Request(
                url, data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}, method="POST")
            with urlrequest.urlopen(request, timeout=timeout) as response:
                response.read()
            entry["sent"] = True
        except urlerror.HTTPError as exc:
            entry["error"] = f"HTTP {exc.code}"
        except Exception as exc:
            entry["error"] = str(exc)
        results.append(entry)

    # The journal is the only place that says which slot was chosen, and
    # telemetry is already tailing it.
    selections = []
    if collector is not None:
        deadline = time.time() + 5
        while time.time() < deadline:
            selections = [event for event in collector.snapshot()
                          if event.get("kind") == "slot_select"
                          and event.get("method") == "id"
                          and (event.get("ts") or 0) >= started]
            if len(selections) >= len(results):
                break
            time.sleep(0.25)

    seen = {event["slot"] for event in selections}
    for entry in results:
        entry["served_by"] = entry["requested_slot"] if entry["requested_slot"] in seen else None

    return {
        "url": url,
        "started_at": started,
        "probes": results,
        "selections": [{"slot": event["slot"], "ts": event.get("ts")} for event in selections],
        "honoured": all(entry["served_by"] is not None for entry in results) and bool(results),
    }


__all__ = [
    "CONTRACT", "LEASE_ORPHAN_SECONDS", "LEASE_STALE_MS", "MINIMUM_PER_SLOT_CONTEXT",
    "INTERACTIVE_SLOT", "BACKGROUND_SLOT", "contract_check", "drift_between",
    "launched_check", "launched_settings", "lease_directory", "lease_summary",
    "owner_home", "probe_slot_pinning", "read_leases", "reap_leases", "reapable",
    "verify",
]
