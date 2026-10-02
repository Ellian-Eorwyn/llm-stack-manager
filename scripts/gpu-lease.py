#!/usr/bin/env python3
"""
gpu-lease.py — lend the chat backend's GPU to ComfyUI for one job, then give it back.

    gpu-lease.py acquire [--holder NAME] [--max-minutes N] [--idle-minutes N] [--force]
    gpu-lease.py touch [--max-minutes N]
    gpu-lease.py release [--reason TEXT]
    gpu-lease.py status
    gpu-lease.py watch          (the watchdog; systemd runs it, not you)
    gpu-lease.py heal           (the backend healer; a systemd timer runs it each minute)

On llms, ComfyUI lives on GPU 1 permanently (its unit's default), idle at a
couple hundred MB beside the 27B (`llm-a`). A generation job needs the whole
card, so `acquire` evicts the 27B — and only the 27B — from GPU 1:

  1. takes a lock, so there is only ever one lease;
  2. refuses inside the quiet window (overnight jobs and evals use the 27B)
     unless --force;
  3. writes the lease file. The proxy sees it and sends new requests to its
     fallback (the Studio) from this moment on;
  4. waits for the backend's in-flight request to finish (/slots);
  5. stops the backend with systemctl. The manager's recorded expectation is
     left alone, so a reboot mid-lease brings the 27B back;
  6. moves ComfyUI onto the lent GPU if it is not already there (a runtime
     env file its unit reads);
  7. arms a watchdog that releases the lease if nobody else does.

Two kinds of lease:

  - fixed (the default): the caller releases it in its `finally`; the watchdog
    releases it 10 minutes after `expires` (`--max-minutes`, default 30) in
    case the caller died.
  - idle (`--idle-minutes N`, for SillyTavern sessions): nobody has to release
    it. The watchdog releases it once ComfyUI has been idle for N minutes,
    whoever its jobs came from: activity is the newest finished job in
    ComfyUI's /history, a non-empty queue at any check, and `touch`. While it
    is not idle the watchdog keeps waiting. `--max-minutes` (default 720) is a
    hard ceiling so a stuck lease still ends, and inside the quiet window an
    unforced idle lease ends after 5 idle minutes, so the overnight jobs get
    their 27B back.

Taking an idle lease again with the same holder renews it instead of failing,
and `touch` marks activity now (and with --max-minutes moves the end out).

`release` is idempotent and runs from the caller's `finally` and from the
watchdog: it waits for ComfyUI's queue, unloads its models (so the 27B fits
back on the card), makes sure ComfyUI is on GPU 1, starts the backend and
waits until it answers, then removes the lease file so the proxy serves
locally again. ComfyUI never touches GPU 0. Output is one JSON object; the
exit status is non-zero when something is left wrong, with the reason in
"error".

`heal` restarts a backend that is running but has stopped serving. NInfer
latches itself "unavailable" (/health 503, process alive, so systemd's
Restart= never fires) after a worker failure it cannot recover from — on
2026-10-01 and 10-02 a `std::bad_alloc` on an image request with an
uncapped output, retried three times by Hermes. Without a lease, with the
unit active past its load time and /health failing on two checks at least
HEAL_CONFIRM_SECONDS apart, it restarts the unit and waits for /health. At
most HEAL_MAX_PER_HOUR restarts an hour; past that it logs once and leaves
the backend alone (the proxy keeps serving from the fallback).

Standard library only: it runs with the system python3.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

UID = os.getuid()
RUNTIME_DIR = Path(os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{UID}")
LEASE_FILE = Path(os.environ.get("CHAT_FALLBACK_LEASE_FILE") or RUNTIME_DIR / "gpu1-lease.json")
LOCK_FILE = RUNTIME_DIR / "gpu1-lease.lock"
COMFY_DEVICE_FILE = Path(os.environ.get("GPU_LEASE_COMFY_DEVICE_FILE") or RUNTIME_DIR / "comfyui-device.env")
LOG_FILE = Path(os.environ.get("GPU_LEASE_LOG") or Path.home() / ".local/state/gpu-lease.jsonl")

BACKEND_UNIT = os.environ.get("GPU_LEASE_BACKEND_UNIT", "llm-a")
BACKEND_URL = os.environ.get("GPU_LEASE_BACKEND_URL", "http://127.0.0.1:8010").rstrip("/")
COMFY_UNIT = os.environ.get("GPU_LEASE_COMFY_UNIT", "comfyui")
COMFY_URL = os.environ.get("GPU_LEASE_COMFY_URL", "http://100.124.56.11:8188").rstrip("/")
LENT_DEVICE = os.environ.get("GPU_LEASE_DEVICE", "1")
QUIET_WINDOW = os.environ.get("GPU_LEASE_QUIET", "23:00-06:30")
WATCHDOG_UNIT = "gpu1-lease-watchdog"

DRAIN_TIMEOUT = float(os.environ.get("GPU_LEASE_DRAIN_SECONDS", "600"))
COMFY_READY_TIMEOUT = float(os.environ.get("GPU_LEASE_COMFY_READY_SECONDS", "300"))
COMFY_QUEUE_TIMEOUT = float(os.environ.get("GPU_LEASE_COMFY_QUEUE_SECONDS", "300"))
BACKEND_READY_TIMEOUT = float(os.environ.get("GPU_LEASE_BACKEND_READY_SECONDS", "300"))
POLL = float(os.environ.get("GPU_LEASE_POLL_SECONDS", "2"))
WATCH_POLL = float(os.environ.get("GPU_LEASE_WATCH_SECONDS", "60"))
FIXED_GRACE = 600  # a fixed lease's watchdog waits this long past `expires`
QUIET_IDLE_MINUTES = float(os.environ.get("GPU_LEASE_QUIET_IDLE_MINUTES", "5"))
DEFAULT_MAX_MINUTES = 30
DEFAULT_IDLE_CEILING_MINUTES = 720

HEAL_STATE_FILE = RUNTIME_DIR / "backend-heal.json"
HEAL_GRACE_SECONDS = float(os.environ.get("GPU_LEASE_HEAL_GRACE_SECONDS", "300"))  # load time
HEAL_CONFIRM_SECONDS = float(os.environ.get("GPU_LEASE_HEAL_CONFIRM_SECONDS", "90"))
HEAL_MAX_PER_HOUR = int(os.environ.get("GPU_LEASE_HEAL_MAX_PER_HOUR", "3"))


class LeaseError(Exception):
    pass


# --- small helpers (patched in tests) ---------------------------------------

def now() -> datetime:
    return datetime.now(timezone.utc).astimezone()


def sleep(seconds: float):
    time.sleep(seconds)


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if check and proc.returncode != 0:
        raise LeaseError(f"{' '.join(cmd)} failed: {(proc.stderr or proc.stdout).strip()[:300]}")
    return proc


def http_json(url: str, data: dict | None = None, timeout: float = 5.0):
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, method="POST" if data is not None else "GET",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    return json.loads(raw) if raw.strip() else {}


def http_status(url: str, timeout: float = 5.0) -> int:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except (urllib.error.URLError, OSError):
        return 0


# --- state ---------------------------------------------------------------------

def in_quiet_window(t: datetime, window: str = QUIET_WINDOW) -> bool:
    if not window:
        return False
    start_s, end_s = window.split("-")
    start = int(start_s[:2]) * 60 + int(start_s[3:])
    end = int(end_s[:2]) * 60 + int(end_s[3:])
    minute = t.hour * 60 + t.minute
    if start <= end:
        return start <= minute < end
    return minute >= start or minute < end


def read_lease() -> dict | None:
    try:
        return json.loads(LEASE_FILE.read_text())
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        return {"unreadable": True}


def write_lease(lease: dict):
    # Atomic, so the proxy and `status` never read half a file.
    tmp = LEASE_FILE.with_name(LEASE_FILE.name + ".tmp")
    tmp.write_text(json.dumps(lease))
    os.replace(tmp, LEASE_FILE)


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, now().tzinfo).isoformat(timespec="seconds")


def ts_of(value: str | None) -> float | None:
    try:
        return datetime.fromisoformat(value).timestamp() if value else None
    except ValueError:
        return None


def log_event(event: dict):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a") as fh:
        fh.write(json.dumps(event, sort_keys=True) + "\n")


def backend_busy() -> bool | None:
    """True while the backend is generating; None when it does not answer."""
    try:
        slots = http_json(f"{BACKEND_URL}/slots")
    except (urllib.error.URLError, OSError, ValueError):
        return None
    return any(s.get("is_processing") for s in slots if isinstance(s, dict))


def backend_healthy() -> bool:
    return http_status(f"{BACKEND_URL}/health") == 200


def backend_unit_state() -> tuple[str, float | None]:
    """(ActiveState, epoch seconds it entered that state) of the backend unit."""
    out = run(["systemctl", "show", BACKEND_UNIT, "-p", "ActiveState", "-p", "ActiveEnterTimestamp",
               "--timestamp=unix"], check=False).stdout
    props = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
    since = props.get("ActiveEnterTimestamp", "").lstrip("@")
    return props.get("ActiveState", "unknown"), float(since) if since.isdigit() else None


def comfy_queue_len() -> int | None:
    try:
        q = http_json(f"{COMFY_URL}/queue")
    except (urllib.error.URLError, OSError, ValueError):
        return None
    return len(q.get("queue_running", [])) + len(q.get("queue_pending", []))


def comfy_last_finished() -> float | None:
    """When ComfyUI last finished (or started) a job, as epoch seconds.

    /history keeps finished prompts in memory, newest last; each one's status
    messages carry millisecond timestamps. Any client counts: SillyTavern,
    the web UI, Hermes. None when ComfyUI does not answer or has no history
    (a restart empties it; the lease's own `last_activity` covers that).
    """
    try:
        hist = http_json(f"{COMFY_URL}/history?max_items=5")
    except (urllib.error.URLError, OSError, ValueError):
        return None
    newest = None
    for item in (hist.values() if isinstance(hist, dict) else []):
        for msg in ((item or {}).get("status") or {}).get("messages") or []:
            if isinstance(msg, list) and len(msg) == 2 and isinstance(msg[1], dict):
                stamp = msg[1].get("timestamp")
                if isinstance(stamp, (int, float)):
                    newest = max(newest or 0.0, stamp / 1000)
    return newest


def comfy_gpus() -> set[str]:
    """Physical GPU indices the ComfyUI unit's processes hold, from nvidia-smi.

    ComfyUI's own /system_stats cannot say: `--cuda-device 1` hides the other
    cards, so it names physical GPU 1 "cuda:0".
    """
    cg = run(["systemctl", "--user", "show", "-p", "ControlGroup", "--value", COMFY_UNIT], check=False).stdout.strip()
    try:
        pids = set(Path(f"/sys/fs/cgroup{cg}/cgroup.procs").read_text().split()) if cg else set()
    except OSError:
        pids = set()
    gpus = run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], check=False).stdout
    index_of = {u.strip(): i.strip() for i, u in (line.split(",", 1) for line in gpus.splitlines() if "," in line)}
    apps = run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"], check=False).stdout
    held = set()
    for line in apps.splitlines():
        if "," not in line:
            continue
        pid, uuid = (x.strip() for x in line.split(",", 1))
        if pid in pids and uuid in index_of:
            held.add(index_of[uuid])
    return held


def comfy_device() -> str | None:
    """The physical GPU ComfyUI runs on, e.g. '1'; None while it is not answering."""
    try:
        http_json(f"{COMFY_URL}/system_stats")
    except (urllib.error.URLError, OSError, ValueError):
        return None
    held = comfy_gpus()
    return next(iter(held)) if len(held) == 1 else None


def wait_until(predicate, timeout: float, what: str):
    deadline = time.monotonic() + timeout
    while True:
        if predicate():
            return
        if time.monotonic() >= deadline:
            raise LeaseError(f"timed out after {int(timeout)} s waiting for {what}")
        sleep(POLL)


def move_comfy(device: str):
    """Put ComfyUI on the physical GPU `device`, restarting it only if it is
    elsewhere. Its unit defaults to GPU 1, so this is a no-op in the normal
    flow; the env file exists for a one-off GPU 0 (tight VRAM) or a re-home."""
    if comfy_device() == device:
        return
    if COMFY_DEVICE_FILE.exists() or comfy_device() in (None,):
        COMFY_DEVICE_FILE.write_text(f"COMFY_CUDA_DEVICE={device}\n")
    else:
        COMFY_DEVICE_FILE.unlink(missing_ok=True)
    run(["systemctl", "--user", "restart", COMFY_UNIT])
    wait_until(lambda: comfy_device() == device, COMFY_READY_TIMEOUT, f"ComfyUI on GPU {device}")


def clear_watchdog():
    # The .timer is the watchdog before 2026-10-01 (one fixed release time).
    run(["systemctl", "--user", "stop", f"{WATCHDOG_UNIT}.service", f"{WATCHDOG_UNIT}.timer"], check=False)
    run(["systemctl", "--user", "reset-failed", f"{WATCHDOG_UNIT}.service"], check=False)


def watchdog_command() -> list[str]:
    # A small loop (`watch`) rather than a timer at a fixed time: an idle lease
    # has no end time to arm for. Restart covers a crash, and a release that
    # left something wrong (exit 1) is retried five minutes later.
    return [
        "systemd-run", "--user", "--collect", f"--unit={WATCHDOG_UNIT}",
        "-p", "Restart=on-failure", "-p", "RestartSec=300",
        sys.executable, str(Path(__file__).resolve()), "watch",
    ]


def watchdog_active() -> bool:
    out = run(["systemctl", "--user", "is-active", f"{WATCHDOG_UNIT}.service"], check=False).stdout.strip()
    return out in ("active", "activating", "reloading")


def arm_watchdog():
    clear_watchdog()
    run(watchdog_command())


@contextmanager
def locked():
    """The one-lease lock; raises LeaseError if another command holds it."""
    with LOCK_FILE.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise LeaseError("another gpu-lease command is running") from None
        yield


# --- idleness -------------------------------------------------------------------

def last_activity(lease: dict, t: float) -> tuple[float, bool]:
    """(newest activity as epoch seconds, ComfyUI busy right now)."""
    stamps = [ts_of(lease.get("last_activity")), ts_of(lease.get("started")), comfy_last_finished()]
    busy = bool(comfy_queue_len())
    if busy:
        stamps.append(t)
    return max(s for s in stamps if s is not None), busy


def idle_limit_seconds(lease: dict, t: datetime) -> float:
    limit = float(lease["idle_minutes"]) * 60
    if in_quiet_window(t) and not lease.get("forced"):
        limit = min(limit, QUIET_IDLE_MINUTES * 60)
    return limit


def due(lease: dict, t: datetime, activity: float, busy: bool) -> str | None:
    """Why the watchdog should release now, or None to keep waiting."""
    end = ts_of(lease.get("expires"))
    tt = t.timestamp()
    if not lease.get("idle_minutes"):
        if end is not None and tt >= end + FIXED_GRACE:
            return "watchdog: past its end"
        return None
    if end is not None and tt >= end:
        return "watchdog: hard ceiling"
    if not busy and tt - activity >= idle_limit_seconds(lease, t):
        return f"watchdog: ComfyUI idle {int((tt - activity) // 60)} min"
    return None


def releases_at(lease: dict, activity: float, t: datetime) -> float | None:
    """When the watchdog will release if nothing happens meanwhile."""
    end = ts_of(lease.get("expires"))
    if not lease.get("idle_minutes"):
        return None if end is None else end + FIXED_GRACE
    idle_end = activity + idle_limit_seconds(lease, t)
    return idle_end if end is None else min(idle_end, end)


# --- commands -------------------------------------------------------------------

def acquire(holder: str, max_minutes: int | None, force: bool, idle_minutes: int | None = None) -> dict:
    t0 = time.monotonic()
    started = now()
    if max_minutes is None:
        max_minutes = DEFAULT_IDLE_CEILING_MINUTES if idle_minutes else DEFAULT_MAX_MINUTES
    existing = read_lease()
    if (idle_minutes and existing is not None and existing.get("holder") == holder
            and existing.get("idle_minutes")):
        # Taking an idle lease again (e.g. "enable SillyTavern mode" twice)
        # renews it. A fixed lease is still refused: its first caller will
        # release it in its `finally`, under the second one.
        existing["idle_minutes"] = idle_minutes
        existing["forced"] = bool(existing.get("forced") or force)
        return {**touch(max_minutes, lease=existing), "renewed": True}
    if in_quiet_window(started) and not force:
        raise LeaseError(f"inside the quiet window {QUIET_WINDOW}; overnight jobs use the 27B (use --force to override)")
    if existing is not None:
        raise LeaseError(f"a lease is already held: {json.dumps(existing)}")
    if comfy_queue_len() is None:
        raise LeaseError(f"ComfyUI is not answering at {COMFY_URL}")
    # Someone else's job (the UI, another Hermes run) finishes first: a
    # ComfyUI restart would drop it, and the 27B stopping would OOM it.
    try:
        wait_until(lambda: comfy_queue_len() == 0, COMFY_QUEUE_TIMEOUT, "ComfyUI's current jobs to finish")
    except LeaseError:
        raise LeaseError(f"ComfyUI still has {comfy_queue_len()} job(s) after {int(COMFY_QUEUE_TIMEOUT)} s; "
                         "evicting the 27B would drop them") from None

    lease = {
        "holder": holder,
        "started": started.isoformat(timespec="seconds"),
        "expires": datetime.fromtimestamp(started.timestamp() + max_minutes * 60, started.tzinfo).isoformat(timespec="seconds"),
        "idle_minutes": idle_minutes,
        "forced": force,
        "last_activity": started.isoformat(timespec="seconds"),
        "backend_unit": BACKEND_UNIT,
        "device": LENT_DEVICE,
    }
    write_lease(lease)  # the proxy falls back from here on
    try:
        wait_until(lambda: backend_busy() is not True, DRAIN_TIMEOUT, "the 27B to finish its current request")
        run(["sudo", "-n", "systemctl", "stop", BACKEND_UNIT])
        move_comfy(LENT_DEVICE)  # usually already there: no restart
        arm_watchdog()
    except BaseException:
        # Put everything back rather than leave the backend down with no lease.
        release("acquire failed")
        raise
    lease["acquire_seconds"] = round(time.monotonic() - t0, 1)
    write_lease(lease)
    log_event({"event": "acquire", **lease})
    return {"ok": True, **lease}


def release(reason: str) -> dict:
    t0 = time.monotonic()
    lease = read_lease()
    errors = []
    # Let a running job finish, then stop whatever is left.
    try:
        wait_until(lambda: not comfy_queue_len(), COMFY_QUEUE_TIMEOUT, "ComfyUI's queue to empty")
    except LeaseError as exc:
        errors.append(str(exc))
        try:
            http_json(f"{COMFY_URL}/queue", {"clear": True})
            http_json(f"{COMFY_URL}/interrupt", {})
        except (urllib.error.URLError, OSError, ValueError):
            pass
    try:
        # Unload so the 27B fits back beside ComfyUI's idle footprint; the
        # next generation pays a one-time model load (~25 s), not a restart.
        http_json(f"{COMFY_URL}/free", {"unload_models": True, "free_memory": True})
    except (urllib.error.URLError, OSError, ValueError):
        pass  # the restart below frees it anyway
    try:
        move_comfy(LENT_DEVICE)  # ComfyUI stays on GPU 1; restart only if it drifted
    except LeaseError as exc:
        errors.append(str(exc))
    try:
        run(["sudo", "-n", "systemctl", "start", BACKEND_UNIT])
        wait_until(backend_healthy, BACKEND_READY_TIMEOUT, f"{BACKEND_UNIT} to answer /health")
    except LeaseError as exc:
        errors.append(str(exc))
    # The lease file goes only once the backend answers: until then the proxy
    # keeps serving from the fallback instead of returning 503s.
    if backend_healthy():
        LEASE_FILE.unlink(missing_ok=True)
    if not reason.startswith("watchdog"):
        clear_watchdog()  # (the watchdog itself just exits)
    result = {
        "ok": not errors,
        "reason": reason,
        "held": lease,
        "release_seconds": round(time.monotonic() - t0, 1),
        "backend_healthy": backend_healthy(),
        "lease_file": LEASE_FILE.exists(),
    }
    if errors:
        result["error"] = "; ".join(errors)
    log_event({"event": "release", "at": now().isoformat(timespec="seconds"), **result})
    return result


def touch(max_minutes: int | None = None, lease: dict | None = None) -> dict:
    """Mark activity now; with max_minutes, move the end out to at least now + that."""
    lease = lease or read_lease()
    if lease is None or lease.get("unreadable"):
        raise LeaseError("no lease is held" if lease is None else "the lease file is unreadable")
    t = now()
    lease["last_activity"] = t.isoformat(timespec="seconds")
    if max_minutes:
        new_end = t.timestamp() + max_minutes * 60
        if new_end > (ts_of(lease.get("expires")) or 0):
            lease["expires"] = iso(new_end)
    write_lease(lease)
    rearmed = not watchdog_active()
    if rearmed:
        arm_watchdog()
    log_event({"event": "touch", "at": lease["last_activity"], "expires": lease.get("expires"),
               "holder": lease.get("holder"), "watchdog_rearmed": rearmed})
    return {"ok": True, **lease, "watchdog_rearmed": rearmed}


def watch() -> int:
    """The watchdog loop: release the lease when it is due, else keep checking.

    Each check records the newest ComfyUI activity in the lease file, so a
    ComfyUI restart (which empties /history) does not reset the idle clock.
    """
    while True:
        lease = read_lease()
        if lease is None:
            return 0  # released by someone else
        if not lease.get("unreadable"):
            try:
                with locked():
                    lease = read_lease()
                    if lease is None:
                        return 0
                    t = now()
                    activity, busy = last_activity(lease, t.timestamp())
                    if activity > (ts_of(lease.get("last_activity")) or 0):
                        lease["last_activity"] = iso(activity)
                        write_lease(lease)
                    reason = due(lease, t, activity, busy)
                    if reason:
                        return 0 if release(reason).get("ok") else 1
            except LeaseError:
                pass  # another command holds the lock; look again next round
        sleep(WATCH_POLL)


def read_heal_state() -> dict:
    try:
        return json.loads(HEAL_STATE_FILE.read_text())
    except (OSError, ValueError):
        return {}


def write_heal_state(state: dict):
    tmp = HEAL_STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, sort_keys=True))
    tmp.replace(HEAL_STATE_FILE)


def heal() -> dict:
    """One check of the backend; restart it if it is running but latched. Caller holds the lock."""
    t = now().timestamp()
    state = read_heal_state()
    restarts = [r for r in state.get("restarts", []) if t - r < 3600]
    state["restarts"] = restarts

    def done(action: str, **extra) -> dict:
        if action != "watching":
            state.pop("unhealthy_since", None)
        write_heal_state(state)
        return {"ok": True, "action": action, **extra}

    if read_lease() is not None:
        return done("none", why="lease held")
    unit_state, since = backend_unit_state()
    if unit_state != "active":
        # Stopped on purpose, failed (systemd's Restart= handles that) or mid-start.
        return done("none", why=f"{BACKEND_UNIT} is {unit_state}")
    if since is not None and t - since < HEAL_GRACE_SECONDS:
        return done("none", why="backend still loading")
    code = http_status(f"{BACKEND_URL}/health")
    if code == 200:
        return done("none", why="healthy")
    first = state.get("unhealthy_since")
    if first is None or t - first < HEAL_CONFIRM_SECONDS:
        state.setdefault("unhealthy_since", t)
        return done("watching", health=code)
    if len(restarts) >= HEAL_MAX_PER_HOUR:
        if not state.get("gave_up_logged"):
            state["gave_up_logged"] = True
            log_event({"event": "heal", "at": iso(t), "action": "gave up", "health": code,
                       "restarts_last_hour": len(restarts)})
        return done("gave up", health=code, restarts_last_hour=len(restarts))
    state.pop("gave_up_logged", None)
    run(["sudo", "-n", "systemctl", "restart", BACKEND_UNIT])
    restarts.append(t)
    try:
        wait_until(backend_healthy, BACKEND_READY_TIMEOUT, f"{BACKEND_UNIT} to answer /health")
        recovered = True
    except LeaseError:
        recovered = False
    event = {"event": "heal", "at": iso(t), "action": "restarted", "health": code,
             "unhealthy_seconds": round(t - first), "recovered": recovered}
    log_event(event)
    state["last_restart"] = event
    return done("restarted", health=code, recovered=recovered)


def status() -> dict:
    lease = read_lease()
    t = now()
    out = {
        "lease": lease,
        "backend_healthy": backend_healthy(),
        "comfy_device": comfy_device(),
        "now": t.isoformat(timespec="seconds"),
    }
    heal_state = read_heal_state()
    if heal_state.get("last_restart"):
        out["last_heal"] = heal_state["last_restart"]
    if heal_state.get("gave_up_logged"):
        out["heal_gave_up"] = True
    if lease and lease.get("expires"):
        out["expired"] = datetime.fromisoformat(lease["expires"]) < t
    if lease and not lease.get("unreadable"):
        activity, busy = last_activity(lease, t.timestamp())
        out["comfy_busy"] = busy
        out["idle_minutes"] = round((t.timestamp() - activity) / 60, 1)
        out["last_activity"] = iso(activity)
        end = releases_at(lease, activity, t)
        out["releases_at"] = None if end is None else iso(end)
        out["watchdog_active"] = watchdog_active()
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("acquire")
    a.add_argument("--holder", default="comfyui")
    a.add_argument("--max-minutes", type=int, default=None,
                   help=f"fixed lease: its length (default {DEFAULT_MAX_MINUTES}); "
                        f"idle lease: the hard ceiling (default {DEFAULT_IDLE_CEILING_MINUTES})")
    a.add_argument("--idle-minutes", type=int, default=None,
                   help="release once ComfyUI has been idle this long (no caller release needed)")
    a.add_argument("--force", action="store_true", help="ignore the quiet window")
    t = sub.add_parser("touch", help="mark activity now; keeps an idle lease from ending")
    t.add_argument("--max-minutes", type=int, default=None, help="also move the end out to now + N")
    r = sub.add_parser("release")
    r.add_argument("--reason", default="done")
    sub.add_parser("status")
    sub.add_parser("watch", help="the watchdog loop (systemd runs it)")
    sub.add_parser("heal", help="restart a backend that runs but no longer serves (a timer runs it)")
    args = ap.parse_args(argv)

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    try:
        if args.cmd == "status":
            print(json.dumps(status()))
            return 0
        if args.cmd == "watch":
            return watch()
        if args.cmd == "heal":
            try:
                with locked():
                    result = heal()
            except LeaseError as exc:
                if "another gpu-lease command" not in str(exc):
                    raise
                # acquire/release own the backend right now; check again next minute.
                result = {"ok": True, "action": "none", "why": str(exc)}
            print(json.dumps(result))
            return 0
        with locked():
            if args.cmd == "acquire":
                result = acquire(args.holder, args.max_minutes, args.force, args.idle_minutes)
            elif args.cmd == "touch":
                result = touch(args.max_minutes)
            else:
                result = release(args.reason)
    except LeaseError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1
    print(json.dumps(result))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
