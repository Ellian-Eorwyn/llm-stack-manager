#!/usr/bin/env python3
"""
gpu-lease.py — lend the chat backend's GPU to ComfyUI for one job, then give it back.

    gpu-lease.py acquire [--holder NAME] [--max-minutes 30] [--force]
    gpu-lease.py release [--reason TEXT]
    gpu-lease.py status

On llms, GPU 1 holds the 27B (`llm-a`) and GPU 0 is shared by the task model,
embeddings and ComfyUI. A heavy image job wants a whole card. `acquire`:

  1. takes a lock, so there is only ever one lease;
  2. refuses inside the quiet window (overnight jobs and evals use the 27B)
     unless --force;
  3. writes the lease file. The proxy sees it and sends new requests to its
     fallback (the Studio) from this moment on;
  4. waits for the backend's in-flight request to finish (/slots);
  5. stops the backend with systemctl. The manager's recorded expectation is
     left alone, so a reboot mid-lease brings the 27B back;
  6. restarts ComfyUI on the lent GPU (a runtime env file its unit reads);
  7. arms a watchdog that releases the lease if nobody else does.

`release` is idempotent and runs from the caller's `finally` and from the
watchdog: it waits for ComfyUI's queue, unloads its models, puts it back on its
own GPU, starts the backend and waits until it answers, then removes the lease
file so the proxy serves locally again. Output is one JSON object; the exit
status is non-zero when something is left wrong, with the reason in "error".

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
COMFY_READY_TIMEOUT = float(os.environ.get("GPU_LEASE_COMFY_READY_SECONDS", "180"))
COMFY_QUEUE_TIMEOUT = float(os.environ.get("GPU_LEASE_COMFY_QUEUE_SECONDS", "300"))
BACKEND_READY_TIMEOUT = float(os.environ.get("GPU_LEASE_BACKEND_READY_SECONDS", "300"))
POLL = float(os.environ.get("GPU_LEASE_POLL_SECONDS", "2"))


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


def comfy_queue_len() -> int | None:
    try:
        q = http_json(f"{COMFY_URL}/queue")
    except (urllib.error.URLError, OSError, ValueError):
        return None
    return len(q.get("queue_running", [])) + len(q.get("queue_pending", []))


def comfy_device() -> str | None:
    """The CUDA index ComfyUI reports, e.g. '1'; None while it is not answering."""
    try:
        stats = http_json(f"{COMFY_URL}/system_stats")
    except (urllib.error.URLError, OSError, ValueError):
        return None
    for dev in stats.get("devices", []):
        name = str(dev.get("name", ""))
        if name.startswith("cuda:"):
            return name.split(":", 1)[1].split()[0]
    return None


def wait_until(predicate, timeout: float, what: str):
    deadline = time.monotonic() + timeout
    while True:
        if predicate():
            return
        if time.monotonic() >= deadline:
            raise LeaseError(f"timed out after {int(timeout)} s waiting for {what}")
        sleep(POLL)


def restart_comfy(device: str | None):
    if device is None:
        COMFY_DEVICE_FILE.unlink(missing_ok=True)
    else:
        COMFY_DEVICE_FILE.write_text(f"COMFY_CUDA_DEVICE={device}\n")
    run(["systemctl", "--user", "restart", COMFY_UNIT])
    want = device or "0"
    wait_until(lambda: comfy_device() == want, COMFY_READY_TIMEOUT, f"ComfyUI on cuda:{want}")


def clear_watchdog():
    run(["systemctl", "--user", "stop", f"{WATCHDOG_UNIT}.timer"], check=False)
    run(["systemctl", "--user", "reset-failed", f"{WATCHDOG_UNIT}.service"], check=False)


def watchdog_command(max_minutes: int) -> list[str]:
    return [
        "systemd-run", "--user", "--collect", f"--unit={WATCHDOG_UNIT}",
        f"--on-active={max_minutes * 60 + 600}",
        sys.executable, str(Path(__file__).resolve()), "release", "--reason", "watchdog",
    ]


# --- commands -------------------------------------------------------------------

def acquire(holder: str, max_minutes: int, force: bool) -> dict:
    t0 = time.monotonic()
    started = now()
    if in_quiet_window(started) and not force:
        raise LeaseError(f"inside the quiet window {QUIET_WINDOW}; overnight jobs use the 27B (use --force to override)")
    existing = read_lease()
    if existing is not None:
        raise LeaseError(f"a lease is already held: {json.dumps(existing)}")
    queued = comfy_queue_len()
    if queued is None:
        raise LeaseError(f"ComfyUI is not answering at {COMFY_URL}")
    if queued:
        raise LeaseError(f"ComfyUI has {queued} job(s) queued; restarting it would drop them")

    lease = {
        "holder": holder,
        "started": started.isoformat(timespec="seconds"),
        "expires": datetime.fromtimestamp(started.timestamp() + max_minutes * 60, started.tzinfo).isoformat(timespec="seconds"),
        "backend_unit": BACKEND_UNIT,
        "device": LENT_DEVICE,
    }
    LEASE_FILE.write_text(json.dumps(lease))  # the proxy falls back from here on
    try:
        wait_until(lambda: backend_busy() is not True, DRAIN_TIMEOUT, "the 27B to finish its current request")
        run(["sudo", "-n", "systemctl", "stop", BACKEND_UNIT])
        restart_comfy(LENT_DEVICE)
        clear_watchdog()
        run(watchdog_command(max_minutes))
    except BaseException:
        # Put everything back rather than leave the backend down with no lease.
        release("acquire failed")
        raise
    lease["acquire_seconds"] = round(time.monotonic() - t0, 1)
    LEASE_FILE.write_text(json.dumps(lease))
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
        http_json(f"{COMFY_URL}/free", {"unload_models": True, "free_memory": True})
    except (urllib.error.URLError, OSError, ValueError):
        pass  # the restart below frees it anyway
    try:
        if COMFY_DEVICE_FILE.exists() or comfy_device() not in (None, "0"):
            restart_comfy(None)
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
    if reason != "watchdog":
        clear_watchdog()
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


def status() -> dict:
    lease = read_lease()
    out = {
        "lease": lease,
        "backend_healthy": backend_healthy(),
        "comfy_device": comfy_device(),
        "now": now().isoformat(timespec="seconds"),
    }
    if lease and lease.get("expires"):
        out["expired"] = datetime.fromisoformat(lease["expires"]) < now()
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("acquire")
    a.add_argument("--holder", default="comfyui")
    a.add_argument("--max-minutes", type=int, default=30)
    a.add_argument("--force", action="store_true", help="ignore the quiet window")
    r = sub.add_parser("release")
    r.add_argument("--reason", default="done")
    sub.add_parser("status")
    args = ap.parse_args(argv)

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    try:
        if args.cmd == "status":
            print(json.dumps(status()))
            return 0
        with LOCK_FILE.open("w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise LeaseError("another gpu-lease command is running")
            if args.cmd == "acquire":
                result = acquire(args.holder, args.max_minutes, args.force)
            else:
                result = release(args.reason)
    except LeaseError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1
    print(json.dumps(result))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
