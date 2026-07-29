#!/usr/bin/env python3
"""
Service health for the LLM Stack Manager.

`systemctl is-active` answers "is the process running", and the services panel
has always treated that as "is the service working". Those are different
questions, and the gap between them is where this stack's failures live:

  * `glmocr-sdk` runs happily with its OCR upstream dead. It answers `/health`
    with `{"status":"ok"}` because its own process is fine; every document it is
    handed then fails against `127.0.0.1:8009`, which nothing is listening on.
  * A unit that crashed reads exactly like a unit somebody stopped, because
    `is-active` returns non-`active` for both and the caller compared against
    the string `"active"`.
  * Half the auxiliary services are off at any time, and nothing said whether
    that was deliberate.

So health is assembled from four inputs rather than one:

  * the systemd unit state, now including `failed` (see `ServiceManager.state`);
  * a readiness probe against the port the service actually serves — `/props`
    for llama.cpp backends, `/v1/models` for the proxies, `/health` for the SDK
    and Honcho, a TCP connect for Playwright;
  * the upstreams it declares, so a service whose dependency is down reports
    `degraded` instead of `active`;
  * an expectation, recorded when the operator starts or stops it, so an
    intentionally-off service reads as `stopped` rather than as a fault.

Expectation has to be recorded because nothing else on the box carries it.
`systemctl is-enabled` says `disabled` for units that are running right now, and
the `*_ENABLED` env flags say `on` for services that are deliberately stopped —
they mean "configured", not "should be up". The one exception is a flag set to
`off`, which the start scripts honour by exiting 0 immediately; that genuinely
does mean the unit can never come up, so it is treated as expected-off.

Probes run on a background sweep rather than inside `/api/status`. The UI polls
status every 5 seconds and that poll already spawns a `systemctl` call per
service; a hung backend answering a 1.5s probe inline is exactly the case that
most needs to render, and it is the case that would blow the interval.
"""

import json
import socket
import sys
import threading
import time
from pathlib import Path

# Explicit rather than relying on the caller's path, so loading this module by
# path (as the tests do) resolves its sibling the same way systemd does.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import telemetry

STACK_DIR = Path(__file__).resolve().parent.parent
EXPECTATIONS_FILE = STACK_DIR / "config" / "service-expectations.json"

# Shorter than telemetry's, because a full sweep is serial and a wedged backend
# should not hold the next one up.
PROBE_TIMEOUT_SECONDS = 1.5
SWEEP_INTERVAL_SECONDS = 10.0

# States the services panel renders. `active`, `inactive` and `unknown` keep the
# meanings they always had; the other three are new.
STATE_ACTIVE = "active"
STATE_DEGRADED = "degraded"      # running, but not serving
STATE_FAILED = "failed"          # systemd says the unit failed
STATE_STOPPED = "stopped"        # not running, and not expected to be
STATE_INACTIVE = "inactive"      # not running, but expected on
STATE_UNKNOWN = "unknown"        # unit not installed


# ---------------------------------------------------------------------------
# probes
# ---------------------------------------------------------------------------

def tcp_port_open(host: str, port: str | int, timeout: float = PROBE_TIMEOUT_SECONDS) -> tuple[bool, str]:
    """Whether something is listening. `0.0.0.0` is a bind address, not a route."""
    try:
        target_host = "127.0.0.1" if host in {"0.0.0.0", "::", ""} else host
        with socket.create_connection((target_host, int(port)), timeout=timeout):
            return True, ""
    except Exception as exc:
        return False, str(exc)


def _llama_probes() -> dict[str, dict]:
    """`/props` probes for every unit telemetry already knows how to reach.

    `SERVICES[*]["ports"]` is a display string ("8010 internal / llms:8010"), so
    the machine-readable port map is telemetry's, and this reuses it rather than
    starting a third one.
    """
    probes = {}
    for spec in telemetry.BACKEND_TARGETS:
        for unit in spec["units"]:
            probes[unit] = {
                "kind": "http",
                "path": "/props",
                "host_key": spec["host_key"],
                "port_key": spec["port_key"],
                "default_port": telemetry.DEFAULT_BACKEND_PORTS.get(spec["port_key"], ""),
            }
    return probes


# What "ready" means for each service. A service with no entry here is judged on
# its unit state and upstreams alone — `honcho-deriver` is a worker with no
# listening socket, and `searxng` is already a composite of uwsgi, nginx and a
# socket file reported through its own manager.
SERVICE_PROBES = _llama_probes() | {
    "chat-proxy": {
        "kind": "http", "path": "/v1/models",
        "host_key": "LISTEN_HOST", "port_key": "NOTHINK_PORT", "default_port": "8004",
    },
    "chat-proxy2": {
        "kind": "http", "path": "/v1/models",
        "host_key": "LISTEN_HOST", "port_key": "NOTHINK2_PORT", "default_port": "8104",
    },
    "glmocr-sdk": {
        "kind": "http", "path": "/health",
        "host_key": "GLMOCR_SDK_HOST", "port_key": "GLMOCR_SDK_PORT", "default_port": "5002",
        "expect_field": ("status", "ok"),
    },
    "honcho-api": {
        "kind": "http", "path": "/health",
        "host_key": "HONCHO_HOST", "port_key": "HONCHO_PORT", "default_port": "8090",
    },
    "playwright-server": {
        "kind": "tcp",
        "host_key": "PLAYWRIGHT_HOST", "port_key": "PLAYWRIGHT_PORT", "default_port": "3001",
    },
}


# Each entry is a list of any-of groups: every group must be satisfied by at
# least one of its members. The primary backend is served by whichever of three
# mutually exclusive units is up, which is why the inner list exists.
#
# Sourced from the graph the installer already writes as systemd `After=`/
# `Wants=` (install.sh), from `setup_engine.COMPONENT_DEPENDENCIES`, and from
# the OCR upstream that `scripts/start-glmocr-sdk.sh` bakes into the SDK's
# generated config. `tests/test_health.py` asserts it stays consistent with the
# component-level map so the two cannot drift apart.
SERVICE_DEPENDENCIES = {
    "chat-proxy": [["chat-backend-dense", "chat-backend-moe", "chat-backend"]],
    "chat-proxy2": [["chat-backend2"]],
    "glmocr-sdk": [["ocr"]],
    "honcho-api": [["chat-proxy"], ["embed"]],
    "honcho-deriver": [["honcho-api"]],
}

# A flag set to `off` means the start script exits 0 without launching anything,
# so the unit cannot come up and being down is not a fault. Only `off` is
# meaningful here: these flags read `on` for services that are deliberately
# stopped, so `on` says nothing about whether the service should be running.
ENABLED_FLAGS = {
    "glmocr-sdk": "GLMOCR_SDK_ENABLED",
    "searxng": "SEARXNG_ENABLED",
    "playwright-server": "PLAYWRIGHT_ENABLED",
    "honcho-api": "HONCHO_ENABLED",
    "honcho-deriver": "HONCHO_ENABLED",
}


def dependency_units() -> set[str]:
    """Every unit named as somebody's upstream.

    The services panel renders one card for the primary backend, but three
    mutually exclusive units can serve it. The two that have no card still have
    to be asked about, or `chat-proxy` reads as degraded whenever the primary is
    served by a unit the panel does not list.
    """
    return {member for groups in SERVICE_DEPENDENCIES.values()
            for group in groups for member in group}


def endpoint_for(name: str, env: dict) -> tuple[str, str] | None:
    """(host, port) for a service's probe, or None if it has no probe."""
    spec = SERVICE_PROBES.get(name)
    if not spec:
        return None
    port = str(env.get(spec["port_key"]) or spec.get("default_port") or "").strip()
    if not port:
        return None
    host = str(env.get(spec["host_key"]) or "127.0.0.1").strip() or "127.0.0.1"
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1"
    return host, port


def probe(name: str, env: dict, timeout: float = PROBE_TIMEOUT_SECONDS) -> dict | None:
    """Run one service's readiness probe. None when the service has none."""
    spec = SERVICE_PROBES.get(name)
    endpoint = endpoint_for(name, env)
    if not spec or not endpoint:
        return None
    host, port = endpoint
    checked_at = time.time()

    if spec["kind"] == "tcp":
        ok, error = tcp_port_open(host, port, timeout)
        target = f"{host}:{port}"
        return {
            "ok": ok, "target": target, "http_status": None, "checked_at": checked_at,
            "detail": "" if ok else f"nothing is listening on {target}",
        }

    url = f"http://{host}:{port}{spec['path']}"
    body, status = telemetry._http_text(url, timeout=timeout)
    result = {"ok": False, "target": url, "http_status": status, "checked_at": checked_at, "detail": ""}
    if body is None:
        result["detail"] = (f"{url} answered HTTP {status}" if status
                            else f"no answer from {url}")
        return result
    try:
        payload = json.loads(body)
    except ValueError:
        result["detail"] = f"{url} did not answer with JSON"
        return result

    field = spec.get("expect_field")
    if field and (not isinstance(payload, dict) or payload.get(field[0]) != field[1]):
        result["detail"] = f"{url} did not report {field[0]}={field[1]}"
        return result

    result["ok"] = True
    return result


# ---------------------------------------------------------------------------
# expected state
# ---------------------------------------------------------------------------

def read_expectations(path: Path | None = None) -> dict:
    path = path or EXPECTATIONS_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def record_expectation(name: str, expected: str, source: str = "operator",
                       path: Path | None = None) -> dict:
    """Record that a service is meant to be on or off. `auto` forgets it again."""
    path = path or EXPECTATIONS_FILE
    data = read_expectations(path)
    if expected == "auto":
        data.pop(name, None)
    elif expected in {"on", "off"}:
        data[name] = {"expected": expected, "at": int(time.time()), "source": source}
    else:
        raise ValueError(f"expected must be on, off or auto, not {expected!r}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except OSError:
        # An unwritable config dir should not fail a service action; the
        # expectation is an annotation, not the action itself.
        pass
    return data


def expectation_for(name: str, env: dict, expectations: dict) -> str:
    """`on`, `off`, or `unspecified` when nobody has said."""
    flag = ENABLED_FLAGS.get(name)
    if flag and str(env.get(flag, "")).strip().lower() == "off":
        return "off"
    entry = expectations.get(name)
    if isinstance(entry, dict) and entry.get("expected") in {"on", "off"}:
        return entry["expected"]
    return "unspecified"


# ---------------------------------------------------------------------------
# folding it together
# ---------------------------------------------------------------------------

def collect(env: dict, statuses: dict, probes: dict | None = None,
            expectations: dict | None = None) -> dict:
    """One health entry per service, from unit state, probes and upstreams.

    `statuses` is the map `/api/status` already computes, so this adds no
    subprocesses. `probes` is the background sweep's cache; an empty one means
    the first sweep has not landed yet, and every service falls back to being
    judged on its unit state alone rather than being reported as broken.
    """
    probes = probes or {}
    expectations = read_expectations() if expectations is None else expectations
    resolved: dict[str, dict] = {}
    resolving: set[str] = set()

    def resolve(name: str) -> dict:
        if name in resolved:
            return resolved[name]
        if name in resolving:
            # No cycles exist in SERVICE_DEPENDENCIES, but a future edit that
            # introduced one should not hang the status poll.
            return {"state": STATE_UNKNOWN, "unit": statuses.get(name, STATE_UNKNOWN)}
        resolving.add(name)

        unit = statuses.get(name, STATE_UNKNOWN)
        expected = expectation_for(name, env, expectations)
        probe_result = probes.get(name)
        upstreams = []
        for group in SERVICE_DEPENDENCIES.get(name, []):
            states = {member: resolve(member)["state"] for member in group
                      if member in statuses}
            upstreams.append({
                "any_of": list(group),
                "ok": any(state == STATE_ACTIVE for state in states.values()),
                "states": states,
            })

        entry = {
            "state": unit,
            "unit": unit,
            "expected": expected,
            "reason": "",
            "probe": probe_result,
            "upstreams": upstreams,
            "checked_at": (probe_result or {}).get("checked_at"),
        }

        if unit == STATE_FAILED:
            entry["reason"] = "the unit failed — check its logs"
        elif unit == STATE_ACTIVE:
            unmet = [group for group in upstreams if not group["ok"]]
            if probe_result is not None and not probe_result["ok"]:
                entry["state"] = STATE_DEGRADED
                entry["reason"] = f"running, but {probe_result['detail']}"
            elif unmet:
                entry["state"] = STATE_DEGRADED
                entry["reason"] = "running, but " + _upstream_reason(unmet)
        elif unit == STATE_INACTIVE:
            if expected == "on":
                entry["reason"] = "expected to be running"
            else:
                entry["state"] = STATE_STOPPED
                entry["reason"] = ("turned off in the configuration" if expected == "off"
                                   else "not running")

        resolving.discard(name)
        resolved[name] = entry
        return entry

    for name in statuses:
        resolve(name)
    return resolved


def _upstream_reason(unmet: list[dict]) -> str:
    parts = []
    for group in unmet:
        members = group["any_of"]
        listed = members[0] if len(members) == 1 else " or ".join(members)
        parts.append(listed)
    joined = "; ".join(parts)
    plural = "are" if len(unmet) > 1 or len(unmet[0]["any_of"]) > 1 else "is"
    return f"upstream {joined} {plural} not running"


# ---------------------------------------------------------------------------
# background sweep
# ---------------------------------------------------------------------------

class Prober:
    """Probes every running service on its own cadence, off the request path."""

    def __init__(self, interval: float = SWEEP_INTERVAL_SECONDS):
        self.interval = interval
        self._lock = threading.Lock()
        self._results: dict[str, dict] = {}
        self._thread: threading.Thread | None = None
        self._tasks: list = []

    def sweep(self, env: dict, statuses: dict) -> dict:
        """Probe the services that are up. Synchronous, so tests can call it."""
        results = {}
        for name, status in statuses.items():
            if status != STATE_ACTIVE or name not in SERVICE_PROBES:
                continue
            try:
                result = probe(name, env)
            except Exception as exc:
                result = {"ok": False, "target": "", "http_status": None,
                          "checked_at": time.time(), "detail": f"probe failed: {exc}"}
            if result is not None:
                results[name] = result
        with self._lock:
            self._results = results
        return results

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._results)

    def add_task(self, task) -> None:
        """Run `task()` after each sweep. Used for the opt-in lease reaper."""
        self._tasks.append(task)

    def start(self, env_reader, status_reader) -> None:
        """Idempotent. Started on first request so importing the app spawns nothing."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._loop, args=(env_reader, status_reader),
            name="health-prober", daemon=True)
        self._thread.start()

    def _loop(self, env_reader, status_reader) -> None:
        while True:
            try:
                self.sweep(env_reader(), status_reader())
            except Exception:
                # A sweep that raises must not end the thread; the next one may
                # well succeed, and a dead prober silently freezes every card.
                pass
            for task in list(self._tasks):
                try:
                    task()
                except Exception:
                    pass
            time.sleep(self.interval)


PROBER = Prober()


def pid_alive(pid) -> bool:
    """Whether a recorded PID still exists. Used by the lease reaper too."""
    try:
        return Path(f"/proc/{int(pid)}").exists()
    except (TypeError, ValueError):
        return False


__all__ = [
    "PROBER", "Prober", "SERVICE_DEPENDENCIES", "SERVICE_PROBES", "ENABLED_FLAGS",
    "STATE_ACTIVE", "STATE_DEGRADED", "STATE_FAILED", "STATE_INACTIVE",
    "STATE_STOPPED", "STATE_UNKNOWN", "collect", "endpoint_for", "expectation_for",
    "pid_alive", "probe", "read_expectations", "record_expectation", "tcp_port_open",
    "dependency_units",
]
