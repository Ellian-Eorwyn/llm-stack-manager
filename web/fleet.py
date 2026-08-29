#!/usr/bin/env python3
"""Other machines running this manager, and how this one reaches them.

One interface across several hosts: `llms` with its two 3090s, a Mac Studio, an
M1 Pro. Every instance ships this code and any of them can be the one you sit
at, so there is no hub build and no `IS_HUB` flag -- an instance *is* a hub
exactly when its `config/fleet.json` lists peers. A hub-only build would mean
the Mac could not watch the Linux box without a second code path, and the Linux
box could not watch the Mac at all.

Two things about the shape here are load-bearing.

**Every call goes through `fetch`.** It is the only place this module touches
the network, which is what makes a fleet testable without a fleet: a spoke in
the test suite is two Flask `test_client()`s and one `patch.object`. `fleet` is
in `ModuleBoundaryTests.BEHAVIOUR_MODULES` for that reason, and one
`from fleet import fetch` anywhere would defeat it silently.

**Nothing here raises.** A page that shows five machines must not fail because
one of them is off. Every transport failure -- DNS, refused, timeout, a body
that is not JSON, a 401 -- comes back as data with a reason, and the poller
keeps the last good payload beside it so the card goes stale rather than blank.
"""

from __future__ import annotations

import json
import re
import threading
import time
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

import core
import public_api

#: An operator-chosen slug, assigned when the host is added.
#:
#: Not `platform.node()`. That is a value the *remote* controls -- it changes
#: with DNS, with mDNS, with a re-image -- and keying the registry on it would
#: make this hub's idea of which machine is which depend on what the other
#: machine says it is called. The reported hostname is kept as
#: `expected_hostname` and checked, which is a different and better use for it.
HOST_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}$")

DEFAULT_TIMEOUT_SECONDS = 4.0
#: A restart loads a model. Nothing here uses it yet -- the write half does.
SERVICE_ACTION_TIMEOUT_SECONDS = 45.0

DEFAULT_POLL_SECONDS = 5
MIN_POLL_SECONDS = 2
MAX_POLL_SECONDS = 60

#: Re-read a peer's `/api/v1/schema` this often. Version negotiation is not
#: free and does not change between releases.
SCHEMA_TTL_SECONDS = 300

#: The sections a fleet card renders. `config` and `deployment` are left out of
#: the fast path deliberately: they are the expensive half of a snapshot and
#: nothing on a monitoring card reads them.
CARD_SECTIONS = ("stack", "gpus", "backends", "services", "host", "router", "alerts")

_FIELDS = ("id", "label", "host", "scheme", "read_port", "control_port",
           "expected_hostname", "enabled", "control")
_SECRET_FIELDS = ("token", "control_token")


# ---------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------

def _blank() -> dict:
    return {"version": 1, "hosts": []}


def load() -> dict:
    """The peer list, or an empty one. Never raises: a corrupt file must not
    take down the manager that is the only way to fix it."""
    try:
        if core.FLEET_FILE.exists():
            data = json.loads(core.FLEET_FILE.read_text())
            if isinstance(data, dict) and isinstance(data.get("hosts"), list):
                return data
    except Exception:
        pass
    return _blank()


def save(registry: dict) -> None:
    core.FLEET_FILE.parent.mkdir(parents=True, exist_ok=True)
    core.FLEET_FILE.write_text(json.dumps(registry, indent=2) + "\n")


def hosts(enabled_only: bool = True) -> list[dict]:
    return [h for h in load().get("hosts", [])
            if not enabled_only or h.get("enabled", True)]


def get(host_id: str) -> dict | None:
    return next((h for h in load().get("hosts", []) if h.get("id") == host_id), None)


def redacted(host: dict) -> dict:
    """A peer as the UI may see it: every field except the credentials.

    Tokens are write-only through this API for the same reason they are on the
    control listener -- you can set one, you cannot read one back. `has_token`
    is the part a form needs.
    """
    out = {field: host.get(field) for field in _FIELDS}
    for field in _SECRET_FIELDS:
        out[f"has_{field}"] = bool(str(host.get(field) or "").strip())
    return out


def validate(host: dict, existing_ids: set[str], replacing: str = "") -> str:
    """Why this entry cannot be saved, or "" if it can."""
    host_id = str(host.get("id") or "").strip()
    if not HOST_ID_RE.match(host_id):
        return ("id must be lowercase letters, digits and hyphens, "
                "starting with a letter or digit, up to 32 characters")
    if host_id != replacing and host_id in existing_ids:
        return f"a host called {host_id} is already in the fleet"
    if not str(host.get("host") or "").strip():
        return "host is required"
    for field in ("read_port", "control_port"):
        try:
            port = int(host.get(field) or 0)
        except (TypeError, ValueError):
            return f"{field} must be a number"
        if not 1 <= port <= 65535:
            return f"{field} must be a port number"
    return ""


def base_url(host: dict, port_field: str = "read_port") -> str:
    scheme = str(host.get("scheme") or "http").strip() or "http"
    return f"{scheme}://{host.get('host')}:{host.get(port_field)}"


# ---------------------------------------------------------------------------
# the one place this module touches the network
# ---------------------------------------------------------------------------

def fetch(url: str, token: str = "", method: str = "GET", payload=None,
          timeout: float = DEFAULT_TIMEOUT_SECONDS,
          headers: dict | None = None) -> tuple[int, dict, str]:
    """One HTTP call to a peer. Returns (status, body, error).

    Not `core.http_json`: that raises on 4xx and has no way to return a status,
    and a peer's 401 is *data* here -- the hub has to render "this host
    rejected the token" rather than throw. `core.http_multipart` already
    returns a status for the same reason.

    stdlib only. `web/requirements.txt` has no `requests` and keeping it that
    way is stated policy.

    Never raises. Status 0 means the request did not complete, and the third
    element says why in words an operator can act on.
    """
    data = None
    request_headers = dict(headers or {})
    if token:
        request_headers["Authorization"] = f"Bearer {token}"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    request = urlrequest.Request(url, data=data, method=method, headers=request_headers)
    try:
        with urlrequest.urlopen(request, timeout=timeout) as response:
            return response.status, _decode(response.read()), ""
    except urlerror.HTTPError as exc:
        # A status with a body: the peer answered, it just said no.
        try:
            return exc.code, _decode(exc.read()), ""
        except Exception:
            return exc.code, {}, str(exc.reason or exc)
    except urlerror.URLError as exc:
        reason = getattr(exc, "reason", exc)
        if isinstance(reason, TimeoutError):
            return 0, {}, f"timed out after {timeout:g}s"
        return 0, {}, f"unreachable: {reason}"
    except TimeoutError:
        return 0, {}, f"timed out after {timeout:g}s"
    except Exception as exc:
        return 0, {}, f"{type(exc).__name__}: {exc}"


def _decode(body: bytes) -> dict:
    if not body:
        return {}
    try:
        parsed = json.loads(body.decode("utf-8"))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {"data": parsed}


def call(host: dict, path: str, port_field: str = "read_port",
         token_field: str = "token", **kwargs) -> tuple[int, dict, str]:
    """`fetch` against a peer, with its address and credential filled in."""
    return fetch(base_url(host, port_field) + path,
                 str(host.get(token_field) or ""), **kwargs)


# ---------------------------------------------------------------------------
# what a hub asks a peer
# ---------------------------------------------------------------------------

def probe(host: dict) -> dict:
    """One round trip to each listener, for the add form and the fleet page.

    Run before a peer is saved, so a typo is caught while the operator is
    looking at it rather than as a dead card later.
    """
    read_status, read_body, read_error = call(host, "/api/v1/health")
    control_status, _control_body, control_error = call(
        host, "/api/control/v1/health", port_field="control_port",
        token_field="control_token")
    reported = ""
    if read_status == 200:
        snap_status, snap, _ = call(host, "/api/v1/snapshot?include=stack")
        if snap_status == 200:
            reported = str(snap.get("stack", {}).get("hostname") or "")
    return {
        "read": {"ok": read_status == 200, "status": read_status,
                 "error": read_error or _rejected(read_status)},
        "control": {"ok": control_status == 200, "status": control_status,
                    "error": control_error or _rejected(control_status)},
        "api_version": str(read_body.get("api_version") or ""),
        "hostname": reported,
        "hostname_matches": (not host.get("expected_hostname")
                             or reported == host.get("expected_hostname")),
    }


def _rejected(status: int) -> str:
    if status == 401:
        return "this host rejected the token"
    if status == 503:
        return "reachable, but its control API has no token set"
    if status and status != 200:
        return f"answered {status}"
    return ""


def schema_of(host: dict) -> dict:
    status, body, error = call(host, "/api/v1/schema")
    return body if status == 200 else {"error": error or _rejected(status)}


def compatible(local: str, remote: str) -> bool:
    """Whether a controller speaking `local` may act on a host speaking `remote`.

    Majors only, so an additive change stays permissive by default. A mismatch
    means refusing to *control* with a reason -- monitoring still works, and a
    form that half works is worse than a form that says why it is disabled.
    """
    try:
        return remote.split(".")[0] == local.split(".")[0]
    except Exception:
        return False


def sections_for(host_schema: dict) -> str:
    """The `include=` a peer will actually accept.

    `resolve_sections` 400s on a section it does not know, which would blank
    the whole card rather than drop one panel. So a hub that has learned a new
    section asks an older peer only for the ones that peer lists.
    """
    known = host_schema.get("sections")
    if not isinstance(known, list) or not known:
        return ",".join(CARD_SECTIONS)
    return ",".join(s for s in CARD_SECTIONS if s in known) or "stack"


def snapshot_of(host: dict, host_schema: dict | None = None) -> tuple[dict, str]:
    """(snapshot, error) for one peer."""
    include = sections_for(host_schema or {})
    status, body, error = call(host, f"/api/v1/snapshot?include={include}")
    if status == 200:
        return body, ""
    return {}, error or _rejected(status) or "no snapshot"


# ---------------------------------------------------------------------------
# the poller
# ---------------------------------------------------------------------------

def poll_seconds(env: dict) -> int:
    try:
        value = int(str(env.get("LLM_FLEET_POLL_SECONDS") or DEFAULT_POLL_SECONDS).strip())
    except ValueError:
        value = DEFAULT_POLL_SECONDS
    return max(MIN_POLL_SECONDS, min(MAX_POLL_SECONDS, value))


class FleetCache:
    """One poller for every peer, whatever is watching.

    The naive shape -- fetch on the request thread -- makes the page as slow as
    the slowest machine and turns one host being off into a four-second stall
    on every poll from every open tab. Same argument as
    `public_api.Broadcaster`, same discipline as `deploy.DriftWatcher`: the
    request thread only ever reads the cache.

    Inside a tick every host is fetched on its own thread and joined against a
    deadline, so a hung peer cannot delay the others and cannot stretch the
    tick. Every host always appears in the output -- with its last good payload
    and how old it is when the current attempt failed -- because a card that
    goes stale says more than a card that disappears.

    Polling rather than each peer's `/api/v1/events`: an SSE consumer needs
    reconnect, backoff and dedupe against the snapshot it already holds, and
    `Broadcaster` makes the *peer* collect for as long as the connection is
    held whether or not anyone is looking. Polling costs a peer one assembly
    per tick and nothing at all when the hub is down. Worth revisiting past
    roughly eight hosts.
    """

    def __init__(self, read_env, interval: float | None = None):
        self._read_env = read_env
        self._interval = interval
        self._lock = threading.Lock()
        self._state: dict[str, dict] = {}
        self._schema: dict[str, tuple[float, dict]] = {}
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # -- reading ------------------------------------------------------------

    def snapshot(self) -> list[dict]:
        """Every configured peer, from the cache. Never a network call."""
        with self._lock:
            state = {host_id: dict(entry) for host_id, entry in self._state.items()}
        now = time.time()
        out = []
        for host in hosts():
            entry = state.get(host["id"]) or {}
            payload = entry.get("payload")
            last_ok = entry.get("last_ok_at")
            out.append({
                "id": host["id"],
                "label": host.get("label") or host["id"],
                "ok": bool(entry.get("ok")),
                "error": entry.get("error", "never polled"),
                "checked_at": entry.get("checked_at"),
                "last_ok_at": last_ok,
                "stale_for_seconds": (round(now - last_ok, 1)
                                      if last_ok and not entry.get("ok") else 0),
                "controllable": bool(entry.get("controllable")),
                "control_refused": entry.get("control_refused", ""),
                "payload": payload,
            })
        return out

    def one(self, host_id: str) -> dict | None:
        return next((h for h in self.snapshot() if h["id"] == host_id), None)

    # -- collecting ---------------------------------------------------------

    def _schema_for(self, host: dict) -> dict:
        host_id = host["id"]
        now = time.time()
        with self._lock:
            cached = self._schema.get(host_id)
        if cached and now - cached[0] < SCHEMA_TTL_SECONDS:
            return cached[1]
        fetched = schema_of(host)
        with self._lock:
            self._schema[host_id] = (now, fetched)
        return fetched

    def _collect_one(self, host: dict, out: dict) -> None:
        host_schema = self._schema_for(host)
        payload, error = snapshot_of(host, host_schema)
        remote_version = str(host_schema.get("api_version") or "")
        speaks = compatible(public_api.API_VERSION, remote_version) if remote_version else True
        refused = "" if speaks else (
            f"{host['id']} speaks state API {remote_version}; this manager speaks "
            f"{public_api.API_VERSION}. Update one of them before controlling it.")
        out[host["id"]] = {
            "ok": bool(payload),
            "error": error,
            "payload": payload or None,
            "checked_at": time.time(),
            "controllable": bool(host.get("control")) and speaks,
            "control_refused": refused,
        }

    def collect(self) -> None:
        """One tick. Synchronous, so a test and the fleet page can force it."""
        peers = hosts()
        if not peers:
            with self._lock:
                self._state = {}
            return
        results: dict[str, dict] = {}
        threads = [threading.Thread(target=self._collect_one, args=(host, results),
                                    name=f"fleet-{host['id']}", daemon=True)
                   for host in peers]
        for thread in threads:
            thread.start()
        # Twice the per-request timeout: enough for the schema call and the
        # snapshot behind it, and still bounded whatever the peer does.
        deadline = 2 * DEFAULT_TIMEOUT_SECONDS
        for thread in threads:
            thread.join(timeout=deadline)
        with self._lock:
            for host in peers:
                entry = results.get(host["id"])
                previous = self._state.get(host["id"], {})
                if entry is None:
                    entry = {"ok": False, "error": "did not answer in time",
                             "payload": None, "checked_at": time.time(),
                             "controllable": False, "control_refused": ""}
                if entry["ok"]:
                    entry["last_ok_at"] = entry["checked_at"]
                else:
                    entry["payload"] = entry["payload"] or previous.get("payload")
                    entry["last_ok_at"] = previous.get("last_ok_at")
                self._state[host["id"]] = entry

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.collect()
            except Exception:
                # A poller that dies takes the whole page with it, silently.
                pass
            self._stop.wait(self._interval or poll_seconds(self._read_env()))

    def ensure_running(self) -> None:
        """Start polling, once, and only when there is something to poll."""
        if self._thread and self._thread.is_alive():
            return
        if not hosts():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="fleet", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()


# ---------------------------------------------------------------------------
# adapters
# ---------------------------------------------------------------------------

def status_from_snapshot(payload: dict) -> dict:
    """A peer's snapshot in the shape the services panel already reads.

    `/api/status` is a UI payload and does not exist on the read-only API,
    which is deliberate: `public_api` exists so the external surface is not a
    set of shapes that change whenever a page does. So the adapter lives here,
    on the hub, rather than a page-shaped endpoint being added to every peer.

    One function with one strong test -- its keys must match what the local
    `/api/status` returns for the same fixture -- is the price of leaving
    seventy-eight `fetchJSON` call sites alone.
    """
    services = payload.get("services") or []
    statuses = {service["name"]: service.get("state", "unknown") for service in services}
    # The snapshot calls it `unit_state`, because "state" there is the health
    # verdict and the two are genuinely different -- a unit can be `active` and
    # the service `degraded`. The panel reads `unit`, so the rename is undone
    # here rather than in the panel.
    health = {
        service["name"]: {
            "state": service.get("state", "unknown"),
            "unit": service.get("unit_state", ""),
            "expected": service.get("expected", ""),
            "reason": service.get("reason", ""),
            "restarts": service.get("restarts", 0),
            "probe": service.get("probe", {}),
            "upstreams": service.get("upstreams", []),
            "checked_at": service.get("checked_at"),
        }
        for service in services
    }
    # `/api/status` reads its contexts from the env, so it has an entry for
    # every configured backend including the stopped ones. A snapshot carries
    # the running geometry instead, so a remote host reports context for the
    # backends telemetry actually reached -- fewer entries, and the ones there
    # are are what the backend is really serving rather than what it was told.
    contexts = {}
    for backend in payload.get("backends") or []:
        rollup = backend.get("context")
        unit = backend.get("unit")
        if not unit or not isinstance(rollup, dict):
            continue
        total = rollup.get("configured_total") or rollup.get("n_ctx_total")
        per_slot = rollup.get("configured_per_slot") or rollup.get("n_ctx_per_slot")
        if not total:
            continue
        contexts[unit] = {
            "total_context": total,
            "per_slot_context": per_slot,
            "slots": (round(total / per_slot) if per_slot else rollup.get("slots_total") or 1),
        }
    return {
        "services": statuses,
        "health": health,
        "gpus": payload.get("gpus") or [],
        "contexts": contexts,
        "deployment": payload.get("deployment") or {},
    }
