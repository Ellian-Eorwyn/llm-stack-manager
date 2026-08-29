#!/usr/bin/env python3
"""
The hub half of the fleet: this manager's own port, serving other machines.

Registered on the manager app (8077) and on neither of the others. The peer
list holds credentials for every machine in the fleet, so the ability to edit
it must not exist on a port another machine can reach -- a control channel that
can be told to trust a new peer is a lateral-movement primitive, and there is
no reason to build one.

**Proxied paths are a whitelist, never a forwarder.** Each is a fixed rule with
a fixed destination. A generic `/api/fleet/<id>/<path:rest>` on this
unauthenticated port would let anything that can reach 8077 issue authenticated
requests to every machine in the fleet.

Reads only, for now. The write half -- config saves and service actions against
a remote host -- is its own commit, on top of a read path that has been used.
"""

from __future__ import annotations

from flask import Blueprint, jsonify, request

import fleet

bp = Blueprint("fleet_api", __name__)

CACHE: "fleet.FleetCache | None" = None


def configure(cache) -> None:
    global CACHE
    CACHE = cache


def _known(host_id: str):
    """(host, error response). A peer this hub has never heard of is a 404."""
    host = fleet.get(host_id)
    if host is None:
        return None, (jsonify(ok=False, error=f"no host called {host_id}"), 404)
    return host, None


# ---------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------

@bp.route("/api/fleet")
def api_fleet():
    """Every peer, from the cache. Never a network call on the request thread."""
    CACHE.ensure_running()
    return jsonify(hosts=CACHE.snapshot())


@bp.route("/api/fleet/hosts")
def api_fleet_hosts_list():
    return jsonify(hosts=[fleet.redacted(h) for h in fleet.hosts(enabled_only=False)])


@bp.route("/api/fleet/hosts", methods=["POST"])
def api_fleet_hosts_add():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify(ok=False, error="Expected JSON object"), 400
    registry = fleet.load()
    existing = {h.get("id") for h in registry["hosts"]}
    entry = _entry_from(body)
    error = fleet.validate(entry, existing)
    if error:
        return jsonify(ok=False, error=error), 400
    if _is_self(entry):
        return jsonify(ok=False, error="that is this machine"), 400
    registry["hosts"].append(entry)
    fleet.save(registry)
    CACHE.ensure_running()
    return jsonify(ok=True, host=fleet.redacted(entry))


@bp.route("/api/fleet/hosts/<host_id>", methods=["PUT"])
def api_fleet_hosts_update(host_id):
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify(ok=False, error="Expected JSON object"), 400
    registry = fleet.load()
    index = next((i for i, h in enumerate(registry["hosts"]) if h.get("id") == host_id), None)
    if index is None:
        return jsonify(ok=False, error=f"no host called {host_id}"), 404
    current = registry["hosts"][index]
    entry = _entry_from(body, current)
    error = fleet.validate(entry, {h.get("id") for h in registry["hosts"]}, replacing=host_id)
    if error:
        return jsonify(ok=False, error=error), 400
    registry["hosts"][index] = entry
    fleet.save(registry)
    return jsonify(ok=True, host=fleet.redacted(entry))


@bp.route("/api/fleet/hosts/<host_id>", methods=["DELETE"])
def api_fleet_hosts_delete(host_id):
    registry = fleet.load()
    remaining = [h for h in registry["hosts"] if h.get("id") != host_id]
    if len(remaining) == len(registry["hosts"]):
        return jsonify(ok=False, error=f"no host called {host_id}"), 404
    registry["hosts"] = remaining
    fleet.save(registry)
    return jsonify(ok=True)


@bp.route("/api/fleet/hosts/<host_id>/test", methods=["POST"])
def api_fleet_host_test(host_id):
    """One round trip to each listener, so a typo is caught at add time.

    On the request thread on purpose: the operator is looking at the form and
    waiting for the answer, which is the one case where blocking is what they
    asked for.
    """
    host, missing = _known(host_id)
    if missing:
        return missing
    return jsonify(ok=True, result=fleet.probe(host))


def _entry_from(body: dict, current: dict | None = None) -> dict:
    """A registry entry from a form, keeping a token the form did not resend.

    Tokens are write-only through this API, so the edit form never has the
    current value to send back. An absent field means unchanged; an empty
    string means clear it.
    """
    current = current or {}
    entry = {
        "id": str(body.get("id") or current.get("id") or "").strip().lower(),
        "label": str(body.get("label") or current.get("label") or "").strip(),
        "host": str(body.get("host") or current.get("host") or "").strip(),
        "scheme": str(body.get("scheme") or current.get("scheme") or "http").strip(),
        "read_port": int(body.get("read_port") or current.get("read_port") or 8078),
        "control_port": int(body.get("control_port") or current.get("control_port") or 8079),
        "expected_hostname": str(body.get("expected_hostname")
                                 if body.get("expected_hostname") is not None
                                 else current.get("expected_hostname") or "").strip(),
        "enabled": bool(body.get("enabled", current.get("enabled", True))),
        # Monitor-only until someone says otherwise. `llms` should be watched
        # long before it is driven.
        "control": bool(body.get("control", current.get("control", False))),
    }
    for field in ("token", "control_token"):
        entry[field] = (str(body[field]) if field in body
                        else str(current.get(field) or ""))
    return entry


def _is_self(entry: dict) -> bool:
    """Refuse an entry that points back here.

    Otherwise a fleet route can call a fleet route, and the poller can spend a
    tick waiting on the thread that is running it.
    """
    import config_env
    env = config_env.read_env()
    if str(entry.get("host")) not in {"127.0.0.1", "::1", "localhost"}:
        return False
    return str(entry.get("read_port")) == str(env.get("LLM_API_PORT") or "8078")


# ---------------------------------------------------------------------------
# proxied reads
# ---------------------------------------------------------------------------

@bp.route("/api/fleet/<host_id>/snapshot")
def api_fleet_snapshot(host_id):
    host, missing = _known(host_id)
    if missing:
        return missing
    status, body, error = fleet.call(host, "/api/v1/snapshot?" + request.query_string.decode())
    return (jsonify(body), status) if status else (jsonify(ok=False, error=error), 502)


@bp.route("/api/fleet/<host_id>/status")
def api_fleet_status(host_id):
    """The services panel's own shape, synthesised from the peer's snapshot.

    `/api/status` is a UI payload and deliberately does not exist on the
    read-only API; adding it there would be the exact thing `public_api`'s
    docstring forbids. So the adapter is here, and the page's seventy-eight
    `fetchJSON` call sites do not change.
    """
    cached = CACHE.one(host_id)
    if cached is None:
        return jsonify(ok=False, error=f"no host called {host_id}"), 404
    if not cached.get("payload"):
        return jsonify(ok=False, error=cached.get("error") or "no data yet",
                       stale_for_seconds=cached.get("stale_for_seconds")), 502
    payload = fleet.status_from_snapshot(cached["payload"])
    payload["fleet"] = {k: cached[k] for k in ("ok", "error", "last_ok_at",
                                               "stale_for_seconds")}
    return jsonify(payload)


@bp.route("/api/fleet/<host_id>/logs")
def api_fleet_logs(host_id):
    """Parsed log events, polled rather than streamed.

    `/api/v1/logs` reads the peer's telemetry ring buffer and is
    platform-independent. `/api/v1/logs/raw` shells `journalctl` and is not, so
    it is not proxied at all.
    """
    host, missing = _known(host_id)
    if missing:
        return missing
    status, body, error = fleet.call(host, "/api/v1/logs?" + request.query_string.decode())
    return (jsonify(body), status) if status else (jsonify(ok=False, error=error), 502)
