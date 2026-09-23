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

The write half sits behind two gates the read half does not have: a peer must
be marked `control` in the registry, and the two majors must agree. Both refuse
with a reason rather than sending a form that half works.
"""

from __future__ import annotations

from urllib.parse import quote

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


def _controllable(host_id: str):
    """(host, error response) for a peer this hub may write to.

    Two gates, both opt-in and both refusing with a reason:

    `control` is per peer and defaults false, so adding a machine to watch it
    never grants the ability to stop its backends. The version gate is the one
    `fleet.compatible` describes -- a hub whose major differs would render a
    form built from its own field list against a host that means something else
    by those keys.

    A peer the poller has not reached yet is *not* refused here. The control
    listener authenticates every request itself and answers 401 or 503 on its
    own behalf; guessing on this side would only turn a working write into a
    confusing local error.
    """
    host, missing = _known(host_id)
    if missing:
        return None, missing
    if not host.get("control"):
        return None, (jsonify(
            ok=False,
            error=(f"{host_id} is monitored but not controlled. Enable control "
                   f"for it in the host list first.")), 403)
    entry = CACHE.one(host_id) if CACHE else None
    refused = str((entry or {}).get("control_refused") or "")
    if refused:
        return None, (jsonify(ok=False, error=refused), 409)
    return host, None


def _control(host, path, **kwargs):
    """One call to a peer's control listener, as a Flask response."""
    status, body, error = fleet.call(host, path, port_field="control_port",
                                     token_field="control_token", **kwargs)
    if not status:
        return jsonify(ok=False, error=error or "no answer"), 502
    return jsonify(body), status


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
    platform-independent. `/api/v1/logs/raw` returns journald or launchd-file
    lines verbatim, whose shape depends on the peer, so it is not proxied.
    """
    host, missing = _known(host_id)
    if missing:
        return missing
    status, body, error = fleet.call(host, "/api/v1/logs?" + request.query_string.decode())
    return (jsonify(body), status) if status else (jsonify(ok=False, error=error), 502)


# ---------------------------------------------------------------------------
# proxied writes
# ---------------------------------------------------------------------------
#
# Each mirrors the local path it stands in for, so `fleet.js` can rewrite
# `/api/config` to `/api/fleet/<id>/config` and leave every call site alone.
# `/config/fields` is the exception with no local twin: this page renders its
# own form from Jinja and only needs a field list when the form belongs to
# somebody else.

@bp.route("/api/fleet/<host_id>/config")
def api_fleet_config(host_id):
    host, refused = _controllable(host_id)
    if refused:
        return refused
    return _control(host, "/api/control/v1/config")


@bp.route("/api/fleet/<host_id>/config/fields")
def api_fleet_config_fields(host_id):
    """The peer's own field list, for rendering a form this hub does not have."""
    host, refused = _controllable(host_id)
    if refused:
        return refused
    return _control(host, "/api/control/v1/config/fields")


@bp.route("/api/fleet/<host_id>/config/preflight", methods=["POST"])
def api_fleet_config_preflight(host_id):
    host, refused = _controllable(host_id)
    if refused:
        return refused
    return _control(host, "/api/control/v1/config/preflight",
                    method="POST", payload=request.get_json(silent=True) or {})


@bp.route("/api/fleet/<host_id>/config", methods=["POST"])
def api_fleet_config_save(host_id):
    """A remote save, with `ignored_keys` treated as a failure.

    `allowed_config_keys` unions in whatever the *target's* env file holds, so
    what is writable is a fact about the target and not about this hub. A key
    the peer does not know is dropped silently and the save returns 200 -- which
    is indistinguishable from having worked. A hub is exactly where that
    happens: it renders a form from a field list that may be a version ahead,
    and the operator watching a green toast has no way to tell that the setting
    they changed went nowhere.

    So a non-empty `ignored_keys` is reported as a 409 even though the peer
    called it a success. Whatever the peer *did* write stays written -- this
    does not roll anything back -- and the names are handed back so the message
    can say which settings did not land.
    """
    host, refused = _controllable(host_id)
    if refused:
        return refused
    headers = {}
    if request.headers.get("If-Match"):
        headers["If-Match"] = request.headers["If-Match"]
    path = "/api/control/v1/config"
    if str(request.args.get("force", "")).lower() in {"1", "true", "yes", "on"}:
        path += "?force=1"
    status, body, error = fleet.call(
        host, path, port_field="control_port", token_field="control_token",
        method="POST", payload=request.get_json(silent=True) or {},
        headers=headers or None)
    if not status:
        return jsonify(ok=False, error=error or "no answer"), 502
    ignored = body.get("ignored_keys") if isinstance(body, dict) else None
    if status == 200 and ignored:
        return jsonify({**body, "ok": False, "error": (
            f"{host_id} does not know "
            f"{'these settings' if len(ignored) > 1 else 'this setting'}: "
            f"{', '.join(ignored)}. Nothing was written for "
            f"{'them' if len(ignored) > 1 else 'it'}.")}), 409
    return jsonify(body), status


@bp.route("/api/fleet/<host_id>/saved-configs")
def api_fleet_saved_configs(host_id):
    host, refused = _controllable(host_id)
    if refused:
        return refused
    return _control(host, "/api/control/v1/saved-configs")


@bp.route("/api/fleet/<host_id>/saved-configs/<name>/apply", methods=["POST"])
def api_fleet_saved_config_apply(host_id, name):
    host, refused = _controllable(host_id)
    if refused:
        return refused
    return _control(host, f"/api/control/v1/saved-configs/{quote(name, safe='')}/apply",
                    method="POST", payload=request.get_json(silent=True) or {})


@bp.route("/api/fleet/<host_id>/services")
def api_fleet_services(host_id):
    host, refused = _controllable(host_id)
    if refused:
        return refused
    return _control(host, "/api/control/v1/services")


@bp.route("/api/fleet/<host_id>/service/<name>/<action>", methods=["POST"])
def api_fleet_service_action(host_id, name, action):
    """Singular `service`, matching the local route this stands in for.

    The control listener spells it `services`; the page calls
    `/api/service/<name>/<action>`. The rewrite is a prefix substitution, so
    this side has to match the page, and the translation happens here.
    """
    host, refused = _controllable(host_id)
    if refused:
        return refused
    return _control(host,
                    f"/api/control/v1/services/{quote(name, safe='')}/{quote(action, safe='')}",
                    method="POST")
