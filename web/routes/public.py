#!/usr/bin/env python3
"""
The HTTP surface of the read-only state API.

Every rule here is a GET. That is not a convention — it is the security model.
`app.py` serves this blueprint from a second Flask app on its own port, and the
routes that can stop a service or read the env file are not registered on it, so
there is nothing on that port to abuse. The optional token narrows who can read;
the missing routes are what stop anyone writing.

Registered without a `url_prefix`, like every other blueprint in `web/routes/`,
so the paths are exactly what they read as here.

The one thing worth knowing before changing this file: **do not add a POST**.
If remote control is ever wanted it belongs on a separate blueprint with its own
decision behind it, not smuggled in beside the telemetry.
"""

from __future__ import annotations

import json
import queue
import subprocess
import time

from flask import Blueprint, Response, current_app, jsonify, request, stream_with_context

import config_env
import public_api
import telemetry

bp = Blueprint("public_api", __name__)

# Set by `app.py` at registration. Module-level rather than passed per call
# because Flask routes take no arguments but everything here needs it.
PROVIDERS: public_api.Providers | None = None
BROADCASTER: public_api.Broadcaster | None = None

# How long a client waits with nothing to read before the stream sends a comment
# to prove it is still there. Long enough not to be chatter, short enough to beat
# the idle timeout of anything likely to sit in front of this.
HEARTBEAT_SECONDS = 15

LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost", ""}
DEFAULT_RAW_LOG_LINES = 100
MAX_RAW_LOG_LINES = 2000
MAX_LOG_EVENTS = 1000


def configure(providers: public_api.Providers) -> public_api.Broadcaster:
    """Hand the blueprint the `app.py` callables it is allowed to use."""
    global PROVIDERS, BROADCASTER
    PROVIDERS = providers
    BROADCASTER = public_api.Broadcaster(providers, api_settings)
    return BROADCASTER


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------

def api_settings() -> dict:
    """The `LLM_API_*` block, read fresh so a config change takes effect."""
    env = config_env.read_env()
    host = str(env.get("LLM_API_HOST") or "127.0.0.1").strip()
    token = str(env.get("LLM_API_TOKEN") or "").strip()
    events = {part.strip() for part in str(env.get("LLM_API_WEBHOOK_EVENTS") or "").split(",") if part.strip()}
    return {
        "enabled": str(env.get("LLM_API_ENABLED") or "on").strip().lower() == "on",
        "host": host,
        "port": int(str(env.get("LLM_API_PORT") or "8078").strip() or 8078),
        "token": token,
        "allow_origins": [o.strip() for o in str(env.get("LLM_API_ALLOW_ORIGINS") or "").split(",") if o.strip()],
        "interval": env.get("LLM_API_STREAM_INTERVAL"),
        "webhook_url": str(env.get("LLM_API_WEBHOOK_URL") or "").strip(),
        "webhook_events": events,
        "window": telemetry.DEFAULT_WINDOW_SECONDS,
        "bind_warning": bind_warning(host, token),
    }


def bind_warning(host: str, token: str) -> str:
    """Said out loud when the API is reachable off-box with nothing in the way.

    Not an error: an open read-only API on a trusted tailnet is a legitimate
    choice, and it is the configured one by default. But it should never be a
    thing someone discovers later, so it is reported on the stream, in the
    alerts, and on the console at startup.
    """
    if host in LOOPBACK_HOSTS or token:
        return ""
    return (f"The state API is bound to {host} with no LLM_API_TOKEN set — "
            f"anything that can reach this port can read the full stack state.")


# ---------------------------------------------------------------------------
# auth and CORS
# ---------------------------------------------------------------------------

def _request_token() -> str:
    header = request.headers.get("Authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    # Browsers cannot set headers on an EventSource, so the query parameter is
    # the only way a page can authenticate to the stream at all.
    return (request.args.get("token") or "").strip()


@bp.before_request
def require_token():
    if request.method == "OPTIONS" or request.endpoint == "public_api.api_v1_health":
        return None
    # Only the dedicated listener enforces. On the manager's own port this
    # blueprint is as open as everything else already there, and pretending
    # otherwise would imply a protection that port does not have.
    if not current_app.config.get("PUBLIC_API_ENFORCE_TOKEN"):
        return None
    expected = api_settings()["token"]
    if not expected:
        return None
    if _request_token() != expected:
        return jsonify(error="unauthorized",
                       detail="Set Authorization: Bearer <token>, or ?token= for EventSource."), 401
    return None


@bp.after_request
def allow_origin(response):
    origins = api_settings()["allow_origins"]
    origin = request.headers.get("Origin")
    if origins and origin and (origin in origins or "*" in origins):
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
        response.headers["Access-Control-Allow-Headers"] = "Authorization"
    return response


def _snapshot(sections: list[str] | None):
    settings = api_settings()
    window = telemetry.clamp_window(request.args.get("window"), telemetry.DEFAULT_WINDOW_SECONDS)
    return public_api.snapshot(PROVIDERS, sections, window=window,
                               bind_warning=settings["bind_warning"])


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------

@bp.route("/api/v1/health")
def api_v1_health():
    """Liveness only, and deliberately unauthenticated.

    A health check that needs a credential is one that stops being checked.
    It reports nothing about the stack — only that this process is answering.
    """
    return jsonify(ok=True, api_version=public_api.API_VERSION, at=time.time())


@bp.route("/api/v1/schema")
def api_v1_schema():
    return jsonify(public_api.schema())


@bp.route("/api/v1/snapshot")
def api_v1_snapshot():
    sections, unknown = public_api.resolve_sections(request.args.get("include"))
    if unknown:
        return jsonify(error="unknown_section", unknown=unknown,
                       known=list(public_api.SECTIONS)), 400
    return jsonify(_snapshot(sections))


@bp.route("/api/v1/gpu")
def api_v1_gpu():
    return jsonify(_snapshot(["gpus"]))


@bp.route("/api/v1/backends")
def api_v1_backends():
    return jsonify(_snapshot(["backends"]))


@bp.route("/api/v1/services")
def api_v1_services():
    return jsonify(_snapshot(["services"]))


@bp.route("/api/v1/alerts")
def api_v1_alerts():
    return jsonify(_snapshot(["alerts"]))


@bp.route("/api/v1/metrics")
def api_v1_metrics():
    payload = _snapshot(list(public_api.SECTIONS))
    return Response(public_api.render_prometheus(payload),
                    mimetype="text/plain; version=0.0.4; charset=utf-8")


# ---------------------------------------------------------------------------
# logs
# ---------------------------------------------------------------------------

def known_units() -> set[str]:
    """Units this API will read logs for.

    An allow-list because the parameter reaches `journalctl`: without one, a
    unit name is an arbitrary argument to a command run as root.
    """
    env = config_env.read_env()
    units = {service["name"] for service in PROVIDERS.services_table(env)}
    for spec in telemetry.BACKEND_TARGETS:
        units.update(spec["units"])
    units.add(telemetry.ROUTER_UNIT)
    return units


def _requested_units() -> tuple[list[str], list[str]]:
    allowed = known_units()
    asked = [part.strip() for part in (request.args.get("unit") or "").split(",") if part.strip()]
    if not asked:
        env = config_env.read_env()
        targets = telemetry.resolve_targets(env, PROVIDERS.service_status)
        return [t["unit"] for t in targets if t.get("unit")], []
    return [u for u in asked if u in allowed], [u for u in asked if u not in allowed]


@bp.route("/api/v1/logs")
def api_v1_logs():
    units, unknown = _requested_units()
    if unknown:
        return jsonify(error="unknown_unit", unknown=unknown, known=sorted(known_units())), 400

    kinds = {part.strip() for part in (request.args.get("kind") or "").split(",") if part.strip()}
    try:
        limit = min(MAX_LOG_EVENTS, max(1, int(request.args.get("limit", 200))))
    except ValueError:
        limit = 200
    try:
        since = float(request.args["since"]) if request.args.get("since") else None
    except ValueError:
        since = None

    window = telemetry.clamp_window(request.args.get("window"), telemetry.DEFAULT_WINDOW_SECONDS)
    events = public_api.log_events(units, window, kinds or None, since, limit)
    return jsonify(units=units, window_seconds=window, count=len(events), events=events)


@bp.route("/api/v1/logs/raw")
def api_v1_logs_raw():
    units, unknown = _requested_units()
    if unknown:
        return jsonify(error="unknown_unit", unknown=unknown, known=sorted(known_units())), 400
    if not units:
        return jsonify(error="no_unit", detail="No active backend to read; pass ?unit="), 400

    try:
        lines = min(MAX_RAW_LOG_LINES, max(1, int(request.args.get("lines", DEFAULT_RAW_LOG_LINES))))
    except ValueError:
        lines = DEFAULT_RAW_LOG_LINES

    output = {}
    for unit in units:
        # One-shot, never `-f`. A follow here would leave a journalctl per client
        # alive for as long as they cared to hold the socket open.
        try:
            result = subprocess.run(
                ["journalctl", "-u", unit, "-n", str(lines), "--no-pager", "--output=short-iso"],
                capture_output=True, text=True, timeout=15,
            )
            output[unit] = result.stdout.splitlines()
        except (OSError, subprocess.SubprocessError) as exc:
            output[unit] = [f"<failed to read journal: {exc}>"]
    return jsonify(units=units, lines=lines, logs=output)


# ---------------------------------------------------------------------------
# the live stream
# ---------------------------------------------------------------------------

@bp.route("/api/v1/events")
def api_v1_events():
    """Server-sent events: one shared collector, fanned out to every client.

    The interval belongs to the broadcaster, not the connection, so a client
    asking for one second does not make the box collect faster for everyone —
    `?interval=` is honoured as the floor the loop runs at while it is the
    fastest thing connected.
    """
    include, unknown = _event_types(request.args.get("include"))
    if unknown:
        return jsonify(error="unknown_event_type", unknown=unknown,
                       known=sorted(EVENT_TYPES)), 400

    interval = public_api.clamp_interval(
        request.args.get("interval"), public_api.clamp_interval(api_settings().get("interval")))
    subscription = BROADCASTER.subscribe(include, interval)

    def generate():
        try:
            yield _sse("hello", {
                "api_version": public_api.API_VERSION,
                "interval": interval,
                "include": sorted(include) if include else sorted(EVENT_TYPES),
            })
            while True:
                try:
                    event_type, data = subscription.queue.get(timeout=HEARTBEAT_SECONDS)
                except queue.Empty:
                    yield ": ping\n\n"
                    continue
                if subscription.dropped:
                    yield _sse("overrun", {"detail": "client fell behind and was dropped"})
                    return
                yield _sse(event_type, data)
        finally:
            BROADCASTER.unsubscribe(subscription)

    return Response(stream_with_context(generate()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                             "Connection": "keep-alive"})


EVENT_TYPES = {"snapshot", "delta", "alert", "log"}


def _event_types(requested: str | None) -> tuple[list[str] | None, list[str]]:
    if not requested:
        return None, []
    names = [part.strip() for part in requested.split(",") if part.strip()]
    return ([n for n in names if n in EVENT_TYPES],
            [n for n in names if n not in EVENT_TYPES])


def _sse(event_type: str, data) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
