#!/usr/bin/env python3
"""
The HTTP surface of the control API.

`app.py` serves this blueprint from a *third* Flask app on its own port. Three
apps rather than one with careful checks, because the property worth having is
structural: the read-only listener cannot serve a write because it never
learned the route, and this one cannot serve the whole stack's state because it
never learned that one. A check that has to be right on every route is a check
that will eventually be missed on one.

Registered on neither the manager app nor the state app.

Three deliberate differences from `routes/public.py`, each of which follows
from this port being able to stop a backend:

1. **No `?token=`.** The read API accepts one because `EventSource` cannot set
   headers, and there is no stream here. A token in a query string is a token
   in werkzeug's access log and in every reverse proxy in front of it.
2. **An unset token is 503, not open.** On 8078 an empty token is a
   configuration; here it is a mistake, and defaulting to open would be the
   wrong way to resolve it.
3. **A non-loopback bind with no token refuses to listen at all**, where the
   read API only warns.
"""

from __future__ import annotations

from flask import Blueprint, current_app, jsonify, request

import config_env
import control_api
import public_api

bp = Blueprint("control_api", __name__)

# Set by `app.py` at registration, for the reason `routes/public.py` states:
# Flask routes take no arguments and everything here needs it.
PROVIDERS: control_api.ControlProviders | None = None

LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost", ""}


def configure(providers: control_api.ControlProviders) -> control_api.ControlProviders:
    global PROVIDERS
    PROVIDERS = providers
    return providers


def control_settings() -> dict:
    """The `LLM_CONTROL_*` block, read fresh so a config change takes effect.

    Off by default. An upgrade must never open a write port on a host whose
    operator did not ask for one.
    """
    env = config_env.read_env()
    host = str(env.get("LLM_CONTROL_HOST") or "127.0.0.1").strip()
    return {
        "enabled": str(env.get("LLM_CONTROL_ENABLED") or "off").strip().lower() == "on",
        "host": host,
        "port": int(str(env.get("LLM_CONTROL_PORT") or "8079").strip() or 8079),
        "token": str(env.get("LLM_CONTROL_TOKEN") or "").strip(),
        "allow_secrets": str(env.get("LLM_CONTROL_ALLOW_SECRETS") or "off").strip().lower() == "on",
    }


def refuses_to_bind(settings: dict) -> str:
    """Why this listener will not start, or "" if it will.

    A refusal rather than a warning. `routes/public.py:bind_warning` says its
    piece and binds anyway, because an open read-only API on a tailnet is a
    choice someone can reasonably make. There is no equivalent reading of an
    open port that can stop a backend and rewrite the env file.
    """
    if settings["host"] not in LOOPBACK_HOSTS and not settings["token"]:
        return (f"LLM_CONTROL_HOST is {settings['host']} and LLM_CONTROL_TOKEN is unset. "
                f"A control API reachable off-box with no token would let anything "
                f"that can route to it stop a backend.")
    return ""


def _bearer_token() -> str:
    header = request.headers.get("Authorization", "")
    return header[7:].strip() if header.lower().startswith("bearer ") else ""


@bp.before_request
def require_control_token():
    if request.endpoint == "control_api.api_control_v1_health":
        return None
    if not current_app.config.get("CONTROL_API_ENFORCE_TOKEN"):
        return None
    expected = control_settings()["token"]
    if not expected:
        return jsonify(error="control_disabled",
                       detail="LLM_CONTROL_TOKEN is unset; this listener accepts nothing."), 503
    if not public_api.token_matches(_bearer_token(), expected):
        return jsonify(error="unauthorized",
                       detail="Send Authorization: Bearer <token>."), 401
    return None


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------

@bp.route("/api/control/v1/health")
def api_control_v1_health():
    """Unauthenticated liveness, the one exemption.

    A fleet has to be able to tell "unreachable" from "rejected my token", and
    it should not need a credential to learn that a host is up.
    """
    return jsonify(ok=True, api_version=control_api.API_VERSION)


@bp.route("/api/control/v1/schema")
def api_control_v1_schema():
    return jsonify(control_api.schema(control_settings()))


@bp.route("/api/control/v1/config")
def api_control_v1_config():
    env = PROVIDERS.read_env()
    return jsonify(config=control_api.masked_config(env),
                   config_etag=control_api.config_etag(env))


@bp.route("/api/control/v1/config/fields")
def api_control_v1_config_fields():
    return jsonify(control_api.field_schema())


@bp.route("/api/control/v1/config/preflight", methods=["POST"])
def api_control_v1_config_preflight():
    updates = request.get_json(silent=True)
    if not isinstance(updates, dict):
        return jsonify(ok=False, error="Expected JSON object"), 400
    return jsonify(PROVIDERS.preflight(updates))


@bp.route("/api/control/v1/config", methods=["POST"])
def api_control_v1_config_save():
    updates = request.get_json(silent=True)
    if not isinstance(updates, dict):
        return jsonify(ok=False, error="Expected JSON object"), 400

    # Optimistic concurrency. Two writers over an unlocked read-modify-write
    # lose one of the two saves, and a hub cannot see the other writer.
    expected = request.headers.get("If-Match", "").strip()
    if expected:
        current = control_api.config_etag(PROVIDERS.read_env())
        if expected != current:
            return jsonify(ok=False, error="stale_config", config_etag=current,
                           detail="The configuration changed since you read it."), 409

    settings = control_settings()
    kept, dropped = control_api.drop_secrets(updates, settings["allow_secrets"])
    forced = str(request.args.get("force", "")).lower() in {"1", "true", "yes", "on"}
    payload, status = PROVIDERS.save_config(kept, forced)
    payload["dropped_secret_keys"] = dropped
    if status == 200:
        payload["config_etag"] = control_api.config_etag(PROVIDERS.read_env())
    return jsonify(payload), status


@bp.route("/api/control/v1/saved-configs")
def api_control_v1_saved_configs():
    return jsonify(PROVIDERS.saved_configs())


@bp.route("/api/control/v1/saved-configs/<name>/apply", methods=["POST"])
def api_control_v1_saved_config_apply(name):
    body = request.get_json(silent=True) or {}
    result = PROVIDERS.apply_saved_config(name, bool(body.get("launch")))
    return jsonify(result), (200 if result.get("ok") else 400)


@bp.route("/api/control/v1/services")
def api_control_v1_services():
    return jsonify(control_api.services(PROVIDERS))


@bp.route("/api/control/v1/services/<name>/<action>", methods=["POST"])
def api_control_v1_service_action(name, action):
    payload, status = PROVIDERS.service_action(name, action)
    return jsonify(payload), status
