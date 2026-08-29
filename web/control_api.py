#!/usr/bin/env python3
"""What the control API does, with no Flask in it.

The twin of `public_api.py`, and deliberately the same shape: a bundle of
callables the application hands over, and functions that turn them into
payloads. `web/routes/control.py` is the HTTP half.

The split is what makes this testable without a second machine -- a spoke is a
`test_client()` over `create_control_api_app()` -- and it is what keeps the
module-boundary rule satisfied: nothing here imports `app`.

**Where this differs from the read-only API, and why.**

`public_api` answers "what is this stack doing". This answers "change it", and
the three policy differences follow from that:

- An unset token is a refusal, not an open door. On 8078 an empty
  `LLM_API_TOKEN` means "an open read-only API", which is a real and defensible
  configuration on a trusted tailnet. An open listener that can stop
  `llm-a` is not.
- Secrets are write-only. `GET /config` reports that a key is set and never
  what it is set to, and `POST /config` refuses to write one unless
  `LLM_CONTROL_ALLOW_SECRETS` says otherwise.
- Every route is a mutation or the read a mutation needs. There is no snapshot
  here; a controller that wants state reads 8078.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Callable

import config_fields
import platforms
import public_api

#: Bumped with `public_api.API_VERSION`'s major when a controller written for
#: this API would stop working. A hub refuses to *control* a host whose major
#: differs, so this number is a promise rather than a label.
API_VERSION = "1.0"

#: What a secret key looks like. The same compiled pattern the read-only API
#: redacts with, reached through the module so the two cannot drift into
#: disagreeing about what counts -- which is how one of them ends up being the
#: one that leaks `TRANSCRIPT_API_TOKEN`.
SECRET_KEY_RE = public_api.SECRET_KEY_RE


@dataclass
class ControlProviders:
    """The application's own functions, handed over rather than imported.

    Every field is a callable for the reason `public_api.Providers` states:
    binding the function itself captures it at import time and leaves
    `patch.object` with nothing to patch.
    """

    read_env: Callable[[], dict]
    save_config: Callable[[dict, bool], tuple[dict, int]]
    preflight: Callable[[dict], dict]
    service_action: Callable[[str, str], tuple[dict, int]]
    services_table: Callable[[dict], list]
    service_status: Callable[[str], str]
    saved_configs: Callable[[], list]
    apply_saved_config: Callable[[str, bool], dict]


def config_etag(env: dict) -> str:
    """A fingerprint of the values, for optimistic concurrency.

    Of the values and not of the file: comments and key order are not what a
    conflicting write would clobber. `update_env_values` is an unlocked
    read-modify-write over a text file, so two writers -- two browser tabs, or
    a hub and someone sitting at the machine -- can lose one of the two saves.
    A client that sends back the etag it rendered from can be told so.
    """
    canonical = json.dumps({k: str(v) for k, v in sorted(env.items())}, sort_keys=True)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def masked_config(env: dict) -> dict:
    """Every key, with the secrets reported as set rather than as themselves.

    A control API has to serve the whole configuration -- a controller cannot
    edit what it cannot read -- which makes it the one surface where the
    read-only API's section allow-list does not apply. So the secret rule has
    to hold on its own here, and it holds in the strong direction: you can set
    an API key on a remote host and you can never read one back.
    """
    out = {}
    for key, value in sorted(env.items()):
        if SECRET_KEY_RE.search(key):
            out[key] = {"value": None, "secret": True, "set": bool(str(value or "").strip())}
        else:
            out[key] = {"value": value, "secret": False, "set": bool(str(value or "").strip())}
    return out


def drop_secrets(updates: dict, allow: bool) -> tuple[dict, list[str]]:
    """(what may be written, what was refused).

    Default off, because "monitor and edit the model configuration" does not
    require pushing credentials across the network, and a control channel that
    can is a larger thing to secure than one that cannot.
    """
    if allow:
        return dict(updates), []
    kept, dropped = {}, []
    for key, value in updates.items():
        if SECRET_KEY_RE.search(key):
            dropped.append(key)
        else:
            kept[key] = value
    return kept, sorted(dropped)


def field_schema() -> dict:
    """This host's configuration surface, and what it cannot act on.

    `fields_digest` covers the *shape* -- keys, types, options -- so it changes
    when a controller would have to render something different and not when a
    hint is reworded. A hub compares it against its own to decide whether the
    form it already has is the right one.
    """
    inert = platforms.active().inert_config_capabilities
    fields, omitted = config_fields.applicable_fields(inert)
    shape = [[f.get("key", ""), f.get("type", ""),
              sorted(str(o) for o in f.get("options", []) or [])] for f in fields]
    digest = hashlib.sha256(json.dumps(shape, sort_keys=True).encode("utf-8")).hexdigest()
    sections: dict[str, list[str]] = {}
    for field in fields:
        sections.setdefault(field.get("section", ""), []).append(field.get("key", ""))
    return {
        "api_version": API_VERSION,
        "platform": platforms.active().name,
        "fields_digest": "sha256:" + digest,
        "sections": sections,
        "fields": fields,
        "omitted": omitted,
    }


def schema(settings: dict) -> dict:
    """What a controller needs to know before it tries anything.

    Version first: a hub whose major differs must refuse to control this host
    with a reason, rather than send a form that half works.
    """
    return {
        "api_version": API_VERSION,
        "state_api_version": public_api.API_VERSION,
        "platform": platforms.active().name,
        "allow_secrets": bool(settings.get("allow_secrets")),
        "endpoints": [
            "GET  /api/control/v1/health",
            "GET  /api/control/v1/schema",
            "GET  /api/control/v1/config",
            "GET  /api/control/v1/config/fields",
            "POST /api/control/v1/config",
            "POST /api/control/v1/config/preflight",
            "GET  /api/control/v1/saved-configs",
            "POST /api/control/v1/saved-configs/<name>/apply",
            "GET  /api/control/v1/services",
            "POST /api/control/v1/services/<name>/<action>",
        ],
    }


def services(providers: ControlProviders) -> list[dict]:
    """Name, label and state for everything that can be started or stopped.

    The reads a mutation needs, and no more: a controller has to know what it
    may act on and whether it is running. Anything richer belongs on the state
    API, which already assembles it.
    """
    env = providers.read_env()
    out = []
    for service in providers.services_table(env):
        out.append({
            "name": service["name"],
            "label": service.get("label", service["name"]),
            "group": service.get("group", ""),
            "state": providers.service_status(service["name"]),
        })
    return out
