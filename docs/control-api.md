# The control API

The write half of `docs/local-api.md`, on a third port. The read-only state API
on 8078 answers *what is this stack doing*; this answers *change it*.

It exists so one interface can configure several machines. It is off by
default, and turning it on is a decision about this host rather than something
an upgrade did.

## Why a third listener

The manager is one Flask app on 8077 with everything on it — the UI, the setup
wizard, model downloads, and every mutating route. It is not safe to expose,
and it never has been.

So there are three apps, not one with careful checks:

| Port | App | Carries |
|---|---|---|
| 8077 | the manager | everything, including the UI. Local only. |
| 8078 | `create_state_api_app()` | `/api/v1/*`, every rule a GET. |
| 8079 | `create_control_api_app()` | `/api/control/v1/*`, ten rules, no state. |

The property worth having is structural. The read-only listener cannot serve a
write because it never learned the route, and the control listener cannot serve
the whole stack's state because it never learned that one. A check that has to
be right on every route is a check that will eventually be missed on one;
`StateApiAppTests` and `ControlApiAppTests` assert both directions, and the
control app's rule list is pinned so that adding one is a deliberate act.

## Turning it on

```sh
LLM_CONTROL_ENABLED=on
LLM_CONTROL_HOST=127.0.0.1        # or this host's tailnet address
LLM_CONTROL_PORT=8079
LLM_CONTROL_TOKEN=                # required; blank refuses every request
LLM_CONTROL_ALLOW_SECRETS=off
```

Takes effect when the manager restarts.

**Two tokens, deliberately.** `LLM_API_TOKEN` reads the stack's state;
`LLM_CONTROL_TOKEN` stops its backends. Those are different blast radii and
must be independently rotatable — a single value would make the read-only
dashboard credential also the kill switch. Use different values.

## Three rules that differ from the read-only API

Each follows from this port being able to stop a backend.

**A non-loopback bind with no token refuses to listen.** `routes/public.py`
warns and binds anyway, because an open read-only API on a trusted tailnet is a
legitimate choice someone can make. There is no equivalent reading of an open
port that can rewrite `llm-stack.env`.

**An unset token is 503, not "no authentication required".** On 8078 an empty
token means an open read-only API, which is a configuration. Here it is a
mistake, and defaulting to open is the wrong way to resolve one.

**No `?token=`.** The read API accepts a query parameter because `EventSource`
cannot set headers. There is no stream here, so the query form buys nothing and
costs a secret in werkzeug's access log and in every reverse proxy in front of
it. Header only.

## Secrets are write-only

`GET /config` reports that a key is set and never what it is set to:

```json
"HF_TOKEN": {"value": null, "secret": true, "set": true}
```

`POST /config` refuses to write one unless `LLM_CONTROL_ALLOW_SECRETS=on`, and
says which keys it dropped. Default off, because "monitor and edit the model
configuration" does not require pushing credentials across the network.

What counts as a secret is `public_api.SECRET_KEY_RE`, reached through the
module rather than copied — two surfaces with independent notions of "looks
like a secret" is how one of them ends up being the one that leaks.

## Concurrent writes

`config_env.update_env_values` is an unlocked read-modify-write over a text
file. Two writers lose one of the two saves, and that has always been possible
with two browser tabs; a remote controller makes it likelier.

`GET /api/control/v1/config` returns a `config_etag` over the values.
`POST` may send it back as `If-Match`, and a mismatch is a 409 carrying the
current etag rather than a write. The etag covers the values, not the file:
comments and key order are not what a conflicting write would clobber.

## Keys a host does not know

`config_env.allowed_config_keys` unions in whatever the target's env file
already holds, so **what is writable is a fact about the target**. A controller
that sends a key this host has never heard of gets a 200 with that key listed
in `ignored_keys`, and the value is not written.

This matters most across a version gap. A hub updated to the renamed
`LLM_A_CTX_SIZE` writing to a host that still says `CHAT_PRIMARY_CTX_SIZE`
would otherwise get a success with nothing changed, which is the worst possible
answer because it looks like a save. **A non-empty `ignored_keys` on a remote
save is an error, not a success**, and a client should render it as one.

The other direction is safe: an older controller sending `CHAT_PRIMARY_*` to a
renamed host has its keys mapped through `LEGACY_ENV_KEY_MAP`. That asymmetry
is why the rename must add the old names to that map *before* a fleet spans a
version gap, not after.

## The routes

| Rule | Method | |
|---|---|---|
| `/api/control/v1/health` | GET | Liveness. The one unauthenticated route: a controller has to tell "unreachable" from "rejected my token" without a credential. |
| `/api/control/v1/schema` | GET | `api_version`, platform, endpoints. A controller whose major differs must refuse to control this host with a reason rather than send a form that half works. |
| `/api/control/v1/config` | GET | Every key, secrets masked, plus `config_etag`. |
| `/api/control/v1/config` | POST | Save. `?force=1` overrides the budget refusal; `If-Match` guards against a concurrent write. |
| `/api/control/v1/config/fields` | GET | This host's field registry and `fields_digest`, plus what the platform cannot act on and why. |
| `/api/control/v1/config/preflight` | POST | The budget model's opinion, no write. |
| `/api/control/v1/saved-configs` | GET | The saved profiles. |
| `/api/control/v1/saved-configs/<name>/apply` | POST | Apply one. |
| `/api/control/v1/services` | GET | Name, label and state — the read a service action needs. |
| `/api/control/v1/services/<name>/<action>` | POST | `start`, `stop`, `restart`. Goes through the manager's own rule, including the refusal to touch a unit the model router holds. |

Not here, deliberately: the setup wizard, HuggingFace downloads, application
updates, log streaming, and anything that reads the whole stack's state. The
first three are large surfaces that belong to the machine being sat at; the
last is what 8078 is for.

## What it does not do yet

Nothing on this host reaches *out*. This is the spoke half; a hub that polls
several of these is the next piece of work, and the peer list, the fleet page
and the host selector arrive with it.
