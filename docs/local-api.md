# The read-only state API

Other applications on this box, the LAN, or the tailnet can read what the stack
is doing: GPU utilisation, how much VRAM each model holds on each card, how much
context the slots have consumed, service health, backend log events, and a list
of things currently worth acting on.

It listens on **port 8078** by default, separately from the manager.

## 1. Why it is not on port 8077

The manager runs as root. It can start and stop services, rewrite
`config/llm-stack.env`, trigger an update that restarts the process, and
`GET /api/config` returns the entire env file including every API key in it.
None of that is authenticated, because it was built for a trusted-LAN setup UI
and the README says so.

So "let other apps read the stack state" could not be answered by pointing more
machines at 8077. The read-only routes live on a **second Flask app in the same
process**, on its own port, and the mutating routes are not registered on it:

```
8077  llm-manager     root, no auth, full control   -> localhost / trusted LAN
8078  llm-stack-api   GET only, optional token      -> safe to point a tailnet at
```

The property that matters is that `POST /api/service/embed/stop` on 8078
returns 404 because that rule **does not exist on that app**, not because a
check rejected it. A check has to be right on every route, and will eventually
be missed on one. `StateApiAppTests` asserts both halves: every rule is GET, and
the mutating paths are absent.

Same process, though, and that is deliberate: the journal tailers, the health
prober and the GPU cache are already running there, so an extra client costs a
dict serialisation rather than another `nvidia-smi`.

## 2. Configuration

```sh
LLM_API_ENABLED=on
LLM_API_HOST=127.0.0.1      # 0.0.0.0 for the LAN; `tailscale ip -4` for the tailnet only
LLM_API_PORT=8078
LLM_API_TOKEN=              # blank = no auth
LLM_API_ALLOW_ORIGINS=      # CORS; blank sends no CORS headers
LLM_API_STREAM_INTERVAL=2
LLM_API_WEBHOOK_URL=
LLM_API_WEBHOOK_EVENTS=service_state,alert
```

`ENABLED`, `HOST` and `PORT` are read when the listener binds, so changing them
needs a manager restart. Everything else is read per request and per stream
tick, so it applies immediately.

Authentication is **optional and off by default**. With a token set, requests
carry `Authorization: Bearer <token>`; `?token=` is also accepted, because a
browser's `EventSource` cannot set headers and the stream would otherwise be
unusable from a web page. `/api/v1/health` never requires one — a health check
that needs a credential is one that stops being checked.

Binding to a non-loopback address with no token is allowed. It is a reasonable
choice on a tailnet, and it is the kind of choice that should not be discovered
six months later, so it is printed at startup and reported as an
`api_unauthenticated` alert in the payload itself.

## 3. Endpoints

| Path | What it gives you |
|---|---|
| `/api/v1/schema` | Self-describing: sections, alert codes, thresholds, endpoints. Start here. |
| `/api/v1/snapshot` | Everything. `?include=gpus,backends` narrows it; `?window=` sets the stats window. |
| `/api/v1/gpu` | GPUs, with VRAM grouped by model. |
| `/api/v1/backends` | Per backend: model, slots, context rollup, throughput, config drift. |
| `/api/v1/services` | Per service: health state, probe result, upstreams, restarts. |
| `/api/v1/alerts` | What is wrong, with a stable code each. |
| `/api/v1/logs` | Parsed backend events. `?unit=`, `?kind=`, `?since=`, `?limit=`. |
| `/api/v1/logs/raw` | Unparsed journal lines. `?unit=`, `?lines=`. |
| `/api/v1/metrics` | Prometheus exposition of the same snapshot. |
| `/api/v1/events` | SSE stream. `?include=`, `?interval=`. |
| `/api/v1/health` | Liveness of the API itself. |

An unknown `include=` value is a 400 listing the known names, rather than a
payload silently missing the one section the caller wanted.

## 4. The three things that are new rather than repackaged

**VRAM per model.** `nvidia-smi` reports processes, so `gpus[].models[]` joins
each PID to a unit and a model:

- The **unit** comes from `/proc/<pid>/cgroup`. Matching against each unit's
  MainPID finds only the process systemd started, and the ones holding VRAM are
  often children — `llama-router` forks a `llama-server` per resident model, and
  none of those is a MainPID. The cgroup names the unit for every process in it,
  from one file read and no subprocess.
- The **model** comes from that process's own `--model` and `--alias`
  arguments, falling back to what the backend reports at `/props`.

Blocks are keyed on `(unit, model)`, which is what makes the router legible: its
children share a unit but each names a different model, so `task` and `embed`
appear as separate blocks with separate numbers instead of one merged
`llama-router` row. Those carry `"attribution": "router"` and a
`router_resident` flag from the router's own `/models` view — a model holding
VRAM that the router does not consider resident is worth being able to see.

Backends that own their unit are `"exclusive"`. A process whose model cannot be
determined is `"unattributed"` rather than dropped: it is competing for the same
VRAM either way.

Three bugs were fixed to make this work, all of which had been silently
producing wrong output:

- `label_gpu_process` referenced an undefined `cmdline`, and the surrounding
  `except Exception: pass` swallowed the `NameError` — discarding *every*
  attribution while the payload still looked well-formed. The `except` is now
  narrow, so the next bug of that shape surfaces.
- Matching a service name as a plain substring made every `glmocr-sdk` process
  report as `ocr`, because "ocr" occurs inside "glmocr" and `ocr` comes first in
  `SERVICES`. Matching is now longest-name-first and never mid-word.
- Every model the router held was attributed to whichever service name happened
  to appear in its command line, because nothing looked at the cgroup.

**Context across slots.** `/props` gives the geometry, `/slots` gives the
occupancy, and neither alone says how full a backend is. `backends[].context`
sums them. Two numbers matter and they are different:

- `used_pct` — the whole backend.
- `max_slot_pct` — the fullest single slot.

A backend at 30% overall still rejects the next prompt if it lands on a slot at
95%, because `--ctx-size` is divided by `--parallel` and a request is measured
against the quotient. `max_slot_pct` is the one that predicts a refusal.

**Coded alerts.** `telemetry.warnings_for` writes prose for a banner. A program
needs to branch, so `alerts[]` carries a stable `code`, the `subject` it is
about, and a `detail` object. Codes are listed in `/api/v1/schema`; the tests
assert every code the API can emit is documented there.

`config_runtime_mismatch` is the one worth calling out. A backend reads its
config at launch, so an edit that has not been restarted into leaves the panel
showing the number that was typed and the model serving the number it started
with. This compares the live `/props` against the saved env and reports the
difference per field.

## 5. The stream

`/api/v1/events` is server-sent events. One collector serves every client:

```
event: snapshot     the full payload, on the interval
event: delta        a service changed state
event: alert        an alert was raised or cleared
event: log          a parsed backend log event
: ping              every 15s, so idle proxies do not reap the connection
```

The naive shape — a thread per connection, each polling — multiplies every
subprocess and every backend probe by the number of connected apps, which on
this box is how a dashboard becomes the reason the GPU is busy. So there is
exactly one loop. `?interval=` is honoured as the floor the loop runs at while
that client is the fastest thing connected, not as a private timer.

A client that stops reading is dropped after `SUBSCRIBER_QUEUE_DEPTH` queued
events rather than buffered indefinitely; it gets an `overrun` event first.

The first tick establishes a baseline and emits no deltas. Without that, every
manager restart would report the entire service table as transitions.

## 6. Webhooks

Set `LLM_API_WEBHOOK_URL` and state changes are POSTed instead of polled. This
is the only **outbound** capability in the manager, which is worth knowing given
what it runs as. Delivery is a bounded queue and one worker, so a slow receiver
cannot stall collection; a full queue drops the event with a log line.

When a token is set the body is signed with `X-LLM-Stack-Signature`
(`sha256=` + HMAC-SHA256 over the raw body, keyed by `LLM_API_TOKEN`).

## 7. Examples

```bash
TOK=...   # only if LLM_API_TOKEN is set

curl -s localhost:8078/api/v1/schema | jq .

# is anything actually running right now
curl -s localhost:8078/api/v1/snapshot?include=stack | jq .stack

# what is each model holding, per card
curl -s localhost:8078/api/v1/gpu | jq '.gpus[] | {index, util, models}'

# how close is the primary backend to its context limit
curl -s localhost:8078/api/v1/backends | jq '.backends[] | {unit, context}'

# anything wrong
curl -s localhost:8078/api/v1/alerts | jq '.alerts[] | {level, code, text}'

# live
curl -N localhost:8078/api/v1/events
curl -N "localhost:8078/api/v1/events?include=alert,delta&interval=5"

# scrape
curl -s localhost:8078/api/v1/metrics
```

From a browser page, with `LLM_API_ALLOW_ORIGINS` set:

```js
const events = new EventSource(`http://llms:8078/api/v1/events?token=${TOK}`);
events.addEventListener("snapshot", e => render(JSON.parse(e.data)));
events.addEventListener("alert",    e => notify(JSON.parse(e.data).alert));
```

## 8. Cost

`/api/status` used to fire roughly seventeen subprocesses per call — two
`nvidia-smi` and one `systemctl show` per unit — uncached, on the UI's
five-second poll. Adding external consumers on top of that would have hurt, so
`get_gpu_info()` and `service_main_pids()` now sit behind a two-second
`core.ttl_cache`. It collapses the duplicates within a single poll while leaving
readings a second apart genuinely distinct, and the existing UI benefits by the
same amount.

Log endpoints read from the ring buffer `telemetry.REGISTRY` already maintains
rather than spawning `journalctl`. `/api/v1/logs/raw` does run one, but one-shot
with `-n`, never `-f`: a follow there would leave a process alive per client for
as long as they held the socket. The unit name is checked against an allow-list
before it reaches the command line, because that command runs as root.
