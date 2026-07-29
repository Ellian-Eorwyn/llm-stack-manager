# "Active" did not mean "working"

What the services panel reported, what that cost, and what replaced it.

## 1. One question was standing in for another

`get_service_status()` asked `systemctl is-active` and nothing else. That answers
"is the process running". The panel presented it as "is the service working",
and on this box those came apart in three ways.

**A service can run with its upstream dead.** `glmocr-sdk` answers `/health`
with `{"status":"ok"}` from its own process. Every document it is handed goes to
`127.0.0.1:8009`, which is written into `config/glmocr-sdk.json` by
`scripts/start-glmocr-sdk.sh` and which nothing is listening on when `ocr` is
stopped. The card was green. The same shape was live during this work on the
secondary proxy: `chat-proxy2` running and answering `/v1/models`, with
`chat-backend2` stopped.

**A crashed unit looked like a stopped one.** `is-active` returns non-`active`
for both, and the caller compared against the string `"active"`, so `failed`
collapsed into `inactive`. There was a `failed` pill in the CSS that nothing
could ever produce for a systemd service.

**Half the stack is off at any time and nothing said whether that was meant.**
`ocr`, `rerank`, `task`, `embed2`, `chat-backend2`, `honcho-api` and
`honcho-deriver` were all inactive, rendered identically to a service that had
just died.

## 2. What replaced it

Health is assembled from four inputs, in `web/health.py`:

| Input | Source |
| --- | --- |
| Unit state, now including `failed` | `ServiceManager.state()` — one `systemctl show` instead of two calls |
| Readiness | `SERVICE_PROBES` — `/props`, `/v1/models`, `/health`, or a TCP connect |
| Upstreams | `SERVICE_DEPENDENCIES` — any-of groups, resolved transitively |
| Expectation | `config/service-expectations.json` |

The states a card can now show:

| State | Meaning |
| --- | --- |
| `active` | running, probe answered, upstreams up |
| `degraded` | running, but the probe failed or an upstream is down |
| `failed` | systemd says the unit failed |
| `stopped` | not running, and not expected to be — neutral |
| `inactive` | not running, but expected on |
| `unknown` | unit not installed |

`get_service_status()` still returns only the unit status, because six callers
compare it against `"active"` — telemetry's `resolve_targets`, the active-model
snapshot, the saved-config apply path and three status routes. A backend
rendered `degraded` must not vanish from telemetry. The richer verdict rides
alongside as a parallel `health` map on `/api/status`; `services` keeps exactly
the shape it always had.

## 3. Expectation had to be recorded, because nothing carried it

Three plausible sources were checked live and all three were wrong:

| Candidate | Why not |
| --- | --- |
| `systemctl is-enabled` | reads `disabled` for `chat-backend-dense` and `chat-proxy` while they are running |
| `*_ENABLED` env flags | read `on` for `glmocr-sdk`, `searxng` and `honcho` while those are stopped — they mean "configured", not "should be up" |
| `setup_engine` component selection | `config/install-state.json` does not exist, so it falls back to `CORE_DEFAULTS` |

So the manager records it: a successful start or restart from the UI records
`on`, a successful stop records `off`, and applying a saved profile records the
profile's `_active_services` for every service in it. `POST
/api/service/<name>/expect` sets it without cycling the service.

An unrecorded service that is not running renders `stopped`, not amber. The
panel only claims a service *should* be up when somebody said so. The single
exception is a `*_ENABLED` flag set to `off`, which does mean expected-off — the
start script exits 0 without launching anything, so the unit cannot come up.

## 4. Probes run off the request path

`/api/status` is polled every five seconds and already spawns a `systemctl` call
per service. Ten HTTP probes inline would have blown that interval precisely
when a backend was wedged — the case that most needs to render.

`health.Prober` is a daemon thread on a ten-second sweep holding a snapshot
behind a lock; `/api/status` reads the cache. It starts on the first request, so
importing `app.py` in a test spawns nothing. Before the first sweep lands, every
card falls back to its unit state rather than reporting as broken.

Measured after the change: `/api/status` returns in 0.26s (dominated by
`nvidia-smi`), a full probe sweep of a healthy stack takes 24ms, and merging
`is_active` + `is_installed` into one `systemctl show` halved the subprocess
count per poll.

## 5. `validate.sh`

It used to fail on any service that was off, which made a clean run mean
"everything is installed" rather than "everything that should be running works".
It now reads the same expectation file and skips what is deliberately down,
generalising the `HONCHO_ENABLED` special case it already had. It also probes
the backend on `CHAT_BACKEND_PORT` directly — checking only the proxy ports
meant a backend that had died behind a live proxy read as a proxy fault — and
warns when the OCR SDK is up without its backend.

## 6. What enforces this

| Claim | Test |
| --- | --- |
| A service whose upstream is down is degraded | `CollectTests.test_a_service_whose_upstream_is_down_is_degraded` |
| Degradation follows the chain | `CollectTests.test_degradation_propagates_along_the_chain` |
| A crashed unit reports as failed | `ServiceHealthTests.test_a_crashed_unit_is_no_longer_indistinguishable_from_a_stopped_one` |
| Off on purpose is not a fault | `CollectTests.test_a_stopped_service_nobody_asked_for_is_not_a_fault` |
| `*_ENABLED=on` is not evidence of intent | `ExpectationTests.test_an_enabled_flag_set_to_on_says_nothing` |
| The service and component graphs agree | `DependencyGraphTests.test_service_graph_agrees_with_the_component_graph` |
| `/api/status` keeps its shape | `ServiceHealthTests.test_status_carries_health_without_changing_the_services_map` |
| A missing probe does not blank a card | `CollectTests.test_a_service_with_no_probe_yet_is_judged_on_its_unit_state` |
