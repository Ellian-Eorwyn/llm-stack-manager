# Four models, one VRAM budget

Why the auxiliary models stopped being services, and what that changed.

## 1. Reserving memory for a model nobody is using

`embed`, `ocr`, `rerank` and `task` were four systemd units, each holding a
llama-server, each holding VRAM from the moment it started until somebody
stopped it. None of them is busy for more than a few seconds at a time.

On a box whose primary chat backend takes ~39 GB of the 48 GB across two 3090s,
that arithmetic does not work. It failed concretely: `ocr.service` sitting at
`Result=exit-code` with

```
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 590.07 MiB on device 0:
    cudaMalloc failed: out of memory
llama_model_load: error loading model: unable to allocate CUDA0 buffer
```

It could not find 590 MiB, because `embed` was holding 3.6 GB to serve a request
that had not arrived yet. `docs/service-health.md` §3 describes the same failure
from the other end: the OCR SDK pulled `ocr` up through `Wants=`, it could not
allocate, and `Restart=on-failure` bounced it 32 times.

## 2. llama.cpp already had the mechanism

The pinned build has a **router mode**: `llama-server` started with no model
reads an INI preset, spawns one child per model on demand, and evicts the
least-recently-used once `--models-max` are resident.
`server_models::unload_lru()` blocks on a condition variable until the evicted
model reaches `UNLOADED` before the next one loads, so a swap cannot half-happen.

Three properties made this worth adopting over a proxy of our own:

**Requests already carry the routing key.** The router dispatches on the `model`
field in the request body, and every caller in this stack already sends the
right name — `llm-chat-proxy.py` forces `model = "embed"` on embeddings,
`config/glmocr-sdk.json` sends `"ocr"`, `config/honcho.env` sends `embed`.
Nothing had to change about what anyone sends.

**Concurrency is handled.** The GLM-OCR SDK fans a document out to 16 workers
(`GLMOCR_MAX_WORKERS`). Measured against a live router: 16 concurrent requests
for a non-resident model produced **one** spawn, not sixteen.

**Observability endpoints cannot cause a load.** `GET /health`, `/props` and
`/models` are documented as never triggering a model load or resetting the idle
timer. This is load-bearing: the services panel polls every five seconds, and
under a design where probing could load a model, the panel would have become a
swap generator pointed at its own GPUs.

## 3. What it looks like now

One unit, `llama-router`, running `llama-server` in router mode against a preset
rendered from the config keys that already existed:

```
chat-proxy, honcho ──▶ :8005 ─┐
rerank clients      ──▶ :8006 ─┼─ nginx ─▶ 127.0.0.1:8013  llama-router
task clients        ──▶ :8007 ─┤                                │
glmocr-sdk, Flask   ──▶ :8009 ─┘                    [embed] [ocr] [rank] [task]
```

The per-model ports still answer. `scripts/install-model-router-nginx.sh` fronts
each one onto the router, so nothing on the LAN — including integrations this
repo does not know about — had to be repointed. The router picks the model from
the body, so nothing is rewritten in transit.

`scripts/render-models-ini.py` generates `config/models.ini` at service start
from the `EMBED_*`, `OCR_*`, `RERANK_*` and `TASK_*` keys. The config UI stays
the one place these models are configured; the preset is a build artefact and
says so in its header.

## 4. Two details that bite

**The section name is the routing key.** `server_model_meta::update_args`
overwrites each child's `--alias` with its preset section name, so the section
name *is* the model id clients must send. The reranker is the trap:
its env prefix is `RERANK` but `RERANK_MODEL_NAME` is `rank`. A section named
for the prefix would silently rename the model out from under every caller.

**Top-level preset keys become a routable model.** Anything before the first
`[section]` header lands in the preset named `default`, which the router then
advertises on `/v1/models` as a real model with no model path — visible to
everything that enumerates models, and a load error for anyone who asks. Putting
`version = 1` under `[*]` keeps the version and loses the phantom.

A third, smaller one: the router treats the llama.cpp cache as a model source in
addition to the preset, so anything ever pulled with `-hf` becomes routable.
`start-model-router.sh` points `LLAMA_CACHE` at a directory of its own so the
preset is the only source.

## 5. What the panel says

A pooled model is not a unit any more, so `systemctl is-active` reports it
inactive whether or not the router is serving it happily. Reporting that as
"stopped on purpose" would be true of the unit and misleading about the model.

`telemetry.pooled_units()` names the units the router owns, and `health.py` uses
it twice: `expectation_for()` returns `off` for them regardless of what the panel
last recorded, so a stale `on` from before the switch cannot resurrect them; and
the inactive branch reports *"held by the model router — the model loads on
demand and is not run as a unit"*.

Upstreams move with it. `SERVICE_DEPENDENCIES` still says `glmocr-sdk` needs
`ocr`, because that is the shape of the stack as installed and
`tests/test_health.py` checks it against `setup_engine.COMPONENT_DEPENDENCIES`.
Router mode is a runtime choice, so `ROUTER_DEPENDENCY_OVERRIDES` supplies the
substitution — `glmocr-sdk` depends on `llama-router` instead — and
`dependencies_for()` picks between them.

The router card lists its models with their live state and a Load/Unload button,
from `/api/model-router`.

## 6. Turning it on, and off

`MODEL_ROUTER_ENABLED=off` is the default, and with it off nothing about the
stack differs from before: the four units bind their own ports exactly as they
did. Turning it on and re-running the installer moves the units aside, installs
the router, and adds the nginx shims. Turning it off and re-running removes the
shims and gives the ports back.

The member units stay installed but stopped rather than being removed, so that
rollback is a config flag rather than a reinstall. The manager refuses to start
one while the router owns it — starting it would fight nginx for the port and
put a second copy of the model on the GPU — and says so with a 409 rather than
half-succeeding.

## 7. The part that is still a judgement call

`--models-max` is a **count, not a memory budget**. The router evicts
least-recently-used when the count is reached, but knows nothing about how large
the survivors are. With a cap of 2 it will happily try to hold `embed` (~3.6 GB)
and `ocr` together in whatever is free.

There is no way to express "2 GB of models" to it. Size the cap against measured
free VRAM, and use `1` for strict one-at-a-time.

The other live risk is Honcho. `config/honcho.env` points its embeddings at
`127.0.0.1:8005` with `EMBED_MESSAGES=true`, so its deriver embeds every message
as background work. With Honcho running, that is a continuous unattended load
trigger that will evict OCR seconds after it loads. Pointing Honcho at `embed2`
on 8011 — outside the pool, 639 MB, and already the right 1024 dimensions —
keeps it out of the way.
