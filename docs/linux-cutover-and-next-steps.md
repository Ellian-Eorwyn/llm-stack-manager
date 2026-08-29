# Handoff: the cutover on `llms`, and what comes after

Everything below is on `main` and green on both CI runners — 870 tests,
`ubuntu-24.04` and `macos-latest`. **None of it has run on `llms`.** That is the
first job, and §2 is the part to read before touching anything.

This supersedes the earlier `macos-and-slot-refactor-handoff.md`, whose "next
steps" §5 is now all done except the parts listed in §5 here.

---

## 1. What this is for

Two goals, in this order.

**One repo that runs properly on both Linux/NVIDIA and macOS/Apple Silicon.**
Not a fork. Every existing Linux capability stays; Apple Silicon is a
first-class target using MLX and Metal-backed llama.cpp. The hardware is a Linux
box with 2× RTX 3090 (`llms`, in production), an M5 Ultra Mac Studio arriving,
and a 16 GB M1 Pro used for small-model testing.

**One interface across all of them.** Monitor first, control second. The read
half is done and needs no change on `llms` beyond exposing its existing state
API; the write half is §5.

Decisions taken with the owner. Settled — do not relitigate:

| Question | Decision |
|---|---|
| Slot naming | Peers: `llm-a` / `llm-b`, identical proxy trios, no hierarchy. |
| Persona tuning | Per-slot keys with today's shared keys as fallback. |
| GPU indices | Always absolute — stop per-slot `CUDA_VISIBLE_DEVICES` renumbering. |
| Router config | Per member: GPU placement, pooled-or-dedicated, context/cache. |
| Config migration | Read old keys, write new ones, via `LEGACY_ENV_KEY_MAP`. |
| Fleet auth | Bearer token over Tailscale, on a third listener. Reads stay on 8078. |
| Fleet control scope | Config keys, saved configs, service start/stop/restart. Not the setup wizard, HF downloads or app updates. |

## 2. Before the first `install.sh` on `llms`

**2.1 The proxy ports and model aliases are an external contract.**
`docs/pi-forge-scheduling-contract.md:131-144` pins them, and pi-forge and
open-webui depend on them:

| Port | Model | Thinking | Temp | Consumer |
|---|---|---|---|---|
| 8003 | `think` | on | 0.7 | the only endpoint both thinking-enabled and memory-backed |
| 8004 | `chat` | off | 0.7 | pi-forge `forge-chat-local`, open-webui |
| 8008 | `code` | on | 0.6 | pi-forge `forge-local` |
| 8012 | routed | per profile | — | nginx → `llms:8010` |

pi-forge's "think" role points at **:8008, not :8003**, deliberately. Slot B is
8103/8104/8108/8112 with aliases `think2`/`chat2`/`code2`.

Rename services and config keys; **never ports, aliases or persona semantics.**

**2.2 `install.sh` stops five units, unconditionally.** `install.sh:471`:

```bash
for unit in think nothink embed2 chat-backend chat-backend-moe; do
    systemctl disable --now "${unit}" ...
```

That is deliberate — a host predating their retirement would otherwise keep a
unit whose launcher no longer exists — but it means **the first `install.sh` run
on `llms` stops them.** Confirm nothing on that box still depends on any of
them, especially anything pointed at port 8010 expecting the MoE model.
`restore-active-stack.sh` now stops the same five for the same reason.

**2.3 The procedure is the README's "Production Cutover With Rollback".**
`scripts/backup-active-stack.sh` and `scripts/rollback-to-backup.sh` exist for
it. Take the backup first; service renames are *not* covered by the config
migration, and units on that box are named `chat-backend-dense`.

**2.4 Compare before and after.** `http://llms:8078/api/v1/snapshot` answers
today and will answer after. GPU state, per-slot context, service health and a
redacted config section — capture it on both sides of the cutover.

## 3. What changed since `llms` last saw this code

Behaviour changes an operator would notice, all deliberate. The mechanical
refactors are in the git log and do not need repeating here.

- **Telemetry probed the wrong port for the secondary backend.**
  `CHAT_BACKEND2_PORT` and `CHAT_BACKEND2_HOST` were written in `telemetry.py`
  and nowhere else; the field, the launcher and the proxy all say
  `CHAT2_BACKEND_PORT`. The panel used 8020 whatever was configured, and looked
  right only because the defaults agreed. **If `llms` runs a non-default
  secondary port, its panel has been reporting a backend it never reached.**
- **`update.sh` restarts the same set on macOS as on Linux.** The two lists were
  one entry apart (`playwright-server`) for no recorded reason.
- **The setup wizard no longer offers "Second embedding backend".** The slot went
  in `8f448f3`; the checkbox that installed it did not, so selecting it asked for
  a component with no unit and no launcher.
- **A Mac is no longer offered GPU placement it cannot perform** — 27 controls
  across 7 sections. No effect on Linux: `applicable_fields` withholds nothing
  when `Platform.inert_config_capabilities` is empty, which it is there.
- **Nine launchers are one.** `start-chat-backend-dense.sh`,
  `start-chat-backend2.sh` and `start-task.sh` are fifteen-line shims that
  `exec scripts/start-backend.sh <slot>`. The installed units still name those
  paths, so an already-installed host keeps working; they go once the units name
  `start-backend.sh`.
- **Two new listeners exist but neither binds by default.** `LLM_CONTROL_ENABLED`
  is `off`, and the fleet poller does not start without `config/fleet.json`.

## 4. Setting up the fleet across the two boxes

Nothing here changes how either machine serves models. It is opt-in on both
sides.

**On `llms` (the peer being watched):**

```sh
LLM_API_ENABLED=on
LLM_API_HOST=<its tailnet address>    # not 0.0.0.0
LLM_API_TOKEN=<a value>               # required off-box
```

That is all the read half needs. Leave `LLM_CONTROL_ENABLED=off` until the
write half exists and you want it.

**On the Mac (the hub):** add `llms` through the UI's host picker, or write
`config/fleet.json` directly:

```json
{"version": 1, "hosts": [
  {"id": "llms", "label": "llms (2x 3090)", "host": "llms.<tailnet>.ts.net",
   "scheme": "http", "read_port": 8078, "control_port": 8079,
   "token": "<the LLM_API_TOKEN above>", "control_token": "",
   "expected_hostname": "llms", "enabled": true, "control": false}]}
```

`control: false` is the default and the right setting until the cutover is
proven. The file is gitignored; it holds credentials and never leaves the box —
`GET /api/fleet/hosts` reports `has_token`, never the value.

Any instance can be the hub; there is no hub build. An instance *is* a hub when
its `fleet.json` lists peers.

Read `docs/control-api.md` before turning on `LLM_CONTROL_ENABLED` anywhere. The
short version: it refuses to bind off-box without a token, an unset token is a
503 rather than an open door, and it uses a *different* credential from the read
API on purpose.

## 5. What is left

In order. Each step should leave the tree releasable.

**Step 9 — the fleet write half. Done.** Both parts, plus a fifth schema case
the four below do not name: two hosts can share a key set and still disagree
about a field's type or options, which is what `fields_digest` catches and
neither key set can see. `ignored_keys` is enforced at the hub as a 409 rather
than only rendered, so every client gets the rule.

- `/api/fleet/<id>/config` (GET/POST), `/config/fields`, `/saved-configs`,
  `/service/<name>/<action>` on the hub, each proxying to the peer's control
  listener. Add them to `fleet.FLEET_PROXIED` in `web/static/js/fleet.js` and to
  `RouteInventoryTests.EXPECTED`. The spoke half already exists and is tested.
- The config tab has to render *another host's* field schema.
  `GET /api/control/v1/config/fields` already returns it with a `fields_digest`
  and an `omitted` map. Every field wrapper carries `data-cfg-key`, which is
  what makes DOM-level filtering possible without a second renderer. Four cases
  are worth handling: digest and key sets match (fill the form already on the
  page), key sets differ (hide the omitted ones and say how many and why),
  digest differs (a generic `text|number|select|toggle` renderer for the extras;
  bespoke types say "edit this on the host itself" because they call local-only
  endpoints anyway), major version differs (read-only with a banner naming both).

  **`ignored_keys` is not optional.** `allowed_config_keys` unions in whatever
  the *target's* env file holds, so what is writable is a fact about the target.
  A non-empty `ignored_keys` on a remote save must render as an **error**, never
  a success — otherwise a hub that knows a renamed key gets a 200 with nothing
  changed, which looks exactly like a save.

**Step 10 — rename to `llm-a`/`llm-b`. Done.** Units, config prefixes, budget
names, components, labels and sections. Ports, aliases and persona semantics
untouched per §2.1, and `CHAT_BACKEND_PORT` / `CHAT2_BACKEND_PORT` / their hosts
keep their names because they *are* the port contract.

**This one needs a cutover on `llms`.** The launchers were renamed with the
units, so the installed `chat-backend-dense.service` points at a script that no
longer exists: it keeps running, but it cannot restart until `install.sh` has
retired it and installed `llm-a`. Run `install.sh` then `restore-active-stack.sh`
before the next reboot, not after.

Three migrations carry existing hosts across it, all read-generously /
write-strictly: `LEGACY_ENV_KEY_MAP` gains 110 generated entries so a config
holding only `CHAT_PRIMARY_*` still reads; `setup_engine.LEGACY_COMPONENTS` maps
the old component names, without which every `install-state.json` on disk names
components that no longer resolve and the boot path starts no chat backend at
all; and `health.LEGACY_UNIT_NAMES` carries `service-expectations.json`, without
which `chat-backend2: off` stops applying and the boot path starts a second 27B
onto a GPU holding the first.

Verified on `llms` by building the command line from the renamed registry
against the untouched live config and diffing it against the running process:
78 arguments, identical.

The original note, kept because the reasoning still applies to step 11: `web/backends/slots.py` is
the one source and twelve tables derive from it, asserted key-for-key in
`tests/test_slot_registry.py`. Ports and aliases unchanged (§2.1). Two things
the legacy map does not cover:

- `config_env.py:92-97` backfills every `CHAT_PRIMARY_*` field from its bare
  `CHAT_*` twin through a loop hardcoded to those two prefixes, with none of them
  listed in `LEGACY_ENV_KEY_MAP`. Rename and it silently stops firing.
- The `backend_defaults` block and the ~200 `setdefault` lines below it are
  literal key names.

Flatten the map rather than chaining: `tests/test_llm_stack_manager.py`
asserts no canonical key is itself a legacy key, and the existing
`CHAT_MOE_* → CHAT2_*` entries show the intended style. **Add the old names
before a fleet spans a version gap, not after** — an older hub writing
`CHAT_PRIMARY_*` to a renamed host is then safe, while the reverse is not.

**Step 11 — proxy registry and per-slot personas.** `start-chat-proxy2.sh` is a
hand copy of `start-chat-proxy.sh` differing in exactly three ways: seven env
overrides, three alias overrides, and 19 memory-gateway exports replaced by a
hardcoded `MEMORY_GATEWAY_ENABLED=off`. The other 33 persona keys are byte-for-
byte duplicated, so both proxies share every `THINK_TEMP` and
`CODE_REASONING_EFFORT` — there is no `THINK2_TEMP`. Collapse to one script
taking a slot argument, and add per-slot keys falling back to the shared ones,
which is the idiom `scripts/llm-chat-proxy.py:135-138` already uses.

Fix in passing: `THINK_REASONING_EFFORT` and `CODE_REASONING_EFFORT` are read by
the proxy but exported by neither script. Latent, because systemd's
`EnvironmentFile` supplies them — but `start-chat-proxy.sh:15` sources bare, so
running it by hand drops both.

**Step 12 — router per-member config.** Mostly surfacing.
`render-models-ini.py` already maps `N_GPU_LAYERS`/`MAIN_GPU`/`DEVICE`/
`SPLIT_MODE`/`TENSOR_SPLIT` per member, but:

- **EMBED and RERANK have no `_MAIN_GPU` and no `_DEVICE` field at all**, and
  ASR has `_MAIN_GPU` but no `_DEVICE`. Verified still true. Generalise the block
  the way `_transcription_engine_fields()` does — a table, no identity-key
  special cases — rather than `_clone_chat_backend_field`, whose `secondary`
  branch is dead code that would emit `CHAT_SECONDARY_*` where the rest of the
  tree says `CHAT2_*`.
- `<MEMBER>_POOLED` per member, with `MODEL_ROUTER_MEMBERS` derived. That default
  string is still duplicated in **12 files**.
- Relabel `MODEL_ROUTER_GPU_VISIBLE_DEVICES` as the superset the router may
  touch. It cannot be per-member: `CUDA_VISIBLE_DEVICES` is process-level and the
  router spawns its own children.
- **Absolute GPU indices.** `start-backend.sh:68` still exports
  `CUDA_VISIBLE_DEVICES` from the slot's own key, so an index means different
  cards under the router and as a dedicated unit. Dropping the renumbering
  **changes placement for existing dedicated units on `llms`** and needs a
  deliberate moment plus a cutover note.
- `split-mode = tensor` silently discards placement. The UI must show that.
- `MEMBER_PORTS` in `install-model-router-nginx.sh` is the one member table with
  no test; give it the key-set assertion the other two have, allowing for the
  deliberately portless `ASR`.
- **Confirm per-member residency exists in the pinned llama.cpp before building
  a setting for it.** A documented limitation beats a control that does nothing.

## 6. Where to be suspicious

- **The systemd path is exercised only by `bash -n` and a sandboxed harness.**
  Argv equivalence is proven; unit *wiring* is not. `After=`, `Wants=` and the
  removed `Conflicts=` are checked by tests, not by a running systemd.
- **`restore-active-stack.sh` and `activate-selected-stack.sh` still map
  components to units in shell, twice.** Named in
  `tests/test_slot_registry.py` rather than hidden. Collapsing it needs Python
  on the install path.
- **A remote host reports context only for backends telemetry reached**, where
  the local page shows all six including stopped ones. A snapshot carries running
  geometry; `/api/status` reads the env.
- **`*_DEVICE` still defaults to `CUDA0` on a fresh Mac install.** Cosmetic — the
  launcher drops it with a message rather than failing — but wrong.
- **A stale duplicate exists.** `~/Applications/LLMs/llm-stack-manager` on the M1
  Pro is a separate checkout, well behind, still serving MLX embeddings on 8005
  and transcription on 8014 as LaunchAgents. Its uncommitted work is in `main`;
  it has not been replaced.

## 7. How to verify anything here

```bash
bash test.sh                       # 870 tests, green on macOS and Ubuntu
```

The tools that make a change here safe rather than hopeful:

- **Argv equivalence.** `tests/launcher_harness.py` with two golden files:
  `launcher-argv.golden.json` (empty `models/`) and
  `launcher-loaded-model.golden.json` (model files present, `budget.py` stubbed
  to record the question). The second pins `--mmproj` and the memory-fit report,
  neither of which the first can see. Diff both before and after any launcher or
  registry change.
- **`tests/test_slot_equivalence.py`** runs the real shell against the registry
  across 26 configurations chosen where the shell branches. It is what says the
  launcher deletion was safe, and it keeps saying it.
- **`tests/platform_harness.py`** — `as_linux()` / `as_darwin()` run either
  platform's code path on either host. Use it rather than skipping tests.
  `LLM_STACK_PLATFORM` does the same for the shell half, and `platforms.detect()`
  honours it so the two cannot disagree.
- **`tests/test_fleet.py`** — a peer is two Flask `test_client()`s over the same
  app factories, spliced in at `patch.object(fleet, "fetch", ...)`. No second
  machine required.
- **`tests/test_slot_registry.py`** pins every derived table against the literal
  it replaced, and `stack-services.sh` against the registry.

Two hazards worth knowing before you write tests here:

- `unittest.mock.patch` mutates module state, so a patch entered on a worker
  thread can overlap another test's. Run `bash test.sh` twice; it has caught a
  real one.
- The manager caches its Jinja template and stamps static assets with the commit
  hash, so an edited template or script needs a restart and a forced reload
  before a browser shows it.

## 8. Conventions this repo holds to

Load-bearing and tested.

- **Reach behaviour through the module** — `core.read_env()`, not
  `from core import read_env`. `ModuleBoundaryTests` enforces it, and
  `fleet` and `control_api` are in that list because the fleet tests depend on
  substituting `fleet.fetch`.
- **Read generously, write strictly.** Legacy env names are readable forever and
  writable never, so they drain out as settings are re-saved.
- **A helper must never stop a backend from starting.** `backend-preflight.sh`
  degrades to permissive when `budget.py` cannot form an opinion. Command
  *construction* is the exception — that must be right or not run, which is why a
  bad chat-template id raises rather than degrades.
- **Three Flask apps, not one with checks.** 8077 carries everything and is local
  only; 8078 is read-only; 8079 can only write. The read app cannot serve a write
  because it never learned the route. `StateApiAppTests` and `ControlApiAppTests`
  assert both directions.
- **Say what was measured, not what was assumed.** The docs here cite the failure
  that motivated each design; keep doing that.
