# Handoff

The current state of this project and what is next. **Keep this file current
rather than adding another handoff beside it** — that has gone wrong twice now,
and a stale handoff is worse than none, because it is the first thing a fresh
agent trusts.

Everything below is on `main` and green on both CI runners. Steps 1–12 of the
slot simplification and the fleet work are **done**.

## 1. What this is for

**One repo that runs properly on Linux/NVIDIA and macOS/Apple Silicon.** Not a
fork. Every Linux capability stays; Apple Silicon is a first-class target using
MLX and Metal-backed llama.cpp.

**One interface across all of them.** The hardware is `llms` (Linux, 2× RTX
3090, production), an M1 Pro, and an M5 Ultra Mac Studio arriving.

Settled, do not relitigate: `llm-a`/`llm-b` peer naming; per-slot persona keys
falling back to the shared ones; absolute GPU indices; `LEGACY_ENV_KEY_MAP` for
renames; bearer token over Tailscale on a third listener; fleet control scoped
to config keys, saved configs and service actions — not the setup wizard, HF
downloads or app updates.

## 2. Where both machines stand

**`llms`** — Linux, production. Cut over and running the renamed slots:
`llm-a` + `llm-a-proxy` active, `llama-router` active, `transcript-backend`
active, `llm-b` + `llm-b-proxy` stopped on purpose. Its read API answers on
8078 with a token; its control API answers on 8079.

**The M1 Pro** — the fleet hub. Stack directory
`~/Applications/LLMs/llm-stack-manager`, three LaunchAgents in the user domain:
the manager (8077 + 8078), MLX embeddings (8005), Parakeet transcription
(8014). `config/fleet.json` lists `llms` with `control: false`. It polls across
the tailnet and renders 13 remote services.

Both are opt-in on both sides: no `fleet.json` means no poller and no extra
thread; `LLM_CONTROL_ENABLED=off` means nothing binds.

## 3. What is left

**A nav group per machine — the next substantial piece.** Today the fleet is a
*mode switch*: a header `<select>` picks one host and `applyFleetMode()` swaps
the page into "remote". The owner wants each machine to be a **place you
navigate to**, so controls for one cannot act on another. Sidebar gains a group
per machine — "This Mac", "llms", "Studio" — each with its own Services /
Configuration / Logs.

The shape, decided: **`self` becomes a first-class host id in the data model but
never a URL prefix.** `fleetPath()` stays the one seam and gains
`if (fleetHost === SELF) return url;`. That is how "one shape in the browser"
and "the call sites do not know a fleet exists" stay true together.

Do **not** add `/api/fleet/self/*` routes. The local endpoints return
`preflight`, `restart_needed` and `ignored_keys` that their control-API twins do
not; `RouteInventoryTests.EXPECTED` would grow a second copy of every proxied
path; and `routes/fleet.py:_is_self()` exists precisely to stop the hub pointing
at itself.

Seven commits, each releasable:

1. **`self` is a name the backend reserves.** `fleet.SELF_ID`; `validate()`
   refuses it as a peer id — otherwise a peer called `self` shadows the local
   machine in `fleet.get()`, `CACHE.one()` and every nav lookup; `self_entry()`;
   `api_fleet()` prepends it, built from `public_api.snapshot(...)` in-process
   rather than through `FleetCache`, which would call back into the app that
   started it. `_known`/`_controllable` 404 it explicitly. No new routes.
2. **The services catalog becomes data.** `public_api.services` gains `ports`
   and `config_section` (both already on `SERVICES` via `app._slot_service`);
   `/api/status` and `fleet.status_from_snapshot` both gain `catalog`.
   `StatusAdapterTests` pins that the two agree. Nothing reads it yet.
3. **One services renderer.** `renderServices(hostId, payload)` replaces both
   the Jinja cards and `renderRemoteServices`. **Element ids become host-scoped
   in this same commit** — `card-<host>-<name>` — or `getElementById('card-llm-a')`
   returns whichever section rendered first. Layout keys in `shell.js` become
   per-host too, or reordering one machine reorders another.
4. **The sidebar grows a group per machine.** A `<template id="host-nav-group">`
   cloned per host, `data-host` stamped on the group and every `.tab-btn`;
   `showTab(tab, host = fleetHost)` — the default is what preserves the 108
   inline handlers. Delete the header picker: two ways to answer "which machine"
   can disagree. **Peer groups simply do not contain the local-only entries** —
   a control that is not in the DOM cannot fire, which is stronger than
   `disabled`.
5. **One poll for all sections.** `poll()` fetches `/api/fleet` once and renders
   every section; the local-only extras run only when the active host is `self`.
   Render `stale_for_seconds` on every peer section *always* — one visible
   section made staleness obvious, four side by side do not.
6. **Isolation hardening.** Mutating functions take an explicit host:
   `svcAction(host, name, action, btn)`, `bulkAction(host, action)`,
   `saveCfgSection(host, section, btn)`. `svcAction` refuses when
   `btn.closest('[data-host]').dataset.host !== host`, so a card rendered for
   one machine cannot fire against another after a switch. `bulkAction` must
   scope its DOM scan — unscoped it is Stop All on every machine in the fleet
   from one click. `cfgCurrent`/`cfgDirty` keyed by host.
7. **A Fleet section** for the host CRUD that `/api/fleet/hosts` already serves
   and nothing renders. Where the Studio gets added.

**Two smaller things:**

- **The remote config tab has never been exercised.** It is reachable now, but
  only when a peer has `control: true`, and `llms` is deliberately
  `control: false`. Turning it on is the owner's call. When it happens, the
  case worth watching is a Metal hub against a CUDA peer: the form should render
  *the peer's* fields, say how many are hidden and why **in the peer's own
  words**, and badge settings the peer has that this build does not.
- **Per-slot persona keys work but have no UI.** `LLM_B_THINK_TEMP` resolves in
  front of `THINK_TEMP` through `backends/proxies.py`'s prefix mechanism, and
  nothing declares them in `CONFIG_FIELDS`, so they are env-file-only. That may
  be deliberate — 36 keys × 2 slots would bloat the form — but it is not
  written down anywhere as a decision.

## 4. Gotchas that cost real time

Each of these was found the hard way. None is obvious from the code.

- **The stack directory must not sit under `~/Documents`, `~/Desktop` or
  `~/Downloads`.** TCC-protected: a LaunchAgent cannot execute anything there
  and every service fails with `Operation not permitted`, while the file
  permissions, ownership and plist are all correct — and the same script runs
  fine from a terminal, because an interactive shell has the grant and launchd
  does not. `docs/mlx-macos.md` has the full account.
- **`install.sh` needs no root on macOS in the user domain**, and demanding it
  would put root-owned plists in a user LaunchAgents directory that `launchctl`
  then refuses to bootstrap.
- **The two platform branches of `install.sh` drift.** The launchd half once
  installed all thirteen services ungated and swept none.
  `InstallParityTests` now asserts parity against `setup_engine`'s own component
  map; keep it green rather than adding a second list.
- **`unittest.mock.patch` mutates module state**, so a patch entered on a worker
  thread can overlap another test's and leak a canned value into whatever runs
  next. **Run `bash test.sh` twice** — it has caught a real ordering bug.
- **The manager caches its Jinja template and stamps assets with the commit
  hash.** An edited template or script needs a manager restart *and* a forced
  reload before a browser shows it. Editing and re-testing without both is how
  you conclude a working fix does not work.
- **A fine-tune's dataset is built by `scripts/finetune/`, and its runs live at
  `/mnt/LLMs/unsloth/runs/<name>/`, not in this repo.** `corpus.py report --run
  <name>` prints everything about one in a single call. The three defaults in
  `train.py` that look arbitrary — explicit `target_modules`, a response part of
  `</think>\n\n`, and `reasoning_effort medium` — each prevent a specific,
  documented failure; `docs/fine-tuning.md` says which.
- **`llamacpp.build(..., said=[])` used to throw the caller's list away.**
  `said or []` swapped an empty list for a fresh one, and every caller passes an
  empty list, so no message any option wrote — an ignored `--device`, an inert
  `--fit-ctx`, a rejected n-gram mode — had ever reached the journal. Fixed to
  `said if said is not None`; `SaidPropagationTests` pins it. If you see notes in
  the journal that were never there before, this is why.
- **`document.hidden` is true in a headless browser pane**, and `poll()` returns
  early on it. A blank fleet view there is the visibility guard, not a bug.

## 5. How to verify

```bash
bash test.sh && bash test.sh      # 984+ tests, both runners
```

The tools that make a change here safe rather than hopeful:

- **Argv equivalence.** `tests/launcher_harness.py` with two golden files:
  `launcher-argv.golden.json` (empty `models/`) and
  `launcher-loaded-model.golden.json` (model files present, `budget.py` stubbed
  to record the question). The second pins `--mmproj` and the memory-fit report,
  neither of which the first can see. Diff both before and after any launcher or
  registry change.
- **`tests/test_slot_equivalence.py`** runs the real shell against the registry
  across configurations chosen where the shell branches.
- **`tests/platform_harness.py`** — `as_linux()` / `as_darwin()` run either
  platform's code path on either host. `LLM_STACK_PLATFORM` does the same for
  the shell, and `platforms.detect()` honours it so the two cannot disagree.
- **`tests/test_fleet.py`** — a peer is two Flask `test_client()`s spliced in at
  `patch.object(fleet, "fetch", ...)`. No second machine required. It exercises
  the logic and **nothing about HTTP, latency or DNS**, so anything network-shaped
  needs a real peer.
- **`tests/test_slot_registry.py`** pins every derived table against the literal
  it replaced, `stack-services.sh` against the registry, and the two install
  branches against each other.

For fleet work specifically, verify against the real `llms`, not a stub. The
stale-peer path — point a peer at a closed port mid-poll, confirm the card goes
**stale with its last payload and an age** rather than blank — has now been run
once for real and should be re-run after any change to `FleetCache`.

## 6. Conventions

Load-bearing and tested.

- **Reach behaviour through the module** — `core.read_env()`, not
  `from core import read_env`. `ModuleBoundaryTests` enforces it, and `fleet`
  and `control_api` are in that list because the fleet suite depends on
  substituting `fleet.fetch`.
- **Read generously, write strictly.** Legacy env names are readable forever and
  writable never, so they drain out as settings are re-saved.
- **A helper must never stop a backend from starting.** `backend-preflight.sh`
  degrades to permissive when `budget.py` cannot form an opinion. Command
  *construction* is the exception — it must be right or not run.
- **Three Flask apps, not one with checks.** 8077 carries everything and is
  local only; 8078 is read-only; 8079 can only write. The read app cannot serve
  a write because it never learned the route. `StateApiAppTests` and
  `ControlApiAppTests` assert both directions.
- **Proxied paths are a whitelist of fixed rules, keyed by method**, never a
  forwarder. A generic `/api/fleet/<id>/<path:rest>` on the unauthenticated
  manager port would let anything that reaches 8077 issue authenticated requests
  to every machine in the fleet.
- **Never touch ports, model aliases or persona semantics.**
  `docs/pi-forge-scheduling-contract.md` pins 8003/8004/8008/8012 and the
  `think`/`chat`/`code` aliases as an external contract; pi-forge and open-webui
  depend on them.
- **Do not run tree-wide substitutions in a deployed checkout.** `config/` holds
  gitignored operator data beside source; doing this on `llms` overwrote live
  config three times in one session.
- **Say what was measured, not what was assumed.** The docs here cite the
  failure that motivated each design. Keep doing that.
