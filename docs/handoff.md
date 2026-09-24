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

**The M5 Ultra Mac Studio** — arrived, 96 GiB unified. Same stack directory;
LaunchAgents for the manager, `llm-a` (Qwen3.8 27B on 8010), `llm-a-proxy`
(8003/8004/8008), `embed` (Qwen3-Embedding-4B, 8005), `task` (Qwen3.5 9B,
8007), `rerank` (Qwen3-Reranker-4B Q8_0 from `Voodisss/…-GGUF-llama_cpp`,
8006, stopped for now) and `transcript-backend` (Parakeet TDT 0.6B v3 on MLX,
8014). No `fleet.json` yet. `bash validate.sh` passes against all of it.
Everything is on the tailnet as **studio:<port>** — the manager at
http://studio:8077, models at e.g. http://studio:8004/v1 — by
`scripts/tailscale-serve.sh` (`on|off|status`): services stay bound to
127.0.0.1 and `tailscale serve --bg` proxies tailnet traffic to each installed
service's port (tailnet only; not on the LAN). `llms` exposes its proxies
(8003/8004/8008/8012) itself; its backend and pooled aux models are
loopback-only until `sudo scripts/tailscale-serve.sh` is run there.

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

**Meeting ingestion, the owner's next goal.** New recordings (Zoom, Teams,
Apple Voice Memos, and later a `~/Documents/meetings` capture folder) should be
picked up automatically, deduplicated, transcribed with speakers, given real
names, and filed into the Obsidian vaults by Hermes as source transcripts plus
linked notes. What exists today is the speaker-labelled transcript
(`diarize=true`), with enrolled voices named (`docs/voiceprints.md`; profiles
built from the owner's MacWhisper library, local only). Watching folders,
dedup and filing belong to Hermes, not this repo. `llms` (the Linux sidecar, `scripts/transcribe-server.py`)
has no diarization yet. NeMo's `SortformerEncLabelModel` loads the same
checkpoint there.

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
- **`launchctl list <label>` prints an OpenStep plist, not JSON** — on the
  shell side too. `svc_is_active` in `scripts/cross-platform.sh` fed it to
  `json.loads`, so every launchd service read as inactive (affecting
  `update.sh`, `restore-active-stack.sh` and `validate.sh`). It now parses
  `"PID" = N;` the way `platforms/darwin.py` does; `test_validate_script.py`
  runs both against a stub `launchctl`.
- **`config/llm-stack.env` sets `STACK_DIR`.** A script that sources it and then
  sources a sibling via `${STACK_DIR}` picks up the *installed* tree's copy,
  not its own — which is how a worktree's `validate.sh` silently ran the old
  helper. Source siblings before the config.
- **`manage-transcript-service.sh status` exits 3 for "stopped"** (LSB). On a
  host with no transcript-backend unit, `app.get_service_status` goes through
  it; reading every non-zero exit as `failed` put a standing "Transcription has
  failed" alert on any Mac with `TRANSCRIPT_ENABLED=off`.
- **A Mac answers "what is using the GPU" from `proc_pid_rusage`.** There is
  no `nvidia-smi`; `darwin.gpu_compute_apps` reads each model server's memory
  (no root, same-user only) and attributes it by walking the process tree up
  to the launchd job; `pid_unit` does the same walk, so router children are
  attributed too. The figure is `phys_footprint` **plus the resident pages of
  mapped `.gguf`/`.safetensors` regions** (`proc_pidinfo` region walk):
  footprint leaves out an mmap-loaded model's clean weight pages (task read
  5.4 GiB holding 12.4), and `resident_size` is useless once Metal wires a
  process's buffers (llm-a read 15 GiB on it while holding 37).
- **llama.cpp unpins a model's Metal buffers 3 minutes after its last
  request** (`GGML_METAL_RESIDENCY_KEEP_ALIVE_S`, `ggml-metal-device.m`). The
  memory then turns from wired into ordinary pageable pages that macOS
  compresses and swaps — which is why "used" swung from 91 GB to 52 GB with
  nothing stopped. `METAL_KEEP_MODELS_RESIDENT=on` (default on a Mac) has
  every llama.cpp launcher keep them wired for 100 days; ~0.1% CPU.
- **The Mac's header bar shows pressure, not "used".** Models, plus
  `macOS` = wired memory the models do not account for (kernel, GPU driver,
  WindowServer; ~8 GB on the Studio). Compressed pages and the file cache are
  left off because macOS reclaims them; `gpu_vram_low` / `host_memory_low`
  still read MemAvailable. Linux's card is unchanged.
- **Stop on a Mac persists across logins.** `bootout` alone lasts until the
  next login, when launchd loads every LaunchAgent and KeepAlive starts it;
  Stop now also `launchctl disable`s the job and Start `enable`s it. Check
  with `launchctl print-disabled gui/$(id -u)`.
- **On the Studio, "macOS & apps" was ~25 GB and was mostly not apps.** Of 91 GB
  used: 74 GB wired (≈65 GB of it the model servers' Metal buffers; the rest
  kernel, GPU driver, WindowServer ~1.7 GB) and ~9.4 GB compressor. Desktop
  apps are ~15 GB of footprint but mostly compressed. `vmmap` shows 21.5 GB
  of llm-a as swapped/compressed — more than the compressor holds, so partly
  GPU-wired pages outside the CPU page table, but the box is at its limit.
- **launchd has no journal.** `Platform.log_command` is the one place logs are
  read from: journald on Linux, the files named in the job's plist on macOS.
  Those lines carry no wall-clock time (llama.cpp's stamp counts from process
  start), so Mac telemetry is stamped as lines arrive and never backfills.
- **A component left out at setup has no LaunchAgent.** On a Mac,
  `service_start` runs `scripts/install-launchd-service.sh` for it first; that
  script's table must match `install_mac_service` in `install.sh`
  (`InstallLaunchdServiceTests` holds them together).
- **`update.sh` on a Mac (user domain) does not run `install.sh`.** With no
  component selection that installs an agent for every component, and launchd
  starts any KeepAlive agent at the next login. It refreshes the agents already
  in `~/Library/LaunchAgents` through `install-launchd-service.sh`, restarts only
  running services with `launchctl kickstart -k`, and the manager **last** — the
  Update button runs `update.sh` as the manager's child, so a bootout of the
  manager would kill the update before its bootstrap and leave it down.
- **Most Qwen3-Reranker GGUFs score garbage in llama.cpp** (~1e-23; missing
  `cls.output.weight`, llama.cpp#16407). The `Voodisss/Qwen3-Reranker-*-GGUF-llama_cpp`
  conversions score correctly with `--reranking`; check any other against a
  query with an obvious answer before trusting it.
- **Embedding and rerank slots need `*_UBATCH_SIZE` ≥ the longest input.**
  llama.cpp pools a sequence inside one micro-batch and forces `n_batch =
  n_ubatch`, so the shipped `512` returns HTTP 500 "input (N tokens) is too
  large" for any passage over 512 tokens. The Studio runs both at 8192
  (= `*_CTX_SIZE`); `llms` and `config/llm-stack.env.example` still ship 512.
- **mlx-audio is pinned to a git commit; the rest of the MLX runtime is not
  pinned.** Nemotron 3 Diarization landed in mlx-audio after its 0.5.5 release,
  so `install-mlx-runtime.sh` installs `9ada37c` (`MLX_AUDIO_SPEC` overrides it).
  Move back to a PyPI release once one includes the model. The Studio ran
  Parakeet on 0.5.5 and runs it unchanged on the pin. The runtime needs
  `python-multipart`, which the installer names explicitly.
- **`document.hidden` is true in a headless browser pane**, and `poll()` returns
  early on it. A blank fleet view there is the visibility guard, not a bug.
- **Upstream llama.cpp no longer accepts `--no-mmap`** (now `--load-mode none`),
  and every slot with `*_NO_MMAP=true` passes it. Bumping the pin in
  `dependencies.json` past `7e4c0a9` fails every such backend at start, on both
  platforms, until `COMMON_TOGGLES` learns `--load-mode`. Worth doing: on the
  Studio, master is no faster on short prompts but decodes 30% faster at ~100k
  context (`docs/mtplx.md`).
- **The Studio's llm-a runs at ~19 tok/s in real use, not the ~53 a short
  prompt shows**: its logged requests sit at ~160k tokens of context. MTPLX
  (an MTPLX pack as LLM A's model, `docs/mtplx.md`) measured 2.9x llama.cpp
  there, and the Studio's llm-a runs on it.

## 5. How to verify

```bash
bash test.sh && bash test.sh      # 1100+ tests, both runners
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
