# Handoff: macOS support and the slot simplification

Branch `macos-platform-support`, nine commits on top of `da66564`, 715 tests
green on macOS. **Nothing is pushed.** Read §1 and §2 before touching anything.

---

## 1. What this is for

Two goals, in this order.

**One repo that runs properly on both Linux/NVIDIA and macOS/Apple Silicon.**
Not a fork. Every existing Linux capability stays; Apple Silicon becomes a
first-class target using MLX and Metal-backed llama.cpp. The hardware is a Linux
box with 2× RTX 3090 (`llms`, in production), an M5 Ultra Mac Studio arriving,
and a 16 GB M1 Pro used for small-model testing.

**A slot model that says what it means.** The owner's words:

> What I really want are 2 primary slots for larger llms that each have proxies
> for delivering the chat, think, code configurations (chat is no think/instruct,
> code is thinking with stricter temp, think is thinking with looser temp). Then
> a Task model for one small model that can run concurrently if vram permits.
> Then different slots for embed, ocr, and transcribe. The dense vs moe isn't
> really relevant, because the model type is irrelevant, it's just about having
> multiple slots available to run multiple models at once, especially on the
> linux machine. The model router is great in theory, but i'd love for it to be
> more configurable so I can decide if I want an embed model and transcribe model
> on different gpus, for instance.

Decisions taken with the owner, which are settled and should not be relitigated:

| Question | Decision |
|---|---|
| embed2 / rerank | Drop embed2, keep rerank. Aux set: embed, rerank, ocr, transcribe. |
| Slot naming | Peers: `llm-a` / `llm-b`, identical proxy trios, no hierarchy. |
| Persona tuning | Per-slot keys with today's shared keys as fallback. |
| GPU indices | Always absolute — stop per-slot `CUDA_VISIBLE_DEVICES` renumbering. |
| Router config | Per member: GPU placement, pooled-or-dedicated, residency, context/cache. |
| Config migration | Read old keys, write new ones, via the existing `LEGACY_ENV_KEY_MAP`. |
| Mac components | Phased; anything with no Apple Silicon story is capability-gated, not deleted. |
| Multi-host | Monitor first, control later. Not started. |

The full plan, including the phasing this handoff continues, is at
`~/.claude/plans/can-we-first-make-zippy-clock.md`.

## 2. The two hard constraints

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

**Rename services and config keys; never ports, aliases or persona semantics.**
The personas already match what the owner described — only that both slots
should get the full trio, independently tunable.

**2.2 `llms` is in production.** It serves real traffic. None of this branch has
run there. Service renames are *not* covered by the config migration — units on
that box are named `chat-backend-dense`. `scripts/backup-active-stack.sh` and
`scripts/rollback-to-backup.sh` exist for the cutover; the README's "Production
Cutover With Rollback" section is the procedure.

## 3. What landed

### 3.1 Platform layer (`3d5707b`)

`web/platforms/` — an interface plus `linux.py` and `darwin.py`.
`core.ServiceManager`, `core.read_meminfo`, `health.pid_alive`,
`app.get_gpu_info` and `budget`'s readers are one-line delegations now.

It exists because macOS support added in `00faf13` — the 4th-oldest of 67
commits — was never run in CI and rotted through ~63 commits of Linux-first
work, into **four defects that broke nothing loudly**:

- `launchctl list` was parsed as JSON. It emits an OpenStep plist, so
  `json.loads` raised every time, was caught, and the PID defaulted to 0 —
  **every service read as inactive on macOS, permanently**.
- `n_restarts` hardcoded to `0`, which reads as "healthy" rather than "unknown"
  and silently disabled flap detection.
- `_PAGE_MIB = 4/1024` on a 16 KiB-page architecture — Mac swap rates under-
  reported 4×.
- The generated plist emitted `WaitFor` and `Unbootstraps`, neither of which is
  a real launchd key, so service ordering silently did not exist.

Underneath all four: every Linux-specific reader swallows its exception and
returns an empty default. On a Mac the manager reported a host with no memory,
no swap and no GPUs while the machine was 20.9 GB into swap.

**The rule that follows: a missing method raises; an empty dict does not.**
`tests/platform_harness.py` runs *either* adapter on *either* host so neither
path can rot again, and CI now runs `macos-latest` alongside `ubuntu-24.04`.

Design decisions worth not undoing:
- GPU memory on macOS is reported against **host** memory. IOAccelerator's
  `Alloc system memory` counts virtual allocations — 18,102 MiB on a 16 GiB
  machine, i.e. 110% used and an unclearable `gpu_vram_low`. Kept as
  `driver_alloc_mib`, named for what it is.
- Temperature and power are `null`, not `0`, on macOS: unreadable without root,
  and a fabricated zero renders as a cold idle card. The UI shows an em-dash.
- `gpu_compute_apps()` returns `None` on macOS, not `[]`. There is no per-process
  device-memory accounting; `[]` would assert the GPU is idle.

### 3.2 Wizard on Apple Silicon (`8fcb4e3`)

`collect_preflight()` required `os`, `architecture`, `systemd`, `nvidia_driver`
and `cuda_compatibility` — **all five fail on Apple Silicon**, so the wizard
could not complete. Platform-dispatched now; every required check passes on the
M1 Pro. Services install as **user-domain LaunchAgents** (`gui/<uid>`), with
`LLM_LAUNCHD_DOMAIN=system` for the old behaviour.

### 3.3 MLX engines (`06ea2c3`)

Folded in 651 uncommitted lines that had been running on the M1 Pro for nine
days at `~/Applications/LLMs/llm-stack-manager` (a separate checkout, still
live). MLX embeddings and Parakeet transcription, selectable per slot via
`EMBED_ENGINE` / `TRANSCRIPT_ENGINE`, plus `scripts/install-mlx-runtime.sh`
which did not exist.

**A live defect fixed here:** the manager reported both MLX services *degraded
while they served correctly*, because it probed `/props` (which the MLX server
404s) and expected `{"status":"ok"}` where they return `"healthy"`.
`health.ENGINE_PROBES` carries the per-engine definition. Do **not** fake a
`/props` payload — telemetry parses it for slot accounting and would produce
numbers about a server with no slots.

### 3.4 llama.cpp on Metal (`04547d3`)

Verified end to end: 0.5B Q4_K_M at **99.7 tok/s**, GPU 0% → **93%** measured
through the adapter, and `probe_props`/`probe_slots`/`probe_metrics` all reading
it unchanged. Three of the four changes were only findable by running it:

- **The device is `MTL0`, not `Metal0`.** The build check required the substring
  `cuda`; rewriting it to look for `metal` still refused correct builds.
- `--device CUDA0` is the shipped default and fails at load on Metal — dropped
  with a reason, not translated.
- Split modes collapse to `none`: one device, one pool.

The budget model gained a unified-memory mode. `CUDA_CONTEXT_MIB = 400` is now
`platform.device_context_mib` (Metal: 128), and the fit check branches — CUDA
errors and the unit dies; unified memory *succeeds* and pages, so the verdict is
`memory_overcommit_swaps` and its text says "this will not fail to start".

### 3.5 The launcher work (`e351cb2`, `c602efd`)

**macOS ships bash 3.2.57**, where expanding an empty array under `set -u` is an
error. Six of the nine launchers died on `"${SPEC_ARGS[@]}"` before reaching
llama-server. 72 expansions now use `${A[@]+"${A[@]}"}`.

Verifying that needed something that did not exist. `tests/launcher_harness.py`
runs a launcher in a sandboxed copy of the stack whose `LLAMA_SERVER_BIN` prints
its arguments; `tests/launcher-argv.golden.json` pins the result.
**This golden file is the safety net for everything that follows** — it is what
makes launcher consolidation a safe refactor rather than a hopeful one.

Then `web/backends/` — slots as data, one launcher (`scripts/start-backend.sh`).
Four aux slots migrated: 290 lines of shell → 61, argv byte-identical.

### 3.6 The simplification (`8f448f3`, `7d37200`)

Retired: `think`, `nothink`, `embed2`, `chat-backend`, `chat-backend-moe`,
`switch-chat-model.sh`, `BUILTIN_CHAT_VARIANTS`, `CHAT_SECONDARY_*`, the
`Conflicts=` triangle. **−1,224 lines net.** Services 16 → 15, config fields
587 → 561.

The reasoning that matters: `chat-backend-moe` and `chat-backend2` were two
different answers to "what is a secondary backend" — an *alternative* model
sharing port 8010, versus a *concurrent* slot owning 8020. The owner wants
concurrency, so the first retired.

This also closed a macOS hazard: `cross-platform.sh` cannot enforce `Conflicts=`
(launchd has no equivalent, so it became an XML comment), which meant
`switch-chat-model.sh` was the only thing stopping dense and moe both binding
8010 on a Mac.

Three bugs fixed in passing:
- `/api/switch` could never run — `models = models.load_custom_models()` shadows
  the module and raises `UnboundLocalError`. Repaired and repointed at the
  primary slot's own keys; the custom-model catalogue and its UI buttons still
  work, they configure a slot now.
- `update.sh` restart lists named `chat-proxy` but not `chat-proxy2`.
- Legacy `CHAT_MOE_*` keys now backfill `CHAT2_*`, so a host with a MoE model and
  no second slot gets it promoted rather than orphaned. `normalize_env_keys` only
  backfills, so an existing `CHAT2_*` is never overwritten. Both directions
  tested.

---

## 4. Where to be suspicious

Things a fresh reader should check rather than trust.

**4.1 Nothing on this branch has run on `llms`.** All verification was on the
M1 Pro plus unit tests. `install.sh`, `update.sh`, `validate.sh` and the shell
scripts changed substantially (~1,191 insertions / 1,358 deletions across
`scripts/` and the top-level scripts). The systemd path is exercised only by
`bash -n` and the sandboxed harness.

**4.2 The retirement sweep is destructive by design.**
`install.sh` now unconditionally runs:

```bash
for unit in think nothink embed2 chat-backend chat-backend-moe; do
    systemctl disable --now "${unit}" ...
```

That is deliberate — a host that predates the wizard would otherwise keep a unit
whose launcher no longer exists. But it means **the first `install.sh` run on
`llms` stops those units.** Confirm nothing on that box still depends on them,
especially anything pointed at port 8010 expecting the MoE model.

**4.3 Argv equivalence is proven, behaviour is not.** The golden file proves the
command line is unchanged. It does not prove the *units* are wired the same —
`After=`, `Wants=`, boot ordering and the removed `Conflicts=` all changed and
are only checked by tests, not by a running systemd.

**4.4 Unverified claims I did not confirm:**
- Whether a LaunchDaemon can reach Metal. The user-domain default is right on the
  evidence (an MLX server has served from a user agent for days), but the
  negative was never tested.
- Per-member router *residency* pinning may not exist in the pinned llama.cpp.
  **Confirm before building a setting for it.**
- The Mac budget ceiling is optimistic when `iogpu.wired_limit_mb` is unset:
  llama.cpp reports 12,124 MiB usable of 16,384 installed, and macOS's default
  fraction is unpublished. Stated in code and docs; do not paper over it.

**4.5 Mistakes I made and caught — the same trap is open.** Twice I rewrote a
function from memory while "moving" it and changed behaviour: `_gpu_number`
truncated `power.draw` 20.53 → 20, and `cuda_path_version` demanded exactly two
version components, which picks the *oldest* toolkit on a host with
`/usr/local/cuda-13`. Both were caught by comparing against the original.
**When moving code, diff the output, don't re-derive it.**

**4.6 A stale duplicate exists.** `~/Applications/LLMs/llm-stack-manager` is a
separate checkout, two commits behind, still serving MLX embeddings on 8005 and
transcription on 8014 as LaunchAgents. Its uncommitted work is now in this
branch, but **it is still running and its `web/core.py` predates the platform
layer.** The owner said it can be replaced; that has not been done.

---

## 5. Next steps

Continuing the plan's phasing. Each step should leave the tree releasable.

**Step 3 — migrate the chat and task slots into `web/backends/slots.py`.**
Keep current names and keys; rename nothing yet. This is the risky refactor and
should be done alone.

- `spec.Flag.keys` already models the three-level fallback chain
  (`CHAT_PRIMARY_*` → `CHAT_DENSE_*` → `CHAT_*`); the `!` prefix marks an
  absolute key.
- ~1,100 lines of near-duplicate shell collapse to ~80 lines of `Slot` data.
  `start-chat-backend-dense.sh` and `start-chat-backend2.sh` differ by 22 lines
  after prefix substitution.
- **Verify with the golden file**: `llm-a`'s command line must equal the old
  `chat-backend-dense`'s argument for argument. Show that diff before deleting
  anything.
- The task slot's speculative-decoding surface is the largest single piece. If
  it resists, leave it a shell launcher behind the registry rather than
  blocking the rest.

**Step 4 — collapse the six copies of the slot map.** The same
slot→prefix→unit relationship is written out independently in
`config_fields.SHARED_CHAT_BACKEND_RESTART`, `app.SERVICE_ENV_PREFIXES`,
`telemetry.BACKEND_TARGETS`, `budget.BACKEND_PREFIXES`, `public_api`'s
model-path fallbacks and `setup_engine`'s component maps — plus shell copies in
`restore-active-stack.sh`, `stack-services.sh`, `activate-selected-stack.sh`,
`llm-stack-manager` and `update.sh`. Make `slots.py` the source and derive the
rest. **This is what makes step 5 a one-line change instead of a twelve-file
sweep.**

**Step 5 — rename** to `llm-a`/`llm-b` and `LLM_A_*`/`LLM_B_*`, with
`LEGACY_ENV_KEY_MAP` carrying the old names. Ports and aliases unchanged (§2.1).

**Step 6 — proxy registry and per-slot personas.** There is no proxy registry;
the literal `2` appears in 13 places and `start-chat-proxy2.sh` is a hand-copied
duplicate that re-exports only **seven** keys. Every `THINK_TEMP`,
`CODE_REASONING_EFFORT` is *shared between both proxies* — there is no
`THINK2_TEMP`. Add per-slot keys falling back to the shared ones, which is the
idiom the proxy already uses (`THINK_TEMP` → `CHAT_TEMP` → default). Also make
`MEMORY_GATEWAY_ENABLED` per-slot; `start-chat-proxy2.sh:82` hardcodes it off.

**Step 7 — router per-member config.** Mostly *surfacing*, not building:

- `render-models-ini.py:88-92` already maps `N_GPU_LAYERS`/`MAIN_GPU`/`DEVICE`/
  `SPLIT_MODE`/`TENSOR_SPLIT` into the preset per member. `OCR_DEVICE=CUDA1`
  already pins OCR to the second card today.
- The `ASR` member already has the full 14-field UI surface. **EMBED and RERANK
  have no `*_MAIN_GPU` or `*_DEVICE` at all.** Generalise the ASR block to every
  member, the way `_clone_chat_backend_field` generalises the chat block.
- `MODEL_ROUTER_GPU_VISIBLE_DEVICES` cannot be per-member —
  `CUDA_VISIBLE_DEVICES` is process-level and the router spawns children itself.
  Relabel it as the *superset* the router may touch.
- Per-member pooled-or-dedicated: a `<MEMBER>_POOLED` toggle, with
  `MODEL_ROUTER_MEMBERS` derived. That default string is duplicated in **nine**
  places.
- **Absolute GPU indices** (decided): today `OCR_GPU_VISIBLE_DEVICES=1`
  renumbers so host GPU 1 becomes `CUDA0` for a dedicated unit, while under the
  router indices are absolute — the same `OCR_MAIN_GPU` means different cards in
  the two modes. Drop the renumbering. **This changes placement for existing
  dedicated units on `llms` and must be called out at cutover.**
- `split-mode = tensor` silently discards placement
  (`render-models-ini.py:326-328` pops `tensor-split` and `main-gpu`). The UI
  must show that.
- `MEMBER_PORTS` is the only one of the three member tables with no test; give it
  the key-set assertion the others have, allowing for the portless `ASR`.

**Step 8 — docs and cutover notes.**

**Then, from the original plan:** fleet view (federating the existing read-only
`/api/v1` across hosts — needs *no* change on `llms`, its API already serves what
a controller wants), and after that an authenticated control channel on a
**third** listener. Do not add writes to 8078: its safety property is that
mutating rules *do not exist* on that app, asserted structurally by
`StateApiAppTests`.

## 6. How to verify anything here

```bash
bash test.sh                       # 715 tests, green on macOS and Ubuntu
```

- **Argv equivalence** — the most valuable tool in the repo:
  `tests/launcher_harness.py` + `tests/launcher-argv.golden.json`. Any launcher
  change should be diffed against it before and after.
- `tests/platform_harness.py` — `as_linux()` / `as_darwin()` run either
  platform's code path on either host. Use it rather than skipping tests.
- `tests/test_deploy.py` asserts `deploy.BACKEND_SENSITIVE_PATHS` matches
  `update.sh`'s copy. `RouteInventoryTests` pins 84 URL rules;
  `ModuleBoundaryTests` forbids importing `app` and binding module functions.
- **The live box** answers at `http://llms:8078/api/v1/snapshot` — GPU state,
  per-slot context, service health, and a redacted config section. Useful for
  before/after comparison across the cutover.
- On the M1 Pro: `deps/llama.cpp/build/bin/llama-server` is a working Metal
  build, and `models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` (469 MB) is a test model
  that loads in seconds.

## 7. Conventions this repo holds to

Worth reading before writing code here; they are load-bearing and tested.

- **Reach behaviour through the module** — `core.read_env()`, not
  `from core import read_env`. `ModuleBoundaryTests` enforces it.
- **Read generously, write strictly.** Legacy env names are readable forever and
  writable never, so they drain out as settings are re-saved. See
  `docs/context-accounting-and-config-surface.md` §4.
- **A helper must never stop a backend from starting.** `backend-preflight.sh`
  degrades to permissive when `budget.py` cannot form an opinion. Command
  *construction* is the exception — that must be right or not run.
- **Say what was measured, not what was assumed.** The docs in this repo cite
  the failure that motivated each design; keep doing that.
