# Two platforms, one manager

Why the `if IS_MAC:` branches became a package, and what they were hiding.

## 1. The macOS support that was already here did not work

`00faf13 "Added Mac Compatibility"` is the fourth-oldest of sixty-seven commits.
Everything after it — router mode, the budget model, telemetry, the health
model, the state API, transcription, split-mode vetting — was written
Linux-first, and the macOS branches were not maintained alongside any of it.

None of that decay produced a failing test, because no test ever ran on macOS.
Four defects, all found by running the existing code on an M1 Pro:

**`launchctl list` was parsed as JSON.** It emits an OpenStep plist:

```
{
	"Label" = "com.llmstack.embed";
	"LastExitStatus" = 0;
	"PID" = 4242;
};
```

`json.loads` raised on every call, the exception was caught, and the PID
defaulted to `0`. So `state()["active"]` was `False` for every service on macOS,
permanently, and `get_pid()` always returned 0.

**`n_restarts` was hardcoded to `0`.** launchd has no `NRestarts`, and a
constant does not read as "unknown" — it reads as "healthy". This silently
disabled `health.RestartTracker`, which `docs/service-health.md` describes as
the most valuable signal in the health model.

**The page size was compiled in as 4 KiB.** `telemetry._PAGE_MIB = 4 / 1024`.
Apple silicon uses 16 KiB pages, so every swap rate reported on a Mac was a
quarter of the real figure — and that number feeds the `host_swapping` alert.

**The generated plist carried two keys launchd does not have.**
`generate_launchd_plist` emitted `WaitFor` and `Unbootstraps` to express
systemd's `After=` and `Conflicts=`. launchd ignores unrecognised keys without
complaint, so the units looked ordered and were not.

Underneath all four is one property: **every Linux-specific reader in this
codebase swallows its exception and returns an empty default.** That is correct
on Linux, where those failures are transient. As a porting strategy it is
disastrous. Measured on the M1 Pro before this package existed:

```
read_meminfo()          -> {}
budget._nvidia_gpus()   -> []
host_memory(...)        -> {'mem_total_mib': 0, 'mem_used_pct': None, ...}
```

A host with no memory, no swap and no GPUs. And because `mem_available_pct` was
`None`, neither `host_memory_low` nor `host_swapping` could fire — on a machine
that was, at the time, 93% into 22 GB of swap.

**An abstract method that has not been implemented raises. An empty dict does
not.** That is the whole argument for `web/platforms/`.

## 2. The shape

```
web/platforms/base.py     the interface; the docstrings are the contract
web/platforms/linux.py    systemd, nvidia-smi, /proc     (moved, not rewritten)
web/platforms/darwin.py   launchd, ioreg, vm_stat, ps
web/platforms/__init__.py active(), detect(), set_active()
```

Reach behaviour through the module — `platforms.active().meminfo()` — never by
binding `active` or a method to a local name. Binding captures the object at
import time and makes it unsubstitutable, which is the rule
`docs/repo-layout-and-deploy-drift.md` already states and `ModuleBoundaryTests`
already guards. Nothing in the package may import `app`.

`active()` is cached, and that matters: the Darwin adapter holds the synthesised
restart counts, which only mean anything as a series of observations by one
object. A fresh adapter per call would reset them on every poll and quietly
recreate the fixed-zero `n_restarts` this package exists to fix.

### Platform facts, not interpretation

`gpu_compute_apps()` returns raw `{gpu_uuid, pid, process_name, used_memory}`
rows. Turning those into "which unit, which model, which alias" stays in
`app.py`, so both platforms get the same labelling from the same code and the
adapters stay small enough to read.

### `None` and `[]` are different answers

- `gpu_compute_apps()` returns `None` on macOS. IOAccelerator reports allocation
  driver-wide with no per-process breakdown, so there is no Darwin equivalent of
  `nvidia-smi --query-compute-apps`. `[]` would assert the GPU is idle.
- Temperature and power are `None` on macOS, not `0`. They are not readable
  without elevated privileges, and a monitoring daemon should not need root. A
  fabricated `0` renders as a cold, idle card — indistinguishable from good
  news. The UI renders `null` as an em-dash (`gpuReading` in
  `static/js/telemetry.js`).

## 3. Two questions macOS answers differently

**Restart counts are synthesised.** The adapter remembers each service's main
PID between polls; a PID that changes into a *running* process without us having
asked is a restart. Going to `0` is a stop — it is the coming back that counts,
and counting both would double every bounce. `service_start`/`stop`/`restart`
mark the next change as expected, so an operator pressing "restart" does not
make a healthy service look like one that cannot stay up. That matches systemd's
`NRestarts`, which counts automatic restarts only.

**Unified memory is reported against host memory.** The tempting field is
IOAccelerator's `Alloc system memory`, and it is the wrong one: it counts the
driver's *virtual* allocations, so on a 16 GiB M1 Pro it reads 18,102 MiB —
110% used, a negative free figure, and a `gpu_vram_low` alert that can never
clear. What actually constrains loading a model on this architecture is host
memory pressure, which is what `MemAvailable` measures. The driver's allocation
figure is kept as `driver_alloc_mib`, named for what it is rather than dressed
up as VRAM, and the payload carries `unified_memory: true` so a consumer is not
left to infer it.

Two smaller ones:

- `meminfo()` reports `/proc/meminfo`'s key names in KiB **on both platforms**.
  Every consumer already reads those names and every test already patches that
  shape; a second vocabulary would have to be translated at every call site.
  macOS's `MemAvailable` is free + inactive + speculative + purgeable — the
  pages the kernel can reclaim without writing anything out. Wired and active
  cannot be reclaimed, and compressor pages are already compressed.
- `swap_counters()` reads `Swapins`/`Swapouts`, not `Pageins`/`Pageouts`. The
  pagein counter includes every demand-paged file read — launching an
  application moves it by hundreds of thousands — so it would report any busy
  machine as permanently swapping. Swapins/Swapouts are the true analogue of
  Linux's `pswpin`/`pswpout`.

## 4. Both paths run on both runners

`tests/platform_harness.py` provides `as_linux()` and `as_darwin()`, each
optionally stubbing that adapter's `run_cmd`, so a test can hand back canned
`systemctl` or `ioreg` output and assert on what the adapter makes of it.

This is the part that keeps the rot from returning. Without it, the systemd and
nvidia-smi paths would be exercised on Linux only and the launchd and ioreg
paths on macOS only, each free to decay on the other — which is exactly how the
four defects in §1 arrived. `tests/test_platforms.py` asserts both adapters
implement the whole interface, that `service_state` returns the same keys *and
the same types* on both, and carries a named regression test for each of the
four.

CI runs the same suite on `ubuntu-24.04` and `macos-latest`.

## 5. Setup on macOS

`collect_preflight()` used to ask one hardcoded set of questions — Ubuntu 24.04,
x86-64, systemd, an NVIDIA driver, a compatible CUDA toolkit — and require all
five. On Apple silicon **all five fail at once**, so the wizard could not
complete there, and the checks it displayed described a machine the operator was
never going to have.

The platform now contributes its own checks and names which of them are
required; `setup_engine` adds the three that are the same everywhere
(privileges, disk, connectivity). On this M1 Pro:

```
ok       : True
platform : darwin
  [ok  ] os                 macOS 26.5.2
  [ok  ] architecture       arm64
  [ok  ] launchd            launchd
  [ok  ] metal_gpu          Apple M1 Pro GPU
  [ok  ] unified_memory     16 GiB unified
  [ok  ] xcode_tools        /Applications/Xcode.app/Contents/Developer
  [ok  ] homebrew           /opt/homebrew/bin/brew
  [ok  ] private_network    192.168.4.0/22
  [FAIL] firewall
```

`firewall` fails and is advisory, exactly as the UFW check is on Linux. macOS
filters by application rather than by port and source subnet, so there is no
equivalent of the UFW rules the Linux installer writes; `firewall_rules()`
returns `[]` and the check says so, because an empty rule list must not be
mistaken for a protected host.

Two details worth keeping in mind:

- **The preflight `gpus` are not `gpu_info()`'s shape.** The planner reads
  `memory_total_mib` / `memory_free_mib`; the status payload uses `mem_total` /
  `mem_free`. Handing the planner the wrong one raises `KeyError` on the first
  model it places. `PreflightContractTests` pins this on both platforms.
- **`detect_private_network()` has to count a hexadecimal netmask.** BSD
  `ifconfig` prints `0xfffffc00` and has no flag to print a prefix length.

### The launchd domain

Services install as **per-user LaunchAgents in `gui/<uid>`** by default, with
plists under the service user's `~/Library/LaunchAgents`. `LLM_LAUNCHD_DOMAIN=system`
restores the old `/Library/LaunchDaemons` behaviour.

A LaunchDaemon runs at boot with no user session, and Metal is built around an
interactive one. Every tool that serves models on a Mac has converged on running
a host-native process in the user's session — Docker ships vLLM for macOS that
way because there is no GPU passthrough for Metal in containers, and GPUStack
dropped macOS entirely at v2 for the same reason.

It is also settled empirically on this hardware: an MLX embedding server has
been serving from a per-user LaunchAgent for days.

### Building against the accelerator

`dependencies.json` used to carry `-DGGML_CUDA=ON` as a literal, and the builder
followed it with an nvcc lookup and an `nvidia-smi` compute-capability probe.
On a Mac that path did not fail with "wrong platform" — it failed with "Could
not detect NVIDIA GPU compute capability", which reads as broken hardware.

`accelerator_cmake_args()` now supplies them: `-DGGML_CUDA=ON` plus the located
compiler and detected architectures on Linux, `-DGGML_METAL=ON` on macOS, where
Metal ships with the OS and there is nothing to locate or probe. It raises
rather than returning `[]` if the accelerator is unusable — a silent fallback
produces a CPU-only build that starts, serves, and is an order of magnitude
slower with nothing saying why.

`scripts/install-system-dependencies.sh` grows a macOS branch that installs the
build prerequisites with Homebrew. It **refuses to run as root** there:
Homebrew will not run as root, and the services it is preparing for are
per-user agents.

## 6. What is not done yet

- **Logs.** `journalctl` is still called directly in `telemetry.py` (seed and
  follow), `routes/public.py` and `app.py`. The launchd plists already redirect
  to `logs/<unit>.stdout.log`, so the replacement is a file tailer, and
  `telemetry.parse_line()` is already a pure function over a line.
- **`pid_unit` cannot see children on macOS.** There is no cgroup to read, so it
  matches main PIDs only — and the pooled router's children are exactly the
  processes worth attributing. `label_gpu_process`'s command-line fallback
  carries more weight there as a result.
- **The launchd domain.** Services are still bootstrapped into whichever domain
  their plist is found in, preferring the user domain. Metal access is built
  around an interactive session and the ecosystem has converged on LaunchAgents,
  but this has not been verified on hardware yet; see the plan's open items.
- **The budget model** still charges `CUDA_CONTEXT_MIB` per device and treats
  VRAM and host RAM as separate budgets. On unified memory they are one pool,
  gated by `iogpu.wired_limit_mb`.
- **The setup wizard** still fails preflight on Apple silicon: five of its nine
  required checks (`os`, `architecture`, `systemd`, `nvidia_driver`,
  `cuda_compatibility`) cannot pass there.
