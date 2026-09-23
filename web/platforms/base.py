#!/usr/bin/env python3
"""The interface every supported platform implements.

Why this exists as an interface rather than as `if IS_MAC:` branches.

The manager grew on Linux, and macOS support was added early and then not
maintained through sixty-odd commits of Linux-first work. It did not break
loudly. Every Linux-specific reader in this codebase is written to swallow its
exception and return an empty default -- `/proc/meminfo` missing returns `{}`,
`nvidia-smi` missing returns `[]` -- which is right on Linux, where those
failures are transient, and catastrophic as a porting strategy, because the
result is a manager that reports a host with no memory, no swap and no GPUs
while the machine underneath it is swapping itself to death.

Measured on an M1 Pro before this package existed:

    read_meminfo()          -> {}
    budget._nvidia_gpus()   -> []
    host_memory(...)        -> {'mem_total_mib': 0, 'mem_used_pct': None, ...}

and, because `mem_available_pct` was `None`, neither `host_memory_low` nor
`host_swapping` could fire.

An abstract method that a platform has not implemented raises. An empty dict
does not. That difference is the entire reason for this file.

The contract each method owes its callers is in its docstring here rather than
in the implementations, because the point is that both implementations answer
the same question in the same units.
"""

from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod


class Platform(ABC):
    """One host operating system, and how to ask it the questions we ask."""

    #: Short name, matching `sys.platform` vocabulary: "linux" or "darwin".
    name: str = ""

    #: What the service manager calls a unit, for use in operator-facing text
    #: ("service" reads oddly next to `systemctl`, "unit" reads oddly next to
    #: `launchctl`). Message text is not a place to leak the abstraction.
    unit_noun: str = "service"

    # -- process helpers ----------------------------------------------------

    @staticmethod
    def run_cmd(cmd, timeout=30) -> subprocess.CompletedProcess:
        """Run a command, and treat "there is no such command" as a failure.

        Every caller here already branches on `returncode`, because a probe
        that cannot answer is an ordinary outcome: a box with no NVIDIA driver
        has no `nvidia-smi`, and a Mac with no Xcode has no `xcode-select`. An
        exception makes that ordinary outcome the caller's problem, and the
        callers that remembered wrapped it in `try/except Exception` -- which
        is how `_cuda.probe` survives a host with no driver while
        `darwin.preflight` does not survive a host with no `sw_vers`.

        127 is the shell's own code for "command not found", so a caller that
        checks `returncode == 0` needs to know nothing about this.
        """
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except (FileNotFoundError, NotADirectoryError, PermissionError) as exc:
            return subprocess.CompletedProcess(cmd, 127, "", str(exc))
        except subprocess.TimeoutExpired as exc:
            return subprocess.CompletedProcess(cmd, 124, "", str(exc))

    # -- services -----------------------------------------------------------

    @abstractmethod
    def service_state(self, name: str) -> dict:
        """Load and activation state in one call.

        Returns the keys the health model reads, and every one of them every
        time:

            installed    bool  a unit definition exists on disk
            active       bool  it is running right now
            failed       bool  it stopped because it failed
            starting     bool  it is mid-launch (not yet active, not failed)
            active_state str   the platform's own word, for the detail view
            sub_state    str   ditto, finer grained; "" if the platform has none
            result       str   why it last stopped; "" if unknown
            main_pid     int   0 when not running
            n_restarts   int   cumulative restarts since the counter was reset

        `n_restarts` is the one that carries weight. The count alone proves
        nothing; a count that *climbs between polls* is a service failing to
        come up, which is the single most valuable signal in the health model
        (docs/service-health.md). A platform without a native counter must
        synthesise one rather than return a constant -- a hardcoded 0 does not
        read as "unknown", it reads as "healthy".
        """

    @abstractmethod
    def service_start(self, name: str, timeout: int = 30) -> subprocess.CompletedProcess: ...

    @abstractmethod
    def service_stop(self, name: str, timeout: int = 30) -> subprocess.CompletedProcess: ...

    @abstractmethod
    def service_restart(self, name: str, timeout: int = 120) -> tuple[int, str]:
        """Returns (returncode, combined output)."""

    @abstractmethod
    def service_pid(self, name: str) -> int:
        """The unit's main PID, or 0. Cheaper than `service_state` where the
        platform offers a narrower query; may just read `service_state`."""

    # -- host memory --------------------------------------------------------

    @property
    @abstractmethod
    def page_bytes(self) -> int:
        """Bytes per page of virtual memory.

        Read from the platform, never assumed. Apple Silicon uses 16 KiB pages
        against Linux's 4 KiB, and the constant `4 / 1024` that used to be
        compiled into the swap-rate maths under-reported every Mac by 4x.
        """

    @abstractmethod
    def meminfo(self) -> dict[str, int]:
        """Host memory in the shape `/proc/meminfo` uses: **KiB**, with at
        least `MemTotal`, `MemAvailable`, `SwapTotal` and `SwapFree`.

        Keeping Linux's key names and units on both platforms is deliberate.
        Every consumer (`telemetry.host_memory`, `budget._host_memory`, the
        build-parallelism estimate in `app.py`) already reads these names, and
        the tests already patch this exact shape. A second vocabulary would buy
        nothing and would have to be translated at every call site.

        Returns `{}` only if the platform genuinely could not be asked.
        """

    @abstractmethod
    def swap_counters(self) -> tuple[int, int] | None:
        """Cumulative (pages swapped in, pages swapped out) since boot, or
        `None` if the platform cannot say.

        Cumulative, not a rate: the caller turns two readings into a rate, and
        needs to know the counters came from the same source both times.
        """

    # -- pids ---------------------------------------------------------------

    @abstractmethod
    def pid_alive(self, pid) -> bool: ...

    @abstractmethod
    def pid_cmdline(self, pid) -> str:
        """The process's full command line, space-separated, or ""."""

    @abstractmethod
    def pid_unit(self, pid) -> str:
        """The service unit a PID belongs to, or "".

        This is what makes "which model is holding this memory" answerable.
        Matching a PID against each unit's main PID finds only the process the
        service manager started directly, and the processes holding GPU memory
        are frequently children -- the model router forks one server per
        resident model and none of them is a main PID.
        """

    # -- gpus ---------------------------------------------------------------

    @abstractmethod
    def gpu_info(self) -> list[dict]:
        """One dict per GPU. Keys, with `None` for anything unavailable:

            index, uuid, name, mem_used, mem_total, mem_free, mem_pct  (MiB)
            util, mem_util, temp, power_watts, power_limit_watts
            clock_sm_mhz, clock_mem_mhz, fan_pct, pstate, processes

        `None` rather than `0` for a reading this platform cannot take. A
        fabricated zero for temperature or power renders as a cold, idle card,
        which is indistinguishable from good news.
        """

    # -- setup ---------------------------------------------------------------

    @abstractmethod
    def preflight(self) -> dict:
        """Whether this host can run the stack, in this platform's terms.

        The setup wizard used to ask one hardcoded set of questions -- Ubuntu
        24.04, x86-64, systemd, an NVIDIA driver, a compatible CUDA toolkit --
        and require all five. On Apple silicon all five fail, so the wizard
        could not complete at all, and the checks it showed described a machine
        the operator was never going to have.

        Returns:

            checks    name -> {"ok": bool, ...detail}. The UI renders these
                      generically from `value` / `warning` / `error`, so a
                      platform is free to ask its own questions.
            required  the check names that must pass for `ok` to be true.
                      Everything else is advisory and renders as a warning.
            gpus      accelerators in the shape `plan_gpu_placement` reads:
                      `index`, `name`, `memory_total_mib`, `memory_free_mib`.
                      This is deliberately *not* `gpu_info()`'s shape -- that
                      one is for the live status payload and uses different key
                      names, and handing it to the planner raises a KeyError on
                      the first model it tries to place.
            network   {"interface", "address", "cidr"}, or {} if there is no
                      private network to bind to.
            extra     top-level keys merged into the payload, for anything
                      only one platform has.

        The caller adds the checks that are the same everywhere -- privileges,
        disk, connectivity -- so a platform only answers what it alone knows.
        """

    @abstractmethod
    def accelerator_cmake_args(self) -> list[str]:
        """The cmake flags that build ggml for this host's accelerator.

        `dependencies.json` used to carry `-DGGML_CUDA=ON` as a literal, and the
        builder followed it with an nvcc lookup and an `nvidia-smi` probe for
        compute capabilities. On a Mac that path does not fail with "wrong
        platform", it fails with "Could not detect NVIDIA GPU compute
        capability" — which reads as broken hardware rather than as a build
        configured for someone else's machine.

        Raises rather than returning `[]` if the accelerator is unusable: a
        silent fallback here produces a CPU-only build that starts, serves, and
        is slower by an order of magnitude with nothing saying why.
        """

    #: Whether device memory and host memory are the same pool. On Apple
    #: silicon they are, which changes what "it does not fit" means: CUDA fails
    #: the allocation and the unit dies, where unified memory succeeds and the
    #: host starts swapping. Loud versus slow, and a different fix each time.
    unified_memory: bool = False

    # -- logs ---------------------------------------------------------------

    #: Whether service logs carry the host's own timestamps in journald's
    #: `short-iso-precise` shape. Telemetry backfills history only from a log
    #: that says when each line was written.
    journal_logs: bool = True

    def log_command(self, unit: str, lines: int, follow: bool = False,
                    precise: bool = False) -> list[str]:
        """The command that prints the last `lines` of a service's log, and
        keeps printing new ones when `follow` is set.

        journald by default. A service manager that writes plain files instead
        overrides this; callers only ever read stdout line by line.
        """
        return (["journalctl", "-u", unit]
                + (["-f"] if follow else [])
                + ["-n", str(lines), "--no-pager",
                   "--output=short-iso-precise" if precise else "--output=short-iso"])

    #: Configuration capabilities this host does not have, and why.
    #:
    #: A control that reports success and changes nothing is worse than a
    #: missing one: the operator sets it, the UI says saved, the backend starts,
    #: and the setting is not in the command line. Naming the capability here
    #: lets `config_fields.applicable_fields` leave those controls out and say
    #: what is missing, instead of rendering a GPU-placement panel on a machine
    #: with one GPU that cannot be placed.
    #:
    #: Empty means the whole configuration surface applies.
    inert_config_capabilities: dict[str, str] = {}

    @property
    @abstractmethod
    def device_context_mib(self) -> int:
        """Fixed per-device overhead a backend pays just for existing.

        Runtime context, command buffers and allocator slack — everything
        present before a single weight is loaded. Estimated, not measured; the
        budget model reports it separately from the exact figures so it is
        clear which part of a prediction is arithmetic and which is judgement.
        """

    @abstractmethod
    def verify_accelerated_build(self, probe_output: str) -> str:
        """Check `llama-server --list-devices` output names this accelerator.

        Returns "" when the build is accelerated, or the reason it is not.

        A CPU-only llama.cpp build is not a broken build -- it compiles, starts,
        serves, and answers correctly. It is simply an order of magnitude
        slower and will exhaust host RAM on a model the GPU would have held, so
        the failure surfaces days later as "the box is slow" rather than as a
        build error. Hence a positive check for the expected backend rather than
        a check for the absence of errors.
        """

    @abstractmethod
    def firewall_rules(self, ports: list[int], cidr: str) -> list[list[str]]:
        """Commands that open `ports` to `cidr` only, or `[]` if this platform
        has no firewall this installer manages.

        `[]` is not "the host is open"; it is "nothing here will be changed on
        your behalf". The preflight check is what tells the operator which of
        those they are looking at.
        """

    @abstractmethod
    def gpu_compute_apps(self) -> list[dict] | None:
        """Raw per-process device-memory rows: `{gpu_uuid, pid, process_name,
        used_memory}` in MiB. `None` means this platform cannot attribute
        device memory to a process at all.

        `None` and `[]` say different things and callers depend on the
        difference: `[]` is "nothing is on the GPU", `None` is "do not draw
        conclusions from the absence of rows here". On macOS there is no
        `nvidia-smi --query-compute-apps`; unified memory has no device pool,
        so the rows there are each model-serving process's `phys_footprint`
        (which includes its wired Metal buffers), and `None` only when that
        cannot be read.

        Deliberately raw. Turning a row into "which unit, which model, which
        alias" is interpretation, not a platform fact, and it lives with the
        rest of the interpretation in `app.py` so that both platforms get the
        same labelling from the same code.
        """
