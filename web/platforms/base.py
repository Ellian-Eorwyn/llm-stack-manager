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
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

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
        conclusions from the absence of rows here". macOS returns `None` --
        IOAccelerator reports allocation driver-wide, with no per-process
        breakdown, so there is no Darwin equivalent of
        `nvidia-smi --query-compute-apps`.

        Deliberately raw. Turning a row into "which unit, which model, which
        alias" is interpretation, not a platform fact, and it lives with the
        rest of the interpretation in `app.py` so that both platforms get the
        same labelling from the same code.
        """
