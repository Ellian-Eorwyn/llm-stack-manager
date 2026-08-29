#!/usr/bin/env python3
"""macOS / Apple Silicon: launchd, ioreg, vm_stat.

Four things here are not translations of the Linux implementation but answers
to questions macOS poses differently. They are the reason this file is longer
than `linux.py` despite doing less.

1. `launchctl list <label>` does not emit JSON. It emits an OpenStep plist
   (`"PID" = 1234;`). The code this replaces called `json.loads` on it, which
   raised every time, was caught, and returned a PID of 0 -- so on macOS every
   service read as inactive, permanently, with no error anywhere.

2. launchd has no `NRestarts`. The code this replaces returned a hardcoded 0,
   which does not read as "unknown", it reads as "healthy", and silently
   disabled the flap detection that `docs/service-health.md` calls the most
   valuable signal in the health model. Here the count is synthesised by
   watching the main PID change between polls.

3. Unified memory means there is no separate pool of device memory to report,
   and IOAccelerator publishes allocation driver-wide with no per-process
   breakdown. `gpu_compute_apps` therefore returns `None` -- "cannot say" --
   rather than `[]`, which would claim the GPU is idle.

4. Temperature and power are not readable without elevated privileges, so they
   are reported as `None`. A monitoring daemon should not need root, and a
   fabricated 0 would render as a cold, idle GPU.
"""

from __future__ import annotations

import ipaddress
import os
import plistlib
import re
import shutil
import subprocess
import threading

from . import base


_PLIST_INT_RE = {
    "PID": re.compile(r'"PID"\s*=\s*(\d+)\s*;'),
    "LastExitStatus": re.compile(r'"LastExitStatus"\s*=\s*(-?\d+)\s*;'),
}

_VM_STAT_PAGE_SIZE_RE = re.compile(r"page size of (\d+) bytes")
_SWAPUSAGE_RE = re.compile(r"total\s*=\s*([\d.]+)M\s+used\s*=\s*([\d.]+)M\s+free\s*=\s*([\d.]+)M")

LABEL_PREFIX = "com.llmstack"


class DarwinPlatform(base.Platform):
    name = "darwin"
    unit_noun = "service"

    def __init__(self):
        # Synthesised restart tracking. `_seen_pid` is the main PID at the last
        # poll; `_restarts` counts the times it changed without us asking.
        self._lock = threading.Lock()
        self._seen_pid: dict[str, int] = {}
        self._restarts: dict[str, int] = {}
        # Labels whose next PID change we caused ourselves. systemd's NRestarts
        # counts automatic restarts only, and an operator pressing "restart"
        # must not look like a service that cannot stay up.
        self._expected_change: set[str] = set()

    # -- launchd ------------------------------------------------------------

    @staticmethod
    def label_for(name: str) -> str:
        return f"{LABEL_PREFIX}.{name}"

    def _domain_and_plist(self, name: str) -> tuple[str, str]:
        """Which launchd domain this service lives in, and its plist path.

        Probed rather than assumed. A GUI (per-user) agent and a system daemon
        are bootstrapped into different domains and `launchctl` will not find a
        job in the wrong one. Metal-using workloads want the user domain -- GPU
        access is built around an interactive session -- but an install may
        predate that, so the domain follows wherever the plist actually is.
        """
        label = self.label_for(name)
        candidates = [
            (f"gui/{os.getuid()}", os.path.expanduser(f"~/Library/LaunchAgents/{label}.plist")),
            (f"gui/{os.getuid()}", f"/Library/LaunchAgents/{label}.plist"),
            ("system", f"/Library/LaunchDaemons/{label}.plist"),
        ]
        for domain, path in candidates:
            if os.path.exists(path):
                return domain, path
        # Nothing installed yet: name the preferred destination so callers that
        # report "not installed" point at where it should go.
        return candidates[0]

    def _job_facts(self, name: str) -> dict:
        """PID and last exit status from `launchctl list <label>`.

        Parsed with targeted regexes rather than a full OpenStep plist parser:
        we need two integers, and a hand-rolled parser for a format with nested
        dicts and unquoted `mach-port-object` values would be a liability for no
        gain.
        """
        r = self.run_cmd(["launchctl", "list", self.label_for(name)], timeout=5)
        if r.returncode != 0:
            return {"loaded": False, "pid": 0, "last_exit": None}
        text = r.stdout or ""
        facts = {"loaded": True, "pid": 0, "last_exit": None}
        match = _PLIST_INT_RE["PID"].search(text)
        if match:
            facts["pid"] = int(match.group(1))
        match = _PLIST_INT_RE["LastExitStatus"].search(text)
        if match:
            facts["last_exit"] = int(match.group(1))
        return facts

    def _note_pid(self, name: str, pid: int) -> int:
        """Fold this poll's PID into the synthesised restart count."""
        with self._lock:
            previous = self._seen_pid.get(name)
            if previous is None:
                self._seen_pid[name] = pid
                self._restarts.setdefault(name, 0)
            elif pid != previous:
                self._seen_pid[name] = pid
                # Only a transition into a *running* process is a restart. Going
                # to 0 is a stop; it is the coming back that counts, and counting
                # both would double every bounce.
                if pid:
                    if name in self._expected_change:
                        self._expected_change.discard(name)
                    else:
                        self._restarts[name] = self._restarts.get(name, 0) + 1
            return self._restarts.get(name, 0)

    def _expect_change(self, name: str) -> None:
        with self._lock:
            self._expected_change.add(name)

    # -- services -----------------------------------------------------------

    def service_state(self, name: str) -> dict:
        _domain, plist = self._domain_and_plist(name)
        facts = self._job_facts(name)
        pid = facts["pid"]
        n_restarts = self._note_pid(name, pid)
        last_exit = facts["last_exit"]

        # launchd has no "activating". A job that is loaded, has no PID and
        # exited non-zero is the launchd shape of a unit that cannot start:
        # KeepAlive is bouncing it and we are looking between bounces.
        failed = bool(facts["loaded"] and not pid and last_exit not in (None, 0))
        return {
            "installed": os.path.exists(plist),
            "active": pid > 0,
            "failed": failed,
            # Nothing in launchd distinguishes "starting" from "running", and
            # inventing the distinction would put every service through a state
            # it never actually reports leaving.
            "starting": False,
            "active_state": "active" if pid > 0 else ("failed" if failed else "inactive"),
            "sub_state": "",
            "result": "" if last_exit in (None, 0) else f"exit-code-{last_exit}",
            "main_pid": pid,
            "n_restarts": n_restarts,
        }

    def service_start(self, name: str, timeout: int = 30):
        domain, plist = self._domain_and_plist(name)
        self._expect_change(name)
        self.run_cmd(["launchctl", "bootout", f"{domain}/{self.label_for(name)}"])
        return self.run_cmd(["launchctl", "bootstrap", domain, plist], timeout=timeout)

    def service_stop(self, name: str, timeout: int = 30):
        domain, _plist = self._domain_and_plist(name)
        self._expect_change(name)
        return self.run_cmd(
            ["launchctl", "bootout", f"{domain}/{self.label_for(name)}"], timeout=timeout)

    def service_restart(self, name: str, timeout: int = 120) -> tuple[int, str]:
        self.service_stop(name, timeout=timeout)
        r = self.service_start(name, timeout=timeout)
        return r.returncode, (r.stdout + r.stderr).strip()

    def service_pid(self, name: str) -> int:
        return self._job_facts(name)["pid"]

    # -- host memory --------------------------------------------------------

    _page_bytes_cache: int | None = None

    @property
    def page_bytes(self) -> int:
        if self._page_bytes_cache is None:
            # `vm_stat` states its own page size in its header, so the number is
            # taken from the same command whose counters it scales rather than
            # from a constant that has to be remembered separately. Apple Silicon
            # is 16 KiB; the constant this replaces assumed 4 KiB and
            # under-reported every Mac swap rate by a factor of four.
            type(self)._page_bytes_cache = 16384
            try:
                out = self.run_cmd(["vm_stat"], timeout=5).stdout or ""
                match = _VM_STAT_PAGE_SIZE_RE.search(out)
                if match:
                    type(self)._page_bytes_cache = int(match.group(1))
            except Exception:
                pass
        return self._page_bytes_cache

    def _vm_stat(self) -> dict[str, int]:
        counters: dict[str, int] = {}
        try:
            out = self.run_cmd(["vm_stat"], timeout=5).stdout or ""
        except Exception:
            return counters
        for line in out.splitlines():
            key, _, value = line.partition(":")
            value = value.strip().rstrip(".")
            if value.isdigit():
                counters[key.strip()] = int(value)
        return counters

    def meminfo(self) -> dict[str, int]:
        """Reported in `/proc/meminfo`'s key names and KiB units.

        `MemAvailable` on Linux means "allocatable without swapping". The
        macOS equivalent is free plus the pages the kernel can reclaim without
        writing anything out: inactive, speculative and purgeable. Wired and
        active cannot be reclaimed, and pages held by the compressor are already
        compressed -- counting either would report memory that is not there.
        """
        counters = self._vm_stat()
        if not counters:
            return {}
        page_kib = self.page_bytes / 1024

        def kib(*keys) -> int:
            return round(sum(counters.get(key, 0) for key in keys) * page_kib)

        total_kib = 0
        try:
            total_kib = round(
                int(self.run_cmd(["sysctl", "-n", "hw.memsize"], timeout=5).stdout.strip()) / 1024)
        except Exception:
            # Every page the kernel accounts for, which is the whole of RAM.
            total_kib = kib("Pages free", "Pages active", "Pages inactive",
                            "Pages speculative", "Pages wired down",
                            "Pages occupied by compressor")

        info = {
            "MemTotal": total_kib,
            "MemAvailable": kib("Pages free", "Pages inactive",
                                "Pages speculative", "Pages purgeable"),
            "SwapTotal": 0,
            "SwapFree": 0,
        }
        try:
            out = self.run_cmd(["sysctl", "-n", "vm.swapusage"], timeout=5).stdout or ""
            match = _SWAPUSAGE_RE.search(out)
            if match:
                info["SwapTotal"] = round(float(match.group(1)) * 1024)
                info["SwapFree"] = round(float(match.group(3)) * 1024)
        except Exception:
            pass
        return info

    def swap_counters(self) -> tuple[int, int] | None:
        """`Swapins`/`Swapouts`, not `Pageins`/`Pageouts`.

        The pagein counter includes every demand-paged file read -- launching an
        application moves it by hundreds of thousands -- so using it as a swap
        signal would report a busy machine as permanently swapping. Swapins and
        Swapouts are the true analogue of Linux's `pswpin`/`pswpout`.
        """
        counters = self._vm_stat()
        if "Swapins" not in counters or "Swapouts" not in counters:
            return None
        return counters["Swapins"], counters["Swapouts"]

    # -- pids ---------------------------------------------------------------

    def pid_alive(self, pid) -> bool:
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            return False
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            # It exists; we are simply not allowed to signal it.
            return True
        except OSError:
            return False
        return True

    def pid_cmdline(self, pid) -> str:
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            return ""
        try:
            r = self.run_cmd(["ps", "-o", "command=", "-p", str(pid)], timeout=5)
        except (OSError, subprocess.SubprocessError):
            return ""
        return (r.stdout or "").strip()

    def pid_unit(self, pid) -> str:
        """The service a PID belongs to.

        There is no cgroup to read, so this can only match a job's *main* PID
        from `launchctl list`. Children are therefore invisible to it -- which
        matters, because the pooled model router's children are exactly the
        processes worth attributing. Callers already fall back to matching the
        command line when this returns "".
        """
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            return ""
        if pid <= 0:
            return ""
        try:
            out = self.run_cmd(["launchctl", "list"], timeout=5).stdout or ""
        except (OSError, subprocess.SubprocessError):
            return ""
        for line in out.splitlines():
            parts = line.split("\t")
            if len(parts) < 3 or not parts[0].isdigit():
                continue
            if int(parts[0]) == pid and parts[2].startswith(LABEL_PREFIX + "."):
                return parts[2][len(LABEL_PREFIX) + 1:]
        return ""

    # -- gpus ---------------------------------------------------------------

    def gpu_info(self) -> list[dict]:
        """One entry for the integrated GPU, from IOAccelerator.

        Matched on the generic `IOAccelerator` class rather than the concrete
        one: the concrete class is chip-generation specific (`AGXAcceleratorG13X`
        on an M1 Pro) and would need a new name for every Apple silicon
        revision. Read as a plist and parsed with `plistlib` rather than
        scraped from `ioreg`'s brace-nested text form, which keeps the manager's
        deliberate one-dependency posture intact.
        """
        try:
            r = self.run_cmd(
                ["ioreg", "-rc", "IOAccelerator", "-d", "1", "-a"], timeout=5)
            entries = plistlib.loads((r.stdout or "").encode()) if r.stdout.strip() else []
        except Exception:
            return []
        if not isinstance(entries, list):
            entries = [entries]

        # Unified memory: the GPU's memory *is* the host's memory, so that is
        # what is reported here.
        #
        # The tempting field is IOAccelerator's "Alloc system memory", and it is
        # the wrong one. It counts the driver's virtual allocations, so on this
        # 16 GiB M1 Pro it reads 18,102 MiB -- 110% of physical RAM, a negative
        # free figure, and a `gpu_vram_low` alert that can never clear. What
        # actually constrains loading a model on this architecture is host
        # memory pressure, which is what `MemAvailable` measures. The driver
        # allocation figure is kept alongside, named for what it is, rather than
        # dressed up as VRAM.
        meminfo = self.meminfo()
        mem_total = round(meminfo.get("MemTotal", 0) / 1024)
        mem_free = round(meminfo.get("MemAvailable", 0) / 1024)
        mem_used = max(0, mem_total - mem_free)

        # How much of that the GPU may actually wire down.
        #
        # Not the same as installed RAM, and the gap is large: llama.cpp on this
        # 16 GiB M1 Pro reports `MTL0: Apple M1 Pro (12124 MiB, 12123 MiB free)`
        # -- about 74%. macOS caps the GPU's wired working set, and
        # `iogpu.wired_limit_mb` is the knob. It reads 0 for "driver default",
        # which Apple does not publish, so the cap is reported when it has been
        # set explicitly and left null when it has not, rather than guessing at
        # a fraction and presenting the guess as a measurement.
        wired_limit_mib = None
        try:
            raw = self.run_cmd(
                ["sysctl", "-n", "iogpu.wired_limit_mb"], timeout=5).stdout.strip()
            wired_limit_mib = int(raw) or None
        except Exception:
            pass

        # Friendlier than IOAccelerator's class name, which is a chip-generation
        # code ("AGXAcceleratorG13X") that means nothing to an operator.
        try:
            chip = self.run_cmd(
                ["sysctl", "-n", "machdep.cpu.brand_string"], timeout=5).stdout.strip()
        except Exception:
            chip = ""

        gpus = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            stats = entry.get("PerformanceStatistics") or {}
            alloc_bytes = stats.get("Alloc system memory") or 0
            gpus.append({
                "index": index,
                "uuid": f"apple-gpu-{index}",
                "name": f"{chip} GPU" if chip else (entry.get("IOClass") or "Apple GPU"),
                "mem_used": mem_used,
                "mem_total": mem_total,
                "mem_free": mem_free,
                "mem_pct": round(100 * mem_used / max(mem_total, 1)),
                # Unified memory has no separate device pool, so this is host
                # memory being reported in a GPU-shaped slot. Said out loud in
                # the payload so a consumer is not left to infer it.
                "unified_memory": True,
                "driver_alloc_mib": round(alloc_bytes / (1024 * 1024)),
                # None means "driver default", not "no limit".
                "wired_limit_mib": wired_limit_mib,
                "util": stats.get("Device Utilization %"),
                "mem_util": None,
                # Not readable without elevated privileges. Reported as unknown
                # rather than as zero, which would render as a cold idle GPU.
                "temp": None,
                "power_watts": None,
                "power_limit_watts": None,
                "clock_sm_mhz": None,
                "clock_mem_mhz": None,
                "fan_pct": None,
                "pstate": None,
                "processes": [],
            })
        return gpus

    def gpu_compute_apps(self) -> list[dict] | None:
        # IOAccelerator publishes allocation driver-wide with no per-process
        # breakdown, and there is no Darwin equivalent of
        # `nvidia-smi --query-compute-apps`. `None` says so; `[]` would claim
        # the GPU is idle.
        return None

    # -- setup ---------------------------------------------------------------

    #: macOS 14 (Sonoma) is the floor: it is the oldest release still receiving
    #: security updates, and llama.cpp's Metal backend targets Metal 3.
    MINIMUM_MACOS_MAJOR = 14

    def detect_private_network(self) -> dict[str, str]:
        """The LAN interface and its subnet.

        `ip -j route` does not exist here. `route -n get default` names the
        interface, and `ifconfig` gives the address and a **hexadecimal**
        netmask (`0xfffffc00`), which has to be counted into a prefix length --
        BSD ifconfig has printed it that way for thirty years and there is no
        flag to change it.
        """
        try:
            route = self.run_cmd(["route", "-n", "get", "default"], timeout=5).stdout or ""
            match = re.search(r"^\s*interface:\s*(\S+)", route, re.MULTILINE)
            if not match:
                return {}
            interface = match.group(1)

            config = self.run_cmd(["ifconfig", interface], timeout=5).stdout or ""
            for line in config.splitlines():
                fields = line.split()
                if len(fields) < 4 or fields[0] != "inet":
                    continue
                ip = ipaddress.ip_address(fields[1])
                if not ip.is_private:
                    continue
                netmask = fields[3]
                prefix = (bin(int(netmask, 16)).count("1") if netmask.startswith("0x")
                          else ipaddress.ip_network(f"0.0.0.0/{netmask}").prefixlen)
                network = ipaddress.ip_network(f"{ip}/{prefix}", strict=False)
                return {"interface": interface, "address": str(ip), "cidr": str(network)}
        except Exception:
            pass
        return {}

    def preflight(self) -> dict:
        version = (self.run_cmd(["sw_vers", "-productVersion"], timeout=5).stdout or "").strip()
        try:
            major = int(version.split(".")[0])
        except (ValueError, IndexError):
            major = 0
        machine = (self.run_cmd(["uname", "-m"], timeout=5).stdout or "").strip()

        checks: dict[str, dict] = {
            "os": {
                "ok": major >= self.MINIMUM_MACOS_MAJOR,
                "value": f"macOS {version}" if version else "macOS (version unknown)",
                "required": f"macOS {self.MINIMUM_MACOS_MAJOR} or newer",
            },
            # Intel Macs have no Metal-capable unified memory worth serving
            # models from, and no Apple silicon GPU to target.
            "architecture": {"ok": machine == "arm64", "value": machine, "required": "arm64"},
            "launchd": {"ok": os.path.exists("/bin/launchctl"), "value": "launchd"},
        }

        devices = self.gpu_info()
        checks["metal_gpu"] = {
            "ok": bool(devices),
            "gpu_count": len(devices),
            "value": devices[0]["name"] if devices else "",
            "error": "" if devices else "No IOAccelerator device found",
        }
        # Translated into the planner's vocabulary rather than passed through:
        # `gpu_info()` is the live status shape and names the same numbers
        # `mem_total` / `mem_free`, which the planner does not read.
        gpus = [{
            "index": device["index"],
            "name": device["name"],
            "memory_total_mib": device["mem_total"],
            "memory_free_mib": device["mem_free"],
        } for device in devices]

        # Unified memory is the hard ceiling on model size here, so it is a
        # preflight fact rather than something to discover at load time.
        total_mib = round(self.meminfo().get("MemTotal", 0) / 1024)
        checks["unified_memory"] = {
            "ok": total_mib > 0,
            "value": f"{total_mib / 1024:.0f} GiB unified",
            "warning": ("Under 32 GiB: expect small models only"
                        if 0 < total_mib < 32 * 1024 else ""),
        }

        # llama.cpp is built from source against Metal, which needs a toolchain.
        clt = self.run_cmd(["xcode-select", "-p"], timeout=10)
        checks["xcode_tools"] = {
            "ok": clt.returncode == 0,
            "value": (clt.stdout or "").strip(),
            "error": "" if clt.returncode == 0 else "Run: xcode-select --install",
        }
        brew = shutil.which("brew")
        checks["homebrew"] = {
            "ok": bool(brew),
            "value": brew or "",
            "error": "" if brew else "Homebrew is how dependencies are installed on macOS",
        }

        network = self.detect_private_network()
        checks["private_network"] = {"ok": bool(network), **network}

        # macOS's application firewall filters by application, not by port and
        # source subnet, so there is no equivalent of the UFW rules the Linux
        # installer writes. Reported so the operator knows the manager is not
        # being protected by anything this installer set up -- it is
        # unauthenticated and must stay off untrusted networks.
        state = self.run_cmd(
            ["/usr/libexec/ApplicationFirewall/socketfilterfw", "--getglobalstate"], timeout=5)
        enabled = "State = 1" in (state.stdout or "")
        checks["firewall"] = {
            "ok": enabled,
            "active": enabled,
            "warning": "macOS filters by application, not by port: no rules are "
                       "installed for the stack's ports. Keep it on a trusted network.",
        }

        return {
            "checks": checks,
            "required": ["os", "architecture", "launchd", "metal_gpu",
                         "unified_memory", "xcode_tools", "homebrew", "private_network"],
            "gpus": gpus,
            "network": network,
            "extra": {},
        }

    def firewall_rules(self, ports: list[int], cidr: str) -> list[list[str]]:
        # Nothing is changed on the operator's behalf here; see the firewall
        # check above for why, and what it says instead.
        return []

    def accelerator_cmake_args(self) -> list[str]:
        """Metal needs no toolkit and no architecture probe.

        It ships with the OS and ggml targets it directly, so unlike CUDA there
        is no compiler to locate and no compute capability to detect -- the one
        flag is the whole configuration.
        """
        if not self.gpu_info():
            raise RuntimeError("No Metal device found; refusing to configure a CPU-only build")
        return ["-DGGML_METAL=ON"]

    CPU_ONLY_MARKERS = (
        "compiled without support for gpu offload",
        "no usable gpu found",
    )

    unified_memory = True

    # Everything about multi-GPU placement. `resolve_split_opts` collapses every
    # split mode to `none` here and drops the tensor split and the main-GPU
    # index with it, so these are not settings that behave differently on a Mac
    # -- they are settings that do nothing at all.
    inert_config_capabilities = {
        "gpu_visible_devices":
            "CUDA_VISIBLE_DEVICES is not read by a Metal build.",
        "gpu_indices":
            "One Metal device, so there is no index to choose and nothing to "
            "split across.",
        "split_modes":
            "One device, so every split mode resolves to none.",
    }

    #: Metal has no separate runtime context to stand up: the backend allocates
    #: command buffers and a residency set out of the same pool the weights go
    #: in. Materially smaller than CUDA's, and an estimate either way.
    @property
    def device_context_mib(self) -> int:
        return 128

    #: What llama.cpp calls the Metal device in `--list-devices`. It is `MTL0`,
    #: not `Metal0` -- checking for "metal" matches nothing and refuses a
    #: correctly built binary:
    #:
    #:     Available devices:
    #:       MTL0: Apple M1 Pro (12124 MiB, 12123 MiB free)
    #:       BLAS: Accelerate (0 MiB, 0 MiB free)
    DEVICE_PREFIX = "MTL"

    def verify_accelerated_build(self, probe_output: str) -> str:
        lowered = (probe_output or "").lower()
        if self.DEVICE_PREFIX.lower() not in lowered:
            return f"the build reports no {self.DEVICE_PREFIX}* device"
        for marker in self.CPU_ONLY_MARKERS:
            if marker in lowered:
                return f"the build reports: {marker}"
        return ""
