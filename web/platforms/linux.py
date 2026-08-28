#!/usr/bin/env python3
"""Linux: systemd, nvidia-smi, /proc.

The implementations here are the ones this project has always run in
production, moved rather than rewritten. Comments explaining *why* a particular
call is shaped the way it is have been kept with the code they explain -- they
document failures that were paid for once already.
"""

from __future__ import annotations

import ipaddress
import json
import platform as _platform
import re
import shutil
import subprocess
from pathlib import Path

from . import base


# Order matters: this is the `--query-gpu` field list and the column order it
# comes back in. The first seven are what the UI has always shown; the rest are
# for API consumers, and every one of them can be `[N/A]` on some card or driver
# -- an eGPU reports no fan, a datacentre card no power limit -- so they parse to
# None rather than failing the row.
GPU_QUERY_FIELDS = [
    "index", "uuid", "name", "memory.used", "memory.total",
    "utilization.gpu", "temperature.gpu",
    "utilization.memory", "power.draw", "enforced.power.limit",
    "clocks.current.sm", "clocks.current.memory", "fan.speed", "pstate",
]

_CGROUP_UNIT_RE = re.compile(r"/([\w\-.@\\]+)\.service\b")


def _gpu_number(value: str):
    """A numeric nvidia-smi field, or None for the several ways it says N/A."""
    text = (value or "").strip()
    if not text or text.startswith("[") or text.lower() in {"n/a", "unknown"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return int(number) if number.is_integer() else round(number, 2)


class LinuxPlatform(base.Platform):
    name = "linux"
    unit_noun = "unit"

    # -- services -----------------------------------------------------------

    def service_state(self, name: str) -> dict:
        """`is-active` collapses a crashed unit into "not active", which is how
        a unit that died and a unit somebody stopped came to look identical. One
        `systemctl show` answers both questions, and costs one subprocess where
        `is_active` plus `is_installed` cost two -- the status poll runs this for
        every service every five seconds.
        """
        r = self.run_cmd(
            ["systemctl", "show", name,
             "--property=LoadState,ActiveState,SubState,Result,MainPID,NRestarts"],
            timeout=5)
        fields = {}
        for line in (r.stdout or "").splitlines():
            key, _, value = line.partition("=")
            fields[key.strip()] = value.strip()
        active_state = fields.get("ActiveState", "")

        def as_int(key):
            try:
                return int(fields.get(key, "0") or "0")
            except ValueError:
                return 0

        return {
            "installed": r.returncode == 0 and fields.get("LoadState", "") not in ("", "not-found"),
            "active": active_state == "active",
            # systemd's own definition. `Result` is reported alongside for the
            # detail view but is deliberately not part of this test: it survives
            # a later `stop`, so a unit that crashed once and was then stopped
            # on purpose would otherwise read as failed forever.
            "failed": active_state == "failed",
            # A unit that cannot start spends most of its time here rather than
            # in `failed`, because `Restart=` bounces it before anyone looks.
            "starting": active_state in ("activating", "reloading"),
            "active_state": active_state,
            "sub_state": fields.get("SubState", ""),
            "result": fields.get("Result", ""),
            "main_pid": as_int("MainPID"),
            # Cumulative since the unit was last reset. The count alone proves
            # nothing; a count that climbs between polls is a service failing to
            # come up, which is the state a panel most needs to shout about.
            "n_restarts": as_int("NRestarts"),
        }

    def service_start(self, name: str, timeout: int = 30):
        return self.run_cmd(["systemctl", "start", name], timeout=timeout)

    def service_stop(self, name: str, timeout: int = 30):
        return self.run_cmd(["systemctl", "stop", name], timeout=timeout)

    def service_restart(self, name: str, timeout: int = 120) -> tuple[int, str]:
        r = self.run_cmd(["systemctl", "restart", name], timeout=timeout)
        return r.returncode, (r.stdout + r.stderr).strip()

    def service_pid(self, name: str) -> int:
        r = self.run_cmd(
            ["systemctl", "show", name, "--property=MainPID", "--value"], timeout=2)
        try:
            return int((r.stdout or "0").strip() or "0")
        except Exception:
            return 0

    # -- host memory --------------------------------------------------------

    @property
    def page_bytes(self) -> int:
        return 4096

    def meminfo(self) -> dict[str, int]:
        data: dict[str, int] = {}
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    key, _, raw_value = line.partition(":")
                    parts = raw_value.strip().split()
                    if parts and parts[0].isdigit():
                        data[key] = int(parts[0])
        except Exception:
            pass
        return data

    def swap_counters(self) -> tuple[int, int] | None:
        counters = {}
        try:
            with open("/proc/vmstat", "r", encoding="utf-8") as handle:
                for line in handle:
                    key, _, value = line.partition(" ")
                    if key in ("pswpin", "pswpout"):
                        counters[key] = int(value.strip())
        except (OSError, ValueError):
            return None
        if len(counters) < 2:
            return None
        return counters["pswpin"], counters["pswpout"]

    # -- pids ---------------------------------------------------------------

    def pid_alive(self, pid) -> bool:
        try:
            return Path(f"/proc/{int(pid)}").exists()
        except (TypeError, ValueError):
            return False

    def pid_cmdline(self, pid) -> str:
        """How a process was actually launched, which is not always how it is
        configured."""
        try:
            return Path(f"/proc/{int(pid)}/cmdline").read_text(
                errors="ignore").replace("\x00", " ").strip()
        except Exception:
            return ""

    def pid_unit(self, pid) -> str:
        """The systemd unit a PID belongs to, read from its cgroup.

        Matching against each unit's MainPID only ever finds the process systemd
        started, and the interesting ones are often children: `llama-router`
        forks a `llama-server` per resident model, and it is those children that
        hold the VRAM. The cgroup names the unit for every process in it, parent
        or child, from one file read and no subprocess.
        """
        try:
            text = Path(f"/proc/{int(pid)}/cgroup").read_text(errors="ignore")
        except (OSError, ValueError):
            return ""
        match = _CGROUP_UNIT_RE.search(text)
        return match.group(1) if match else ""

    # -- gpus ---------------------------------------------------------------

    def gpu_info(self) -> list[dict]:
        try:
            r = self.run_cmd(
                ["nvidia-smi",
                 "--query-gpu=" + ",".join(GPU_QUERY_FIELDS),
                 "--format=csv,noheader,nounits"],
                timeout=5)
        except Exception:
            return []
        gpus = []
        for line in (r.stdout or "").strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 7:
                continue
            try:
                index = int(parts[0])
                mem_used, mem_total = int(parts[3]), int(parts[4])
            except ValueError:
                continue

            def field(position: int) -> str:
                return parts[position] if position < len(parts) else ""

            gpus.append({
                "index":     index,
                "uuid":      parts[1],
                "name":      parts[2],
                "mem_used":  mem_used,
                "mem_total": mem_total,
                "util":      _gpu_number(parts[5]),
                "temp":      _gpu_number(parts[6]),
                "mem_pct":   round(100 * mem_used / max(mem_total, 1)),
                "mem_free":  max(0, mem_total - mem_used),
                "mem_util":          _gpu_number(field(7)),
                "power_watts":       _gpu_number(field(8)),
                "power_limit_watts": _gpu_number(field(9)),
                "clock_sm_mhz":      _gpu_number(field(10)),
                "clock_mem_mhz":     _gpu_number(field(11)),
                "fan_pct":           _gpu_number(field(12)),
                "pstate":            field(13) or None,
                "processes": [],
            })
        return gpus

    def gpu_compute_apps(self) -> list[dict] | None:
        rows: list[dict] = []
        try:
            r = self.run_cmd(
                ["nvidia-smi",
                 "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                 "--format=csv,noheader,nounits"],
                timeout=5)
        # Narrow on purpose: a bare `except Exception` here once spent an unknown
        # length of time swallowing a NameError that discarded every attribution
        # while the payload still looked well-formed.
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            print(f"[llm-manager] GPU process attribution failed: {exc}", flush=True)
            return None
        for line in (r.stdout or "").strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            try:
                pid = int(parts[1])
                used = int(float(parts[3]))
            except ValueError:
                continue
            rows.append({
                "gpu_uuid": parts[0],
                "pid": pid,
                "process_name": parts[2],
                "used_memory": used,
            })
        return rows

    # -- setup ---------------------------------------------------------------

    def _os_release(self) -> dict[str, str]:
        data: dict[str, str] = {}
        try:
            for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
                key, _, value = line.partition("=")
                if key:
                    data[key.strip()] = value.strip().strip('"')
        except OSError:
            pass
        return data

    def detect_private_network(self) -> dict[str, str]:
        try:
            route = self.run_cmd(["ip", "-j", "route", "show", "default"], timeout=5)
            entries = json.loads(route.stdout or "[]")
            interface = entries[0].get("dev", "") if entries else ""
            if not interface:
                return {}
            addr = self.run_cmd(["ip", "-j", "address", "show", "dev", interface], timeout=5)
            for info in json.loads(addr.stdout or "[]"):
                for item in info.get("addr_info", []):
                    if item.get("family") != "inet":
                        continue
                    ip = ipaddress.ip_address(item["local"])
                    if not ip.is_private:
                        continue
                    network = ipaddress.ip_network(f"{ip}/{item['prefixlen']}", strict=False)
                    return {"interface": interface, "address": str(ip), "cidr": str(network)}
        except Exception:
            pass
        return {}

    def preflight(self) -> dict:
        from . import _cuda

        os_release = self._os_release()
        machine = _platform.machine()
        checks: dict[str, dict] = {
            "os": {
                "ok": os_release.get("ID") == "ubuntu" and os_release.get("VERSION_ID") == "24.04",
                "value": f"{os_release.get('ID', _platform.system())} "
                         f"{os_release.get('VERSION_ID', _platform.release())}".strip(),
                "required": "Ubuntu 24.04",
            },
            "architecture": {"ok": machine in {"x86_64", "amd64"}, "value": machine,
                             "required": "x86_64"},
            "systemd": {"ok": Path("/run/systemd/system").exists(), "value": "systemd"},
        }

        gpus, cuda_version, error = _cuda.probe(self)
        checks["nvidia_driver"] = {"ok": bool(gpus), "gpu_count": len(gpus), "error": error}
        toolkit = _cuda.choose_toolkit(cuda_version)
        checks["cuda_compatibility"] = {
            "ok": bool(toolkit),
            "driver_cuda": cuda_version,
            "selected_toolkit": toolkit,
            "error": "" if toolkit else
                     "Driver does not report compatibility with a supported CUDA toolkit",
        }

        network = self.detect_private_network()
        checks["private_network"] = {"ok": bool(network), **network}

        ufw_active = False
        if shutil.which("ufw"):
            try:
                ufw_active = "Status: active" in self.run_cmd(["ufw", "status"], timeout=5).stdout
            except Exception:
                pass
        checks["firewall"] = {"ok": ufw_active, "active": ufw_active,
                              "warning": "" if ufw_active else "No active UFW firewall detected"}

        return {
            "checks": checks,
            "required": ["os", "architecture", "systemd", "nvidia_driver",
                         "cuda_compatibility", "private_network"],
            "gpus": gpus,
            "network": network,
            "extra": {"cuda_toolkit": toolkit},
        }

    def firewall_rules(self, ports: list[int], cidr: str) -> list[list[str]]:
        network = ipaddress.ip_network(cidr, strict=False)
        if not network.is_private:
            raise ValueError("Firewall source must be a private network")
        return [["ufw", "allow", "from", str(network), "to", "any",
                 "port", str(port), "proto", "tcp"] for port in ports]

    def accelerator_cmake_args(self) -> list[str]:
        import shutil

        args = ["-DGGML_CUDA=ON"]
        candidates = sorted(Path("/usr/local").glob("cuda-*/bin/nvcc"),
                            key=_cuda_path_version, reverse=True)
        nvcc = str(candidates[0]) if candidates else (shutil.which("nvcc") or "")
        if not nvcc:
            raise RuntimeError(
                "CUDA toolkit compiler nvcc was not found; "
                "run the setup system-dependencies stage")
        args.append(f"-DCMAKE_CUDA_COMPILER={nvcc}")

        probe = self.run_cmd(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], timeout=10)
        architectures = sorted({
            line.strip().replace(".", "") for line in (probe.stdout or "").splitlines()
            if re.fullmatch(r"\s*\d+\.\d+\s*", line)})
        if not architectures:
            raise RuntimeError(
                "Could not detect NVIDIA GPU compute capability with nvidia-smi")
        args.append(f"-DCMAKE_CUDA_ARCHITECTURES={';'.join(architectures)}")
        return args

    #: Phrases llama.cpp prints when it has no usable GPU backend.
    CPU_ONLY_MARKERS = (
        "compiled without support for gpu offload",
        "no usable gpu found",
        "ggml_cuda: not found",
    )

    #: A CUDA context costs roughly this much per device before any model
    #: is loaded. The figure predates this package and is kept unchanged.
    @property
    def device_context_mib(self) -> int:
        return 400

    def verify_accelerated_build(self, probe_output: str) -> str:
        lowered = (probe_output or "").lower()
        if "cuda" not in lowered:
            return "the build reports no CUDA device"
        for marker in self.CPU_ONLY_MARKERS:
            if marker in lowered:
                return f"the build reports: {marker}"
        return ""



def _cuda_path_version(path: Path) -> tuple[int, ...]:
    """Sort key for /usr/local/cuda-*/bin/nvcc, newest first.

    Variable length on purpose: an install can be `cuda-13`, `cuda-13.3` or
    `cuda-13.3.1`, and a pattern demanding exactly two components silently
    sorts the single-component ones last -- picking the oldest toolkit on a host
    that has a `cuda-13` directory.
    """
    match = re.search(r"cuda-([0-9]+(?:\.[0-9]+)*)", str(path))
    return tuple(int(part) for part in match.group(1).split(".")) if match else (0,)
