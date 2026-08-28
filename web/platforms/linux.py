#!/usr/bin/env python3
"""Linux: systemd, nvidia-smi, /proc.

The implementations here are the ones this project has always run in
production, moved rather than rewritten. Comments explaining *why* a particular
call is shaped the way it is have been kept with the code they explain -- they
document failures that were paid for once already.
"""

from __future__ import annotations

import re
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
