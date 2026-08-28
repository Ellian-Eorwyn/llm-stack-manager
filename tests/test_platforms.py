"""The platform adapters, and the promises they make to each other.

The macOS support that existed before this package did not fail loudly. It
returned `{}` for host memory, `[]` for GPUs, `0` for restart counts and `False`
for every service's activity, and no test noticed, because every test that
touched those paths ran on Linux where they worked.

So these tests are mostly about *shape*: both adapters answering the same
question with the same keys, and the specific fabrications that were previously
mistaken for answers being caught if they come back.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import platform_harness  # noqa: E402
from platform_harness import platforms  # noqa: E402
from platform_harness import DarwinPlatform, LinuxPlatform  # noqa: E402


def _completed(stdout="", returncode=0):
    return subprocess.CompletedProcess([], returncode, stdout, "")


class InterfaceConformanceTests(unittest.TestCase):
    """Neither adapter may quietly omit a method or a key."""

    ADAPTERS = (LinuxPlatform, DarwinPlatform)

    def test_both_adapters_are_constructible(self):
        # An abstract method left unimplemented fails here rather than at the
        # call site on the one platform nobody is testing on.
        for adapter in self.ADAPTERS:
            with self.subTest(adapter.__name__):
                self.assertIsInstance(adapter(), platforms.Platform)

    def test_both_adapters_name_themselves_distinctly(self):
        names = {adapter().name for adapter in self.ADAPTERS}
        self.assertEqual(names, {"linux", "darwin"})

    SERVICE_STATE_KEYS = {
        "installed", "active", "failed", "starting", "active_state",
        "sub_state", "result", "main_pid", "n_restarts",
    }

    def test_service_state_has_the_same_keys_on_both_platforms(self):
        with platform_harness.as_linux(run_cmd=lambda *a, **k: _completed("")):
            linux_state = LinuxPlatform().service_state("embed")
        with platform_harness.as_darwin(run_cmd=lambda *a, **k: _completed("", 1)):
            darwin_state = DarwinPlatform().service_state("embed")

        self.assertEqual(set(linux_state), self.SERVICE_STATE_KEYS)
        self.assertEqual(set(darwin_state), self.SERVICE_STATE_KEYS)

    def test_service_state_types_agree_across_platforms(self):
        with platform_harness.as_linux(run_cmd=lambda *a, **k: _completed("")):
            linux_state = LinuxPlatform().service_state("embed")
        with platform_harness.as_darwin(run_cmd=lambda *a, **k: _completed("", 1)):
            darwin_state = DarwinPlatform().service_state("embed")
        for key in self.SERVICE_STATE_KEYS:
            with self.subTest(key):
                self.assertIs(type(linux_state[key]), type(darwin_state[key]))


class DarwinLaunchctlTests(unittest.TestCase):
    """`launchctl list` emits an OpenStep plist, not JSON."""

    # Trimmed from real output. The nested dict and the unquoted
    # `mach-port-object` are what a naive parser trips over.
    LIST_OUTPUT = """{
\t"LimitLoadToSessionType" = "Aqua";
\t"MachServices" = {
\t\t"com.example.thing" = mach-port-object;
\t};
\t"Label" = "com.llmstack.embed";
\t"LastExitStatus" = 0;
\t"PID" = 4242;
};
"""

    def test_the_pid_is_read_from_the_openstep_plist(self):
        # This is the regression that mattered most: the previous code called
        # json.loads on this, which raised every time and was caught, so the PID
        # was always 0 and every service on macOS read as inactive forever.
        with platform_harness.as_darwin(run_cmd=lambda *a, **k: _completed(self.LIST_OUTPUT)):
            state = DarwinPlatform().service_state("embed")
        self.assertEqual(state["main_pid"], 4242)
        self.assertTrue(state["active"])

    def test_a_job_that_is_not_loaded_is_not_active(self):
        with platform_harness.as_darwin(run_cmd=lambda *a, **k: _completed("", 1)):
            state = DarwinPlatform().service_state("embed")
        self.assertFalse(state["active"])
        self.assertEqual(state["main_pid"], 0)

    def test_a_loaded_job_with_no_pid_and_a_bad_exit_is_failed(self):
        output = '{\n\t"Label" = "com.llmstack.ocr";\n\t"LastExitStatus" = 134;\n};\n'
        with platform_harness.as_darwin(run_cmd=lambda *a, **k: _completed(output)):
            state = DarwinPlatform().service_state("ocr")
        self.assertTrue(state["failed"])
        self.assertEqual(state["result"], "exit-code-134")


class DarwinRestartCountTests(unittest.TestCase):
    """launchd has no NRestarts, so it is synthesised from PID changes.

    Returning a constant 0 here -- what the code this replaces did -- does not
    read as "unknown". It reads as "healthy", and it silently disabled the flap
    detection that docs/service-health.md calls the most valuable signal in the
    health model.
    """

    @staticmethod
    def _listing(pid):
        body = f'\t"PID" = {pid};\n' if pid else ""
        return '{\n\t"Label" = "com.llmstack.ocr";\n' + body + '\t"LastExitStatus" = 0;\n};\n'

    def _poll(self, platform, pid):
        with patch.object(DarwinPlatform, "run_cmd",
                          staticmethod(lambda *a, **k: _completed(self._listing(pid)))):
            return platform.service_state("ocr")

    def test_a_stable_pid_is_not_a_restart(self):
        platform = DarwinPlatform()
        with platform_harness.as_platform(platform):
            for _ in range(3):
                state = self._poll(platform, 100)
        self.assertEqual(state["n_restarts"], 0)

    def test_a_pid_that_changes_underneath_us_counts_as_a_restart(self):
        platform = DarwinPlatform()
        with platform_harness.as_platform(platform):
            self._poll(platform, 100)
            self._poll(platform, 0)
            state = self._poll(platform, 101)
        self.assertEqual(state["n_restarts"], 1)

    def test_a_service_bouncing_repeatedly_keeps_climbing(self):
        # The count alone proves nothing; a count that climbs between polls is
        # a service that cannot stay up, and that is what must be visible.
        platform = DarwinPlatform()
        with platform_harness.as_platform(platform):
            for pid in (100, 0, 101, 0, 102, 0, 103):
                state = self._poll(platform, pid)
        self.assertEqual(state["n_restarts"], 3)

    def test_an_operator_restart_is_not_counted_as_flapping(self):
        # systemd's NRestarts counts automatic restarts. Someone pressing
        # "restart" in the UI must not make a healthy service look like one that
        # cannot come up.
        platform = DarwinPlatform()
        with platform_harness.as_platform(platform):
            self._poll(platform, 100)
            with patch.object(DarwinPlatform, "run_cmd",
                              staticmethod(lambda *a, **k: _completed(""))):
                platform.service_restart("ocr")
            state = self._poll(platform, 101)
        self.assertEqual(state["n_restarts"], 0)


class UnifiedMemoryTests(unittest.TestCase):

    VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                     4258.
Pages active:                                 293573.
Pages inactive:                               292709.
Pages speculative:                              1297.
Pages wired down:                             226078.
Pages purgeable:                                5230.
Pages occupied by compressor:                 190994.
Pageins:                                   375827494.
Pageouts:                                    7090774.
Swapins:                                    74905437.
Swapouts:                                  112407584.
"""

    def _run(self, cmd, timeout=30):
        if cmd[0] == "vm_stat":
            return _completed(self.VM_STAT)
        if cmd[:2] == ["sysctl", "-n"] and cmd[2] == "hw.memsize":
            return _completed(str(16 * 1024**3))
        if cmd[:2] == ["sysctl", "-n"] and cmd[2] == "vm.swapusage":
            return _completed("total = 22528.00M  used = 20865.44M  free = 1662.56M  (encrypted)")
        return _completed("")

    def test_the_page_size_comes_from_the_platform(self):
        # This was the constant `4 / 1024`. Apple silicon uses 16 KiB pages, so
        # every swap rate reported on a Mac was a quarter of the real figure --
        # and that is the number feeding the host_swapping alert.
        with platform_harness.as_darwin(run_cmd=self._run) as platform:
            type(platform)._page_bytes_cache = None
            self.assertEqual(platform.page_bytes, 16384)
        self.assertEqual(LinuxPlatform().page_bytes, 4096)

    def test_meminfo_uses_the_proc_meminfo_key_names_and_units(self):
        # Keeping Linux's vocabulary on both platforms is what lets every
        # consumer and every existing test stay unchanged.
        with platform_harness.as_darwin(run_cmd=self._run) as platform:
            type(platform)._page_bytes_cache = None
            info = platform.meminfo()
        self.assertEqual(info["MemTotal"], 16 * 1024 * 1024)
        self.assertEqual(info["SwapTotal"], round(22528.00 * 1024))
        self.assertEqual(info["SwapFree"], round(1662.56 * 1024))
        # free + inactive + speculative + purgeable, in KiB at 16 KiB pages.
        self.assertEqual(info["MemAvailable"], (4258 + 292709 + 1297 + 5230) * 16)

    def test_meminfo_is_never_silently_empty_on_a_working_host(self):
        with platform_harness.as_darwin(run_cmd=self._run) as platform:
            info = platform.meminfo()
        for key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
            self.assertIn(key, info)
        self.assertGreater(info["MemTotal"], 0)

    def test_swap_rate_comes_from_swapins_not_pageins(self):
        # Pageins counts every demand-paged file read -- launching an
        # application moves it by hundreds of thousands -- so using it would
        # report any busy machine as permanently swapping.
        with platform_harness.as_darwin(run_cmd=self._run) as platform:
            self.assertEqual(platform.swap_counters(), (74905437, 112407584))


class GpuAttributionTests(unittest.TestCase):

    def test_darwin_cannot_attribute_device_memory_and_says_so(self):
        # None and [] mean different things here. [] asserts the GPU is idle;
        # None says do not draw conclusions from the absence of rows.
        with platform_harness.as_darwin() as platform:
            self.assertIsNone(platform.gpu_compute_apps())

    def test_linux_parses_nvidia_compute_apps(self):
        rows = "GPU-aaa, 4242, llama-server, 17104\nGPU-bbb, 4243, python, 428\n"
        with platform_harness.as_linux(run_cmd=lambda *a, **k: _completed(rows)) as platform:
            apps = platform.gpu_compute_apps()
        self.assertEqual(len(apps), 2)
        self.assertEqual(apps[0], {"gpu_uuid": "GPU-aaa", "pid": 4242,
                                   "process_name": "llama-server", "used_memory": 17104})

    def test_unavailable_gpu_readings_are_none_rather_than_zero(self):
        # A fabricated 0 for temperature or power renders as a cold, idle card,
        # which is indistinguishable from good news.
        ioreg = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
            '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">'
            '<plist version="1.0"><array><dict>'
            '<key>IOClass</key><string>AGXAcceleratorG13X</string>'
            '<key>PerformanceStatistics</key><dict>'
            '<key>Device Utilization %</key><integer>44</integer>'
            '<key>Alloc system memory</key><integer>19067781120</integer>'
            '</dict></dict></array></plist>'
        )

        def run(cmd, timeout=30):
            if cmd[0] == "ioreg":
                return _completed(ioreg)
            return UnifiedMemoryTests()._run(cmd, timeout)

        with platform_harness.as_darwin(run_cmd=run) as platform:
            gpus = platform.gpu_info()
        self.assertEqual(len(gpus), 1)
        gpu = gpus[0]
        for key in ("temp", "power_watts", "power_limit_watts", "fan_pct"):
            with self.subTest(key):
                self.assertIsNone(gpu[key])
        self.assertEqual(gpu["util"], 44)

    def test_unified_memory_is_reported_against_host_memory(self):
        # IOAccelerator's "Alloc system memory" counts virtual allocations: on a
        # 16 GiB machine it reads 18,102 MiB, which as a VRAM figure gives 110%
        # used, a negative free figure, and a gpu_vram_low alert that can never
        # clear. What constrains loading a model here is host memory pressure.
        ioreg = (
            '<plist version="1.0"><array><dict>'
            '<key>IOClass</key><string>AGXAcceleratorG13X</string>'
            '<key>PerformanceStatistics</key><dict>'
            '<key>Alloc system memory</key><integer>19067781120</integer>'
            '</dict></dict></array></plist>'
        )

        def run(cmd, timeout=30):
            if cmd[0] == "ioreg":
                return _completed(ioreg)
            return UnifiedMemoryTests()._run(cmd, timeout)

        with platform_harness.as_darwin(run_cmd=run) as platform:
            gpu = platform.gpu_info()[0]
        self.assertTrue(gpu["unified_memory"])
        self.assertLessEqual(gpu["mem_used"], gpu["mem_total"])
        self.assertGreaterEqual(gpu["mem_free"], 0)
        self.assertLessEqual(gpu["mem_pct"], 100)
        # Kept, but named for what it is rather than dressed up as VRAM.
        self.assertEqual(gpu["driver_alloc_mib"], round(19067781120 / 1024**2))


class ActiveAdapterTests(unittest.TestCase):

    def test_the_active_adapter_is_cached(self):
        # The Darwin adapter carries the synthesised restart counts, which are
        # only meaningful as a series of observations by one object. A new
        # adapter per call would reset them on every poll and reproduce exactly
        # the fixed-zero n_restarts this package exists to fix.
        platforms.set_active(None)
        try:
            self.assertIs(platforms.active(), platforms.active())
        finally:
            platforms.set_active(None)

    def test_detect_matches_the_host(self):
        expected = "darwin" if sys.platform == "darwin" else "linux"
        self.assertEqual(platforms.detect().name, expected)


if __name__ == "__main__":
    unittest.main()
