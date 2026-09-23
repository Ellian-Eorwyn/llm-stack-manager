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

import os
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

    # launchd runs a wrapper script per job; the model server is its child,
    # and the router's per-model servers are grandchildren.
    LAUNCHCTL_LIST = ("PID\tStatus\tLabel\n"
                      "100\t0\tcom.llmstack.llm-a\n"
                      "200\t0\tcom.llmstack.llm-manager\n"
                      "300\t0\tcom.llmstack.llama-router\n"
                      "400\t0\tcom.apple.something\n")
    UID = 501
    PS = ("  100     1  501 /bin/bash\n"
          "  101   100  501 /stack/deps/llama.cpp/build/bin/llama-server\n"
          "  200     1  501 /stack/.venv/bin/python3\n"
          "  300     1  501 /bin/bash\n"
          "  301   300  501 /stack/deps/llama.cpp/build/bin/llama-server\n"
          "  302   301  501 /stack/deps/llama.cpp/build/bin/llama-server\n"
          "  500     1  501 /Applications/Safari.app/Contents/MacOS/Safari\n"
          "  600     1  501 /opt/homebrew/bin/llama-server\n"
          "  700     1    0 /usr/libexec/llama-server\n")
    FOOTPRINTS = {100: 3, 101: 36954, 200: 900, 301: 40, 302: 5120,
                  500: 2048, 600: 1024, 700: 4096}

    def _darwin_run(self, cmd, timeout=30):
        if cmd[:2] == ["launchctl", "list"]:
            return _completed(self.LAUNCHCTL_LIST)
        if cmd[0] == "ps":
            return _completed(self.PS)
        return _completed("", 1)

    def _darwin_apps(self, footprint=None):
        footprint = footprint or (lambda pid: self.FOOTPRINTS.get(pid, 1))
        with (
            platform_harness.as_darwin(run_cmd=self._darwin_run) as platform,
            patch.object(DarwinPlatform, "process_memory_mib", staticmethod(footprint)),
            patch("platforms.darwin.os.getuid", return_value=self.UID),
        ):
            return platform.gpu_compute_apps()

    def test_darwin_attributes_unified_memory_to_model_servers(self):
        # There is no nvidia-smi here, and it used to return None, so the header
        # said "No compute processes" with a 27B model loaded. The kernel's
        # per-process accounting, which counts wired Metal buffers, is the answer.
        apps = {row["pid"]: row for row in self._darwin_apps()}
        self.assertEqual(apps[101]["used_memory"], 36954)
        self.assertEqual(apps[101]["process_name"], "llama-server")
        self.assertEqual(apps[101]["gpu_uuid"], "apple-gpu-0")
        # A router grandchild is found through the process tree.
        self.assertIn(302, apps)
        # A model server started by hand still counts.
        self.assertIn(600, apps)

    def test_darwin_leaves_out_what_is_not_serving_a_model(self):
        apps = {row["pid"] for row in self._darwin_apps()}
        self.assertNotIn(200, apps)   # the manager itself
        self.assertNotIn(500, apps)   # an unrelated app
        self.assertNotIn(100, apps)   # a wrapper shell, below the floor
        self.assertNotIn(301, apps)   # a router child holding nothing
        self.assertNotIn(700, apps)   # another user's; unreadable anyway

    def test_darwin_says_so_when_footprints_cannot_be_read(self):
        # None and [] mean different things here. [] asserts nothing is loaded;
        # None says do not draw conclusions from the absence of rows.
        self.assertIsNone(self._darwin_apps(footprint=lambda pid: None))

    def _memory_mib(self, footprint_mib, resident_mib, mapped_mib):
        import platforms.darwin as darwin

        class FakeLibproc:
            @staticmethod
            def proc_pid_rusage(pid, flavor, ref):
                ref._obj.ri_phys_footprint = footprint_mib * 1024 * 1024
                ref._obj.ri_resident_size = resident_mib * 1024 * 1024
                return 0

        with (
            patch.object(darwin, "_libproc", FakeLibproc()),
            patch.object(darwin, "_mapped_weights_mib", return_value=mapped_mib),
        ):
            return darwin._process_memory_mib(4242)

    def test_an_mmap_loaded_model_counts_its_weights(self):
        # TASK_NO_MMAP=false: the weights are clean pages of the GGUF, which
        # phys_footprint leaves out. The task model read 5.4 GiB this way while
        # holding 12.4 GiB -- less than its own 6.9 GiB weights file.
        self.assertEqual(self._memory_mib(5556, 5500, mapped_mib=7102), 12658)

    def test_resident_size_is_not_used_for_wired_gpu_memory(self):
        # Once Metal wires a process's buffers, resident_size stops reflecting
        # them: llm-a read 15 GiB on it while its footprint was 37.
        self.assertEqual(self._memory_mib(37593, 15600, mapped_mib=0), 37593)

    @unittest.skipUnless(sys.platform == "darwin", "reads this process's own regions")
    def test_mapped_weights_are_read_from_the_process_regions(self):
        # A real mapping of a real file, read back through proc_pidinfo.
        import mmap
        import tempfile
        import platforms.darwin as darwin
        with tempfile.NamedTemporaryFile(suffix=".gguf") as handle:
            handle.write(b"\1" * (8 * 1024 * 1024))
            handle.flush()
            with mmap.mmap(handle.fileno(), 0, prot=mmap.PROT_READ) as mapped:
                sum(mapped[i] for i in range(0, len(mapped), 4096))   # fault it in
                self.assertGreaterEqual(darwin._mapped_weights_mib(os.getpid()), 7)
        self.assertEqual(darwin._mapped_weights_mib(os.getpid()), 0)

    def test_darwin_maps_a_child_process_to_its_launchd_job(self):
        with platform_harness.as_darwin(run_cmd=self._darwin_run) as platform:
            self.assertEqual(platform.pid_unit(101), "llm-a")
            self.assertEqual(platform.pid_unit(302), "llama-router")
            self.assertEqual(platform.pid_unit(100), "llm-a")
            self.assertEqual(platform.pid_unit(500), "")

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


class PreflightContractTests(unittest.TestCase):
    """The wizard has to be able to run on both platforms, and its planner has
    to be able to read what preflight hands it on both platforms."""

    SECTIONS = {"checks", "required", "gpus", "network", "extra"}

    def _preflight(self, harness):
        with harness() as platform:
            return platform.preflight()

    def test_both_platforms_return_the_same_sections(self):
        for harness in (platform_harness.as_linux, platform_harness.as_darwin):
            with self.subTest(harness.__name__):
                self.assertEqual(set(self._preflight(harness)), self.SECTIONS)

    def test_every_required_check_actually_exists(self):
        # A name in `required` with no matching check is a KeyError in
        # collect_preflight, on the platform nobody is running the wizard on.
        for harness in (platform_harness.as_linux, platform_harness.as_darwin):
            with self.subTest(harness.__name__):
                result = self._preflight(harness)
                missing = set(result["required"]) - set(result["checks"])
                self.assertEqual(missing, set())

    def test_every_check_carries_an_ok_flag(self):
        for harness in (platform_harness.as_linux, platform_harness.as_darwin):
            for name, check in self._preflight(harness)["checks"].items():
                with self.subTest(harness.__name__, check=name):
                    self.assertIn("ok", check)
                    self.assertIs(type(check["ok"]), bool)

    def test_preflight_gpus_are_in_the_shape_the_planner_reads(self):
        # `gpu_info()` names these mem_total/mem_free; the planner reads
        # memory_total_mib/memory_free_mib and raises KeyError on the first
        # model it places if handed the wrong one.
        for harness in (platform_harness.as_linux, platform_harness.as_darwin):
            for gpu in self._preflight(harness)["gpus"]:
                with self.subTest(harness.__name__):
                    self.assertLessEqual(
                        {"index", "name", "memory_total_mib", "memory_free_mib"}, set(gpu))

    def test_the_darwin_firewall_installs_no_rules_rather_than_wrong_ones(self):
        # macOS filters by application, not by port and source subnet, so there
        # is nothing here that corresponds to the UFW rules. Empty is the honest
        # answer; the preflight check is what tells the operator.
        with platform_harness.as_darwin() as platform:
            self.assertEqual(platform.firewall_rules([8077], "192.168.4.0/22"), [])

    def test_the_linux_firewall_refuses_a_public_source(self):
        with platform_harness.as_linux() as platform:
            with self.assertRaises(ValueError):
                platform.firewall_rules([8077], "8.8.8.0/24")
            rules = platform.firewall_rules([8077], "192.168.4.0/22")
            self.assertEqual(rules[0][:4], ["ufw", "allow", "from", "192.168.4.0/22"])


class DarwinNetworkTests(unittest.TestCase):

    IFCONFIG = (
        "en4: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500\n"
        "\tinet 192.168.4.32 netmask 0xfffffc00 broadcast 192.168.7.255\n"
    )
    ROUTE = "   route to: default\n    gateway: 192.168.4.1\n  interface: en4\n"

    def _run(self, cmd, timeout=30):
        if cmd[0] == "route":
            return _completed(self.ROUTE)
        if cmd[0] == "ifconfig":
            return _completed(self.IFCONFIG)
        return _completed("")

    def test_the_hexadecimal_netmask_becomes_a_prefix_length(self):
        # BSD ifconfig prints the mask as 0xfffffc00 and has no flag to change
        # it, so it has to be counted into a prefix rather than parsed as one.
        with platform_harness.as_darwin(run_cmd=self._run) as platform:
            self.assertEqual(platform.detect_private_network(), {
                "interface": "en4", "address": "192.168.4.32", "cidr": "192.168.4.0/22"})

    def test_a_public_address_is_not_offered_as_a_private_network(self):
        # Not 203.0.113.x: Python counts the documentation ranges as private,
        # so TEST-NET-3 passes `is_private` and would not exercise this at all.
        public = self.IFCONFIG.replace("192.168.4.32", "8.8.4.4")
        with platform_harness.as_darwin(
                run_cmd=lambda cmd, timeout=30: _completed(
                    self.ROUTE if cmd[0] == "route" else public)) as platform:
            self.assertEqual(platform.detect_private_network(), {})


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



class LogSourceTests(unittest.TestCase):
    """The Logs tab and request telemetry read `journalctl`, which a Mac does
    not have -- so both were blank there while launchd wrote every line to the
    files each job's plist names."""

    def test_linux_reads_the_journal(self):
        cmd = LinuxPlatform().log_command("llm-a", 100, follow=True)
        self.assertEqual(cmd[:3], ["journalctl", "-u", "llm-a"])
        self.assertIn("-f", cmd)
        self.assertIn("--output=short-iso", cmd)
        self.assertIn("--output=short-iso-precise",
                      LinuxPlatform().log_command("llm-a", 0, precise=True))
        self.assertTrue(LinuxPlatform.journal_logs)

    def test_darwin_tails_the_files_its_plist_names(self):
        import plistlib
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            err, out = pathlib.Path(tmp, "llm-a.stderr.log"), pathlib.Path(tmp, "llm-a.stdout.log")
            err.write_text("x\n")
            out.write_text("y\n")
            plist = pathlib.Path(tmp, "com.llmstack.llm-a.plist")
            plist.write_bytes(plistlib.dumps({"StandardErrorPath": str(err),
                                              "StandardOutPath": str(out)}))
            platform = DarwinPlatform()
            with patch.object(DarwinPlatform, "_domain_and_plist",
                              lambda self, name: ("gui/501", str(plist))):
                cmd = platform.log_command("llm-a", 100, follow=True)
                self.assertEqual(cmd, ["tail", "-n", "100", "-F", str(err), str(out)])
                self.assertEqual(platform.log_command("llm-a", 5),
                                 ["tail", "-n", "5", str(err), str(out)])
        self.assertFalse(DarwinPlatform.journal_logs)

    def test_darwin_says_why_when_there_is_nothing_to_read(self):
        with patch.object(DarwinPlatform, "_domain_and_plist",
                          lambda self, name: ("gui/501", "/nonexistent.plist")):
            cmd = DarwinPlatform().log_command("embed", 100, follow=True)
        self.assertEqual(cmd[0], "echo")
        self.assertIn("embed", cmd[1])


class DarwinStartInstallsMissingAgentTests(unittest.TestCase):
    def test_a_service_with_no_plist_is_installed_then_started(self):
        # Bootstrapping a plist that does not exist fails with launchctl's
        # "Input/output error"; a component left out at setup had no other way
        # to be started from the manager.
        calls, installed = [], []

        def run(cmd, timeout=30):
            calls.append(cmd)
            if cmd[0] == "bash":
                installed.append(True)
            return _completed("")

        exists = lambda path: bool(installed) or not path.endswith("com.llmstack.task.plist")
        with (
            platform_harness.as_darwin(run_cmd=run) as platform,
            patch("platforms.darwin.os.path.exists", side_effect=exists),
        ):
            result = platform.service_start("task")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(calls[0][0], "bash")
        self.assertTrue(calls[0][1].endswith("scripts/install-launchd-service.sh"))
        self.assertEqual(calls[0][2], "task")
        self.assertEqual(calls[-1][:2], ["launchctl", "bootstrap"])

    def test_an_install_that_fails_is_reported_not_bootstrapped(self):
        calls = []

        def run(cmd, timeout=30):
            calls.append(cmd)
            return _completed("", 2)

        with (
            platform_harness.as_darwin(run_cmd=run) as platform,
            patch("platforms.darwin.os.path.exists", return_value=False),
        ):
            result = platform.service_start("glmocr-sdk")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(len(calls), 1)

if __name__ == "__main__":
    unittest.main()


class EngineSelectionTests(unittest.TestCase):
    """A slot served by a different engine is judged by that engine's rules.

    Both of these were live defects when the MLX servers were first wired in:
    the manager reported them degraded while they served correctly, because it
    was asking a llama-server question of something that is not llama-server.
    """

    def setUp(self):
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "web"))
        import health
        self.health = health

    def test_the_default_embedding_probe_is_still_llama_servers(self):
        spec = self.health.probe_spec("embed", {})
        self.assertEqual(spec["path"], "/props")

    def test_the_mlx_embedding_server_is_not_asked_for_props(self):
        # It has no /props and answers 404. Faking one would be worse than this
        # table: telemetry parses that payload for slot and context accounting,
        # and an imitation would produce numbers about a server with no slots.
        spec = self.health.probe_spec("embed", {"EMBED_ENGINE": "mlx"})
        self.assertEqual(spec["path"], "/health")
        self.assertEqual(spec["expect_field"], ("status", "healthy"))

    def test_the_parakeet_server_reports_healthy_not_ok(self):
        # The sidecar returns {"status": "ok"}; Parakeet returns
        # {"status": "healthy"}. Probing for the wrong one fails a working
        # service.
        sidecar = self.health.probe_spec("transcript-backend", {})
        parakeet = self.health.probe_spec(
            "transcript-backend", {"TRANSCRIPT_ENGINE": "parakeet-mlx"})
        self.assertEqual(sidecar["expect_field"], ("status", "ok"))
        self.assertEqual(parakeet["expect_field"], ("status", "healthy"))

    def test_an_unknown_engine_falls_back_rather_than_losing_the_probe(self):
        # A typo in the config should not silently remove a service's readiness
        # check; it should behave as the default engine does.
        spec = self.health.probe_spec("embed", {"EMBED_ENGINE": "typo"})
        self.assertEqual(spec, self.health.SERVICE_PROBES["embed"])

    def test_every_engine_probe_names_a_service_that_exists(self):
        for (service, engine), spec in self.health.ENGINE_PROBES.items():
            with self.subTest(service=service, engine=engine):
                self.assertIn(service, self.health.SERVICE_PROBES)
                self.assertIn(service, self.health.SERVICE_ENGINE_KEYS)
                for key in ("kind", "path", "host_key", "port_key"):
                    self.assertIn(key, spec)


class AcceleratedBuildTests(unittest.TestCase):
    """A CPU-only llama.cpp build is not a broken build.

    It compiles, starts, serves and answers correctly -- it is simply an order
    of magnitude slower and will exhaust host RAM on a model the GPU would have
    held. So the check is positive (does it name the accelerator?) rather than
    a search for error strings.
    """

    # Verbatim from a Metal build of the pinned revision on an M1 Pro. The
    # device is MTL0, not Metal0: a check for "metal" matches nothing here and
    # refuses a correct binary, which is exactly what happened the first time.
    METAL = ("Available devices:\n"
             "  MTL0: Apple M1 Pro (12124 MiB, 12123 MiB free)\n"
             "  BLAS: Accelerate (0 MiB, 0 MiB free)")
    CUDA = "Available devices:\n  CUDA0: NVIDIA GeForce RTX 3090 (24576 MiB, 6507 MiB free)"
    CPU_ONLY = "warning: llama.cpp was compiled without support for GPU offload"

    def test_each_platform_accepts_only_its_own_accelerator(self):
        cases = {
            ("linux", self.CUDA): "",
            ("darwin", self.METAL): "",
        }
        for (name, probe), expected in cases.items():
            harness = platform_harness.as_linux if name == "linux" else platform_harness.as_darwin
            with harness() as platform, self.subTest(name):
                self.assertEqual(platform.verify_accelerated_build(probe), expected)

    def test_a_metal_build_is_refused_on_linux_and_vice_versa(self):
        with platform_harness.as_linux() as platform:
            self.assertTrue(platform.verify_accelerated_build(self.METAL))
        with platform_harness.as_darwin() as platform:
            self.assertTrue(platform.verify_accelerated_build(self.CUDA))

    def test_a_cpu_only_build_is_refused_on_both(self):
        for harness in (platform_harness.as_linux, platform_harness.as_darwin):
            with harness() as platform, self.subTest(platform.name):
                self.assertTrue(platform.verify_accelerated_build(self.CPU_ONLY))

    def test_metal_needs_no_toolkit_or_architecture_probe(self):
        # CUDA needs nvcc located and compute capabilities detected. Metal ships
        # with the OS: one flag is the whole configuration.
        #
        # `ioreg` is stubbed because the check is still a positive one -- it
        # refuses to configure a build with no Metal device -- and a Linux
        # runner has no IOAccelerator to find.
        ioreg = ('<plist version="1.0"><array><dict>'
                 '<key>IOClass</key><string>AGXAcceleratorG13X</string>'
                 '<key>PerformanceStatistics</key><dict>'
                 '<key>Alloc system memory</key><integer>19067781120</integer>'
                 '</dict></dict></array></plist>')

        def run(cmd, timeout=30):
            if cmd[0] == "ioreg":
                return _completed(ioreg)
            return UnifiedMemoryTests()._run(cmd, timeout)

        with platform_harness.as_darwin(run_cmd=run) as platform:
            self.assertEqual(platform.accelerator_cmake_args(), ["-DGGML_METAL=ON"])

    def test_a_host_with_no_metal_device_is_refused_rather_than_built_cpu_only(self):
        with platform_harness.as_darwin(run_cmd=lambda *a, **k: _completed("", 1)) as platform:
            with self.assertRaises(RuntimeError):
                platform.accelerator_cmake_args()


class UnifiedMemoryBudgetTests(unittest.TestCase):
    """Overcommitting unified memory is a different failure from overcommitting
    VRAM, and needs a different verdict."""

    def setUp(self):
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "web"))
        import budget
        self.budget = budget

    PREDICTION = {
        "vram": {"upper_mib": 12000, "per_device": [{"upper_mib": 12000}]},
        "host": {"checkpoint_total_mib": 6000},
    }

    def test_weights_and_prompt_cache_are_charged_to_the_same_pool(self):
        # On a discrete GPU the prompt cache is host RAM and a separate budget.
        # Here it is the same memory the model is in, so omitting it would
        # under-count by the whole cache: 12,000 fits 16,384 and 18,000 does not.
        with platform_harness.as_darwin():
            issues = self.budget._unified_memory_issues(
                self.PREDICTION, [{"mem_total": 16384, "mem_free": 8000}])
        self.assertEqual([i["code"] for i in issues], ["memory_overcommit_swaps"])
        self.assertIn("page", issues[0]["text"])

    def test_the_verdict_says_it_will_not_fail_to_start(self):
        # The operator response differs from vram_overcommit: there is no crash
        # loop to find, just a machine that got slow.
        with platform_harness.as_darwin():
            issues = self.budget._unified_memory_issues(
                self.PREDICTION, [{"mem_total": 16384}])
        self.assertNotIn("vram_overcommit", [i["code"] for i in issues])
        self.assertIn("will not fail to start", issues[0]["text"])

    def test_a_configuration_that_fits_reports_nothing(self):
        small = {"vram": {"upper_mib": 2000, "per_device": []},
                 "host": {"checkpoint_total_mib": 500}}
        with platform_harness.as_darwin():
            self.assertEqual(self.budget._unified_memory_issues(
                small, [{"mem_total": 16384, "mem_free": 12000}]), [])

    def test_fitting_installed_ram_but_not_free_ram_is_only_a_warning(self):
        small = {"vram": {"upper_mib": 6000, "per_device": []},
                 "host": {"checkpoint_total_mib": 1000}}
        with platform_harness.as_darwin():
            issues = self.budget._unified_memory_issues(
                small, [{"mem_total": 16384, "mem_free": 4000}])
        self.assertEqual([i["code"] for i in issues], ["memory_tight"])
        self.assertEqual(issues[0]["level"], "warn")

    def test_an_explicit_wired_limit_lowers_the_ceiling(self):
        # llama.cpp reports 12,124 MiB usable of 16,384 installed on an M1 Pro:
        # macOS caps what the GPU may wire. Where the cap has been set
        # explicitly it is the real ceiling and installed RAM is not.
        prediction = {"vram": {"upper_mib": 9000, "per_device": []},
                      "host": {"checkpoint_total_mib": 1000}}
        with platform_harness.as_darwin():
            without = self.budget._unified_memory_issues(
                prediction, [{"mem_total": 16384, "mem_free": 16000}])
            with_cap = self.budget._unified_memory_issues(
                prediction, [{"mem_total": 16384, "mem_free": 16000,
                              "wired_limit_mib": 8192}])
        self.assertEqual(without, [])
        self.assertEqual([i["code"] for i in with_cap], ["memory_overcommit_swaps"])

    def test_the_per_device_context_overhead_is_not_a_cuda_constant(self):
        # 400 MiB is CUDA's; Metal's is materially smaller, and charging CUDA's
        # figure per device inflated every Mac prediction.
        with platform_harness.as_linux() as linux:
            self.assertEqual(linux.device_context_mib, 400)
        with platform_harness.as_darwin() as darwin:
            self.assertLess(darwin.device_context_mib, 400)


class InertConfigTests(unittest.TestCase):
    """What a platform cannot configure, and why.

    A control that reports success and changes nothing is the worst of the
    three states a setting can be in: worse than missing, because the operator
    sets it and believes it took, and worse than wrong, because nothing says
    so. On a Metal build `CUDA_VISIBLE_DEVICES` is not read at all and
    `resolve_split_opts` collapses every split mode before it reaches
    llama-server, so the whole GPU-placement surface is in that state.
    """

    def test_linux_withholds_nothing(self):
        with platform_harness.as_linux() as platform:
            self.assertEqual(platform.inert_config_capabilities, {})

    def test_darwin_names_the_three_it_cannot_act_on(self):
        with platform_harness.as_darwin() as platform:
            self.assertEqual(set(platform.inert_config_capabilities),
                             {"gpu_visible_devices", "gpu_indices", "split_modes"})

    def test_every_reason_is_a_sentence_an_operator_can_act_on(self):
        # These are rendered into the page verbatim, so "unsupported" would be
        # a worse answer than the field simply being absent.
        with platform_harness.as_darwin() as platform:
            for capability, reason in platform.inert_config_capabilities.items():
                with self.subTest(capability):
                    self.assertTrue(reason.endswith("."), reason)
                    self.assertGreater(len(reason.split()), 5, reason)

    def test_the_capability_names_are_ones_the_config_surface_asks_about(self):
        # A reason for a capability nothing consults is a reason nobody sees.
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "web"))
        import config_fields
        with platform_harness.as_darwin() as platform:
            self.assertEqual(set(platform.inert_config_capabilities),
                             set(config_fields.CAPABILITY_BY_SUFFIX.values()))
