"""gpu-lease.py with systemctl and HTTP replaced by a small fake llms."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from unittest import mock


def _load():
    root = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("gpu_lease", root / "scripts" / "gpu-lease.py")
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


lease = _load()
NOON = datetime(2026, 9, 28, 12, 0, tzinfo=timezone(timedelta(hours=-7)))


class FakeLlms:
    """Just enough of llms: the backend unit, ComfyUI's device and queue."""

    def __init__(self, tmp: pathlib.Path):
        self.tmp = tmp
        self.backend_up = True
        self.backend_busy_polls = 0
        self.comfy_device = "1"  # the unit's default: ComfyUI's home card
        self.comfy_queue = 0
        self.history: list[float] = []  # epoch seconds of finished ComfyUI jobs
        self.watchdog_up = False
        self.commands: list[list[str]] = []
        self.fail: dict[str, str] = {}
        self.backend_latched = False  # NInfer's "unavailable": running, /health 503
        self.backend_since = NOON.timestamp() - 3600  # when the unit last (re)started
        self.restart_heals = True

    def run(self, cmd, check=True):
        self.commands.append(cmd)
        joined = " ".join(cmd)
        for needle, err in self.fail.items():
            if needle in joined:
                if check:
                    raise lease.LeaseError(err)
                return mock.Mock(returncode=1, stdout="", stderr=err)
        if cmd[:4] == ["sudo", "-n", "systemctl", "stop"]:
            self.backend_up = False
        elif cmd[:4] == ["sudo", "-n", "systemctl", "start"]:
            self.backend_up = True
        elif cmd[:4] == ["sudo", "-n", "systemctl", "restart"]:
            self.backend_up = True
            self.backend_latched = not self.restart_heals
        elif cmd[:2] == ["systemctl", "show"]:
            active = "active" if self.backend_up else "inactive"
            return mock.Mock(returncode=0, stderr="",
                             stdout=f"ActiveState={active}\nActiveEnterTimestamp=@{int(self.backend_since)}\n")
        elif cmd[:4] == ["systemctl", "--user", "show", "-p"]:
            return mock.Mock(returncode=0, stdout="/user.slice/app.slice/comfyui.service\n", stderr="")
        elif cmd[:2] == ["nvidia-smi", "--query-gpu=index,uuid"]:
            return mock.Mock(returncode=0, stdout="0, GPU-aaa\n1, GPU-bbb\n", stderr="")
        elif cmd[:2] == ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid"]:
            uuid = {"0": "GPU-aaa", "1": "GPU-bbb"}[self.comfy_device]
            return mock.Mock(returncode=0, stdout=f"4242, {uuid}\n999, GPU-bbb\n", stderr="")
        elif cmd[0] == "systemd-run":
            self.watchdog_up = True
        elif cmd[:3] == ["systemctl", "--user", "stop"] and "gpu1-lease-watchdog.service" in cmd:
            self.watchdog_up = False
        elif cmd[:3] == ["systemctl", "--user", "is-active"]:
            return mock.Mock(returncode=0 if self.watchdog_up else 3,
                             stdout="active\n" if self.watchdog_up else "inactive\n", stderr="")
        elif cmd[:3] == ["systemctl", "--user", "restart"]:
            dev_file = lease.COMFY_DEVICE_FILE
            self.comfy_device = dev_file.read_text().strip().split("=")[1] if dev_file.exists() else "1"
            self.comfy_queue = 0
        return mock.Mock(returncode=0, stdout="", stderr="")

    def http_json(self, url, data=None, timeout=5.0):
        if url.endswith("/slots"):
            if not self.backend_up:
                raise OSError("refused")
            busy = self.backend_busy_polls > 0
            self.backend_busy_polls = max(0, self.backend_busy_polls - 1)
            return [{"id": 0, "is_processing": busy}]
        if url.endswith("/queue") and data is None:
            return {"queue_running": [[1]] * self.comfy_queue, "queue_pending": []}
        if "/history" in url:
            return {f"p{i}": {"status": {"completed": True, "messages": [
                ["execution_start", {"timestamp": int(ts * 1000) - 5000}],
                ["execution_success", {"timestamp": int(ts * 1000)}]]}}
                for i, ts in enumerate(self.history[-5:])}
        if url.endswith("/system_stats"):
            # ComfyUI names whatever card it sees cuda:0, whichever it is.
            return {"devices": [{"name": "cuda:0 NVIDIA GeForce RTX 3090 : cudaMallocAsync"}]}
        return {}

    def http_status(self, url, timeout=5.0):
        if not self.backend_up:
            return 0
        return 503 if self.backend_latched else 200


class FakeLlmsCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = pathlib.Path(self._tmp.name)
        self.fake = FakeLlms(tmp)
        self.patches = mock.patch.multiple(
            lease,
            LEASE_FILE=tmp / "gpu1-lease.json",
            LOCK_FILE=tmp / "gpu1-lease.lock",
            COMFY_DEVICE_FILE=tmp / "comfyui-device.env",
            LOG_FILE=tmp / "gpu-lease.jsonl",
            HEAL_STATE_FILE=tmp / "backend-heal.json",
            RUNTIME_DIR=tmp,
            run=self.fake.run,
            Path=self._fake_path,
            http_json=self.fake.http_json,
            http_status=self.fake.http_status,
            sleep=lambda s: None,
            now=lambda: NOON,
            DRAIN_TIMEOUT=0.2,
            COMFY_READY_TIMEOUT=0.2,
            COMFY_QUEUE_TIMEOUT=0.2,
            BACKEND_READY_TIMEOUT=0.2,
            POLL=0.01,
        )
        self.patches.start()

    def _fake_path(self, p, *rest):
        # The cgroup's process list: ComfyUI is pid 4242 (999 is someone else).
        if str(p).endswith("cgroup.procs"):
            fake = mock.Mock()
            fake.read_text.return_value = "4242\n"
            return fake
        return pathlib.Path(p, *rest)

    def tearDown(self):
        self.patches.stop()
        self._tmp.cleanup()


class LeaseTests(FakeLlmsCase):
    def test_acquire_stops_the_backend_and_moves_comfy(self):
        self.fake.comfy_device = "0"  # drifted off its home card
        result = lease.acquire("test", 30, False)
        self.assertTrue(result["ok"])
        self.assertTrue(lease.LEASE_FILE.exists())
        self.assertFalse(self.fake.backend_up)
        self.assertEqual(self.fake.comfy_device, "1")
        order = [" ".join(c) for c in self.fake.commands]
        stop = order.index("sudo -n systemctl stop llm-a")
        restart = order.index("systemctl --user restart comfyui")
        self.assertLess(stop, restart)  # the card is empty before ComfyUI moves in
        watchdog = [c for c in self.fake.commands if c[0] == "systemd-run"][0]
        self.assertIn("Restart=on-failure", watchdog)
        self.assertEqual(watchdog[-1], "watch")
        self.assertEqual(result["expires"], (NOON + timedelta(minutes=30)).isoformat())
        self.assertIsNone(result["idle_minutes"])

    def test_release_puts_everything_back(self):
        lease.acquire("test", 30, False)
        result = lease.release("done")
        self.assertTrue(result["ok"], result)
        self.assertTrue(self.fake.backend_up)
        self.assertEqual(self.fake.comfy_device, "1")  # ComfyUI never leaves GPU 1
        self.assertFalse(lease.LEASE_FILE.exists())
        self.assertFalse(lease.COMFY_DEVICE_FILE.exists())
        events = [json.loads(l)["event"] for l in lease.LOG_FILE.read_text().splitlines()]
        self.assertEqual(events, ["acquire", "release"])

    def test_acquire_does_not_restart_comfy_when_already_home(self):
        lease.acquire("test", 30, False)
        self.assertNotIn(["systemctl", "--user", "restart", "comfyui"], self.fake.commands)

    def test_release_is_idempotent(self):
        self.assertTrue(lease.release("done")["ok"])
        self.assertTrue(lease.release("done")["ok"])
        self.assertNotIn(["systemctl", "--user", "restart", "comfyui"], self.fake.commands)

    def test_quiet_window_refuses_without_force(self):
        with mock.patch.object(lease, "now", lambda: NOON.replace(hour=23, minute=30)):
            with self.assertRaises(lease.LeaseError):
                lease.acquire("test", 30, False)
            self.assertFalse(lease.LEASE_FILE.exists())
            self.assertTrue(lease.acquire("test", 30, True)["ok"])

    def test_quiet_window_wraps_midnight(self):
        self.assertTrue(lease.in_quiet_window(NOON.replace(hour=2), "23:00-06:30"))
        self.assertTrue(lease.in_quiet_window(NOON.replace(hour=6, minute=29), "23:00-06:30"))
        self.assertFalse(lease.in_quiet_window(NOON.replace(hour=6, minute=30), "23:00-06:30"))
        self.assertFalse(lease.in_quiet_window(NOON, "23:00-06:30"))
        self.assertFalse(lease.in_quiet_window(NOON, ""))

    def test_second_acquire_is_refused(self):
        lease.acquire("one", 30, False)
        with self.assertRaises(lease.LeaseError):
            lease.acquire("two", 30, False)

    def test_busy_comfy_queue_is_refused_before_anything_changes(self):
        self.fake.comfy_queue = 2  # never drains
        with self.assertRaises(lease.LeaseError):
            lease.acquire("test", 30, False)
        self.assertTrue(self.fake.backend_up)
        self.assertFalse(lease.LEASE_FILE.exists())

    def test_someone_elses_job_is_waited_for(self):
        self.fake.comfy_queue = 1
        real = self.fake.http_json
        polls = {"n": 0}

        def draining(url, data=None, timeout=5.0):
            if url.endswith("/queue") and data is None:
                polls["n"] += 1
                if polls["n"] > 3:
                    self.fake.comfy_queue = 0
            return real(url, data, timeout)

        with mock.patch.object(lease, "http_json", draining):
            self.assertTrue(lease.acquire("test", 30, False)["ok"])

    def test_backend_that_stays_busy_gives_up_and_restores(self):
        self.fake.backend_busy_polls = 10**6
        with self.assertRaises(lease.LeaseError):
            lease.acquire("test", 30, False)
        self.assertTrue(self.fake.backend_up)
        self.assertFalse(lease.LEASE_FILE.exists())
        self.assertNotIn(["sudo", "-n", "systemctl", "stop", "llm-a"], self.fake.commands)

    def test_backend_that_finishes_its_request_is_waited_for(self):
        self.fake.backend_busy_polls = 3
        self.assertTrue(lease.acquire("test", 30, False)["ok"])

    def test_backend_that_will_not_start_keeps_the_fallback(self):
        lease.acquire("test", 30, False)
        self.fake.fail["systemctl start llm-a"] = "unit failed"
        result = lease.release("done")
        self.assertFalse(result["ok"])
        self.assertIn("unit failed", result["error"])
        # The proxy keeps falling back rather than returning 503s.
        self.assertTrue(lease.LEASE_FILE.exists())

    def test_status_reports_expiry(self):
        lease.acquire("test", 30, False)
        with mock.patch.object(lease, "now", lambda: NOON + timedelta(minutes=31)):
            self.assertTrue(lease.status()["expired"])
        self.assertFalse(lease.status()["expired"])



class Clock:
    """now() and sleep() for the watchdog loop: sleeping moves time on."""

    def __init__(self, start: datetime, limit: int = 2000):
        self.t = start
        self.sleeps = 0
        self.limit = limit

    def now(self):
        return self.t

    def sleep(self, seconds):
        if seconds < 1:
            return  # wait_until's polls inside release: not the watchdog's clock
        self.sleeps += 1
        if self.sleeps > self.limit:
            raise AssertionError("the watchdog never released")
        self.t += timedelta(seconds=seconds)

    def epoch(self, minutes: float) -> float:
        return (NOON + timedelta(minutes=minutes)).timestamp()


class IdleLeaseTests(FakeLlmsCase):
    """--idle-minutes: the watchdog releases once ComfyUI has been idle."""

    def setUp(self):
        super().setUp()
        self.clock = Clock(NOON)
        self.more = mock.patch.multiple(lease, now=self.clock.now, sleep=self.clock.sleep, WATCH_POLL=60)
        self.more.start()

    def tearDown(self):
        self.more.stop()
        super().tearDown()

    def watch_until_released(self):
        code = lease.watch()
        minutes = (self.clock.t - NOON).total_seconds() / 60
        events = [json.loads(l) for l in lease.LOG_FILE.read_text().splitlines()]
        return code, minutes, events[-1]

    def test_idle_lease_defaults_to_a_long_ceiling(self):
        result = lease.acquire("sillytavern", None, False, 60)
        self.assertEqual(result["idle_minutes"], 60)
        self.assertEqual(result["expires"], (NOON + timedelta(minutes=720)).isoformat())

    def test_released_after_an_idle_hour(self):
        lease.acquire("sillytavern", None, False, 60)
        code, minutes, event = self.watch_until_released()
        self.assertEqual(code, 0)
        self.assertTrue(60 <= minutes <= 61, minutes)
        self.assertEqual(event["event"], "release")
        self.assertTrue(event["reason"].startswith("watchdog: ComfyUI idle"))
        self.assertTrue(self.fake.backend_up)
        self.assertFalse(lease.LEASE_FILE.exists())

    def test_comfy_jobs_from_any_client_keep_it(self):
        # SillyTavern's own ComfyUI calls show up only in /history.
        lease.acquire("sillytavern", None, False, 60)
        self.fake.history = [self.clock.epoch(45)]
        code, minutes, _ = self.watch_until_released()
        self.assertTrue(105 <= minutes <= 106, minutes)

    def test_a_running_job_keeps_it_even_without_history(self):
        lease.acquire("sillytavern", None, False, 60)
        real_sleep = self.clock.sleep

        def sleep(seconds):
            real_sleep(seconds)
            # ComfyUI busy from minute 50 to 80, then restarted (history empty).
            self.fake.comfy_queue = 1 if 50 <= (self.clock.t - NOON).total_seconds() / 60 < 80 else 0

        with mock.patch.object(lease, "sleep", sleep):
            code, minutes, _ = self.watch_until_released()
        self.assertTrue(139 <= minutes <= 141, minutes)

    def test_touch_resets_the_idle_clock(self):
        lease.acquire("sillytavern", None, False, 60)
        self.clock.t = NOON + timedelta(minutes=50)
        result = lease.touch()
        self.assertTrue(result["ok"])
        self.assertEqual(result["last_activity"], self.clock.t.isoformat())
        code, minutes, _ = self.watch_until_released()
        self.assertTrue(110 <= minutes <= 111, minutes)

    def test_hard_ceiling_ends_a_busy_lease(self):
        lease.acquire("sillytavern", 120, False, 60)
        self.fake.comfy_queue = 1  # a stuck job: never idle
        with mock.patch.object(lease, "COMFY_QUEUE_TIMEOUT", 0.05):
            code, minutes, event = self.watch_until_released()
        self.assertTrue(120 <= minutes <= 121, minutes)
        self.assertEqual(event["reason"], "watchdog: hard ceiling")
        self.assertTrue(self.fake.backend_up)

    def test_quiet_window_shortens_the_idle_time(self):
        self.clock.t = NOON.replace(hour=22, minute=30)
        start = self.clock.t
        lease.acquire("sillytavern", None, False, 60)
        self.fake.history = [(start + timedelta(minutes=29)).timestamp()]
        code, _, event = self.watch_until_released()
        # Idle from 22:59; at 23:00 the limit drops to 5 minutes.
        self.assertEqual((self.clock.t - start).total_seconds() // 60, 34)
        self.assertTrue(event["reason"].startswith("watchdog: ComfyUI idle"))

    def test_forced_lease_keeps_its_idle_time_overnight(self):
        self.clock.t = NOON.replace(hour=23, minute=30)
        start = self.clock.t
        lease.acquire("sillytavern", None, True, 60)
        code, _, _ = self.watch_until_released()
        self.assertEqual((self.clock.t - start).total_seconds() // 60, 60)

    def test_fixed_lease_waits_for_its_end_plus_grace(self):
        lease.acquire("comfyui-edit", 30, False)
        self.fake.history = []  # idle all along: irrelevant for a fixed lease
        code, minutes, event = self.watch_until_released()
        self.assertTrue(40 <= minutes <= 41, minutes)
        self.assertEqual(event["reason"], "watchdog: past its end")

    def test_watchdog_does_not_stop_itself(self):
        lease.acquire("sillytavern", None, False, 60)
        self.fake.commands.clear()
        self.watch_until_released()
        self.assertFalse(any(c[:3] == ["systemctl", "--user", "stop"] for c in self.fake.commands))

    def test_watch_exits_when_released_elsewhere(self):
        self.assertEqual(lease.watch(), 0)

    def test_failed_release_exits_nonzero_for_a_retry(self):
        lease.acquire("sillytavern", None, False, 60)
        self.fake.fail["systemctl start llm-a"] = "unit failed"
        code, _, _ = self.watch_until_released()
        self.assertEqual(code, 1)  # systemd restarts the watchdog
        self.assertTrue(lease.LEASE_FILE.exists())  # the proxy keeps the fallback

    def test_same_holder_renews_an_idle_lease(self):
        lease.acquire("sillytavern", None, False, 60)
        self.fake.commands.clear()
        self.clock.t = NOON + timedelta(minutes=30)
        result = lease.acquire("sillytavern", None, False, 45)
        self.assertTrue(result["renewed"])
        self.assertEqual(result["idle_minutes"], 45)
        self.assertEqual(result["expires"], (self.clock.t + timedelta(minutes=720)).isoformat())
        self.assertNotIn(["sudo", "-n", "systemctl", "stop", "llm-a"], self.fake.commands)
        self.assertFalse(result["watchdog_rearmed"])

    def test_other_holders_and_fixed_leases_are_still_refused(self):
        lease.acquire("sillytavern", None, False, 60)
        with self.assertRaises(lease.LeaseError):
            lease.acquire("hermes", None, False, 60)
        with self.assertRaises(lease.LeaseError):
            lease.acquire("sillytavern", 30, False)  # fixed: would release under the holder

    def test_touch_rearms_a_dead_watchdog_and_extends_the_end(self):
        lease.acquire("sillytavern", None, False, 60)
        self.fake.watchdog_up = False
        result = lease.touch(900)
        self.assertTrue(result["watchdog_rearmed"])
        self.assertEqual(result["expires"], (NOON + timedelta(minutes=900)).isoformat())

    def test_touch_without_a_lease_fails(self):
        with self.assertRaises(lease.LeaseError):
            lease.touch()

    def test_status_reports_idle_time(self):
        lease.acquire("sillytavern", None, False, 60)
        self.fake.history = [self.clock.epoch(10)]
        self.clock.t = NOON + timedelta(minutes=25)
        st = lease.status()
        self.assertEqual(st["idle_minutes"], 15.0)
        self.assertEqual(st["releases_at"], (NOON + timedelta(minutes=70)).isoformat())
        self.assertTrue(st["watchdog_active"])
        self.assertFalse(st["comfy_busy"])
        self.assertFalse(st["expired"])

    def test_main_routes_the_new_commands(self):
        out = []
        with mock.patch("builtins.print", lambda s: out.append(json.loads(s))):
            self.assertEqual(lease.main(["acquire", "--holder", "sillytavern", "--idle-minutes", "60"]), 0)
            self.assertEqual(lease.main(["touch"]), 0)
            self.assertEqual(lease.main(["release"]), 0)
        self.assertEqual(out[0]["idle_minutes"], 60)
        self.assertIn("last_activity", out[1])
        self.assertTrue(out[2]["ok"])


class StickyLeaseTests(FakeLlmsCase):
    """--sticky (ComfyUI mode): held until a release without a holder."""

    def setUp(self):
        super().setUp()
        self.clock = Clock(NOON, limit=3 * 24 * 60)
        self.more = mock.patch.multiple(lease, now=self.clock.now, sleep=self.clock.sleep, WATCH_POLL=60)
        self.more.start()

    def tearDown(self):
        self.more.stop()
        super().tearDown()

    def test_sticky_lease_has_no_end(self):
        result = lease.acquire("comfy-mode", None, False, sticky=True)
        self.assertTrue(result["ok"])
        self.assertTrue(result["sticky"])
        self.assertIsNone(result["expires"])
        self.assertIsNone(result["idle_minutes"])
        self.assertFalse(self.fake.backend_up)
        st = lease.status()
        self.assertIsNone(st["releases_at"])
        self.assertNotIn("expired", st)

    def test_watchdog_never_releases_it_even_overnight(self):
        lease.acquire("comfy-mode", None, False, sticky=True)
        # Two idle days, through two quiet windows: still held.
        with mock.patch.object(lease, "read_lease", side_effect=self._stop_after(2 * 24 * 60)):
            self.assertEqual(lease.watch(), 0)
        self.assertTrue(lease.LEASE_FILE.exists())
        self.assertFalse(self.fake.backend_up)

    def _stop_after(self, minutes):
        real = lease.read_lease  # taken before the patch below replaces it

        def read():
            if (self.clock.t - NOON).total_seconds() / 60 >= minutes:
                return None  # end the loop as a release elsewhere would
            return real()
        return read

    def test_a_jobs_release_keeps_it(self):
        lease.acquire("comfy-mode", None, False, sticky=True)
        result = lease.release("done", holder="comfyui-edit")
        self.assertTrue(result["ok"])
        self.assertTrue(result["kept"])
        self.assertTrue(lease.LEASE_FILE.exists())
        self.assertFalse(self.fake.backend_up)

    def test_release_without_holder_turns_it_off(self):
        lease.acquire("comfy-mode", None, False, sticky=True)
        result = lease.release("comfy mode off")
        self.assertTrue(result["ok"], result)
        self.assertFalse(lease.LEASE_FILE.exists())
        self.assertTrue(self.fake.backend_up)

    def test_turning_it_on_converts_a_held_job_lease(self):
        lease.acquire("comfyui-edit", 30, False)
        result = lease.acquire("comfy-mode", None, False, sticky=True)
        self.assertTrue(result["converted"])
        self.assertEqual(result["converted_from"], "comfyui-edit")
        self.assertIsNone(result["expires"])
        # The job finishes and releases its own lease: ComfyUI mode stays on.
        self.assertTrue(lease.release("done", holder="comfyui-edit")["kept"])
        self.assertTrue(lease.LEASE_FILE.exists())

    def test_on_twice_is_a_no_op(self):
        lease.acquire("comfy-mode", None, False, sticky=True)
        n = len(self.fake.commands)
        self.assertTrue(lease.acquire("comfy-mode", None, False, sticky=True)["already"])
        self.assertEqual(len(self.fake.commands), n)

    def test_idle_acquire_shares_it_and_fixed_is_refused(self):
        lease.acquire("comfy-mode", None, False, sticky=True)
        shared = lease.acquire("sillytavern", None, False, 60)
        self.assertTrue(shared["shared"])
        self.assertTrue(lease.read_lease()["sticky"])
        with self.assertRaises(lease.LeaseError):
            lease.acquire("comfyui-edit", 30, False)

    def test_quiet_window_needs_force_to_turn_it_on(self):
        self.clock.t = NOON.replace(hour=23, minute=30)
        with self.assertRaises(lease.LeaseError):
            lease.acquire("comfy-mode", None, False, sticky=True)
        self.assertTrue(lease.acquire("comfy-mode", None, True, sticky=True)["ok"])

    def test_main_sticky_defaults_the_holder(self):
        out = []
        with mock.patch("builtins.print", lambda s: out.append(json.loads(s))):
            self.assertEqual(lease.main(["acquire", "--sticky"]), 0)
            self.assertEqual(lease.main(["release", "--holder", "comfyui"]), 0)
            self.assertEqual(lease.main(["release", "--reason", "comfy mode off"]), 0)
        self.assertEqual(out[0]["holder"], "comfy-mode")
        self.assertTrue(out[1]["kept"])
        self.assertFalse(lease.LEASE_FILE.exists())


class HealTests(FakeLlmsCase):
    """heal: restart a backend that runs but has latched itself unavailable."""

    def setUp(self):
        super().setUp()
        self.clock = Clock(NOON)
        self.more = mock.patch.multiple(lease, now=self.clock.now)
        self.more.start()

    def tearDown(self):
        self.more.stop()
        super().tearDown()

    def check(self, after_seconds: float = 0):
        self.clock.t += timedelta(seconds=after_seconds)
        return lease.heal()

    def restarts(self):
        return [c for c in self.fake.commands if c[:4] == ["sudo", "-n", "systemctl", "restart"]]

    def heal_events(self):
        if not lease.LOG_FILE.exists():
            return []
        return [e for e in map(json.loads, lease.LOG_FILE.read_text().splitlines()) if e["event"] == "heal"]

    def test_a_healthy_backend_is_left_alone(self):
        self.assertEqual(self.check()["action"], "none")
        self.assertEqual(self.restarts(), [])

    def test_a_latched_backend_is_restarted_after_two_checks(self):
        self.fake.backend_latched = True
        self.assertEqual(self.check()["action"], "watching")
        self.assertEqual(self.check(60)["action"], "watching")  # 60 s < 90 s: not yet confirmed
        result = self.check(60)
        self.assertEqual(result["action"], "restarted")
        self.assertTrue(result["recovered"])
        self.assertEqual(len(self.restarts()), 1)
        self.assertEqual(self.heal_events()[-1]["health"], 503)
        self.assertIn("last_heal", lease.status())

    def test_one_bad_check_then_healthy_does_not_restart(self):
        self.fake.backend_latched = True
        self.check()
        self.fake.backend_latched = False
        self.assertEqual(self.check(120)["action"], "none")
        self.fake.backend_latched = True
        self.assertEqual(self.check(120)["action"], "watching")  # the clock started again
        self.assertEqual(self.restarts(), [])

    def test_a_lease_means_hands_off(self):
        lease.acquire("comfyui", 30, False)
        self.fake.backend_up, self.fake.backend_latched = True, True
        self.check()
        self.assertEqual(self.check(300)["why"], "lease held")
        self.assertEqual(self.restarts(), [])

    def test_a_stopped_unit_is_not_started(self):
        self.fake.backend_up = False
        self.check()
        self.assertEqual(self.check(300)["action"], "none")
        self.assertEqual(self.restarts(), [])

    def test_a_backend_still_loading_is_given_time(self):
        self.fake.backend_latched = True
        self.fake.backend_since = NOON.timestamp() - 60
        self.check()
        self.assertEqual(self.check(120)["why"], "backend still loading")
        self.assertEqual(self.restarts(), [])

    def test_gives_up_after_three_restarts_an_hour_and_logs_once(self):
        self.fake.backend_latched = True
        self.fake.restart_heals = False
        actions = []
        for _ in range(12):
            actions.append(self.check(100)["action"])
            self.fake.backend_since = self.clock.t.timestamp() - 3600  # past the load grace
        self.assertEqual(len(self.restarts()), 3)
        self.assertIn("gave up", actions)
        self.assertEqual([e["action"] for e in self.heal_events()].count("gave up"), 1)
        self.assertTrue(lease.status()["heal_gave_up"])

    def test_main_heal_skips_quietly_while_a_lease_command_runs(self):
        out = []
        with mock.patch("builtins.print", lambda s: out.append(json.loads(s))), \
                mock.patch.object(lease, "locked", side_effect=lease.LeaseError("another gpu-lease command is running")):
            self.assertEqual(lease.main(["heal"]), 0)
        self.assertEqual(out[0]["action"], "none")


if __name__ == "__main__":
    unittest.main()
