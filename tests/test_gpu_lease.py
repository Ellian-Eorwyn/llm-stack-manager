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
        self.comfy_device = "0"
        self.comfy_queue = 0
        self.commands: list[list[str]] = []
        self.fail: dict[str, str] = {}

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
        elif cmd[:3] == ["systemctl", "--user", "restart"]:
            dev_file = lease.COMFY_DEVICE_FILE
            self.comfy_device = dev_file.read_text().strip().split("=")[1] if dev_file.exists() else "0"
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
        if url.endswith("/system_stats"):
            return {"devices": [{"name": f"cuda:{self.comfy_device} NVIDIA GeForce RTX 3090 : cudaMallocAsync"}]}
        return {}

    def http_status(self, url, timeout=5.0):
        return 200 if self.backend_up else 0


class LeaseTests(unittest.TestCase):
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
            RUNTIME_DIR=tmp,
            run=self.fake.run,
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

    def tearDown(self):
        self.patches.stop()
        self._tmp.cleanup()

    def test_acquire_stops_the_backend_and_moves_comfy(self):
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
        self.assertIn("--on-active=2400", watchdog)
        self.assertEqual(watchdog[-3:], ["release", "--reason", "watchdog"])

    def test_release_puts_everything_back(self):
        lease.acquire("test", 30, False)
        result = lease.release("done")
        self.assertTrue(result["ok"], result)
        self.assertTrue(self.fake.backend_up)
        self.assertEqual(self.fake.comfy_device, "0")
        self.assertFalse(lease.LEASE_FILE.exists())
        self.assertFalse(lease.COMFY_DEVICE_FILE.exists())
        events = [json.loads(l)["event"] for l in lease.LOG_FILE.read_text().splitlines()]
        self.assertEqual(events, ["acquire", "release"])

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
        self.fake.comfy_queue = 2
        with self.assertRaises(lease.LeaseError):
            lease.acquire("test", 30, False)
        self.assertTrue(self.fake.backend_up)
        self.assertFalse(lease.LEASE_FILE.exists())

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


if __name__ == "__main__":
    unittest.main()
