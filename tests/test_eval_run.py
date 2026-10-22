"""eval-run.py: what `serve` leaves for a driving tool to check, and where it
puts a candidate on a CUDA host. Nothing here starts a server."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import pathlib
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock


def _load():
    root = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("eval_run", root / "scripts" / "eval-run.py")
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


evalrun = _load()


def _dead_pid() -> int:
    child = subprocess.Popen(["true"])
    child.wait()  # reaped: the pid no longer names a process
    return child.pid


class StatusTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.state = pathlib.Path(self._tmp.name) / ".eval-server.json"
        self.patches = mock.patch.multiple(evalrun, STATE=self.state, _is_ready=mock.Mock(return_value=True))
        self.patches.start()

    def tearDown(self):
        self.patches.stop()
        self._tmp.cleanup()

    def serving(self, pid: int):
        self.state.write_text(json.dumps({"pid": pid, "candidate": "cand", "url": "http://127.0.0.1:8020",
                                          "slot": "llm-b"}))

    def test_nothing_served(self):
        out, code = evalrun.status()
        self.assertEqual(code, 1)
        self.assertEqual(out, {"candidate": None, "pid": None, "alive": False, "ready": False, "url": None})
        evalrun._is_ready.assert_not_called()

    def test_serving_and_ready(self):
        self.serving(os.getpid())
        out, code = evalrun.status()
        self.assertEqual(code, 0)
        self.assertEqual(out, {"candidate": "cand", "pid": os.getpid(), "alive": True, "ready": True,
                               "url": "http://127.0.0.1:8020"})
        evalrun._is_ready.assert_called_once_with("http://127.0.0.1:8020")

    def test_running_but_not_ready(self):
        self.serving(os.getpid())
        evalrun._is_ready.return_value = False
        out, code = evalrun.status()
        self.assertEqual(code, 2)
        self.assertTrue(out["alive"])
        self.assertFalse(out["ready"])

    def test_a_dead_server_is_not_running_even_if_the_port_answers(self):
        # Someone else may be on the port; it is not the candidate.
        self.serving(_dead_pid())
        out, code = evalrun.status()
        self.assertEqual(code, 1)
        self.assertFalse(out["alive"])
        self.assertEqual(out["candidate"], "cand")
        evalrun._is_ready.assert_not_called()

    def test_main_prints_json_and_needs_no_candidates_file(self):
        self.serving(os.getpid())
        buf = io.StringIO()
        with redirect_stdout(buf), mock.patch.object(evalrun, "load_candidates",
                                                     side_effect=AssertionError("read candidates")):
            code = evalrun.main(["status"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(buf.getvalue())["candidate"], "cand")


if __name__ == "__main__":
    unittest.main()
