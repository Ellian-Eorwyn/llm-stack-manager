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


class CardTests(unittest.TestCase):
    """The slot's GPU_VISIBLE_DEVICES reaches the process, as start-backend.sh does it."""

    #: A two-card host's env file: LLM B's saved card is GPU 0.
    BASE = {"LLM_B_GPU_VISIBLE_DEVICES": "0"}

    def env(self, settings=None, extra=None, base=None):
        candidate = {"name": "cand", "settings": settings or {}, "env": extra or {}}
        return evalrun.candidate_env("llm-b", candidate, dict(self.BASE if base is None else base))

    def test_the_candidates_card_is_the_visible_one(self):
        env = self.env({"GPU_VISIBLE_DEVICES": "1"})
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1")
        self.assertEqual(env["LLM_B_GPU_VISIBLE_DEVICES"], "1")

    def test_without_one_the_slots_card_is_used_as_the_launcher_would(self):
        self.assertEqual(self.env()["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(self.env(base={"CHAT2_GPU_VISIBLE_DEVICES": "0"})["CUDA_VISIBLE_DEVICES"], "0")

    def test_nothing_set_leaves_the_environment_alone(self):
        self.assertNotIn("CUDA_VISIBLE_DEVICES", self.env(base={}))

    def test_absolute_indices_are_not_renumbered(self):
        env = self.env({"GPU_VISIBLE_DEVICES": "1"}, base={**self.BASE, "LLM_ABSOLUTE_GPU_INDICES": "on"})
        self.assertNotIn("CUDA_VISIBLE_DEVICES", env)

    def test_a_candidates_own_env_wins(self):
        env = self.env({"GPU_VISIBLE_DEVICES": "1"}, extra={"CUDA_VISIBLE_DEVICES": "1,0"})
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1,0")

    def test_shown_names_the_cards(self):
        self.assertEqual(evalrun.shown(["llama-server", "-m", "m.gguf"], {"CUDA_VISIBLE_DEVICES": "1"}),
                         "CUDA_VISIBLE_DEVICES=1 llama-server -m m.gguf")
        self.assertEqual(evalrun.shown(["mtplx", "serve"], {}), "mtplx serve")


class ServeTests(unittest.TestCase):
    """serve() up to the launch, with the launch itself a mock."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = pathlib.Path(self._tmp.name)
        (tmp / "benchmarks").mkdir()
        self.popen = mock.Mock(return_value=mock.Mock(pid=4242))
        self.stop = mock.Mock()
        self.patches = mock.patch.multiple(
            evalrun, ROOT=tmp, STATE=tmp / "benchmarks" / ".eval-server.json",
            sourced_env=mock.Mock(return_value=dict(CardTests.BASE)),
            stop=self.stop, _port_owner=mock.Mock(return_value=None),
            build=mock.Mock(return_value=["llama-server", "-m", "m.gguf"]),
            wait_healthy=mock.Mock(return_value=True),
            _cuda_host=mock.Mock(return_value=True))
        self.patches.start()
        self.more = mock.patch.object(evalrun.subprocess, "Popen", self.popen)
        self.more.start()

    def tearDown(self):
        self.more.stop()
        self.patches.stop()
        self._tmp.cleanup()

    def test_the_card_reaches_the_server_and_its_log(self):
        url = evalrun.serve("llm-b", {"name": "cand", "settings": {"GPU_VISIBLE_DEVICES": "1"}},
                            pathlib.Path("llm-stack.env"))
        self.assertEqual(url, "http://127.0.0.1:8020")
        self.assertEqual(self.popen.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "1")
        log = next((evalrun.ROOT / "benchmarks" / "cand").glob("server-*.log"))
        self.assertTrue(log.read_text().startswith("CUDA_VISIBLE_DEVICES=1 llama-server"))

    def test_a_cuda_host_refuses_a_candidate_without_a_card_before_stopping_anything(self):
        with self.assertRaises(SystemExit) as caught:
            evalrun.serve("llm-b", {"name": "cand", "settings": {"MODEL_PATH": "m.gguf"}},
                          pathlib.Path("llm-stack.env"))
        message = str(caught.exception)
        self.assertIn('"GPU_VISIBLE_DEVICES"', message)
        self.assertIn("LLM_B_GPU_VISIBLE_DEVICES=0", message)  # what it would have inherited
        self.stop.assert_not_called()  # a server already up is left alone
        self.popen.assert_not_called()

    def test_without_nvidia_smi_no_card_is_needed(self):
        evalrun._cuda_host.return_value = False
        evalrun.serve("llm-b", {"name": "cand", "settings": {}}, pathlib.Path("llm-stack.env"))
        self.popen.assert_called_once()

    def test_an_empty_card_is_no_card(self):
        with self.assertRaises(SystemExit):
            evalrun.serve("llm-b", {"name": "cand", "settings": {"GPU_VISIBLE_DEVICES": " "}},
                          pathlib.Path("llm-stack.env"))

    def test_a_slot_with_no_saved_card_says_so(self):
        evalrun.sourced_env.return_value = {}
        with self.assertRaises(SystemExit) as caught:
            evalrun.serve("llm-b", {"name": "cand"}, pathlib.Path("llm-stack.env"))
        self.assertIn("every card", str(caught.exception))

    def test_run_skips_it_and_plan_says_so(self):
        candidates = pathlib.Path(self._tmp.name) / "candidates.json"
        candidates.write_text(json.dumps({"slot": "llm-b", "candidates": [
            {"name": "placed", "settings": {"GPU_VISIBLE_DEVICES": "1"}},
            {"name": "unplaced", "settings": {}}]}))
        common = ["--candidates", str(candidates), "--env-file", "llm-stack.env"]
        buf = io.StringIO()
        with redirect_stdout(buf):
            evalrun.main(["plan", *common])
        plan = buf.getvalue()
        self.assertIn("placed:\n  CUDA_VISIBLE_DEVICES=1 llama-server", plan)
        self.assertIn("WILL NOT SERVE -- unplaced:", plan)
        buf = io.StringIO()
        with redirect_stdout(buf), mock.patch.object(evalrun.subprocess, "run"):
            evalrun.main(["run", "--only", "unplaced", "--suites", "speed", *common])
        self.assertIn("skipped: unplaced: this host has NVIDIA GPUs", buf.getvalue())
        self.popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
