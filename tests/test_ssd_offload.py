"""SSD offload: one setting set per slot, translated per engine.

The property that matters most is the negative one -- a slot left at
`resident` runs exactly the command it ran before any of this existed, which
`tests/test_launchers.py` pins against the golden files. These cover the rest:
what an offloaded slot adds on each engine, and what it leaves alone.
"""

from __future__ import annotations

import os
import pathlib
import shlex
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from platform_harness import as_darwin  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "web"))
import backends  # noqa: E402
import budget  # noqa: E402
import config_fields  # noqa: E402
from backends import offload  # noqa: E402

BASE = {
    "LLAMA_SERVER_BIN": "/bin/llama-server",
    "LISTEN_HOST": "0.0.0.0",
    "LLM_B_MODEL_PATH": "/models/flash-next.gguf",
    "LLM_B_NO_MMAP": "true",
    "LLM_B_MLOCK": "true",
}

#: Every offload setting, so a test can show that at `resident` none of them
#: reach the command line.
ALL_SET = {"LLM_B_RAM_BUDGET_GB": "64", "LLM_B_NGRAM_PREWARM": "auto",
           "LLM_B_EXPERT_CACHE_GB": "40", "LLM_B_EXPERT_STREAM_IO_THREADS": "4"}


def fake_server(directory: pathlib.Path, name: str, *flags: str) -> str:
    """A llama-server whose --help lists `flags`."""
    path = directory / name
    path.write_text("#!/bin/sh\n" + "".join(f"echo '  {flag}'\n" for flag in flags))
    path.chmod(0o755)
    return str(path)


class OffloadTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = pathlib.Path(tmp.name)
        # Answers are cached per binary path; each test's binaries are new.
        offload._help_text.cache_clear()
        self.addCleanup(offload._help_text.cache_clear)


class LlamaCppOffloadTests(OffloadTestCase):
    def build(self, **env):
        return backends.build_command("llm-b", dict(BASE, **env))

    def test_resident_ignores_every_offload_setting(self):
        self.assertEqual(self.build(**ALL_SET),
                         self.build())
        self.assertEqual(self.build(LLM_B_MEMORY_MODE="resident", **ALL_SET),
                         self.build())

    def test_an_unknown_mode_is_resident(self):
        self.assertEqual(self.build(LLM_B_MEMORY_MODE="swap-everything"), self.build())

    def test_offload_leaves_the_weights_pageable(self):
        argv = self.build(LLM_B_MEMORY_MODE="ssd-offload")
        self.assertNotIn("--no-mmap", argv)
        self.assertNotIn("--mlock", argv)
        self.assertIn("--no-mmap", self.build())

    def test_the_ngram_table_is_kept_out_of_the_metal_buffers(self):
        argv = self.build(LLM_B_MEMORY_MODE="ssd-offload")
        at = argv.index("--override-tensor")
        self.assertEqual(argv[at + 1], f"{offload.NGRAM_TENSOR_PATTERN}=CPU")

    def test_an_operator_override_tensor_is_theirs(self):
        argv = self.build(LLM_B_MEMORY_MODE="ssd-offload",
                          LLM_B_CUSTOM_ARGS_JSON='["-ot blk\\\\..*_exps=CPU"]')
        self.assertEqual(argv.count("-ot"), 1)
        self.assertNotIn("--override-tensor", argv)

    def test_the_pinned_build_gets_no_flags_it_lacks(self):
        binary = fake_server(self.tmp, "old-server", "--mlock")
        argv = self.build(LLM_B_MEMORY_MODE="ssd-offload", LLAMA_SERVER_BIN=binary, **ALL_SET)
        self.assertNotIn("--load-mode", argv)
        self.assertFalse([a for a in argv if a.startswith("--moe-stream")])

    def test_a_build_with_load_mode_is_told_mmap(self):
        binary = fake_server(self.tmp, "server", "--load-mode")
        argv = self.build(LLM_B_MEMORY_MODE="ssd-offload", LLAMA_SERVER_BIN=binary)
        self.assertEqual(argv[argv.index("--load-mode") + 1], "mmap")

    def test_a_streaming_build_streams_with_the_configured_cache(self):
        binary = fake_server(self.tmp, "stream", "--load-mode", "--moe-stream",
                             "--moe-stream-direct")
        argv = self.build(LLM_B_MEMORY_MODE="ssd-offload", LLM_B_LLAMA_SERVER_BIN=binary,
                          **ALL_SET)
        at = argv.index("--moe-stream")
        self.assertEqual(argv[at:at + 6], ["--moe-stream", "--moe-stream-cache", "40",
                                           "--moe-stream-io-threads", "4",
                                           "--moe-stream-direct"])

    def test_a_blank_cache_lets_the_build_choose(self):
        binary = fake_server(self.tmp, "stream", "--moe-stream")
        argv = self.build(LLM_B_MEMORY_MODE="ssd-offload", LLM_B_LLAMA_SERVER_BIN=binary)
        self.assertIn("--moe-stream", argv)
        self.assertNotIn("--moe-stream-cache", argv)
        self.assertEqual(argv[argv.index("--moe-stream-io-threads") + 1],
                         offload.DEFAULT_IO_THREADS)

    def test_a_typo_never_reaches_a_flag(self):
        binary = fake_server(self.tmp, "stream", "--moe-stream")
        argv = self.build(LLM_B_MEMORY_MODE="ssd-offload", LLM_B_LLAMA_SERVER_BIN=binary,
                          LLM_B_EXPERT_CACHE_GB="forty")
        self.assertNotIn("--moe-stream-cache", argv)

    def test_a_slot_binary_serves_that_slot_only(self):
        env = dict(BASE, LLM_B_LLAMA_SERVER_BIN="/opt/fork/llama-server",
                   LLM_A_MODEL_PATH="/models/a.gguf")
        self.assertEqual(backends.build_command("llm-b", env)[0], "/opt/fork/llama-server")
        self.assertEqual(backends.build_command("llm-a", env)[0], "/bin/llama-server")

    def test_a_binary_that_will_not_run_supports_nothing(self):
        self.assertFalse(offload.server_supports(str(self.tmp / "absent"), "--load-mode"))


class MtplxOffloadTests(OffloadTestCase):
    def setUp(self):
        super().setUp()
        self.pack = self.tmp / "Qwen3.8-Flash-Next-MTPLX-Bare-Speed"
        self.pack.mkdir()
        (self.pack / "mtplx_runtime.json").write_text("{}")
        self.env = dict(BASE, STACK_DIR="/stack", LLM_B_MODEL_PATH=str(self.pack))

    def build(self, **env):
        with as_darwin():
            return backends.build_command("llm-b", dict(self.env, **env))

    def test_resident_ignores_every_offload_setting(self):
        self.assertEqual(self.build(**ALL_SET), self.build())

    def test_the_budget_caps_mtplx_and_the_table_prewarm_is_passed(self):
        argv = self.build(LLM_B_MEMORY_MODE="ssd-offload", **ALL_SET)
        serve = argv.index("serve")
        self.assertIn(f"MTPLX_MEMORY_LIMIT_BYTES={64 * 1024}M", argv[:serve])
        self.assertEqual(argv[argv.index("--ngram-prewarm") + 1], "auto")

    def test_a_prewarm_size_is_a_gib_count(self):
        argv = self.build(LLM_B_MEMORY_MODE="ssd-offload", LLM_B_NGRAM_PREWARM="12")
        self.assertEqual(argv[argv.index("--ngram-prewarm") + 1], "12")

    def test_an_expert_cache_does_not_stop_a_pack_starting(self):
        # MTPLX has no expert streaming; the same slot may have been set up for
        # a streaming GGUF and switched to a pack by choosing the model.
        argv = self.build(LLM_B_MEMORY_MODE="ssd-offload", LLM_B_EXPERT_CACHE_GB="40")
        self.assertFalse([a for a in argv if "moe-stream" in a])
        self.assertNotIn("--override-tensor", argv)


class SlotFactsTests(unittest.TestCase):
    def facts(self, **env) -> str:
        run_env = dict(os.environ, STACK_DIR=str(ROOT), **BASE, **env)
        done = subprocess.run([sys.executable, str(ROOT / "scripts/lib/slot-facts.py"), "llm-b"],
                              env=run_env, capture_output=True, text=True, check=True)
        return done.stdout

    def test_the_launcher_learns_the_mode_and_binary(self):
        out = self.facts(LLM_B_MEMORY_MODE="ssd-offload",
                         LLM_B_LLAMA_SERVER_BIN="/opt/fork/llama-server")
        self.assertIn("FACT_MEMORY_MODE=ssd-offload", out)
        self.assertIn("FACT_LLAMA_SERVER_BIN=/opt/fork/llama-server", out)

    def test_the_report_names_the_budget_only_when_offloaded(self):
        resident = self.facts(LLM_B_RAM_BUDGET_GB="64")
        self.assertNotIn("memory_mode=", resident)
        offloaded = self.facts(LLM_B_MEMORY_MODE="ssd-offload", LLM_B_RAM_BUDGET_GB="64")
        line = next(l for l in offloaded.splitlines() if l.startswith("FACT_PREFLIGHT="))
        self.assertIn("ram_budget_gb=64", shlex.split(line[len("FACT_PREFLIGHT=("):-1]))


class ConfigSurfaceTests(unittest.TestCase):
    def test_both_chat_slots_have_the_settings_and_restart_on_them(self):
        keys = {f["key"] for f in config_fields.CONFIG_FIELDS if f["section"] == "SSD Offload"}
        for slot, unit in (("LLM_A", "llm-a"), ("LLM_B", "llm-b")):
            for suffix in ("MEMORY_MODE", "RAM_BUDGET_GB", "NGRAM_PREWARM",
                           "EXPERT_CACHE_GB", "EXPERT_STREAM_IO_THREADS", "LLAMA_SERVER_BIN"):
                self.assertIn(f"{slot}_{suffix}", keys)
                self.assertEqual(config_fields.RESTART_HINTS[f"{slot}_{suffix}"], [unit])
        self.assertIn("SSD Offload", config_fields.CORE_CONFIG_SECTIONS)

    def test_resident_is_the_first_choice(self):
        field = next(f for f in config_fields.CONFIG_FIELDS if f["key"] == "LLM_A_MEMORY_MODE")
        self.assertEqual(field["options"][0], "resident")


class BudgetTests(unittest.TestCase):
    #: A 111 GB GGUF whose KV, compute and overhead come to 10 GiB.
    PREDICTION = {"vram": {"total_mib": 113_664 + 10_240, "weights_mib": 113_664}}
    GEOMETRY = {"file_size_mib": 113_664}

    def test_the_budget_decides_what_is_resident(self):
        got = budget.offload_weights(self.GEOMETRY, {"memory_mode": "ssd-offload",
                                                     "ram_budget_gb": "64"}, self.PREDICTION)
        self.assertEqual(got["resident_weights_mib"], 64 * 1024 - 10_240)
        self.assertEqual(got["on_ssd_mib"], 113_664 - (64 * 1024 - 10_240))

    def test_a_budget_bigger_than_the_model_holds_it_all(self):
        got = budget.offload_weights(self.GEOMETRY, {"memory_mode": "ssd-offload",
                                                     "ram_budget_gb": "200"}, self.PREDICTION)
        self.assertEqual((got["resident_weights_mib"], got["on_ssd_mib"]), (113_664, 0))

    def test_no_budget_prices_the_whole_file_and_says_so(self):
        got = budget.offload_weights(self.GEOMETRY, {"memory_mode": "ssd-offload"},
                                     self.PREDICTION)
        self.assertEqual(got["resident_weights_mib"], 113_664)
        self.assertIn("no RAM budget", got["note"])

    def test_resident_is_not_an_offload(self):
        self.assertIsNone(budget.offload_weights(self.GEOMETRY, {"ram_budget_gb": "64"},
                                                 self.PREDICTION))

    def test_the_env_carries_the_mode_to_the_budget(self):
        settings = budget.settings_from_env({"LLM_B_MEMORY_MODE": "ssd-offload",
                                             "LLM_B_RAM_BUDGET_GB": "64"}, "llm-b")
        self.assertEqual((settings["memory_mode"], settings["ram_budget_gb"]),
                         ("ssd-offload", "64"))

    def test_a_pack_reports_its_streamed_table_apart(self):
        with tempfile.TemporaryDirectory() as tmp:
            pack = pathlib.Path(tmp)
            (pack / "mtplx_runtime.json").write_text("{}")
            (pack / "model-00001-of-00001.safetensors").write_bytes(b"\0" * (3 * 1024 * 1024))
            (pack / "ngram-table.safetensors").write_bytes(b"\0" * (2 * 1024 * 1024))
            result = budget.budget_for({"LLM_B_MODEL_PATH": str(pack)}, "llm-b")
        self.assertEqual(result["pack"], {"weights_mib": 3, "ssd_streamed_mib": 2})


if __name__ == "__main__":
    unittest.main()
