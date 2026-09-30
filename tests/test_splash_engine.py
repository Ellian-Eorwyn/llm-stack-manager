"""The Splash engine: a chat slot served by `splash serve`."""

from __future__ import annotations

import pathlib
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from platform_harness import as_darwin, as_linux  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "web"))
import backends  # noqa: E402
import config_fields  # noqa: E402
import health  # noqa: E402

BASE = {"LLAMA_SERVER_BIN": "/bin/llama-server", "SPLASH_BIN": "/opt/splash",
        "LLM_B_ENGINE": "splash", "LLM_B_MODEL_PATH": "unsloth/Qwen3.8-27B-GGUF:Q8_0",
        "LLM_B_MODEL_NAME": "qwen3.8-27b", "LLM_B_CTX_SIZE": "262144"}


def build(**env):
    with as_darwin():
        return backends.build_command("llm-b", dict(BASE, **env))


def flag(argv, name):
    return argv[argv.index(name) + 1] if name in argv else None


class SplashEngineTests(unittest.TestCase):
    def test_the_slot_becomes_a_splash_command(self):
        argv = build()
        self.assertEqual(argv[:2], ["/opt/splash", "serve"])
        self.assertEqual(flag(argv, "--model"), "unsloth/Qwen3.8-27B-GGUF:Q8_0")
        self.assertEqual(flag(argv, "--port"), "8020")
        self.assertEqual(flag(argv, "--host"), "127.0.0.1")
        self.assertEqual(flag(argv, "--served-model-name"), "qwen3.8-27b")
        self.assertEqual(flag(argv, "--max-context"), "256K")
        self.assertIn("--no-webui", argv)
        self.assertNotIn("--language-only", argv)

    def test_vision_is_on_unless_turned_off(self):
        self.assertIn("--language-only", build(LLM_B_SPLASH_VISION="off"))
        self.assertNotIn("--language-only", build(LLM_B_SPLASH_VISION="on"))

    def test_the_kv_cache_is_lossless_unless_asked(self):
        self.assertEqual(flag(build(), "--kv-format"), "bf16")
        self.assertEqual(flag(build(LLM_B_SPLASH_KV_FORMAT="int8"), "--kv-format"), "int8")
        self.assertEqual(flag(build(LLM_B_SPLASH_KV_FORMAT="q4"), "--kv-format"), "bf16")

    def test_the_slot_reasoning_level_is_the_default(self):
        self.assertEqual(flag(build(LLM_B_REASONING_EFFORT="none"), "--default-reasoning-effort"), "none")
        self.assertEqual(flag(build(LLM_B_REASONING_EFFORT="medium"), "--default-reasoning-effort"), "medium")
        self.assertIsNone(flag(build(LLM_B_REASONING_EFFORT=""), "--default-reasoning-effort"))

    def test_context_is_capped_at_splash_maximum(self):
        self.assertEqual(flag(build(LLM_B_CTX_SIZE="1048576"), "--max-context"), "256K")
        self.assertEqual(flag(build(LLM_B_CTX_SIZE="131072"), "--max-context"), "128K")

    def test_memory_is_capped_below_splash_auto(self):
        with mock.patch.object(backends.splash, "physical_gib", return_value=96.0):
            self.assertEqual(flag(build(), "--max-memory"), "48G")
            self.assertEqual(flag(build(LLM_B_RAM_BUDGET_GB="50"), "--max-memory"), "48G")
            argv = build(LLM_B_MEMORY_MODE="ssd-offload", LLM_B_RAM_BUDGET_GB="50")
            self.assertEqual(flag(argv, "--max-memory"), "50G")
            argv = build(LLM_B_MEMORY_MODE="ssd-offload", LLM_B_RAM_BUDGET_GB="50",
                         LLM_B_SPLASH_MAX_MEMORY_GB="44")
            self.assertEqual(flag(argv, "--max-memory"), "44G")
            self.assertNotIn("--max-memory", build(LLM_B_SPLASH_MAX_MEMORY_GB="auto"))

    def test_the_ssd_tier_is_off_unless_sized(self):
        self.assertNotIn("--max-cache-disk", build())
        self.assertEqual(flag(build(LLM_B_SPLASH_MAX_CACHE_DISK_GB="40"), "--max-cache-disk"), "40G")
        for off in ("0", "off", ""):
            self.assertNotIn("--max-cache-disk", build(LLM_B_SPLASH_MAX_CACHE_DISK_GB=off))

    def test_a_local_path_or_gguf_is_refused_with_the_reason(self):
        for model in ("/models/q.gguf", "q.gguf", ""):
            with self.assertRaises(SystemExit) as caught:
                build(LLM_B_MODEL_PATH=model)
            self.assertIn("Hugging Face model reference", str(caught.exception))

    def test_only_loopback_and_only_macos(self):
        with self.assertRaises(SystemExit):
            build(CHAT2_BACKEND_HOST="0.0.0.0")
        with as_linux(), self.assertRaises(SystemExit):
            backends.build_command("llm-b", dict(BASE))

    def test_auto_never_picks_splash(self):
        env = dict(BASE, LLM_B_ENGINE="auto")
        with as_darwin():
            self.assertEqual(backends.SLOTS["llm-b"].engine(env), "llamacpp")

    def test_health_waits_for_the_model(self):
        spec = health.ENGINE_PROBES[("llm-b", "splash")]
        self.assertEqual((spec["path"], spec["expect_field"]), ("/ready", ("status", "ready")))

    def test_the_proxy_learns_the_engine_and_the_served_name(self):
        from backends.proxies import PROXIES, resolve
        env = dict(BASE, LLM_A_ENGINE="splash", LLM_A_MODEL_PATH="unsloth/Qwen3.8-27B-GGUF:Q8_0",
                   LLM_A_MODEL_NAME="qwen3.8-27b")
        with as_darwin():
            out = resolve(PROXIES["llm-a-proxy"], env)
        self.assertEqual((out["CHAT_BACKEND_ENGINE"], out["CHAT_BACKEND_MODEL"]),
                         ("splash", "qwen3.8-27b"))

    def test_the_ui_offers_it(self):
        field = next(f for f in config_fields.CONFIG_FIELDS if f["key"] == "LLM_A_ENGINE")
        self.assertIn("splash", field["options"])


if __name__ == "__main__":
    unittest.main()
