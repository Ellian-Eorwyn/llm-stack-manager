"""The NInfer engine: a chat slot served by `ninfer-serve` on an RTX 3090."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from platform_harness import as_darwin, as_linux  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "web"))
import backends  # noqa: E402
import budget  # noqa: E402
import config_fields  # noqa: E402
import health  # noqa: E402

MODEL = "/models/qwen3_8_27b.ninfer"
BASE = {"STACK_DIR": "/stack", "LLM_A_ENGINE": "ninfer", "LLM_A_MODEL_PATH": MODEL,
        "LLM_A_MODEL_NAME": "chat-dense", "LLM_A_CTX_SIZE": "131072",
        "LLM_A_GPU_VISIBLE_DEVICES": "1"}


def build(**env):
    with as_linux():
        return backends.build_command("llm-a", dict(BASE, **env))


def flag(argv, name):
    return argv[argv.index(name) + 1] if name in argv else None


def _load_proxy():
    spec = importlib.util.spec_from_file_location("llm_chat_proxy_ninfer",
                                                  ROOT / "scripts" / "llm-chat-proxy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NinferEngineTests(unittest.TestCase):
    def test_the_slot_becomes_an_ninfer_command(self):
        argv = build()
        self.assertEqual(argv[:2], ["/stack/deps/ninfer-3090/build-sm86/apps/ninfer-serve", MODEL])
        self.assertEqual(flag(argv, "--model-id"), "chat-dense")
        self.assertEqual(flag(argv, "--port"), "8010")
        self.assertEqual(flag(argv, "--max-context"), "131072")
        self.assertEqual(flag(argv, "--kv-capacity"), "131072")
        self.assertEqual(flag(argv, "--max-concurrency"), "1")
        self.assertEqual(flag(argv, "--kv-dtype"), "int8")
        self.assertEqual((flag(argv, "--spec"), flag(argv, "--draft-tokens")), ("mtp", "3"))
        self.assertIn("--lm-head-draft", argv)
        self.assertEqual(flag(argv, "--vision-residency"), "overlay")
        self.assertIn("--preserve-thinking", argv)
        self.assertNotIn("--prefill-cublas", argv)

    def test_the_card_is_chosen_by_the_visible_list(self):
        self.assertEqual(flag(build(), "--device"), "0")
        self.assertEqual(flag(build(LLM_ABSOLUTE_GPU_INDICES="on"), "--device"), "1")

    def test_speculation_vision_and_kv_follow_the_slot(self):
        argv = build(LLM_A_NINFER_SPEC="dflash2", LLM_A_NINFER_KV_DTYPE="rk8v4",
                     LLM_A_NINFER_VISION="off", LLM_A_NINFER_PREFILL_CUBLAS="on")
        self.assertEqual((flag(argv, "--spec"), flag(argv, "--draft-tokens")), ("dflash2", "7"))
        self.assertEqual(flag(argv, "--kv-dtype"), "rk8v4")
        self.assertNotIn("--vision", argv)
        self.assertIn("--prefill-cublas", argv)
        off = build(LLM_A_NINFER_SPEC="off")
        self.assertNotIn("--spec", off)
        self.assertNotIn("--lm-head-draft", off)
        self.assertEqual(flag(build(LLM_A_NINFER_KV_DTYPE="fp8"), "--kv-dtype"), "int8")

    def test_the_slot_reasoning_level_is_the_default(self):
        self.assertEqual(flag(build(LLM_A_REASONING_EFFORT="low"), "--reasoning-effort"), "low")
        self.assertIsNone(flag(build(LLM_A_REASONING_EFFORT="none"), "--reasoning-effort"))
        self.assertIsNone(flag(build(), "--reasoning-effort"))

    def test_the_pool_can_be_sized_apart_from_the_context(self):
        self.assertEqual(flag(build(LLM_A_NINFER_KV_CAPACITY="auto"), "--kv-capacity"), "auto")
        self.assertEqual(flag(build(LLM_A_NINFER_KV_CAPACITY="200000"), "--kv-capacity"), "200000")

    def test_a_gguf_or_another_host_is_refused_with_the_reason(self):
        with self.assertRaises(SystemExit) as caught:
            build(LLM_A_MODEL_PATH="/models/q.gguf")
        self.assertIn(".ninfer", str(caught.exception))
        with as_darwin(), self.assertRaises(SystemExit):
            backends.build_command("llm-a", dict(BASE))

    def test_auto_picks_ninfer_for_its_artifact(self):
        slot = backends.SLOTS["llm-a"]
        self.assertEqual(slot.engine(dict(BASE, LLM_A_ENGINE="auto")), "ninfer")
        self.assertEqual(slot.engine(dict(BASE, LLM_A_ENGINE="auto",
                                          LLM_A_MODEL_PATH="/m/q.gguf")), "llamacpp")

    def test_health_waits_for_the_model(self):
        spec = health.ENGINE_PROBES[("llm-a", "ninfer")]
        self.assertEqual((spec["path"], spec["expect_field"]), ("/health", ("status", "ok")))

    def test_the_proxy_learns_the_engine_and_the_served_name(self):
        from backends.proxies import PROXIES, resolve
        with as_linux():
            out = resolve(PROXIES["llm-a-proxy"], dict(BASE))
        self.assertEqual((out["CHAT_BACKEND_ENGINE"], out["CHAT_BACKEND_MODEL"]),
                         ("ninfer", "chat-dense"))

    def test_budget_does_not_read_it_as_a_gguf(self):
        with tempfile.NamedTemporaryFile(suffix=".ninfer") as artifact:
            artifact.write(b"\0" * 1024)
            artifact.flush()
            result = budget.budget_for(dict(BASE, LLM_A_MODEL_PATH=artifact.name), "llm-a")
        self.assertIn("NInfer", result["error"])
        self.assertIsNone(result["verdict"])

    def test_the_ui_offers_it(self):
        field = next(f for f in config_fields.CONFIG_FIELDS if f["key"] == "LLM_A_ENGINE")
        self.assertIn("ninfer", field["options"])
        keys = {f["key"] for f in config_fields.CONFIG_FIELDS}
        self.assertTrue({"NINFER_BIN", "LLM_A_NINFER_SPEC", "LLM_B_NINFER_KV_DTYPE"} <= keys)


class NinferProxyTests(unittest.TestCase):
    def setUp(self):
        self.proxy = _load_proxy()
        self.proxy.BACKEND_ENGINE = "ninfer"
        self.proxy.BACKEND_MODEL = "chat-dense"

    def test_chat_keeps_what_ninfer_takes_and_drops_what_it_refuses(self):
        payload = {"model": "think", "messages": [], "top_k": 40, "repetition_penalty": 1.1,
                   "logit_bias": {"42": 5}, "logprobs": True, "top_logprobs": 3,
                   "tool_choice": "required", "temperature": 0.7}
        self.proxy._adapt_for_ninfer(payload, "chat", True)
        self.assertEqual(payload, {"model": "chat-dense", "messages": [], "top_k": 20,
                                   "tool_choice": "auto", "temperature": 0.7})

    def test_neutral_values_are_left_alone(self):
        payload = {"model": "think", "repetition_penalty": 1, "logit_bias": {"42": 0},
                   "logprobs": False, "top_k": 20}
        self.proxy._adapt_for_ninfer(payload, "chat", True)
        self.assertEqual(payload["repetition_penalty"], 1)
        self.assertEqual(payload["logit_bias"], {"42": 0})
        self.assertIs(payload["logprobs"], False)

    def test_thinking_off_never_carries_a_level(self):
        payload = {"reasoning_effort": "low",
                   "chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": "low"}}
        self.proxy._adapt_for_ninfer(payload, "chat", True)
        self.assertEqual(payload, {"reasoning_effort": "none",
                                   "chat_template_kwargs": {"enable_thinking": False}})

    def test_responses_get_only_the_fields_they_accept(self):
        payload = {"model": "think", "input": "hi", "reasoning_effort": "low", "top_k": 20,
                   "min_p": 0.0, "repeat_penalty": 1.0, "reasoning_format": "deepseek",
                   "chat_template_kwargs": {"enable_thinking": True}}
        self.proxy._adapt_for_ninfer(payload, "responses", True)
        self.assertEqual(payload, {"model": "chat-dense", "input": "hi",
                                   "reasoning": {"effort": "low"},
                                   "chat_template_kwargs": {"enable_thinking": True}})

    def test_responses_with_thinking_off_say_none(self):
        payload = {"input": "hi", "chat_template_kwargs": {"enable_thinking": False}}
        self.proxy._adapt_for_ninfer(payload, "responses", True)
        self.assertEqual(payload["reasoning"], {"effort": "none"})

    def test_a_models_lookup_only_renames(self):
        payload = {"model": "think", "top_k": 99}
        self.proxy._adapt_for_ninfer(payload, "models", False)
        self.assertEqual(payload, {"model": "chat-dense", "top_k": 99})


if __name__ == "__main__":
    unittest.main()
