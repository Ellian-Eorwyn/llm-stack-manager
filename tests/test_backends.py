"""The slot registry and the command builders.

The launchers these replace could only be checked by starting a service and
reading the journal. Here the same question -- what would this slot run? -- is a
function call, so the fallback chains, the defaults and the engine choice can
each be asserted directly.

`tests/test_launchers.py` remains the end-to-end check: it runs the real shell
and compares against a command line captured from the original scripts.
"""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import platform_harness  # noqa: E402
from platform_harness import as_darwin, as_linux  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "web"))
import backends  # noqa: E402
from backends import llamacpp  # noqa: E402
from backends.slots import SLOTS  # noqa: E402
from backends.spec import Flag, Slot, Toggle, lookup  # noqa: E402


def flags(argv: list[str]) -> dict:
    out, i = {}, 0
    while i < len(argv):
        if argv[i].startswith("--"):
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                out[argv[i]] = argv[i + 1]; i += 2
            else:
                out[argv[i]] = True; i += 1
        else:
            i += 1
    return out


BASE = {
    "LLAMA_SERVER_BIN": "/bin/llama-server",
    "LISTEN_HOST": "0.0.0.0",
    "EMBEDDING_MODEL_PATH": "/models/embed.gguf",
    "RERANKER_MODEL_PATH": "/models/rerank.gguf",
    "OCR_MODEL_PATH": "/models/ocr.gguf",
    "EMBED_CTX_SIZE": "8192", "EMBED_BATCH_SIZE": "2048",
    "RERANK_CTX_SIZE": "8192", "RERANK_BATCH_SIZE": "4096",
}


class FallbackChainTests(unittest.TestCase):
    """The chains are load-bearing configuration, not tidiable accidents."""

    def test_a_slot_falls_back_to_the_chat_micro_batch(self):
        # `${EMBED_UBATCH_SIZE:-${CHAT_UBATCH_SIZE:-512}}` shipped for years:
        # "the embedding slot inherits the chat slot's ubatch" is a fact about
        # the configuration surface, and normalising it away changes behaviour
        # on every host that relies on it.
        with as_linux():
            f = flags(backends.build_command("embed", dict(BASE, CHAT_UBATCH_SIZE="1024")))
            self.assertEqual(f["--ubatch-size"], "1024")
            f = flags(backends.build_command("embed", dict(BASE, CHAT_UBATCH_SIZE="1024",
                                                           EMBED_UBATCH_SIZE="256")))
            self.assertEqual(f["--ubatch-size"], "256")

    def test_the_slots_own_key_wins_over_the_shared_one(self):
        with as_linux():
            f = flags(backends.build_command("embed", dict(BASE, CHAT_N_GPU_LAYERS="20",
                                                           EMBED_N_GPU_LAYERS="7")))
        self.assertEqual(f["--n-gpu-layers"], "7")

    def test_an_empty_value_falls_through_rather_than_being_passed(self):
        # Passing "" is not the same as not passing: llama.cpp reads
        # `--tensor-split ""` as an explicit empty split and refuses it.
        with as_linux():
            f = flags(backends.build_command("embed", dict(BASE, EMBED_N_GPU_LAYERS="")))
        self.assertEqual(f["--n-gpu-layers"], "-1")

    def test_auxiliary_bind_hosts_override_the_shared_host_on_both_platforms(self):
        # Tailscale Serve owns the tailnet address on these ports, while other
        # services may still need the shared wildcard listener.
        for platform in (as_linux, as_darwin):
            with platform():
                for slot in ("embed", "task"):
                    with self.subTest(platform=platform.__name__, slot=slot):
                        key = f"{slot.upper()}_HOST"
                        self.assertEqual(flags(backends.build_command(slot, BASE))["--host"],
                                         "0.0.0.0")
                        self.assertEqual(flags(backends.build_command(
                            slot, dict(BASE, **{key: "127.0.0.1"})))["--host"],
                            "127.0.0.1")


class SlotIdentityTests(unittest.TestCase):

    def test_the_reranker_answers_to_rank_not_rerank(self):
        # The router's INI section name overwrites the child's --alias, and
        # every caller in the stack sends "rank". See docs/model-router.md.
        with as_linux():
            self.assertEqual(flags(backends.build_command("rerank", BASE))["--alias"], "rank")

    def test_each_slot_declares_a_distinct_default_port(self):
        ports = {name: slot.port_default for name, slot in SLOTS.items()}
        self.assertEqual(len(set(ports.values())), len(ports), ports)

    def test_the_embedding_slots_ask_for_embeddings_and_the_reranker_does_not(self):
        with as_linux():
            self.assertIn("--embedding", backends.build_command("embed", BASE))
            self.assertIn("--reranking", backends.build_command("rerank", BASE))
            self.assertNotIn("--reranking", backends.build_command("embed", BASE))

    def test_ocr_is_given_a_deterministic_sampler(self):
        # A layout model wants the same answer twice, not a sampled one.
        with as_linux():
            f = flags(backends.build_command("ocr", BASE))
        self.assertEqual(f["--temp"], "0.1")
        self.assertEqual(f["--top-k"], "1")

    def test_ocr_takes_no_reasoning_format(self):
        with as_linux():
            self.assertNotIn("--reasoning-format", backends.build_command("ocr", BASE))


class PlacementTests(unittest.TestCase):

    def test_the_launchers_vetted_flags_win(self):
        # resolve_split_opts refuses modes that fail after exec, and vets
        # `tensor` against the model's architecture. That verdict must not be
        # recomputed here from the raw setting it already rejected.
        env = dict(BASE, EMBED_SPLIT_MODE="row",
                   LLM_BACKEND_PLACEMENT_JSON=json.dumps(["--split-mode", "layer"]))
        with as_linux():
            argv = backends.build_command("embed", env)
        self.assertIn("layer", argv)
        self.assertNotIn("row", argv)

    def test_metal_collapses_placement_to_a_single_device(self):
        with as_darwin():
            argv = backends.build_command("embed", dict(BASE, EMBED_SPLIT_MODE="layer",
                                                        EMBED_TENSOR_SPLIT="1,1"))
        self.assertEqual(flags(argv)["--split-mode"], "none")
        self.assertNotIn("--tensor-split", argv)

    def test_auto_expands_to_one_share_per_visible_device(self):
        self.assertEqual(llamacpp.even_tensor_split("auto", "0"), "1")
        self.assertEqual(llamacpp.even_tensor_split("auto", "0,1"), "1,1")
        self.assertEqual(llamacpp.even_tensor_split("auto", "0, 1, 2"), "1,1,1")
        # A slot pinned to one card of two wants "1", not "1,1".
        self.assertEqual(llamacpp.even_tensor_split("auto", ""), "1")
        # An explicit ratio is left alone; empty stays empty and is omitted.
        self.assertEqual(llamacpp.even_tensor_split("3,2", "0,1"), "3,2")
        self.assertEqual(llamacpp.even_tensor_split("", "0,1"), "")


class CustomArgumentTests(unittest.TestCase):

    def test_operator_arguments_are_shell_split_and_appended(self):
        with as_linux():
            argv = backends.build_command(
                "ocr", dict(BASE, OCR_CUSTOM_ARGS_JSON=json.dumps(["--foo bar", "--baz"])))
        self.assertEqual(argv[-3:], ["--foo", "bar", "--baz"])

    def test_a_malformed_custom_args_field_does_not_stop_the_backend(self):
        # It is a text box in a web form. A backend that will not start because
        # someone left a stray bracket in it is a worse outcome than ignoring it.
        for broken in ("{not json", "[1, 2, 3]", ""):
            with self.subTest(broken), as_linux():
                argv = backends.build_command("ocr", dict(BASE, OCR_CUSTOM_ARGS_JSON=broken))
                self.assertIn("--model", argv)

    def test_passthrough_arguments_come_last(self):
        with as_linux():
            argv = backends.build_command("embed", BASE, ["--verbose"])
        self.assertEqual(argv[-1], "--verbose")


class EngineSelectionTests(unittest.TestCase):

    def test_a_slot_defaults_to_llamacpp(self):
        with as_linux():
            argv = backends.build_command("embed", BASE)
        self.assertTrue(argv[0].endswith("llama-server"))

    def test_the_mlx_engine_runs_its_own_server(self):
        with as_darwin():
            argv = backends.build_command(
                "embed", dict(BASE, EMBED_ENGINE="mlx", STACK_DIR="/stack",
                              MLX_RUNTIME_VENV="/venv"))
        self.assertEqual(argv[0], "/venv/bin/python")
        self.assertEqual(argv[1], "/stack/scripts/mlx-embedding-server.py")
        self.assertEqual(flags(argv)["--port"], "8005")

    def test_an_unknown_engine_is_refused_rather_than_defaulted(self):
        # Falling back to llama.cpp would serve the slot from the wrong engine
        # and look like it worked.
        with as_linux(), self.assertRaises(SystemExit):
            backends.build_command("embed", dict(BASE, EMBED_ENGINE="banana"))

    def test_asking_mlx_for_a_slot_it_cannot_serve_says_so(self):
        with as_darwin(), self.assertRaises(SystemExit):
            backends.build_command("rerank", dict(BASE, RERANK_ENGINE="mlx"))


class MtplxEngineTests(unittest.TestCase):
    """A chat slot served by MTPLX reads the settings it already had."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.pack = pathlib.Path(tmp.name) / "Qwen3.8-27B-MTPLX-Optimized-Speed"
        self.pack.mkdir()
        (self.pack / "mtplx_runtime.json").write_text("{}")
        self.env = dict(BASE, STACK_DIR="/stack", LLM_A_ENGINE="mtplx",
                        LLM_A_MODEL_PATH=str(self.pack), LLM_A_MODEL_NAME="qwen3.8-27b",
                        LLM_A_CTX_SIZE="262144", LLM_A_TEMP="0.7", LLM_A_TOP_P="0.9",
                        LLM_A_TOP_K="40", LLM_A_REASONING_EFFORT="medium",
                        LLM_A_CACHE_RAM="8192")

    def build(self, slot="llm-a", **overrides):
        with as_darwin():
            return backends.build_command(slot, dict(self.env, **overrides))

    def test_the_slot_settings_become_mtplx_flags(self):
        argv = self.build()
        self.assertEqual(argv[:5], ["/usr/bin/env", "PYTHONUNBUFFERED=1",
                                    "MTPLX_SESSION_BANK_MAX_BYTES=8192M",
                                    "MTPLX_SESSION_BANK_IDLE_TTL_S=300",
                                    "/stack/deps/mtplx-venv/bin/mtplx"])
        self.assertEqual(argv[5], "serve")
        got = flags(argv)
        self.assertEqual(got["--model"], str(self.pack))
        self.assertEqual(got["--model-id"], "qwen3.8-27b")
        self.assertEqual(got["--host"], "127.0.0.1")
        self.assertEqual(got["--port"], "8010")
        self.assertEqual(got["--context-window"], "262144")
        self.assertEqual(got["--default-temperature"], "0.7")
        self.assertEqual(got["--default-top-p"], "0.9")
        self.assertEqual(got["--default-top-k"], "40")
        self.assertEqual(got["--reasoning-effort"], "medium")
        self.assertEqual(got["--preserve-thinking"], "on")

    def test_the_reply_text_carries_no_stats_footer(self):
        # MTPLX appends a tokens-per-second line to every reply without it.
        self.assertIn("--no-stats-footer", self.build())

    def test_the_proxy_is_not_asked_for_a_key(self):
        self.assertIn("--no-auth", self.build())

    def test_unset_sampling_matches_what_llama_server_would_get(self):
        env = {k: v for k, v in self.env.items()
               if k not in ("LLM_A_TEMP", "LLM_A_TOP_P", "LLM_A_TOP_K")}
        with as_darwin():
            got = flags(backends.build_command("llm-a", env))
        self.assertEqual((got["--default-temperature"], got["--default-top-p"],
                          got["--default-top-k"]), ("1.0", "0.95", "20"))

    def test_the_legacy_chat_keys_still_configure_it(self):
        env = {k: v for k, v in self.env.items() if k != "LLM_A_CTX_SIZE"}
        with as_darwin():
            got = flags(backends.build_command("llm-a", dict(env, CHAT_CTX_SIZE="65536")))
        self.assertEqual(got["--context-window"], "65536")

    def test_the_second_chat_slot_binds_its_own_port(self):
        got = flags(self.build("llm-b", LLM_B_ENGINE="mtplx", LLM_B_MODEL_PATH=str(self.pack)))
        self.assertEqual(got["--port"], "8020")

    def test_a_gguf_left_behind_by_the_engine_switch_is_refused(self):
        with self.assertRaises(SystemExit) as caught:
            self.build(LLM_A_MODEL_PATH="/models/Qwen3.8-27B-Q6_K.gguf")
        self.assertIn("not an MTPLX pack", str(caught.exception))

    def test_a_wider_bind_is_refused_because_mtplx_would_demand_a_key(self):
        with self.assertRaises(SystemExit):
            self.build(CHAT_BACKEND_HOST="0.0.0.0")

    def test_llama_server_custom_args_are_not_passed_to_it(self):
        argv = self.build(LLM_A_CUSTOM_ARGS_JSON='["--no-warmup"]',
                          LLM_A_MTPLX_ARGS_JSON='["--profile sustained"]')
        self.assertNotIn("--no-warmup", argv)
        self.assertEqual(argv[-2:], ["--profile", "sustained"])

    def test_an_idle_conversation_is_released_after_five_minutes(self):
        # An hour, MTPLX's default, stranded ~8 GB per compacted agent session.
        self.assertIn("MTPLX_SESSION_BANK_IDLE_TTL_S=300", self.build())
        self.assertIn("MTPLX_SESSION_BANK_IDLE_TTL_S=900",
                      self.build(LLM_A_MTPLX_SESSION_TTL="900"))
        self.assertIn("MTPLX_SESSION_BANK_IDLE_TTL_S=0",
                      self.build(LLM_A_MTPLX_SESSION_TTL="0"))

    def test_a_zero_cache_ram_leaves_the_cache_to_mtplx(self):
        argv = self.build(LLM_A_CACHE_RAM="0")
        self.assertFalse(any(a.startswith("MTPLX_SESSION_BANK_MAX_BYTES") for a in argv))

    def test_a_linux_host_is_told_to_use_llamacpp(self):
        # A config copied off a Mac must not restart-loop on a missing venv.
        with as_linux(), self.assertRaises(SystemExit) as caught:
            backends.build_command("llm-a", self.env)
        self.assertIn("LLM_A_ENGINE=llamacpp", str(caught.exception))

    def test_linux_still_builds_llama_server_for_the_same_slot(self):
        env = {k: v for k, v in self.env.items() if k != "LLM_A_ENGINE"}
        with as_linux():
            argv = backends.build_command("llm-a", dict(env, LLM_A_MODEL_PATH="/m/a.gguf"))
        self.assertTrue(argv[0].endswith("llama-server"))

    def test_auto_serves_a_pack_on_mtplx_and_a_gguf_on_llamacpp(self):
        # Switching engines is choosing the model, as switching GGUFs always was.
        env = {k: v for k, v in self.env.items() if k != "LLM_A_ENGINE"}
        with as_darwin():
            pack = backends.build_command("llm-a", env)
            gguf = backends.build_command("llm-a", dict(env, LLM_A_MODEL_PATH="/m/a.gguf"))
        self.assertEqual(pack[pack.index("serve") - 1], "/stack/deps/mtplx-venv/bin/mtplx")
        self.assertTrue(gguf[0].endswith("llama-server"))

    def test_an_explicit_engine_still_overrides_auto(self):
        self.assertEqual(SLOTS["llm-a"].engine(dict(self.env, LLM_A_ENGINE="llamacpp")), "llamacpp")

    def test_the_gguf_kv_cache_types_do_not_quantize_mtplx(self):
        # q8_0 costs llama.cpp little; MTPLX 2.12 decoded a third as fast with
        # a quantized cache at 100k context. Carrying it over would be silent.
        argv = self.build(LLM_A_CACHE_TYPE_K="q8_0", LLM_A_CACHE_TYPE_V="q8_0")
        self.assertNotIn("--paged-kv-quantization", argv)

    def test_mtplx_kv_quantization_is_its_own_setting(self):
        for mode in ("q8", "q4"):
            with self.subTest(mode=mode):
                got = flags(self.build(LLM_A_MTPLX_KV_QUANT=mode))
                self.assertEqual(got["--paged-kv-quantization"], mode)
        self.assertNotIn("--paged-kv-quantization", self.build(LLM_A_MTPLX_KV_QUANT="off"))

    def test_it_serves_only_the_chat_slots(self):
        with as_darwin(), self.assertRaises(SystemExit):
            backends.build_command("task", dict(self.env, TASK_ENGINE="mtplx",
                                                TASK_MODEL_PATH=str(self.pack)))


class PrefixChainTests(unittest.TestCase):
    """A slot may answer to more than one prefix.

    The primary chat slot resolves `LLM_A_X` and then `CHAT_X` for forty
    keys. Writing forty two-entry chains out would be honest but unreadable, so
    the slot carries the legacy prefix instead and every relative key is tried
    under each in turn.

    Five keys need more than that -- the ones that were once spelled
    `CHAT_DENSE_*` -- and they get an explicit chain rather than a third prefix
    that would silently start honouring `CHAT_DENSE_TEMP`, a key nothing in the
    tree declares.
    """

    SLOT = Slot(name="demo", prefix="DEMO_ONE", legacy_prefixes=("DEMO",),
                model_keys=("!DEMO_MODEL_PATH",), alias_default="demo",
                port_default="9000")

    def test_the_slots_own_prefix_wins(self):
        self.assertEqual(
            lookup({"DEMO_ONE_TEMP": "0.5", "DEMO_TEMP": "1.0"}, ("TEMP",), self.SLOT.prefixes),
            "0.5")

    def test_the_legacy_prefix_is_read_behind_it(self):
        self.assertEqual(lookup({"DEMO_TEMP": "1.0"}, ("TEMP",), self.SLOT.prefixes), "1.0")

    def test_an_absolute_key_is_not_prefixed_at_all(self):
        self.assertEqual(lookup({"LISTEN_HOST": "0.0.0.0"}, ("!LISTEN_HOST",),
                                self.SLOT.prefixes), "0.0.0.0")

    def test_nothing_set_is_none_rather_than_empty(self):
        # None and "" are different answers: "" is a value someone chose.
        self.assertIsNone(lookup({}, ("TEMP",), self.SLOT.prefixes))

    def test_a_single_prefix_slot_is_unchanged(self):
        self.assertEqual(SLOTS["embed"].prefixes, ("EMBED",))


class ClearedMeansClearedTests(unittest.TestCase):
    """`${X:-d}` and `${X-d}` are two different settings.

    The launchers use the first for required values, so an empty one lands on a
    working default, and the second for the optional paths and knobs, so
    clearing the new key means cleared. Reading the second as the first is why
    `--fit-ctx` kept being passed alongside `--fit off` after it had been
    cleared in the UI -- and a prefix chain would reintroduce it under two key
    names instead of one.
    """

    PREFIXES = ("DEMO_ONE", "DEMO")

    def test_an_emptied_key_stops_the_search_when_empty_is_set(self):
        self.assertEqual(
            lookup({"DEMO_ONE_FIT_CTX": "", "DEMO_FIT_CTX": "8192"},
                   ("FIT_CTX",), self.PREFIXES, empty_is_set=True),
            "")

    def test_an_emptied_key_falls_through_when_it_is_not(self):
        self.assertEqual(
            lookup({"DEMO_ONE_FIT_CTX": "", "DEMO_FIT_CTX": "8192"},
                   ("FIT_CTX",), self.PREFIXES),
            "8192")

    def test_an_absent_key_falls_through_either_way(self):
        for empty_is_set in (True, False):
            with self.subTest(empty_is_set=empty_is_set):
                self.assertEqual(
                    lookup({"DEMO_FIT_CTX": "8192"}, ("FIT_CTX",), self.PREFIXES,
                           empty_is_set=empty_is_set),
                    "8192")

    def test_a_cleared_flag_is_omitted_rather_than_passed_empty(self):
        flag = Flag("--fit-ctx", ("FIT_CTX",), empty_is_set=True)
        self.assertEqual(flag.resolve({"DEMO_ONE_FIT_CTX": ""}, self.PREFIXES), [])

    def test_a_toggle_reads_the_legacy_prefix(self):
        toggle = Toggle("--metrics", "METRICS", when="on", default="off")
        self.assertEqual(toggle.resolve({"DEMO_METRICS": "on"}, self.PREFIXES), ["--metrics"])
        self.assertEqual(toggle.resolve({"DEMO_ONE_METRICS": "off", "DEMO_METRICS": "on"},
                                        self.PREFIXES), [])


class SlotKeyOverrideTests(unittest.TestCase):
    """Ports, aliases and mmproj paths do not all follow the prefix.

    The chat slots bind `CHAT_BACKEND_PORT`, not `LLM_A_PORT`, and a
    registry that derived the key from the prefix would move both backends to a
    port nothing else in the stack talks to.
    """

    def _slot(self, **kwargs):
        return Slot(name="demo", prefix="DEMO", model_keys=("!DEMO_MODEL_PATH",),
                    alias_default="demo", port_default="9000", **kwargs)

    def test_a_slot_may_name_the_port_key_itself(self):
        slot = self._slot(port_keys=("!OTHER_BACKEND_PORT",))
        with as_linux():
            argv = llamacpp.build(slot, dict(BASE, OTHER_BACKEND_PORT="8010"))
        self.assertEqual(flags(argv)["--port"], "8010")

    def test_the_alias_resolves_through_a_chain_before_its_default(self):
        slot = self._slot(alias_keys=("MODEL_NAME", "!LEGACY_ALIAS"))
        with as_linux():
            self.assertEqual(flags(llamacpp.build(slot, dict(BASE)))["--alias"], "demo")
            self.assertEqual(
                flags(llamacpp.build(slot, dict(BASE, LEGACY_ALIAS="old")))["--alias"], "old")
            self.assertEqual(
                flags(llamacpp.build(slot, dict(BASE, LEGACY_ALIAS="old",
                                                DEMO_MODEL_NAME="new")))["--alias"], "new")

    def test_a_key_chain_overrides_the_shared_one_for_this_slot_only(self):
        # The five keys once spelled CHAT_DENSE_* are the reason this exists,
        # and they are why the chain is written absolute: a relative key is
        # tried under every prefix before the next entry is reached, so mixing
        # the two forms would put the legacy prefix ahead of the middle key.
        slot = self._slot(legacy_prefixes=("SHARED",),
                          key_chains={"CTX_SIZE": ("!DEMO_CTX_SIZE", "!MIDDLE_CTX_SIZE",
                                                   "!SHARED_CTX_SIZE")})
        with as_linux():
            f = flags(llamacpp.build(slot, dict(BASE, MIDDLE_CTX_SIZE="4096",
                                                SHARED_CTX_SIZE="8192")))
        self.assertEqual(f["--ctx-size"], "4096")
        # And only for this slot: the shared COMMON_FLAGS entry is untouched,
        # so no other slot starts reading the key this one was given.
        with as_linux():
            f = flags(llamacpp.build(SLOTS["embed"], dict(BASE, MIDDLE_CTX_SIZE="4096")))
        self.assertNotEqual(f.get("--ctx-size"), "4096")

    def test_a_slot_default_still_applies_under_an_overridden_chain(self):
        slot = self._slot(defaults={"CTX_SIZE": "2048"},
                          key_chains={"CTX_SIZE": ("!ONLY_CTX_SIZE",)})
        with as_linux():
            self.assertEqual(flags(llamacpp.build(slot, dict(BASE)))["--ctx-size"], "2048")


if __name__ == "__main__":
    unittest.main()


class LoraAdapterTests(unittest.TestCase):
    """`--lora-scaled`, which is how a fine-tune reaches a running backend.

    The adapter is applied on top of the base model rather than merged into it,
    so what these assert is the loading half: which files the launcher hands to
    llama-server, and at what starting scale. Which of them is actually applied
    is changed later over HTTP and is not a launch flag at all.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.stack = pathlib.Path(self.tmp.name)
        (self.stack / "models" / "loras").mkdir(parents=True)
        for name in ("voice.gguf", "other.gguf"):
            (self.stack / "models" / "loras" / name).write_bytes(b"GGUF")
        self.addCleanup(self.tmp.cleanup)

    def build(self, slot="llm-a", **env):
        said = ["-"]  # non-empty: an empty list is falsy and used to be dropped
        argv = backends.build_command(
            slot, dict(BASE, STACK_DIR=str(self.stack),
                       LLM_A_MODEL_PATH="/models/a.gguf",
                       LLM_B_MODEL_PATH="/models/b.gguf",
                       TASK_MODEL_PATH="/models/t.gguf", **env),
            said=said)
        return argv, said[1:]

    def lora_args(self, argv):
        return [a for i, a in enumerate(argv)
                if a.startswith("--lora") or (i and argv[i - 1] == "--lora-scaled")]

    def test_no_adapters_emits_no_flags(self):
        """A slot nobody configured must run exactly the command it always did."""
        with as_linux():
            argv, _ = self.build()
        self.assertEqual(self.lora_args(argv), [])

    def test_a_bare_name_resolves_under_models_loras(self):
        with as_linux():
            argv, _ = self.build(LLM_A_LORA_PATHS="voice.gguf")
        self.assertEqual(self.lora_args(argv), [
            "--lora-scaled", f"{self.stack}/models/loras/voice.gguf:0",
            "--lora-init-without-apply",
        ])

    def test_preload_unapplied_boots_at_zero_rather_than_trusting_the_flag(self):
        """`--lora-init-without-apply` is not sufficient on its own.

        The server README says adapters loaded with it "start at scale 0.0".
        In llama-server they do not: `common.cpp` only skips the one-time
        `common_set_adapter_lora` at startup, and `server-context.cpp` then does
        `slot.lora = params_base.lora_adapters` for every task and re-applies it
        per batch, restoring each adapter's configured scale on the first
        request. Measured on build b10434. `:0` is what actually holds.
        """
        with as_linux():
            argv, _ = self.build(LLM_A_LORA_PATHS="voice.gguf",
                                 LLM_A_LORA_SCALES="0.8")
        self.assertIn(f"{self.stack}/models/loras/voice.gguf:0", argv)
        self.assertNotIn(f"{self.stack}/models/loras/voice.gguf:0.8", argv)

    def test_scales_pair_positionally_and_missing_ones_default_to_one(self):
        """With preloading off, the configured scales are what the slot boots at."""
        with as_linux():
            argv, _ = self.build(LLM_A_LORA_PATHS="voice.gguf,other.gguf",
                                 LLM_A_LORA_SCALES="0.8",
                                 LLM_A_LORA_INIT_WITHOUT_APPLY="off")
        self.assertEqual(self.lora_args(argv), [
            "--lora-scaled", f"{self.stack}/models/loras/voice.gguf:0.8",
            "--lora-scaled", f"{self.stack}/models/loras/other.gguf:1.0",
        ])

    def test_init_without_apply_can_be_turned_off(self):
        """Off means every adapter applies from startup, and they stack."""
        with as_linux():
            argv, _ = self.build(LLM_A_LORA_PATHS="voice.gguf",
                                 LLM_A_LORA_INIT_WITHOUT_APPLY="off")
        self.assertNotIn("--lora-init-without-apply", argv)
        self.assertIn("--lora-scaled", argv)

    def test_a_missing_adapter_is_dropped_with_a_reason(self):
        """llama-server exits on a missing adapter, which reads as a crash loop."""
        with as_linux():
            argv, said = self.build(LLM_A_LORA_PATHS="ghost.gguf")
        self.assertEqual(self.lora_args(argv), [])
        self.assertTrue(any("ghost.gguf" in m for m in said), said)

    def test_a_scale_that_is_not_a_number_falls_back_to_one(self):
        with as_linux():
            argv, said = self.build(LLM_A_LORA_PATHS="voice.gguf",
                                    LLM_A_LORA_SCALES="loud")
        # The bad scale still resolves to 1.0; preloading then boots it at 0.
        self.assertIn(f"{self.stack}/models/loras/voice.gguf:0", argv)
        self.assertTrue(any("loud" in m for m in said), said)

    def test_a_path_with_a_colon_is_refused_rather_than_mangled(self):
        """`--lora-scaled` splits on the last colon; a path holding one lies."""
        odd = self.stack / "models" / "loras" / "a:b.gguf"
        odd.write_bytes(b"GGUF")
        with as_linux():
            argv, said = self.build(LLM_A_LORA_PATHS="a:b.gguf")
        self.assertEqual(self.lora_args(argv), [])
        self.assertTrue(any("colon" in m for m in said), said)

    def test_the_operators_own_lora_flag_wins(self):
        with as_linux():
            argv, _ = self.build(LLM_A_LORA_PATHS="voice.gguf",
                                 LLM_A_CUSTOM_ARGS_JSON='["--lora", "/x.gguf"]')
        self.assertNotIn("--lora-scaled", argv)
        self.assertEqual(argv[-2:], ["--lora", "/x.gguf"])

    def test_every_adapter_capable_slot_reads_its_own_prefix(self):
        for slot, key in (("llm-a", "LLM_A_LORA_PATHS"),
                          ("llm-b", "LLM_B_LORA_PATHS"),
                          ("task", "TASK_LORA_PATHS")):
            with self.subTest(slot=slot), as_linux():
                argv, _ = self.build(slot, **{key: "voice.gguf"})
            self.assertIn(f"{self.stack}/models/loras/voice.gguf:0", argv)

    def test_llm_a_inherits_the_legacy_chat_prefix(self):
        """`LLM_A_*` falls back through `CHAT_PRIMARY_*` to `CHAT_*`, as the
        rest of the slot's settings do."""
        with as_linux():
            argv, _ = self.build(CHAT_LORA_PATHS="voice.gguf")
        self.assertIn(f"{self.stack}/models/loras/voice.gguf:0", argv)


class SaidPropagationTests(unittest.TestCase):
    """The messages an option writes have to reach the caller's list.

    `said or []` swapped the caller's empty list for a fresh one, and every
    caller passes an empty list -- so nothing an option said had ever been
    printed by `scripts/lib/build-backend-command.py`.
    """

    def test_messages_reach_an_empty_list_the_caller_passed(self):
        with tempfile.TemporaryDirectory() as tmp:
            said = []
            with as_linux():
                backends.build_command(
                    "llm-a",
                    dict(BASE, STACK_DIR=tmp, LLM_A_MODEL_PATH="/models/a.gguf",
                         LLM_A_LORA_PATHS="ghost.gguf"),
                    said=said)
            self.assertTrue(any("ghost.gguf" in m for m in said), said)
