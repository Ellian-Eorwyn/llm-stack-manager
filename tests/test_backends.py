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


class PrefixChainTests(unittest.TestCase):
    """A slot may answer to more than one prefix.

    The primary chat slot resolves `CHAT_PRIMARY_X` and then `CHAT_X` for forty
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

    The chat slots bind `CHAT_BACKEND_PORT`, not `CHAT_PRIMARY_PORT`, and a
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
