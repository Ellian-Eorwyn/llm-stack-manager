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


if __name__ == "__main__":
    unittest.main()
