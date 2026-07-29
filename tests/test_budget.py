from __future__ import annotations

import importlib.util
import pathlib
import struct
import tempfile
import unittest


def _load_budget_module():
    root = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "llm_stack_manager_budget", root / "web" / "budget.py")
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


budget = _load_budget_module()

# The metadata of the model this box actually runs, read off the real file.
# Everything downstream is checked against numbers measured on that backend, so
# these values are the fixture the rest of the suite depends on.
QWEN36_27B = {
    "general.architecture": "qwen35",
    "general.name": "Qwen3.6-27B",
    "general.file_type": 18,
    "qwen35.block_count": 65,
    "qwen35.nextn_predict_layers": 1,
    "qwen35.context_length": 262144,
    "qwen35.embedding_length": 5120,
    "qwen35.attention.head_count": 24,
    "qwen35.attention.head_count_kv": 4,
    "qwen35.attention.key_length": 256,
    "qwen35.attention.value_length": 256,
    "qwen35.full_attention_interval": 4,
    "qwen35.ssm.inner_size": 6144,
    "qwen35.ssm.state_size": 128,
    "qwen35.ssm.conv_kernel": 4,
    "qwen35.ssm.group_count": 16,
}

# Interleaved local/global attention, taken from the Gemma4-31B this box runs.
# Five sliding-window layers to every global one, and the two classes differ in
# both head count and head dimension: the window layers are wide (16 heads x
# 256) but only ever hold 1024 tokens, while the layers that actually grow with
# context are narrow (4 heads x 512). Pricing all 60 layers as wide global ones
# predicted 254,998 MiB of KV against roughly 11,050 MiB it holds.
_GEMMA_PATTERN = [(index % 6) != 5 for index in range(60)]
GEMMA4_31B = {
    "general.architecture": "gemma4",
    "general.name": "Gemma4-31B",
    "general.file_type": 15,
    "gemma4.block_count": 60,
    "gemma4.context_length": 262144,
    "gemma4.embedding_length": 5376,
    "gemma4.attention.head_count": 32,
    "gemma4.attention.head_count_kv": [16 if swa else 4 for swa in _GEMMA_PATTERN],
    "gemma4.attention.key_length": 512,
    "gemma4.attention.value_length": 512,
    "gemma4.attention.key_length_swa": 256,
    "gemma4.attention.value_length_swa": 256,
    "gemma4.attention.sliding_window": 1024,
    "gemma4.attention.sliding_window_pattern": _GEMMA_PATTERN,
}

# A conventional dense model: no SSM keys, no attention interval, so every
# layer holds KV and checkpoints carry no fixed cost.
DENSE_8B = {
    "general.architecture": "llama",
    "general.name": "Dense 8B",
    "general.file_type": 15,
    "llama.block_count": 32,
    "llama.context_length": 8192,
    "llama.embedding_length": 4096,
    "llama.attention.head_count": 32,
    "llama.attention.head_count_kv": 8,
    "llama.attention.key_length": 128,
    "llama.attention.value_length": 128,
}


# --------------------------------------------------------------------------
# a GGUF file, so the reader is tested against bytes rather than a mock
# --------------------------------------------------------------------------

_TYPE_STRING, _TYPE_ARRAY, _TYPE_UINT32, _TYPE_BOOL = 8, 9, 4, 7


def _encode_string(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _encode_value(value) -> bytes:
    if isinstance(value, str):
        return struct.pack("<I", _TYPE_STRING) + _encode_string(value)
    if isinstance(value, list):
        # Per-layer metadata arrives as bool or int arrays (Gemma's sliding
        # window pattern and per-layer KV head counts); the tokenizer's token
        # list arrives as a string array.
        if value and isinstance(value[0], bool):
            body = b"".join(struct.pack("<?", bool(item)) for item in value)
            item_type = _TYPE_BOOL
        elif value and isinstance(value[0], int):
            body = b"".join(struct.pack("<I", int(item)) for item in value)
            item_type = _TYPE_UINT32
        else:
            body = b"".join(_encode_string(item) for item in value)
            item_type = _TYPE_STRING
        return struct.pack("<I", _TYPE_ARRAY) + struct.pack("<I", item_type) \
            + struct.pack("<Q", len(value)) + body
    return struct.pack("<I", _TYPE_UINT32) + struct.pack("<I", int(value))


def write_gguf(path: pathlib.Path, metadata: dict, trailing_bytes: int = 4096) -> pathlib.Path:
    """Write a GGUF file with a real header and filler where tensors would be."""
    body = b"".join(_encode_string(key) + _encode_value(value) for key, value in metadata.items())
    header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(metadata))
    path.write_bytes(header + body + b"\0" * trailing_bytes)
    return path


class GGUFReaderTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = pathlib.Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_reads_metadata_and_file_size(self):
        path = write_gguf(self.tmp / "m.gguf", QWEN36_27B, trailing_bytes=1024)
        metadata = budget.read_gguf_metadata(path)
        self.assertEqual(metadata["general.architecture"], "qwen35")
        self.assertEqual(metadata["qwen35.block_count"], 65)
        self.assertEqual(metadata["__file_size__"], path.stat().st_size)

    def test_large_string_arrays_are_skipped_but_counted(self):
        """The tokenizer vocabulary is never materialised, only measured."""
        metadata = dict(QWEN36_27B)
        metadata["tokenizer.ggml.tokens"] = [f"tok{i}" for i in range(2000)]
        path = write_gguf(self.tmp / "vocab.gguf", metadata)

        parsed = budget.read_gguf_metadata(path)
        self.assertEqual(parsed["tokenizer.ggml.tokens"], {"__len__": 2000})
        # Skipping must not desynchronise the reader: keys after the array
        # still parse, which is the whole risk of seeking past a value.
        self.assertEqual(parsed["qwen35.ssm.group_count"], 16)
        self.assertEqual(budget.model_geometry(parsed)["vocab_size"], 2000)

    def test_rejects_non_gguf(self):
        path = self.tmp / "not-a-model.gguf"
        path.write_bytes(b"<html>404</html>" + b"\0" * 128)
        with self.assertRaises(budget.GGUFError):
            budget.read_gguf_metadata(path)

    def test_rejects_unsupported_version(self):
        path = self.tmp / "v9.gguf"
        path.write_bytes(b"GGUF" + struct.pack("<I", 9) + struct.pack("<Q", 0) + struct.pack("<Q", 0))
        with self.assertRaises(budget.GGUFError):
            budget.read_gguf_metadata(path)


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

class GeometryTest(unittest.TestCase):
    def test_hybrid_layer_split(self):
        """65 blocks, one of them an MTP head, one full-attention layer in four."""
        geometry = budget.model_geometry(QWEN36_27B)
        self.assertEqual(geometry["layers"], 64)
        self.assertEqual(geometry["full_attention_layers"], 16)
        self.assertEqual(geometry["recurrent_layers"], 48)
        self.assertTrue(geometry["is_hybrid"])

    def test_dense_model_has_no_recurrent_layers(self):
        geometry = budget.model_geometry(DENSE_8B)
        self.assertEqual(geometry["layers"], 32)
        self.assertEqual(geometry["full_attention_layers"], 32)
        self.assertEqual(geometry["recurrent_layers"], 0)
        self.assertFalse(geometry["is_hybrid"])

    def test_recurrent_layer_array_takes_precedence(self):
        """Architectures that publish a per-layer flag array are read directly."""
        metadata = {
            "general.architecture": "falconh",
            "falconh.block_count": 6,
            "falconh.attention.head_count": 8,
            "falconh.attention.head_count_kv": 2,
            "falconh.attention.key_length": 64,
            "falconh.attention.value_length": 64,
            "falconh.embedding_length": 512,
            "falconh.recurrent_layer_arr": [True, False, True, True, False, True],
        }
        geometry = budget.model_geometry(metadata)
        self.assertEqual(geometry["full_attention_layers"], 2)
        self.assertEqual(geometry["recurrent_layers"], 4)

    def test_head_dimensions_fall_back_to_embedding_split(self):
        metadata = {k: v for k, v in DENSE_8B.items()
                    if not k.endswith(("key_length", "value_length"))}
        geometry = budget.model_geometry(metadata)
        self.assertEqual(geometry["key_length"], 4096 // 32)
        self.assertEqual(geometry["value_length"], 4096 // 32)

    def test_swa_support_is_detected_from_sliding_window(self):
        self.assertFalse(budget.model_geometry(QWEN36_27B)["supports_swa"])
        self.assertTrue(budget.model_geometry(
            dict(DENSE_8B, **{"llama.attention.sliding_window": 4096}))["supports_swa"])

    def test_file_type_is_named(self):
        self.assertEqual(budget.model_geometry(QWEN36_27B)["file_type"], "Q6_K")


# --------------------------------------------------------------------------
# memory terms, checked against numbers measured on the live backend
# --------------------------------------------------------------------------

class MemoryTermsTest(unittest.TestCase):
    def setUp(self):
        self.qwen = budget.model_geometry(QWEN36_27B)
        self.dense = budget.model_geometry(DENSE_8B)

    def test_kv_bytes_per_token(self):
        # 16 full-attention layers x 4 KV heads x (256 + 256) x 1.0625 B/elem.
        self.assertEqual(budget.kv_bytes_per_token(self.qwen, "q8_0", "q8_0"), 34816)

    def test_kv_scales_with_cache_quantisation(self):
        q8 = budget.kv_bytes_per_token(self.qwen, "q8_0", "q8_0")
        f16 = budget.kv_bytes_per_token(self.qwen, "f16", "f16")
        self.assertAlmostEqual(f16 / q8, 2 / 1.0625, places=6)

    def test_mixed_k_and_v_quantisation(self):
        mixed = budget.kv_bytes_per_token(self.qwen, "q8_0", "f16")
        self.assertEqual(mixed, 16 * 4 * (256 * (34 / 32) + 256 * 2.0))

    def test_unknown_cache_type_falls_back_to_f16(self):
        self.assertEqual(budget.kv_bytes_per_token(self.qwen, "nonsense", "nonsense"),
                         budget.kv_bytes_per_token(self.qwen, "f16", "f16"))

    def test_recurrent_state_matches_measurement(self):
        """149.6 MiB was measured in the journal; the formula must land on it.

        This is the number the whole eviction story turns on: it is what every
        context checkpoint costs before storing a single token.
        """
        mib = budget.recurrent_state_bytes(self.qwen) / budget.MIB
        self.assertAlmostEqual(mib, 149.6, delta=0.1)

    def test_dense_model_has_no_recurrent_state(self):
        self.assertEqual(budget.recurrent_state_bytes(self.dense), 0.0)

    def test_checkpoint_terms_match_measurement(self):
        """Journal: 149.6 MiB fixed plus roughly 0.002 MiB per token."""
        fixed, per_token = budget.checkpoint_bytes(self.qwen, 262144, "q8_0", "q8_0")
        self.assertAlmostEqual(fixed / budget.MIB, 149.6, delta=0.1)
        self.assertAlmostEqual(per_token / budget.MIB, 0.002, delta=0.0005)

    def test_sliding_window_checkpoints_hold_the_window_not_the_context(self):
        """PARTIAL_ONLY saves kv_swa and skips the base cache, so a checkpoint
        is the window however long the context is."""
        gemma = budget.model_geometry(GEMMA4_31B)
        short, _ = budget.checkpoint_bytes(gemma, 8192, "q8_0", "q8_0")
        long, per_token = budget.checkpoint_bytes(gemma, 255998, "q8_0", "q8_0")
        self.assertEqual(short, long)
        self.assertEqual(per_token, 0.0)
        self.assertAlmostEqual(long / budget.MIB, 425.0, delta=1.0)

    def test_a_plain_attention_model_checkpoints_nothing(self):
        """A checkpoint saves only state that cannot be rebuilt by reprocessing
        the prompt. A full-attention model has none, and llama.cpp does not
        create checkpoints for one — see the `n_swa > 0` guard on
        `do_checkpoint` in server-context.cpp."""
        fixed, per_token = budget.checkpoint_bytes(self.dense, 8192, "q8_0", "q8_0")
        self.assertEqual(fixed, 0.0)
        self.assertEqual(per_token, 0.0)


class InterleavedAttentionTest(unittest.TestCase):
    """Gemma-style local/global attention, the geometry that made the model
    predict 284 GiB for a backend that runs in 37 GiB."""

    def setUp(self):
        self.gemma = budget.model_geometry(GEMMA4_31B)

    def test_layers_are_split_into_window_and_global(self):
        self.assertEqual(self.gemma["swa_layers"], 50)
        self.assertEqual(self.gemma["full_attention_layers"], 10)
        self.assertEqual(self.gemma["recurrent_layers"], 0)
        self.assertTrue(self.gemma["supports_swa"])

    def test_per_layer_head_counts_are_not_collapsed_to_the_maximum(self):
        """Taking the max charged the context-scaling layers 16 heads when they
        have 4 — four times the cost, on exactly the term that grows."""
        self.assertEqual(self.gemma["head_count_kv_swa"], 16)
        self.assertEqual(self.gemma["head_count_kv_full"], 4)

    def test_window_layers_use_their_own_head_dimension(self):
        self.assertEqual(self.gemma["key_length"], 512)
        self.assertEqual(self.gemma["key_length_swa"], 256)

    def test_only_global_layers_scale_with_context(self):
        per_token = budget.kv_bytes_per_token(self.gemma, "q8_0", "q8_0")
        expected = 4 * (512 + 512) * (34 / 32) * 10
        self.assertAlmostEqual(per_token, expected, delta=1.0)

    def test_window_layers_cost_the_window_whatever_the_context(self):
        short = budget.swa_kv_bytes(self.gemma, 8192, "q8_0", "q8_0")
        long = budget.swa_kv_bytes(self.gemma, 255998, "q8_0", "q8_0")
        self.assertEqual(short, long)
        self.assertAlmostEqual(long / budget.MIB, 425.0, delta=1.0)

    def test_swa_full_makes_the_window_layers_hold_everything(self):
        windowed = budget.swa_kv_bytes(self.gemma, 255998, "q8_0", "q8_0")
        full = budget.swa_kv_bytes(self.gemma, 255998, "q8_0", "q8_0", swa_full=True)
        self.assertGreater(full, windowed * 200)

    def test_the_prediction_matches_the_running_backend(self):
        """Live on this box: 18,978 + 18,258 MiB across two GPUs for
        Gemma4-31B-Q4_K_M at --ctx-size 255998 --parallel 1, with a 1145 MiB
        projector. The model predicted 284,145 MiB before this was fixed."""
        prediction = budget.predict(dict(self.gemma, file_size_mib=17821), {
            "ctx_size": 255998, "parallel": 1, "devices": 2, "ubatch": 512,
            "cache_type_k": "q8_0", "cache_type_v": "q8_0",
            "tensor_split": "1,1", "weights_mib": 17821, "projector_mib": 1145,
        })
        self.assertAlmostEqual(prediction["vram"]["kv_mib"], 11050, delta=200)
        self.assertLess(abs(prediction["vram"]["total_mib"] - 37236), 2000)

    def test_it_fits_the_hardware_it_actually_runs_on(self):
        prediction = budget.predict(dict(self.gemma, file_size_mib=17821), {
            "ctx_size": 255998, "parallel": 1, "devices": 2,
            "cache_type_k": "q8_0", "cache_type_v": "q8_0",
            "tensor_split": "1,1", "weights_mib": 17821, "projector_mib": 1145,
        })
        gpus = [{"index": 0, "mem_total": 24576}, {"index": 1, "mem_total": 24576}]
        verdict = budget.evaluate(self.gemma, {}, prediction, gpus, {"mem_available_mib": 25864})
        self.assertNotIn("vram_overcommit", {issue["code"] for issue in verdict["issues"]})

    def test_swa_full_is_reported_as_expensive_rather_than_inert(self):
        settings = {"swa_full": "on", "cache_type_k": "q8_0", "cache_type_v": "q8_0"}
        prediction = budget.predict(dict(self.gemma, file_size_mib=17821),
                                    dict(settings, ctx_size=255998, parallel=1, devices=2))
        verdict = budget.evaluate(self.gemma, settings, prediction, [], {})
        codes = {issue["code"] for issue in verdict["issues"]}
        self.assertIn("swa_full_expensive", codes)
        self.assertNotIn("swa_full_unsupported", codes)

    def test_swa_full_is_never_recommended(self):
        """It was recommended wherever the model supported SWA, which is the
        one place it is expensive rather than merely ignored."""
        recommendation = budget.recommend(
            dict(self.gemma, file_size_mib=17821),
            [{"index": 0, "mem_total": 24576}, {"index": 1, "mem_total": 24576}],
            {"mem_available_mib": 25864})
        self.assertEqual(recommendation["swa_full"], "off")


class PredictionTest(unittest.TestCase):
    def setUp(self):
        self.qwen = budget.model_geometry(QWEN36_27B)
        self.live = {
            "ctx_size": 262144, "parallel": 2, "devices": 2, "ubatch": 512,
            "cache_type_k": "q8_0", "cache_type_v": "q8_0",
            "ctx_checkpoints": 8, "cache_ram": 8192, "tensor_split": "1,1",
            "weights_mib": 21824, "projector_mib": 885, "spec_method": "draft-mtp",
        }

    def test_kv_and_context_accounting(self):
        prediction = budget.predict(self.qwen, self.live)
        self.assertEqual(prediction["per_slot_context"], 131072)
        self.assertEqual(prediction["vram"]["kv_mib"], 8704)

    def test_exact_terms_are_the_sum_of_their_parts(self):
        vram = budget.predict(self.qwen, self.live)["vram"]
        self.assertEqual(
            vram["exact_mib"],
            vram["weights_mib"] + vram["projector_mib"] + vram["kv_mib"]
            + vram["recurrent_mib"] + vram["draft_mib"])

    def test_upper_bound_exceeds_the_point_estimate(self):
        vram = budget.predict(self.qwen, self.live)["vram"]
        self.assertGreater(vram["upper_mib"], vram["total_mib"])
        # Only the estimated half carries the uncertainty band.
        self.assertEqual(vram["upper_mib"] - vram["total_mib"],
                         round(vram["estimated_mib"] * budget.COMPUTE_UNCERTAINTY))

    def test_checkpoints_dominate_the_host_budget(self):
        """32 checkpoints was the configuration that thrashed; 8 is what fits."""
        thirty_two = budget.predict(self.qwen, dict(self.live, ctx_checkpoints=32))["host"]
        eight = budget.predict(self.qwen, self.live)["host"]
        self.assertGreater(thirty_two["cache_ram_shortfall_mib"], 15000)
        self.assertEqual(eight["cache_ram_shortfall_mib"], 0)

    def test_tensor_split_distributes_weights(self):
        vram = budget.predict(self.qwen, dict(self.live, tensor_split="3,1"))["vram"]
        first, second = vram["per_device"]
        self.assertAlmostEqual(first["weights_mib"] / second["weights_mib"], 3.0, delta=0.05)

    def test_missing_tensor_split_divides_evenly(self):
        vram = budget.predict(self.qwen, dict(self.live, tensor_split=""))["vram"]
        first, second = vram["per_device"]
        self.assertEqual(first["weights_mib"], second["weights_mib"])

    def test_draft_cache_only_applies_to_mtp(self):
        with_mtp = budget.predict(self.qwen, self.live)["vram"]["draft_mib"]
        without = budget.predict(self.qwen, dict(self.live, spec_method="off"))["vram"]["draft_mib"]
        self.assertGreater(with_mtp, 0)
        self.assertEqual(without, 0)

    def test_single_slot_context_is_not_divided(self):
        prediction = budget.predict(self.qwen, dict(self.live, parallel=1))
        self.assertEqual(prediction["per_slot_context"], 262144)


# --------------------------------------------------------------------------
# verdicts
# --------------------------------------------------------------------------

class EvaluateTest(unittest.TestCase):
    def setUp(self):
        self.qwen = budget.model_geometry(QWEN36_27B)
        self.gpus = [{"index": 0, "mem_total": 24576}, {"index": 1, "mem_total": 24576}]
        self.host = {"mem_available_mib": 19800}
        self.settings = {
            "ctx_size": 262144, "parallel": 2, "devices": 2,
            "cache_type_k": "q8_0", "cache_type_v": "q8_0",
            "ctx_checkpoints": 8, "cache_ram": 8192, "tensor_split": "1,1",
            "weights_mib": 21824, "projector_mib": 885,
            "swa_full": "off", "fit": "off", "fit_ctx": "",
        }

    def codes(self, **overrides) -> set[str]:
        settings = dict(self.settings, **overrides)
        prediction = budget.predict(self.qwen, settings)
        verdict = budget.evaluate(self.qwen, settings, prediction, self.gpus, self.host)
        return {issue["code"] for issue in verdict["issues"]}

    def test_live_configuration_raises_no_warnings(self):
        codes = self.codes()
        self.assertNotIn("cache_ram_shortfall", codes)
        self.assertNotIn("swa_full_unsupported", codes)
        self.assertNotIn("fit_ctx_without_fit", codes)
        self.assertNotIn("vram_overcommit", codes)

    def test_original_configuration_is_diagnosed_in_full(self):
        """Every finding the journal analysis produced, from config alone."""
        codes = self.codes(ctx_checkpoints=32, swa_full="on", fit_ctx="4096")
        self.assertIn("cache_ram_shortfall", codes)
        self.assertIn("host_ram_overcommit", codes)
        self.assertIn("swa_full_unsupported", codes)
        self.assertIn("fit_ctx_without_fit", codes)

    def test_swa_full_is_accepted_on_a_model_that_has_swa(self):
        swa_model = budget.model_geometry(dict(DENSE_8B, **{"llama.attention.sliding_window": 4096}))
        settings = dict(self.settings, swa_full="on", weights_mib=4000, projector_mib=0)
        verdict = budget.evaluate(swa_model, settings, budget.predict(swa_model, settings),
                                  self.gpus, self.host)
        self.assertNotIn("swa_full_unsupported", {i["code"] for i in verdict["issues"]})

    def test_fit_ctx_is_fine_when_fit_is_on(self):
        self.assertNotIn("fit_ctx_without_fit", self.codes(fit="on", fit_ctx="4096"))

    def test_fit_ctx_of_zero_is_not_flagged(self):
        self.assertNotIn("fit_ctx_without_fit", self.codes(fit="off", fit_ctx="0"))

    def test_cache_reuse_is_flagged_on_a_multimodal_backend(self):
        """llama-server logs 'cache_reuse is not supported by multimodal' and
        carries on, so the setting reads as applied when it is inert."""
        self.assertIn("cache_reuse_with_multimodal",
                      self.codes(cache_reuse="256", mmproj_path="/models/x.mmproj.gguf"))

    def test_cache_reuse_is_fine_without_a_projector(self):
        self.assertNotIn("cache_reuse_with_multimodal",
                         self.codes(cache_reuse="256", mmproj_path=""))

    def test_cache_reuse_of_zero_is_not_flagged(self):
        self.assertNotIn("cache_reuse_with_multimodal",
                         self.codes(cache_reuse="0", mmproj_path="/models/x.mmproj.gguf"))

    def test_vram_overcommit_is_an_error_not_a_warning(self):
        settings = dict(self.settings, ctx_size=1048576)
        prediction = budget.predict(self.qwen, settings)
        verdict = budget.evaluate(self.qwen, settings, prediction, self.gpus, self.host)
        self.assertFalse(verdict["ok"])
        self.assertEqual(verdict["issues"][0]["level"], "error")
        self.assertEqual(verdict["issues"][0]["code"], "vram_overcommit")

    def test_per_slot_context_is_always_stated_for_multi_slot(self):
        self.assertIn("per_slot_context", self.codes())
        self.assertNotIn("per_slot_context", self.codes(parallel=1))

    def test_issues_are_ordered_by_severity(self):
        settings = dict(self.settings, ctx_size=1048576, ctx_checkpoints=32, swa_full="on")
        verdict = budget.evaluate(self.qwen, settings, budget.predict(self.qwen, settings),
                                  self.gpus, self.host)
        levels = [issue["level"] for issue in verdict["issues"]]
        self.assertEqual(levels, sorted(levels, key={"error": 0, "warn": 1, "info": 2}.get))

    def test_missing_gpu_data_does_not_fabricate_a_verdict(self):
        settings = dict(self.settings)
        verdict = budget.evaluate(self.qwen, settings, budget.predict(self.qwen, settings), [], {})
        self.assertTrue(verdict["ok"])
        self.assertNotIn("vram_overcommit", {i["code"] for i in verdict["issues"]})


# --------------------------------------------------------------------------
# env plumbing and recommendations
# --------------------------------------------------------------------------

class SettingsFromEnvTest(unittest.TestCase):
    ENV = {
        "CHAT_PRIMARY_MODEL_PATH": "/models/primary.gguf",
        "CHAT_PRIMARY_MMPROJ_PATH": "/models/primary.mmproj.gguf",
        "CHAT_PRIMARY_CTX_SIZE": "262144",
        "CHAT_PRIMARY_N_PARALLEL": "2",
        "CHAT_PRIMARY_CTX_CHECKPOINTS": "8",
        "CHAT_PRIMARY_FIT_CTX": "",
        "CHAT_PRIMARY_GPU_VISIBLE_DEVICES": "0,1",
        "CHAT2_CTX_SIZE": "65536",
    }

    def test_reads_the_requested_prefix(self):
        settings = budget.settings_from_env(self.ENV, "chat-primary")
        self.assertEqual(settings["ctx_size"], "262144")
        self.assertEqual(settings["parallel"], "2")
        self.assertEqual(settings["devices"], 2)

    def test_empty_values_are_left_unset(self):
        """An explicitly cleared key must not arrive as an empty string, which
        would then be flagged as a configured-but-dead --fit-ctx."""
        self.assertNotIn("fit_ctx", budget.settings_from_env(self.ENV, "chat-primary"))

    def test_prefixes_do_not_leak_between_backends(self):
        self.assertEqual(budget.settings_from_env(self.ENV, "chat-secondary")["ctx_size"], "65536")

    def test_device_count_defaults_to_one(self):
        self.assertEqual(budget.settings_from_env({}, "chat-primary")["devices"], 1)

    def test_unknown_backend_is_rejected(self):
        with self.assertRaises(ValueError):
            budget.settings_from_env(self.ENV, "not-a-backend")


class BudgetForTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = pathlib.Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_missing_model_reports_rather_than_raises(self):
        result = budget.budget_for({"CHAT_PRIMARY_MODEL_PATH": "/nope.gguf"}, "chat-primary")
        self.assertIn("model not found", result["error"])
        self.assertIsNone(result["prediction"])

    def test_unreadable_model_reports_rather_than_raises(self):
        path = self.tmp / "junk.gguf"
        path.write_bytes(b"not gguf at all" + b"\0" * 64)
        result = budget.budget_for({"CHAT_PRIMARY_MODEL_PATH": str(path)}, "chat-primary")
        self.assertIn("could not read model metadata", result["error"])

    def test_full_budget_from_env(self):
        path = write_gguf(self.tmp / "m.gguf", QWEN36_27B, trailing_bytes=64 * 1024)
        result = budget.budget_for({
            "CHAT_PRIMARY_MODEL_PATH": str(path),
            "CHAT_PRIMARY_CTX_SIZE": "131072",
            "CHAT_PRIMARY_N_PARALLEL": "2",
            "CHAT_PRIMARY_CTX_CHECKPOINTS": "8",
            "CHAT_PRIMARY_CACHE_RAM": "8192",
        }, "chat-primary")
        self.assertIsNone(result["error"])
        self.assertEqual(result["prediction"]["per_slot_context"], 65536)
        self.assertEqual(result["geometry"]["architecture"], "qwen35")

    def test_overrides_price_an_unsaved_change(self):
        path = write_gguf(self.tmp / "m.gguf", QWEN36_27B)
        env = {"CHAT_PRIMARY_MODEL_PATH": str(path), "CHAT_PRIMARY_CTX_SIZE": "262144",
               "CHAT_PRIMARY_N_PARALLEL": "2"}
        saved = budget.budget_for(env, "chat-primary")
        edited = budget.budget_for(env, "chat-primary", overrides={"ctx_size": "65536"})
        self.assertEqual(saved["prediction"]["per_slot_context"], 131072)
        self.assertEqual(edited["prediction"]["per_slot_context"], 32768)


class RecommendTest(unittest.TestCase):
    def setUp(self):
        # The fixture is written from metadata alone, so it carries no weight.
        # Recommendations turn on how much VRAM the weights leave for KV, so
        # stand in the real file's size.
        self.qwen = dict(budget.model_geometry(QWEN36_27B), file_size_mib=21824)

    def test_recommendation_fits_the_detected_hardware(self):
        gpus = [{"index": 0, "mem_total": 24576}, {"index": 1, "mem_total": 24576}]
        recommendation = budget.recommend(self.qwen, gpus, {"mem_available_mib": 19800})
        prediction = budget.predict(self.qwen, {
            "ctx_size": recommendation["ctx_size"], "parallel": 2, "devices": 2,
            "cache_type_k": "q8_0", "cache_type_v": "q8_0",
            "weights_mib": self.qwen["file_size_mib"],
        })
        self.assertLessEqual(prediction["vram"]["upper_mib"], sum(g["mem_total"] for g in gpus))
        self.assertGreater(recommendation["ctx_size"], 0)

    def test_smaller_hardware_gets_a_smaller_recommendation(self):
        big = budget.recommend(self.qwen, [{"index": 0, "mem_total": 81920}],
                               {"mem_available_mib": 128000})
        small = budget.recommend(self.qwen, [{"index": 0, "mem_total": 24576}],
                                 {"mem_available_mib": 8000})
        self.assertGreater(big["ctx_size"], small["ctx_size"])

    def test_checkpoints_are_sized_to_available_ram(self):
        """The constant this replaces was 32 regardless of host or model."""
        tight = budget.recommend(self.qwen, [{"index": 0, "mem_total": 24576}],
                                 {"mem_available_mib": 4000})
        roomy = budget.recommend(self.qwen, [{"index": 0, "mem_total": 24576}],
                                 {"mem_available_mib": 128000})
        self.assertLess(tight["ctx_checkpoints"], roomy["ctx_checkpoints"])
        self.assertGreaterEqual(tight["ctx_checkpoints"], 2)
        self.assertLessEqual(roomy["ctx_checkpoints"], 32)

    def test_swa_is_only_recommended_where_it_works(self):
        self.assertEqual(budget.recommend(self.qwen, [{"index": 0, "mem_total": 24576}],
                                          {"mem_available_mib": 19800})["swa_full"], "off")

    def test_the_recommendation_passes_its_own_verdict(self):
        """A recommendation that trips the pre-flight is not a recommendation."""
        gpus = [{"index": 0, "mem_total": 24576}, {"index": 1, "mem_total": 24576}]
        host = {"mem_available_mib": 19800}
        recommendation = budget.recommend(self.qwen, gpus, host)
        settings = {
            "ctx_size": recommendation["ctx_size"],
            "parallel": recommendation["parallel"],
            "ctx_checkpoints": recommendation["ctx_checkpoints"],
            "cache_ram": recommendation["cache_ram"],
            "cache_type_k": recommendation["cache_type_k"],
            "cache_type_v": recommendation["cache_type_v"],
            "cache_reuse": recommendation["cache_reuse"],
            "swa_full": recommendation["swa_full"],
            "fit": recommendation["fit"],
            "fit_ctx": recommendation["fit_ctx"],
            "devices": 2,
            "weights_mib": self.qwen["file_size_mib"],
        }
        verdict = budget.evaluate(
            self.qwen, settings, budget.predict(self.qwen, settings), gpus, host)
        actionable = [issue for issue in verdict["issues"] if issue["level"] != "info"]
        self.assertEqual(actionable, [], actionable)

    def test_it_does_not_recommend_past_the_trained_context(self):
        roomy = budget.recommend(self.qwen, [{"index": 0, "mem_total": 81920},
                                             {"index": 1, "mem_total": 81920}],
                                 {"mem_available_mib": 128000})
        self.assertLessEqual(roomy["ctx_size"], self.qwen["train_context_length"])

    def test_the_projector_and_draft_head_are_priced_in(self):
        """Priced without them, the recommendation is for a backend nobody is
        launching: this box carries an 885 MiB projector and an MTP head."""
        # Sized so the ceiling is VRAM rather than the trained context, which is
        # where the difference between the two is visible at all.
        gpus = [{"index": 0, "mem_total": 20480}, {"index": 1, "mem_total": 20480}]
        host = {"mem_available_mib": 19800}
        bare = budget.recommend(self.qwen, gpus, host,
                                base_settings={"weights_mib": self.qwen["file_size_mib"]})
        loaded = budget.recommend(self.qwen, gpus, host, base_settings={
            "weights_mib": self.qwen["file_size_mib"],
            "projector_mib": 885, "spec_method": "draft-mtp",
        })
        self.assertLess(loaded["ctx_size"], bare["ctx_size"])

    def test_cache_reuse_is_not_recommended_for_a_multimodal_backend(self):
        """llama-server disables --cache-reuse outright when a projector is
        loaded, so a chunk size there is a dead flag."""
        gpus = [{"index": 0, "mem_total": 24576}]
        host = {"mem_available_mib": 19800}
        self.assertEqual(budget.recommend(self.qwen, gpus, host,
                                          base_settings={"projector_mib": 885})["cache_reuse"], 0)
        self.assertEqual(budget.recommend(self.qwen, gpus, host)["cache_reuse"], 256)

    def test_auto_fit_off_clears_the_flag_it_contradicts(self):
        recommendation = budget.recommend(self.qwen, [{"index": 0, "mem_total": 24576}],
                                          {"mem_available_mib": 19800})
        self.assertEqual(recommendation["fit"], "off")
        self.assertEqual(recommendation["fit_ctx"], "")

    def test_the_prompt_cache_budget_matches_what_the_checkpoints_claim(self):
        """The budget used to be a flat quarter of host RAM, which is either far
        more than the checkpoints can use or far less than they need."""
        recommendation = budget.recommend(self.qwen, [{"index": 0, "mem_total": 24576}],
                                          {"mem_available_mib": 128000})
        claimed = (recommendation["checkpoint_each_mib"]
                   * recommendation["ctx_checkpoints"] * recommendation["parallel"])
        self.assertLessEqual(claimed, recommendation["cache_ram"])
        self.assertLess(recommendation["cache_ram"], claimed + 1024)


# --------------------------------------------------------------------------
# the model this box actually runs, when it is present
# --------------------------------------------------------------------------

LIVE_MODEL = pathlib.Path("/mnt/LLMs/llamacpp/llm-stack-git/models/Qwen3.6-27B-Q6_K.gguf")


@unittest.skipUnless(LIVE_MODEL.is_file(), "live model not present on this host")
class LiveModelTest(unittest.TestCase):
    """Guards the fixture above against drift in the real file."""

    def test_real_file_matches_the_fixture(self):
        geometry = budget.model_geometry(budget.read_gguf_metadata(LIVE_MODEL))
        expected = budget.model_geometry(QWEN36_27B)
        for key in ("architecture", "layers", "full_attention_layers",
                    "recurrent_layers", "head_count_kv", "key_length", "value_length"):
            self.assertEqual(geometry[key], expected[key], key)

    def test_recurrent_state_matches_the_journal(self):
        geometry = budget.model_geometry(budget.read_gguf_metadata(LIVE_MODEL))
        self.assertAlmostEqual(
            budget.recurrent_state_bytes(geometry) / budget.MIB, 149.6, delta=0.1)


LIVE_SWA_MODEL = pathlib.Path(
    "/mnt/LLMs/llamacpp/llm-stack-git/models/"
    "Gemma4-31B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf")


@unittest.skipUnless(LIVE_SWA_MODEL.is_file(), "live SWA model not present on this host")
class LiveSlidingWindowModelTest(unittest.TestCase):
    """Guards the interleaved-attention fixture against the real file."""

    def test_real_file_matches_the_fixture(self):
        geometry = budget.model_geometry(budget.read_gguf_metadata(LIVE_SWA_MODEL))
        expected = budget.model_geometry(GEMMA4_31B)
        for key in ("layers", "swa_layers", "full_attention_layers", "sliding_window",
                    "head_count_kv_full", "head_count_kv_swa",
                    "key_length", "key_length_swa", "value_length", "value_length_swa"):
            self.assertEqual(geometry[key], expected[key], key)

    def test_the_live_configuration_is_not_reported_as_overcommitted(self):
        """It runs on this box, so a model that says it cannot is wrong."""
        geometry = budget.model_geometry(budget.read_gguf_metadata(LIVE_SWA_MODEL))
        settings = {"ctx_size": 255998, "parallel": 1, "devices": 2,
                    "cache_type_k": "q8_0", "cache_type_v": "q8_0",
                    "tensor_split": "1,1", "projector_mib": 1145}
        prediction = budget.predict(geometry, settings)
        gpus = [{"index": 0, "mem_total": 24576}, {"index": 1, "mem_total": 24576}]
        verdict = budget.evaluate(geometry, settings, prediction, gpus,
                                  {"mem_available_mib": 25864})
        self.assertNotIn("vram_overcommit", {issue["code"] for issue in verdict["issues"]})
        # Observed live: 18,978 + 18,258 MiB for this process across both GPUs.
        self.assertLess(abs(prediction["vram"]["total_mib"] - 37236), 2000)


if __name__ == "__main__":
    unittest.main()
