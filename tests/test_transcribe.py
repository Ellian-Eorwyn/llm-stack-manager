"""The transcription sidecar and the config surface that feeds it.

The highest-value test here is `LazyImportTests`: it runs in CI, where no ASR
runtime is installed, and asserts that the server still imports, still answers
`/health`, and turns a request for a missing engine into an actionable 503
rather than a dead process. That is the property the whole engine registry
exists to provide, and it is invisible on a developer box where the runtimes
happen to be present.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import threading
import time
import unittest


def _load(name: str, relative: str):
    root = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(name, root / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "web"))

config_fields = _load("transcribe_config_fields", "web/config_fields.py")
config_env = _load("transcribe_config_env", "web/config_env.py")
tsrv = _load("transcribe_server", "scripts/transcribe-server.py")

ENGINE_SUFFIXES = (
    "_BACKEND_TYPE", "_LOCAL_MODEL", "_UPSTREAM_URL", "_MODEL", "_API_KEY",
    "_TRANSCRIBE_PATH", "_STREAM_OUTPUT_ENABLED", "_STREAM_OUTPUT_TARGET",
    "_STREAM_OUTPUT_FORMAT", "_SPEAKER_DETECTION", "_SPEAKER_MODE", "_SPEAKER_COUNT",
)


def _cfg(**overrides):
    cfg = tsrv.default_config()
    cfg["server"]["host"] = "127.0.0.1"
    cfg["router"]["yield_mode"] = "off"
    cfg["limits"]["idle_unload_seconds"] = 0
    for dotted, value in overrides.items():
        section, _, key = dotted.partition("__")
        if key:
            cfg[section][key] = value
        else:
            cfg[section] = value
    return cfg


class EngineRegistryTests(unittest.TestCase):
    def test_ids_and_prefixes_are_unique(self):
        ids = [e["id"] for e in config_fields.TRANSCRIPTION_ENGINES]
        prefixes = [e["env_prefix"] for e in config_fields.TRANSCRIPTION_ENGINES]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(prefixes), len(set(prefixes)))

    def test_every_registry_engine_is_implemented(self):
        """The config offers exactly the engines the server can dispatch."""
        self.assertEqual(
            {e["id"] for e in config_fields.TRANSCRIPTION_ENGINES},
            set(tsrv.ENGINES),
        )

    def test_every_engine_has_a_full_field_block(self):
        keys = {f["key"] for f in config_fields.CONFIG_FIELDS}
        for engine in config_fields.TRANSCRIPTION_ENGINES:
            for suffix in ENGINE_SUFFIXES:
                self.assertIn(engine["env_prefix"] + suffix, keys)

    def test_every_transcription_field_restarts_the_sidecar_only(self):
        for field in config_fields.CONFIG_FIELDS:
            if field["section"] != "Transcription":
                continue
            self.assertEqual(
                config_fields.RESTART_HINTS.get(field["key"]), ["transcript-backend"],
                f"{field['key']} does not restart transcript-backend alone")

    def test_the_default_engine_options_track_the_registry(self):
        field = next(f for f in config_fields.CONFIG_FIELDS
                     if f["key"] == "TRANSCRIPT_ACTIVE_ENGINE")
        self.assertEqual(field["options"],
                         [e["id"] for e in config_fields.TRANSCRIPTION_ENGINES])

    def test_the_transcription_section_renders(self):
        self.assertIn("Transcription", config_fields.CORE_CONFIG_SECTIONS)

    def test_asr_keys_restart_the_router_not_the_sidecar(self):
        """`asr` is a router child, not a unit, so the hint cannot be redirected."""
        self.assertEqual(config_fields.RESTART_HINTS.get("ASR_MODEL_PATH"), ["llama-router"])


class NemoTimestampTests(unittest.TestCase):
    """NeMo fills `segment`, `word` and `char` independently.

    Reading only `segment` left every `words` list empty while the engine
    advertised `word_timestamps: true` — a capability the payload contradicted.
    """

    @staticmethod
    def _item(**stamps):
        return type("Hyp", (), {"text": "hello there world", "timestamp": stamps})()

    SEGMENTS = [{"segment": "hello there", "start": 0.0, "end": 1.0},
                {"segment": "world", "start": 1.0, "end": 2.0}]
    WORDS = [{"word": "hello", "start": 0.0, "end": 0.4},
             {"word": "there", "start": 0.4, "end": 1.0},
             {"word": "world", "start": 1.2, "end": 2.0}]

    def test_words_are_attached_to_their_segment(self):
        segments = tsrv._NemoEngine._segments_from(
            self._item(segment=self.SEGMENTS, word=self.WORDS), "hello there world")
        self.assertEqual([w["word"] for w in segments[0]["words"]], ["hello", "there"])
        self.assertEqual([w["word"] for w in segments[1]["words"]], ["world"])

    def test_a_word_past_every_span_still_lands_somewhere(self):
        stray = self.WORDS + [{"word": "trailing", "start": 99.0, "end": 99.5}]
        segments = tsrv._NemoEngine._segments_from(
            self._item(segment=self.SEGMENTS, word=stray), "x")
        self.assertEqual(segments[-1]["words"][-1]["word"], "trailing")

    def test_word_only_stamps_do_not_nest_words_inside_themselves(self):
        segments = tsrv._NemoEngine._segments_from(self._item(word=self.WORDS), "x")
        self.assertEqual(len(segments), 3)
        self.assertEqual([s["words"] for s in segments], [[], [], []])

    def test_no_stamps_falls_back_to_one_span(self):
        segments = tsrv._NemoEngine._segments_from(self._item(), "hello there world")
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["text"], "hello there world")


class LegacyRenameTests(unittest.TestCase):
    LEGACY = [k for k in config_fields.LEGACY_ENV_KEY_MAP if k.startswith("WHISPERKIT_")]

    def test_all_twelve_whisperkit_keys_are_mapped(self):
        self.assertEqual(len(self.LEGACY), 12)

    def test_legacy_values_backfill_the_canonical_key(self):
        for legacy in self.LEGACY:
            canonical = config_fields.LEGACY_ENV_KEY_MAP[legacy]
            result = config_env.normalize_env_keys({legacy: "carried"})
            self.assertEqual(result[canonical], "carried", legacy)

    def test_writes_are_rewritten_to_the_canonical_key(self):
        for legacy in self.LEGACY:
            canonical = config_fields.LEGACY_ENV_KEY_MAP[legacy]
            updates = config_env.normalize_config_updates({legacy: "x"})
            self.assertIn(canonical, updates)
            self.assertNotIn(legacy, updates)

    def test_each_rename_is_explained(self):
        for legacy in self.LEGACY:
            self.assertIn("WhisperKit", config_fields.DEPRECATED_ENV_KEY_NOTES[legacy])


class StaleModelRepairTests(unittest.TestCase):
    """Whisper sizes left on NeMo slots by the old shared default.

    Every engine's model once defaulted off one `TRANSCRIPT_LOCAL_MODEL_SIZE`,
    so configs written then carry `PARAKEET_V3_LOCAL_MODEL=preset:large-v3`.
    NeMo can never load that, and the failure names the model rather than the
    setting — with parakeet as the default engine it breaks every request.
    """

    def _engine(self, engine_id):
        return config_fields.TRANSCRIPTION_ENGINE_BY_ID[engine_id]

    def test_a_whisper_preset_on_a_nemo_slot_is_replaced(self):
        self.assertEqual(
            config_fields.repair_transcription_model(self._engine("parakeet-v3"), "preset:large-v3"),
            "preset:nvidia/parakeet-tdt-0.6b-v3")

    def test_every_whisper_preset_is_caught(self):
        engine = self._engine("canary-qwen")
        for preset in config_fields.WHISPER_MODEL_PRESETS:
            self.assertEqual(
                config_fields.repair_transcription_model(engine, f"preset:{preset}"),
                "preset:nvidia/canary-qwen-2.5b", preset)

    def test_whisper_keeps_its_own_presets(self):
        engine = self._engine("faster-whisper")
        for preset in ("preset:large-v3", "preset:turbo", "preset:distil-large-v3"):
            self.assertEqual(config_fields.repair_transcription_model(engine, preset), preset)

    def test_a_deliberate_choice_is_never_overwritten(self):
        """Only bare Whisper presets are residue; paths and repo ids are intent."""
        engine = self._engine("parakeet-v3")
        for value in ("local:/models/custom.nemo", "preset:nvidia/parakeet-tdt-1.1b",
                      "preset:some-org/some-model"):
            self.assertEqual(config_fields.repair_transcription_model(engine, value), value)

    def test_the_repair_runs_on_read(self):
        repaired = config_env.normalize_env_keys({"PARAKEET_V3_LOCAL_MODEL": "preset:large-v3"})
        self.assertEqual(repaired["PARAKEET_V3_LOCAL_MODEL"],
                         "preset:nvidia/parakeet-tdt-0.6b-v3")

    def test_no_engine_defaults_to_another_runtimes_model(self):
        defaults = config_env.normalize_env_keys({})
        for engine in config_fields.TRANSCRIPTION_ENGINES:
            value = defaults[f"{engine['env_prefix']}_LOCAL_MODEL"]
            if engine["runtime"] == "faster-whisper" or not value.startswith("preset:"):
                continue
            self.assertNotIn(value.split(":", 1)[1], config_fields.WHISPER_MODEL_PRESETS,
                             f"{engine['id']} defaults to a Whisper model")


class LazyImportTests(unittest.TestCase):
    """An uninstalled runtime must not be able to take the service down."""

    def setUp(self):
        self.client = tsrv.create_app(cfg=_cfg()).test_client()

    def test_all_five_engines_register_without_any_runtime(self):
        self.assertEqual(len(tsrv.ENGINES), 5)

    def test_health_answers_before_anything_is_loaded(self):
        body = self.client.get("/health").get_json()
        self.assertEqual(body["status"], "ok")

    def test_a_missing_runtime_is_a_503_with_an_install_hint(self):
        resp = self.client.post("/transcribe", data={
            "engine": "parakeet-v3", "audio_base64": "UklGRiQAAABXQVZF"})
        self.assertEqual(resp.status_code, 503)
        error = resp.get_json()["error"]
        self.assertEqual(error["type"], "engine_unavailable")
        self.assertIn("install-transcribe.sh", error["hint"])

    def test_the_service_survives_a_missing_runtime(self):
        self.client.post("/transcribe", data={
            "engine": "canary-qwen", "audio_base64": "UklGRiQAAABXQVZF"})
        self.assertEqual(self.client.get("/health").status_code, 200)


class StubEngine(tsrv.Engine):
    """A deterministic engine, so response shape can be tested without a model."""

    engine_id = "stub"
    runtime = "stub"
    capabilities = {"segments": True, "word_timestamps": True, "translate": True,
                    "diarization": False, "language_detect": True}
    loads = 0
    unloads = 0
    delay = 0.0

    def load(self, model_ref):
        StubEngine.loads += 1
        self.model = "stub-model"
        self.model_ref = model_ref

    def unload(self):
        StubEngine.unloads += 1
        self.model = None

    def transcribe(self, path, req):
        if StubEngine.delay:
            time.sleep(StubEngine.delay)
        return {
            "segments": [
                {"id": 0, "start": 0.0, "end": 1.5, "text": "hello there", "speaker": None,
                 "avg_logprob": -0.2, "no_speech_prob": 0.01, "compression_ratio": 1.1,
                 "words": [{"start": 0.0, "end": 0.5, "word": "hello", "probability": 0.9}]},
                {"id": 1, "start": 1.5, "end": 4.25, "text": "general kenobi", "speaker": None,
                 "avg_logprob": -0.3, "no_speech_prob": 0.02, "compression_ratio": 1.2,
                 "words": []},
            ],
            "language": "en", "language_probability": 0.99, "duration": 4.25,
        }


class _StubMixin:
    def setUp(self):
        StubEngine.loads = StubEngine.unloads = 0
        StubEngine.delay = 0.0
        tsrv.ENGINES["stub"] = StubEngine
        cfg = _cfg()
        cfg["engines"]["stub"] = {"model": "preset:stub", "backend_type": "local"}
        cfg["active_engine"] = "stub"
        self.cfg = cfg
        self.client = tsrv.create_app(cfg=cfg).test_client()

    def tearDown(self):
        tsrv.ENGINES.pop("stub", None)

    def _post(self, path="/transcribe", **fields):
        data = {"audio_base64": "UklGRiQAAABXQVZF", **fields}
        return self.client.post(path, data=data)


class ResponseShapeTests(_StubMixin, unittest.TestCase):
    NATIVE_KEYS = ("ok", "request_id", "created", "text", "language", "language_probability",
                   "duration", "segments", "words", "engine", "model", "device",
                   "compute_type", "capabilities", "timings")

    def test_the_native_envelope_carries_every_documented_key(self):
        body = self._post(response_format="json").get_json()
        for key in self.NATIVE_KEYS:
            self.assertIn(key, body)
        self.assertTrue(body["ok"])
        self.assertEqual(body["text"], "hello there general kenobi")

    def test_words_are_also_flattened_to_the_top_level(self):
        body = self._post(response_format="json", word_timestamps="true").get_json()
        self.assertEqual([w["word"] for w in body["words"]], ["hello"])

    def test_verbose_json_is_exactly_the_openai_key_set(self):
        body = self._post(response_format="verbose_json").get_json()
        self.assertEqual(set(body),
                         {"task", "language", "duration", "text", "segments", "words"})

    def test_openai_json_is_only_the_text(self):
        """SDKs expect {"text": ...} from this endpoint and nothing more."""
        body = self._post("/v1/audio/transcriptions", response_format="json").get_json()
        self.assertEqual(set(body), {"text"})

    def test_text_format_is_plain_text(self):
        resp = self._post(response_format="text")
        self.assertTrue(resp.mimetype.startswith("text/plain"))
        # One charset, not two: `mimetype=` would append a second one.
        self.assertEqual(resp.headers["Content-Type"].count("charset"), 1)

    def test_markdown_is_native_only(self):
        resp = self._post("/v1/audio/transcriptions", response_format="markdown")
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.get_json()["error"]["type"], "bad_request")

    def test_an_unknown_engine_is_rejected(self):
        resp = self._post(engine="does-not-exist")
        self.assertEqual(resp.status_code, 400)

    def test_whisper_1_falls_back_to_the_default_engine(self):
        """Stock SDK code hardcodes this and must not 400."""
        body = self._post("/v1/audio/transcriptions", model="whisper-1").get_json()
        self.assertEqual(body["text"], "hello there general kenobi")


class SubtitleFormatTests(_StubMixin, unittest.TestCase):
    def test_srt_uses_comma_fractions_and_one_based_cues(self):
        body = self._post(response_format="srt").get_data(as_text=True)
        lines = body.splitlines()
        self.assertEqual(lines[0], "1")
        self.assertEqual(lines[1], "00:00:00,000 --> 00:00:01,500")
        self.assertEqual(lines[2], "hello there")
        self.assertEqual(lines[4], "2")
        self.assertEqual(lines[5], "00:00:01,500 --> 00:00:04,250")

    def test_vtt_uses_a_header_and_dot_fractions(self):
        body = self._post(response_format="vtt").get_data(as_text=True)
        self.assertTrue(body.startswith("WEBVTT\n"))
        self.assertIn("00:00:01.500 --> 00:00:04.250", body)
        self.assertNotIn(",", body.split("\n")[2])

    def test_markdown_carries_a_heading_and_minute_stamps(self):
        body = self._post(response_format="markdown").get_data(as_text=True)
        self.assertTrue(body.startswith("# Transcript"))
        self.assertIn("**[00:00]** hello there", body)
        self.assertIn("**[00:01]** general kenobi", body)

    def test_stamp_rounding_does_not_produce_a_1000ms_field(self):
        self.assertEqual(tsrv._stamp(1.9999, True), "00:00:02,000")


class VramDisciplineTests(_StubMixin, unittest.TestCase):
    def test_loading_a_second_model_releases_the_first(self):
        manager = tsrv.ModelManager(self.cfg)
        manager.acquire("stub", "preset:a")
        manager.release()
        self.assertEqual(StubEngine.loads, 1)
        manager.acquire("stub", "preset:b")
        manager.release()
        self.assertEqual(StubEngine.unloads, 1)
        self.assertEqual(StubEngine.loads, 2)

    def test_the_same_model_is_not_reloaded(self):
        manager = tsrv.ModelManager(self.cfg)
        manager.acquire("stub", "preset:a")
        manager.release()
        manager.acquire("stub", "preset:a")
        manager.release()
        self.assertEqual(StubEngine.loads, 1)

    def test_an_idle_model_is_released_on_its_own(self):
        cfg = dict(self.cfg)
        cfg["limits"] = dict(cfg["limits"], idle_unload_seconds=0.15)
        manager = tsrv.ModelManager(cfg)
        manager.start_idle_thread()
        try:
            manager.acquire("stub", "preset:a")
            manager.release()
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and manager.snapshot()["resident"] is not None:
                time.sleep(0.05)
            self.assertIsNone(manager.snapshot()["resident"])
        finally:
            manager.stop()

    def test_the_idle_thread_never_evicts_a_running_decode(self):
        cfg = dict(self.cfg)
        cfg["limits"] = dict(cfg["limits"], idle_unload_seconds=0.1)
        manager = tsrv.ModelManager(cfg)
        manager.start_idle_thread()
        try:
            manager.acquire("stub", "preset:a")   # acquire without release: in flight
            time.sleep(0.6)
            self.assertIsNotNone(manager.snapshot()["resident"])
            self.assertEqual(StubEngine.unloads, 0)
            manager.release()
        finally:
            manager.stop()

    def test_a_failed_load_releases_what_it_allocated(self):
        """A load that dies partway has usually allocated something already.

        `_resident` is never assigned in that case, so nothing else will free
        it — the memory stays pinned until the process exits and the next
        attempt fails for the same reason with less room than before.
        """
        released = []

        class HalfLoadingEngine(StubEngine):
            def load(self, model_ref):
                raise tsrv.ModelLoadFailed("CUDA out of memory")

            def unload(self):
                released.append(True)

        tsrv.ENGINES["half"] = HalfLoadingEngine
        self.addCleanup(tsrv.ENGINES.pop, "half", None)
        manager = tsrv.ModelManager(self.cfg)
        with self.assertRaises(tsrv.ModelLoadFailed):
            manager.acquire("half", "preset:a")
        self.assertEqual(released, [True], "the half-loaded engine was not unloaded")
        self.assertIsNone(manager.snapshot()["resident"])

    def test_a_failed_load_leaves_the_manager_usable(self):
        class BrokenEngine(StubEngine):
            def load(self, model_ref):
                raise tsrv.ModelLoadFailed("nope")

        tsrv.ENGINES["broken"] = BrokenEngine
        self.addCleanup(tsrv.ENGINES.pop, "broken", None)
        manager = tsrv.ModelManager(self.cfg)
        with self.assertRaises(tsrv.ModelLoadFailed):
            manager.acquire("broken", "preset:a")
        engine = manager.acquire("stub", "preset:a")   # must still work
        manager.release()
        self.assertIsNotNone(engine)

    def test_unload_reports_whether_anything_was_resident(self):
        manager = tsrv.ModelManager(self.cfg)
        self.assertFalse(manager.unload())
        manager.acquire("stub", "preset:a")
        manager.release()
        self.assertTrue(manager.unload())

    def test_the_unload_route_frees_the_model(self):
        self._post()
        self.assertIsNotNone(self.client.get("/engines").get_json()["resident"])
        self.client.post("/unload")
        self.assertIsNone(self.client.get("/engines").get_json()["resident"])


class VramBudgetTests(_StubMixin, unittest.TestCase):
    """A window size moves peak usage; only a budget bounds it."""

    def _manager_with_budget(self, budget_mb, recorder):
        cfg = dict(self.cfg)
        cfg["limits"] = dict(cfg["limits"], max_vram_mb=budget_mb)
        fake = type(sys)("torch")
        fake.cuda = type(sys)("torch.cuda")
        fake.cuda.is_available = lambda: True
        fake.cuda.empty_cache = lambda: None
        fake.cuda.ipc_collect = lambda: None
        fake.cuda.set_per_process_memory_fraction = lambda f, d=0: recorder.append((f, d))
        fake.cuda.get_device_properties = lambda i: type(
            "P", (), {"total_memory": 24 * 1024 * 1024 * 1024})()
        sys.modules["torch"] = fake
        self.addCleanup(sys.modules.pop, "torch", None)
        return tsrv.ModelManager(cfg)

    def test_a_budget_caps_the_allocator(self):
        calls = []
        self._manager_with_budget(2500, calls).acquire("stub", "preset:a")
        self.assertEqual(len(calls), 1)
        fraction, device = calls[0]
        # The CUDA context is outside torch's allocator, so it comes off the top.
        expected = (2500 - tsrv.CUDA_CONTEXT_MB) / (24 * 1024)
        self.assertAlmostEqual(fraction, expected, places=4)
        self.assertEqual(device, 0)

    def test_zero_means_no_budget(self):
        calls = []
        self._manager_with_budget(0, calls).acquire("stub", "preset:a")
        self.assertEqual(calls, [])

    def test_a_budget_below_the_context_cost_is_refused_not_applied(self):
        """Setting a fraction of zero would fail every allocation instead."""
        calls = []
        self._manager_with_budget(tsrv.CUDA_CONTEXT_MB - 50, calls).acquire("stub", "preset:a")
        self.assertEqual(calls, [])

    def test_a_budget_never_blocks_a_load_when_torch_is_absent(self):
        cfg = dict(self.cfg)
        cfg["limits"] = dict(cfg["limits"], max_vram_mb=2500)
        sys.modules["torch"] = None      # import raises
        self.addCleanup(sys.modules.pop, "torch", None)
        manager = tsrv.ModelManager(cfg)
        self.assertIsNotNone(manager.acquire("stub", "preset:a"))


class RouterYieldTests(_StubMixin, unittest.TestCase):
    class _FakeResponse:
        def __init__(self, status=200, text="", payload=None):
            self.status_code, self.text, self._payload = status, text, payload or {}

        def json(self):
            return self._payload

    def _manager(self, mode, recorder):
        cfg = dict(self.cfg)
        cfg["router"] = dict(cfg["router"], yield_mode=mode)
        manager = tsrv.ModelManager(cfg)
        fake = type(sys)("requests")
        fake.post = lambda url, **kw: recorder.append((url, kw)) or self._FakeResponse()
        fake.get = lambda url, **kw: self._FakeResponse(
            payload={"data": [{"id": "ocr", "status": {"value": "loaded"}},
                               {"id": "task", "status": {"value": "unloaded"}}]})
        sys.modules["requests"] = fake
        self.addCleanup(sys.modules.pop, "requests", None)
        return manager

    def test_yield_off_sends_nothing(self):
        calls = []
        self._manager("off", calls)._router_yield()
        self.assertEqual(calls, [])

    def test_yield_asr_unloads_only_the_audio_model(self):
        calls = []
        self._manager("asr", calls)._router_yield()
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0][0].endswith("/models/unload"))
        self.assertEqual(calls[0][1]["json"], {"model": "asr"})

    def test_yield_all_unloads_only_resident_models(self):
        calls = []
        self._manager("all", calls)._router_yield()
        self.assertEqual([c[1]["json"]["model"] for c in calls], ["ocr"])

    def test_a_model_that_is_not_running_is_not_a_failure(self):
        """400 'model is not running' is the common path, not an error."""
        manager = self._manager("asr", [])
        sys.modules["requests"].post = lambda url, **kw: self._FakeResponse(
            status=400, text="model is not running")
        with self.assertLogs(tsrv.log, level="WARNING") as captured:
            tsrv.log.warning("sentinel")   # assertLogs needs at least one record
            manager._router_yield()
        self.assertEqual([r for r in captured.output if "not running" in r], [])

    def test_a_missing_requests_module_does_not_fail_the_request(self):
        """Yielding is an optimisation; it must never mask a real answer."""
        cfg = dict(self.cfg)
        cfg["router"] = dict(cfg["router"], yield_mode="asr")
        manager = tsrv.ModelManager(cfg)
        sys.modules["requests"] = None       # import raises
        self.addCleanup(sys.modules.pop, "requests", None)
        manager._router_yield()              # must not raise
        engine = manager.acquire("stub", "preset:a")
        manager.release()
        self.assertIsNotNone(engine)


class RouterEngineTests(_StubMixin, unittest.TestCase):
    def test_the_router_engine_reports_no_timeline(self):
        self.assertFalse(tsrv.ENGINES["router"].capabilities["segments"])
        self.assertFalse(tsrv.ENGINES["router"].capabilities["word_timestamps"])

    def test_subtitles_are_refused_rather_than_fabricated(self):
        for fmt in ("srt", "vtt", "verbose_json"):
            resp = self._post(engine="router", response_format=fmt)
            self.assertEqual(resp.status_code, 422, fmt)
            self.assertEqual(resp.get_json()["error"]["type"], "unsupported_capability")

    def test_degraded_output_can_be_opted_into(self):
        cfg = _cfg()
        cfg["router"] = dict(cfg["router"], allow_degraded="on")
        tsrv.check_timeline_support(tsrv.ENGINES["router"].capabilities, "srt", cfg, "router")

    def test_plain_json_is_always_allowed(self):
        cfg = _cfg()
        tsrv.check_timeline_support(tsrv.ENGINES["router"].capabilities, "json", cfg, "router")


class UrlFetchTests(_StubMixin, unittest.TestCase):
    def test_a_blank_allow_list_denies_every_fetch(self):
        resp = self.client.post("/transcribe", data={"url": "http://example.com/a.wav"})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("not allowed", resp.get_json()["error"]["message"])

    def test_no_audio_at_all_is_a_clear_error(self):
        resp = self.client.post("/transcribe", data={})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("no audio supplied", resp.get_json()["error"]["message"])


class AuthTests(unittest.TestCase):
    def setUp(self):
        cfg = _cfg()
        cfg["server"]["token"] = "s3cret"
        cfg["engines"]["stub"] = {"model": "preset:stub", "backend_type": "local"}
        cfg["active_engine"] = "stub"
        tsrv.ENGINES["stub"] = StubEngine
        self.addCleanup(tsrv.ENGINES.pop, "stub", None)
        self.client = tsrv.create_app(cfg=cfg).test_client()

    def test_health_never_needs_a_token(self):
        """Otherwise the health probe reports a working service as down."""
        self.assertEqual(self.client.get("/health").status_code, 200)

    def test_a_missing_token_is_rejected(self):
        self.assertEqual(self.client.get("/engines").status_code, 401)

    def test_a_bearer_token_is_accepted(self):
        resp = self.client.get("/engines", headers={"Authorization": "Bearer s3cret"})
        self.assertEqual(resp.status_code, 200)

    def test_the_error_shape_matches_the_endpoint_family(self):
        self.assertIn("error", self.client.post("/v1/audio/transcriptions").get_json())
        self.assertIs(self.client.post("/transcribe").get_json()["ok"], False)


class AsyncThresholdTests(_StubMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self._real_probe = tsrv.probe_duration
        tsrv.probe_duration = lambda path: 1200.0
        self.addCleanup(setattr, tsrv, "probe_duration", self._real_probe)

    def test_long_audio_returns_a_job_from_the_native_route(self):
        resp = self._post()
        self.assertEqual(resp.status_code, 202)
        body = resp.get_json()
        self.assertIn("job_id", body)
        self.assertEqual(body["poll"], f"/jobs/{body['job_id']}")

    def test_long_audio_is_refused_by_the_openai_route(self):
        """An SDK cannot poll, so a job id would be useless to it."""
        resp = self._post("/v1/audio/transcriptions")
        self.assertEqual(resp.status_code, 413)
        self.assertIn("/transcribe", resp.get_json()["error"]["message"])

    def test_a_job_runs_to_completion_and_can_be_read_back(self):
        job_id = self._post().get_json()["job_id"]
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            body = self.client.get(f"/jobs/{job_id}").get_json()
            if body["status"] in ("done", "error"):
                break
            time.sleep(0.05)
        self.assertEqual(body["status"], "done")
        self.assertEqual(body["result"]["text"], "hello there general kenobi")

    def test_an_unknown_job_is_a_404(self):
        self.assertEqual(self.client.get("/jobs/nope").status_code, 404)


class DurationFallbackTests(_StubMixin, unittest.TestCase):
    """An engine that reports no duration must not report zero.

    NeMo returns text and, unless timestamps are requested, no timeline at all.
    Chaining the fallback off the last segment's end then reports a real
    transcription as 0.0 seconds of audio at a realtime factor of 0 — which is
    exactly the number someone benchmarking the engine would read.
    """

    class SilentEngine(StubEngine):
        def transcribe(self, path, req):
            return {"segments": [{"id": 0, "start": 0.0, "end": 0.0, "text": "hello",
                                  "speaker": None, "avg_logprob": 0.0, "no_speech_prob": 0.0,
                                  "compression_ratio": 0.0, "words": []}],
                    "language": "", "language_probability": 0.0, "duration": 0.0}

    def setUp(self):
        super().setUp()
        tsrv.ENGINES["stub"] = self.SilentEngine
        self._real_probe = tsrv.probe_duration
        tsrv.probe_duration = lambda path: 42.0
        self.addCleanup(setattr, tsrv, "probe_duration", self._real_probe)

    def test_duration_falls_back_to_the_probe(self):
        body = self._post().get_json()
        self.assertEqual(body["duration"], 42.0)
        self.assertEqual(body["timings"]["audio_seconds"], 42.0)

    def test_realtime_factor_is_meaningful(self):
        self.assertGreater(self._post().get_json()["timings"]["realtime_factor"], 0)


class ProbeDurationTests(unittest.TestCase):
    def test_pyav_answers_when_ffprobe_is_off_the_path(self):
        """The unit's PATH is systemd's, so a brew/opt ffmpeg is invisible to
        it. PyAV is a library in the venv and needs no PATH at all."""
        import shutil as _shutil
        real_which = _shutil.which
        tsrv.shutil.which = lambda name: None      # simulate the service's PATH
        self.addCleanup(setattr, tsrv.shutil, "which", real_which)
        fixture = pathlib.Path(__file__).resolve().parents[1] / "deps/llama.cpp/tools/mtmd/test-2.mp3"
        if not fixture.exists():
            self.skipTest("llama.cpp audio fixture not present")
        try:
            import av  # noqa: F401
        except Exception:
            self.skipTest("PyAV not installed in this environment")
        self.assertAlmostEqual(tsrv.probe_duration(str(fixture)), 17.4, delta=0.5)

    def test_an_unreadable_file_reports_unknown_rather_than_guessing(self):
        """Unknown runs synchronously, which is what the caller asked for."""
        self.assertIsNone(tsrv.probe_duration("/nonexistent/file.wav"))

    def test_a_wav_header_is_read_without_ffprobe(self):
        import struct
        import tempfile
        import wave
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            with wave.open(handle.name, "wb") as writer:
                writer.setnchannels(1)
                writer.setsampwidth(2)
                writer.setframerate(16000)
                writer.writeframes(struct.pack("<h", 0) * 32000)
            self.assertAlmostEqual(tsrv.probe_duration(handle.name), 2.0, places=3)


if __name__ == "__main__":
    unittest.main()
