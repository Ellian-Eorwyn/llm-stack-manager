from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import unittest
from unittest.mock import patch


def _load_app_module():
    root = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "llm_stack_manager_app_public", root / "web" / "app.py")
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


manager = _load_app_module()
core = sys.modules["core"]
config_env = sys.modules["config_env"]
public_api = sys.modules["public_api"]
telemetry = sys.modules["telemetry"]
public_routes = sys.modules["routes.public"]


# A two-GPU box with the primary chat backend and the router both resident, which
# is the arrangement the VRAM attribution has to get right: one process it can
# name a model for, one it cannot split.
GPUS = [
    {
        "index": 0, "uuid": "GPU-aaa", "name": "RTX 5090",
        "mem_used": 30000, "mem_total": 32607, "mem_free": 2607,
        "util": 74, "temp": 61, "mem_pct": 92, "mem_util": 40,
        "power_watts": 402, "power_limit_watts": 575, "fan_pct": 63,
        "clock_sm_mhz": 2610, "clock_mem_mhz": 14001, "pstate": "P0",
        # The router runs one child per resident model, all under its own unit,
        # each naming its own --model. Two of these are that.
        "processes": [
            {"pid": 1001, "name": "chat-backend-dense", "process_name": "llama-server",
             "used_memory": 26000, "model": "/models/glm-4.6-Q4_K_M.gguf", "alias": "chat-dense"},
            {"pid": 1002, "name": "llama-router", "process_name": "llama-server",
             "used_memory": 2500, "model": "/models/nomic-embed.gguf", "alias": "embed"},
            {"pid": 1004, "name": "llama-router", "process_name": "llama-server",
             "used_memory": 1000, "model": "/models/qwen3-4b-task.gguf", "alias": "task"},
            {"pid": 1003, "name": "some-other-thing", "process_name": "blender",
             "used_memory": 500, "model": "", "alias": ""},
        ],
    },
    {
        "index": 1, "uuid": "GPU-bbb", "name": "RTX 3090",
        "mem_used": 800, "mem_total": 24576, "mem_free": 23776,
        "util": 0, "temp": 34, "mem_pct": 3, "mem_util": 0,
        "power_watts": 22, "power_limit_watts": 350, "fan_pct": 30,
        "clock_sm_mhz": 210, "clock_mem_mhz": 405, "pstate": "P8",
        "processes": [],
    },
]

PROPS = {
    "model_path": "/models/glm-4.6-Q4_K_M.gguf",
    "model_alias": "glm-4.6",
    "model_ftype": "Q4_K_M",
    "build_info": "b1234",
    "total_slots": 2,
    "n_ctx_per_slot": 131072,
    "n_ctx_total": 262144,
    "modalities": {"vision": False},
    "is_sleeping": False,
}

SLOTS = [
    {"id": 0, "n_ctx": 131072, "is_processing": True, "n_prompt_tokens": 118000,
     "n_prompt_tokens_cache": 90000, "speculative": True, "ctx_pct": 90.0},
    {"id": 1, "n_ctx": 131072, "is_processing": False, "n_prompt_tokens": 12000,
     "n_prompt_tokens_cache": 8000, "speculative": True, "ctx_pct": 9.2},
]

ENV = {
    "CHAT_BACKEND_PORT": "8010", "CHAT_BACKEND_HOST": "127.0.0.1",
    "CHAT_PRIMARY_MODEL_PATH": "/models/glm-4.6-Q4_K_M.gguf",
    "CHAT_PRIMARY_CTX_SIZE": "262144", "CHAT_PRIMARY_N_PARALLEL": "2",
    "CHAT_PRIMARY_FLASH_ATTN": "on", "CHAT_PRIMARY_CACHE_TYPE_K": "q8_0",
    "MODEL_ROUTER_ENABLED": "on", "MODEL_ROUTER_MAX": "1",
    "LLM_API_HOST": "127.0.0.1", "LLM_API_PORT": "8078", "LLM_API_TOKEN": "",
    "LLM_API_ENABLED": "on", "LLM_API_STREAM_INTERVAL": "2",
    "LLM_API_ALLOW_ORIGINS": "", "LLM_API_WEBHOOK_URL": "",
    "LLM_API_WEBHOOK_EVENTS": "service_state,alert",
    # The reason the redaction tests exist.
    "HF_TOKEN": "hf_secretvalue", "NEO4J_PASSWORD": "hunter2",
    "OPENAI_API_KEY": "sk-do-not-leak", "GLMOCR_SDK_AUTH_SECRET": "shh",
}

HOST_MEM = {
    "mem_total_mib": 128000, "mem_available_mib": 40000, "mem_used_mib": 88000,
    "mem_used_pct": 69, "mem_available_pct": 31,
    "swap_total_mib": 8192, "swap_used_mib": 0, "swap_used_pct": 0,
    "swap_activity": {"available": True, "active": False},
}

SERVICE_HEALTH = {
    "chat-backend-dense": {"state": "active", "unit": "active", "expected": "on", "reason": "",
                           "probe": {"ok": True, "checked_at": 1.0}, "upstreams": [], "restarts": 0,
                           "checked_at": 1.0},
    "embed": {"state": "degraded", "unit": "active", "expected": "on",
              "reason": "probe failed: connection refused", "probe": {"ok": False}, "upstreams": [],
              "restarts": 0, "checked_at": 1.0},
    "llama-router": {"state": "active", "unit": "active", "expected": "on", "reason": "",
                     "probe": {"ok": True}, "upstreams": [], "restarts": 0, "checked_at": 1.0},
    "chat-proxy": {"state": "failed", "unit": "failed", "expected": "on", "reason": "",
                   "probe": None, "upstreams": [], "restarts": 3, "checked_at": 1.0},
}

ROUTER = {
    "ok": True, "enabled": True, "reachable": True, "max_resident": "1",
    "models": [
        {"id": "embed", "state": "loaded", "path": "/models/nomic-embed.gguf"},
        {"id": "rank", "state": "unloaded", "path": "/models/bge-reranker.gguf"},
    ],
}


def build_providers(**overrides) -> public_api.Providers:
    """A Providers bundle that needs no systemd, no GPU and no backend."""
    defaults = dict(
        read_env=lambda: dict(ENV),
        service_status=lambda name: "active" if name in {"chat-backend-dense", "llama-router"} else "inactive",
        service_health=lambda env: ({}, dict(SERVICE_HEALTH)),
        gpu_info=lambda: [dict(g) for g in GPUS],
        context_summary=lambda env: {
            "chat-backend-dense": {"total_context": 262144, "slots": 2, "per_slot_context": 131072}},
        deployment=lambda: {"summary": "up to date", "behind": 0, "dirty": False},
        router_overview=lambda env: dict(ROUTER),
        services_table=lambda env: [
            {"name": "chat-backend-dense", "label": "Primary Backend", "group": "chat", "desc": "Primary model backend"},
            {"name": "embed", "label": "Embedding", "group": "auxiliary", "desc": "Embedding model"},
            {"name": "llama-router", "label": "Model Router", "group": "auxiliary", "desc": "Loads on demand"},
            {"name": "chat-proxy", "label": "Primary Proxy", "group": "chat", "desc": "Routes think/chat/code"},
        ],
    )
    defaults.update(overrides)
    return public_api.Providers(**defaults)


class _TelemetryStub:
    """Stands in for the probes and the journal, so nothing here touches a socket."""

    def __enter__(self):
        self._patches = [
            patch.object(telemetry, "probe_props", return_value=dict(PROPS)),
            patch.object(telemetry, "probe_slots", return_value=[dict(s) for s in SLOTS]),
            patch.object(telemetry, "probe_metrics", return_value=({"llamacpp:n_decode_total": 5.0}, True)),
            patch.object(telemetry, "summarize", return_value={
                "throughput": {"generation_tps": {"p50": 41.2, "p90": 52.0}, "last_generation_tps": 44.1},
                "cache": {"launches": 100, "evictions": 40, "evictions_per_launch": 0.4},
                "scheduling": {"select_to_launch_seconds": {"p90": 1.4, "p99": 2.0}, "select_methods": {}},
                "context": {"released_tokens": 0, "overflow_count": 1,
                            "overflows": [{"requested": 155751, "available": 131072}]},
            }),
            patch.object(core, "read_meminfo", return_value={}),
            patch.object(telemetry, "host_memory", return_value=dict(HOST_MEM)),
        ]
        for item in self._patches:
            item.start()
        # A real collector would spawn a journalctl tailer per unit.
        self._registry = patch.object(
            telemetry.REGISTRY, "collector",
            return_value=type("C", (), {"snapshot": staticmethod(lambda: []), "error": None})())
        self._registry.start()
        return self

    def __exit__(self, *exc):
        self._registry.stop()
        for item in reversed(self._patches):
            item.stop()
        return False


class ContextRollupTests(unittest.TestCase):
    def test_sums_tokens_across_slots_and_reports_the_fullest(self):
        rollup = public_api.context_rollup(
            {"props": dict(PROPS), "slots": [dict(s) for s in SLOTS]},
            {"total_context": 262144, "slots": 2, "per_slot_context": 131072})
        self.assertEqual(rollup["used_tokens"], 130000)
        self.assertEqual(rollup["cached_tokens"], 98000)
        self.assertEqual(rollup["n_ctx_total"], 262144)
        self.assertEqual(rollup["slots_total"], 2)
        self.assertEqual(rollup["slots_busy"], 1)
        self.assertEqual(rollup["free_tokens"], 132144)
        self.assertAlmostEqual(rollup["used_pct"], 49.6, places=1)
        # The backend is half full, but one slot is nearly out. That slot is the
        # one that will reject the next request, so it has to be reported.
        self.assertEqual(rollup["max_slot_pct"], 90.0)

    def test_a_stopped_backend_rolls_up_to_empty_rather_than_crashing(self):
        rollup = public_api.context_rollup({"props": None, "slots": None}, None)
        self.assertEqual(rollup["used_tokens"], 0)
        self.assertIsNone(rollup["n_ctx_total"])
        self.assertIsNone(rollup["used_pct"])
        self.assertIsNone(rollup["max_slot_pct"])
        self.assertEqual(rollup["slots_busy"], 0)

    def test_total_context_is_derived_when_props_omits_it(self):
        props = dict(PROPS, n_ctx_total=None)
        rollup = public_api.context_rollup({"props": props, "slots": [dict(s) for s in SLOTS]}, None)
        self.assertEqual(rollup["n_ctx_total"], 262144)


class VramAttributionTests(unittest.TestCase):
    def test_a_backend_model_is_attributed_to_the_unit_serving_it(self):
        backends = [{"name": "chat-primary", "unit": "chat-backend-dense", "props": dict(PROPS)}]
        blocks = public_api.gpu_model_blocks(GPUS[0], backends, ROUTER)
        block = next(b for b in blocks if b["unit"] == "chat-backend-dense")
        self.assertEqual(block["vram_mib"], 26000)
        self.assertEqual(block["model"], "/models/glm-4.6-Q4_K_M.gguf")
        self.assertEqual(block["attribution"], "exclusive")

    def test_each_router_child_gets_its_own_model_block(self):
        # Grouping by unit alone would merge these into one 3500 MiB block for
        # `llama-router` and lose which model the VRAM is actually going to.
        blocks = public_api.gpu_model_blocks(GPUS[0], [], ROUTER)
        router_blocks = {b["model_alias"]: b for b in blocks if b["unit"] == "llama-router"}
        self.assertEqual(set(router_blocks), {"embed", "task"})
        self.assertEqual(router_blocks["embed"]["vram_mib"], 2500)
        self.assertEqual(router_blocks["task"]["vram_mib"], 1000)
        self.assertEqual(router_blocks["embed"]["model"], "/models/nomic-embed.gguf")
        for block in router_blocks.values():
            self.assertEqual(block["attribution"], "router")

    def test_router_residency_is_reported_against_the_routers_own_view(self):
        blocks = public_api.gpu_model_blocks(GPUS[0], [], ROUTER)
        by_alias = {b["model_alias"]: b for b in blocks if b["unit"] == "llama-router"}
        # ROUTER reports embed loaded and rank unloaded; task is holding VRAM
        # without the router calling it resident, which is worth being able to see.
        self.assertTrue(by_alias["embed"]["router_resident"])
        self.assertFalse(by_alias["task"]["router_resident"])

    def test_a_process_with_no_identifiable_model_is_still_reported(self):
        blocks = public_api.gpu_model_blocks(GPUS[0], [], ROUTER)
        block = next(b for b in blocks if b["unit"] == "some-other-thing")
        self.assertEqual(block["attribution"], "unattributed")
        self.assertIsNone(block["model"])
        self.assertEqual(block["vram_mib"], 500)

    def test_blocks_are_ordered_by_the_vram_they_hold(self):
        blocks = public_api.gpu_model_blocks(GPUS[0], [], None)
        self.assertEqual([b["vram_mib"] for b in blocks], [26000, 2500, 1000, 500])

    def test_a_gpu_between_tokens_still_reads_as_busy(self):
        idle_core = dict(GPUS[0], util=0, mem_util=38)
        annotated = public_api.annotate_gpus([idle_core], [], None)
        self.assertTrue(annotated[0]["busy"])

    def test_a_genuinely_idle_gpu_reads_as_idle(self):
        annotated = public_api.annotate_gpus([dict(GPUS[1])], [], None)
        self.assertFalse(annotated[0]["busy"])


class GpuProcessLabellingTests(unittest.TestCase):
    """Regression cover for the NameError that silently emptied every label."""

    def test_the_ocr_sdk_is_not_mistaken_for_the_ocr_backend(self):
        # "ocr" occurs inside "glmocr". A plain substring test moved the SDK's
        # VRAM onto the OCR model's row, because `ocr` comes first in SERVICES.
        with (patch.object(manager, "process_unit", return_value=""),
              patch.object(manager, "process_cmdline",
                           return_value="/usr/bin/python3 /opt/stack/scripts/glmocr-sdk-server.py --port 5002")):
            label = manager.label_gpu_process(4242, "/usr/bin/python3", {})
        self.assertEqual(label, "glmocr-sdk")

    def test_the_ocr_backend_is_still_matched_on_its_own_name(self):
        with (patch.object(manager, "process_unit", return_value=""),
              patch.object(manager, "process_cmdline",
                           return_value="/opt/llama.cpp/llama-server --alias ocr --port 8009")):
            self.assertEqual(manager.label_gpu_process(4243, "llama-server", {}), "ocr")

    def test_an_unmanaged_python_process_falls_back_to_its_script(self):
        # Not a service the manager runs, but it is holding VRAM, so it needs a
        # name more useful than "python3".
        with (patch.object(manager, "process_unit", return_value=""),
              patch.object(manager, "process_cmdline",
                           return_value="/usr/bin/python3 /home/ellie/train_lora.py --epochs 3")):
            label = manager.label_gpu_process(4244, "/usr/bin/python3", {})
        self.assertEqual(label, "train_lora.py")

    def test_a_known_unit_wins_over_any_guess(self):
        with patch.object(manager, "process_unit", return_value=""):
            self.assertEqual(manager.label_gpu_process(7, "llama-server", {7: "chat-backend-dense"}),
                             "chat-backend-dense")

    def test_a_router_child_is_attributed_by_its_cgroup_not_its_parent_pid(self):
        # The router forks a llama-server per resident model. None of them is the
        # unit's MainPID, so a MainPID map alone never finds them.
        with patch.object(manager, "process_unit", return_value="llama-router"):
            self.assertEqual(manager.label_gpu_process(3406554, "llama-server", {}), "llama-router")

    def test_a_cgroup_naming_something_that_is_not_a_managed_service_is_ignored(self):
        # A desktop process sits in user@1000.service; that is not an answer.
        with (patch.object(manager, "process_unit", return_value="user@1000"),
              patch.object(manager, "process_cmdline", return_value="/usr/bin/blender")):
            self.assertEqual(manager.label_gpu_process(50, "blender", {}), "blender")

    def test_a_launcher_in_the_cmdline_names_the_service(self):
        with patch.object(manager, "process_cmdline",
                          return_value="/bin/bash /opt/stack/scripts/start-embed.sh"):
            self.assertEqual(manager.label_gpu_process(9, "bash", {}), "embed")

    def test_attribution_survives_a_process_that_has_already_exited(self):
        # /proc/<pid> disappearing mid-read is normal, not exceptional.
        with (patch.object(manager, "process_unit", return_value=""),
              patch.object(manager, "process_cmdline", return_value="")):
            self.assertEqual(manager.label_gpu_process(11, "llama-server", {}), "llama-server")

    def test_the_unit_is_read_from_the_cgroup_of_a_real_process(self):
        # This process is in some unit or session; the point is that reading it
        # neither raises nor invents a name.
        self.assertIsInstance(manager.process_unit(1), str)
        self.assertEqual(manager.process_unit(-1), "")
        self.assertEqual(manager.process_unit(999999999), "")


class ProcessModelArgsTests(unittest.TestCase):
    def test_model_and_alias_are_pulled_from_a_llama_server_command_line(self):
        cmdline = ("/opt/llama.cpp/llama-server --host 127.0.0.1 --jinja --alias task "
                   "--ctx-size 65538 --model /models/Qwen3.5-4B-UD-Q6_K_XL.gguf --parallel 1")
        self.assertEqual(manager.process_model_args(cmdline),
                         ("/models/Qwen3.5-4B-UD-Q6_K_XL.gguf", "task"))

    def test_a_command_line_without_them_yields_empties(self):
        self.assertEqual(manager.process_model_args("/usr/bin/blender --background"), ("", ""))

    def test_a_trailing_flag_with_no_value_does_not_raise(self):
        self.assertEqual(manager.process_model_args("llama-server --model"), ("", ""))


class ConfigDriftTests(unittest.TestCase):
    def test_a_context_change_that_has_not_been_restarted_into_is_reported(self):
        backend = {"unit": "chat-backend-dense", "props": dict(PROPS)}
        drift = public_api.backend_drift(
            backend, dict(ENV), {"per_slot_context": 65536, "total_context": 131072})
        self.assertEqual(len(drift), 1)
        self.assertEqual(drift[0]["field"], "n_ctx_per_slot")
        self.assertEqual(drift[0]["running"], 131072)
        self.assertEqual(drift[0]["configured"], 65536)

    def test_a_swapped_model_is_reported(self):
        env = dict(ENV, CHAT_PRIMARY_MODEL_PATH="/models/qwen3-next-Q5.gguf")
        drift = public_api.backend_drift(
            {"unit": "chat-backend-dense", "props": dict(PROPS)}, env,
            {"per_slot_context": 131072})
        self.assertEqual([d["field"] for d in drift], ["model_path"])

    def test_matching_config_reports_nothing(self):
        drift = public_api.backend_drift(
            {"unit": "chat-backend-dense", "props": dict(PROPS)}, dict(ENV),
            {"per_slot_context": 131072})
        self.assertEqual(drift, [])

    def test_the_launcher_fallback_chain_is_honoured(self):
        # start-chat-backend-dense.sh falls back through three keys; reading only
        # the first would report drift against a backend doing as it was told.
        env = {"CHAT_MODEL_PATH": "/models/glm-4.6-Q4_K_M.gguf"}
        self.assertEqual(public_api.configured_model_path(env, "chat-backend-dense"),
                         "/models/glm-4.6-Q4_K_M.gguf")


class RedactionTests(unittest.TestCase):
    def test_no_secret_survives_the_config_section(self):
        config = public_api.redacted_config(dict(ENV))
        flat = json.dumps(config)
        for secret in ("hf_secretvalue", "hunter2", "sk-do-not-leak", "shh"):
            self.assertNotIn(secret, flat)
        for key in ("HF_TOKEN", "NEO4J_PASSWORD", "OPENAI_API_KEY", "GLMOCR_SDK_AUTH_SECRET"):
            self.assertNotIn(key, flat)

    def test_the_launch_settings_that_explain_behaviour_do_survive(self):
        config = public_api.redacted_config(dict(ENV))
        self.assertEqual(config["Primary Backend"]["CHAT_PRIMARY_CTX_SIZE"], "262144")
        self.assertEqual(config["Primary Backend"]["CHAT_PRIMARY_FLASH_ATTN"], "on")

    def test_a_secret_added_to_an_allowed_section_is_still_dropped(self):
        # The allow-list is the intended control; this is the one that has to
        # hold when someone extends it without thinking about who reads it.
        extra = {"section": "Model Router", "key": "MODEL_ROUTER_API_KEY",
                 "label": "Router Key", "type": "text"}
        with patch.object(public_api.config_fields, "CONFIG_FIELDS",
                          list(public_api.config_fields.CONFIG_FIELDS) + [extra]):
            config = public_api.redacted_config(dict(ENV, MODEL_ROUTER_API_KEY="leak-me"))
        self.assertNotIn("leak-me", json.dumps(config))

    def test_every_allow_listed_key_passes_the_secret_pattern(self):
        for field in public_api._config_allowlist():
            self.assertIsNone(public_api.SECRET_KEY_RE.search(field["key"]), field["key"])


class AlertTests(unittest.TestCase):
    def setUp(self):
        with _TelemetryStub():
            self.payload = public_api.snapshot(build_providers())
        self.codes = {alert["code"] for alert in self.payload["alerts"]}

    def test_conditions_carry_stable_codes(self):
        self.assertIn("cache_thrash", self.codes)
        self.assertIn("slot_delay", self.codes)
        self.assertIn("ctx_overflow", self.codes)
        self.assertIn("context_high", self.codes)
        self.assertIn("service_degraded", self.codes)
        self.assertIn("service_flapping", self.codes)

    def test_a_flapping_service_outranks_its_failed_state(self):
        flapping = [a for a in self.payload["alerts"] if a["code"] == "service_flapping"]
        self.assertEqual(len(flapping), 1)
        self.assertEqual(flapping[0]["subject"], "chat-proxy")
        self.assertNotIn("service_failed", {a["code"] for a in self.payload["alerts"]})

    def test_errors_sort_above_warnings(self):
        levels = [alert["level"] for alert in self.payload["alerts"]]
        self.assertEqual(levels, sorted(levels, key=lambda l: {"error": 0, "warn": 1, "info": 2}[l]))

    def test_every_emitted_code_is_documented_in_the_schema(self):
        documented = set(public_api.schema()["alert_codes"])
        self.assertEqual(self.codes - documented, set())

    def test_an_open_bind_is_reported_rather_than_being_silent(self):
        with _TelemetryStub():
            payload = public_api.snapshot(build_providers(), bind_warning="bound wide open")
        codes = {a["code"] for a in payload["alerts"]}
        self.assertIn("api_unauthenticated", codes)

    def test_a_healthy_stack_raises_nothing(self):
        providers = build_providers(
            service_health=lambda env: ({}, {"chat-backend-dense": dict(SERVICE_HEALTH["chat-backend-dense"])}),
            gpu_info=lambda: [dict(GPUS[1])],
            services_table=lambda env: [{"name": "chat-backend-dense", "label": "Primary Backend",
                                         "group": "chat", "desc": ""}],
        )
        quiet_slots = [dict(SLOTS[1], id=0), dict(SLOTS[1], id=1)]
        with _TelemetryStub():
            with (patch.object(telemetry, "summarize", return_value={
                      "throughput": {}, "cache": {}, "scheduling": {}, "context": {}}),
                  patch.object(telemetry, "probe_slots", return_value=quiet_slots)):
                payload = public_api.snapshot(providers)
        self.assertEqual([a for a in payload["alerts"] if a["level"] != "info"], [])


class SnapshotTests(unittest.TestCase):
    def test_every_section_is_present_by_default(self):
        with _TelemetryStub():
            payload = public_api.snapshot(build_providers())
        for section in public_api.SECTIONS:
            self.assertIn(section, payload)
        self.assertEqual(payload["api_version"], public_api.API_VERSION)

    def test_include_narrows_the_payload(self):
        sections, unknown = public_api.resolve_sections("gpus,services")
        self.assertEqual(unknown, [])
        with _TelemetryStub():
            payload = public_api.snapshot(build_providers(), sections)
        self.assertIn("gpus", payload)
        self.assertIn("services", payload)
        self.assertNotIn("config", payload)
        self.assertNotIn("backends", payload)

    def test_an_unknown_section_is_reported_not_ignored(self):
        sections, unknown = public_api.resolve_sections("gpu,services")
        self.assertEqual(unknown, ["gpu"])
        self.assertEqual(sections, ["services"])

    def test_the_roll_up_counts_what_is_up_and_what_is_wrong(self):
        with _TelemetryStub():
            payload = public_api.snapshot(build_providers())
        stack = payload["stack"]
        self.assertEqual(stack["services_total"], 4)
        self.assertEqual(stack["services_active"], 2)
        self.assertTrue(stack["busy"])
        self.assertTrue(stack["router_enabled"])
        self.assertEqual(stack["alert_counts"]["error"], 2)

    def test_expensive_work_happens_once_however_many_sections_ask_for_it(self):
        calls = []
        providers = build_providers(gpu_info=lambda: (calls.append(1), [dict(g) for g in GPUS])[1])
        with _TelemetryStub():
            public_api.snapshot(providers)
        # gpus, backends, alerts and stack all need this.
        self.assertEqual(len(calls), 1)


def parse_exposition(text: str) -> dict[str, float]:
    """Prometheus exposition text -> {series: value}.

    Not `telemetry.parse_prometheus`: that one splits on the first space, which
    is fine for llama-server's unlabelled counters and wrong the moment a label
    value contains a space — as `name="RTX 5090"` does. The value is always the
    last field, so this splits from the right.
    """
    parsed = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        series, _, value = line.rpartition(" ")
        parsed[series] = float(value)
    return parsed


class PrometheusTests(unittest.TestCase):
    def setUp(self):
        with _TelemetryStub():
            self.payload = public_api.snapshot(build_providers())
        self.text = public_api.render_prometheus(self.payload)
        self.parsed = parse_exposition(self.text)

    def test_every_line_is_a_well_formed_series_and_value(self):
        self.assertTrue(self.parsed)
        for series, value in self.parsed.items():
            self.assertTrue(series and not series.endswith(" "), series)
            self.assertIsInstance(value, float)
        self.assertEqual(self.text[-1], "\n")

    def test_unlabelled_metrics_also_read_with_the_llama_parser(self):
        flat = telemetry.parse_prometheus(self.text)
        self.assertIn("llmstack_host_memory_used_bytes", flat)

    def test_gpu_and_model_vram_are_exposed(self):
        self.assertIn('llmstack_gpu_utilization_percent{gpu="0",name="RTX 5090",uuid="GPU-aaa"}', self.text)
        self.assertIn('llmstack_model_vram_bytes{gpu="0",unit="chat-backend-dense",'
                      'model="glm-4.6-Q4_K_M.gguf",attribution="exclusive"}', self.text)

    def test_context_and_service_state_are_exposed(self):
        self.assertIn("llmstack_backend_context_used_tokens", self.text)
        self.assertIn("llmstack_slot_context_used_tokens", self.text)
        self.assertIn('llmstack_service_up{name="embed",state="degraded"} 0', self.text)
        self.assertIn('llmstack_service_up{name="chat-backend-dense",state="active"} 1', self.text)

    def test_values_are_bytes_not_mebibytes(self):
        self.assertEqual(self.parsed['llmstack_gpu_memory_used_bytes{gpu="0",name="RTX 5090",uuid="GPU-aaa"}'],
                         30000 * 1024 * 1024)

    def test_every_metric_carries_help_and_type(self):
        for name, _kind, _help in public_api._METRIC_META:
            self.assertIn(f"# HELP {name} ", self.text)
            self.assertIn(f"# TYPE {name} ", self.text)

    def test_a_label_with_a_quote_cannot_break_the_format(self):
        payload = {"gpus": [dict(GPUS[1], name='Rogue" GPU', models=[])]}
        text = public_api.render_prometheus(payload)
        self.assertIn(r'name="Rogue\" GPU"', text)
        self.assertTrue(parse_exposition(text))


class BroadcasterTests(unittest.TestCase):
    def _broadcaster(self, providers=None, **settings):
        """A broadcaster whose loop thread never starts.

        `subscribe()` starts it in production, which is right there and wrong
        here: a live loop would collect on its own schedule, against unstubbed
        providers, and shell out to nvidia-smi from a test run.
        """
        base = {"interval": 2, "window": 3600, "webhook_url": "", "webhook_events": set(),
                "token": "", "bind_warning": ""}
        base.update(settings)
        broadcaster = public_api.Broadcaster(providers or build_providers(), lambda: dict(base))
        patcher = patch.object(broadcaster, "ensure_running")
        patcher.start()
        self.addCleanup(patcher.stop)
        return broadcaster

    def test_one_collection_serves_every_subscriber(self):
        calls = []
        providers = build_providers(gpu_info=lambda: (calls.append(1), [dict(g) for g in GPUS])[1])
        broadcaster = self._broadcaster(providers)
        first = broadcaster.subscribe()
        second = broadcaster.subscribe()
        with _TelemetryStub():
            broadcaster.tick()
        # The whole point of the shared loop: two clients, one nvidia-smi.
        self.assertEqual(len(calls), 1)
        self.assertEqual(first.queue.get_nowait()[0], "snapshot")
        self.assertEqual(second.queue.get_nowait()[0], "snapshot")

    def test_the_first_tick_does_not_report_the_whole_stack_as_transitions(self):
        broadcaster = self._broadcaster()
        subscriber = broadcaster.subscribe(["delta"])
        with _TelemetryStub():
            broadcaster.tick()
        self.assertTrue(subscriber.queue.empty())

    def test_a_service_changing_state_becomes_a_delta(self):
        broadcaster = self._broadcaster()
        subscriber = broadcaster.subscribe(["delta"])
        with _TelemetryStub():
            broadcaster.tick()
            broadcaster.providers = build_providers(
                service_health=lambda env: ({}, dict(SERVICE_HEALTH, **{
                    "embed": dict(SERVICE_HEALTH["embed"], state="active")})))
            broadcaster.tick()
        event_type, event = subscriber.queue.get_nowait()
        self.assertEqual(event_type, "delta")
        self.assertEqual((event["name"], event["from"], event["to"]), ("embed", "degraded", "active"))

    def test_a_subscriber_only_receives_what_it_asked_for(self):
        broadcaster = self._broadcaster()
        subscriber = broadcaster.subscribe(["alert"])
        with _TelemetryStub():
            broadcaster.tick()
        self.assertTrue(subscriber.queue.empty())

    def test_a_client_that_stops_reading_is_dropped_not_buffered_forever(self):
        subscription = public_api.Subscription(None)
        for _ in range(public_api.SUBSCRIBER_QUEUE_DEPTH + 5):
            subscription.offer("snapshot", {})
        self.assertTrue(subscription.dropped)
        self.assertEqual(subscription.queue.qsize(), public_api.SUBSCRIBER_QUEUE_DEPTH)

    def test_the_fastest_subscriber_sets_the_cadence(self):
        broadcaster = self._broadcaster(interval=10)
        broadcaster.subscribe(interval=10)
        broadcaster.subscribe(interval=2)
        self.assertEqual(broadcaster._current_interval({"interval": 10}), 2)

    def test_an_interval_outside_the_range_is_clamped(self):
        self.assertEqual(public_api.clamp_interval(0), public_api.MIN_STREAM_INTERVAL)
        self.assertEqual(public_api.clamp_interval(9999), public_api.MAX_STREAM_INTERVAL)
        self.assertEqual(public_api.clamp_interval("nonsense"), public_api.DEFAULT_STREAM_INTERVAL)

    def test_webhooks_fire_on_transitions_only_and_only_when_subscribed(self):
        sent = []
        broadcaster = self._broadcaster(webhook_url="http://receiver.local/hook",
                                        webhook_events={"service_state"}, token="t0k")
        with _TelemetryStub(), patch.object(public_api.WEBHOOKS, "send",
                                            side_effect=lambda *a: sent.append(a)):
            broadcaster.tick()
            self.assertEqual(sent, [])  # first tick establishes the baseline
            broadcaster.providers = build_providers(
                service_health=lambda env: ({}, dict(SERVICE_HEALTH, **{
                    "embed": dict(SERVICE_HEALTH["embed"], state="active")})))
            broadcaster.tick()
        self.assertEqual(len(sent), 1)
        url, token, event = sent[0]
        self.assertEqual(url, "http://receiver.local/hook")
        self.assertEqual(event["kind"], "service_state")

    def test_the_webhook_signature_is_an_hmac_of_the_body(self):
        body = b'{"hello":"world"}'
        signature = public_api.sign_webhook(body, "t0k")
        self.assertTrue(signature.startswith("sha256="))
        self.assertEqual(signature, public_api.sign_webhook(body, "t0k"))
        self.assertNotEqual(signature, public_api.sign_webhook(body, "other"))


class StateApiAppTests(unittest.TestCase):
    """The port other machines reach: what it carries, and what it must not."""

    def setUp(self):
        self.state_app = manager.create_state_api_app()
        manager.get_gpu_info.cache_clear()
        manager.service_main_pids.cache_clear()

    def test_every_rule_is_read_only(self):
        for rule in self.state_app.url_map.iter_rules():
            self.assertEqual(rule.methods - {"HEAD", "OPTIONS"}, {"GET"}, rule.rule)

    def test_the_mutating_routes_do_not_exist_on_it(self):
        rules = {str(rule) for rule in self.state_app.url_map.iter_rules()}
        for path in ("/api/config", "/api/service/<name>/<action>", "/api/status",
                     "/api/app/update", "/api/saved-configs", "/"):
            self.assertNotIn(path, rules)

    def test_it_does_not_serve_the_ui_assets(self):
        self.assertNotIn("/static/<path:filename>",
                         {str(rule) for rule in self.state_app.url_map.iter_rules()})

    def test_stopping_a_service_through_it_is_not_possible(self):
        with self.state_app.test_client() as client:
            self.assertEqual(client.post("/api/service/embed/stop").status_code, 404)
            self.assertEqual(client.get("/api/config").status_code, 404)

    def test_health_answers_without_a_token_even_when_one_is_set(self):
        with (self.state_app.test_client() as client,
              patch.object(config_env, "read_env", return_value=dict(ENV, LLM_API_TOKEN="s3cret"))):
            response = client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["ok"])

    def test_a_token_is_required_once_one_is_configured(self):
        with (self.state_app.test_client() as client,
              patch.object(config_env, "read_env", return_value=dict(ENV, LLM_API_TOKEN="s3cret"))):
            self.assertEqual(client.get("/api/v1/schema").status_code, 401)
            self.assertEqual(
                client.get("/api/v1/schema", headers={"Authorization": "Bearer s3cret"}).status_code, 200)
            self.assertEqual(client.get("/api/v1/schema?token=s3cret").status_code, 200)
            self.assertEqual(client.get("/api/v1/schema?token=wrong").status_code, 401)

    def test_no_token_configured_means_no_token_required(self):
        with (self.state_app.test_client() as client,
              patch.object(config_env, "read_env", return_value=dict(ENV))):
            self.assertEqual(client.get("/api/v1/schema").status_code, 200)

    def test_the_managers_own_port_does_not_start_demanding_a_token(self):
        # 8077 is as open as it has always been; implying otherwise would suggest
        # a protection it does not have.
        with (manager.app.test_client() as client,
              patch.object(config_env, "read_env", return_value=dict(ENV, LLM_API_TOKEN="s3cret"))):
            self.assertEqual(client.get("/api/v1/schema").status_code, 200)

    def test_cors_headers_appear_only_for_configured_origins(self):
        env = dict(ENV, LLM_API_ALLOW_ORIGINS="http://dash.local")
        with (self.state_app.test_client() as client,
              patch.object(config_env, "read_env", return_value=env)):
            allowed = client.get("/api/v1/health", headers={"Origin": "http://dash.local"})
            denied = client.get("/api/v1/health", headers={"Origin": "http://evil.local"})
        self.assertEqual(allowed.headers.get("Access-Control-Allow-Origin"), "http://dash.local")
        self.assertIsNone(denied.headers.get("Access-Control-Allow-Origin"))

    def test_no_cors_headers_at_all_by_default(self):
        with (self.state_app.test_client() as client,
              patch.object(config_env, "read_env", return_value=dict(ENV))):
            response = client.get("/api/v1/health", headers={"Origin": "http://dash.local"})
        self.assertIsNone(response.headers.get("Access-Control-Allow-Origin"))

    def test_an_unknown_include_is_a_400_with_the_known_names(self):
        with (self.state_app.test_client() as client,
              patch.object(config_env, "read_env", return_value=dict(ENV))):
            response = client.get("/api/v1/snapshot?include=gpu")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["unknown"], ["gpu"])
        self.assertIn("gpus", response.get_json()["known"])

    def test_a_journal_unit_outside_the_allow_list_is_refused(self):
        # The parameter reaches journalctl, which runs as root.
        with (self.state_app.test_client() as client,
              patch.object(config_env, "read_env", return_value=dict(ENV))):
            response = client.get("/api/v1/logs/raw?unit=../../etc/shadow")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "unknown_unit")

    def test_the_snapshot_endpoint_serves_the_payload(self):
        with (self.state_app.test_client() as client,
              patch.object(config_env, "read_env", return_value=dict(ENV)),
              _TelemetryStub()):
            payload = client.get("/api/v1/snapshot").get_json()
        self.assertEqual(payload["api_version"], public_api.API_VERSION)
        self.assertEqual(len(payload["gpus"]), 2)
        self.assertNotIn("hf_secretvalue", json.dumps(payload))

    def test_the_metrics_endpoint_serves_prometheus_text(self):
        with (self.state_app.test_client() as client,
              patch.object(config_env, "read_env", return_value=dict(ENV)),
              _TelemetryStub()):
            response = client.get("/api/v1/metrics")
        self.assertTrue(response.mimetype.startswith("text/plain"))
        self.assertIn("llmstack_gpu_utilization_percent", response.get_data(as_text=True))

    def test_the_schema_documents_every_route_it_serves(self):
        documented = set(public_api.schema()["endpoints"])
        served = {str(rule) for rule in self.state_app.url_map.iter_rules()}
        self.assertEqual(served - documented, set())


class SettingsTests(unittest.TestCase):
    def test_a_loopback_bind_is_not_warned_about(self):
        self.assertEqual(public_routes.bind_warning("127.0.0.1", ""), "")

    def test_an_open_bind_with_a_token_is_not_warned_about(self):
        self.assertEqual(public_routes.bind_warning("0.0.0.0", "s3cret"), "")

    def test_an_open_bind_with_no_token_is_warned_about(self):
        self.assertIn("no LLM_API_TOKEN", public_routes.bind_warning("100.64.0.5", ""))

    def test_settings_come_from_the_env_so_a_change_takes_effect(self):
        with patch.object(config_env, "read_env",
                          return_value=dict(ENV, LLM_API_PORT="9999", LLM_API_TOKEN="abc")):
            settings = public_routes.api_settings()
        self.assertEqual(settings["port"], 9999)
        self.assertEqual(settings["token"], "abc")

    def test_the_state_api_defaults_to_loopback(self):
        normalized = config_env.normalize_env_keys({})
        self.assertEqual(normalized["LLM_API_HOST"], "127.0.0.1")
        self.assertEqual(normalized["LLM_API_ENABLED"], "on")
        self.assertEqual(normalized["LLM_API_TOKEN"], "")


class TtlCacheTests(unittest.TestCase):
    def setUp(self):
        core.CACHE_TTL_SECONDS = None

    def tearDown(self):
        core.CACHE_TTL_SECONDS = None

    def test_repeat_calls_inside_the_window_do_not_recompute(self):
        calls = []

        @core.ttl_cache(30)
        def expensive():
            calls.append(1)
            return len(calls)

        self.assertEqual(expensive(), 1)
        self.assertEqual(expensive(), 1)
        self.assertEqual(len(calls), 1)

    def test_clearing_forces_a_recompute(self):
        calls = []

        @core.ttl_cache(30)
        def expensive():
            calls.append(1)
            return len(calls)

        expensive()
        expensive.cache_clear()
        expensive()
        self.assertEqual(len(calls), 2)

    def test_tests_can_switch_caching_off_globally(self):
        calls = []

        @core.ttl_cache(30)
        def expensive():
            calls.append(1)
            return len(calls)

        core.CACHE_TTL_SECONDS = 0
        expensive()
        expensive()
        self.assertEqual(len(calls), 2)


class LogEventTests(unittest.TestCase):
    EVENTS = [
        {"ts": 100.0, "unit": "chat-backend-dense", "kind": "generation", "tg_tps": 40.0},
        {"ts": 200.0, "unit": "chat-backend-dense", "kind": "context_overflow", "requested": 155751},
        {"ts": 300.0, "unit": "chat-backend-dense", "kind": "generation", "tg_tps": 42.0},
    ]

    def _registry(self):
        events = self.EVENTS

        class Collector:
            error = None

            @staticmethod
            def snapshot():
                return list(events)

        class Registry:
            @staticmethod
            def collector(unit, window):
                return Collector()

        return Registry()

    def test_events_come_from_the_existing_tailer_not_a_new_process(self):
        with patch.object(telemetry, "REGISTRY") as registry:
            registry.collector.return_value = type(
                "C", (), {"snapshot": staticmethod(lambda: list(self.EVENTS)), "error": None})()
            events = public_api.log_events(["chat-backend-dense"], 3600)
        self.assertEqual(len(events), 3)

    def test_kind_filters(self):
        events = public_api.log_events(["chat-backend-dense"], 3600, kinds={"context_overflow"},
                                       registry=self._registry())
        self.assertEqual([e["kind"] for e in events], ["context_overflow"])

    def test_since_returns_only_what_is_newer(self):
        events = public_api.log_events(["chat-backend-dense"], 3600, since=150.0,
                                       registry=self._registry())
        self.assertEqual([e["ts"] for e in events], [200.0, 300.0])

    def test_limit_keeps_the_most_recent(self):
        events = public_api.log_events(["chat-backend-dense"], 3600, limit=2,
                                       registry=self._registry())
        self.assertEqual([e["ts"] for e in events], [200.0, 300.0])


if __name__ == "__main__":
    unittest.main()
