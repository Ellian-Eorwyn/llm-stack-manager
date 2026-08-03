from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest


def _load_renderer():
    root = pathlib.Path(__file__).resolve().parents[1]
    path = root / "scripts" / "render-models-ini.py"
    spec = importlib.util.spec_from_file_location("render_models_ini", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


renderer = _load_renderer()

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "web"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import health  # noqa: E402
import setup_engine  # noqa: E402
import telemetry  # noqa: E402


BASE_ENV = {
    "MODEL_ROUTER_ENABLED": "on",
    "MODEL_ROUTER_MEMBERS": "EMBED,OCR,RERANK,TASK",
    "EMBEDDING_MODEL_PATH": "/models/embed.gguf",
    "EMBED_MODEL_NAME": "embed",
    "EMBED_CTX_SIZE": "8192",
    "RERANKER_MODEL_PATH": "/models/rerank.gguf",
    "RERANK_MODEL_NAME": "rank",
    "OCR_MODEL_PATH": "/models/ocr.gguf",
    "OCR_MMPROJ_PATH": "/models/ocr-mmproj.gguf",
    "OCR_MODEL_NAME": "ocr",
    "TASK_MODEL_PATH": "/models/task.gguf",
    "TASK_MODEL_NAME": "task",
}


def _sections(text: str) -> dict[str, dict[str, str]]:
    sections: dict[str, dict[str, str]] = {}
    current = None
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            sections[current] = {}
        elif "=" in line and current is not None:
            key, _, value = line.partition("=")
            sections[current][key.strip()] = value.strip()
    return sections


class SectionNameTests(unittest.TestCase):
    def test_sections_are_named_by_model_name_not_env_prefix(self):
        """The section name is the routing key clients must send.

        `update_args` overwrites the child's `--alias` with the section name, so
        a section named for the env prefix would silently change the name every
        caller has to use. The reranker is the one that catches this: its prefix
        is RERANK but `RERANK_MODEL_NAME` is `rank`.
        """
        sections = _sections(renderer.render(BASE_ENV))
        self.assertIn("rank", sections)
        self.assertNotIn("rerank", sections)
        self.assertEqual({"embed", "ocr", "rank", "task"}, set(sections) - {"*"})

    def test_a_missing_model_name_falls_back_to_the_conventional_id(self):
        env = dict(BASE_ENV)
        del env["RERANK_MODEL_NAME"]
        self.assertIn("rank", _sections(renderer.render(env)))

    def test_two_members_claiming_one_name_do_not_both_render(self):
        env = dict(BASE_ENV, MODEL_ROUTER_MEMBERS="EMBED,EMBED2",
                   EMBED2_MODEL_PATH="/models/embed2.gguf", EMBED2_MODEL_NAME="embed")
        warnings = []
        sections = _sections(renderer.render(env, warn=warnings.append))
        self.assertEqual({"embed"}, set(sections) - {"*"})
        self.assertTrue(any("already used" in w for w in warnings))


class GlobalSectionTests(unittest.TestCase):
    def test_version_lives_under_the_global_section(self):
        """Top-level keys would become a routable model called `default`.

        Anything before the first header lands in the preset named `default`,
        which the router then advertises on /v1/models as a real model with no
        model path — visible to every client that enumerates models, and a load
        error for anyone who asks for it.
        """
        text = renderer.render(BASE_ENV)
        sections = _sections(text)
        self.assertEqual(sections["*"]["version"], "1")
        body = [line for line in text.splitlines()
                if line.strip() and not line.strip().startswith(";")]
        self.assertEqual(body[0].strip(), "[*]")


class OptionRenderingTests(unittest.TestCase):
    def test_nothing_loads_until_it_is_asked_for(self):
        for name, options in _sections(renderer.render(BASE_ENV)).items():
            if name == "*":
                continue
            self.assertEqual(options.get("load-on-startup"), "false", name)

    def test_router_controlled_keys_are_never_emitted(self):
        for name, options in _sections(renderer.render(BASE_ENV)).items():
            for key in ("host", "port", "alias", "api-key"):
                self.assertNotIn(key, options, f"{name} must not set {key}")

    def test_an_unset_option_is_omitted_rather_than_written_empty(self):
        """`tensor-split = ` reads as an explicit empty split, not as absent."""
        env = dict(BASE_ENV, EMBED_TENSOR_SPLIT="", EMBED_DEVICE="")
        embed = _sections(renderer.render(env))["embed"]
        self.assertNotIn("tensor-split", embed)
        self.assertNotIn("device", embed)

    def test_auto_tensor_split_is_dropped(self):
        """`auto` is a convention of the start scripts, which expand it in bash.

        llama.cpp has no such value, so passing it through would be rejected.
        """
        env = dict(BASE_ENV, OCR_TENSOR_SPLIT="auto")
        self.assertNotIn("tensor-split", _sections(renderer.render(env))["ocr"])

    def test_values_that_would_be_truncated_by_a_comment_are_dropped(self):
        env = dict(BASE_ENV, EMBED_TEMP="0.7 ; sneaky")
        self.assertNotIn("temp", _sections(renderer.render(env))["embed"])

    def test_flag_values_pass_through_for_llama_cpp_to_negate(self):
        """`to_args` swaps in the `--no-` form itself when a value reads falsey,
        so the renderer must not try to guess negative flag names."""
        env = dict(BASE_ENV, OCR_KV_OFFLOAD="off", EMBED_NO_MMAP="true")
        sections = _sections(renderer.render(env))
        self.assertEqual(sections["ocr"]["kv-offload"], "off")
        self.assertEqual(sections["embed"]["no-mmap"], "true")

    def test_each_kind_of_server_gets_the_flag_that_makes_it_that_kind(self):
        sections = _sections(renderer.render(BASE_ENV))
        self.assertEqual(sections["embed"]["embedding"], "true")
        self.assertEqual(sections["embed"]["pooling"], "mean")
        self.assertEqual(sections["rank"]["reranking"], "true")
        self.assertNotIn("reranking", sections["embed"])
        self.assertNotIn("embedding", sections["task"])

    def test_mmproj_is_emitted_only_when_configured(self):
        sections = _sections(renderer.render(BASE_ENV))
        self.assertEqual(sections["ocr"]["mmproj"], "/models/ocr-mmproj.gguf")
        self.assertNotIn("mmproj", sections["embed"])
        env = dict(BASE_ENV, OCR_MMPROJ_PATH="")
        self.assertNotIn("mmproj", _sections(renderer.render(env))["ocr"])

    def test_shell_quoting_is_stripped_from_env_values(self):
        env = dict(BASE_ENV, EMBED_CTX_SIZE='"4096"')
        self.assertEqual(_sections(renderer.render(env))["embed"]["ctx-size"], "4096")


class CustomArgumentTests(unittest.TestCase):
    def test_custom_flags_and_options_become_preset_keys(self):
        env = dict(BASE_ENV, OCR_CUSTOM_ARGS_JSON='["--verbose", "--seed 42"]')
        ocr = _sections(renderer.render(env))["ocr"]
        self.assertEqual(ocr["verbose"], "true")
        self.assertEqual(ocr["seed"], "42")

    def test_custom_args_cannot_smuggle_in_a_router_controlled_key(self):
        env = dict(BASE_ENV, OCR_CUSTOM_ARGS_JSON='["--port 9999"]')
        self.assertNotIn("port", _sections(renderer.render(env))["ocr"])

    def test_an_empty_custom_args_list_changes_nothing(self):
        env = dict(BASE_ENV, OCR_CUSTOM_ARGS_JSON="[]")
        self.assertEqual(_sections(renderer.render(env))["ocr"],
                         _sections(renderer.render(BASE_ENV))["ocr"])


class MemberSelectionTests(unittest.TestCase):
    def test_a_member_without_a_model_path_is_skipped_not_fatal(self):
        """One unconfigured model must not take the other three offline."""
        env = dict(BASE_ENV, OCR_MODEL_PATH="")
        warnings = []
        sections = _sections(renderer.render(env, warn=warnings.append))
        self.assertNotIn("ocr", sections)
        self.assertEqual({"embed", "rank", "task"}, set(sections) - {"*"})
        self.assertTrue(any("OCR_MODEL_PATH" in w for w in warnings))

    def test_an_unknown_member_is_reported(self):
        env = dict(BASE_ENV, MODEL_ROUTER_MEMBERS="EMBED,NOSUCH")
        warnings = []
        renderer.render(env, warn=warnings.append)
        self.assertTrue(any("NOSUCH" in w for w in warnings))

    def test_no_renderable_member_is_an_error(self):
        env = dict(BASE_ENV, MODEL_ROUTER_MEMBERS="NOSUCH")
        with self.assertRaises(renderer.RenderError):
            renderer.render(env, warn=lambda _: None)

    def test_members_default_to_the_four_auxiliary_models(self):
        env = dict(BASE_ENV)
        del env["MODEL_ROUTER_MEMBERS"]
        self.assertEqual({"embed", "ocr", "rank", "task"},
                         set(_sections(renderer.render(env))) - {"*"})


class PooledUnitTests(unittest.TestCase):
    def test_nothing_is_pooled_while_the_router_is_off(self):
        self.assertEqual(telemetry.pooled_units({"MODEL_ROUTER_ENABLED": "off"}), set())
        self.assertEqual(telemetry.pooled_units({}), set())

    def test_pooled_units_follow_the_configured_members(self):
        self.assertEqual(telemetry.pooled_units(BASE_ENV),
                         {"embed", "ocr", "rerank", "task"})
        env = dict(BASE_ENV, MODEL_ROUTER_MEMBERS="EMBED, OCR")
        self.assertEqual(telemetry.pooled_units(env), {"embed", "ocr"})

    def test_every_renderable_member_maps_to_a_unit(self):
        """The two tables are keyed by the same prefixes and must not drift:
        a member the renderer serves but telemetry does not know about would be
        reported as stopped-on-purpose while the router was serving it."""
        self.assertEqual(set(renderer.MEMBERS), set(telemetry.ROUTER_MEMBER_UNITS))


class PlacementBudgetTests(unittest.TestCase):
    """A pooled model must not be charged for memory it never holds."""

    GPUS = [{"index": 0, "memory_total_mib": 24576, "memory_free_mib": 24576}]
    MODELS = {
        "primary": {"size_mib": 8000},
        "embedding": {"size_mib": 4000},
        "ocr": {"size_mib": 4000},
        "task": {"size_mib": 4000},
    }

    def setUp(self):
        # `estimate_model_mib` reads whatever shape the wizard collected; the
        # tests only care that the arithmetic over it changes, so pin it.
        self._original = setup_engine.estimate_model_mib
        setup_engine.estimate_model_mib = lambda model: int(model["size_mib"])
        self.addCleanup(setattr, setup_engine, "estimate_model_mib", self._original)

    def test_a_group_costs_its_largest_member_not_their_sum(self):
        plain = setup_engine._required_mib(self.MODELS)
        pooled = setup_engine._required_mib(
            self.MODELS, exclusive_groups=(("embedding", "ocr", "task"),))
        self.assertEqual(plain, 20000)
        self.assertEqual(pooled, 12000)

    def test_a_group_member_that_was_not_selected_is_ignored(self):
        models = {"primary": {"size_mib": 8000}, "ocr": {"size_mib": 4000}}
        self.assertEqual(
            setup_engine._required_mib(models, exclusive_groups=(("embedding", "ocr"),)),
            12000)

    def test_a_group_with_no_selected_member_costs_nothing(self):
        models = {"primary": {"size_mib": 8000}}
        self.assertEqual(
            setup_engine._required_mib(models, exclusive_groups=(("embedding", "ocr"),)),
            8000)

    def test_pooling_can_rescue_a_plan_that_would_otherwise_be_refused(self):
        gpus = [{"index": 0, "memory_total_mib": 20000, "memory_free_mib": 20000}]
        refused = setup_engine.plan_gpu_placement(gpus, self.MODELS)
        self.assertFalse(refused["ok"])
        allowed = setup_engine.plan_gpu_placement(
            gpus, self.MODELS, exclusive_groups=(("embedding", "ocr", "task"),))
        self.assertTrue(allowed["ok"], allowed.get("error"))

    def test_omitting_groups_leaves_the_old_arithmetic_untouched(self):
        self.assertEqual(setup_engine._required_mib(self.MODELS),
                         sum(m["size_mib"] for m in self.MODELS.values()))


class RouterHealthTests(unittest.TestCase):
    STATUSES = {
        "llama-router": health.STATE_ACTIVE,
        "embed": health.STATE_INACTIVE,
        "ocr": health.STATE_INACTIVE,
        "glmocr-sdk": health.STATE_ACTIVE,
    }

    def test_a_pooled_model_is_not_reported_as_a_fault(self):
        entries = health.collect(BASE_ENV, self.STATUSES, probes={}, expectations={})
        self.assertEqual(entries["embed"]["state"], health.STATE_STOPPED)
        self.assertIn("model router", entries["embed"]["reason"])

    def test_a_stale_on_expectation_does_not_resurrect_a_pooled_model(self):
        """Turning the router on leaves whatever the panel last recorded behind."""
        expectations = {"embed": {"expected": "on"}}
        entries = health.collect(BASE_ENV, self.STATUSES, probes={},
                                 expectations=expectations)
        self.assertEqual(entries["embed"]["expected"], "off")
        self.assertEqual(entries["embed"]["state"], health.STATE_STOPPED)

    def test_the_ocr_sdk_is_healthy_when_the_router_holds_its_model(self):
        entries = health.collect(BASE_ENV, self.STATUSES, probes={}, expectations={})
        self.assertEqual(entries["glmocr-sdk"]["state"], health.STATE_ACTIVE)

    def test_the_ocr_sdk_is_degraded_when_the_router_is_down(self):
        statuses = dict(self.STATUSES, **{"llama-router": health.STATE_INACTIVE})
        entries = health.collect(BASE_ENV, statuses, probes={}, expectations={})
        self.assertEqual(entries["glmocr-sdk"]["state"], health.STATE_DEGRADED)

    def test_with_the_router_off_the_ocr_sdk_judges_the_ocr_unit_again(self):
        env = dict(BASE_ENV, MODEL_ROUTER_ENABLED="off")
        entries = health.collect(env, self.STATUSES, probes={}, expectations={})
        self.assertEqual(entries["glmocr-sdk"]["state"], health.STATE_DEGRADED)
        self.assertEqual(entries["glmocr-sdk"]["upstreams"][0]["any_of"], ["ocr"])

    def test_the_router_being_off_is_not_a_fault(self):
        env = dict(BASE_ENV, MODEL_ROUTER_ENABLED="off")
        statuses = {"llama-router": health.STATE_INACTIVE}
        entries = health.collect(env, statuses, probes={}, expectations={})
        self.assertEqual(entries["llama-router"]["state"], health.STATE_STOPPED)
        self.assertEqual(entries["llama-router"]["reason"],
                         "turned off in the configuration")

    def test_the_router_is_probed_on_an_endpoint_that_cannot_cause_a_load(self):
        """The services page sweeps every five seconds. `/props` would need a
        `?model=` and the inference paths would load a model, so a probe on
        anything else would turn the panel into a swap generator."""
        self.assertEqual(health.SERVICE_PROBES["llama-router"]["path"], "/health")
        self.assertEqual(health.SERVICE_PROBES["llama-router"]["port_key"],
                         "MODEL_ROUTER_PORT")


if __name__ == "__main__":
    unittest.main()
