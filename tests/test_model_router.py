from __future__ import annotations

import importlib.util
import pathlib
import re
import sys
import tempfile
import unittest
from unittest import mock


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
        # The section name *is* the routing key, so two members answering to one
        # name would make which model serves a request depend on render order.
        env = dict(BASE_ENV, MODEL_ROUTER_MEMBERS="EMBED,RERANK",
                   RERANK_MODEL_NAME="embed")
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


class TaskSettingCoverageTests(unittest.TestCase):
    """A setting the UI offers has to survive the trip into the preset.

    The router never runs `start-task.sh`; it builds each child's argv from this
    file. So anything the start script derives and the renderer does not is a
    control the config UI claims to have and silently does not — which is how
    `TASK_THINKING=off` came to be honoured by the unit and lost by the router,
    leaving :8007 reasoning on every request.
    """

    # Suffixes the preset deliberately does not carry. PORT and
    # GPU_VISIBLE_DEVICES belong to the router, which assigns a free port per
    # child and owns CUDA_VISIBLE_DEVICES for all of them. The SPEC_* family
    # needs the method-dependent branching of start-task.sh:127-214, which a
    # flat suffix table cannot express; it is off in the shipped config.
    # POOLED decides whether the router owns this model at all, which is a
    # question about the pool rather than a setting the model is started
    # with. It reaches the preset by the member being present or absent.
    KNOWINGLY_DROPPED = {"PORT", "GPU_VISIBLE_DEVICES", "POOLED"}

    def _task(self, **overrides):
        return _sections(renderer.render(dict(BASE_ENV, **overrides)))["task"]

    def test_every_task_setting_is_carried_or_knowingly_dropped(self):
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "web"))
        import config_fields  # noqa: PLC0415

        carried = set(renderer.VALUE_OPTIONS) | set(renderer.FLAG_OPTIONS) | {
            "MODEL_PATH", "MMPROJ_PATH", "MODEL_NAME", "CUSTOM_ARGS_JSON",
            "THINKING", "REASONING_EFFORT", "CHAT_TEMPLATE_ID",
            "LOAD_ON_STARTUP",
        }
        declared = {
            field["key"][len("TASK_"):]
            for field in config_fields.CONFIG_FIELDS
            if field.get("key", "").startswith("TASK_")
        }
        dropped = {
            suffix for suffix in declared - carried
            if not suffix.startswith("SPEC_")
        }
        self.assertEqual(dropped, self.KNOWINGLY_DROPPED)

    def test_load_on_startup_is_a_bool_not_an_on_off_flag(self):
        """The router parses it as a bool and reads anything else as false.

        Every other flag here passes through verbatim because llama.cpp accepts
        `on`/`off` for them. Writing `on` for this one would be read as `off`
        and the model would stay lazy while the UI claimed otherwise.
        """
        self.assertEqual(self._task(TASK_LOAD_ON_STARTUP="on")["load-on-startup"], "true")
        self.assertEqual(self._task(TASK_LOAD_ON_STARTUP="off")["load-on-startup"], "false")

    def test_a_member_nobody_configured_stays_lazy(self):
        """Unset means false: an unconsidered model costs nothing until used."""
        self.assertEqual(self._task()["load-on-startup"], "false")

    def test_thinking_rides_in_as_a_chat_template_kwarg(self):
        """Thinking is a template variable, not a llama.cpp flag."""
        self.assertEqual(
            self._task(TASK_THINKING="off")["chat-template-kwargs"],
            '{"enable_thinking":false}',
        )

    def test_thinking_level_rides_along_only_while_thinking_is_on(self):
        """Qwen 3.8's template raises on a level it does not know, and ignores
        any level once thinking is off — so an unusable one is never sent."""
        self.assertEqual(
            self._task(TASK_THINKING="on", TASK_REASONING_EFFORT="low")["chat-template-kwargs"],
            '{"enable_thinking":true, "reasoning_effort": "low"}',
        )
        self.assertEqual(
            self._task(TASK_THINKING="off", TASK_REASONING_EFFORT="low")["chat-template-kwargs"],
            '{"enable_thinking":false}',
        )
        self.assertEqual(
            self._task(TASK_THINKING="on", TASK_REASONING_EFFORT="high")["chat-template-kwargs"],
            '{"enable_thinking":true}',
        )
        self.assertEqual(
            self._task(TASK_THINKING="on")["chat-template-kwargs"],
            '{"enable_thinking":true}',
        )

    def test_a_member_with_nothing_to_think_about_gets_no_kwargs(self):
        sections = _sections(renderer.render(dict(BASE_ENV, TASK_THINKING="off")))
        self.assertNotIn("chat-template-kwargs", sections["embed"])
        self.assertNotIn("chat-template-kwargs", sections["rank"])

    def test_a_chat_template_id_resolves_to_the_file_it_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            templates = pathlib.Path(tmp) / "config" / "chat-templates"
            templates.mkdir(parents=True)
            (templates / "custom.jinja").write_text("{{ messages }}")
            with mock.patch.object(renderer, "STACK_DIR", pathlib.Path(tmp)):
                task = self._task(TASK_CHAT_TEMPLATE_ID="custom")
        self.assertEqual(task["chat-template-file"], str(templates / "custom.jinja"))

    def test_a_missing_chat_template_skips_the_member_rather_than_guessing(self):
        """Falling back to the model's built-in template is the worse failure.

        A skipped member is loudly absent; a silent fallback answers every
        request with a subtly wrong prompt format.
        """
        warnings: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(renderer, "STACK_DIR", pathlib.Path(tmp)):
                sections = _sections(renderer.render(
                    dict(BASE_ENV, TASK_CHAT_TEMPLATE_ID="absent"),
                    warn=warnings.append,
                ))
        self.assertNotIn("task", sections)
        self.assertTrue(any("chat template not found" in text for text in warnings))

    def test_a_chat_template_id_cannot_name_a_path(self):
        warnings: list[str] = []
        sections = _sections(renderer.render(
            dict(BASE_ENV, TASK_CHAT_TEMPLATE_ID="../../etc/passwd"),
            warn=warnings.append,
        ))
        self.assertNotIn("task", sections)
        self.assertTrue(any("invalid" in text.lower() for text in warnings))

    def test_the_cache_and_fit_knobs_reach_the_preset(self):
        task = self._task(
            TASK_CACHE_IDLE_SLOTS="on",
            TASK_CACHE_REUSE="256",
            TASK_FIT="on",
            TASK_FIT_TARGET="2048",
            TASK_FIT_CTX="4096",
        )
        self.assertEqual(task["cache-idle-slots"], "on")
        self.assertEqual(task["cache-reuse"], "256")
        self.assertEqual(task["fit-target"], "2048")
        self.assertEqual(task["fit-ctx"], "4096")

    def test_fit_ctx_is_dropped_when_auto_fit_is_off(self):
        """`add_fit_ctx_opt` drops it, so the preset has to drop it too."""
        self.assertNotIn("fit-ctx", self._task(TASK_FIT="off", TASK_FIT_CTX="4096"))

    def test_zero_reads_as_unset_rather_than_as_a_literal_zero(self):
        task = self._task(TASK_FIT="on", TASK_FIT_CTX="0", TASK_CACHE_REUSE="0")
        self.assertNotIn("fit-ctx", task)
        self.assertNotIn("cache-reuse", task)


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


class AbsoluteGpuIndexTests(unittest.TestCase):
    """`MAIN_GPU` means two different cards depending on who starts the model.

    A slot with `GPU_VISIBLE_DEVICES=1` and `MAIN_GPU=0` runs on physical GPU 1,
    because llama.cpp only sees one device and calls it 0. The same `MAIN_GPU=0`
    under `llama-router`, which sets its own visible list, runs on physical
    GPU 0. Absolute indices fix that and cannot just be switched on: the stored
    values were written in the renumbered space, so changing the meaning moves
    models between cards. On the host this was written for, `rerank` and `ocr`
    would both have jumped from GPU 1 to GPU 0 with nothing in the config
    changing.
    """

    def setUp(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location(
            "gpu_indices", root / "scripts" / "lib" / "gpu-indices.py")
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)

    def test_a_slot_pinned_to_the_second_card_is_reported_as_moving(self):
        rows = {r["slot"]: r for r in self.mod.translate(
            {"RERANK_GPU_VISIBLE_DEVICES": "1", "RERANK_MAIN_GPU": "0"})}
        self.assertTrue(rows["rerank"]["moves"])
        self.assertEqual(rows["rerank"]["main_gpu"], 1)

    def test_an_identity_visible_list_moves_nothing(self):
        rows = {r["slot"]: r for r in self.mod.translate(
            {"LLM_A_GPU_VISIBLE_DEVICES": "0,1", "LLM_A_MAIN_GPU": "0"})}
        self.assertFalse(rows["llm-a"]["moves"])
        self.assertEqual(rows["llm-a"]["main_gpu"], 0)

    def test_a_tensor_split_is_widened_to_the_whole_machine(self):
        """One weight per *visible* device becomes one per device, with zeros
        for the cards the slot never used -- otherwise the split silently
        describes a different set of GPUs."""
        rows = {r["slot"]: r for r in self.mod.translate(
            {"RERANK_GPU_VISIBLE_DEVICES": "1", "RERANK_MAIN_GPU": "0",
             "RERANK_TENSOR_SPLIT": "1"})}
        self.assertEqual(rows["rerank"]["tensor_split"], "0,1")

    def test_the_switch_is_off_by_default(self):
        """It changes where models run. That is an operator's decision, made
        after reading the report, not a default that arrives with an update."""
        script = (pathlib.Path(__file__).resolve().parents[1]
                  / "scripts" / "start-backend.sh").read_text()
        self.assertIn('LLM_ABSOLUTE_GPU_INDICES:-off', script)
        self.assertIn("CUDA_VISIBLE_DEVICES", script)


class PoolMembershipTests(unittest.TestCase):
    """One source for which models the router owns.

    The default string was written out in thirteen files -- four shell scripts,
    five Python modules, the field hint and the docs -- which is twelve chances
    to add a member and have half the stack disagree about whether it is in the
    pool. `ASR` proved the point: in the member table, out of the default
    string, and every copy had to get that distinction right.
    """

    def setUp(self):
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "web"))
        from backends import router  # noqa: PLC0415
        self.router = router

    def test_the_default_pool_leaves_the_audio_model_out(self):
        """Pooling ASR is opt-in: its only caller is the transcription sidecar,
        and it would compete for VRAM with models serving interactive traffic."""
        self.assertIn("ASR", self.router.MEMBER_PREFIXES)
        self.assertNotIn("ASR", self.router.DEFAULT_POOLED)
        self.assertEqual(self.router.pooled_members({}), list(self.router.DEFAULT_POOLED))

    def test_a_host_that_only_has_the_string_keeps_working(self):
        """Every host is this host until someone touches a switch."""
        self.assertEqual(
            self.router.pooled_members({"MODEL_ROUTER_MEMBERS": "EMBED,ASR"}),
            ["EMBED", "ASR"])

    def test_inherit_is_not_a_switch(self):
        """A select cannot render "unset" without showing its first option, so
        the field says `inherit` explicitly. It must not count as configured, or
        every host would silently start overriding its own string."""
        env = {"MODEL_ROUTER_MEMBERS": "EMBED,ASR"}
        env.update({self.router.pooled_key(p): "inherit"
                    for p in self.router.MEMBER_PREFIXES})
        self.assertEqual(self.router.pooled_members(env), ["EMBED", "ASR"])

    def test_one_switch_makes_the_switches_authoritative(self):
        """A mixture -- some members switched, the rest from a stale string --
        is the ambiguity this replaces, not a feature."""
        self.assertEqual(
            self.router.pooled_members({"MODEL_ROUTER_MEMBERS": "EMBED,ASR",
                                        "OCR_POOLED": "on"}),
            ["EMBED", "OCR", "RERANK", "TASK"])

    def test_a_member_switched_off_leaves_the_pool(self):
        self.assertNotIn("OCR", self.router.pooled_members({"OCR_POOLED": "off"}))

    def test_the_order_is_the_registry_not_the_string(self):
        """Two hosts listing the same members differently must render the same
        preset, or a diff of two configs is unreadable."""
        self.assertEqual(
            self.router.pooled_members({"MODEL_ROUTER_MEMBERS": "TASK,EMBED"}),
            self.router.pooled_members({"MODEL_ROUTER_MEMBERS": "EMBED,TASK"}))

    def test_a_name_nothing_serves_is_reported(self):
        warnings = []
        self.router.pooled_members({"MODEL_ROUTER_MEMBERS": "EMBED,NOSUCH"},
                                   warn=warnings.append)
        self.assertTrue(any("NOSUCH" in w for w in warnings))

    def test_every_consumer_derives_rather_than_repeating_the_default(self):
        """The duplication this exists to end. A copy that drifts is a stack
        that disagrees with itself about what the router owns."""
        root = pathlib.Path(__file__).resolve().parents[1]
        allowed = {"web/backends/router.py", "scripts/lib/router-members.py"}
        offenders = []
        for path in list(root.glob("scripts/**/*.py")) + list(root.glob("scripts/*.sh")) \
                + list(root.glob("web/**/*.py")) + [root / "install.sh"]:
            rel = str(path.relative_to(root))
            if rel in allowed or ".test-venv" in rel or "/.venv/" in rel:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if "EMBED,OCR,RERANK,TASK" in text and "e.g." not in text:
                offenders.append(rel)
        self.assertEqual(offenders, [])


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

    def test_more_eager_members_than_the_router_may_hold_are_capped(self):
        """The router throws rather than degrading -- "number of models to load
        on startup (4) exceeds models_max (2)" -- so two individually reasonable
        settings combine into a router that will not come up."""
        env = dict(BASE_ENV, MODEL_ROUTER_MAX="2",
                   EMBED_LOAD_ON_STARTUP="on", OCR_LOAD_ON_STARTUP="on",
                   RERANK_LOAD_ON_STARTUP="on", TASK_LOAD_ON_STARTUP="on")
        warnings = []
        sections = _sections(renderer.render(env, warn=warnings.append))
        eager = [name for name, opts in sections.items()
                 if opts.get("load-on-startup") == "true"]
        self.assertEqual(len(eager), 2)
        self.assertTrue(any("MODEL_ROUTER_MAX" in w for w in warnings))

    def test_the_cap_drops_by_registry_order_not_by_the_string(self):
        """Registry order, not the order the string happens to list them in --
        two hosts naming the same members differently must render the same
        preset, so the cap has to drop the same one on both."""
        env = dict(BASE_ENV, MODEL_ROUTER_MEMBERS="TASK,EMBED,OCR", MODEL_ROUTER_MAX="1",
                   EMBED_LOAD_ON_STARTUP="on", OCR_LOAD_ON_STARTUP="on",
                   TASK_LOAD_ON_STARTUP="on")
        sections = _sections(renderer.render(env, warn=lambda _m: None))
        eager = [name for name, opts in sections.items()
                 if opts.get("load-on-startup") == "true"]
        self.assertEqual(eager, ["embed"])

    def test_a_pool_within_its_limit_is_left_alone(self):
        env = dict(BASE_ENV, MODEL_ROUTER_MAX="2", EMBED_LOAD_ON_STARTUP="on")
        sections = _sections(renderer.render(env, warn=lambda _m: None))
        self.assertEqual(sections["embed"]["load-on-startup"], "true")

    def test_the_nginx_member_table_agrees_with_the_renderer(self):
        """The third member table, and the one that had no test.

        `install-model-router-nginx.sh` fronts each member's public port onto
        the router. A member the renderer serves but this table omits keeps
        answering on the router's own port and nowhere a caller looks; a member
        here that the renderer does not serve gets an nginx block pointing at a
        model the router will refuse.

        ASR is the deliberate exception -- its only caller is the transcription
        sidecar, which posts to the router directly, so a public shim would be a
        second door to the same room.
        """
        script = (pathlib.Path(__file__).resolve().parents[1]
                  / "scripts" / "install-model-router-nginx.sh").read_text()
        block = re.search(r"declare -A MEMBER_PORTS=\((.*?)\n\)", script, re.S)
        self.assertIsNotNone(block, "MEMBER_PORTS changed shape")
        named = set(re.findall(r"^\s*\[(\w+)\]=", block.group(1), re.M))
        self.assertEqual(named, set(renderer.MEMBERS))
        portless = {m for m in named
                    if re.search(rf"^\s*\[{m}\]=\"\$\{{\w+:-\}}\"", block.group(1), re.M)
                    and m == "ASR"}
        self.assertEqual(portless, {"ASR"})

    def test_the_audio_model_is_pooled_only_when_asked_for(self):
        """ASR is opt-in: being in MEMBERS is not the same as being in the pool."""
        self.assertIn("ASR", renderer.MEMBERS)
        self.assertNotIn("asr", telemetry.pooled_units(BASE_ENV))
        env = dict(BASE_ENV, MODEL_ROUTER_MEMBERS="EMBED,OCR,RERANK,TASK,ASR")
        self.assertIn("asr", telemetry.pooled_units(env))

    def test_the_audio_model_renders_with_its_projector(self):
        """llama.cpp refuses transcription without an audio projector, so a
        section that omits mmproj would load and then fail every request."""
        env = dict(
            BASE_ENV,
            MODEL_ROUTER_MEMBERS="ASR",
            ASR_MODEL_PATH="/models/voxtral.gguf",
            ASR_MMPROJ_PATH="/models/voxtral-mmproj.gguf",
            ASR_JINJA="on",
        )
        rendered = renderer.render(env)
        self.assertIn("[asr]", rendered)
        self.assertIn("model = /models/voxtral.gguf", rendered)
        self.assertIn("mmproj = /models/voxtral-mmproj.gguf", rendered)
        self.assertIn("jinja = on", rendered)
        self.assertIn("load-on-startup = false", rendered)

    def test_the_audio_model_is_absent_from_a_default_render(self):
        self.assertNotIn("[asr]", renderer.render(BASE_ENV))


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
