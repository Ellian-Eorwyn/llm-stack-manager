from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import unittest
from collections import Counter
from unittest.mock import patch


def _load_app_module():
    root = pathlib.Path(__file__).resolve().parents[1]
    app_path = root / "web" / "app.py"
    spec = importlib.util.spec_from_file_location("llm_stack_manager_app", app_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


manager = _load_app_module()
# The sibling modules app.py imported on its way up, taken from sys.modules
# rather than loaded again — a second copy would have its own globals, and
# patching one of them would leave the application calling the other.
core = sys.modules["core"]
config_env = sys.modules["config_env"]
config_fields = sys.modules["config_fields"]
models = sys.modules["models"]

# Imported after the app, for the same reason: it reaches the `platforms`
# module app.py already put in sys.modules, so a substituted adapter is the one
# the application sees.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import platform_harness  # noqa: E402


class ConfigSectionTests(unittest.TestCase):
    def test_primary_and_secondary_backend_fields_are_separate(self):
        sections = {f["key"]: f["section"] for f in manager.CONFIG_FIELDS}
        self.assertEqual(sections["LLM_A_MODEL_PATH"], "LLM A")
        self.assertEqual(sections["LLM_A_SPEC_METHOD"], "LLM A")
        self.assertEqual(sections["LLM_A_CUSTOM_ARGS_JSON"], "LLM A")
        self.assertEqual(sections["LLM_B_LABEL"], "LLM B")
        self.assertEqual(sections["LLM_B_MODEL_PATH"], "LLM B")
        self.assertEqual(sections["LLM_B_SPEC_METHOD"], "LLM B")
        self.assertEqual(sections["LLM_B_CUSTOM_ARGS_JSON"], "LLM B")
        self.assertNotIn("CHAT_SECONDARY_MODEL_PATH", sections)

    def test_primary_and_secondary_backend_restart_independently(self):
        self.assertEqual(manager.RESTART_HINTS["LLM_A_MODEL_PATH"], ["llm-a"])
        self.assertEqual(manager.RESTART_HINTS["LLM_A_BATCH_SIZE"], ["llm-a"])
        self.assertEqual(manager.RESTART_HINTS["LLM_B_MODEL_PATH"], ["llm-b"])
        self.assertEqual(manager.RESTART_HINTS["LLM_B_BATCH_SIZE"], ["llm-b"])

    def test_cache_aware_scheduling_cards_render_for_both_backends(self):
        template_dir = pathlib.Path(__file__).resolve().parents[1] / "web" / "templates"
        original_searchpath = list(manager.app.jinja_loader.searchpath)
        manager.app.jinja_loader.searchpath = [str(template_dir)]
        with (
            manager.app.test_client() as client,
            patch.object(config_env, "read_env", return_value={}),
            patch.object(models, "load_custom_models", return_value=[]),
        ):
            response = client.get("/")
        manager.app.jinja_loader.searchpath = original_searchpath

        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn('data-cache-aware-prefix="LLM_A"', html)
        self.assertIn('data-cache-aware-prefix="LLM_B"', html)
        self.assertEqual(html.count("Enable Cache-Aware Scheduling"), 2)
        self.assertIn("cache-aware-scheduling.js", html)
        self.assertIn("~/.pi-forge/agent/settings.json", html)

    def test_cache_aware_fields_restart_their_own_backends(self):
        for suffix in (
            "N_PARALLEL",
            "CTX_SIZE",
            "CACHE_RAM",
            "CTX_CHECKPOINTS",
            "CACHE_IDLE_SLOTS",
            "FIT",
        ):
            self.assertEqual(manager.RESTART_HINTS[f"LLM_A_{suffix}"], ["llm-a"])
            self.assertEqual(manager.RESTART_HINTS[f"LLM_B_{suffix}"], ["llm-b"])

    def test_primary_and_secondary_backend_normalize_from_legacy_keys(self):
        env = config_env.normalize_env_keys({
            "CHAT_DENSE_LABEL": "Backend Dense",
            "CHAT_DENSE_MODEL_PATH": "/models/primary.gguf",
            "CHAT_DENSE_CTX_SIZE": "32768",
            "CHAT_MOE_LABEL": "Backend MoE",
            "CHAT_MOE_MODEL_PATH": "/models/secondary.gguf",
            "CHAT_MOE_CTX_SIZE": "65536",
            "CHAT_BATCH_SIZE": "2048",
            "CHAT_GPU_VISIBLE_DEVICES": "0,1",
        })
        self.assertEqual(env["LLM_A_LABEL"], "LLM A")
        self.assertEqual(env["LLM_A_MODEL_PATH"], "/models/primary.gguf")
        self.assertEqual(env["LLM_A_CTX_SIZE"], "32768")
        self.assertEqual(env["LLM_A_BATCH_SIZE"], "2048")
        self.assertEqual(env["LLM_A_GPU_VISIBLE_DEVICES"], "0,1")
        # The MoE keys described an alternative model for the *one* shared
        # backend. That slot is retired, so they backfill the concurrent second
        # slot instead of vanishing -- a host with a MoE model and no second
        # slot keeps its model.
        self.assertEqual(env["LLM_B_LABEL"], "LLM B")
        self.assertEqual(env["LLM_B_MODEL_PATH"], "/models/secondary.gguf")
        self.assertEqual(env["LLM_B_CTX_SIZE"], "65536")
        self.assertNotIn("CHAT_SECONDARY_LABEL", env)
        self.assertNotIn("CHAT_SECONDARY_BATCH_SIZE", {f["key"] for f in manager.CONFIG_FIELDS})

    def test_a_configured_second_slot_is_not_overwritten_by_legacy_moe_keys(self):
        """The backfill never overwrites, so the two cannot collide."""
        env = config_env.normalize_env_keys({
            "CHAT_MOE_MODEL_PATH": "/models/old-moe.gguf",
            "LLM_B_MODEL_PATH": "/models/current-b.gguf",
        })
        self.assertEqual(env["LLM_B_MODEL_PATH"], "/models/current-b.gguf")

    def test_removed_backend_fields_are_not_in_config_surface(self):
        sections = {f["key"]: f["section"] for f in manager.CONFIG_FIELDS}
        self.assertNotIn("REMOVED_BACKEND_MODEL_PATH", sections)
        self.assertNotIn("REMOVED_BACKEND_BIN", sections)

    def test_removed_backend_section_is_not_exposed(self):
        fields = {f["key"]: f for f in manager.CONFIG_FIELDS}
        self.assertNotIn("REMOVED_BACKEND_SPEC_METHOD", fields)
        self.assertNotIn("Removed Backend", manager.CORE_CONFIG_SECTIONS)

    def test_ocr_fields_restart_ocr_only(self):
        self.assertEqual(manager.RESTART_HINTS["OCR_MODEL_PATH"], ["ocr"])
        self.assertEqual(manager.RESTART_HINTS["OCR_PORT"], ["ocr"])

    def test_ocr_gpu_placement_fields_are_exposed(self):
        fields = {f["key"]: f for f in manager.CONFIG_FIELDS}
        for key in (
            "OCR_GPU_VISIBLE_DEVICES",
            "OCR_MAIN_GPU",
            "OCR_DEVICE",
            "OCR_SPLIT_MODE",
            "OCR_TENSOR_SPLIT",
        ):
            self.assertIn(key, fields)
            self.assertEqual(manager.RESTART_HINTS[key], ["ocr"])
        self.assertIn("0,1", fields["OCR_GPU_VISIBLE_DEVICES"].get("hint", ""))
        self.assertIn("none", fields["OCR_SPLIT_MODE"]["options"])

    def test_ocr_gpu_default_follows_chat_gpu_devices(self):
        env = config_env.normalize_env_keys({"CHAT_GPU_VISIBLE_DEVICES": "0,1"})
        self.assertEqual(env["OCR_GPU_VISIBLE_DEVICES"], "0,1")

    def test_chat_template_fields_are_exposed(self):
        fields = {f["key"]: f for f in manager.CONFIG_FIELDS}
        self.assertEqual(fields["CHAT_TEMPLATE_MANAGER"]["type"], "template_manager")
        self.assertEqual(fields["LLM_A_TEMPLATE_ID"]["type"], "chat_template")
        self.assertEqual(fields["LLM_B_TEMPLATE_ID"]["type"], "chat_template")
        self.assertEqual(fields["TASK_CHAT_TEMPLATE_ID"]["type"], "chat_template")

    def test_glmocr_sdk_fields_restart_sdk_only(self):
        self.assertIn("GLM-OCR SDK", manager.CORE_CONFIG_SECTIONS)
        self.assertEqual(manager.RESTART_HINTS["GLMOCR_SDK_PORT"], ["glmocr-sdk"])
        self.assertEqual(manager.RESTART_HINTS["GLMOCR_LAYOUT_DEVICE"], ["glmocr-sdk"])

    def test_glmocr_sdk_layout_gpu_is_never_comma_separated(self):
        env = config_env.normalize_env_keys({
            "OCR_GPU_VISIBLE_DEVICES": "0,1",
            "GLMOCR_LAYOUT_CUDA_VISIBLE_DEVICES": "0,1",
            "GLMOCR_LAYOUT_DEVICE": "cuda:0,1",
        })
        self.assertEqual(env["OCR_GPU_VISIBLE_DEVICES"], "0,1")
        self.assertEqual(env["GLMOCR_LAYOUT_CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(env["GLMOCR_LAYOUT_DEVICE"], "cuda:0")

    def test_glmocr_sdk_layout_gpu_default_does_not_inherit_ocr_multi_gpu(self):
        env = config_env.normalize_env_keys({"OCR_GPU_VISIBLE_DEVICES": "0,1"})
        self.assertEqual(env["OCR_GPU_VISIBLE_DEVICES"], "0,1")
        self.assertEqual(env["GLMOCR_LAYOUT_CUDA_VISIBLE_DEVICES"], "")

    def test_gguf_mmproj_classifier_keeps_model_and_projector_separate(self):
        self.assertFalse(models.is_mmproj_gguf("Qwen3.5-27B-Q4_K_M.gguf", 16 * 1024**3))
        self.assertTrue(models.is_mmproj_gguf("mmproj-Qwen3.5-27B-f16.gguf", 800 * 1024**2))
        self.assertTrue(models.is_mmproj_gguf("Qwen3.5-27B.projector.gguf", 800 * 1024**2))
        self.assertTrue(models.is_mmproj_gguf("vision-clip.gguf", 800 * 1024**2))

    def test_update_env_values_does_not_write_unrelated_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_file = pathlib.Path(tmp) / "llm-stack.env"
            config_file.write_text("CHAT_TEMP=1.0\n")
            with patch.object(core, "CONFIG_FILE", config_file):
                config_env.update_env_values({"CHAT_TEMP": "0.6"})

            content = config_file.read_text()

        self.assertIn("CHAT_TEMP=0.6", content)
        self.assertNotIn("REMOVED_BACKEND_LABEL=", content)
        self.assertNotIn("CHAT_DENSE_LABEL=", content)


class OcrExtractTests(unittest.TestCase):
    def test_ocr_extract_builds_multimodal_payload(self):
        captured = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps({"choices": [{"message": {"content": "hello"}}]}).encode()

        def fake_urlopen(req, timeout=0):
            captured["url"] = req.full_url
            captured["payload"] = json.loads(req.data.decode())
            captured["timeout"] = timeout
            return FakeResponse()

        env = {
            "OCR_HOST": "0.0.0.0",
            "OCR_PORT": "8009",
            "OCR_MODEL_NAME": "ocr",
            "OCR_PROMPT": "OCR",
            "OCR_TEMP": "0.1",
            "OCR_TOP_P": "0.95",
            "OCR_TOP_K": "1",
            "OCR_MIN_P": "0.00",
        }
        with (
            manager.app.test_client() as client,
            patch.object(config_env, "read_env", return_value=env),
            patch.object(manager.urlrequest, "urlopen", side_effect=fake_urlopen),
        ):
            resp = client.post("/api/ocr/extract", json={"image_base64": "abc", "mime_type": "image/png"})

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["text"], "hello")
        self.assertEqual(captured["url"], "http://127.0.0.1:8009/v1/chat/completions")
        content = captured["payload"]["messages"][0]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "OCR"})
        self.assertEqual(content[1]["image_url"]["url"], "data:image/png;base64,abc")

    def test_ocr_parse_forwards_sdk_payload_and_normalizes_response(self):
        captured = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps({
                    "markdown_result": "# Parsed",
                    "json_result": [[{"label": "text", "content": "Parsed"}]],
                    "layout_details": [[{"label": "text", "content": "Parsed"}]],
                }).encode()

        def fake_urlopen(req, timeout=0):
            captured["url"] = req.full_url
            captured["payload"] = json.loads(req.data.decode())
            captured["timeout"] = timeout
            return FakeResponse()

        env = {
            "GLMOCR_PUBLIC_URL": "http://127.0.0.1:5002/glmocr/parse",
            "GLMOCR_OCR_REQUEST_TIMEOUT": "222",
        }
        with (
            manager.app.test_client() as client,
            patch.object(config_env, "read_env", return_value=env),
            patch.object(manager.urlrequest, "urlopen", side_effect=fake_urlopen),
        ):
            resp = client.post(
                "/api/ocr/parse",
                json={
                    "images": ["/tmp/doc.pdf"],
                    "need_layout_visualization": True,
                    "start_page_id": 2,
                    "end_page_id": 3,
                },
            )

        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["text"], "# Parsed")
        self.assertEqual(captured["url"], "http://127.0.0.1:5002/glmocr/parse")
        self.assertEqual(captured["timeout"], 222)
        self.assertEqual(captured["payload"]["images"], ["/tmp/doc.pdf"])
        self.assertTrue(captured["payload"]["need_layout_visualization"])
        self.assertEqual(captured["payload"]["start_page_id"], 2)
        self.assertEqual(captured["payload"]["end_page_id"], 3)

    def test_transcribe_overview_merges_the_sidecar_body(self):
        """The sidecar's own reply carries `ok`, so splatting it into
        `jsonify(ok=True, **data)` is a TypeError and a 500 on every call."""
        sidecar = {"ok": True, "active_engine": "faster-whisper", "resident": None,
                   "engines": [{"id": "faster-whisper", "runtime": "faster-whisper"}]}
        env = {"TRANSCRIPT_ENABLED": "on", "TRANSCRIPT_HOST": "127.0.0.1",
               "TRANSCRIPT_PORT": "8014"}
        with (
            manager.app.test_client() as client,
            patch.object(config_env, "read_env", return_value=env),
            patch.object(core, "http_json", return_value=sidecar),
        ):
            resp = client.get("/api/transcribe/overview")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertTrue(body["ok"] and body["enabled"] and body["reachable"])
        self.assertEqual(body["active_engine"], "faster-whisper")

    def test_transcribe_overview_does_not_dial_a_disabled_sidecar(self):
        with (
            manager.app.test_client() as client,
            patch.object(config_env, "read_env", return_value={"TRANSCRIPT_ENABLED": "off"}),
            patch.object(core, "http_json", side_effect=AssertionError("must not be called")),
        ):
            body = client.get("/api/transcribe/overview").get_json()
        self.assertFalse(body["enabled"])
        self.assertFalse(body["reachable"])

    def test_transcribe_url_rewrites_an_unreachable_bind_address(self):
        """`TRANSCRIPT_HOST` may be a wildcard or the literal ${LISTEN_HOST},
        which systemd's EnvironmentFile does not expand. Neither is dialable."""
        self.assertEqual(
            manager._transcribe_base_url({"TRANSCRIPT_HOST": "0.0.0.0", "TRANSCRIPT_PORT": "8014"}),
            "http://127.0.0.1:8014")
        self.assertEqual(
            manager._transcribe_base_url({"TRANSCRIPT_HOST": "${LISTEN_HOST}",
                                          "LISTEN_HOST": "10.0.0.5", "TRANSCRIPT_PORT": "8014"}),
            "http://10.0.0.5:8014")

    def test_ocr_parse_wraps_base64_input(self):
        captured = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps({"md_results": "ok"}).encode()

        def fake_urlopen(req, timeout=0):
            captured["payload"] = json.loads(req.data.decode())
            return FakeResponse()

        with (
            manager.app.test_client() as client,
            patch.object(config_env, "read_env", return_value={"GLMOCR_SDK_HOST": "0.0.0.0", "GLMOCR_SDK_PORT": "5002"}),
            patch.object(manager.urlrequest, "urlopen", side_effect=fake_urlopen),
        ):
            resp = client.post("/api/ocr/parse", json={"image_base64": "abc", "mime_type": "application/pdf"})

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(captured["payload"]["images"], ["data:application/pdf;base64,abc"])


class ChatTemplateTests(unittest.TestCase):
    def test_list_chat_templates_includes_jinja_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            template_dir = pathlib.Path(tmp)
            (template_dir / "custom.jinja").write_text("{{ messages }}")
            (template_dir / "templates.json").write_text(json.dumps({
                "custom": {"name": "Custom Template", "description": "desc", "updated_at": 123}
            }))
            with (
                patch.object(core, "CHAT_TEMPLATES_DIR", template_dir),
                patch.object(core, "CHAT_TEMPLATES_META_FILE", template_dir / "templates.json"),
            ):
                templates = models.list_chat_templates()

        self.assertEqual(templates[0]["id"], "")
        self.assertIn("custom", {item["id"] for item in templates})
        custom = next(item for item in templates if item["id"] == "custom")
        self.assertEqual(custom["name"], "Custom Template")


class HuggingFaceRepoFileTests(unittest.TestCase):
    def test_repo_files_split_model_and_mmproj_candidates(self):
        repo_ref = {
            "repo_id": "owner/repo",
            "revision": "main",
            "repo_url": "https://huggingface.co/owner/repo",
        }
        files = [
            {"path": "Qwen3.5-27B-Q4_K_M.gguf", "name": "Qwen3.5-27B-Q4_K_M.gguf", "size": 16 * 1024**3},
            {"path": "mmproj-Qwen3.5-27B-f16.gguf", "name": "mmproj-Qwen3.5-27B-f16.gguf", "size": 800 * 1024**2},
            {"path": "notes.txt", "name": "notes.txt", "size": 100},
        ]
        with (
            manager.app.test_client() as client,
            patch.object(models, "parse_huggingface_repo_ref", return_value=repo_ref),
            patch.object(models, "list_huggingface_repo_files", return_value=files),
        ):
            resp = client.post("/api/huggingface/repo-files", json={"repo_url": "owner/repo"})

        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertEqual([item["name"] for item in body["model_files"]], ["Qwen3.5-27B-Q4_K_M.gguf"])
        self.assertEqual([item["name"] for item in body["mmproj_files"]], ["mmproj-Qwen3.5-27B-f16.gguf"])
        self.assertEqual(body["model_files"][0]["matched_mmproj"], "mmproj-Qwen3.5-27B-f16.gguf")


class CustomModelApiTests(unittest.TestCase):
    def test_add_custom_model_derives_names_from_model_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            custom_models_file = pathlib.Path(tmp) / "custom-models.json"
            with (
                manager.app.test_client() as client,
                patch.object(core, "CUSTOM_MODELS_FILE", custom_models_file),
            ):
                resp = client.post(
                    "/api/custom-models",
                    json={"model_path": "/models/Qwen3.5-27B-Q4_K_M.gguf"},
                )

        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["model"]["display_name"], "Qwen3.5-27B-Q4_K_M")
        self.assertEqual(body["model"]["model_name"], "qwen3.5-27b-q4_k_m")


class SavedConfigTests(unittest.TestCase):
    def test_save_current_records_exact_form_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved_dir = pathlib.Path(tmp)
            env = {
                "LLM_A_CTX_SIZE": "32768",
                "LLM_A_SPLIT_MODE": "layer",
                "LLM_A_TENSOR_SPLIT": "1,1",
                "LLM_A_GPU_VISIBLE_DEVICES": "0,1",
            }
            form = {
                "LLM_A_CTX_SIZE": "128000",
                "LLM_A_SPLIT_MODE": "none",
                "LLM_A_TENSOR_SPLIT": "",
                "LLM_A_GPU_VISIBLE_DEVICES": "0",
            }

            with (
                manager.app.test_client() as client,
                patch.object(core, "SAVED_CONFIGS_DIR", saved_dir),
                patch.object(config_env, "read_env", return_value=env),
                patch.object(manager, "get_service_status", return_value="inactive"),
            ):
                resp = client.post("/api/saved-configs", json={"name": "OneGpu", "config": form})

            self.assertEqual(resp.status_code, 200)
            data = json.loads((saved_dir / "OneGpu.json").read_text())
            self.assertEqual(data["_config_form"]["LLM_A_CTX_SIZE"], "128000")
            self.assertEqual(data["_config_form"]["LLM_A_SPLIT_MODE"], "none")
            self.assertEqual(data["_config_form"]["LLM_A_TENSOR_SPLIT"], "")
            self.assertEqual(data["_config_form"]["LLM_A_GPU_VISIBLE_DEVICES"], "0")

    def test_apply_saved_config_prefers_form_snapshot_over_top_level_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved_dir = pathlib.Path(tmp)
            config_file = saved_dir / "llm-stack.env"
            config_file.write_text(
                "LLM_A_CTX_SIZE=32768\n"
                "LLM_A_SPLIT_MODE=layer\n"
                "LLM_A_TENSOR_SPLIT=1,1\n"
                "LLM_A_GPU_VISIBLE_DEVICES=0,1\n"
            )
            (saved_dir / "OneGpu.json").write_text(json.dumps({
                "LLM_A_CTX_SIZE": "262144",
                "CHAT_CTX_SIZE": "262144",
                "LLM_A_SPLIT_MODE": "layer",
                "LLM_A_TENSOR_SPLIT": "1,1",
                "LLM_A_GPU_VISIBLE_DEVICES": "0,1",
                "_config_form": {
                    "LLM_A_CTX_SIZE": "128000",
                    "LLM_A_SPLIT_MODE": "none",
                    "LLM_A_TENSOR_SPLIT": "",
                    "LLM_A_GPU_VISIBLE_DEVICES": "0",
                },
                "_active_chat_model": {"variant": None, "service": None, "label": "", "kind": "none"},
            }))

            with (
                patch.object(core, "SAVED_CONFIGS_DIR", saved_dir),
                patch.object(core, "CONFIG_FILE", config_file),
            ):
                result = manager.apply_saved_config("OneGpu", launch=False)
            content = config_file.read_text()

            self.assertTrue(result["ok"])
            self.assertIn("LLM_A_CTX_SIZE=128000", content)
            self.assertIn('LLM_A_TENSOR_SPLIT=""', content)
            self.assertIn("LLM_A_SPLIT_MODE=none", content)
            self.assertIn("LLM_A_GPU_VISIBLE_DEVICES=0", content)

    def test_apply_saved_config_prefers_canonical_pane_key_over_legacy_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved_dir = pathlib.Path(tmp)
            config_file = saved_dir / "llm-stack.env"
            config_file.write_text("CHAT_DENSE_CTX_SIZE=32768\nCHAT_CTX_SIZE=32768\n")
            (saved_dir / "Context.json").write_text(json.dumps({
                "LLM_A_CTX_SIZE": "128000",
                "CHAT_DENSE_CTX_SIZE": "32768",
                "CHAT_CTX_SIZE": "262144",
                "_active_chat_model": {"variant": None, "service": None, "label": "", "kind": "none"},
            }))

            with (
                patch.object(core, "SAVED_CONFIGS_DIR", saved_dir),
                patch.object(core, "CONFIG_FILE", config_file),
            ):
                result = manager.apply_saved_config("Context", launch=False)
            content = config_file.read_text()

            self.assertTrue(result["ok"])
            self.assertIn("LLM_A_CTX_SIZE=128000", content)
            self.assertNotIn("CHAT_DENSE_CTX_SIZE=32768", content)

    def test_apply_saved_config_accepts_numeric_json_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved_dir = pathlib.Path(tmp)
            config_file = saved_dir / "llm-stack.env"
            config_file.write_text("LLM_A_CTX_SIZE=32768\n")
            (saved_dir / "Numeric.json").write_text(json.dumps({
                "LLM_A_CTX_SIZE": 128000,
                "_active_chat_model": {"variant": None, "service": None, "label": "", "kind": "none"},
            }))

            with (
                patch.object(core, "SAVED_CONFIGS_DIR", saved_dir),
                patch.object(core, "CONFIG_FILE", config_file),
            ):
                result = manager.apply_saved_config("Numeric", launch=False)

            self.assertTrue(result["ok"])
            self.assertIn("LLM_A_CTX_SIZE=128000", config_file.read_text())

    def test_save_records_primary_and_secondary_backend_slots(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved_dir = pathlib.Path(tmp)
            env = {
                "LLM_A_LABEL": "LLM A",
                "LLM_B_LABEL": "Local Secondary",
                "LLM_B_MODEL_NAME": "llm-b",
                "LLM_B_MODEL_PATH": "/models/secondary.gguf",
            }

            def fake_status(name):
                return "active" if name in {"llm-a", "llm-b"} else "inactive"

            with (
                manager.app.test_client() as client,
                patch.object(core, "SAVED_CONFIGS_DIR", saved_dir),
                patch.object(config_env, "read_env", return_value=env),
                patch.object(manager, "get_service_status", side_effect=fake_status),
            ):
                resp = client.post("/api/saved-configs", json={"name": "Both"})
                listed = client.get("/api/saved-configs")

        self.assertEqual(resp.status_code, 200)
        body = listed.get_json()[0]
        self.assertEqual(body["active_backend_slots"]["primary"]["label"], "LLM A")
        self.assertEqual(body["active_backend_slots"]["secondary"]["label"], "Local Secondary")
        self.assertEqual(body["active_backend_slots"]["secondary"]["service"], "llm-b")

    def test_apply_saved_config_launches_secondary_from_slot_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved_dir = pathlib.Path(tmp)
            config_file = saved_dir / "llm-stack.env"
            config_file.write_text("LLM_B_LABEL=Old\n")
            (saved_dir / "Secondary.json").write_text(json.dumps({
                "LLM_B_LABEL": "Local Secondary",
                "_active_chat_model": {"variant": None, "service": None, "label": "", "kind": "none"},
                "_active_backend_slots": {
                    "secondary": {
                        "variant": "secondary",
                        "service": "llm-b",
                        "label": "Local Secondary",
                        "kind": "secondary",
                    }
                },
            }))
            started = []

            def fake_status(_name):
                return "inactive"

            class FakeServiceManager(core.ServiceManager):
                @classmethod
                def start(cls, name, timeout=30):
                    started.append(name)
                    return manager.subprocess.CompletedProcess(["start", name], 0, "", "")

            with (
                patch.object(core, "SAVED_CONFIGS_DIR", saved_dir),
                patch.object(core, "CONFIG_FILE", config_file),
                patch.object(manager, "get_service_status", side_effect=fake_status),
                patch.object(core, "ServiceManager", FakeServiceManager),
            ):
                result = manager.apply_saved_config("Secondary", launch=True)
            content = config_file.read_text()

            self.assertTrue(result["ok"])
            self.assertIn("llm-b", started)
            self.assertIn("llm-b-proxy", started)
            self.assertIn('LLM_B_LABEL="Local Secondary"', content)

    def test_patch_saved_config_updates_only_supplied_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved_dir = pathlib.Path(tmp)
            saved_path = saved_dir / "Default.json"
            saved_path.write_text(json.dumps({
                "CHAT_TEMP": "1.0",
                "CHAT_TOP_K": "20",
                "_name": "Default",
            }))
            with patch.object(core, "SAVED_CONFIGS_DIR", saved_dir):
                result = manager.update_saved_config_values("Default", {"CHAT_TEMP": "0.6"})
            data = json.loads(saved_path.read_text())

        self.assertTrue(result["ok"])
        self.assertEqual(data["CHAT_TEMP"], "0.6")
        self.assertEqual(data["CHAT_TOP_K"], "20")
        self.assertEqual(data["_name"], "Default")


class MetricsFlagTests(unittest.TestCase):
    """`--metrics` is what promotes the telemetry panel from journal-derived
    stats to backend counters, so every llama.cpp backend must expose it."""

    def test_every_llamacpp_backend_exposes_a_metrics_toggle(self):
        sections = {f["key"]: f["section"] for f in manager.CONFIG_FIELDS}
        for key, section in [
            ("LLM_A_METRICS", "LLM A"),
            ("LLM_B_METRICS", "LLM B"),
            ("EMBED_METRICS", "Embedding"),
            ("RERANK_METRICS", "Reranker"),
            ("TASK_METRICS", "Task Model"),
            ("OCR_METRICS", "OCR"),
        ]:
            self.assertEqual(sections.get(key), section, key)

    def test_metrics_changes_restart_only_their_own_backend(self):
        self.assertEqual(manager.RESTART_HINTS["LLM_A_METRICS"], ["llm-a"])
        self.assertEqual(manager.RESTART_HINTS["LLM_B_METRICS"], ["llm-b"])
        self.assertEqual(manager.RESTART_HINTS["EMBED_METRICS"], ["embed"])
        self.assertEqual(manager.RESTART_HINTS["TASK_METRICS"], ["task"])

    def test_every_slot_gates_the_flag_on_its_own_env_key(self):
        """Asserted as behaviour rather than as source text.

        Half of this used to grep the launchers for the line that emitted
        `--metrics`. There are no such lines left -- the gate is a Toggle in
        `web/backends/slots.py` -- and a grep would in any case have passed for
        a launcher that never reached its exec, which is the failure the
        launcher harness exists because of.
        """
        env = {"LLAMA_SERVER_BIN": "/bin/llama-server", "LISTEN_HOST": "127.0.0.1",
               "EMBEDDING_MODEL_PATH": "/m.gguf", "RERANKER_MODEL_PATH": "/m.gguf",
               "OCR_MODEL_PATH": "/m.gguf", "LLM_A_MODEL_PATH": "/m.gguf",
               "LLM_B_MODEL_PATH": "/m.gguf", "TASK_MODEL_PATH": "/m.gguf"}
        with platform_harness.as_linux():
            import backends
            for slot, prefix in (("embed", "EMBED"), ("rerank", "RERANK"), ("ocr", "OCR"),
                                 ("llm-a", "LLM_A"),
                                 ("llm-b", "LLM_B"), ("task", "TASK")):
                with self.subTest(slot):
                    self.assertIn("--metrics", backends.build_command(slot, dict(env)))
                    self.assertNotIn("--metrics", backends.build_command(
                        slot, dict(env, **{f"{prefix}_METRICS": "off"})))

    def test_mtp_runs_without_a_draft_model(self):
        """Most MTP GGUFs carry their own blk.N.nextn.* head, and llama.cpp
        builds the MTP draft context against the *target* model when no draft
        is given -- the `else if (spec_mtp)` branch of
        common_speculative_init_result. Demanding a draft path for draft-mtp
        would exit 1 on precisely the models that need no sidecar.
        """
        import backends
        self.assertNotIn("draft-mtp", backends.speculative.NEEDS_DRAFT_MODEL)
        args, _said = backends.speculative.build({"X_SPEC_METHOD": "draft-mtp"}, ("X",))
        self.assertIn("--spec-type", args)
        self.assertNotIn("--spec-draft-model", args)

    def test_thinking_level_reaches_every_chat_launcher(self):
        env = {"LLAMA_SERVER_BIN": "/bin/llama-server", "LISTEN_HOST": "127.0.0.1",
               "LLM_A_MODEL_PATH": "/m.gguf", "LLM_B_MODEL_PATH": "/m.gguf",
               "TASK_MODEL_PATH": "/m.gguf", "TASK_THINKING": "on"}
        with platform_harness.as_linux():
            import backends
            for slot, prefix in (("llm-a", "LLM_A"),
                                 ("llm-b", "LLM_B"), ("task", "TASK")):
                with self.subTest(slot):
                    argv = backends.build_command(
                        slot, dict(env, **{f"{prefix}_REASONING_EFFORT": "medium"}))
                    kwargs = argv[argv.index("--chat-template-kwargs") + 1]
                    self.assertIn('"reasoning_effort": "medium"', kwargs)

    def test_metrics_toggle_is_a_recognised_config_key(self):
        filtered = config_env.filter_config_updates({"LLM_A_METRICS": "off"}, env={})
        self.assertEqual(filtered, {"LLM_A_METRICS": "off"})


class UpdateCliTests(unittest.TestCase):
    """The fast update path must not rebuild llama.cpp or reload models."""

    @classmethod
    def setUpClass(cls):
        root = pathlib.Path(__file__).resolve().parents[1]
        cls.cli = (root / "scripts" / "llm-stack-manager").read_text()
        cls.update = (root / "update.sh").read_text()
        cls.install = (root / "install.sh").read_text()

    def test_cli_is_executable(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        self.assertTrue((root / "scripts" / "llm-stack-manager").stat().st_mode & 0o111)

    def test_plain_update_takes_the_manager_only_path(self):
        self.assertIn('exec bash "${STACK_DIR}/update.sh" --manager-only', self.cli)

    def test_full_update_is_opt_in(self):
        self.assertIn("--full", self.cli)
        self.assertIn('exec bash "${STACK_DIR}/update.sh"\n', self.cli)

    def test_manager_only_skips_dependency_builds(self):
        # --skip-deps is what avoids install-dependencies.py, i.e. the cmake rebuild.
        self.assertRegex(self.update, r"--manager-only\)\s*\n\s*MANAGER_ONLY=1\s*\n\s*SKIP_DEPS=1")

    def test_skip_deps_is_handed_down_to_install(self):
        """Regression: --skip-deps only skipped update.sh's own dependency call.

        install.sh then ran install-dependencies.py --update itself and
        rebuilt llama.cpp from source anyway, so a "fast" update still paid for
        a full CUDA compile.
        """
        self.assertRegex(
            self.update,
            r'env LLM_STACK_SKIP_DEP_UPDATE="\$\{SKIP_DEPS\}"[^\n]*\\\n\s*'
            r'LLM_STACK_SKIP_EXTERNAL_INSTALL="\$\{MANAGER_ONLY\}"[^\n]*\\\n\s*'
            r'bash "\$\{STACK_DIR\}/install\.sh"',
        )

    def test_install_honours_the_dependency_skip_gate(self):
        """The other half of the same regression: the gate must exist and wrap
        the only call that builds llama.cpp."""
        gate = re.search(
            r'if \[\[ "\$\{LLM_STACK_SKIP_DEP_UPDATE:-0\}" == "1" \]\]; then(.*?)\nfi',
            self.install, re.S)
        self.assertIsNotNone(gate, "install.sh lost its dependency-update gate")
        self.assertIn("install-dependencies.py", gate.group(1))
        # And no ungated build call survives elsewhere in the file.
        for line in self.install.splitlines():
            if "install-dependencies.py" in line and "--update" in line:
                self.assertIn(line, gate.group(1),
                              f"ungated dependency build: {line.strip()}")

    def test_manager_only_restarts_no_model_backend(self):
        cheap = re.search(r"CHEAP_RESTART_SERVICES=\(([^)]*)\)", self.update)
        self.assertIsNotNone(cheap)
        services = cheap.group(1).split()
        for backend in ("llm-a",
                        "llm-b", "embed", "rerank", "task", "ocr"):
            self.assertNotIn(backend, services)
        self.assertIn("llm-manager", services)
        self.assertIn("llm-a-proxy", services)

    def test_stale_backend_launchers_are_reported(self):
        self.assertIn("changed_backend_files", self.update)
        self.assertIn("still running the previous code", self.update)

    def test_shared_launcher_code_counts_as_backend_sensitive(self):
        """Launchers source scripts/lib and consult the budget model at startup,
        so a change to either means a running backend is on stale code."""
        paths = re.search(r"BACKEND_SENSITIVE_PATHS=\(([^)]*)\)", self.update)
        self.assertIsNotNone(paths)
        self.assertIn("scripts/lib/", paths.group(1))
        self.assertIn("web/budget.py", paths.group(1))

    def test_install_links_the_cli_onto_path(self):
        self.assertIn('ln -sfn "${STACK_DIR}/scripts/llm-stack-manager" /usr/local/bin/llm-stack-manager',
                      self.install)


class BudgetRouteTests(unittest.TestCase):
    """The endpoint Agent D's recommended-preset work reads from.

    GGUF geometry has to be read server-side — the browser cannot open a 22 GB
    model — so this is the only place a configuration can be priced.
    """

    GPUS = [{"index": 0, "mem_total": 24576, "mem_used": 1000},
            {"index": 1, "mem_total": 24576, "mem_used": 1000}]
    MEMINFO = {"MemTotal": 32787000, "MemAvailable": 20275000,
               "SwapTotal": 8388604, "SwapFree": 8388604}

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        # Reuse the GGUF writer and fixture from the budget tests rather than
        # keeping a second copy of the model metadata in sync.
        spec = importlib.util.spec_from_file_location(
            "llm_stack_manager_budget_tests",
            pathlib.Path(__file__).resolve().parent / "test_budget.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.model = module.write_gguf(
            pathlib.Path(self._tmp.name) / "model.gguf", module.QWEN36_27B)

    def _client(self, env):
        return (
            manager.app.test_client(),
            patch.object(config_env, "read_env", return_value=env),
            patch.object(manager, "get_gpu_info", return_value=self.GPUS),
            patch.object(core, "read_meminfo", return_value=self.MEMINFO),
        )

    def test_budget_prices_the_configured_backend(self):
        env = {"LLM_A_MODEL_PATH": str(self.model),
               "LLM_A_CTX_SIZE": "262144", "LLM_A_N_PARALLEL": "2",
               "LLM_A_CTX_CHECKPOINTS": "8", "LLM_A_CACHE_RAM": "8192"}
        client, *patches = self._client(env)
        with client, patches[0], patches[1], patches[2]:
            response = client.get("/api/backend/budget")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIsNone(payload["error"])
        self.assertEqual(payload["prediction"]["per_slot_context"], 131072)
        self.assertEqual(payload["geometry"]["recurrent_layers"], 48)

    def test_query_parameters_price_an_unsaved_edit(self):
        env = {"LLM_A_MODEL_PATH": str(self.model),
               "LLM_A_CTX_SIZE": "262144", "LLM_A_N_PARALLEL": "2"}
        client, *patches = self._client(env)
        with client, patches[0], patches[1], patches[2]:
            response = client.get("/api/backend/budget?ctx_size=65536")
        self.assertEqual(response.get_json()["prediction"]["per_slot_context"], 32768)

    def test_unknown_query_parameters_are_ignored(self):
        env = {"LLM_A_MODEL_PATH": str(self.model), "LLM_A_CTX_SIZE": "131072"}
        client, *patches = self._client(env)
        with client, patches[0], patches[1], patches[2]:
            response = client.get("/api/backend/budget?model_alias=evil&ctx_size=131072")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["prediction"]["total_context"], 131072)

    def test_unknown_backend_is_rejected(self):
        client, *patches = self._client({})
        with client, patches[0], patches[1], patches[2]:
            response = client.get("/api/backend/budget?backend=nope")
        self.assertEqual(response.status_code, 400)
        self.assertIn("llm-a", response.get_json()["backends"])

    def test_missing_model_reports_without_failing_the_request(self):
        client, *patches = self._client({"LLM_A_MODEL_PATH": "/nope.gguf"})
        with client, patches[0], patches[1], patches[2]:
            response = client.get("/api/backend/budget")
        self.assertEqual(response.status_code, 200)
        self.assertIn("model not found", response.get_json()["error"])

    def test_recommendation_is_derived_from_detected_hardware(self):
        env = {"LLM_A_MODEL_PATH": str(self.model)}
        client, *patches = self._client(env)
        with client, patches[0], patches[1], patches[2]:
            response = client.get("/api/backend/budget/recommend?slots=2")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["parallel"], 2)
        self.assertEqual(payload["per_slot_context"], payload["ctx_size"] // 2)
        # The constant this replaces was 32 on every host and every model.
        self.assertLessEqual(payload["ctx_checkpoints"], 32)
        self.assertGreaterEqual(payload["ctx_checkpoints"], 2)

    def test_recommendation_rejects_an_unreadable_model(self):
        client, *patches = self._client({"LLM_A_MODEL_PATH": "/nope.gguf"})
        with client, patches[0], patches[1], patches[2]:
            response = client.get("/api/backend/budget/recommend")
        self.assertEqual(response.status_code, 400)
        self.assertIn("could not read model metadata", response.get_json()["error"])

    def test_recommendation_stays_within_the_trained_context(self):
        """A recommendation that trips the pre-flight is not a recommendation:
        `evaluate` warns above the trained context, so `recommend` stops there."""
        env = {"LLM_A_MODEL_PATH": str(self.model)}
        client, *patches = self._client(env)
        with client, patches[0], patches[1], patches[2]:
            payload = client.get("/api/backend/budget/recommend?slots=2").get_json()
        self.assertLessEqual(payload["ctx_size"], payload["train_context_length"])

    def test_recommendation_clears_the_flags_it_disables(self):
        env = {"LLM_A_MODEL_PATH": str(self.model)}
        client, *patches = self._client(env)
        with client, patches[0], patches[1], patches[2]:
            payload = client.get("/api/backend/budget/recommend").get_json()
        self.assertEqual(payload["fit"], "off")
        self.assertEqual(payload["fit_ctx"], "")
        # No sliding-window attention in this model, so --swa-full is inert.
        self.assertEqual(payload["swa_full"], "off")


class ConfigFieldRenderingTests(unittest.TestCase):
    """Curated section layouts list the keys each panel holds, so a key in none
    of those lists rendered nowhere — declared, saveable, and invisible. Twenty
    eight fields were in that state, including every backend's --metrics
    toggle."""

    def _render(self, platform=platform_harness.as_linux):
        template_dir = pathlib.Path(__file__).resolve().parents[1] / "web" / "templates"
        original = list(manager.app.jinja_loader.searchpath)
        manager.app.jinja_loader.searchpath = [str(template_dir)]
        try:
            with (
                platform(),
                manager.app.test_client() as client,
                patch.object(config_env, "read_env", return_value={}),
                patch.object(models, "load_custom_models", return_value=[]),
            ):
                return client.get("/").get_data(as_text=True)
        finally:
            manager.app.jinja_loader.searchpath = original

    def test_every_declared_field_reaches_the_page(self):
        html = self._render()
        missing = [f"{f['section']}/{f['key']}" for f in manager.CONFIG_FIELDS
                   if f.get("section") in manager.CORE_CONFIG_SECTIONS
                   and f"cfg-{f['key']}" not in html]
        self.assertEqual(missing, [], f"{len(missing)} config fields render nowhere")

    def test_a_metal_host_is_not_offered_placement_it_cannot_perform(self):
        """The other half of the same rule.

        A field that renders nowhere is invisible; a field that renders on a
        machine that cannot act on it is worse, because it saves, it restarts
        the backend, and the setting never reaches the command line.
        `CUDA_VISIBLE_DEVICES` is not read by a Metal build and
        `resolve_split_opts` collapses every split mode there, so twenty-seven
        controls across seven sections are exactly that.
        """
        html = self._render(platform_harness.as_darwin)
        _fields, omitted = config_fields.applicable_fields(
            platform_harness.DarwinPlatform().inert_config_capabilities)
        # 29 since EMBED and RERANK gained the `_MAIN_GPU` control they were
        # missing: one Metal device means no index to choose, so a Mac is not
        # offered them either. `_DEVICE` is deliberately not withheld -- it is
        # the one placement key a Metal build still reads, and the launcher
        # drops an unusable value with a message rather than failing.
        self.assertEqual(len(omitted), 29)
        for key in omitted:
            with self.subTest(key):
                self.assertNotIn(f'"cfg-{key}"', html)
        # Left out, not silently dropped.
        self.assertIn("settings not shown on this host.", html)
        for reason in set(omitted.values()):
            self.assertIn(reason, html)

    def test_the_settings_a_metal_host_can_act_on_are_still_offered(self):
        # `--device` is not inert on Metal -- MTL0 is a real answer -- so the
        # control stays. Only the shipped CUDA0 default is wrong for the host,
        # which is a different problem.
        html = self._render(platform_harness.as_darwin)
        for key in ("LLM_A_DEVICE", "LLM_A_N_GPU_LAYERS",
                    "LLM_A_CTX_SIZE", "TASK_DEVICE"):
            with self.subTest(key):
                self.assertIn(f"cfg-{key}", html)

    def test_every_field_carries_its_key_in_the_dom(self):
        # `data-cfg-key` is what lets a control be found without re-deriving
        # the curated panel layouts in JavaScript.
        html = self._render()
        for key in ("LLM_A_CTX_SIZE", "OCR_TEMP", "TASK_METRICS"):
            with self.subTest(key):
                self.assertIn(f'data-cfg-key="{key}"', html)

    def test_no_field_is_rendered_twice(self):
        """The catch-all must not duplicate what a curated panel already drew."""
        counts = Counter(re.findall(r'id="cfg-([A-Z0-9_]+)"', self._render()))
        self.assertEqual({key: n for key, n in counts.items() if n > 1}, {})

    def test_the_metrics_toggles_are_reachable(self):
        """Named directly: the backend telemetry work added --metrics keys that
        no panel listed, so the flag it exists to set could not be set."""
        html = self._render()
        for key in ("LLM_A_METRICS", "LLM_B_METRICS", "TASK_METRICS", "OCR_METRICS"):
            self.assertIn(f"cfg-{key}", html, key)

    def test_the_upstream_capture_toggle_is_reachable(self):
        self.assertIn("cfg-UPSTREAM_400_CAPTURE_ENABLED", self._render())


class ContextAccountingTests(unittest.TestCase):
    """--ctx-size is a total that llama.cpp divides by --parallel. Reporting only
    the total is how a backend the UI called 262144 rejected a 155751-token
    request."""

    def test_status_reports_per_slot_context_for_each_backend(self):
        env = {"LLM_A_CTX_SIZE": "262144", "LLM_A_N_PARALLEL": "2",
               "LLM_B_CTX_SIZE": "65536", "LLM_B_N_PARALLEL": "1"}
        with patch.object(config_env, "read_env", return_value=env):
            summary = manager.backend_context_summary()
        self.assertEqual(summary["llm-a"]["per_slot_context"], 131072)
        self.assertEqual(summary["llm-a"]["total_context"], 262144)
        self.assertEqual(summary["llm-b"]["per_slot_context"], 65536)

    def test_a_missing_parallel_setting_means_one_slot(self):
        with patch.object(config_env, "read_env", return_value={"EMBED_CTX_SIZE": "8192"}):
            summary = manager.backend_context_summary()
        self.assertEqual(summary["embed"], {"total_context": 8192, "slots": 1,
                                            "per_slot_context": 8192})

    def test_unconfigured_backends_are_omitted_rather_than_reported_as_zero(self):
        with patch.object(config_env, "read_env", return_value={"LLM_A_CTX_SIZE": "0"}):
            self.assertNotIn("llm-a", manager.backend_context_summary())

    def test_a_non_numeric_context_does_not_break_the_status_poll(self):
        with patch.object(config_env, "read_env", return_value={"OCR_CTX_SIZE": "lots"}):
            self.assertNotIn("ocr", manager.backend_context_summary())


class ServiceHealthTests(unittest.TestCase):
    """"Active" answered whether a process is running. It was being read as
    whether the service works, and those came apart on this box: the OCR SDK
    runs happily with its OCR backend dead."""

    @staticmethod
    def _systemctl(**states):
        """Fake `systemctl show`, keyed by unit, returning ActiveState etc."""
        def run(cmd, timeout=30):
            unit = cmd[2] if len(cmd) > 2 else ""
            spec = states.get(unit, ("inactive", "loaded"))
            active_state, load_state = spec[0], spec[1]
            restarts = spec[2] if len(spec) > 2 else 0
            output = (f"LoadState={load_state}\nActiveState={active_state}\n"
                      f"SubState=dead\nResult=success\nMainPID=0\n"
                      f"NRestarts={restarts}\n")
            return subprocess.CompletedProcess(cmd, 0, output, "")
        return run

    def test_a_unit_mid_launch_is_its_own_state(self):
        # A service that cannot start spends most of its life here, because
        # Restart= bounces it out of `failed` within seconds.
        with platform_harness.as_linux(run_cmd=self._systemctl(ocr=("activating", "loaded"))):
            self.assertEqual(manager.get_service_status("ocr"), "starting")

    def test_statuses_and_restart_counts_come_from_one_pass(self):
        calls = []

        def run(cmd, timeout=30):
            calls.append(cmd[2])
            return subprocess.CompletedProcess(
                cmd, 0, "LoadState=loaded\nActiveState=activating\nSubState=start\n"
                        "Result=exit-code\nMainPID=0\nNRestarts=32\n", "")

        with (
            patch.object(config_env, "read_env", return_value={}),
            patch.object(manager, "patch_service_labels", return_value=[{"name": "ocr"}]),
            platform_harness.as_linux(run_cmd=run),
        ):
            statuses, restarts = manager.service_unit_snapshot()

        self.assertEqual(statuses["ocr"], "starting")
        self.assertEqual(restarts["ocr"], 32)
        # One systemctl call per unit, not one per question asked about it.
        self.assertEqual(len(calls), len(set(calls)))

    def test_a_crashed_unit_is_no_longer_indistinguishable_from_a_stopped_one(self):
        with platform_harness.as_linux(run_cmd=self._systemctl(embed=("failed", "loaded"))):
            self.assertEqual(manager.get_service_status("embed"), "failed")
            self.assertEqual(manager.get_service_status("rerank"), "inactive")

    def test_an_uninstalled_unit_is_unknown(self):
        with platform_harness.as_linux(run_cmd=self._systemctl(ghost=("inactive", "not-found"))):
            self.assertEqual(manager.get_service_status("ghost"), "unknown")

    def test_unit_state_reads_load_and_activation_in_one_call(self):
        calls = []

        def run(cmd, timeout=30):
            calls.append(cmd)
            return subprocess.CompletedProcess(
                cmd, 0, "LoadState=loaded\nActiveState=active\nSubState=running\n"
                        "Result=success\nMainPID=4242\n", "")

        with platform_harness.as_linux(run_cmd=run):
            state = core.ServiceManager.state("embed")
        self.assertEqual(len(calls), 1)
        self.assertTrue(state["active"] and state["installed"])
        self.assertEqual(state["main_pid"], 4242)

    def test_status_carries_health_without_changing_the_services_map(self):
        env = {"LLM_A_CTX_SIZE": "262144", "LLM_A_N_PARALLEL": "2"}
        statuses = {"glmocr-sdk": "active", "ocr": "inactive"}
        with (
            manager.app.test_client() as client,
            patch.object(config_env, "read_env", return_value=env),
            patch.object(manager, "service_unit_snapshot", return_value=(statuses, {})),
            patch.object(manager, "patch_service_labels",
                         return_value=[{"name": "glmocr-sdk"}, {"name": "ocr"}]),
            patch.object(manager, "get_gpu_info", return_value=[]),
            patch.object(manager.health.PROBER, "start"),
            patch.object(manager.health.PROBER, "snapshot", return_value={}),
            patch.object(manager.health, "read_expectations", return_value={}),
        ):
            payload = client.get("/api/status").get_json()

        # The panel's contract with this map has not changed: name to string.
        self.assertEqual(payload["services"], statuses)
        self.assertEqual(payload["health"]["glmocr-sdk"]["state"], "degraded")
        self.assertIn("ocr", payload["health"]["glmocr-sdk"]["reason"])

    def test_a_flapping_service_is_reported_as_failed_through_the_api(self):
        """Live case: ocr at 32 restarts rendered grey or green by turns."""
        manager.health.RESTARTS.reset()
        env = {}
        with (
            manager.app.test_client() as client,
            patch.object(config_env, "read_env", return_value=env),
            patch.object(manager, "patch_service_labels", return_value=[{"name": "ocr"}]),
            patch.object(manager, "get_gpu_info", return_value=[]),
            patch.object(manager.health.PROBER, "start"),
            patch.object(manager.health.PROBER, "snapshot", return_value={}),
            patch.object(manager.health, "read_expectations", return_value={}),
            patch.object(manager, "service_unit_snapshot",
                         side_effect=[({"ocr": "starting"}, {"ocr": 30}),
                                      ({"ocr": "active"}, {"ocr": 33})]),
        ):
            first = client.get("/api/status").get_json()
            second = client.get("/api/status").get_json()

        # Nothing can be concluded from one sample.
        self.assertEqual(first["health"]["ocr"]["state"], "starting")
        # By the second, the count has climbed and the phase no longer matters.
        self.assertEqual(second["health"]["ocr"]["state"], "failed")
        self.assertIn("3 times", second["health"]["ocr"]["reason"])
        manager.health.RESTARTS.reset()

    def test_starting_and_stopping_a_service_records_what_was_meant(self):
        recorded = []
        with patch.object(manager.health, "record_expectation",
                          side_effect=lambda *a, **k: recorded.append((a, k))):
            manager.record_service_expectation("ocr", "stop", True)
            manager.record_service_expectation("ocr", "start", True)
            manager.record_service_expectation("ocr", "restart", True)
            # A stop that failed leaves the service running, so it says nothing.
            manager.record_service_expectation("ocr", "stop", False)
        self.assertEqual([args[1] for args, _ in recorded], ["off", "on", "on"])

    def test_the_expect_route_is_not_swallowed_by_the_action_route(self):
        with (
            manager.app.test_client() as client,
            patch.object(manager, "patch_service_labels", return_value=[{"name": "ocr"}]),
            patch.object(manager.health, "record_expectation") as record,
        ):
            response = client.post("/api/service/ocr/expect", json={"expected": "off"})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["ok"])
        record.assert_called_once()

    def test_upstream_units_without_a_card_are_still_asked_about(self):
        with (
            patch.object(config_env, "read_env", return_value={}),
            patch.object(manager, "patch_service_labels", return_value=[{"name": "llm-a-proxy"}]),
            patch.object(manager, "get_service_status", return_value="inactive"),
        ):
            statuses = manager.all_service_statuses()
        # llm-a-proxy's upstream has no card of its own in this stubbed panel,
        # and still has to be asked about.
        self.assertIn("llm-a", statuses)
        self.assertIn("llm-a-proxy", statuses)


class SchedulingVerifyRouteTests(unittest.TestCase):
    """The pi-forge slot contract worked, and nothing could show that it did."""

    def test_verification_reads_the_stack_and_sends_nothing(self):
        env = {"LLM_A_CTX_SIZE": "262144", "LLM_A_N_PARALLEL": "2",
               "LLM_A_CACHE_IDLE_SLOTS": "on", "LLM_A_FIT": "off"}
        stats = {"scheduling": {"select_methods": {"id": 62},
                                "select_by_id_slots": {"0": 56, "1": 6}}}
        with (
            manager.app.test_client() as client,
            patch.object(config_env, "read_env", return_value=env),
            patch.object(manager, "get_service_status",
                         side_effect=lambda n: "active" if n == "llm-a" else "inactive"),
            patch.object(manager.telemetry, "probe_props",
                         return_value={"total_slots": 2, "n_ctx_per_slot": 131072}),
            patch.object(manager.telemetry, "probe_slots", return_value=[]),
            patch.object(manager.telemetry, "summarize", return_value=stats),
            patch.object(manager.telemetry.REGISTRY, "collector"),
            patch.object(core.ServiceManager, "get_pid", return_value=4242),
            patch.object(manager, "process_cmdline",
                         return_value="llama-server --ctx-size 262144 --parallel 2 "
                                      "--fit off --cache-idle-slots"),
            patch.object(manager.scheduling, "lease_directory", return_value=None),
        ):
            payload = client.get("/api/scheduling/verify").get_json()

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["unit"], "llm-a")
        self.assertTrue(payload["evidence"]["observed"])
        self.assertEqual(payload["runtime"]["n_ctx_per_slot"], 131072)

    def test_a_probe_is_refused_when_no_backend_is_running(self):
        with (
            manager.app.test_client() as client,
            patch.object(config_env, "read_env", return_value={}),
            patch.object(manager, "get_service_status", return_value="inactive"),
            patch.object(manager.scheduling, "lease_directory", return_value=None),
        ):
            response = client.post("/api/scheduling/verify", json={"probe": True})
        self.assertEqual(response.status_code, 409)

    def test_a_post_without_probe_sends_no_requests(self):
        with (
            manager.app.test_client() as client,
            patch.object(config_env, "read_env", return_value={}),
            patch.object(manager, "get_service_status", return_value="inactive"),
            patch.object(manager.scheduling, "lease_directory", return_value=None),
            patch.object(manager.scheduling, "probe_slot_pinning") as probe,
        ):
            response = client.post("/api/scheduling/verify", json={})
        self.assertEqual(response.status_code, 200)
        probe.assert_not_called()

    def test_lease_reaping_is_off_unless_it_is_turned_on(self):
        with (
            patch.object(config_env, "read_env", return_value={"PI_FORGE_LEASE_REAP": "off"}),
            patch.object(manager.scheduling, "reap_leases") as reap,
        ):
            manager.sweep_pi_forge_leases()
        reap.assert_not_called()

        with (
            patch.object(config_env, "read_env", return_value={"PI_FORGE_LEASE_REAP": "on"}),
            patch.object(manager.scheduling, "reap_leases") as reap,
        ):
            manager.sweep_pi_forge_leases()
        reap.assert_called_once()


class ConfigPreflightTests(unittest.TestCase):
    """The config form has always accepted anything, and the cost of that
    arrived later — on restart, or as an eviction storm.

    Pinned to Linux: these assert the discrete-GPU verdicts, where
    overcommitting VRAM fails the allocation. Unified memory answers
    differently and has its own tests.
    """

    GPUS = [{"index": 0, "mem_total": 24576, "mem_used": 1000},
            {"index": 1, "mem_total": 24576, "mem_used": 1000}]
    MEMINFO = {"MemTotal": 32787000, "MemAvailable": 20275000,
               "SwapTotal": 8388604, "SwapFree": 8388604}

    def setUp(self):
        ctx = platform_harness.as_linux()
        ctx.__enter__()
        self.addCleanup(ctx.__exit__, None, None, None)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        spec = importlib.util.spec_from_file_location(
            "llm_stack_manager_budget_tests",
            pathlib.Path(__file__).resolve().parent / "test_budget.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.model = module.write_gguf(
            pathlib.Path(self._tmp.name) / "model.gguf", module.QWEN36_27B)
        self.config_file = pathlib.Path(self._tmp.name) / "llm-stack.env"
        self.config_file.write_text("LLM_A_CTX_SIZE=131072\n")

    def _patches(self, env):
        return (
            patch.object(config_env, "read_env", return_value=env),
            patch.object(manager, "get_gpu_info", return_value=self.GPUS),
            patch.object(core, "read_meminfo", return_value=self.MEMINFO),
            patch.object(core, "CONFIG_FILE", self.config_file),
        )

    def _env(self, **overrides):
        return {"LLM_A_MODEL_PATH": str(self.model),
                "LLM_A_CTX_SIZE": "131072",
                "LLM_A_N_PARALLEL": "2", **overrides}

    def test_untouched_backends_are_not_priced(self):
        """Pricing means reading GGUF metadata off disk; a port change should
        not pay for it."""
        with self._patches(self._env())[0]:
            result = manager.preflight_config({"AGGREGATE_PORT": "8012"})
        self.assertEqual(result["backends"], [])
        self.assertTrue(result["ok"])

    def test_a_configuration_that_cannot_allocate_is_refused(self):
        env = self._env(LLM_A_GPU_VISIBLE_DEVICES="0")
        patches = self._patches(env)
        with manager.app.test_client() as client, patches[0], patches[1], patches[2], patches[3]:
            response = client.post("/api/config", json={"LLM_A_CTX_SIZE": "1048576"})
        self.assertEqual(response.status_code, 409)
        payload = response.get_json()
        self.assertFalse(payload["ok"])
        self.assertEqual([issue["code"] for issue in payload["preflight"]["errors"]],
                         ["vram_overcommit"])
        # Refused means not written.
        self.assertIn("LLM_A_CTX_SIZE=131072", self.config_file.read_text())

    def test_the_operator_can_override_the_prediction(self):
        env = self._env(LLM_A_GPU_VISIBLE_DEVICES="0")
        patches = self._patches(env)
        with manager.app.test_client() as client, patches[0], patches[1], patches[2], patches[3]:
            response = client.post("/api/config?force=1",
                                   json={"LLM_A_CTX_SIZE": "1048576"})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["ok"])
        self.assertTrue(response.get_json()["forced"])
        self.assertIn("LLM_A_CTX_SIZE=1048576", self.config_file.read_text())

    def test_a_workable_configuration_saves_and_reports_its_warnings(self):
        env = self._env(LLM_A_GPU_VISIBLE_DEVICES="0,1")
        patches = self._patches(env)
        with manager.app.test_client() as client, patches[0], patches[1], patches[2], patches[3]:
            response = client.post("/api/config", json={
                "LLM_A_CTX_SIZE": "131072",
                "LLM_A_CTX_CHECKPOINTS": "32",
                "LLM_A_CACHE_RAM": "1024",
            })
        payload = response.get_json()
        self.assertTrue(payload["ok"])
        self.assertFalse(payload["forced"])
        # 32 checkpoints of a hybrid model against 1 GiB of budget is the
        # eviction storm, stated before it happens rather than after.
        self.assertIn("cache_ram_shortfall",
                      {issue["code"] for issue in payload["preflight"]["warnings"]})

    def test_dead_and_contradictory_flags_are_reported_on_save(self):
        env = self._env(LLM_A_GPU_VISIBLE_DEVICES="0,1")
        patches = self._patches(env)
        with manager.app.test_client() as client, patches[0], patches[1], patches[2], patches[3]:
            response = client.post("/api/config", json={
                "LLM_A_SWA_FULL": "on",
                "LLM_A_FIT": "off",
                "LLM_A_FIT_CTX": "4096",
            })
        codes = {issue["code"] for issue in response.get_json()["preflight"]["warnings"]}
        self.assertIn("swa_full_unsupported", codes)
        self.assertIn("fit_ctx_without_fit", codes)

    def test_preflight_endpoint_prices_without_writing(self):
        env = self._env(LLM_A_GPU_VISIBLE_DEVICES="0,1")
        patches = self._patches(env)
        with manager.app.test_client() as client, patches[0], patches[1], patches[2], patches[3]:
            response = client.post("/api/config/preflight",
                                   json={"LLM_A_CTX_SIZE": "262144"})
        payload = response.get_json()
        self.assertEqual(payload["backends"][0]["per_slot_context"], 131072)
        self.assertIn("LLM_A_CTX_SIZE=131072", self.config_file.read_text())

    def test_an_unreadable_model_reports_rather_than_blocking_the_save(self):
        """Pointing at a model that has not been fetched yet is not a
        configuration that fails to fit."""
        patches = self._patches({"LLM_A_MODEL_PATH": "/nope.gguf"})
        with manager.app.test_client() as client, patches[0], patches[1], patches[2], patches[3]:
            response = client.post("/api/config", json={"LLM_A_CTX_SIZE": "262144"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("model not found", response.get_json()["preflight"]["backends"][0]["error"])


class EnvDeprecationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = pathlib.Path(self._tmp.name)
        self.config_file = self.root / "llm-stack.env"
        self.saved_dir = self.root / "saved"
        self.saved_dir.mkdir()

    def _patches(self):
        return (patch.object(core, "CONFIG_FILE", self.config_file),
                patch.object(core, "SAVED_CONFIGS_DIR", self.saved_dir))

    def test_legacy_keys_still_on_disk_are_named_with_their_replacement(self):
        self.config_file.write_text(
            "CHAT_DENSE_MODEL_PATH=/models/a.gguf\nCHAT_PRIMARY_CTX_SIZE=131072\n")
        with self._patches()[0], self._patches()[1]:
            report = manager.collect_env_deprecations()
        # `CHAT_PRIMARY_CTX_SIZE` is here because the slot was renamed again:
        # the spelling that replaced `CHAT_DENSE_*` is itself legacy now, and
        # the report names every generation of the name, not just the oldest.
        self.assertEqual([entry["key"] for entry in report["env_keys"]],
                         ["CHAT_DENSE_MODEL_PATH", "CHAT_PRIMARY_CTX_SIZE"])
        self.assertEqual(report["env_keys"][0]["replacement"], "LLM_A_MODEL_PATH")
        self.assertFalse(report["env_keys"][0]["canonical_present"])

    def test_a_clean_env_file_reports_nothing_to_migrate(self):
        self.config_file.write_text("LLM_A_MODEL_PATH=/models/a.gguf\n")
        with self._patches()[0], self._patches()[1]:
            self.assertEqual(manager.collect_env_deprecations()["migratable"], 0)

    def test_saved_profiles_are_reported_but_not_migrated(self):
        self.config_file.write_text("CHAT_DENSE_CTX_SIZE=131072\n")
        (self.saved_dir / "Qwen.json").write_text(json.dumps({
            "CHAT_DENSE_MODEL_PATH": "/models/a.gguf",
            "_config_form": {"CHAT_MOE_CTX_SIZE": "65536"},
        }))
        with manager.app.test_client() as client, self._patches()[0], self._patches()[1]:
            client.post("/api/config/deprecations/migrate")
            report = client.get("/api/config/deprecations").get_json()

        self.assertEqual(report["migratable"], 0, "the env file should be migrated")
        # The profile is untouched: it is user data, and the read-side backfill
        # keeps it working as written.
        self.assertEqual(report["saved_configs"],
                         [{"name": "Qwen", "keys": ["CHAT_DENSE_MODEL_PATH", "CHAT_MOE_CTX_SIZE"]}])
        self.assertIn("CHAT_DENSE_MODEL_PATH", (self.saved_dir / "Qwen.json").read_text())

    def test_migration_preserves_the_value_under_the_canonical_name(self):
        self.config_file.write_text("CHAT_DENSE_MODEL_PATH=/models/a.gguf\n")
        with manager.app.test_client() as client, self._patches()[0], self._patches()[1]:
            response = client.post("/api/config/deprecations/migrate")
            resolved = config_env.read_env()
        self.assertTrue(response.get_json()["ok"])
        written = self.config_file.read_text()
        self.assertIn("LLM_A_MODEL_PATH=/models/a.gguf", written)
        self.assertNotIn("CHAT_DENSE_MODEL_PATH", written)
        self.assertEqual(resolved["LLM_A_MODEL_PATH"], "/models/a.gguf")

    #: Suffixes both slots take but no `CONFIG_FIELDS` entry declares, so
    #: `LEGACY_ENV_KEY_MAP` -- which is generated from the fields -- never sees
    #: them. The launcher does: it resolves a slot's suffixes over its
    #: `legacy_prefixes`. Reporting from the map alone said 55 on a host with 61
    #: written down, and Migrate then left these six behind a banner reading zero.
    FIELDLESS_SUFFIXES = ("TEMP", "TOP_P", "TOP_K", "MIN_P",
                          "REASONING_FORMAT", "LOG_PREFIX")

    def test_a_legacy_key_no_field_declares_is_still_reported(self):
        self.config_file.write_text(
            "".join(f"CHAT2_{s}=x\n" for s in self.FIELDLESS_SUFFIXES))
        with self._patches()[0], self._patches()[1]:
            report = manager.collect_env_deprecations()
        self.assertEqual(report["migratable"], len(self.FIELDLESS_SUFFIXES))
        self.assertEqual(
            {entry["replacement"] for entry in report["env_keys"]},
            {f"LLM_B_{s}" for s in self.FIELDLESS_SUFFIXES})

    def test_migrating_a_key_no_field_declares_removes_the_old_line(self):
        """The regression that made this worth doing: writing the canonical key
        without collapsing its alias left both spellings in the file, and the
        report -- which only knew the map -- called it clean."""
        self.config_file.write_text("CHAT2_TEMP=1.0\n")
        with manager.app.test_client() as client, self._patches()[0], self._patches()[1]:
            response = client.post("/api/config/deprecations/migrate")
            after = manager.collect_env_deprecations()
        written = self.config_file.read_text()
        self.assertEqual(response.get_json()["skipped"], [])
        self.assertIn("LLM_B_TEMP=1.0", written)
        self.assertNotIn("CHAT2_TEMP", written)
        self.assertEqual(after["migratable"], 0)

    def test_the_frozen_port_key_is_never_migrated(self):
        self.config_file.write_text("CHAT2_BACKEND_PORT=8020\nCHAT2_TEMP=1.0\n")
        with manager.app.test_client() as client, self._patches()[0], self._patches()[1]:
            client.post("/api/config/deprecations/migrate")
        written = self.config_file.read_text()
        self.assertIn("CHAT2_BACKEND_PORT=8020", written)
        self.assertNotIn("LLM_B_BACKEND_PORT", written)

    def test_a_write_collapses_the_legacy_line_it_replaces(self):
        """`update_env_values` renames in place rather than appending beside.
        Both prefixes: `CHAT_*` is llm-a's second legacy spelling and is the one
        an unmigrated host actually holds."""
        self.config_file.write_text(
            "# tuning\nCHAT2_TEMP=1.0\nCHAT_TENSOR_SPLIT=1,1\nCHAT_BACKEND_PORT=8010\n")
        with self._patches()[0]:
            config_env.update_env_values(
                {"LLM_B_TEMP": "0.9", "LLM_A_TENSOR_SPLIT": "1,1.25"})
        written = self.config_file.read_text()
        self.assertNotIn("CHAT2_TEMP", written)
        self.assertNotIn("CHAT_TENSOR_SPLIT", written)
        self.assertIn("LLM_B_TEMP=0.9", written)
        self.assertIn("LLM_A_TENSOR_SPLIT=1,1.25", written)
        # The comment above the renamed line survives, and the port contract is
        # not swept up by the prefix it shares.
        self.assertIn("# tuning", written)
        self.assertIn("CHAT_BACKEND_PORT=8010", written)

    def test_read_env_raw_shows_the_file_without_backfill(self):
        self.config_file.write_text("CHAT_DENSE_MODEL_PATH=/models/a.gguf\n")
        with self._patches()[0]:
            self.assertEqual(config_env.read_env_raw(),
                             {"CHAT_DENSE_MODEL_PATH": "/models/a.gguf"})
            self.assertEqual(config_env.read_env()["LLM_A_MODEL_PATH"], "/models/a.gguf")


class RouteInventoryTests(unittest.TestCase):
    """Every URL the manager serves, pinned.

    The manager's HTTP surface is its whole contract: the UI polls it, the CLI
    reads `/api/backend/telemetry` through it, and pi-forge's scheduling
    verification hangs off `/api/scheduling/verify`. That surface is currently
    defined only by 83 `@app.route` decorators scattered across a 6,000-line
    module, so a route lost while moving code between files is invisible until
    a tab stops loading.

    Endpoints are compared on their *suffix* rather than their full name, so
    registering a route through a blueprint (`graphiti.api_graphiti_status`)
    still matches the same entry. That is deliberate: this test has to survive
    the move it exists to police, while still failing if a rule, a method or a
    view function changes.
    """

    EXPECTED = {
        ("/", ("GET",), "index"),
        ("/api/active-chat-model", ("GET",), "api_active_chat_model"),
        ("/api/app/update", ("POST",), "api_app_update"),
        ("/api/backend/budget", ("GET",), "api_backend_budget"),
        ("/api/backend/budget/recommend", ("GET",), "api_backend_budget_recommend"),
        ("/api/backend/telemetry", ("GET",), "api_backend_telemetry"),
        ("/api/chat-templates", ("GET",), "api_chat_templates_list"),
        ("/api/chat-templates", ("POST",), "api_chat_templates_create"),
        ("/api/chat-templates/<template_id>", ("DELETE",), "api_chat_templates_delete"),
        ("/api/chat-templates/<template_id>", ("GET",), "api_chat_templates_get"),
        ("/api/chat-templates/<template_id>", ("PUT",), "api_chat_templates_update"),
        ("/api/config", ("GET",), "api_config_get"),
        ("/api/config", ("POST",), "api_config_save"),
        ("/api/config/deprecations", ("GET",), "api_config_deprecations"),
        ("/api/config/deprecations/migrate", ("POST",), "api_config_deprecations_migrate"),
        ("/api/config/preflight", ("POST",), "api_config_preflight"),
        ("/api/custom-model-arg-presets/match", ("POST",), "api_custom_model_arg_preset_match"),
        ("/api/custom-models", ("GET",), "api_custom_models_list"),
        ("/api/custom-models", ("POST",), "api_custom_models_add"),
        ("/api/custom-models/<model_id>", ("DELETE",), "api_custom_models_delete"),
        ("/api/custom-models/<model_id>", ("PUT",), "api_custom_models_update"),
        ("/api/deploy/check", ("POST",), "api_deploy_check"),
        ("/api/deploy/status", ("GET",), "api_deploy_status"),
        ("/api/gguf-files", ("GET",), "api_gguf_files"),
        # The hub half of the fleet. Every proxied path is a fixed rule with a
        # fixed destination: a generic `/api/fleet/<id>/<path:rest>` on this
        # unauthenticated port would let anything that can reach 8077 issue
        # authenticated requests to every machine in the fleet.
        ("/api/fleet", ("GET",), "api_fleet"),
        # The write half. Each mirrors the local path it stands in for so the
        # page's own call sites can be rewritten rather than duplicated, which
        # is why `service` here is singular where the control API says
        # `services`.
        ("/api/fleet/<host_id>/config", ("GET",), "api_fleet_config"),
        ("/api/fleet/<host_id>/config", ("POST",), "api_fleet_config_save"),
        ("/api/fleet/<host_id>/config/fields", ("GET",), "api_fleet_config_fields"),
        ("/api/fleet/<host_id>/config/preflight", ("POST",), "api_fleet_config_preflight"),
        ("/api/fleet/<host_id>/logs", ("GET",), "api_fleet_logs"),
        ("/api/fleet/<host_id>/saved-configs", ("GET",), "api_fleet_saved_configs"),
        ("/api/fleet/<host_id>/saved-configs/<name>/apply", ("POST",),
         "api_fleet_saved_config_apply"),
        ("/api/fleet/<host_id>/service/<name>/<action>", ("POST",),
         "api_fleet_service_action"),
        ("/api/fleet/<host_id>/services", ("GET",), "api_fleet_services"),
        ("/api/fleet/<host_id>/snapshot", ("GET",), "api_fleet_snapshot"),
        ("/api/fleet/<host_id>/status", ("GET",), "api_fleet_status"),
        ("/api/fleet/hosts", ("GET",), "api_fleet_hosts_list"),
        ("/api/fleet/hosts", ("POST",), "api_fleet_hosts_add"),
        ("/api/fleet/hosts/<host_id>", ("DELETE",), "api_fleet_hosts_delete"),
        ("/api/fleet/hosts/<host_id>", ("PUT",), "api_fleet_hosts_update"),
        ("/api/fleet/hosts/<host_id>/test", ("POST",), "api_fleet_host_test"),
        ("/api/graphiti/detail/entity/<entity_uuid>", ("GET",), "api_graphiti_entity_detail"),
        ("/api/graphiti/detail/episode/<episode_uuid>", ("GET",), "api_graphiti_episode_detail"),
        ("/api/graphiti/detail/relationship/<relationship_uuid>", ("GET",), "api_graphiti_relationship_detail"),
        ("/api/graphiti/export", ("POST",), "api_graphiti_export"),
        ("/api/graphiti/exports", ("GET",), "api_graphiti_exports_list"),
        ("/api/graphiti/exports/<path:filename>", ("GET",), "api_graphiti_export_download"),
        ("/api/graphiti/neighborhood/<entity_uuid>", ("GET",), "api_graphiti_neighborhood"),
        ("/api/graphiti/recent/entities", ("GET",), "api_graphiti_recent_entities"),
        ("/api/graphiti/recent/episodes", ("GET",), "api_graphiti_recent_episodes"),
        ("/api/graphiti/recent/relationships", ("GET",), "api_graphiti_recent_relationships"),
        ("/api/graphiti/search/entities", ("GET",), "api_graphiti_search_entities"),
        ("/api/graphiti/search/group/<group_id>", ("GET",), "api_graphiti_group_history"),
        ("/api/graphiti/search/memory", ("POST",), "api_graphiti_search_memory"),
        ("/api/graphiti/stats", ("GET",), "api_graphiti_stats"),
        ("/api/graphiti/status", ("GET",), "api_graphiti_status"),
        ("/api/huggingface/downloads", ("GET",), "api_huggingface_downloads_list"),
        ("/api/huggingface/downloads", ("POST",), "api_huggingface_download_create"),
        ("/api/huggingface/downloads/<job_id>", ("GET",), "api_huggingface_download_status"),
        ("/api/huggingface/repo-files", ("POST",), "api_huggingface_repo_files"),
        ("/api/huggingface/transcription-downloads", ("POST",), "api_huggingface_transcription_download_create"),
        ("/api/huggingface/transcription-repo-files", ("POST",), "api_huggingface_transcription_repo_files"),
        ("/api/llamacpp/update", ("POST",), "api_llamacpp_update"),
        ("/api/logs/<name>", ("GET",), "api_logs"),
        ("/api/model-router", ("GET",), "api_model_router"),
        ("/api/model-router/<action>", ("POST",), "api_model_router_action"),
        ("/api/ocr/extract", ("POST",), "api_ocr_extract"),
        ("/api/ocr/parse", ("POST",), "api_ocr_parse"),
        ("/api/playwright/install", ("POST",), "api_playwright_install"),
        ("/api/playwright/status", ("GET",), "api_playwright_status"),
        ("/api/restore-active-stack", ("POST",), "api_default_mode"),
        ("/api/saved-configs", ("GET",), "api_saved_configs_list"),
        ("/api/saved-configs", ("POST",), "api_saved_configs_save"),
        ("/api/saved-configs/<name>", ("DELETE",), "api_saved_configs_delete"),
        ("/api/saved-configs/<name>", ("GET",), "api_saved_configs_load"),
        ("/api/saved-configs/<name>/apply", ("POST",), "api_saved_configs_apply"),
        ("/api/saved-configs/<name>/default", ("DELETE",), "api_saved_configs_clear_default"),
        ("/api/saved-configs/<name>/default", ("POST",), "api_saved_configs_set_default"),
        ("/api/saved-configs/<name>/patch", ("POST",), "api_saved_configs_patch"),
        ("/api/scheduling/leases/reap", ("POST",), "api_scheduling_leases_reap"),
        ("/api/scheduling/verify", ("GET",), "api_scheduling_verify"),
        ("/api/scheduling/verify", ("POST",), "api_scheduling_verify_probe"),
        ("/api/searxng/install", ("POST",), "api_searxng_install"),
        ("/api/searxng/status", ("GET",), "api_searxng_status"),
        ("/api/service/<name>/<action>", ("POST",), "api_service_action"),
        ("/api/service/<name>/expect", ("POST",), "api_service_expect"),
        ("/api/service/<name>/health", ("GET",), "api_service_health"),
        ("/api/setup/jobs/<job_id>", ("GET",), "api_setup_job"),
        ("/api/setup/jobs/<job_id>/retry", ("POST",), "api_setup_job_retry"),
        ("/api/setup/models/inspect", ("POST",), "api_setup_model_inspect"),
        ("/api/setup/placement", ("GET",), "api_setup_placement"),
        ("/api/setup/preflight", ("GET",), "api_setup_preflight"),
        ("/api/setup/repair", ("POST",), "api_setup_repair"),
        ("/api/setup/run", ("POST",), "api_setup_run"),
        ("/api/setup/selection", ("GET", "PUT"), "api_setup_selection"),
        ("/api/setup/uninstall", ("POST",), "api_setup_uninstall"),
        ("/api/setup/validation", ("GET",), "api_setup_validation"),
        ("/api/status", ("GET",), "api_status"),
        ("/api/switch/<variant>", ("POST",), "api_switch"),
        ("/api/transcribe/install", ("POST",), "api_transcribe_install"),
        ("/api/transcribe/install/<job_id>", ("GET",), "api_transcribe_install_status"),
        ("/api/transcribe/overview", ("GET",), "api_transcribe_overview"),
        ("/api/transcribe/test", ("POST",), "api_transcribe_test"),
        ("/api/transcribe/unload", ("POST",), "api_transcribe_unload"),
        ("/api/transcription-capabilities", ("GET",), "api_transcription_capabilities"),
        ("/api/transcription-models/<engine_id>", ("GET",), "api_transcription_models"),
        ("/api/tts/activate/<backend_id>", ("POST",), "api_tts_activate"),
        ("/api/tts/overview", ("GET",), "api_tts_overview"),
        ("/api/tts/test", ("POST",), "api_tts_test"),
        # The read-only state API. Also registered here so `/api/v1/*` works on
        # the manager's own port; the listener other machines reach is built by
        # `create_state_api_app()` and carries these and nothing else.
        ("/api/v1/alerts", ("GET",), "api_v1_alerts"),
        ("/api/v1/backends", ("GET",), "api_v1_backends"),
        ("/api/v1/events", ("GET",), "api_v1_events"),
        ("/api/v1/gpu", ("GET",), "api_v1_gpu"),
        ("/api/v1/health", ("GET",), "api_v1_health"),
        ("/api/v1/logs", ("GET",), "api_v1_logs"),
        ("/api/v1/logs/raw", ("GET",), "api_v1_logs_raw"),
        ("/api/v1/metrics", ("GET",), "api_v1_metrics"),
        ("/api/v1/schema", ("GET",), "api_v1_schema"),
        ("/api/v1/services", ("GET",), "api_v1_services"),
        ("/api/v1/snapshot", ("GET",), "api_v1_snapshot"),
        ("/static/<path:filename>", ("GET",), "static"),
    }

    @staticmethod
    def _inventory() -> set[tuple]:
        inventory = set()
        for rule in manager.app.url_map.iter_rules():
            methods = tuple(sorted(m for m in rule.methods if m not in {"HEAD", "OPTIONS"}))
            inventory.add((str(rule), methods, rule.endpoint.rsplit(".", 1)[-1]))
        return inventory

    def test_the_url_map_matches_the_pinned_inventory(self):
        actual = self._inventory()
        self.assertEqual(
            sorted(self.EXPECTED - actual), [],
            "routes disappeared from the url map",
        )
        self.assertEqual(
            sorted(actual - self.EXPECTED), [],
            "routes were added without being pinned here; add them to EXPECTED "
            "so the next move cannot drop them silently",
        )

    def test_every_route_is_reachable_by_its_endpoint(self):
        """A blueprint registered twice, or under a prefix, changes the rule
        without changing the decorator. Resolving each rule back through the map
        catches that."""
        adapter = manager.app.url_map.bind("localhost")
        for rule, methods, _endpoint in sorted(self.EXPECTED):
            if "<" in rule:
                continue
            for method in methods:
                self.assertTrue(adapter.test(rule, method=method),
                                f"{method} {rule} does not resolve")


class DeployRouteTests(unittest.TestCase):
    """The endpoints the header badge reads.

    `/api/deploy/status` is on the 5s poll path, so the one behaviour that has
    to hold is that it never reaches the network itself — the fetch belongs to
    the watcher's thread.
    """

    def test_status_serves_the_cache_without_fetching_on_the_request_thread(self):
        """The badge is on the 5s poll path. Filling the cache is the watcher
        thread's job; the request only ever reads it."""
        with patch.object(manager.DEPLOY_WATCHER, "start"), \
                patch.object(manager.DEPLOY_WATCHER, "check") as check, \
                patch.object(manager.DEPLOY_WATCHER, "snapshot",
                             return_value={"ok": True, "head_short": "abc1234", "behind": 0}), \
                manager.app.test_client() as client:
            payload = client.get("/api/deploy/status").get_json()
        check.assert_not_called()
        self.assertEqual(payload["head_short"], "abc1234")
        self.assertEqual(payload["summary"]["state"], "current")
        self.assertEqual(payload["remedy"], manager.deploy.REMEDY)

    def test_the_first_check_happens_without_waiting_for_the_interval(self):
        """Otherwise the badge would say nothing for the first 15 minutes after
        a restart, which is exactly when a fresh deploy is being looked at."""
        watcher = manager.deploy.DriftWatcher(core.STACK_DIR, interval=3600)
        done = threading.Event()
        with patch.object(watcher, "check", side_effect=lambda: done.set()):
            watcher.start(lambda: {"LLM_MANAGER_DEPLOY_CHECK_INTERVAL": "3600"})
            self.assertTrue(done.wait(timeout=5), "the watcher thread never ran a check")

    def test_check_forces_a_refresh(self):
        with patch.object(manager.DEPLOY_WATCHER, "check",
                          return_value={"ok": True, "behind": 2, "upstream": "origin/main"}) as check, \
                manager.app.test_client() as client:
            payload = client.post("/api/deploy/check").get_json()
        check.assert_called_once()
        self.assertEqual(payload["summary"]["state"], "behind")
        self.assertIn("2 commits behind", payload["summary"]["message"])

    def test_status_carries_the_deployment_report(self):
        """The badge is fed by the poll the UI already makes, not a new timer."""
        with patch.object(config_env, "read_env", return_value={}), \
                patch.object(manager, "service_health_snapshot", return_value=({}, {})), \
                patch.object(manager, "get_gpu_info", return_value=[]), \
                patch.object(manager, "backend_context_summary", return_value={}), \
                patch.object(manager.DEPLOY_WATCHER, "snapshot",
                             return_value={"ok": True, "behind": 0, "head_short": "abc1234"}), \
                manager.app.test_client() as client:
            payload = client.get("/api/status").get_json()
        self.assertEqual(payload["deployment"]["head_short"], "abc1234")
        self.assertEqual(payload["deployment"]["summary"]["state"], "current")

    def test_the_watcher_points_at_the_tree_the_manager_runs_from(self):
        self.assertEqual(pathlib.Path(manager.DEPLOY_WATCHER.stack_dir), core.STACK_DIR)


class ModuleBoundaryTests(unittest.TestCase):
    """How the split modules are allowed to reach each other.

    A blueprint that does `from models import parse_huggingface_repo_ref` binds
    that function once, at import time. Substituting `models.parse_huggingface_repo_ref`
    afterwards then changes nothing the route can see — it is still holding the
    original. That is not hypothetical: it is how this refactor first broke, and
    the failure is quiet, because the route keeps working and only the test's
    substitution goes missing.

    So the rule is: behaviour is reached through the module, data is bound. The
    tables in `config_fields` are read and never replaced, so binding them is
    both safe and more readable than `config_fields.CONFIG_FIELDS` everywhere.
    """

    # `fleet` and `control_api` are here because the fleet suite rests on
    # `patch.object(fleet, "fetch", ...)` -- a spoke in `tests/test_fleet.py` is
    # two Flask test clients spliced in at that one seam, and a single
    # `from fleet import fetch` anywhere would defeat every test in that file
    # without failing one. `web/fleet.py`'s own docstring says so; until now
    # nothing checked it.
    BEHAVIOUR_MODULES = {"core", "config_env", "models", "graphiti", "deploy",
                         "health", "telemetry", "scheduling", "budget",
                         "fleet", "control_api"}

    # `web/.venv` is the manager's own virtualenv, created in place by
    # start-llm-manager.sh, so on any machine that has run the manager these
    # rules would otherwise be applied to Flask's source instead of ours.
    SKIP_DIRS = {"__pycache__", ".venv", "venv", "site-packages", "node_modules"}

    @classmethod
    def _python_sources(cls):
        web = pathlib.Path(__file__).resolve().parents[1] / "web"
        return [p for p in sorted(web.rglob("*.py"))
                if not cls.SKIP_DIRS & set(p.relative_to(web).parts)]

    def test_behaviour_modules_are_never_bound_into_another_module(self):
        import ast
        offenders = []
        for path in self._python_sources():
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom) and node.module in self.BEHAVIOUR_MODULES:
                    names = ", ".join(a.name for a in node.names)
                    offenders.append(f"{path.name}:{node.lineno} from {node.module} import {names}")
        self.assertEqual(
            offenders, [],
            "these bind names that tests substitute; import the module and call "
            "`module.name()` instead",
        )

    def test_no_module_imports_the_application(self):
        """`app.py` imports the others, so any import back would be a cycle — and
        would load a second copy under a different name when tests load app.py
        by path."""
        import ast
        offenders = []
        for path in self._python_sources():
            if path.name == "app.py":
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import) and any(a.name == "app" for a in node.names):
                    offenders.append(f"{path.name}:{node.lineno}")
                if isinstance(node, ast.ImportFrom) and node.module == "app":
                    offenders.append(f"{path.name}:{node.lineno}")
        self.assertEqual(offenders, [], "modules must not import app.py")

    def test_blueprints_carry_no_url_prefix(self):
        """A prefix would silently rewrite every rule in the group."""
        for name, blueprint in manager.app.blueprints.items():
            with self.subTest(blueprint=name):
                self.assertIn(blueprint.url_prefix, (None, ""),
                              f"blueprint {name} would rewrite its rules")


class ConfigNormalizationTests(unittest.TestCase):
    """The read/write path every config surface funnels through.

    Four functions sit between what the operator types and what reaches
    `llm-stack.env`: `normalize_config_updates` rewrites legacy key names to
    canonical ones on the way in, `allowed_config_keys` decides what may be
    written at all, `filter_config_updates` applies both, and
    `normalize_env_keys` backfills canonical keys from legacy twins on the way
    out. Between them they are why a saved profile written under the old
    `CHAT_DENSE_*` names still loads.

    They had five assertions before this class existed, which is not enough to
    move them into another module with any confidence.
    """

    def test_every_legacy_key_normalizes_to_its_canonical_name(self):
        for legacy, canonical in manager.LEGACY_ENV_KEY_MAP.items():
            with self.subTest(legacy=legacy):
                self.assertEqual(
                    config_env.normalize_config_updates({legacy: "value"}),
                    {canonical: "value"},
                )

    def test_an_explicit_canonical_value_wins_over_its_legacy_alias(self):
        """Both names in one payload is the saved-profile case: profiles written
        before the rename carry both, and the canonical one is the current
        truth."""
        for canonical, aliases in manager.NEW_ENV_KEY_LEGACY_ALIASES.items():
            for legacy in aliases:
                with self.subTest(canonical=canonical, legacy=legacy):
                    normalized = config_env.normalize_config_updates(
                        {canonical: "canonical", legacy: "legacy"})
                    self.assertEqual(normalized[canonical], "canonical")

    def test_legacy_aliases_cover_every_mapped_key(self):
        self.assertEqual(
            sorted(manager.NEW_ENV_KEY_LEGACY_ALIASES),
            sorted(set(manager.LEGACY_ENV_KEY_MAP.values())),
        )
        for canonical, aliases in manager.NEW_ENV_KEY_LEGACY_ALIASES.items():
            for legacy in aliases:
                self.assertEqual(manager.LEGACY_ENV_KEY_MAP[legacy], canonical)

    def test_normalize_env_keys_backfills_every_legacy_key_it_maps(self):
        for legacy, canonical in manager.LEGACY_ENV_KEY_MAP.items():
            with self.subTest(legacy=legacy):
                env = config_env.normalize_env_keys({legacy: "/models/x.gguf"})
                # The two label keys are deliberately renamed away from their
                # old model-shaped defaults; every other key passes through.
                if legacy not in ("CHAT_DENSE_LABEL", "CHAT_MOE_LABEL"):
                    self.assertEqual(env[canonical], "/models/x.gguf")
                else:
                    self.assertTrue(env[canonical])

    def test_normalize_env_keys_backfills_a_rename_no_field_declares(self):
        """The map is generated from `CONFIG_FIELDS`, so a setting with no UI
        control is not in it -- but the launcher still reads it, through the
        slot's `legacy_prefixes`. Without this the migration reported the key
        and then skipped it, because the canonical name resolved to nothing."""
        for suffix in ("TEMP", "TOP_P", "TOP_K", "MIN_P",
                       "REASONING_FORMAT", "LOG_PREFIX"):
            with self.subTest(suffix=suffix):
                self.assertNotIn(f"CHAT2_{suffix}", manager.LEGACY_ENV_KEY_MAP)
                env = config_env.normalize_env_keys({f"CHAT2_{suffix}": "written"})
                self.assertEqual(env[f"LLM_B_{suffix}"], "written")

    def test_normalize_env_keys_leaves_the_frozen_port_alone(self):
        env = config_env.normalize_env_keys({"CHAT2_BACKEND_PORT": "8020"})
        self.assertNotIn("LLM_B_BACKEND_PORT", env)
        self.assertEqual(env["CHAT2_BACKEND_PORT"], "8020")

    def test_the_bare_chat_prefix_is_not_swept_into_slot_a(self):
        """`CHAT_BEE_*` names no slot at all. It shares a prefix with llm-a's
        second legacy spelling, so a prefix rule applied to the write side would
        invent `LLM_A_BEE_*` keys out of it."""
        env = config_env.normalize_env_keys({"CHAT_BEE_LABEL": "bee"})
        self.assertNotIn("LLM_A_BEE_LABEL", env)
        self.assertEqual(config_env.normalize_config_updates({"CHAT_BEE_LABEL": "bee"}),
                         {"CHAT_BEE_LABEL": "bee"})

    def test_an_old_profiles_undeclared_key_applies_under_its_new_name(self):
        """Saved profiles are never rewritten, so the only place their older
        spellings get honoured is on the way out. `CHAT2_TEMP` is in no field
        list, so a map-only rewrite dropped it at the allow-list instead."""
        self.assertEqual(config_env.normalize_config_updates({"CHAT2_TEMP": "0.5"}),
                         {"LLM_B_TEMP": "0.5"})

    def test_normalize_env_keys_does_not_overwrite_a_canonical_value(self):
        env = config_env.normalize_env_keys({
            "CHAT_DENSE_CTX_SIZE": "32768",
            "LLM_A_CTX_SIZE": "131072",
        })
        self.assertEqual(env["LLM_A_CTX_SIZE"], "131072")

    def test_every_code_mirror_reaches_its_slot_key(self):
        for code_key, slot_key in manager.CODE_TO_CHAT_MIRRORS.items():
            with self.subTest(code_key=code_key):
                mirrored = config_env.apply_code_chat_mirrors({code_key: "mirrored"})
                self.assertEqual(mirrored[slot_key], "mirrored")

    def test_an_explicit_slot_value_is_never_overwritten_by_its_code_mirror(self):
        """Regression this function's docstring describes: a full saved config
        carries both, and a legacy CODE_* default must not clobber the saved
        backend value."""
        for code_key, slot_key in manager.CODE_TO_CHAT_MIRRORS.items():
            with self.subTest(code_key=code_key, slot_key=slot_key):
                mirrored = config_env.apply_code_chat_mirrors(
                    {code_key: "from-code", slot_key: "explicit"})
                self.assertEqual(mirrored[slot_key], "explicit")

    def test_every_mirror_targets_the_slot_the_coding_endpoint_serves(self):
        """The coding endpoint is a proxy persona on llm-a, so a backend-level
        `CODE_*` setting belongs to slot A and to nothing else. The table used to
        name the bare `CHAT_*` spelling, and `CODE_CTX_SIZE` additionally named
        `CHAT_MOE_CTX_SIZE` -- which normalizes onto `LLM_B_CTX_SIZE`, so saving
        the Coding Endpoint section resized a backend it has never touched."""
        for code_key, slot_key in manager.CODE_TO_CHAT_MIRRORS.items():
            with self.subTest(code_key=code_key):
                self.assertTrue(slot_key.startswith("LLM_A_"), slot_key)

    def test_context_size_mirrors_onto_slot_a_alone(self):
        mirrored = config_env.apply_code_chat_mirrors({"CODE_CTX_SIZE": "131072"})
        self.assertEqual(mirrored["LLM_A_CTX_SIZE"], "131072")
        self.assertNotIn("LLM_B_CTX_SIZE", mirrored)
        self.assertNotIn("CHAT_CTX_SIZE", mirrored)

    def test_unknown_keys_are_dropped(self):
        filtered = config_env.filter_config_updates(
            {"LLM_A_CTX_SIZE": "131072", "RM_RF_SLASH": "1"}, env={})
        self.assertEqual(filtered, {"LLM_A_CTX_SIZE": "131072"})

    def test_keys_already_in_the_env_file_survive_even_without_a_ui_field(self):
        """Saved configs predate some UI controls; dropping their values on
        apply would silently reconfigure the stack."""
        filtered = config_env.filter_config_updates(
            {"SOME_UNLISTED_KEY": "kept"}, env={"SOME_UNLISTED_KEY": "old"})
        self.assertEqual(filtered, {"SOME_UNLISTED_KEY": "kept"})
        self.assertEqual(config_env.filter_config_updates({"SOME_UNLISTED_KEY": "kept"}, env={}), {})

    def test_allowed_keys_span_every_source_it_claims(self):
        allowed = config_env.allowed_config_keys(env={"ADHOC_KEY": ""})
        self.assertIn("LLM_A_CTX_SIZE", allowed)          # CONFIG_FIELDS
        self.assertIn("CHAT_DENSE_CTX_SIZE", allowed)            # legacy names
        self.assertIn("ADHOC_KEY", allowed)                      # current env
        self.assertTrue(set(manager.RESTART_HINTS) <= allowed)

    def test_values_are_coerced_to_strings_and_none_becomes_empty(self):
        filtered = config_env.filter_config_updates({
            "LLM_A_CTX_SIZE": 131072,
            "LLM_A_FLASH_ATTN": True,
            "LLM_A_FIT_TARGET": None,
        }, env={})
        self.assertEqual(filtered["LLM_A_CTX_SIZE"], "131072")
        self.assertEqual(filtered["LLM_A_FLASH_ATTN"], "True")
        self.assertEqual(filtered["LLM_A_FIT_TARGET"], "")

    def test_structured_values_are_rejected_rather_than_stringified(self):
        filtered = config_env.filter_config_updates(
            {"LLM_A_CTX_SIZE": {"nested": 1}, "LLM_A_MODEL_PATH": ["a"]}, env={})
        self.assertEqual(filtered, {})

    def test_a_non_dict_payload_is_not_an_error(self):
        for payload in (None, [], "LLM_A_CTX_SIZE=1", 7):
            self.assertEqual(config_env.filter_config_updates(payload, env={}), {})

    def test_config_form_snapshot_round_trips_a_saved_profile(self):
        values = {
            "CHAT_DENSE_MODEL_PATH": "/models/a.gguf",
            "LLM_A_CTX_SIZE": "131072",
            "LLM_A_N_PARALLEL": "2",
        }
        snapshot = config_env.config_form_snapshot(values, env={})
        self.assertEqual(snapshot["LLM_A_MODEL_PATH"], "/models/a.gguf")
        self.assertEqual(snapshot["LLM_A_CTX_SIZE"], "131072")
        self.assertEqual(config_env.config_form_snapshot(snapshot, env={}), snapshot)


class SplitModeTests(unittest.TestCase):
    """Two of llama.cpp's four split modes cannot run on this build, and both
    fail after the launcher has exec'd — as a core dump in a restart loop, not
    as an error. See docs/gpu-split-modes.md."""

    ROOT = pathlib.Path(__file__).resolve().parents[1]

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        spec = importlib.util.spec_from_file_location(
            "llm_stack_manager_split_budget",
            pathlib.Path(__file__).resolve().parent / "test_budget.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        tmp = pathlib.Path(self._tmp.name)
        # qwen35 is hybrid and is the family the primary backend runs; gemma4
        # is not, and is the counter-example that must still be allowed.
        self.hybrid = module.write_gguf(tmp / "hybrid.gguf", module.QWEN36_27B)
        self.dense = module.write_gguf(tmp / "dense.gguf", module.GEMMA4_31B)

    # -- the config surface -------------------------------------------------

    def test_row_is_not_offered_anywhere(self):
        self.assertNotIn("row", config_fields.LLAMA_SPLIT_MODE_OPTIONS)
        for field in config_fields.CONFIG_FIELDS:
            if field["key"].endswith("_SPLIT_MODE"):
                self.assertEqual(field["options"], config_fields.LLAMA_SPLIT_MODE_OPTIONS,
                                 f"{field['key']} offers its own split-mode list")
                self.assertTrue(field.get("hint"), f"{field['key']} has no hint")

    def test_every_backend_offers_the_same_modes(self):
        """Embedding and reranking used to offer a different list than the chat
        backends, for no reason anyone recorded."""
        keys = {f["key"] for f in config_fields.CONFIG_FIELDS if f["key"].endswith("_SPLIT_MODE")}
        self.assertIn("LLM_A_SPLIT_MODE", keys)
        self.assertIn("EMBED_SPLIT_MODE", keys)
        self.assertIn("RERANK_SPLIT_MODE", keys)

    def test_an_unsupported_mode_is_not_persisted(self):
        """allowed_config_keys unions in whatever the env already holds, so a
        saved profile written before row was withdrawn can still offer it."""
        out = config_env.filter_config_updates({
            "LLM_A_SPLIT_MODE": "row",
            "LLM_B_SPLIT_MODE": "sideways",
            "TASK_SPLIT_MODE": "tensor",
            "EMBED_SPLIT_MODE": "layer",
            "LLM_A_TENSOR_SPLIT": "1,1",
        })
        self.assertNotIn("LLM_A_SPLIT_MODE", out)
        self.assertNotIn("LLM_B_SPLIT_MODE", out)
        self.assertEqual(out["TASK_SPLIT_MODE"], "tensor")
        self.assertEqual(out["EMBED_SPLIT_MODE"], "layer")
        self.assertEqual(out["LLM_A_TENSOR_SPLIT"], "1,1")

    # -- the launcher helper ------------------------------------------------

    def _resolve(self, mode, model, tensor_split="1,1", main_gpu="0", flash_attn="on",
                 platform="Linux"):
        """Run resolve_split_opts the way a start script does and report both
        the flags it built and what it said about them.

        `platform` is pinned rather than inherited from the host: these assert
        the CUDA placement rules, which do not apply on Metal, and they have to
        keep running on a macOS runner. `LLM_STACK_PLATFORM` is the shell's
        equivalent of `platforms.set_active`.
        """
        script = (
            'set -euo pipefail\n'
            f'STACK_DIR={shlex.quote(str(self.ROOT))}\n'
            'source "${STACK_DIR}/scripts/lib/backend-preflight.sh"\n'
            f'resolve_split_opts "[test]" {shlex.quote(mode)} {shlex.quote(str(model))} '
            f'{shlex.quote(tensor_split)} {shlex.quote(main_gpu)} {shlex.quote(flash_attn)}\n'
            'printf "FLAGS:%s\\n" "${SPLIT_OPTS[*]}"\n'
        )
        proc = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                              env={**os.environ, "LLM_STACK_PLATFORM": platform})
        self.assertEqual(proc.returncode, 0, proc.stderr)
        flags, messages = "", []
        for line in proc.stdout.splitlines():
            if line.startswith("FLAGS:"):
                flags = line[len("FLAGS:"):]
            else:
                messages.append(line)
        return flags.split(), "\n".join(messages)

    def test_metal_collapses_every_mode_to_none(self):
        """One device and one memory pool: every mode that exists to divide a
        model between cards is inapplicable, not merely unsupported."""
        for mode in ("layer", "tensor", "row", "bogus"):
            with self.subTest(mode=mode):
                flags, said = self._resolve(mode, self.dense, platform="Darwin")
                self.assertEqual(flags, ["--split-mode", "none"])
                self.assertIn("Metal", said)

    def test_metal_drops_placement_flags_it_cannot_act_on(self):
        # --tensor-split and --main-gpu ask llama-server to choose between GPUs
        # there is only one of.
        flags, said = self._resolve("none", self.dense, tensor_split="1,1",
                                    main_gpu="1", platform="Darwin")
        self.assertEqual(flags, ["--split-mode", "none"])
        self.assertIn("Tensor Split", said)
        self.assertIn("Main GPU", said)

    def test_row_never_reaches_llama_server(self):
        flags, said = self._resolve("row", self.dense)
        self.assertEqual(flags[:2], ["--split-mode", "layer"])
        self.assertIn("split-buffer", said)

    def test_tensor_is_refused_for_a_hybrid_model(self):
        """The reported crash: qwen35 passes llama.cpp's own arch gate and then
        aborts in the meta backend during the warmup decode."""
        flags, said = self._resolve("tensor", self.hybrid)
        self.assertEqual(flags[:2], ["--split-mode", "layer"])
        self.assertIn("hybrid", said)
        self.assertIn("qwen35", said)

    def test_tensor_is_allowed_for_a_non_hybrid_model(self):
        flags, said = self._resolve("tensor", self.dense)
        self.assertEqual(flags, ["--split-mode", "tensor"])
        self.assertNotIn("--tensor-split", flags)
        self.assertNotIn("--main-gpu", flags)
        self.assertIn("experimental", said)

    def test_tensor_needs_flash_attention(self):
        flags, said = self._resolve("tensor", self.dense, flash_attn="off")
        self.assertEqual(flags[:2], ["--split-mode", "layer"])
        self.assertIn("flash attention", said)

    def test_an_unreadable_model_falls_back_rather_than_guessing(self):
        flags, said = self._resolve("tensor", self.ROOT / "no-such-model.gguf")
        self.assertEqual(flags[:2], ["--split-mode", "layer"])
        self.assertIn("could not read", said)

    def test_an_empty_ratio_is_omitted_not_passed_empty(self):
        """llama.cpp reads `--tensor-split ""` as an explicit empty split and
        refuses it; TASK_TENSOR_SPLIT and LLM_B_TENSOR_SPLIT are both empty."""
        flags, _ = self._resolve("layer", self.dense, tensor_split="")
        self.assertEqual(flags, ["--split-mode", "layer", "--main-gpu", "0"])

    def test_layer_still_passes_placement_through(self):
        flags, said = self._resolve("layer", self.hybrid)
        self.assertEqual(flags, ["--split-mode", "layer", "--main-gpu", "0",
                                 "--tensor-split", "1,1"])
        self.assertEqual(said, "")

    def test_backends_without_a_main_gpu_do_not_gain_one(self):
        """embed and rerank have never emitted --main-gpu."""
        flags, _ = self._resolve("layer", self.dense, main_gpu="")
        self.assertNotIn("--main-gpu", flags)

    # -- the router preset --------------------------------------------------

    def test_router_preset_applies_the_same_refusals(self):
        """Pooled models never touch the start scripts, so the check has to
        exist twice or a router member would crash where a unit would not."""
        spec = importlib.util.spec_from_file_location(
            "render_models_ini", self.ROOT / "scripts" / "render-models-ini.py")
        rmi = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rmi)

        said = []
        self.assertEqual(rmi._resolve_split_mode(str(self.hybrid), "tensor", said.append), "layer")
        self.assertEqual(rmi._resolve_split_mode(str(self.dense), "row", said.append), "layer")
        self.assertEqual(rmi._resolve_split_mode(str(self.dense), "tensor", said.append), "tensor")
        self.assertEqual(rmi._resolve_split_mode(str(self.dense), "layer", said.append), "layer")
        self.assertEqual(len(said), 2)

    def test_router_preset_drops_placement_under_tensor(self):
        spec = importlib.util.spec_from_file_location(
            "render_models_ini", self.ROOT / "scripts" / "render-models-ini.py")
        rmi = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rmi)

        _, options = rmi.render_member("EMBED", {
            "EMBEDDING_MODEL_PATH": str(self.dense),
            "EMBED_SPLIT_MODE": "tensor",
            "EMBED_TENSOR_SPLIT": "1,1",
            "EMBED_MAIN_GPU": "0",
        })
        self.assertEqual(options["split-mode"], "tensor")
        self.assertNotIn("tensor-split", options)
        self.assertNotIn("main-gpu", options)


if __name__ == "__main__":
    unittest.main()
