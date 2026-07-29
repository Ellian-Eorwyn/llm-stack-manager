from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import tempfile
import unittest
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


class ConfigSectionTests(unittest.TestCase):
    def test_primary_and_secondary_backend_fields_are_separate(self):
        sections = {f["key"]: f["section"] for f in manager.CONFIG_FIELDS}
        self.assertEqual(sections["CHAT_PRIMARY_MODEL_PATH"], "Primary Backend")
        self.assertEqual(sections["CHAT_PRIMARY_SPEC_METHOD"], "Primary Backend")
        self.assertEqual(sections["CHAT_PRIMARY_CUSTOM_ARGS_JSON"], "Primary Backend")
        self.assertEqual(sections["CHAT2_LABEL"], "Secondary Backend")
        self.assertEqual(sections["CHAT2_MODEL_PATH"], "Secondary Backend")
        self.assertEqual(sections["CHAT2_SPEC_METHOD"], "Secondary Backend")
        self.assertEqual(sections["CHAT2_CUSTOM_ARGS_JSON"], "Secondary Backend")
        self.assertNotIn("CHAT_SECONDARY_MODEL_PATH", sections)

    def test_primary_and_secondary_backend_restart_independently(self):
        self.assertEqual(manager.RESTART_HINTS["CHAT_PRIMARY_MODEL_PATH"], ["chat-backend-dense"])
        self.assertEqual(manager.RESTART_HINTS["CHAT_PRIMARY_BATCH_SIZE"], ["chat-backend-dense"])
        self.assertEqual(manager.RESTART_HINTS["CHAT2_MODEL_PATH"], ["chat-backend2"])
        self.assertEqual(manager.RESTART_HINTS["CHAT2_BATCH_SIZE"], ["chat-backend2"])

    def test_cache_aware_scheduling_cards_render_for_both_backends(self):
        template_dir = pathlib.Path(__file__).resolve().parents[1] / "web" / "templates"
        original_searchpath = list(manager.app.jinja_loader.searchpath)
        manager.app.jinja_loader.searchpath = [str(template_dir)]
        with (
            manager.app.test_client() as client,
            patch.object(manager, "read_env", return_value={}),
            patch.object(manager, "load_custom_models", return_value=[]),
        ):
            response = client.get("/")
        manager.app.jinja_loader.searchpath = original_searchpath

        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn('data-cache-aware-prefix="CHAT_PRIMARY"', html)
        self.assertIn('data-cache-aware-prefix="CHAT2"', html)
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
            self.assertEqual(manager.RESTART_HINTS[f"CHAT_PRIMARY_{suffix}"], ["chat-backend-dense"])
            self.assertEqual(manager.RESTART_HINTS[f"CHAT2_{suffix}"], ["chat-backend2"])

    def test_primary_and_secondary_backend_normalize_from_legacy_keys(self):
        env = manager.normalize_env_keys({
            "CHAT_DENSE_LABEL": "Backend Dense",
            "CHAT_DENSE_MODEL_PATH": "/models/primary.gguf",
            "CHAT_DENSE_CTX_SIZE": "32768",
            "CHAT_MOE_LABEL": "Backend MoE",
            "CHAT_MOE_MODEL_PATH": "/models/secondary.gguf",
            "CHAT_MOE_CTX_SIZE": "65536",
            "CHAT_BATCH_SIZE": "2048",
            "CHAT_GPU_VISIBLE_DEVICES": "0,1",
        })
        self.assertEqual(env["CHAT_PRIMARY_LABEL"], "Primary Backend")
        self.assertEqual(env["CHAT_PRIMARY_MODEL_PATH"], "/models/primary.gguf")
        self.assertEqual(env["CHAT_PRIMARY_CTX_SIZE"], "32768")
        self.assertEqual(env["CHAT_PRIMARY_BATCH_SIZE"], "2048")
        self.assertEqual(env["CHAT_PRIMARY_GPU_VISIBLE_DEVICES"], "0,1")
        self.assertEqual(env["CHAT_SECONDARY_LABEL"], "Secondary Backend")
        self.assertEqual(env["CHAT_SECONDARY_MODEL_PATH"], "/models/secondary.gguf")
        self.assertEqual(env["CHAT_SECONDARY_CTX_SIZE"], "65536")
        self.assertEqual(env["CHAT2_LABEL"], "Secondary Backend")
        self.assertNotIn("CHAT_SECONDARY_BATCH_SIZE", {f["key"] for f in manager.CONFIG_FIELDS})

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
        env = manager.normalize_env_keys({"CHAT_GPU_VISIBLE_DEVICES": "0,1"})
        self.assertEqual(env["OCR_GPU_VISIBLE_DEVICES"], "0,1")

    def test_chat_template_fields_are_exposed(self):
        fields = {f["key"]: f for f in manager.CONFIG_FIELDS}
        self.assertEqual(fields["CHAT_TEMPLATE_MANAGER"]["type"], "template_manager")
        self.assertEqual(fields["CHAT_PRIMARY_TEMPLATE_ID"]["type"], "chat_template")
        self.assertEqual(fields["CHAT2_TEMPLATE_ID"]["type"], "chat_template")
        self.assertEqual(fields["TASK_CHAT_TEMPLATE_ID"]["type"], "chat_template")

    def test_glmocr_sdk_fields_restart_sdk_only(self):
        self.assertIn("GLM-OCR SDK", manager.CORE_CONFIG_SECTIONS)
        self.assertEqual(manager.RESTART_HINTS["GLMOCR_SDK_PORT"], ["glmocr-sdk"])
        self.assertEqual(manager.RESTART_HINTS["GLMOCR_LAYOUT_DEVICE"], ["glmocr-sdk"])

    def test_glmocr_sdk_layout_gpu_is_never_comma_separated(self):
        env = manager.normalize_env_keys({
            "OCR_GPU_VISIBLE_DEVICES": "0,1",
            "GLMOCR_LAYOUT_CUDA_VISIBLE_DEVICES": "0,1",
            "GLMOCR_LAYOUT_DEVICE": "cuda:0,1",
        })
        self.assertEqual(env["OCR_GPU_VISIBLE_DEVICES"], "0,1")
        self.assertEqual(env["GLMOCR_LAYOUT_CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(env["GLMOCR_LAYOUT_DEVICE"], "cuda:0")

    def test_glmocr_sdk_layout_gpu_default_does_not_inherit_ocr_multi_gpu(self):
        env = manager.normalize_env_keys({"OCR_GPU_VISIBLE_DEVICES": "0,1"})
        self.assertEqual(env["OCR_GPU_VISIBLE_DEVICES"], "0,1")
        self.assertEqual(env["GLMOCR_LAYOUT_CUDA_VISIBLE_DEVICES"], "")

    def test_gguf_mmproj_classifier_keeps_model_and_projector_separate(self):
        self.assertFalse(manager.is_mmproj_gguf("Qwen3.5-27B-Q4_K_M.gguf", 16 * 1024**3))
        self.assertTrue(manager.is_mmproj_gguf("mmproj-Qwen3.5-27B-f16.gguf", 800 * 1024**2))
        self.assertTrue(manager.is_mmproj_gguf("Qwen3.5-27B.projector.gguf", 800 * 1024**2))
        self.assertTrue(manager.is_mmproj_gguf("vision-clip.gguf", 800 * 1024**2))

    def test_update_env_values_does_not_write_unrelated_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_file = pathlib.Path(tmp) / "llm-stack.env"
            config_file.write_text("CHAT_TEMP=1.0\n")
            with patch.object(manager, "CONFIG_FILE", config_file):
                manager.update_env_values({"CHAT_TEMP": "0.6"})

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
            patch.object(manager, "read_env", return_value=env),
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
            patch.object(manager, "read_env", return_value=env),
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
            patch.object(manager, "read_env", return_value={"GLMOCR_SDK_HOST": "0.0.0.0", "GLMOCR_SDK_PORT": "5002"}),
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
                patch.object(manager, "CHAT_TEMPLATES_DIR", template_dir),
                patch.object(manager, "CHAT_TEMPLATES_META_FILE", template_dir / "templates.json"),
            ):
                templates = manager.list_chat_templates()

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
            patch.object(manager, "parse_huggingface_repo_ref", return_value=repo_ref),
            patch.object(manager, "list_huggingface_repo_files", return_value=files),
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
                patch.object(manager, "CUSTOM_MODELS_FILE", custom_models_file),
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
                "CHAT_PRIMARY_CTX_SIZE": "32768",
                "CHAT_PRIMARY_SPLIT_MODE": "layer",
                "CHAT_PRIMARY_TENSOR_SPLIT": "1,1",
                "CHAT_PRIMARY_GPU_VISIBLE_DEVICES": "0,1",
            }
            form = {
                "CHAT_PRIMARY_CTX_SIZE": "128000",
                "CHAT_PRIMARY_SPLIT_MODE": "none",
                "CHAT_PRIMARY_TENSOR_SPLIT": "",
                "CHAT_PRIMARY_GPU_VISIBLE_DEVICES": "0",
            }

            with (
                manager.app.test_client() as client,
                patch.object(manager, "SAVED_CONFIGS_DIR", saved_dir),
                patch.object(manager, "read_env", return_value=env),
                patch.object(manager, "get_service_status", return_value="inactive"),
            ):
                resp = client.post("/api/saved-configs", json={"name": "OneGpu", "config": form})

            self.assertEqual(resp.status_code, 200)
            data = json.loads((saved_dir / "OneGpu.json").read_text())
            self.assertEqual(data["_config_form"]["CHAT_PRIMARY_CTX_SIZE"], "128000")
            self.assertEqual(data["_config_form"]["CHAT_PRIMARY_SPLIT_MODE"], "none")
            self.assertEqual(data["_config_form"]["CHAT_PRIMARY_TENSOR_SPLIT"], "")
            self.assertEqual(data["_config_form"]["CHAT_PRIMARY_GPU_VISIBLE_DEVICES"], "0")

    def test_apply_saved_config_prefers_form_snapshot_over_top_level_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved_dir = pathlib.Path(tmp)
            config_file = saved_dir / "llm-stack.env"
            config_file.write_text(
                "CHAT_PRIMARY_CTX_SIZE=32768\n"
                "CHAT_PRIMARY_SPLIT_MODE=layer\n"
                "CHAT_PRIMARY_TENSOR_SPLIT=1,1\n"
                "CHAT_PRIMARY_GPU_VISIBLE_DEVICES=0,1\n"
            )
            (saved_dir / "OneGpu.json").write_text(json.dumps({
                "CHAT_PRIMARY_CTX_SIZE": "262144",
                "CHAT_CTX_SIZE": "262144",
                "CHAT_PRIMARY_SPLIT_MODE": "layer",
                "CHAT_PRIMARY_TENSOR_SPLIT": "1,1",
                "CHAT_PRIMARY_GPU_VISIBLE_DEVICES": "0,1",
                "_config_form": {
                    "CHAT_PRIMARY_CTX_SIZE": "128000",
                    "CHAT_PRIMARY_SPLIT_MODE": "none",
                    "CHAT_PRIMARY_TENSOR_SPLIT": "",
                    "CHAT_PRIMARY_GPU_VISIBLE_DEVICES": "0",
                },
                "_active_chat_model": {"variant": None, "service": None, "label": "", "kind": "none"},
            }))

            with (
                patch.object(manager, "SAVED_CONFIGS_DIR", saved_dir),
                patch.object(manager, "CONFIG_FILE", config_file),
            ):
                result = manager.apply_saved_config("OneGpu", launch=False)
            content = config_file.read_text()

            self.assertTrue(result["ok"])
            self.assertIn("CHAT_PRIMARY_CTX_SIZE=128000", content)
            self.assertIn('CHAT_PRIMARY_TENSOR_SPLIT=""', content)
            self.assertIn("CHAT_PRIMARY_SPLIT_MODE=none", content)
            self.assertIn("CHAT_PRIMARY_GPU_VISIBLE_DEVICES=0", content)

    def test_apply_saved_config_prefers_canonical_pane_key_over_legacy_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved_dir = pathlib.Path(tmp)
            config_file = saved_dir / "llm-stack.env"
            config_file.write_text("CHAT_DENSE_CTX_SIZE=32768\nCHAT_CTX_SIZE=32768\n")
            (saved_dir / "Context.json").write_text(json.dumps({
                "CHAT_PRIMARY_CTX_SIZE": "128000",
                "CHAT_DENSE_CTX_SIZE": "32768",
                "CHAT_CTX_SIZE": "262144",
                "_active_chat_model": {"variant": None, "service": None, "label": "", "kind": "none"},
            }))

            with (
                patch.object(manager, "SAVED_CONFIGS_DIR", saved_dir),
                patch.object(manager, "CONFIG_FILE", config_file),
            ):
                result = manager.apply_saved_config("Context", launch=False)
            content = config_file.read_text()

            self.assertTrue(result["ok"])
            self.assertIn("CHAT_PRIMARY_CTX_SIZE=128000", content)
            self.assertNotIn("CHAT_DENSE_CTX_SIZE=32768", content)

    def test_apply_saved_config_accepts_numeric_json_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved_dir = pathlib.Path(tmp)
            config_file = saved_dir / "llm-stack.env"
            config_file.write_text("CHAT_PRIMARY_CTX_SIZE=32768\n")
            (saved_dir / "Numeric.json").write_text(json.dumps({
                "CHAT_PRIMARY_CTX_SIZE": 128000,
                "_active_chat_model": {"variant": None, "service": None, "label": "", "kind": "none"},
            }))

            with (
                patch.object(manager, "SAVED_CONFIGS_DIR", saved_dir),
                patch.object(manager, "CONFIG_FILE", config_file),
            ):
                result = manager.apply_saved_config("Numeric", launch=False)

            self.assertTrue(result["ok"])
            self.assertIn("CHAT_PRIMARY_CTX_SIZE=128000", config_file.read_text())

    def test_save_records_primary_and_secondary_backend_slots(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved_dir = pathlib.Path(tmp)
            env = {
                "CHAT_PRIMARY_LABEL": "Primary Backend",
                "CHAT2_LABEL": "Local Secondary",
                "CHAT2_MODEL_NAME": "chat-secondary",
                "CHAT2_MODEL_PATH": "/models/secondary.gguf",
            }

            def fake_status(name):
                return "active" if name in {"chat-backend-dense", "chat-backend2"} else "inactive"

            with (
                manager.app.test_client() as client,
                patch.object(manager, "SAVED_CONFIGS_DIR", saved_dir),
                patch.object(manager, "read_env", return_value=env),
                patch.object(manager, "get_service_status", side_effect=fake_status),
            ):
                resp = client.post("/api/saved-configs", json={"name": "Both"})
                listed = client.get("/api/saved-configs")

        self.assertEqual(resp.status_code, 200)
        body = listed.get_json()[0]
        self.assertEqual(body["active_backend_slots"]["primary"]["label"], "Primary Backend")
        self.assertEqual(body["active_backend_slots"]["secondary"]["label"], "Local Secondary")
        self.assertEqual(body["active_backend_slots"]["secondary"]["service"], "chat-backend2")

    def test_apply_saved_config_launches_secondary_from_slot_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved_dir = pathlib.Path(tmp)
            config_file = saved_dir / "llm-stack.env"
            config_file.write_text("CHAT2_LABEL=Old\n")
            (saved_dir / "Secondary.json").write_text(json.dumps({
                "CHAT2_LABEL": "Local Secondary",
                "_active_chat_model": {"variant": None, "service": None, "label": "", "kind": "none"},
                "_active_backend_slots": {
                    "secondary": {
                        "variant": "secondary",
                        "service": "chat-backend2",
                        "label": "Local Secondary",
                        "kind": "secondary",
                    }
                },
            }))
            started = []

            def fake_status(_name):
                return "inactive"

            class FakeServiceManager(manager.ServiceManager):
                @classmethod
                def start(cls, name, timeout=30):
                    started.append(name)
                    return manager.subprocess.CompletedProcess(["start", name], 0, "", "")

            with (
                patch.object(manager, "SAVED_CONFIGS_DIR", saved_dir),
                patch.object(manager, "CONFIG_FILE", config_file),
                patch.object(manager, "get_service_status", side_effect=fake_status),
                patch.object(manager, "ServiceManager", FakeServiceManager),
            ):
                result = manager.apply_saved_config("Secondary", launch=True)
            content = config_file.read_text()

            self.assertTrue(result["ok"])
            self.assertIn("chat-backend2", started)
            self.assertIn("chat-proxy2", started)
            self.assertIn('CHAT2_LABEL="Local Secondary"', content)

    def test_patch_saved_config_updates_only_supplied_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved_dir = pathlib.Path(tmp)
            saved_path = saved_dir / "Default.json"
            saved_path.write_text(json.dumps({
                "CHAT_TEMP": "1.0",
                "CHAT_TOP_K": "20",
                "_name": "Default",
            }))
            with patch.object(manager, "SAVED_CONFIGS_DIR", saved_dir):
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
            ("CHAT_PRIMARY_METRICS", "Primary Backend"),
            ("CHAT2_METRICS", "Secondary Backend"),
            ("EMBED_METRICS", "Embedding"),
            ("EMBED2_METRICS", "Embedding 2"),
            ("RERANK_METRICS", "Reranker"),
            ("TASK_METRICS", "Task Model"),
            ("OCR_METRICS", "OCR"),
        ]:
            self.assertEqual(sections.get(key), section, key)

    def test_metrics_changes_restart_only_their_own_backend(self):
        self.assertEqual(manager.RESTART_HINTS["CHAT_PRIMARY_METRICS"], ["chat-backend-dense"])
        self.assertEqual(manager.RESTART_HINTS["CHAT2_METRICS"], ["chat-backend2"])
        self.assertEqual(manager.RESTART_HINTS["EMBED_METRICS"], ["embed"])
        self.assertEqual(manager.RESTART_HINTS["TASK_METRICS"], ["task"])

    def test_launchers_gate_the_flag_on_the_env_key(self):
        root = pathlib.Path(__file__).resolve().parents[1] / "scripts"
        for script, prefix in [
            ("start-chat-backend.sh", "CHAT"),
            ("start-chat-backend2.sh", "CHAT2"),
            ("start-chat-backend-moe.sh", "CHAT"),
            ("start-chat-backend-dense.sh", "CHAT"),
            ("start-embed.sh", "EMBED"),
            ("start-embed2.sh", "EMBED2"),
            ("start-rerank.sh", "RERANK"),
            ("start-task.sh", "TASK"),
            ("start-ocr.sh", "OCR"),
        ]:
            text = (root / script).read_text()
            self.assertIn(f'"${{{prefix}_METRICS:-on}}" == "on" ]] && OPTS+=(--metrics)', text, script)

    def test_metrics_toggle_is_a_recognised_config_key(self):
        filtered = manager.filter_config_updates({"CHAT_PRIMARY_METRICS": "off"}, env={})
        self.assertEqual(filtered, {"CHAT_PRIMARY_METRICS": "off"})


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
        for backend in ("chat-backend-dense", "chat-backend-moe", "chat-backend",
                        "chat-backend2", "embed", "rerank", "task", "ocr"):
            self.assertNotIn(backend, services)
        self.assertIn("llm-manager", services)
        self.assertIn("chat-proxy", services)

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
            patch.object(manager, "read_env", return_value=env),
            patch.object(manager, "get_gpu_info", return_value=self.GPUS),
            patch.object(manager, "read_meminfo", return_value=self.MEMINFO),
        )

    def test_budget_prices_the_configured_backend(self):
        env = {"CHAT_PRIMARY_MODEL_PATH": str(self.model),
               "CHAT_PRIMARY_CTX_SIZE": "262144", "CHAT_PRIMARY_N_PARALLEL": "2",
               "CHAT_PRIMARY_CTX_CHECKPOINTS": "8", "CHAT_PRIMARY_CACHE_RAM": "8192"}
        client, *patches = self._client(env)
        with client, patches[0], patches[1], patches[2]:
            response = client.get("/api/backend/budget")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIsNone(payload["error"])
        self.assertEqual(payload["prediction"]["per_slot_context"], 131072)
        self.assertEqual(payload["geometry"]["recurrent_layers"], 48)

    def test_query_parameters_price_an_unsaved_edit(self):
        env = {"CHAT_PRIMARY_MODEL_PATH": str(self.model),
               "CHAT_PRIMARY_CTX_SIZE": "262144", "CHAT_PRIMARY_N_PARALLEL": "2"}
        client, *patches = self._client(env)
        with client, patches[0], patches[1], patches[2]:
            response = client.get("/api/backend/budget?ctx_size=65536")
        self.assertEqual(response.get_json()["prediction"]["per_slot_context"], 32768)

    def test_unknown_query_parameters_are_ignored(self):
        env = {"CHAT_PRIMARY_MODEL_PATH": str(self.model), "CHAT_PRIMARY_CTX_SIZE": "131072"}
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
        self.assertIn("chat-primary", response.get_json()["backends"])

    def test_missing_model_reports_without_failing_the_request(self):
        client, *patches = self._client({"CHAT_PRIMARY_MODEL_PATH": "/nope.gguf"})
        with client, patches[0], patches[1], patches[2]:
            response = client.get("/api/backend/budget")
        self.assertEqual(response.status_code, 200)
        self.assertIn("model not found", response.get_json()["error"])

    def test_recommendation_is_derived_from_detected_hardware(self):
        env = {"CHAT_PRIMARY_MODEL_PATH": str(self.model)}
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
        client, *patches = self._client({"CHAT_PRIMARY_MODEL_PATH": "/nope.gguf"})
        with client, patches[0], patches[1], patches[2]:
            response = client.get("/api/backend/budget/recommend")
        self.assertEqual(response.status_code, 400)
        self.assertIn("could not read model metadata", response.get_json()["error"])


if __name__ == "__main__":
    unittest.main()
