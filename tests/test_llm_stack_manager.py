from __future__ import annotations

import importlib.util
import json
import pathlib
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
    def test_beellama_fields_are_in_dedicated_section(self):
        sections = {f["key"]: f["section"] for f in manager.CONFIG_FIELDS}
        self.assertEqual(sections["CHAT_BEE_MODEL_PATH"], "BeeLLaMA Backend")
        self.assertEqual(sections["BEELLAMA_SERVER_BIN"], "BeeLLaMA Backend")

    def test_ocr_fields_restart_ocr_only(self):
        self.assertEqual(manager.RESTART_HINTS["OCR_MODEL_PATH"], ["ocr"])
        self.assertEqual(manager.RESTART_HINTS["OCR_PORT"], ["ocr"])

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


if __name__ == "__main__":
    unittest.main()
