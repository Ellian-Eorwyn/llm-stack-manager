import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import setup_engine


class ComponentSelectionTests(unittest.TestCase):
    def test_glmocr_selects_ocr_dependency(self):
        self.assertEqual(setup_engine.resolve_components(["glmocr-sdk"]), ["ocr", "glmocr-sdk"])

    def test_honcho_selects_primary_and_embedding(self):
        selected = setup_engine.resolve_components(["honcho"])
        self.assertIn("primary", selected)
        self.assertIn("embedding", selected)
        self.assertIn("honcho", selected)

    def test_selected_ports_are_unique_and_include_manager(self):
        self.assertEqual(setup_engine.selected_ports(["searxng", "playwright"]), [80, 8077])


class CudaDetectionTests(unittest.TestCase):
    def test_nvidia_query_parsing_collects_compute_architectures(self):
        gpus = setup_engine.parse_nvidia_gpus("0, RTX 3090, 8.6, 24576, 24000\n1, RTX 4090, 8.9, 24564, 20000\n")
        self.assertEqual([item["cmake_architecture"] for item in gpus], ["86", "89"])

    def test_toolkit_never_exceeds_driver(self):
        self.assertEqual(setup_engine.choose_cuda_toolkit("13.3"), "13.3")
        self.assertEqual(setup_engine.choose_cuda_toolkit("13.2"), "13.0")
        self.assertEqual(setup_engine.choose_cuda_toolkit("12.8"), "12.8")
        self.assertEqual(setup_engine.choose_cuda_toolkit("12.7"), "")


class ModelValidationTests(unittest.TestCase):
    def test_valid_gguf_header_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.gguf"
            path.write_bytes(b"GGUF" + b"\0" * 64)
            self.assertTrue(setup_engine.validate_gguf(path)["ok"])

    def test_xml_access_denied_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.gguf"
            path.write_text("<?xml version='1.0'?><Error><Code>AccessDenied</Code></Error>")
            result = setup_engine.validate_gguf(path)
            self.assertFalse(result["ok"])
            self.assertIn("Invalid GGUF header", result["error"])

    def test_html_download_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.gguf"
            path.write_text("<html><body>gateway error</body></html>")
            self.assertFalse(setup_engine.validate_gguf(path)["ok"])


class PlacementTests(unittest.TestCase):
    def setUp(self):
        self.gpus = [
            {"index": 0, "memory_total_mib": 24576, "memory_free_mib": 24000},
            {"index": 1, "memory_total_mib": 12288, "memory_free_mib": 12000},
        ]

    def test_primary_prefers_highest_vram_single_gpu(self):
        result = setup_engine.plan_gpu_placement(self.gpus, {"primary": {"size": 4 * 1024**3, "context_mib": 1024}})
        self.assertTrue(result["ok"])
        self.assertEqual(result["assignments"]["primary"]["gpu_indices"], [0])
        self.assertEqual(len(result["assignments"]["glmocr-sdk"]["gpu_indices"]), 1)

    def test_primary_spans_gpus_when_one_cannot_fit(self):
        result = setup_engine.plan_gpu_placement(self.gpus, {"primary": {"size": 25 * 1024**3, "context_mib": 1024}})
        self.assertTrue(result["ok"])
        self.assertEqual(result["assignments"]["primary"]["gpu_indices"], [0, 1])

    def test_over_capacity_blocks_without_override(self):
        result = setup_engine.plan_gpu_placement(self.gpus, {"primary": {"size": 40 * 1024**3, "context_mib": 4096}})
        self.assertFalse(result["ok"])


class StateAndFirewallTests(unittest.TestCase):
    def test_state_round_trip_preserves_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "install-state.json"
            state = setup_engine.default_state()
            state["selection"]["components"] = ["primary"]
            setup_engine.save_state(state, path)
            self.assertEqual(setup_engine.load_state(path)["selection"]["components"], ["primary"])

    def test_firewall_rules_are_private_subnet_scoped(self):
        rules = setup_engine.firewall_rules(["primary"], "192.168.4.0/24")
        self.assertTrue(all("192.168.4.0/24" in rule for rule in rules))
        with self.assertRaises(ValueError):
            setup_engine.firewall_rules(["primary"], "8.8.8.0/24")

    def test_runner_marks_active_jobs_interrupted_on_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "install-state.json"
            state = setup_engine.default_state()
            state["jobs"]["active"] = {"id": "active", "status": "running"}
            setup_engine.save_state(state, path)
            setup_engine.SetupRunner(path)
            self.assertEqual(setup_engine.load_state(path)["jobs"]["active"]["status"], "interrupted")


if __name__ == "__main__":
    unittest.main()
