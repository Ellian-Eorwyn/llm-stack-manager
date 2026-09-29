"""Unified-memory attribution: one row per service, weights counted wherever held."""

from __future__ import annotations

import os
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "web"))
import app  # noqa: E402
from platforms import darwin  # noqa: E402


class MergeTests(unittest.TestCase):
    def rows(self):
        return [
            {"pid": 1, "name": "llm-a", "process_name": "python3", "used_memory": 32000, "model": "hf/x", "alias": "a"},
            {"pid": 2, "name": "llm-a", "process_name": "splash", "used_memory": 300, "model": None, "alias": None},
            {"pid": 3, "name": "llama-router", "process_name": "llama-server", "used_memory": 5000, "model": "m1", "alias": None},
            {"pid": 4, "name": "llama-router", "process_name": "llama-server", "used_memory": 6000, "model": "m2", "alias": None},
        ]

    def test_a_service_is_one_row_holding_its_largest_process(self):
        merged = app._merge_service_processes(self.rows())
        llm_a = next(r for r in merged if r["name"] == "llm-a")
        self.assertEqual((llm_a["used_memory"], llm_a["pid"], sorted(llm_a["pids"])), (32300, 1, [1, 2]))
        self.assertNotIn("_largest", llm_a)

    def test_router_children_holding_different_models_stay_apart(self):
        merged = app._merge_service_processes(self.rows())
        self.assertEqual(sorted(r["model"] for r in merged if r["name"] == "llama-router"), ["m1", "m2"])


@unittest.skipUnless(sys.platform == "darwin", "mincore residency is read on macOS")
class FileResidencyTests(unittest.TestCase):
    def test_a_file_just_written_is_counted_and_cached(self):
        with tempfile.NamedTemporaryFile(delete=False) as handle:
            handle.write(os.urandom(4 * 1024 * 1024))
            path = handle.name.encode()
        self.addCleanup(os.unlink, handle.name)
        darwin._RESIDENT_CACHE.pop(path, None)
        first = darwin._file_resident_bytes(path)
        self.assertGreater(first, 0)
        self.assertLessEqual(first, 4 * 1024 * 1024)
        self.assertIn(path, darwin._RESIDENT_CACHE)

    def test_a_missing_file_is_zero(self):
        self.assertEqual(darwin._file_resident_bytes(b"/nonexistent/weights"), 0)


if __name__ == "__main__":
    unittest.main()
