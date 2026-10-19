"""The Mac transcription server gives its memory back: scripts/mlx-parakeet-server.py.

MLX, FastAPI and the models are stubbed, so this runs in the plain test venv.
What is checked is the bookkeeping: buffers freed after every request, and every
model dropped once TRANSCRIPT_IDLE_UNLOAD_SECONDS pass without one.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import pathlib
import sys
import types
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _stub_modules(cleared: list) -> dict:
    mlx = types.ModuleType("mlx")
    core = types.ModuleType("mlx.core")
    core.bfloat16 = "bf16"
    core.clear_cache = lambda: cleared.append(1)
    mlx.core = core

    class App:
        def __init__(self, *a, **k):
            pass

        def _route(self, *a, **k):
            return lambda fn: fn

        get = post = _route

    fastapi = types.ModuleType("fastapi")
    fastapi.FastAPI = App
    fastapi.File = fastapi.Form = lambda *a, **k: None
    fastapi.HTTPException = type("HTTPException", (Exception,), {})
    fastapi.UploadFile = object
    responses = types.ModuleType("fastapi.responses")
    responses.JSONResponse = responses.PlainTextResponse = lambda x: x
    utils = types.ModuleType("mlx_audio.stt.utils")
    utils.load_model = lambda path: object()
    return {"mlx": mlx, "mlx.core": core, "uvicorn": types.ModuleType("uvicorn"),
            "fastapi": fastapi, "fastapi.responses": responses,
            "mlx_audio": types.ModuleType("mlx_audio"),
            "mlx_audio.stt": types.ModuleType("mlx_audio.stt"), "mlx_audio.stt.utils": utils}


def load_server(idle: str, cleared: list):
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        with mock.patch.dict(sys.modules, _stub_modules(cleared)), \
                mock.patch.dict(os.environ, {"TRANSCRIPT_IDLE_UNLOAD_SECONDS": idle,
                                             "MLX_PARAKEET_MODEL_PATH": "/models/parakeet"}):
            spec = importlib.util.spec_from_file_location("parakeet_server",
                                                          ROOT / "scripts" / "mlx-parakeet-server.py")
            server = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(server)
    finally:
        sys.path.remove(str(ROOT / "scripts"))
    return server


class IdleUnloadTests(unittest.TestCase):
    def setUp(self):
        self.cleared = []
        self.server = load_server("300", self.cleared)

    def test_the_stack_setting_is_read(self):
        self.assertEqual(self.server.IDLE_UNLOAD_SECONDS, 300)
        self.assertEqual(load_server("", []).IDLE_UNLOAD_SECONDS, 300)
        self.assertEqual(load_server("0", []).IDLE_UNLOAD_SECONDS, 0)

    def test_every_request_frees_the_buffer_cache(self):
        self.server._mark_used()
        self.assertEqual(len(self.cleared), 1)

    def test_unload_drops_every_model_once(self):
        s = self.server
        s._model, s._diarizer, s._embedder = object(), object(), object()
        self.assertTrue(s._unload_sync())
        self.assertIsNone(s._model)
        self.assertIsNone(s._diarizer)
        self.assertIsNone(s._embedder)
        self.assertEqual(s._unloads, 1)
        self.assertFalse(s._unload_sync())
        self.assertEqual(s._unloads, 1)

    def test_the_unloader_waits_out_the_idle_time(self):
        s = self.server
        s._model = object()
        clock = [1000.0]
        s._last_used = 1000.0
        sleeps = []

        async def fake_sleep(seconds):
            sleeps.append(seconds)
            clock[0] += 200
            if len(sleeps) >= 3:
                raise asyncio.CancelledError

        with mock.patch.object(s.time, "monotonic", lambda: clock[0]), \
                mock.patch.object(s.asyncio, "sleep", fake_sleep):
            s._inference_lock = asyncio.Lock()
            with self.assertRaises(asyncio.CancelledError):
                asyncio.run(s._idle_unloader())
        # 200 s idle: kept; 400 s: unloaded.
        self.assertIsNone(s._model)
        self.assertEqual(s._unloads, 1)
        self.assertEqual(sleeps[0], 60.0)

    def test_zero_keeps_the_models(self):
        s = load_server("0", [])
        s._model = object()
        asyncio.run(s._idle_unloader())
        self.assertIsNotNone(s._model)


if __name__ == "__main__":
    unittest.main()
