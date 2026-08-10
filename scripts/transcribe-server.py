#!/usr/bin/env python3
"""Speech-to-text sidecar for llm-stack.

Hosts the ASR runtimes llama.cpp cannot: faster-whisper (CTranslate2), NeMo
(Parakeet, Canary) and plain transformers. Exposes an OpenAI-compatible
`/v1/audio/transcriptions` so existing SDKs work by changing `base_url` alone,
plus a richer `/transcribe` that returns segments, word timings and provenance
for agents that store what they get back.

Two design rules run through the whole file.

**Engines import lazily.** `import faster_whisper`, `import nemo` and
`from transformers import pipeline` all happen inside `Engine.load()`, never at
module scope. Registration costs nothing, so a host with only one runtime
installed still starts, still serves `/health`, and answers a request naming a
missing engine with a 503 that says which `--engines` flag would install it.
Importing at module scope is what would turn "NeMo is not installed" into "the
service is down".

**One model is resident at a time, and not for long.** The auxiliary models are
pooled by llama-router on a GPU that is usually near full, so a transcription
model that sits resident is the thing that breaks the rest of the stack. The
manager below loads on demand, unloads after an idle timeout, and can ask the
router to drop its own models first so the two never stack.
"""

from __future__ import annotations

import argparse
import base64
import gc
import json
import logging
import multiprocessing
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
import wave
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from flask import Flask, Response, jsonify, request

log = logging.getLogger("transcribe")

# The states llama.cpp's router reports for a model that is holding VRAM. Quoted
# rather than imported: this process runs in its own venv and cannot reach the
# manager's `web/` tree. Keep in step with `public_api.ROUTER_RESIDENT_STATES`.
ROUTER_RESIDENT_STATES = {"loaded", "ready", "active", "resident"}

RESPONSE_FORMATS = ("json", "verbose_json", "text", "srt", "vtt", "markdown")
# `markdown` is ours. A compatible endpoint that accepts non-standard values is
# worse than one that refuses them, so /v1/* takes only OpenAI's five.
OPENAI_RESPONSE_FORMATS = ("json", "verbose_json", "text", "srt", "vtt")

TRUTHY = {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------

class TranscribeError(Exception):
    """Carries the machine-readable type an agent branches on."""

    status = 400
    type = "bad_request"

    def __init__(self, message: str, hint: str = "", **extra):
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.extra = extra

    def payload(self) -> dict:
        body = {"type": self.type, "message": self.message}
        if self.hint:
            body["hint"] = self.hint
        body.update(self.extra)
        return body


class BadRequest(TranscribeError):
    status, type = 400, "bad_request"


class TooLarge(TranscribeError):
    status, type = 413, "too_large"


class UnsupportedCapability(TranscribeError):
    status, type = 422, "unsupported_capability"


class EngineUnavailable(TranscribeError):
    """The runtime is not installed. The hint names the install flag."""

    status, type = 503, "engine_unavailable"


class ModelLoadFailed(TranscribeError):
    status, type = 503, "model_load_failed"


class DecodeFailed(TranscribeError):
    status, type = 500, "decode_failed"


class UpstreamError(TranscribeError):
    status, type = 502, "upstream_error"


# ---------------------------------------------------------------------------
# engine registry
# ---------------------------------------------------------------------------

ENGINES: dict[str, type["Engine"]] = {}


def engine(engine_id: str, runtime: str, install_extra: str = ""):
    """Register an engine class. Registration only — nothing is imported."""

    def wrap(cls):
        cls.engine_id = engine_id
        cls.runtime = runtime
        cls.install_extra = install_extra
        ENGINES[engine_id] = cls
        return cls

    return wrap


class Engine:
    engine_id = ""
    runtime = ""
    install_extra = ""
    capabilities: dict[str, bool] = {
        "segments": True, "word_timestamps": False,
        "translate": False, "diarization": False, "language_detect": False,
    }

    def __init__(self, cfg: dict, engine_cfg: dict):
        self.cfg = cfg
        self.engine_cfg = engine_cfg
        self.model = None
        self.model_ref = ""

    # -- lifecycle ---------------------------------------------------------
    def load(self, model_ref: str) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def unload(self) -> None:
        self.model = None

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def transcribe(self, path: str, req: "TranscribeRequest") -> dict:  # pragma: no cover
        raise NotImplementedError

    # -- helpers -----------------------------------------------------------
    def _unavailable(self, exc: Exception) -> EngineUnavailable:
        flag = self.install_extra or self.runtime
        return EngineUnavailable(
            f"the {self.runtime} runtime required by {self.engine_id} is not installed ({exc})",
            hint=f"bash scripts/install-transcribe.sh --engines {flag}",
            engine=self.engine_id,
        )

    def _resolve(self, model_ref: str) -> tuple[str, str]:
        """`preset:x` / `local:/path` -> (kind, value)."""
        raw = (model_ref or "").strip()
        if raw.startswith("preset:"):
            return "preset", raw.split(":", 1)[1]
        if raw.startswith("local:"):
            return "local", raw.split(":", 1)[1]
        return ("legacy", raw) if raw else ("", "")


@engine("faster-whisper", "faster-whisper", "faster-whisper")
class FasterWhisperEngine(Engine):
    capabilities = {
        "segments": True, "word_timestamps": True,
        "translate": True, "diarization": False, "language_detect": True,
    }

    def load(self, model_ref: str) -> None:
        try:
            from faster_whisper import WhisperModel
        except Exception as exc:
            raise self._unavailable(exc) from exc
        _, value = self._resolve(model_ref)
        if not value:
            raise ModelLoadFailed("no model configured for faster-whisper",
                                  hint="set FASTER_WHISPER_LOCAL_MODEL", engine=self.engine_id)
        try:
            self.model = WhisperModel(
                value,
                device=self.cfg["runtime"]["device"],
                compute_type=self.cfg["runtime"]["compute_type"],
                download_root=self.engine_cfg.get("models_dir") or None,
            )
        except Exception as exc:
            raise ModelLoadFailed(f"faster-whisper could not load {value!r}: {exc}",
                                  engine=self.engine_id) from exc
        self.model_ref = model_ref

    def transcribe(self, path: str, req: "TranscribeRequest") -> dict:
        kwargs: dict[str, Any] = {
            "beam_size": req.beam_size,
            "task": "translate" if req.translate else "transcribe",
            "word_timestamps": req.word_timestamps,
            "vad_filter": req.vad,
        }
        if req.language:
            kwargs["language"] = req.language
        if req.initial_prompt:
            kwargs["initial_prompt"] = req.initial_prompt
        if req.temperature is not None:
            kwargs["temperature"] = req.temperature
        if req.hotwords:
            kwargs["hotwords"] = req.hotwords
        try:
            segments, info = self.model.transcribe(path, **kwargs)
            out = []
            for idx, seg in enumerate(segments):  # generator: this is the decode
                words = [
                    {"start": round(w.start, 3), "end": round(w.end, 3),
                     "word": w.word, "probability": round(getattr(w, "probability", 0.0), 4)}
                    for w in (getattr(seg, "words", None) or [])
                ]
                out.append({
                    "id": idx,
                    "start": round(seg.start, 3),
                    "end": round(seg.end, 3),
                    "text": (seg.text or "").strip(),
                    "speaker": None,
                    "avg_logprob": round(getattr(seg, "avg_logprob", 0.0), 4),
                    "no_speech_prob": round(getattr(seg, "no_speech_prob", 0.0), 4),
                    "compression_ratio": round(getattr(seg, "compression_ratio", 0.0), 4),
                    "words": words,
                })
        except Exception as exc:
            raise DecodeFailed(f"faster-whisper failed: {exc}", engine=self.engine_id) from exc
        return {
            "segments": out,
            "language": getattr(info, "language", "") or "",
            "language_probability": round(getattr(info, "language_probability", 0.0) or 0.0, 4),
            "duration": round(getattr(info, "duration", 0.0) or 0.0, 3),
        }


class _NemoEngine(Engine):
    capabilities = {
        "segments": True, "word_timestamps": True,
        "translate": False, "diarization": False, "language_detect": False,
    }

    def load(self, model_ref: str) -> None:
        try:
            from nemo.collections.asr.models import ASRModel
        except Exception as exc:
            raise self._unavailable(exc) from exc
        kind, value = self._resolve(model_ref)
        if not value:
            raise ModelLoadFailed(f"no model configured for {self.engine_id}", engine=self.engine_id)
        try:
            if kind == "local" and value.endswith(".nemo"):
                self.model = ASRModel.restore_from(value)
            elif kind == "local":
                self.model = ASRModel.restore_from(str(next(Path(value).glob("*.nemo"))))
            else:
                self.model = ASRModel.from_pretrained(value)
            device = self.cfg["runtime"]["device"]
            if device.startswith("cuda"):
                self.model = self.model.cuda()
            self.model.eval()
        except StopIteration as exc:
            raise ModelLoadFailed(f"no .nemo checkpoint under {value!r}", engine=self.engine_id) from exc
        except Exception as exc:
            raise ModelLoadFailed(f"NeMo could not load {value!r}: {exc}", engine=self.engine_id) from exc
        self.model_ref = model_ref

    def unload(self) -> None:
        self.model = None

    def transcribe(self, path: str, req: "TranscribeRequest") -> dict:
        try:
            kwargs: dict[str, Any] = {"batch_size": 1}
            if req.word_timestamps:
                kwargs["timestamps"] = True
            results = self.model.transcribe([path], **kwargs)
            item = results[0] if results else None
            text = getattr(item, "text", None)
            if text is None:
                text = item if isinstance(item, str) else ""
            segments = self._segments_from(item, text)
        except Exception as exc:
            raise DecodeFailed(f"{self.engine_id} failed: {exc}", engine=self.engine_id) from exc
        return {"segments": segments, "language": req.language or "", "language_probability": 0.0,
                "duration": segments[-1]["end"] if segments else 0.0}

    @staticmethod
    def _segments_from(item: Any, text: str) -> list[dict]:
        """NeMo returns a timestamp dict only when asked; fall back to one span."""
        stamps = getattr(item, "timestamp", None) or {}
        rows = stamps.get("segment") or stamps.get("word") or []
        segments = []
        for idx, row in enumerate(rows):
            segments.append({
                "id": idx,
                "start": round(float(row.get("start", 0.0)), 3),
                "end": round(float(row.get("end", 0.0)), 3),
                "text": str(row.get("segment") or row.get("word") or "").strip(),
                "speaker": None, "avg_logprob": 0.0, "no_speech_prob": 0.0,
                "compression_ratio": 0.0, "words": [],
            })
        if not segments and text:
            segments = [{"id": 0, "start": 0.0, "end": 0.0, "text": text.strip(), "speaker": None,
                         "avg_logprob": 0.0, "no_speech_prob": 0.0, "compression_ratio": 0.0,
                         "words": []}]
        return segments


@engine("parakeet-v3", "nemo", "nemo")
class ParakeetEngine(_NemoEngine):
    pass


@engine("canary-qwen", "nemo", "nemo")
class CanaryQwenEngine(_NemoEngine):
    capabilities = dict(_NemoEngine.capabilities, translate=True)


@engine("hf-asr", "hf", "hf")
class HuggingFaceEngine(Engine):
    capabilities = {
        "segments": True, "word_timestamps": True,
        "translate": True, "diarization": False, "language_detect": True,
    }

    def load(self, model_ref: str) -> None:
        try:
            import torch
            from transformers import pipeline as hf_pipeline
        except Exception as exc:
            raise self._unavailable(exc) from exc
        _, value = self._resolve(model_ref)
        if not value:
            raise ModelLoadFailed("no model configured for hf-asr",
                                  hint="set HF_ASR_LOCAL_MODEL", engine=self.engine_id)
        device = self.cfg["runtime"]["device"]
        try:
            self.model = hf_pipeline(
                "automatic-speech-recognition",
                model=value,
                torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
                device=0 if device.startswith("cuda") else -1,
            )
        except Exception as exc:
            raise ModelLoadFailed(f"transformers could not load {value!r}: {exc}",
                                  engine=self.engine_id) from exc
        self.model_ref = model_ref

    def transcribe(self, path: str, req: "TranscribeRequest") -> dict:
        try:
            kwargs: dict[str, Any] = {"return_timestamps": "word" if req.word_timestamps else True}
            generate: dict[str, Any] = {}
            if req.language:
                generate["language"] = req.language
            if req.translate:
                generate["task"] = "translate"
            if generate:
                kwargs["generate_kwargs"] = generate
            out = self.model(path, **kwargs)
        except Exception as exc:
            raise DecodeFailed(f"hf-asr failed: {exc}", engine=self.engine_id) from exc
        chunks = out.get("chunks") or []
        segments = []
        for idx, chunk in enumerate(chunks):
            start, end = (chunk.get("timestamp") or (0.0, 0.0))[:2]
            segments.append({
                "id": idx, "start": round(float(start or 0.0), 3), "end": round(float(end or 0.0), 3),
                "text": str(chunk.get("text", "")).strip(), "speaker": None,
                "avg_logprob": 0.0, "no_speech_prob": 0.0, "compression_ratio": 0.0, "words": [],
            })
        if not segments:
            segments = [{"id": 0, "start": 0.0, "end": 0.0, "text": str(out.get("text", "")).strip(),
                         "speaker": None, "avg_logprob": 0.0, "no_speech_prob": 0.0,
                         "compression_ratio": 0.0, "words": []}]
        return {"segments": segments, "language": req.language or "", "language_probability": 0.0,
                "duration": segments[-1]["end"]}


@engine("router", "router", "")
class RouterEngine(Engine):
    """Forwards to llama-router, where the audio GGUF is pooled with the rest.

    llama.cpp's transcription endpoint runs an audio LLM through the chat path,
    so it returns prose and nothing else: no segments, no word timings, and
    `response_format` may only ever be `json` (server-chat.cpp rejects the
    others outright). This engine therefore holds no weights and reports itself
    as timestamp-free, and the caller decides whether a single whole-file cue is
    an acceptable substitute or an error.
    """

    capabilities = {
        "segments": False, "word_timestamps": False,
        "translate": False, "diarization": False, "language_detect": False,
    }

    def load(self, model_ref: str) -> None:
        _, value = self._resolve(model_ref)
        self.model_ref = model_ref
        self.model = value or self.cfg["router"]["model"]

    @property
    def loaded(self) -> bool:
        return bool(self.model)

    def transcribe(self, path: str, req: "TranscribeRequest") -> dict:
        import requests

        router = self.cfg["router"]
        url = f"http://{router['host']}:{router['port']}/v1/audio/transcriptions"
        # The multipart field must be named exactly `file`, and response_format
        # must be `json` — anything else is a 400 from llama.cpp itself.
        data = {"model": self.model, "response_format": "json"}
        if req.language:
            data["language"] = req.language
        if req.initial_prompt:
            data["prompt"] = req.initial_prompt
        if req.temperature is not None:
            data["temperature"] = str(req.temperature)
        try:
            with open(path, "rb") as handle:
                resp = requests.post(url, data=data,
                                     files={"file": (os.path.basename(path), handle)},
                                     timeout=self.cfg["limits"]["timeout_seconds"])
        except Exception as exc:
            raise UpstreamError(f"model router unreachable at {url}: {exc}",
                                engine=self.engine_id) from exc
        if resp.status_code >= 400:
            raise UpstreamError(f"model router returned {resp.status_code}: {resp.text[:400]}",
                                engine=self.engine_id)
        try:
            text = (resp.json().get("text") or "").strip()
        except Exception as exc:
            raise UpstreamError(f"model router returned unparseable JSON: {exc}",
                                engine=self.engine_id) from exc
        duration = probe_duration(path) or 0.0
        return {
            "segments": [{"id": 0, "start": 0.0, "end": round(duration, 3), "text": text,
                          "speaker": None, "avg_logprob": 0.0, "no_speech_prob": 0.0,
                          "compression_ratio": 0.0, "words": []}],
            "language": req.language or "", "language_probability": 0.0,
            "duration": round(duration, 3), "degraded": True,
        }


# ---------------------------------------------------------------------------
# audio
# ---------------------------------------------------------------------------

def probe_duration(path: str) -> float | None:
    """Seconds of audio, or None when we genuinely cannot tell.

    Used to decide whether a request goes async, so guessing is worse than
    admitting ignorance: an unknown duration runs synchronously, which is the
    behaviour the caller already asked for.
    """
    try:
        with wave.open(path, "rb") as handle:
            rate = handle.getframerate()
            if rate:
                return handle.getnframes() / float(rate)
    except Exception:
        pass
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=30,
        )
        value = out.stdout.strip()
        return float(value) if value and value != "N/A" else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# VRAM discipline
# ---------------------------------------------------------------------------

class ModelManager:
    """Keeps at most one transcription model on the GPU, and not for long.

    `_lock` guards which model is resident; `_inflight` counts running decodes
    so the idle sweeper can never unload a model out from under one.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._lock = threading.RLock()
        self._sem = threading.BoundedSemaphore(max(1, int(cfg["limits"]["max_concurrency"])))
        self._resident: Engine | None = None
        self._resident_key: tuple[str, str] | None = None
        self._loaded_at = 0.0
        self._last_used = time.monotonic()
        self._inflight = 0
        self._stop = threading.Event()

    # -- residency ---------------------------------------------------------
    def acquire(self, engine_id: str, model_ref: str) -> Engine:
        cls = ENGINES.get(engine_id)
        if cls is None:
            raise BadRequest(f"unknown engine {engine_id!r}",
                             hint=f"known engines: {', '.join(sorted(ENGINES))}")
        key = (engine_id, model_ref)
        with self._lock:
            if self._resident is not None and self._resident_key == key:
                self._last_used = time.monotonic()
                self._inflight += 1
                return self._resident
            if self._resident is not None:
                self._release_locked()
            # Symmetric on purpose. Loading locally while the router holds a
            # model stacks two models on one GPU; proxying to the router while
            # we hold one does exactly the same thing from the other side.
            if cls.runtime == "router":
                pass
            else:
                self._router_yield()
            inst = cls(self.cfg, self.cfg["engines"].get(engine_id, {}))
            inst.load(model_ref)
            self._resident, self._resident_key = inst, key
            self._loaded_at = time.monotonic()
            self._last_used = self._loaded_at
            self._inflight += 1
            log.info("loaded %s (%s)", engine_id, model_ref)
            return inst

    def release(self) -> None:
        with self._lock:
            self._inflight = max(0, self._inflight - 1)
            self._last_used = time.monotonic()

    def unload(self) -> bool:
        with self._lock:
            if self._resident is None:
                return False
            self._release_locked()
            return True

    def _release_locked(self) -> None:
        engine_id = self._resident_key[0] if self._resident_key else "?"
        try:
            self._resident.unload()
        except Exception as exc:  # a failed unload must not wedge the manager
            log.warning("unload of %s raised: %s", engine_id, exc)
        self._resident, self._resident_key = None, None
        gc.collect()
        # Imported opportunistically: a faster-whisper-only install is
        # CTranslate2 and has no torch, and must not be made to grow one.
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
        log.info("released %s", engine_id)

    # -- router ------------------------------------------------------------
    def _router_yield(self) -> None:
        """Best effort, and never fatal.

        Yielding is an optimisation: it frees VRAM we would rather have, but a
        router that is off, unreachable, or a venv without `requests` is not a
        reason to fail a transcription that would otherwise succeed. Letting an
        exception out of here would also mask the `engine_unavailable` error the
        caller actually needs, because this runs before the engine loads.
        """
        try:
            self._router_yield_inner()
        except Exception as exc:
            log.warning("router yield skipped: %s", exc)

    def _router_yield_inner(self) -> None:
        mode = (self.cfg["router"]["yield_mode"] or "off").strip().lower()
        if mode == "off":
            return
        import requests

        base = f"http://{self.cfg['router']['host']}:{self.cfg['router']['port']}"
        targets: list[str] = []
        if mode == "all":
            try:
                # `/models` is documented as never loading a model, so polling
                # it here cannot itself cause the swap we are trying to avoid.
                resp = requests.get(f"{base}/models", timeout=10)
                for item in (resp.json() or {}).get("data", []):
                    state = item.get("status")
                    state = state.get("value") if isinstance(state, dict) else state
                    if str(state or "").lower() in ROUTER_RESIDENT_STATES:
                        targets.append(item.get("id") or item.get("name") or "")
            except Exception as exc:
                log.warning("router yield could not list models: %s", exc)
                return
        else:
            targets = [self.cfg["router"]["model"]]
        for name in [t for t in targets if t]:
            try:
                resp = requests.post(f"{base}/models/unload", json={"model": name}, timeout=60)
            except Exception as exc:
                log.warning("router yield failed for %s: %s", name, exc)
                continue
            # 400 "model is not running" is the common path, not a failure:
            # nothing was resident, which is exactly the state we wanted.
            if resp.status_code >= 400 and "not running" not in resp.text:
                log.warning("router yield failed for %s: %s %s", name, resp.status_code,
                            resp.text[:200])

    def router_status(self) -> dict:
        try:
            import requests
            base = f"http://{self.cfg['router']['host']}:{self.cfg['router']['port']}"
            resp = requests.get(f"{base}/models", timeout=5)
            data = resp.json() or {}
            return {"reachable": True,
                    "models": [item.get("id") or item.get("name") for item in data.get("data", [])]}
        except Exception as exc:
            return {"reachable": False, "error": str(exc), "models": []}

    # -- idle sweeper ------------------------------------------------------
    def start_idle_thread(self) -> None:
        idle = float(self.cfg["limits"]["idle_unload_seconds"])
        if idle <= 0:
            log.info("idle unload disabled")
            return
        thread = threading.Thread(target=self._idle_loop, args=(idle,), daemon=True,
                                  name="transcribe-idle")
        thread.start()

    def _idle_loop(self, idle: float) -> None:
        tick = min(5.0, max(0.05, idle / 4.0))
        while not self._stop.wait(tick):
            with self._lock:
                if (self._resident is not None and self._inflight == 0
                        and time.monotonic() - self._last_used >= idle):
                    log.info("idle for %.0fs, releasing", idle)
                    self._release_locked()

    def stop(self) -> None:
        self._stop.set()

    # -- reporting ---------------------------------------------------------
    def snapshot(self) -> dict:
        idle = float(self.cfg["limits"]["idle_unload_seconds"])
        with self._lock:
            if self._resident is None:
                return {"resident": None}
            idle_for = time.monotonic() - self._last_used
            return {"resident": {
                "engine": self._resident_key[0],
                "model": self._resident_key[1],
                "loaded_at": round(self._loaded_at, 3),
                "idle_for": round(idle_for, 1),
                "unload_in": round(max(0.0, idle - idle_for), 1) if idle > 0 else None,
                "inflight": self._inflight,
            }}

    def guard(self):
        return self._sem


# ---------------------------------------------------------------------------
# request model
# ---------------------------------------------------------------------------

class TranscribeRequest:
    def __init__(self, form: dict, cfg: dict, openai_mode: bool):
        self.cfg = cfg
        self.openai_mode = openai_mode
        self.engine_id, self.model_ref = self._resolve_model(form, cfg)
        self.language = (form.get("language") or "").strip()
        self.initial_prompt = (form.get("prompt") or form.get("initial_prompt") or "").strip()
        self.translate = _bool(form.get("translate"), False)
        self.vad = _bool(form.get("vad"), True)
        self.hotwords = (form.get("hotwords") or "").strip()
        self.beam_size = _int(form.get("beam_size"), 5)
        self.temperature = _float_or_none(form.get("temperature"))
        self.is_async = _bool(form.get("async"), False)
        granularities = form.get("timestamp_granularities") or form.get("timestamp_granularities[]") or ""
        if isinstance(granularities, (list, tuple)):
            granularities = ",".join(str(g) for g in granularities)
        self.word_timestamps = _bool(form.get("word_timestamps"), "word" in str(granularities))
        self.response_format = self._resolve_format(form, cfg, openai_mode)

    @staticmethod
    def _resolve_format(form: dict, cfg: dict, openai_mode: bool) -> str:
        fmt = (form.get("response_format") or "").strip().lower()
        if not fmt:
            fmt = "json" if openai_mode else cfg["limits"]["default_format"]
        allowed = OPENAI_RESPONSE_FORMATS if openai_mode else RESPONSE_FORMATS
        if fmt not in allowed:
            raise BadRequest(
                f"unsupported response_format {fmt!r}",
                hint=("this endpoint accepts " + ", ".join(allowed)
                      + ("; use /transcribe for markdown" if openai_mode else "")),
            )
        return fmt

    @staticmethod
    def _resolve_model(form: dict, cfg: dict) -> tuple[str, str]:
        """Accepts '', 'whisper-1', '<engine>', or '<engine>:<model-ref>'.

        Stock SDK code hardcodes `whisper-1`; refusing it would make every
        unmodified OpenAI client fail on its first call.
        """
        raw = (form.get("engine") or form.get("model") or "").strip()
        if raw in ("", "default", "whisper-1", "whisper-large-v3"):
            engine_id = cfg["active_engine"]
            return engine_id, cfg["engines"].get(engine_id, {}).get("model", "")
        if raw in ENGINES:
            return raw, cfg["engines"].get(raw, {}).get("model", "")
        head, sep, tail = raw.partition(":")
        if sep and head in ENGINES:
            # `<engine>:preset:x` and `<engine>:local:/p` keep their kind prefix;
            # a bare tail is a repo id, which the runtimes resolve themselves.
            return head, tail if tail.startswith(("preset:", "local:")) else f"preset:{tail}"
        raise BadRequest(
            f"unknown model {raw!r}",
            hint="use an engine id, '<engine>:<model>', or omit it for the default",
        )


def _bool(value, default: bool) -> bool:
    if value is None or value == "":
        return default
    return str(value).strip().lower() in TRUTHY


def _int(value, default: int) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return default


def _float_or_none(value):
    try:
        return float(str(value).strip())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# formatters
# ---------------------------------------------------------------------------

def _stamp(seconds: float, comma: bool) -> str:
    seconds = max(0.0, float(seconds))
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis == 1000:  # rounding can carry into the next second
        secs, millis = secs + 1, 0
    sep = "," if comma else "."
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{millis:03d}"


def to_srt(result: dict) -> str:
    lines = []
    for idx, seg in enumerate(result["segments"], start=1):
        lines.append(str(idx))
        lines.append(f"{_stamp(seg['start'], True)} --> {_stamp(seg['end'], True)}")
        lines.append(seg["text"])
        lines.append("")
    return "\n".join(lines)


def to_vtt(result: dict) -> str:
    lines = ["WEBVTT", ""]
    for seg in result["segments"]:
        lines.append(f"{_stamp(seg['start'], False)} --> {_stamp(seg['end'], False)}")
        lines.append(seg["text"])
        lines.append("")
    return "\n".join(lines)


def to_markdown(result: dict) -> str:
    out = ["# Transcript", ""]
    meta = [
        f"- **Engine:** {result['engine']} (`{result['model']}`)",
        f"- **Language:** {result.get('language') or 'unknown'}",
        f"- **Duration:** {result.get('duration', 0):.1f}s",
    ]
    if result.get("degraded"):
        meta.append("- **Note:** this engine returns no timeline; the whole file is one span")
    out.extend(meta)
    out.append("")
    current_speaker = None
    for seg in result["segments"]:
        speaker = seg.get("speaker")
        if speaker and speaker != current_speaker:
            out.extend(["", f"## Speaker {speaker}", ""])
            current_speaker = speaker
        minutes, secs = divmod(int(seg["start"]), 60)
        out.append(f"**[{minutes:02d}:{secs:02d}]** {seg['text']}")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def to_verbose_json(result: dict) -> dict:
    """OpenAI's exact key set, and nothing else."""
    return {
        "task": "translate" if result.get("translated") else "transcribe",
        "language": result.get("language", ""),
        "duration": result.get("duration", 0.0),
        "text": result["text"],
        "segments": result["segments"],
        "words": result.get("words", []),
    }


def render(result: dict, fmt: str) -> Response:
    # `content_type` and not `mimetype`: the latter appends its own charset, so
    # passing one here produces "charset=utf-8; charset=utf-8".
    if fmt == "text":
        return Response(result["text"] + "\n", content_type="text/plain; charset=utf-8")
    if fmt == "srt":
        return Response(to_srt(result), content_type="application/x-subrip; charset=utf-8")
    if fmt == "vtt":
        return Response(to_vtt(result), content_type="text/vtt; charset=utf-8")
    if fmt == "markdown":
        return Response(to_markdown(result), content_type="text/markdown; charset=utf-8")
    if fmt == "verbose_json":
        return jsonify(to_verbose_json(result))
    return jsonify(result)


def check_timeline_support(result_capabilities: dict, fmt: str, cfg: dict, engine_id: str) -> None:
    """Refuse subtitle formats from an engine that has no timeline.

    Emitting a single cue spanning the whole file as though it were a real
    timeline is worse than an error: a caller that asked for subtitles gets
    something that looks valid, loads in a player, and is wrong everywhere.
    """
    if fmt not in ("srt", "vtt", "verbose_json"):
        return
    if result_capabilities.get("segments", True):
        return
    if _bool(cfg["router"]["allow_degraded"], False):
        return
    raise UnsupportedCapability(
        f"the {engine_id} engine returns no timestamps, so {fmt} would be fabricated",
        hint=("set TRANSCRIPT_ROUTER_ALLOW_DEGRADED=on to accept a single whole-file cue, "
              "or use a local engine such as faster-whisper"),
        engine=engine_id,
    )


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


def create_job(engine_id: str, model_ref: str) -> dict:
    job = {"id": uuid.uuid4().hex[:16], "status": "queued", "stage": "queued",
           "progress": 0, "created_at": time.time(), "updated_at": time.time(),
           "engine": engine_id, "model": model_ref, "result": None, "error": None}
    with JOBS_LOCK:
        JOBS[job["id"]] = job
    return job


def update_job(job_id: str, **fields) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job:
            job.update(fields)
            job["updated_at"] = time.time()


def sweep_jobs(ttl: float) -> None:
    cutoff = time.time() - ttl
    with JOBS_LOCK:
        for job_id in [k for k, v in JOBS.items()
                       if v["updated_at"] < cutoff and v["status"] in ("done", "error")]:
            JOBS.pop(job_id, None)


# ---------------------------------------------------------------------------
# app
# ---------------------------------------------------------------------------

def load_config(path: str | None) -> dict:
    if path and Path(path).is_file():
        return json.loads(Path(path).read_text())
    return default_config()


def default_config() -> dict:
    """Env-driven fallback so the server is runnable without the start script."""
    def env(key, default=""):
        return os.environ.get(key, default)

    engines = {}
    for engine_id, prefix in (("faster-whisper", "FASTER_WHISPER"), ("parakeet-v3", "PARAKEET_V3"),
                              ("canary-qwen", "CANARY_QWEN"), ("hf-asr", "HF_ASR"),
                              ("router", "ROUTER_ASR")):
        engines[engine_id] = {
            "model": env(f"{prefix}_LOCAL_MODEL", ""),
            "backend_type": env(f"{prefix}_BACKEND_TYPE", "local"),
            "models_dir": "",
        }
    return {
        "server": {
            "host": env("TRANSCRIPT_HOST", "127.0.0.1"),
            "port": int(env("TRANSCRIPT_PORT", "8014")),
            "token": env("TRANSCRIPT_API_TOKEN", ""),
            "log_level": env("TRANSCRIPT_LOG_LEVEL", "INFO"),
        },
        "active_engine": env("TRANSCRIPT_ACTIVE_ENGINE", "faster-whisper"),
        "runtime": {
            "device": env("TRANSCRIPT_LOCAL_DEVICE", "cuda"),
            "compute_type": env("TRANSCRIPT_LOCAL_COMPUTE_TYPE", "float16"),
        },
        "router": {
            "host": env("MODEL_ROUTER_HOST", "127.0.0.1"),
            "port": int(env("MODEL_ROUTER_PORT", "8013")),
            "model": env("ASR_MODEL_NAME", "asr"),
            "yield_mode": env("TRANSCRIPT_ROUTER_YIELD", "asr"),
            "allow_degraded": env("TRANSCRIPT_ROUTER_ALLOW_DEGRADED", "off"),
        },
        "limits": {
            "idle_unload_seconds": float(env("TRANSCRIPT_IDLE_UNLOAD_SECONDS", "300")),
            "max_concurrency": int(env("TRANSCRIPT_MAX_CONCURRENCY", "1")),
            "max_upload_mb": int(env("TRANSCRIPT_MAX_UPLOAD_MB", "512")),
            "async_threshold_seconds": float(env("TRANSCRIPT_ASYNC_THRESHOLD_SECONDS", "900")),
            "job_ttl_seconds": float(env("TRANSCRIPT_JOB_TTL_SECONDS", "3600")),
            "timeout_seconds": float(env("TRANSCRIPT_TIMEOUT_SECONDS", "600")),
            "default_format": env("TRANSCRIPT_DEFAULT_FORMAT", "json"),
            "url_allow_hosts": env("TRANSCRIPT_URL_ALLOW_HOSTS", ""),
            "oai_allow_long": env("TRANSCRIPT_OAI_ALLOW_LONG", "off"),
            "work_dir": env("TRANSCRIPT_WORK_DIR", tempfile.gettempdir()),
        },
        "engines": engines,
    }


def create_app(config_path: str | None = None, cfg: dict | None = None) -> Flask:
    cfg = cfg or load_config(config_path)
    logging.basicConfig(
        level=getattr(logging, str(cfg["server"]["log_level"]).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    manager = ModelManager(cfg)
    work_dir = Path(cfg["limits"]["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = int(cfg["limits"]["max_upload_mb"]) * 1024 * 1024
    app.config["transcribe_cfg"] = cfg
    app.config["transcribe_manager"] = manager

    # -- plumbing ----------------------------------------------------------
    def authorised() -> bool:
        token = (cfg["server"]["token"] or "").strip()
        if not token:
            return True
        header = request.headers.get("Authorization", "")
        if header.startswith("Bearer "):
            return header[7:].strip() == token
        return request.args.get("token", "") == token

    def openai_error(exc: TranscribeError):
        # SDKs parse OpenAI's shape and nothing else, and that shape has no
        # field for a hint — so fold it into the message rather than drop the
        # one part of the error that says what to do about it.
        message = f"{exc.message} ({exc.hint})" if exc.hint else exc.message
        return jsonify({"error": {"message": message, "type": exc.type, "code": exc.type}}), exc.status

    def native_error(exc: TranscribeError):
        return jsonify({"ok": False, "error": exc.payload()}), exc.status

    @app.before_request
    def _auth():
        if request.path in ("/health", "/v1/health"):
            return None
        if authorised():
            return None
        exc = TranscribeError("missing or invalid access token")
        exc.status, exc.type = 401, "unauthorized"
        return openai_error(exc) if request.path.startswith("/v1/") else native_error(exc)

    def spool(req_files, form) -> tuple[str, str]:
        """Land the audio on disk and return (path, original filename)."""
        upload = req_files.get("file") if req_files else None
        if upload is not None and upload.filename:
            suffix = Path(upload.filename).suffix or ".audio"
            handle = tempfile.NamedTemporaryFile(dir=work_dir, suffix=suffix, delete=False)
            upload.save(handle.name)
            handle.close()
            return handle.name, upload.filename
        b64 = form.get("audio_base64") or ""
        if b64:
            name = form.get("filename") or "audio.wav"
            raw = b64.strip()
            if raw.startswith("data:") and "," in raw:
                header, _, payload = raw.partition(",")
                if ";base64" not in header:
                    raise BadRequest("data URI must be base64-encoded")
                raw = payload
            try:
                blob = base64.b64decode(raw)
            except Exception as exc:
                raise BadRequest(f"audio_base64 is not valid base64: {exc}") from exc
            handle = tempfile.NamedTemporaryFile(dir=work_dir,
                                                 suffix=Path(name).suffix or ".wav", delete=False)
            handle.write(blob)
            handle.close()
            return handle.name, name
        url = (form.get("url") or "").strip()
        if url:
            return fetch_url(url), Path(urlparse(url).path).name or "audio"
        raise BadRequest("no audio supplied",
                         hint="send multipart 'file', or JSON 'audio_base64', or 'url'")

    def fetch_url(url: str) -> str:
        allowed = {h.strip().lower() for h in
                   (cfg["limits"]["url_allow_hosts"] or "").split(",") if h.strip()}
        host = (urlparse(url).hostname or "").lower()
        # Blank means deny, not allow: this process can reach the whole tailnet.
        if not allowed or host not in allowed:
            raise BadRequest(
                f"fetching audio from {host or url!r} is not allowed",
                hint="add the host to TRANSCRIPT_URL_ALLOW_HOSTS (blank denies every fetch)",
            )
        import requests
        try:
            resp = requests.get(url, timeout=cfg["limits"]["timeout_seconds"], stream=True)
            resp.raise_for_status()
        except Exception as exc:
            raise UpstreamError(f"could not fetch {url}: {exc}") from exc
        cap = int(cfg["limits"]["max_upload_mb"]) * 1024 * 1024
        handle = tempfile.NamedTemporaryFile(dir=work_dir,
                                             suffix=Path(urlparse(url).path).suffix or ".audio",
                                             delete=False)
        written = 0
        for chunk in resp.iter_content(1 << 16):
            written += len(chunk)
            if written > cap:
                handle.close()
                os.unlink(handle.name)
                raise TooLarge(f"remote audio exceeds {cfg['limits']['max_upload_mb']}MB")
            handle.write(chunk)
        handle.close()
        return handle.name

    def run_transcription(path: str, req: TranscribeRequest, filename: str) -> dict:
        started = time.monotonic()
        with manager.guard():
            queued_ms = (time.monotonic() - started) * 1000
            load_started = time.monotonic()
            inst = manager.acquire(req.engine_id, req.model_ref)
            load_ms = (time.monotonic() - load_started) * 1000
            try:
                decode_started = time.monotonic()
                raw = inst.transcribe(path, req)
                decode_ms = (time.monotonic() - decode_started) * 1000
            finally:
                manager.release()

        segments = raw.get("segments") or []
        text = " ".join(seg["text"] for seg in segments if seg["text"]).strip()
        words = [w for seg in segments for w in (seg.get("words") or [])]
        duration = float(raw.get("duration") or (segments[-1]["end"] if segments else 0.0))
        total_ms = (time.monotonic() - started) * 1000
        return {
            "ok": True,
            "request_id": uuid.uuid4().hex[:12],
            "created": int(time.time()),
            "filename": filename,
            "text": text,
            "language": raw.get("language", ""),
            "language_probability": raw.get("language_probability", 0.0),
            "duration": round(duration, 3),
            "segments": segments,
            "words": words,
            "engine": req.engine_id,
            "model": req.model_ref,
            "device": cfg["runtime"]["device"],
            "compute_type": cfg["runtime"]["compute_type"],
            "translated": req.translate,
            "degraded": bool(raw.get("degraded")),
            "capabilities": dict(inst.capabilities),
            "timings": {
                "queued_ms": round(queued_ms, 1),
                "load_ms": round(load_ms, 1),
                "decode_ms": round(decode_ms, 1),
                "total_ms": round(total_ms, 1),
                "audio_seconds": round(duration, 3),
                "realtime_factor": round(duration / (decode_ms / 1000), 2) if decode_ms > 0 else 0.0,
            },
        }

    def collect_form() -> dict:
        form = dict(request.form)
        if request.is_json:
            body = request.get_json(silent=True) or {}
            if isinstance(body, dict):
                form.update({k: v for k, v in body.items()})
        return form

    def handle(openai_mode: bool):
        form = collect_form()
        req = TranscribeRequest(form, cfg, openai_mode)
        cls = ENGINES.get(req.engine_id)
        if cls is None:
            raise BadRequest(f"unknown engine {req.engine_id!r}")
        check_timeline_support(cls.capabilities, req.response_format, cfg, req.engine_id)

        path, filename = spool(request.files, form)
        cleanup = True
        try:
            duration = probe_duration(path)
            threshold = float(cfg["limits"]["async_threshold_seconds"])
            long_audio = duration is not None and threshold > 0 and duration > threshold
            if openai_mode and long_audio and not _bool(cfg["limits"]["oai_allow_long"], False):
                raise TooLarge(
                    f"audio is {duration:.0f}s, over the {threshold:.0f}s synchronous limit",
                    hint="POST /transcribe instead — it returns a job id you can poll",
                )
            if not openai_mode and (req.is_async or long_audio):
                job = create_job(req.engine_id, req.model_ref)
                cleanup = False  # the worker owns the file now
                threading.Thread(target=_run_job, args=(job["id"], path, req, filename),
                                 daemon=True, name=f"transcribe-job-{job['id']}").start()
                return jsonify({
                    "ok": True, "job_id": job["id"], "status": "queued",
                    "poll": f"/jobs/{job['id']}",
                    "estimated_seconds": round(duration / 20.0, 1) if duration else None,
                }), 202
            result = run_transcription(path, req, filename)
            return render(result, req.response_format)
        finally:
            if cleanup:
                _unlink(path)

    def _run_job(job_id: str, path: str, req: TranscribeRequest, filename: str) -> None:
        update_job(job_id, status="running", stage="transcribing", progress=10)
        try:
            result = run_transcription(path, req, filename)
            update_job(job_id, status="done", stage="complete", progress=100, result=result)
        except TranscribeError as exc:
            update_job(job_id, status="error", stage="failed", error=exc.payload())
        except Exception as exc:
            log.exception("job %s failed", job_id)
            update_job(job_id, status="error", stage="failed",
                       error={"type": "decode_failed", "message": str(exc)})
        finally:
            _unlink(path)
            sweep_jobs(float(cfg["limits"]["job_ttl_seconds"]))

    # -- routes ------------------------------------------------------------
    @app.route("/health")
    @app.route("/v1/health")
    def health():
        return jsonify({"status": "ok", "service": "transcript-backend",
                        "engines": sorted(ENGINES)})

    @app.route("/engines")
    def engines_route():
        snap = manager.snapshot()
        rows = []
        for engine_id, cls in sorted(ENGINES.items()):
            rows.append({
                "id": engine_id,
                "runtime": cls.runtime,
                "install_extra": cls.install_extra,
                "capabilities": dict(cls.capabilities),
                "model": cfg["engines"].get(engine_id, {}).get("model", ""),
                "backend_type": cfg["engines"].get(engine_id, {}).get("backend_type", "local"),
            })
        return jsonify({
            "ok": True, "active_engine": cfg["active_engine"], "engines": rows,
            "resident": snap["resident"],
            "idle_unload_seconds": cfg["limits"]["idle_unload_seconds"],
            "router": dict(manager.router_status(), yield_mode=cfg["router"]["yield_mode"]),
            "device": cfg["runtime"]["device"],
        })

    @app.route("/v1/models")
    def list_models():
        data = []
        for engine_id, cls in sorted(ENGINES.items()):
            data.append({"id": engine_id, "object": "model", "owned_by": cls.runtime})
            model = cfg["engines"].get(engine_id, {}).get("model", "")
            if model:
                data.append({"id": f"{engine_id}:{model}", "object": "model", "owned_by": cls.runtime})
        return jsonify({"object": "list", "data": data})

    @app.route("/unload", methods=["POST"])
    def unload_route():
        return jsonify({"ok": True, "unloaded": manager.unload(), **manager.snapshot()})

    @app.route("/jobs")
    def jobs_route():
        with JOBS_LOCK:
            return jsonify({"ok": True, "jobs": [
                {k: v for k, v in job.items() if k != "result"} for job in JOBS.values()
            ]})

    @app.route("/jobs/<job_id>")
    def job_route(job_id):
        with JOBS_LOCK:
            job = JOBS.get(job_id)
        if not job:
            return jsonify({"ok": False, "error": {"type": "not_found",
                                                   "message": f"no job {job_id}"}}), 404
        return jsonify({"ok": True, **job})

    @app.route("/jobs/<job_id>", methods=["DELETE"])
    def job_delete(job_id):
        with JOBS_LOCK:
            existed = JOBS.pop(job_id, None) is not None
        return jsonify({"ok": True, "deleted": existed})

    @app.route("/transcribe", methods=["POST"])
    def transcribe_route():
        try:
            return handle(openai_mode=False)
        except TranscribeError as exc:
            return native_error(exc)

    @app.route("/v1/audio/transcriptions", methods=["POST"])
    def oai_transcriptions():
        try:
            response = handle(openai_mode=True)
        except TranscribeError as exc:
            return openai_error(exc)
        # OpenAI's `json` for this endpoint is {"text": ...} and nothing more.
        if isinstance(response, Response) and response.mimetype == "application/json":
            body = response.get_json()
            if isinstance(body, dict) and body.get("ok") is True and "text" in body:
                return jsonify({"text": body["text"]})
        return response

    @app.route("/v1/audio/translations", methods=["POST"])
    def oai_translations():
        try:
            return handle_translation()
        except TranscribeError as exc:
            return openai_error(exc)

    def handle_translation():
        form = collect_form()
        form["translate"] = "true"
        req = TranscribeRequest(form, cfg, openai_mode=True)
        cls = ENGINES.get(req.engine_id)
        if cls is not None and not cls.capabilities.get("translate"):
            raise UnsupportedCapability(f"the {req.engine_id} engine cannot translate",
                                        engine=req.engine_id)
        check_timeline_support(cls.capabilities, req.response_format, cfg, req.engine_id)
        path, filename = spool(request.files, form)
        try:
            result = run_transcription(path, req, filename)
            if req.response_format == "json":
                return jsonify({"text": result["text"]})
            return render(result, req.response_format)
        finally:
            _unlink(path)

    @app.errorhandler(413)
    def _too_large(_exc):
        body = {"type": "too_large",
                "message": f"upload exceeds TRANSCRIPT_MAX_UPLOAD_MB ({cfg['limits']['max_upload_mb']}MB)"}
        if request.path.startswith("/v1/"):
            return jsonify({"error": {"message": body["message"], "type": "too_large",
                                      "code": "too_large"}}), 413
        return jsonify({"ok": False, "error": body}), 413

    manager.start_idle_thread()
    return app


def _unlink(path: str) -> None:
    try:
        os.unlink(path)
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="llm-stack transcription sidecar")
    parser.add_argument("--config", default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    # NeMo and torch dataloaders fork badly; the OCR SDK sets this for the same
    # reason. Harmless for the CTranslate2-only install.
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    app = create_app(args.config)
    cfg = app.config["transcribe_cfg"]
    host = args.host or cfg["server"]["host"]
    port = args.port or int(cfg["server"]["port"])
    log.info("transcription sidecar on %s:%s (engines: %s)", host, port, ", ".join(sorted(ENGINES)))
    try:
        app.run(host=host, port=port, threaded=True)
    finally:
        app.config["transcribe_manager"].stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
