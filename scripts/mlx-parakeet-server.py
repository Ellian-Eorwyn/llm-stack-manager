#!/usr/bin/env python3
"""English-only OpenAI-compatible Parakeet v3 server using Apple MLX."""

from __future__ import annotations

import argparse
import asyncio
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

import mlx.core as mx
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from mlx_audio.stt.utils import load_model


MODEL_PATH = os.environ.get("MLX_PARAKEET_MODEL_PATH", "")
MODEL_ALIAS = os.environ.get("MLX_PARAKEET_MODEL_NAME", "parakeet-v3-en")
CHUNK_SECONDS = float(os.environ.get("MLX_PARAKEET_CHUNK_SECONDS", "60"))
OVERLAP_SECONDS = float(os.environ.get("MLX_PARAKEET_OVERLAP_SECONDS", "5"))
MAX_UPLOAD_MB = int(os.environ.get("TRANSCRIPT_MAX_UPLOAD_MB", "512"))
WORK_DIR = Path(os.environ.get("TRANSCRIPT_WORK_DIR", "logs/transcript/work"))

_model = None
_load_lock = asyncio.Lock()
_inference_lock = asyncio.Lock()
_started_at = time.time()
_requests = 0


def _load_model_sync():
    global _model
    if _model is None:
        if not MODEL_PATH:
            raise RuntimeError("MLX_PARAKEET_MODEL_PATH is not configured")
        _model = load_model(MODEL_PATH)
    return _model


async def _ensure_model():
    if _model is None:
        async with _load_lock:
            if _model is None:
                await asyncio.to_thread(_load_model_sync)
    return _model


def _tokens_to_words(tokens) -> list[dict]:
    """Join Parakeet's tokenizer pieces into OpenAI-style word entries."""
    words = []
    current = None
    boundary_pending = False
    for token in tokens or []:
        piece = token.text or ""
        if not piece.strip():
            boundary_pending = True
            continue
        starts_word = piece[:1].isspace() or boundary_pending or current is None
        cleaned = piece.lstrip() if starts_word else piece
        if starts_word:
            if current is not None:
                words.append(current)
            current = {
                "word": cleaned,
                "start": float(token.start),
                "end": float(token.end),
            }
        else:
            current["word"] += cleaned
            current["end"] = float(token.end)
        boundary_pending = False
    if current is not None:
        words.append(current)
    return words


def _transcribe_sync(path: str):
    result = _load_model_sync().generate(
        path,
        dtype=mx.bfloat16,
        chunk_duration=CHUNK_SECONDS,
        overlap_duration=OVERLAP_SECONDS,
    )
    if not (getattr(result, "text", "") or "").strip():
        raise ValueError("no speech was detected in the uploaded audio")
    sentences = []
    words = []
    for index, sentence in enumerate(getattr(result, "sentences", []) or []):
        sentence_words = _tokens_to_words(getattr(sentence, "tokens", []) or [])
        words.extend(sentence_words)
        sentences.append(
            {
                "id": index,
                "start": float(sentence.start),
                "end": float(sentence.end),
                "text": sentence.text.strip(),
                "words": sentence_words,
            }
        )
    return {
        "text": result.text,
        "language": "en",
        "segments": sentences,
        "words": words,
        "model": MODEL_ALIAS,
        "backend": "mlx-audio",
    }


def _timestamp(seconds: float, *, vtt: bool = False) -> str:
    millis = max(0, round(seconds * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    separator = "." if vtt else ","
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def _subtitles(payload: dict, *, vtt: bool) -> str:
    lines = ["WEBVTT", ""] if vtt else []
    for index, segment in enumerate(payload["segments"], 1):
        lines.extend(
            [
                str(index),
                f"{_timestamp(segment['start'], vtt=vtt)} --> {_timestamp(segment['end'], vtt=vtt)}",
                segment["text"].strip(),
                "",
            ]
        )
    return "\n".join(lines)


@asynccontextmanager
async def lifespan(_: FastAPI):
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    await _ensure_model()
    yield
    mx.clear_cache()


app = FastAPI(title="Parakeet v3 English MLX", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "backend": "mlx-audio",
        "engine": "parakeet-v3",
        "language": "en",
        "model": MODEL_ALIAS,
        "model_path": MODEL_PATH,
        "loaded": _model is not None,
        "requests": _requests,
        "uptime_seconds": round(time.time() - _started_at, 3),
    }


@app.get("/engines")
async def engines():
    return {
        "engines": [
            {
                "id": "parakeet-v3",
                "installed": True,
                "resident": _model is not None,
                "model": MODEL_ALIAS,
                "language": "en",
            }
        ]
    }


@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ALIAS,
                "object": "model",
                "created": int(_started_at),
                "owned_by": "local-mlx",
            }
        ],
    }


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(MODEL_ALIAS),
    language: str = Form("en"),
    response_format: str = Form("json"),
    word_timestamps: bool = Form(False),
):
    del word_timestamps  # Word timestamps are always available from Parakeet.
    global _requests
    normalized_language = (language or "en").lower().replace("_", "-")
    if normalized_language not in {"en", "en-us", "en-gb", "english"}:
        raise HTTPException(status_code=400, detail="this endpoint is configured for English only")
    if model not in {MODEL_ALIAS, "parakeet-v3", "whisper-1"}:
        raise HTTPException(status_code=404, detail=f"unknown model: {model}")
    if response_format not in {"json", "verbose_json", "text", "srt", "vtt"}:
        raise HTTPException(status_code=400, detail="unsupported response_format")

    content = await file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"upload exceeds {MAX_UPLOAD_MB} MiB")
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(dir=WORK_DIR, suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        await _ensure_model()
        async with _inference_lock:
            payload = await asyncio.to_thread(_transcribe_sync, tmp_path)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"transcription failed: {exc}") from exc
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)

    _requests += 1
    if response_format == "text":
        return PlainTextResponse(payload["text"])
    if response_format == "srt":
        return PlainTextResponse(_subtitles(payload, vtt=False))
    if response_format == "vtt":
        return PlainTextResponse(_subtitles(payload, vtt=True))
    if response_format == "json":
        return JSONResponse({"text": payload["text"]})
    return JSONResponse(payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("TRANSCRIPT_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("TRANSCRIPT_PORT", "8014")))
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, workers=1, server_header=False)


if __name__ == "__main__":
    main()
