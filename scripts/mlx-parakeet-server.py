#!/usr/bin/env python3
"""English-only OpenAI-compatible Parakeet v3 server using Apple MLX.

`diarize=true` adds speakers from NVIDIA Nemotron 3 Diarization (up to eight,
labelled `speaker_0`… in order of arrival); see speaker_attribution.py for how
the two models' timelines are joined. Where voice profiles exist
(scripts/enroll-voiceprints.py), each speaker is also compared with them and
named when the match is clear; see speaker_identity.py.
"""

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

import speaker_attribution
import speaker_identity


MODEL_PATH = os.environ.get("MLX_PARAKEET_MODEL_PATH", "")
MODEL_ALIAS = os.environ.get("MLX_PARAKEET_MODEL_NAME", "parakeet-v3-en")
CHUNK_SECONDS = float(os.environ.get("MLX_PARAKEET_CHUNK_SECONDS", "60"))
OVERLAP_SECONDS = float(os.environ.get("MLX_PARAKEET_OVERLAP_SECONDS", "5"))
MAX_UPLOAD_MB = int(os.environ.get("TRANSCRIPT_MAX_UPLOAD_MB", "512"))
WORK_DIR = Path(os.environ.get("TRANSCRIPT_WORK_DIR", "logs/transcript/work"))
DIARIZATION_MODEL_PATH = os.environ.get("MLX_DIARIZATION_MODEL_PATH", "")
DIARIZATION_THRESHOLD = float(os.environ.get("MLX_DIARIZATION_THRESHOLD", "0.5"))
VOICEPRINT_MODEL_PATH = os.environ.get("MLX_VOICEPRINT_MODEL_PATH", "")
VOICEPRINT_PROFILES = os.environ.get("MLX_VOICEPRINT_PROFILES", "")
# Blank: the threshold and margin the enrollment's own evaluation chose.
VOICEPRINT_THRESHOLD = os.environ.get("MLX_VOICEPRINT_THRESHOLD", "")
VOICEPRINT_MARGIN = os.environ.get("MLX_VOICEPRINT_MARGIN", "")

_model = None
_diarizer = None
_embedder = None
_profiles: dict = {"mtime": None, "data": None}
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


def _load_diarizer_sync():
    """Loaded on first use: most requests never ask for speakers."""
    global _diarizer
    if _diarizer is None:
        if not DIARIZATION_MODEL_PATH:
            raise RuntimeError("MLX_DIARIZATION_MODEL_PATH is not configured")
        from mlx_audio.vad import load as load_diarization

        _diarizer = load_diarization(DIARIZATION_MODEL_PATH, strict=True)
    return _diarizer


def _voiceprint_profiles():
    """Profiles re-read when the file changes, so re-enrolling needs no restart."""
    if not (VOICEPRINT_MODEL_PATH and VOICEPRINT_PROFILES):
        return None
    try:
        mtime = Path(VOICEPRINT_PROFILES).stat().st_mtime
    except OSError:
        return None
    if _profiles["mtime"] != mtime:
        _profiles.update(mtime=mtime, data=speaker_identity.load(VOICEPRINT_PROFILES))
    return _profiles["data"]


def _identify(payload: dict, path: str) -> None:
    """Put enrolled names on the diarized speakers where the voice is a clear match."""
    global _embedder
    data = _voiceprint_profiles()
    if not data or not payload.get("diarization"):
        return
    import numpy as np
    from speaker_embedding import SAMPLE_RATE, SpeakerEmbedder, decode

    if _embedder is None:
        _embedder = SpeakerEmbedder(VOICEPRINT_MODEL_PATH)
    audio = decode(path)
    clusters = {}
    for speaker, clips in speaker_identity.clean_spans(payload["diarization"]).items():
        vecs = [_embedder.embed(audio[int(a * SAMPLE_RATE):int(b * SAMPLE_RATE)]) for a, b in clips]
        if vecs:
            clusters[speaker] = np.mean(vecs, axis=0).tolist()
    threshold = float(VOICEPRINT_THRESHOLD or data.get("threshold") or speaker_identity.DEFAULT_THRESHOLD)
    margin = float(VOICEPRINT_MARGIN or data.get("margin") or speaker_identity.DEFAULT_MARGIN)
    matches = speaker_identity.match(clusters, data["profiles"], threshold, margin)
    people = [m["name"] for m in matches.values() if m["name"]]
    labels = {sp: speaker_identity.choose_label(m["name"], data["profiles"][m["name"]], people)
              for sp, m in matches.items() if m["name"]}
    for entry in payload["speakers"]:
        m = matches.get(entry["id"])
        if m:
            entry.update(name=labels.get(entry["id"]), voice_match=m["name"], voice_score=m["score"],
                         runner_up=m["runner_up"], runner_up_score=m["runner_up_score"])
    for segment in payload["segments"]:
        segment["speaker_name"] = labels.get(segment["speaker"])
    payload["identification"] = {"profiles": len(data["profiles"]), "threshold": threshold,
                                 "margin": margin, "built": data.get("built")}


def _diarize_sync(path: str):
    """Returns (segments, probabilities as nested lists, seconds per frame)."""
    model = _load_diarizer_sync()
    result = model.generate(path, threshold=DIARIZATION_THRESHOLD)
    processor = model.config.processor_config
    frame_seconds = (processor.hop_length / processor.sampling_rate
                     * model.config.output_subsampling_factor)
    segments = [
        {"start": round(float(seg.start), 3), "end": round(float(seg.end), 3),
         "speaker": speaker_attribution.label(int(seg.speaker))}
        for seg in result.segments
    ]
    return segments, result.speaker_probs.tolist(), frame_seconds


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


def _transcribe_sync(path: str, diarize: bool = False, identify: bool = True):
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
    payload = {
        "text": result.text,
        "language": "en",
        "segments": sentences,
        "words": words,
        "model": MODEL_ALIAS,
        "backend": "mlx-audio",
    }
    if diarize:
        _add_speakers(payload, path)
        if identify:
            _identify(payload, path)
    return payload


def _add_speakers(payload: dict, path: str) -> None:
    """Re-cut `segments` at speaker changes and label every word."""
    diarization, probs, frame_seconds = _diarize_sync(path)
    words = speaker_attribution.smooth(speaker_attribution.attribute(
        payload["words"], probs, frame_seconds, DIARIZATION_THRESHOLD))
    by_sentence, cursor = [], 0
    for sentence in payload["segments"]:
        count = len(sentence["words"])
        by_sentence.append(words[cursor:cursor + count])
        cursor += count
    segments = speaker_attribution.turns(by_sentence)
    payload.update(
        segments=segments,
        words=words,
        speakers=speaker_attribution.speaker_summary(segments),
        diarization=diarization,
        diarization_model=Path(DIARIZATION_MODEL_PATH).name,
    )


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
                (f"[{segment.get('speaker_name') or segment['speaker']}] " if segment.get("speaker") else "")
                + segment["text"].strip(),
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


def _diarization_status() -> dict:
    return {"configured": bool(DIARIZATION_MODEL_PATH),
            "model": Path(DIARIZATION_MODEL_PATH).name if DIARIZATION_MODEL_PATH else None,
            "loaded": _diarizer is not None,
            "voiceprints": _voiceprint_status()}


def _voiceprint_status() -> dict:
    data = _voiceprint_profiles()
    return {"configured": bool(VOICEPRINT_MODEL_PATH and VOICEPRINT_PROFILES),
            "profiles": len(data["profiles"]) if data else 0,
            "built": data.get("built") if data else None}


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
        "diarization": _diarization_status(),
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
                "capabilities": {"word_timestamps": True,
                                 "diarization": bool(DIARIZATION_MODEL_PATH)},
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


async def _save_upload(file: UploadFile) -> str:
    content = await file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"upload exceeds {MAX_UPLOAD_MB} MiB")
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=WORK_DIR, suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        return tmp.name


def _require_diarization():
    if not DIARIZATION_MODEL_PATH:
        raise HTTPException(
            status_code=503,
            detail="diarization is not configured: set MLX_DIARIZATION_MODEL_PATH "
                   "(scripts/install-mlx-runtime.sh --transcribe-only fetches the model)")


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(MODEL_ALIAS),
    language: str = Form("en"),
    response_format: str = Form("json"),
    word_timestamps: bool = Form(False),
    diarize: bool = Form(False),
    identify: bool = Form(True),
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
    if diarize:
        _require_diarization()

    tmp_path = ""
    try:
        tmp_path = await _save_upload(file)
        await _ensure_model()
        async with _inference_lock:
            payload = await asyncio.to_thread(_transcribe_sync, tmp_path, diarize, identify)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"transcription failed: {exc}") from exc
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)

    _requests += 1
    if response_format == "text":
        if diarize:
            return PlainTextResponse(speaker_attribution.render_text(payload["segments"]))
        return PlainTextResponse(payload["text"])
    if response_format == "srt":
        return PlainTextResponse(_subtitles(payload, vtt=False))
    if response_format == "vtt":
        return PlainTextResponse(_subtitles(payload, vtt=True))
    if response_format == "json":
        return JSONResponse({"text": payload["text"]})
    return JSONResponse(payload)


@app.post("/v1/audio/diarize")
async def diarize_only(file: UploadFile = File(...)):
    """Who spoke when, without a transcript — for pairing with another ASR."""
    _require_diarization()
    tmp_path = ""
    try:
        tmp_path = await _save_upload(file)
        async with _inference_lock:
            started = time.time()
            segments, _, _ = await asyncio.to_thread(_diarize_sync, tmp_path)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"diarization failed: {exc}") from exc
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)
    speakers = sorted({seg["speaker"] for seg in segments},
                      key=lambda sp: int(sp.rsplit("_", 1)[1]))
    return JSONResponse({"segments": segments, "speakers": speakers,
                         "model": Path(DIARIZATION_MODEL_PATH).name,
                         "elapsed_seconds": round(time.time() - started, 3)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("TRANSCRIPT_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("TRANSCRIPT_PORT", "8014")))
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, workers=1, server_header=False)


if __name__ == "__main__":
    main()
