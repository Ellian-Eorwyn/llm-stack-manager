#!/usr/bin/env python3
"""Hermes's voice: Qwen3-TTS behind an OpenAI-style speech API, on llms GPU 0 (2026-10-06).

Serves POST /v1/audio/speech (OpenAI contract: model, input, voice, response_format
pcm|wav, optional instructions) and streams 16-bit mono PCM while it is generated,
announcing its rate in X-Audio-Sample-Rate (Hermes's tts_streaming reads that).
One model stays loaded (faster-qwen3-tts, CUDA graphs); requests run one at a time.

Voices come from a JSON file (VOICE_TTS_VOICES, default ~/AI/voice-tts/voices.json):
    {"default": "sohee",
     "voices": {"sohee": {"speaker": "sohee", "instruct": ""}}}
A request's voice may also be any preset speaker name; an unknown voice gets the
default (logged), so a client's "alloy" still speaks.

Guards:
  * length cap: max_new_tokens from the text (MAX_SECONDS_BASE + chars * MAX_SECONDS_PER_CHAR,
    12.5 codec frames a second), so a sampling runaway can't talk for a minute;
  * a client that disconnects (barge-in) stops generation at the next chunk;
  * input over MAX_CHARS is refused (Hermes sends one sentence at a time).

Run (systemd/user/voice-tts.service does this):
    CUDA_VISIBLE_DEVICES=0 ~/AI/voice-tts/venv/bin/python voice-tts-server.py --host 100.124.56.11 --port 8016
Check:
    curl -s http://llms:8016/health
    curl -s http://llms:8016/v1/audio/speech -H 'Content-Type: application/json' \
      -d '{"input":"Hello there.","voice":"sohee","response_format":"wav"}' -o /tmp/hello.wav
"""

import argparse
import asyncio
import json
import logging
import os
import queue
import struct
import threading
import time
from pathlib import Path


MODEL_ID = os.environ.get("VOICE_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
VOICES_FILE = Path(os.environ.get("VOICE_TTS_VOICES", str(Path.home() / "AI/voice-tts/voices.json")))
LANGUAGE = os.environ.get("VOICE_TTS_LANGUAGE", "English")
CHUNK_SIZE = int(os.environ.get("VOICE_TTS_CHUNK", "8"))  # codec frames per streamed chunk (~0.64 s)
FRAME_HZ = 12.5
MAX_SECONDS_BASE = 3.0
MAX_SECONDS_PER_CHAR = 0.15  # ~2x a slow speaker; sentence 3 of the audition (230 chars) caps at 37 s
MAX_CHARS = 2000

log = logging.getLogger("voice-tts")
state: dict = {"model": None, "voices": {}, "default": None, "speakers": [], "sr": 24000,
               "loaded_at": None, "served": 0, "capped": 0, "cancelled": 0, "last": None}
lock = threading.Lock()


def load_voices() -> None:
    cfg = {"default": "sohee", "voices": {"sohee": {"speaker": "sohee"}}}
    if VOICES_FILE.exists():
        cfg = json.loads(VOICES_FILE.read_text())
    voices = {name.lower(): v for name, v in (cfg.get("voices") or {}).items()}
    for name, v in voices.items():
        if v.get("speaker", name).lower() not in state["speakers"]:
            raise SystemExit(f"voice {name!r}: speaker {v.get('speaker')!r} not in {state['speakers']}")
    state["voices"] = voices
    state["default"] = (cfg.get("default") or next(iter(voices), "sohee")).lower()


def resolve(voice: str | None, instructions: str | None) -> tuple[str, str | None, str]:
    name = (voice or "").lower() or state["default"]
    if name in state["voices"]:
        v = state["voices"][name]
        return v.get("speaker", name).lower(), (instructions or v.get("instruct") or None), name
    if name in state["speakers"]:
        return name, instructions or None, name
    log.info("unknown voice %r; using %r", voice, state["default"])
    return resolve(state["default"], instructions)


def max_tokens_for(text: str) -> int:
    return int((MAX_SECONDS_BASE + len(text) * MAX_SECONDS_PER_CHAR) * FRAME_HZ)


def pcm16(audio) -> bytes:
    import numpy as np
    a = np.asarray(audio, dtype=np.float32).reshape(-1)
    return np.clip(a * 32767, -32768, 32767).astype("<i2").tobytes()


def wav_header(sr: int) -> bytes:
    # streaming WAV: sizes set to the maximum, as players accept for live audio
    return (b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVEfmt " +
            struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16) + b"data" + struct.pack("<I", 0xFFFFFFFF))


def generate(text: str, speaker: str, instruct: str | None, out: queue.Queue, stop: threading.Event) -> None:
    """Producer thread: put PCM bytes, then None. Holds the model lock throughout."""
    import numpy as np
    t0 = time.perf_counter()
    first = None
    samples = 0
    limit = max_tokens_for(text)
    capped = cancelled = False
    try:
        with lock:
            gen = state["model"].generate_custom_voice_streaming(
                text=text, speaker=speaker, language=LANGUAGE, instruct=instruct,
                max_new_tokens=limit, chunk_size=CHUNK_SIZE)
            for chunk, _sr, _timing in gen:
                if stop.is_set():
                    cancelled = True
                    break
                first = first or time.perf_counter() - t0
                samples += np.asarray(chunk).size
                out.put(pcm16(chunk))
            gen.close()
        seconds = samples / state["sr"]
        capped = not cancelled and seconds >= limit / FRAME_HZ - 1.0
    except Exception as exc:  # reported to the client as a short stream; logged in full
        log.exception("generation failed: %s", exc)
        out.put(exc)
    finally:
        out.put(None)
        seconds = samples / state["sr"]
        state["served"] += 1
        state["capped"] += int(capped)
        state["cancelled"] += int(cancelled)
        state["last"] = {"at": time.strftime("%Y-%m-%d %H:%M:%S"), "chars": len(text),
                         "first_audio_ms": round((first or 0) * 1000), "audio_s": round(seconds, 2),
                         "total_s": round(time.perf_counter() - t0, 2), "capped": capped,
                         "cancelled": cancelled}
        log.info("speak %s", json.dumps({"speaker": speaker, **state["last"]}))


def build_app():
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel

    app = FastAPI(title="voice-tts")

    class Speech(BaseModel):
        input: str
        model: str | None = None
        voice: str | None = None
        response_format: str = "pcm"
        instructions: str | None = None
        speed: float | None = None  # accepted, ignored (Qwen3-TTS has no speed control)

    @app.get("/health")
    def health():
        import torch
        return {"ok": state["model"] is not None, "model": MODEL_ID, "default_voice": state["default"],
                "voices": sorted(state["voices"]), "speakers": state["speakers"], "sample_rate": state["sr"],
                "loaded_at": state["loaded_at"], "busy": lock.locked(), "served": state["served"],
                "capped": state["capped"], "cancelled": state["cancelled"], "last": state["last"],
                "gpu_gb": round(torch.cuda.memory_reserved() / 2**30, 2)}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": "qwen3-tts", "object": "model", "owned_by": "local"}]}

    @app.get("/v1/audio/voices")
    def voices():
        return {"default": state["default"], "voices": sorted(state["voices"]), "speakers": state["speakers"]}

    @app.post("/v1/audio/speech")
    async def speech(req: Speech, request: Request):
        text = req.input.strip()
        fmt = req.response_format.lower()
        if not text:
            raise HTTPException(400, "input is empty")
        if len(text) > MAX_CHARS:
            raise HTTPException(400, f"input over {MAX_CHARS} characters; send one sentence or paragraph at a time")
        if fmt not in ("pcm", "wav"):
            raise HTTPException(400, f"response_format {fmt!r} not supported; use pcm or wav")
        speaker, instruct, name = resolve(req.voice, req.instructions)
        out: queue.Queue = queue.Queue()
        stop = threading.Event()
        threading.Thread(target=generate, args=(text, speaker, instruct, out, stop), daemon=True).start()
        loop = asyncio.get_running_loop()

        async def body():
            if fmt == "wav":
                yield wav_header(state["sr"])
            try:
                while True:
                    item = await loop.run_in_executor(None, out.get)
                    if item is None or isinstance(item, Exception):
                        return
                    if await request.is_disconnected():
                        return
                    yield item
            finally:
                stop.set()  # client gone (barge-in) or done: stop generating

        sr = state["sr"]
        media = "audio/wav" if fmt == "wav" else f"audio/pcm; rate={sr}"
        return StreamingResponse(body(), media_type=media,
                                 headers={"X-Audio-Sample-Rate": str(sr), "X-Voice": name})

    return app


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8016)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    from faster_qwen3_tts import FasterQwen3TTS
    t = time.perf_counter()
    model = FasterQwen3TTS.from_pretrained(MODEL_ID)
    state["speakers"] = sorted(s.lower() for s in model.model.model.config.talker_config.spk_id)
    state["sr"] = int(model.sample_rate)
    state["model"] = model
    load_voices()
    for _ in model.generate_custom_voice_streaming(text="Warming up.", speaker=resolve(None, None)[0],
                                                   language=LANGUAGE, chunk_size=CHUNK_SIZE):
        pass
    state["loaded_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    log.info("loaded %s in %.1fs; voices %s (default %s); %d Hz", MODEL_ID, time.perf_counter() - t,
             sorted(state["voices"]), state["default"], state["sr"])

    import uvicorn
    uvicorn.run(build_app(), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
