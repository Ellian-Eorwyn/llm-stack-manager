#!/usr/bin/env python3
"""Hermes's voice: Qwen3-TTS behind an OpenAI-style speech API, on llms GPU 0 (2026-10-06).

Serves POST /v1/audio/speech (OpenAI contract: model, input, voice, response_format
pcm|wav, optional instructions) and streams 16-bit mono PCM while it is generated,
announcing its rate in X-Audio-Sample-Rate (Hermes's tts_streaming reads that).
One model stays loaded (faster-qwen3-tts, CUDA graphs); requests run one at a time.

Voices come from a JSON file (VOICE_TTS_VOICES, default ~/AI/voice-tts/voices.json), of two kinds:
    {"default": "sohee",
     "voices": {"sohee": {"speaker": "sohee", "instruct": ""},                    # a preset speaker
                "deep":  {"ref_audio": "voices/deep/reference.wav",              # a clone (2026-10-06):
                          "ref_text_file": "voices/deep/reference.txt"}}}       #   clip + what it says
Paths are relative to the voices file. Presets need the CustomVoice model, clones the Base
model; the server loads only the models its voices need (both fit on GPU 0 in one process,
~10 GB). A voice whose clip or speaker is missing is skipped and listed in /health.
A request's voice may also be any preset speaker name, or "default" (what Hermes asks for, so
changing "default" in voices.json changes Hermes's voice); an unknown voice gets the default
(logged), so a client's "alloy" still speaks.

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


MODELS = {"preset": os.environ.get("VOICE_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"),
          "clone": os.environ.get("VOICE_TTS_CLONE_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")}
VOICES_FILE = Path(os.environ.get("VOICE_TTS_VOICES", str(Path.home() / "AI/voice-tts/voices.json")))
LANGUAGE = os.environ.get("VOICE_TTS_LANGUAGE", "English")
CHUNK_SIZE = int(os.environ.get("VOICE_TTS_CHUNK", "8"))  # codec frames per streamed chunk (~0.64 s)
FRAME_HZ = 12.5
MAX_SECONDS_BASE = 1.5
MAX_SECONDS_PER_CHAR = 0.12  # ~1.6x a slow speaker; the audition's 14 s sentence (230 chars) caps at 29 s
MAX_CHARS = 2000

log = logging.getLogger("voice-tts")
state: dict = {"models": {}, "voices": {}, "skipped": {}, "default": None, "speakers": [], "sr": 24000,
               "loaded_at": None, "served": 0, "capped": 0, "cancelled": 0, "last": None}
lock = threading.Lock()


def read_config() -> dict:
    if VOICES_FILE.exists():
        return json.loads(VOICES_FILE.read_text())
    return {"default": "sohee", "voices": {"sohee": {"speaker": "sohee"}}}


def kinds_needed(cfg: dict) -> set:
    return {"clone" if "ref_audio" in v else "preset" for v in (cfg.get("voices") or {}).values()} or {"preset"}


def load_voices() -> None:
    """Read the voices file against the loaded models; bad voices are skipped, not fatal."""
    cfg = read_config()
    base = VOICES_FILE.parent
    voices, skipped = {}, {}
    for name, v in (cfg.get("voices") or {}).items():
        name = name.lower()
        if "ref_audio" in v:
            audio = (base / v["ref_audio"]).resolve()
            text = v.get("ref_text") or ((base / v["ref_text_file"]).read_text().strip()
                                         if v.get("ref_text_file") and (base / v["ref_text_file"]).exists() else "")
            if "clone" not in state["models"]:
                skipped[name] = "the clone model isn't loaded"
            elif not audio.exists() or not text:
                skipped[name] = f"missing {'clip ' + str(audio) if not audio.exists() else 'reference text'}"
            else:
                voices[name] = {"kind": "clone", "ref_audio": str(audio), "ref_text": text}
        else:
            speaker = v.get("speaker", name).lower()
            if speaker not in state["speakers"]:
                skipped[name] = f"speaker {speaker!r} not in the preset model"
            else:
                voices[name] = {"kind": "preset", "speaker": speaker, "instruct": v.get("instruct") or None}
    for name, why in skipped.items():
        log.error("voice %r skipped: %s", name, why)
    if not voices:
        raise SystemExit(f"no usable voices in {VOICES_FILE}: {skipped}")
    state["voices"], state["skipped"] = voices, skipped
    default = (cfg.get("default") or "").lower()
    state["default"] = default if default in voices else next(iter(voices))


def resolve(voice: str | None, instructions: str | None) -> tuple[str, dict]:
    """(name, spec) for a request; spec has kind preset (speaker, instruct) or clone (ref_audio, ref_text)."""
    name = (voice or "").lower()
    if name in ("", "default"):  # Hermes asks for "default", so the server's default is the one switch
        name = state["default"]
    if name in state["voices"]:
        v = dict(state["voices"][name])
        if v["kind"] == "preset":
            v["instruct"] = instructions or v.get("instruct")
        return name, v
    if name in state["speakers"]:
        return name, {"kind": "preset", "speaker": name, "instruct": instructions or None}
    log.info("unknown voice %r; using %r", voice, state["default"])
    return resolve(state["default"], instructions)


def stream_for(text: str, spec: dict, limit: int):
    if spec["kind"] == "clone":
        return state["models"]["clone"].generate_voice_clone_streaming(
            text=text, language=LANGUAGE, ref_audio=spec["ref_audio"], ref_text=spec["ref_text"],
            max_new_tokens=limit, chunk_size=CHUNK_SIZE)
    return state["models"]["preset"].generate_custom_voice_streaming(
        text=text, speaker=spec["speaker"], language=LANGUAGE, instruct=spec.get("instruct"),
        max_new_tokens=limit, chunk_size=CHUNK_SIZE)


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


def generate(text: str, name: str, spec: dict, out: queue.Queue, stop: threading.Event) -> None:
    """Producer thread: put PCM bytes, then None. Holds the model lock throughout."""
    import numpy as np
    t0 = time.perf_counter()
    first = None
    samples = 0
    limit = max_tokens_for(text)
    capped = cancelled = False
    try:
        with lock:
            gen = stream_for(text, spec, limit)
            for chunk, _sr, _timing in gen:
                if stop.is_set():
                    cancelled = True
                    break
                first = first or time.perf_counter() - t0
                samples += np.asarray(chunk).size
                out.put(pcm16(chunk))
            gen.close()
        seconds = samples / state["sr"]
        capped = not cancelled and seconds >= limit / FRAME_HZ - 0.25  # within ~3 codec frames of the cap
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
        log.info("speak %s", json.dumps({"voice": name, **state["last"]}))


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
        return {"ok": bool(state["models"]), "models": {k: MODELS[k] for k in state["models"]},
                "default_voice": state["default"], "voices": sorted(state["voices"]),
                "clones": sorted(n for n, v in state["voices"].items() if v["kind"] == "clone"),
                "skipped": state["skipped"], "speakers": state["speakers"], "sample_rate": state["sr"],
                "loaded_at": state["loaded_at"], "busy": lock.locked(), "served": state["served"],
                "capped": state["capped"], "cancelled": state["cancelled"], "last": state["last"],
                "gpu_gb": round(torch.cuda.memory_reserved() / 2**30, 2)}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": "qwen3-tts", "object": "model", "owned_by": "local"}]}

    @app.get("/v1/audio/voices")
    def voices():
        return {"default": state["default"], "voices": sorted(state["voices"]), "speakers": state["speakers"],
                "skipped": state["skipped"]}

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
        name, spec = resolve(req.voice, req.instructions)
        out: queue.Queue = queue.Queue()
        stop = threading.Event()
        threading.Thread(target=generate, args=(text, name, spec, out, stop), daemon=True).start()
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
    for kind in sorted(kinds_needed(read_config()), key=lambda k: k != "preset"):
        model = FasterQwen3TTS.from_pretrained(MODELS[kind])
        state["models"][kind] = model
        state["sr"] = int(model.sample_rate)
        if kind == "preset":
            state["speakers"] = sorted(s.lower() for s in model.model.model.config.talker_config.spk_id)
    load_voices()
    for name, spec in state["voices"].items():  # builds CUDA graphs once; caches each clone's encoded clip
        if spec["kind"] == "clone" or name == state["default"]:
            for _ in stream_for("Warming up.", spec, max_tokens_for("Warming up.")):
                pass
    state["loaded_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    log.info("loaded %s in %.1fs; voices %s (default %s); skipped %s; %d Hz",
             sorted(MODELS[k] for k in state["models"]), time.perf_counter() - t, sorted(state["voices"]),
             state["default"], sorted(state["skipped"]), state["sr"])

    import uvicorn
    uvicorn.run(build_app(), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
