# voice-tts: Hermes's voice on llms GPU 0

Added 2026-10-06 for Hermes's voice mode: spoken conversation in the Hermes desktop app,
with replies spoken as the model writes them.

## What it is

- **Model:** Qwen3-TTS 1.7B CustomVoice (Apache-2.0), run by
  [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) with CUDA graphs.
- **Voice:** the preset speaker `sohee`, Ellie's pick from an audition of every preset (the page is
  `~/Documents/Hermes Media/Voice Audition/2026-10-06/` on the Studio).
- **Where:** llms GPU 0, beside the embeddings and the 9B task model. It uses about 4.3 GB, so GPU 0
  holds about 18 of its 24 GB. GPU 0 was chosen over the Studio: the Studio was faster (127 ms against
  ~250 ms first audio), but llms sees less traffic.
- **API:** `scripts/voice-tts-server.py` serves an OpenAI-style speech API on `100.124.56.11:8016`
  (tailnet only, no auth, like the other llms services).
  - `POST /v1/audio/speech` takes `input`, `voice`, `response_format` (`pcm` or `wav`) and
    optional `instructions` (a style note, e.g. "calm, unhurried").
  - It streams 16-bit mono PCM at 24 kHz as it's generated, with `X-Audio-Sample-Rate: 24000`.
  - Also `GET /health`, `/v1/models` and `/v1/audio/voices`.
- **Speed** (measured 2026-10-06 on a 3090):
  - about 250–300 ms to first audio;
  - 2.9× faster than real time;
  - loading and warm-up take about 10 s once the weights are cached.

## Guards

- **One request at a time.** CUDA graphs aren't shareable. Hermes sends one sentence per request, in order.
- **Length cap.** `max_new_tokens` is computed from the text: 1.5 s plus 0.12 s per character. In the
  audition, Qwen3-TTS sometimes failed to stop: "Okay. Keep going." came out as 20 s of audio. `/health`
  counts `capped` replies.
- **Barge-in.** When the client disconnects, generation stops at the next chunk (about 0.6 s of audio).
  `/health` counts `cancelled`.
- **Unknown voices.** A voice not in the registry falls back to the default, so a client asking for
  `alloy` still gets a voice.
- **Input limit.** Input over 2,000 characters is refused.

## Files on llms

| What | Where |
|---|---|
| venv | `~/AI/voice-tts/venv` (torch 2.11 cu128, torchaudio cu128, transformers 5.15.1, faster-qwen3-tts 0.5.4) |
| voices | `~/AI/voice-tts/voices.json`: `{"default": "sohee", "voices": {"sohee": {"speaker": "sohee", "instruct": ""}}}` |
| weights | `~/.cache/huggingface` (the unit runs with `HF_HUB_OFFLINE=1`) |
| audition script | in the hermes repo, `scripts/voice_audition.py` |

venv pitfalls:
- **torchaudio:** pip's default torchaudio is built for CUDA 13. Install it from the cu128 index too, or
  transformers can't import `AutoProcessor`.
- **transformers:** 5.19 breaks `qwen_tts`'s rope shim (`MimiConfig` has no `rope_theta`), so keep it at 5.15.1.

Rebuild the venv:

```
python3 -m venv ~/AI/voice-tts/venv
~/AI/voice-tts/venv/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
~/AI/voice-tts/venv/bin/pip install faster-qwen3-tts==0.5.4 transformers==5.15.1 fastapi "uvicorn[standard]"
```

## Install, check, restart

Install on llms:

```
cp /mnt/LLMs/llamacpp/llm-stack-git/systemd/user/voice-tts.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now voice-tts.service
```

Check and listen:

```
curl -s http://llms:8016/health
curl -s http://llms:8016/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"input":"Hello there.","response_format":"wav"}' -o /tmp/hello.wav
```

Logs: `journalctl --user -u voice-tts -n 20`. There's one `speak {...}` line per request, with
first-audio ms, audio seconds, and whether it was capped or cancelled.

**Changing voices:** edit `voices.json`, then `systemctl --user restart voice-tts`. Hermes's desktop
falls back to Piper while the service is down.

## Not yet

- **Custom voices.** Cloning and voice design need the Base and VoiceDesign checkpoints. Loading
  either beside CustomVoice adds about 4 GB on GPU 0. Fine-tuning has to be planned first.
