# voice-tts: Hermes's voice on llms GPU 0

Added 2026-10-06 for Hermes's voice mode: spoken conversation in the Hermes desktop app,
with replies spoken as the model writes them.

## What it is

- **Model:** Qwen3-TTS 1.7B (Apache-2.0), run by
  [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) with CUDA graphs. Since 2026-10-07
  only the Base (clone) model is loaded: every served voice is a clone (see "Voices" below).
- **Voices:** deep, earth, lama, spark, ellie (clones). It started on the preset speaker `sohee` (Ellie's
  pick from the preset audition, `~/Documents/Hermes Media/Voice Audition/2026-10-06/` on the Studio).
- **Where:** llms GPU 0, beside the embeddings and the 9B task model. It uses about 5.7 GB, so GPU 0
  holds about 19.7 of its 24.6 GB. GPU 0 was chosen over the Studio: the Studio was faster (127 ms against
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
| voices | `~/AI/voice-tts/voices.json` (below), clone clips in `~/AI/voice-tts/voices/<name>/` (`reference.wav`, `reference.txt`, `voice.json`) |
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

## Managed by the stack (2026-10-09)

voice-tts is a stack service like the others: a card on **Services**, a **Voice TTS** page in the
sidebar, and a **Voice TTS** section under Configuration. Until 10-09 it was a systemd *user* unit
(`systemctl --user`), which the manager, running as root, could neither see nor start.

- **Unit:** `/etc/systemd/system/voice-tts.service` (runs as the checkout's owner), written by
  `install.sh` when `VOICE_TTS_ENABLED=on`. It runs `scripts/start-voice-tts.sh`, which reads
  `config/llm-stack.env`.
- **Settings** (`VOICE_TTS_*`): enabled, host, port, public URL, GPU (`CUDA_VISIBLE_DEVICES`),
  the venv's python, the voices file, the clone and preset models, language, stream chunk,
  offline weights (`HF_HUB_OFFLINE`), ffmpeg. Blank model/voices keys keep the server's defaults.
  Changing any of them asks for a restart of voice-tts.
- **Voice TTS page:** start/stop/restart, the server's `/health` (models, GPU memory, served,
  capped, cancelled, the last request), the voices and which loaded, a **Set Default** that edits
  `default` in the voices file (keeping `voices.json.bak`; live, no restart), and a spoken test.
- **Health:** the card is green once `/health` answers `ok: true`, after load and CUDA-graph
  warm-up (about a minute). With `VOICE_TTS_ENABLED=off` the start script exits cleanly and a
  stopped card is not a fault.
- **Boot:** started by `llm-stack-restore` when its expectation is on (starting it from the
  manager records that) and `VOICE_TTS_ENABLED=on`.

This host's settings: `VOICE_TTS_ENABLED=on`, `VOICE_TTS_HOST=100.124.56.11`, `VOICE_TTS_PORT=8016`,
`VOICE_TTS_GPU=0`, `VOICE_TTS_PYTHON=/home/ellie/AI/voice-tts/venv/bin/python`.

Moving a host from the old user unit (once, as the owner, then root):

```
systemctl --user disable --now voice-tts.service
rm ~/.config/systemd/user/voice-tts.service && systemctl --user daemon-reload
sudo bash /mnt/LLMs/llamacpp/llm-stack-git/install.sh    # or write the unit by hand, as install_unit does
sudo systemctl daemon-reload && sudo systemctl start voice-tts
```

Check and listen:

```
curl -s http://llms:8016/health
curl -s http://llms:8016/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"input":"Hello there.","response_format":"wav"}' -o /tmp/hello.wav
```

Logs: the Logs button on the page, or `journalctl -u voice-tts -n 20`. There's one `speak {...}` line
per request, with first-audio ms, audio seconds, and whether it was capped or cancelled.

**Changing voices:** set the default from the Voice TTS page (live). From the Studio, the hermes repo's
`scripts/voice_design.py enable <name>... [--default]` or `disable <name>` (restart), or
`default <name>` (live, no restart). enable copies the clip, edits `voices.json` (keeping `voices.json.bak`), restarts the
service once it is idle and checks the voice loaded. If its restart and pause still call the old
`systemctl --user` unit, they need moving to `sudo systemctl ... voice-tts` or the manager's
`POST /api/service/voice-tts/{stop,start,restart}`. By hand: edit `voices.json`, then restart voice-tts from the
manager. Hermes's desktop falls back to Piper while the service is down
(about a minute with both models).

## Voices: presets and clones (2026-10-06)

```
{"default": "sohee",
 "voices": {"sohee": {"speaker": "sohee"},
            "deep":  {"ref_audio": "voices/deep/reference.wav", "ref_text_file": "voices/deep/reference.txt"}}}
```

- A **preset** names a CustomVoice speaker (and may carry an `instruct` style). A **clone** is a
  10–20 s clip plus exactly what it says, spoken by the Base model (`VOICE_TTS_CLONE_MODEL`).
- The server loads only the models its voices need. Both together: ~10 GB in one process. Measured
  10-06 with the clone model beside the running server: GPU 0 at 23.8 of 24.6 GB as two processes, so
  one process (one CUDA context) leaves ~1.2 GB. The other tenants (9B, embeddings, Frigate)
  allocate up front.
- A clone's first audio is ~300 ms, the same as a preset. Each clone's clip is encoded once, at startup.
- Ellie kept `sohee` as the native preset (a sohee clone sounded worse to them) until 2026-10-07, when
  they chose clones only to free ~4.4 GB on GPU 0 (spare went from ~1.2 GB to ~4.8 GB). To bring sohee
  back: add `"sohee": {"speaker": "sohee"}` to `voices.json` and restart; the preset model loads again.
  Styling a preset for a new voice doesn't need it here: the audition loads it for that run.
- A voice whose clip or speaker is missing is skipped and listed under `skipped` in `/health`, so
  one bad voice can't stop the server. If no voice is usable, it refuses to start.
- Making voices (VoiceDesign from a description, cloning a recording, listening pages) is the hermes
  repo's `scripts/voice_design.py` and the Hermes skill `voice-maker`. VoiceDesign runs as a one-off
  process and needs ~5 GB, so the script pauses this service while it renders.

## Not yet

- **Fine-tuning** has to be planned first (roadmap 6e in the hermes repo).
