# Transcription

A speech-to-text sidecar on port **8014**, unit `transcript-backend`. Agents on
the tailnet POST an audio file and parse structured JSON back.

It exists as a separate Python service rather than another llama-server because
the models worth running for ASR are not llama.cpp models. Parakeet and
Canary-Qwen are NeMo checkpoints; Whisper large-v3 in its fast form is a
CTranslate2 export. llama.cpp can only transcribe with an audio *LLM* — which it
does well, and which this service can also reach — but it cannot load any of the
three models above.

## Engines

| id | runtime | models | installed by |
|---|---|---|---|
| `faster-whisper` | CTranslate2 | `large-v3`, `turbo`, `distil-large-v3`, every Whisper size | default |
| `parakeet-v3` | NeMo | `nvidia/parakeet-tdt-0.6b-v3` | `--engines nemo` |
| `canary-qwen` | NeMo | `nvidia/canary-qwen-2.5b` | `--engines nemo` |
| `hf-asr` | transformers | any HF ASR repo — the "download and test" slot | `--engines hf` |
| `router` | llama-router | an audio-capable GGUF pooled with embed/ocr/rank/task | always |

```bash
bash scripts/install-transcribe.sh                  # faster-whisper only
bash scripts/install-transcribe.sh --engines nemo   # + Parakeet and Canary
bash scripts/install-transcribe.sh --engines nemo,hf
```

Or from the manager: the Transcription page has an **Install / Update** button
that runs the same script as a polled job, because a torch install takes minutes
and no HTTP request should sit through it.

### Python version

`torch` publishes wheels for **3.9–3.13**, and faster-whisper does not need
torch at all. A host whose `python3` is newer than that — 3.14 here — runs
Whisper perfectly and then fails to install NeMo with nothing more useful than
`No matching distribution found for torch`.

The installer therefore picks its own interpreter when `nemo` or `hf` are
requested, preferring `python3.12`. If a venv already exists on an interpreter
torch cannot use, it says so and stops rather than failing three minutes into a
pip run; `--recreate` rebuilds it on a compatible one. `TRANSCRIPT_PYTHON`
overrides the choice.

Because the engines share one venv, adding NeMo to a 3.14 faster-whisper install
means rebuilding both:

```bash
bash scripts/install-transcribe.sh --engines faster-whisper,nemo --recreate
```

torch comes from PyPI, whose wheels bundle their own CUDA runtime and work with
any recent driver. Set `TRANSCRIPT_TORCH_INDEX_URL` only to force a specific
build — hardcoding a `/whl/cuXXX` index silently pins you to older and older
torch as it ages.

Engines import lazily. A runtime that is not installed costs nothing at startup
and answers its first request with `503 engine_unavailable` naming the flag that
would install it — it cannot take the service down.

Each engine has its own model slot (`FASTER_WHISPER_LOCAL_MODEL`,
`PARAKEET_V3_LOCAL_MODEL`, …) taking `preset:<name>` or `local:<absolute-path>`,
and its own folder under `models/transcription/<engine-id>/` that the config
page can download into from Hugging Face.

## Sharing VRAM with the pooled models

This is the constraint the design is built around. The auxiliary models are
pooled by `llama-router` on GPUs that are usually near full, so a transcription
model that sits resident is the thing that breaks everything else.

- **One model is resident at a time.** Asking for a different engine or model
  releases the current one first.
- **Nothing is loaded until the first request.** Starting the unit costs a
  socket. `GET /engines` reports `"resident": null`, and that is the normal
  steady state, not a fault.
- **Idle models are released.** `TRANSCRIPT_IDLE_UNLOAD_SECONDS` (default 300)
  drops the weights and calls `torch.cuda.empty_cache()` where torch exists. A
  running decode is never evicted out from under itself.
- **The router is asked to yield first.** `TRANSCRIPT_ROUTER_YIELD=asr`
  (default) POSTs the router's `/models/unload` before a local model loads, so
  the two never stack. `all` evicts everything the router holds; `off` disables
  it. Yielding is best-effort: an unreachable router logs a warning and the
  transcription proceeds.

`POST /unload` frees the model immediately — the panel's **Free VRAM** button.

## Measured on this host

70.5 minutes of lecture audio (64 kbps mono MP3), one RTX 3090. Peak VRAM is
sampled from `nvidia-smi` for the whole process, so it includes the CUDA
context and the allocator's cached pool, not just live tensors.

| | faster-whisper large-v3 | parakeet-tdt-0.6b-v3 |
|---|---|---|
| Wall clock | 371 s | **47 s** |
| Realtime factor | 11.6× | **192×** |
| Peak VRAM | 5,496 MiB | **1,909 MiB** |
| Segments | **876** | 71 |
| Transcript | 56,418 chars | 55,477 chars |

Parakeet is **8× faster on a fifth of the VRAM** for a transcript of
near-identical length. Two caveats the table does not carry:

**Segment granularity is not comparable.** Whisper returns real utterance
boundaries. NeMo returns none unless `word_timestamps=true` is requested, so
Parakeet's segments are just its decode windows and their timings are
window-granular. Ask for word timestamps if you need subtitles or alignment.

**Whisper applies inverse text normalisation more consistently** — "July 21st,
1969" against Parakeet's "July twenty first, nineteen sixty nine" on one clip.
That matters if agents parse dates or figures out of the transcript.

### Getting Parakeet into 2 GB

The 1,909 MiB figure needs three settings together, and each was worth roughly
a factor on its own:

```
TRANSCRIPT_LOCAL_COMPUTE_TYPE=float16
TRANSCRIPT_NEMO_CHUNK_SECONDS=60
TRANSCRIPT_MAX_VRAM_MB=2500
```

Left at fp32 with 300-second windows the same file peaks at 6,490 MiB. The
precision setting is the big one: NeMo restores checkpoints in fp32 onto the
GPU by default, so the engine restores on the **CPU**, casts, and only then
moves — casting after the move still pays a 2.4 GB fp32 peak for a 0.6B model,
which is enough to make a 2.5 GB budget unloadable.

Window size trades VRAM against nothing much below ~100 s: 60 s and 100 s
windows land within 20 MiB of each other, while 150 s costs a further gigabyte.
Shorter windows also mean finer segment timings when word timestamps are off.

**`TRANSCRIPT_MAX_VRAM_MB` only binds torch.** It works by capping torch's
allocator, so it constrains the `nemo` and `hf` engines and is inert for
`faster-whisper` — CTranslate2 allocates outside torch entirely and was
measured at 2,532 MiB under a 2,500 MiB budget. Treat it as a guard for the
torch engines, not a guarantee for the process.

## The `router` engine

`MODEL_ROUTER_MEMBERS=EMBED,OCR,RERANK,TASK,ASR` adds an audio GGUF to the
router's pool, where it is loaded and evicted by exactly the same LRU as the
others. Point `ASR_MODEL_PATH` and `ASR_MMPROJ_PATH` at an audio-capable model
(Voxtral, Qwen3-Audio, Granite-Speech).

`ASR_MMPROJ_PATH` is not optional: llama.cpp refuses transcription unless the
model carries an audio projector. If the router answers *"The current model does
not support audio input"*, that is what is missing — check the projector against
the encoders in `deps/llama.cpp/tools/mtmd/` (`whisper-enc.cpp`, `conformer.cpp`,
`qwen3a.cpp`, `granite-speech.cpp`) before looking at the sidecar.

**This engine returns no timeline.** llama.cpp runs the audio LLM through its
chat path, so the reply is prose: no segments, no word timings, and
`response_format` may only ever be `json` (`server-chat.cpp` rejects the rest).
Asking it for `srt`, `vtt` or `verbose_json` therefore returns `422
unsupported_capability` rather than a single whole-file cue dressed up as a real
timeline — a subtitle file that loads in a player and is wrong everywhere is
worse than an error. Set `TRANSCRIPT_ROUTER_ALLOW_DEGRADED=on` to accept the
single cue, which is then marked `"degraded": true`.

## API

`POST /v1/audio/transcriptions` is byte-compatible with OpenAI, so an existing
SDK works by changing `base_url` alone. `model` accepts `whisper-1` (mapped to
your default engine), an engine id, or `<engine>:<model-ref>`.

```bash
curl -F file=@meeting.m4a -F response_format=verbose_json \
     http://100.124.56.11:8014/v1/audio/transcriptions
```

`POST /transcribe` is the native route: multipart `file`, or JSON
`audio_base64`, or `url` (denied unless the host is in
`TRANSCRIPT_URL_ALLOW_HOSTS`, which is blank by default). It adds `engine`,
`translate`, `word_timestamps`, `vad`, `beam_size`, `hotwords`, `async`, and
`response_format=markdown`.

```json
{"ok": true, "request_id": "a3f9c1d2", "text": "…", "language": "en",
 "language_probability": 0.993, "duration": 3.04,
 "segments": [{"id": 0, "start": 0.0, "end": 3.04, "text": "…", "speaker": null,
               "avg_logprob": -0.21, "no_speech_prob": 0.004,
               "words": [{"start": 0.0, "end": 0.22, "word": "The", "probability": 0.98}]}],
 "words": [], "engine": "faster-whisper", "model": "preset:large-v3",
 "device": "cuda", "compute_type": "float16",
 "capabilities": {"word_timestamps": true, "diarization": false, "translate": true},
 "timings": {"decode_ms": 214, "total_ms": 218, "audio_seconds": 3.04,
             "realtime_factor": 13.9}}
```

`segments[]` is field-for-field OpenAI's `verbose_json`, so an agent written
against OpenAI reads it unchanged. `engine`, `model` and `capabilities` are
there so a stored transcript stays reproducible and a missing field is explained
rather than silently null.

Formats: `json` · `verbose_json` · `text` · `srt` · `vtt` · `markdown`
(native-only — `/v1/*` refuses it, because a compatible endpoint that accepts
non-standard values is worse than one that does not).

Errors are `{"ok": false, "error": {"type", "message", "hint"}}` with types
`bad_request`, `too_large`, `unsupported_capability`, `engine_unavailable`,
`model_load_failed`, `decode_failed`, `upstream_error`. On `/v1/*` routes the
body is OpenAI's shape instead, with the hint folded into `message` — that shape
has nowhere else to put it and SDKs surface nothing else.

### Long audio

Over `TRANSCRIPT_ASYNC_THRESHOLD_SECONDS` (default 900), `/transcribe` returns
`202` with a job id:

```json
{"ok": true, "job_id": "9f2c…", "status": "queued", "poll": "/jobs/9f2c…"}
```

Poll `GET /jobs/<id>` until `status` is `done` or `error`. `/v1/audio/transcriptions`
never goes async — an SDK cannot poll — and returns `413` pointing here instead.

### Other routes

`GET /health` · `GET /engines` (installed engines, resident model, idle
countdown, router reachability) · `GET /v1/models` · `POST /unload` ·
`GET|DELETE /jobs[/<id>]` · `POST /v1/audio/translations`.

## Access

Bound to `TRANSCRIPT_HOST` (loopback by default; set a Tailscale address to
reach it from other machines) on `TRANSCRIPT_PORT`. `TRANSCRIPT_API_TOKEN` is
blank by default, meaning no authentication; set it and requests need
`Authorization: Bearer <token>`. `/health` never requires one, or the health
probe would report a working service as down.

## Testing without installing anything

The sidecar runs standalone, as your own user, with no systemd and no GPU:

```bash
bash scripts/install-transcribe.sh --engines faster-whisper

TRANSCRIPT_ENABLED=on TRANSCRIPT_HOST=127.0.0.1 TRANSCRIPT_PORT=8114 \
TRANSCRIPT_ACTIVE_ENGINE=faster-whisper \
FASTER_WHISPER_LOCAL_MODEL=preset:tiny.en \
TRANSCRIPT_LOCAL_DEVICE=cpu TRANSCRIPT_LOCAL_COMPUTE_TYPE=int8 \
TRANSCRIPT_ROUTER_YIELD=off \
bash scripts/start-transcribe.sh

curl -F file=@deps/llama.cpp/tools/mtmd/test-2.mp3 \
     -F response_format=verbose_json \
     http://127.0.0.1:8114/v1/audio/transcriptions
```

`scripts/manage-transcript-service.sh {start|stop|restart|status}` does the same
with a pidfile and a log under `logs/transcript/`, and is what the manager falls
back to on a host where no unit is installed.
