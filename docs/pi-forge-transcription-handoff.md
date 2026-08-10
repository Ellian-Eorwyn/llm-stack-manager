# Handoff: pointing pi-forge at the llm-stack transcription backend

You are updating **pi-forge** to transcribe audio through the llm-stack
transcription sidecar instead of whatever it uses today. This document is the
whole contract. You do not need access to the llm-stack repository or the host
it runs on — everything here is reachable over the tailnet.

Written 2026-08-10 against llm-stack commit `acbad35`. Every claim in it was
checked against the running service on that date, not against the source.

---

## 1. What you are calling

| | |
|---|---|
| Host | `llms.tailfad058.ts.net` (Tailscale MagicDNS), or `100.124.56.11` |
| Port | `8014` |
| Base URL | `http://100.124.56.11:8014` |
| Auth | **None currently.** See §6 before assuming that stays true. |
| Transport | Plain HTTP over the tailnet. No TLS — the tailnet is the boundary. |

It is a speech-to-text service, nothing else. It does not do TTS, diarization,
or translation-to-arbitrary-languages (English-only translation exists on one
engine, see §5).

Two surfaces:

- **`POST /v1/audio/transcriptions`** — byte-compatible with OpenAI. If pi-forge
  already uses an OpenAI client for audio, change `base_url` and you are done.
- **`POST /transcribe`** — native. Richer response, more parameters, and the
  only one that handles long audio. **Prefer this one for pi-forge.**

---

## 2. The single most important thing

**Audio longer than 900 seconds returns `202 Accepted` with a job id, not a
transcript.**

```json
{"ok": true, "job_id": "55cb265ae1f74840", "status": "queued",
 "poll": "/jobs/55cb265ae1f74840", "estimated_seconds": 211.5}
```

Poll `GET /jobs/<id>` until `status` is `done` or `error`. On `done` the full
result is under `result`. Jobs are discarded 3600s after they finish.

If pi-forge assumes every `/transcribe` call returns a transcript, it will
silently store a job envelope as if it were a transcription and you will not
notice until something reads the text. **Branch on the status code, or on the
presence of `job_id`, before touching `text`.**

`/v1/audio/transcriptions` never goes async — an OpenAI SDK cannot poll — so it
answers `413` for long audio with a message pointing at `/transcribe`. That is
the whole reason to prefer the native route.

A 70-minute file currently completes in about 20 seconds of decode, so polling
every 2-3 seconds is reasonable. Do not poll faster than 1s. Rates vary with
what else is on the GPU — measured between 208x and 240x realtime on the same
file — so treat `estimated_seconds` as a hint, not a deadline.

---

## 3. Making a request

`POST /transcribe` accepts `multipart/form-data` or `application/json`.

```bash
# multipart, the normal path
curl -X POST http://100.124.56.11:8014/transcribe \
  -F file=@meeting.m4a \
  -F response_format=json \
  -F word_timestamps=true
```

```bash
# JSON with inline audio, for callers that already hold bytes
curl -X POST http://100.124.56.11:8014/transcribe \
  -H 'Content-Type: application/json' \
  -d '{"audio_base64": "<base64 or a data: URI>", "filename": "clip.wav"}'
```

Audio is supplied exactly one of three ways: multipart `file`, JSON
`audio_base64`, or `url`. **`url` is refused by default** — the allow-list
(`TRANSCRIPT_URL_ALLOW_HOSTS`) is empty, which denies every fetch, because the
service can reach the whole tailnet. Do not build on `url` without asking the
stack owner to open it.

Upload cap is **512 MB**; over that you get `413`. Anything ffmpeg can decode
works — mp3, m4a, wav, flac, opus, and video containers.

### Parameters worth knowing

| Field | Default | Notes |
|---|---|---|
| `engine` | `parakeet-v3` | See §5. Omit unless you need a specific one. |
| `model` | engine's configured model | `<engine>` or `<engine>:<model-ref>` |
| `response_format` | `json` | `json`, `verbose_json`, `text`, `srt`, `vtt`, `markdown` |
| `word_timestamps` | `false` | **Read §5 before leaving this off.** |
| `language` | auto | ISO code, e.g. `en` |
| `translate` | `false` | To English. Not supported on every engine. |
| `async` | `false` | Force the job path regardless of length |
| `vad` | `true` | Voice-activity filtering |
| `initial_prompt` / `prompt` | — | Biases decoding |
| `hotwords` | — | faster-whisper only |

---

## 4. The response

`response_format=json` (the default) returns:

```json
{
  "ok": true,
  "request_id": "a3f9c1d2",
  "created": 1786700000,
  "filename": "meeting.m4a",
  "text": "The quick brown fox...",
  "language": "en",
  "language_probability": 0.993,
  "duration": 4230.9,
  "segments": [
    {"id": 0, "start": 0.0, "end": 60.0, "text": "...", "speaker": null,
     "avg_logprob": -0.21, "no_speech_prob": 0.004, "compression_ratio": 1.38,
     "words": [{"start": 0.0, "end": 0.22, "word": "The", "probability": 0.98}]}
  ],
  "words": [{"start": 0.0, "end": 0.22, "word": "The", "probability": 0.98}],
  "engine": "parakeet-v3",
  "model": "preset:nvidia/parakeet-tdt-0.6b-v3",
  "device": "cuda",
  "compute_type": "float16",
  "translated": false,
  "degraded": false,
  "capabilities": {"word_timestamps": true, "diarization": false, "translate": false},
  "timings": {"queued_ms": 2, "load_ms": 0, "decode_ms": 20304, "total_ms": 20511,
              "audio_seconds": 4230.9, "realtime_factor": 208.4}
}
```

Notes for whatever you build on it:

- `ok` is always present on the native route. Branch on it rather than on the
  status code if that is easier — both agree.
- `segments[]` is field-for-field OpenAI's `verbose_json` schema plus `speaker`,
  so parsers written against OpenAI work unchanged.
- `words` is a flat top-level list *in addition to* the per-segment `words`.
  It is empty unless `word_timestamps=true`.
- **Store `engine`, `model` and `capabilities` alongside the transcript.** They
  are what makes a stored transcript reproducible later, and they explain a
  missing field instead of leaving you guessing whether it was a bug.
- `degraded: true` means the timeline is synthetic — only the `router` engine
  does this, and only if someone enables it.

`verbose_json` returns OpenAI's exact key set and nothing else. `text` returns
`text/plain`. `srt`/`vtt` return subtitle files. `markdown` is native-only
(`/v1/*` refuses it) and returns a headed document with `**[mm:ss]**` stamps.

### Errors

```json
{"ok": false, "error": {"type": "engine_unavailable",
                        "message": "the nemo runtime ... is not installed",
                        "hint": "bash scripts/install-transcribe.sh --engines nemo"}}
```

Branch on `error.type`, never on the message text:

| type | status | meaning |
|---|---|---|
| `bad_request` | 400 | Bad parameter, unknown engine, no audio supplied |
| `too_large` | 413 | Over the upload cap, or long audio on the OpenAI route |
| `unsupported_capability` | 422 | Asked an engine for something it cannot do |
| `engine_unavailable` | 503 | Runtime not installed on the host. `hint` says how. |
| `model_load_failed` | 503 | Usually CUDA OOM or a misconfigured model |
| `decode_failed` | 500 | The decode itself failed |
| `timeout` / `upstream_error` | 504 / 502 | — |

On `/v1/*` routes you get OpenAI's `{"error": {"message", "type", "code"}}`
instead, with the hint folded into `message`.

**`503` is usually transient-ish and worth surfacing rather than retrying
blindly** — `engine_unavailable` will never fix itself without someone
installing something, and retrying `model_load_failed` in a tight loop makes
GPU pressure worse. Back off, and give up after two attempts.

---

## 5. Engines, and the two traps

`GET /engines` lists what is installed, what is resident, and each engine's
capabilities. Query it rather than hardcoding assumptions.

All five engines are *registered*, but "registered" is not "usable". As of
2026-08-10 every runtime is installed on the host — faster-whisper, NeMo, and
transformers (the last arrived as a NeMo dependency) — so what separates them is
whether a model is configured:

| engine | runtime | state |
|---|---|---|
| `parakeet-v3` | NeMo | **ready** — the default |
| `faster-whisper` | CTranslate2 | **ready** |
| `canary-qwen` | NeMo | ready, but downloads ~2.5 GB on first use |
| `hf-asr` | transformers | runtime present, **no model set** → `model_load_failed` until `HF_ASR_LOCAL_MODEL` is |
| `router` | llama-router | needs `ASR` in `MODEL_ROUTER_MEMBERS`, which it is not |

Naming an engine whose *runtime* is missing returns `engine_unavailable` with
the install command in `hint`; naming one whose *model* is unset returns
`model_load_failed`. Neither takes the service down. Do not treat
`engine_unavailable` as the only "engine not usable" signal — check both, and
prefer asking `GET /engines` up front.

Measured on 70.5 minutes of speech on the host's RTX 3090:

| | faster-whisper large-v3 | **parakeet-v3** (default) |
|---|---|---|
| Realtime factor | 11.6× | **~210×** |
| VRAM | 5,496 MiB | **~1,950 MiB** |
| Segments | 876 | 71 |

**Trap 1 — `word_timestamps` is not optional if you need a timeline.** Whisper
returns real utterance boundaries either way. NeMo returns *no timeline at all*
unless `word_timestamps=true` is requested: without it, Parakeet's "segments"
are just its 60-second decode windows, every subtitle cue spans a whole window,
and `words` is empty. With it you get real segments *and* populated per-segment
`words`, plus the flat top-level `words` list.

So: if pi-forge needs anything time-aligned — subtitles, seeking, speaker turns,
chunking for downstream summarisation — **pass `word_timestamps=true`**. If you
only need the text, leave it off and save the work. Do not infer from
`capabilities.word_timestamps: true` that you will get words without asking.

**Trap 2 — text normalisation differs by engine.** faster-whisper writes
"July 21st, 1969"; Parakeet has written "July twenty first, nineteen sixty nine"
on the same audio. If anything downstream parses dates, figures, or identifiers
out of the transcript, either normalise yourself or pin `engine=faster-whisper`
for those jobs and accept it being ~20× slower.

---

## 6. Things that will change under you

Write pi-forge so these are configuration, not constants:

- **The base URL.** Put it behind one setting. The host may move off
  `100.124.56.11`, and the MagicDNS name is the more stable of the two.
- **Auth.** `TRANSCRIPT_API_TOKEN` is empty today, so no header is needed. It is
  a supported setting and may be turned on. Support sending
  `Authorization: Bearer <token>` from config now, even if it is unset — it is
  three lines today and an outage later.
- **The default engine.** It is `parakeet-v3` now and was `faster-whisper` a day
  ago. If pi-forge needs a specific engine's behaviour, **name it explicitly**
  rather than relying on the default.
- **The async threshold.** 900s today. Handle `202` regardless of what you think
  the threshold is.

Check liveness with `GET /health` → `{"status": "ok"}`. It never requires a
token and never loads a model, so it is safe to poll.

`GET /engines` reports `resident: null` most of the time. **That is normal, not
a fault.** The service loads a model on first request and releases it after 300s
idle, so it does not sit on VRAM the rest of the stack needs. The consequence
for you: the *first* request after an idle period pays a model load — about 25s
for Parakeet. Set client timeouts to at least 120s, and do not treat a slow
first call as a failure.

---

## 7. Suggested shape for the pi-forge side

1. One config block: base URL, optional token, default engine, timeouts.
2. One `transcribe(path_or_bytes, **opts)` entry point that:
   - POSTs multipart to `/transcribe`
   - returns the result on `200`
   - **on `202`, polls `poll` until terminal, then returns `result`**
   - raises a typed error carrying `error.type` and `error.hint` otherwise
3. Store the whole JSON envelope, not just `text`. `engine`, `model`, `duration`
   and `timings` cost nothing to keep and answer most later questions.
4. Default `word_timestamps` to whatever pi-forge actually needs — see Trap 1.

A five-minute smoke test before wiring anything up:

```bash
curl -sf http://100.124.56.11:8014/health
curl -sf http://100.124.56.11:8014/engines | jq '{active_engine, resident, engines: [.engines[].id]}'
curl -sf -X POST http://100.124.56.11:8014/transcribe -F file=@short-clip.mp3 | jq '{ok, engine, text}'
```

If `/health` answers and `/engines` lists `parakeet-v3`, the backend is fine and
anything else is on the pi-forge side.

---

## 8. Where the source lives

Host `LLMs`, repo `/mnt/LLMs/llamacpp/llm-stack-git`, GitHub
`Ellian-Eorwyn/llm-stack-manager`.

- `docs/transcription.md` — fuller reference, VRAM tuning, benchmark method
- `scripts/transcribe-server.py` — the service; the API is all in `create_app`
- `web/config_fields.py` — every setting, under section `Transcription`

You should not need to change anything there. If you find you do, that is a
conversation with the stack owner rather than a patch from pi-forge's side —
the two repos are deployed independently and this API is the seam.
