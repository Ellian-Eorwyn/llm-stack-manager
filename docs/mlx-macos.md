# Apple Silicon MLX profile

Two services that serve from Apple's MLX framework instead of llama.cpp: an
OpenAI-compatible embedding server and an English-only Parakeet v3 transcription
server. Both are selected per slot, so a host runs one or the other and nothing
else changes:

```sh
EMBED_ENGINE=mlx              # llamacpp | mlx
TRANSCRIPT_ENGINE=parakeet-mlx  # sidecar  | parakeet-mlx
```

`install.sh` reads these to choose the launcher, on both the systemd and launchd
paths. Asking for an MLX engine on a host that is not Apple silicon fails the
install rather than quietly serving from llama.cpp.

The reference configuration is an M1 Pro with 16 GiB unified memory running only
the manager, embeddings and transcription, with the llama.cpp auxiliary stack
disabled in the local environment file.

## Installing the runtime

```bash
bash scripts/install-mlx-runtime.sh
```

Creates `deps/mlx-runtime-venv` and fetches the models at the revisions pinned
in `config/mlx-models.lock.json`. It is **not** run with sudo: the venv has to
belong to the user the LaunchAgents run as.

The runtime is kept apart from the manager's own environment on purpose. The
manager depends on nothing but Flask and is kept that way; the MLX tree is mlx,
mlx-embeddings, mlx-audio, fastapi, uvicorn and numpy. Separating them means an
MLX upgrade cannot take the management UI down with it, and the UI stays
installable on a host serving no models at all.

Revisions are pinned rather than tracked: an embedding model that silently
changes revision changes every vector it has ever produced, and nothing
downstream would notice.

## Where the stack directory may live

Not under `~/Documents`, `~/Desktop` or `~/Downloads`. Those are TCC-protected,
and a LaunchAgent has no permission for them: every service fails at exec with

```
bash: .../scripts/launchd-wrapper-embed.sh: Operation not permitted
shell-init: error retrieving current directory: getcwd: cannot access parent
directories: Operation not permitted
```

The files are executable and the same script runs fine from a terminal, because
an interactive shell has been granted access and launchd has not. Nothing about
the permissions, the ownership or the plist is wrong, which is what makes it
cost an afternoon.

`~/Applications/LLMs/llm-stack-manager` is the tested location. Anywhere outside
the protected directories works.

## Readiness

These servers are not llama-server and must not be probed as though they were.
The MLX embedding server has no `/props` and answers 404; both report
`{"status": "healthy"}` where the transcription sidecar reports
`{"status": "ok"}`. `health.ENGINE_PROBES` carries the per-engine definition —
without it the manager reports both as degraded while they serve correctly.

Faking a `/props` payload would be worse than the table: telemetry parses it for
slot and context accounting, and an imitation would produce numbers about a
server that has no slots.

## Endpoints

| Service | URL | Model |
|---|---|---|
| Manager UI | `http://127.0.0.1:8077` | — |
| Embeddings | `http://127.0.0.1:8005/v1/embeddings` | `embed` |
| Transcription | `http://127.0.0.1:8014/v1/audio/transcriptions` | `parakeet-v3-en` |

All listeners are loopback-only. The manager is intentionally unauthenticated,
so do not change its bind address without adding an appropriate access layer.

The embedding endpoint uses the 4-bit DWQ MLX conversion of
Qwen3-Embedding-0.6B and returns normalized 1024-dimensional vectors. It also
accepts OpenAI's optional `dimensions` field for Matryoshka truncation.

Parakeet v3's checkpoint is multilingual. This local endpoint deliberately
accepts only `en`, `en-US`, `en-GB`, or `English` requests and rejects other
language values. It uses bfloat16, 60-second chunks, five-second overlaps, and
one request at a time to keep memory use predictable on a 16 GiB machine.

## Examples

```bash
curl http://127.0.0.1:8005/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"embed","input":["First document","Second document"]}'

curl http://127.0.0.1:8014/v1/audio/transcriptions \
  -F file=@meeting.m4a \
  -F model=parakeet-v3-en \
  -F language=en \
  -F response_format=verbose_json
```

## Speaker diarization

`-F diarize=true` labels who said what, using NVIDIA's Nemotron 3 Diarization
(100M parameters, up to **eight** speakers, overlapping speech detected) through
mlx-audio's MLX port. It runs beside Parakeet in the same process, loads on the
first request that asks for it, and holds about 200 MB.

```bash
curl http://127.0.0.1:8014/v1/audio/transcriptions \
  -F file=@meeting.m4a -F diarize=true -F response_format=verbose_json
```

With `diarize=true`, `verbose_json` changes in these ways:

- `segments[]` are cut wherever the speaker changes as well as at sentence
  ends, and each has `speaker` (`speaker_0`, `speaker_1`, … in order of
  arrival) and `overlap`.
- `words[]` each carry `speaker`, `speaker_score`, `overlap`, `candidates`
  (everyone heard during the word) and, where the smoothing pass filled the
  speaker in, `speaker_inferred: true`.
- `speakers[]` gives each speaker's first appearance, talk time and word count.
  This is what a naming step works from.
- `diarization[]` is the diarizer's own who-spoke-when timeline.

`text` becomes one `speaker_N: …` paragraph per turn, and `srt`/`vtt` prefix
each cue with `[speaker_N]`. `json` is unchanged, because it has nowhere to put
a speaker. `POST /v1/audio/diarize` returns only the timeline, for pairing with
another ASR.

**Labels are anonymous and per recording.** `speaker_0` is whoever spoke first
*in this file*. Attaching real names is a separate step (an LLM reading the
introductions, or voice enrollment), which this service does not do.

**How words get speakers.** The two models' timelines are joined afterwards.
Each word takes the speaker whose mean activity over the word's interval clears
`MLX_DIARIZATION_THRESHOLD`. Two cases are repaired rather than guessed
(`scripts/speaker_attribution.py`):

- Parakeet's word timestamps are emission times, so a turn's first word is
  routinely stamped just before the diarizer hears that speaker, and its last
  word just after. A word that begins a phrase joins the speaker who follows
  it; any other word stays with the speaker before it.
- A word spoken during crosstalk is only ever given to a speaker who was
  actually talking at the time. If neither neighbour was, the word keeps
  `speaker: null` and the `candidates` list.

**Measured on the Studio (M5 Ultra).** NVIDIA's 97 s eight-voice demo clip took
1.5 s for transcription plus diarization, with 2 of 285 words left unassigned,
both in deliberate crosstalk. A 40-minute file took 37 s at about 3 GB
resident. On that clip the model found **6 of the 8 voices**: two voices that
joined later were given the IDs of earlier ones. The demo uses synthetic TTS
voices, so treat that as a known limit rather than a measured rate for real
meetings.

The model is not in an mlx-audio release yet (0.5.5 predates it).
`install-mlx-runtime.sh` pins mlx-audio to the commit that added it; replace the
pin once a release includes `mlx_audio.vad.models.nemotron_diarization`.

## Service management

The services are per-user LaunchAgents in the `gui/<uid>` domain. They start at
login and need no root. The plists are generated by `install.sh` — they are not
checked in, because a plist carries absolute paths and one written for a
particular home directory is not installable anywhere else.

Metal is why they are agents rather than daemons; see `docs/platform-layer.md`.

```bash
launchctl kickstart -k gui/$(id -u)/com.llmstack.embed
launchctl kickstart -k gui/$(id -u)/com.llmstack.transcript-backend
launchctl kickstart -k gui/$(id -u)/com.llmstack.llm-manager
```

Logs are under `logs/` as `embed.*.log`, `transcript-backend.*.log`, and
`llm-manager.*.log`. Local runtime/model paths and performance knobs are in the
ignored `config/llm-stack.env`. Exact model revisions, weight hashes, runtime
versions, and hardware are recorded in `config/mlx-models.lock.json`.
