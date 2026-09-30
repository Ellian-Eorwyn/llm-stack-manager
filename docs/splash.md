# Splash: Qwen3.8-27B at 8-bit, fast at long context

[Splash](https://github.com/incoai/splash) is an Apple-silicon server built for
two models only: Qwen3.8-27B and Qwen3.6-35B-A3B. It uses hand-written Metal
kernels and a DFlash2 drafter it pairs with the model automatically. Set a chat
slot's engine to `splash` and it serves that slot.

## Measured on the Studio (M5 Ultra, 96 GB), 2026-09-29

Both columns run Qwen3.8-27B at 8-bit, with thinking off and the same prompts
(`scripts/bench-offload.py`).

| Decode, chat · code tok/s | MTPLX Optimized-Quality, 20 s idle window | Splash, `unsloth/Qwen3.8-27B-GGUF:Q8_0`, bf16 KV |
|---|---|---|
| Short prompt | 61 · 143 | 70 · **205** |
| 128k context | 40 · 73 | **56 · 148** |
| 200k context | 38 · 59 | **51 · 133** |
| Memory at 200k | ~64 GiB | **~48 GiB** |

- **Same answers.** On the same questions (GSM8K 100, Banking77 100,
  FaithEval 120), the two engines gave the same outcome on 99%, 99% and 97% of
  items.
- **Scores.** Splash scored 95 / 82 / 70; MTPLX scored 94 / 83 / 72.
- **Stable under load.** Six minutes of four concurrent streams, beside a
  second process saturating the GPU, produced no crash, no errors and no empty
  replies.
- **One request at a time.** Splash serves requests serially, so parallel
  agents queue.

## Setting it up

1. Install it:

   ```bash
   brew install incoai/tap/splash
   ```

2. Configure the slot:
   - `LLM_A_ENGINE=splash`
   - `LLM_A_MODEL_PATH=unsloth/Qwen3.8-27B-GGUF:Q8_0`. This is a Hugging Face
     reference, not a file path. It downloads into the ordinary Hugging Face
     cache on first start.
   - `LLM_A_SPLASH_REVISION`: optional, pins that repo's commit.
   - `LLM_A_SPLASH_KV_FORMAT=bf16`: the default here. Splash's own default,
     int8, is not lossless.
3. Restart the slot and its proxy.

## How settings carry over

- **Carried over:** the alias, context size (capped at Splash's 256K) and the
  reasoning level (as `--default-reasoning-effort`).
- **Memory is always capped** with `--max-memory`: `LLM_A_SPLASH_MAX_MEMORY_GB`
  if set, else the SSD-offload RAM budget, else half of RAM (48 GiB on 96 GB).
  Conversations evicted from memory go to the SSD (`--max-cache-disk`,
  `LLM_A_SPLASH_MAX_CACHE_DISK_GB`, 40 GiB unless set, `0` for none). See
  [Memory](#memory).
- **Vision** (images and PDFs) is on by default; Splash downloads the repo's
  `mmproj` (~0.9 GB) on first start. `LLM_A_SPLASH_VISION=off` serves text
  only and saves that memory.
- **Not carried over:** llama.cpp placement, cache types, draft settings (Splash
  picks its own drafter) and `CUSTOM_ARGS_JSON`. Splash's own flags go in
  `LLM_A_SPLASH_ARGS_JSON`.

## Memory

Left to itself (`--max-memory auto`), Splash may use Metal's recommended
working set: 85% of RAM, 78 GiB on the 96 GB Studio. It keeps the KV cache and
recurrent state of every finished conversation, and evicts only when that limit
is reached. A benchmark run never gets there. A day of Hermes sessions does:
70 kept conversations held 35 GiB of KV and 13 GiB of state on top of the
weights, 80 GiB in all. With macOS and the other services on top, the Mac ran
out of swap and froze.

The stack therefore passes a cap: half of RAM, 48 GiB on the Studio. The 27B
Q8_0 plan is about 33 GiB of weights, drafter, vision and buffers, which leaves
about 15 GiB for KV and state: one bf16 conversation of about 230K tokens.
Past that, Splash evicts the oldest instead of growing. `auto` restores
Splash's own limit.

The cap was 60 GiB until 2026-09-30. Every page of KV Splash keeps is locked
into RAM while it serves a request, so a smaller cap also means less memory
changing hands at each request; see [Stalls](#stalls).

Evicted conversations go to the SSD (`--max-cache-disk`, 40 GiB by default).
Resuming an older Hermes session then reads its KV back from disk instead of
processing its whole prompt again.

Splash reports its plan and live use on `/status` (`memory_plan.budget`,
`memory_actual`) and `/metrics` (`splash_memory_current_bytes`,
`splash_state_evictions_total`). The manager's memory panel counts the
KV pages too; `footprint` does not.

## Stalls

With Splash serving an agent, the Mac froze for 1–30 s after nearly every
turn. The pointer and window dragging kept working, but apps could not
navigate, Chrome could not switch tabs, and windows sometimes went blank. It
was worst when one conversation grew toward 200K tokens and was compacted
again and again.

The cause, measured on 2026-09-30 with a probe that times what apps ask the
system for:

- Splash locks its whole KV cache into RAM (Metal residency) while it serves a
  request: wired memory rises from 35 to 54 GiB at 48 GiB of cap.
- About a second after the last request ends, it gives the cache back. While
  the kernel unlocks those ~17 GiB, creating an IOSurface blocks. That is the
  call every app makes for a new window buffer, tab or view. Stacks taken
  during a stall show the call waiting in `IOSurfaceClientCreateChild` inside
  the kernel, with Splash's own threads idle.
- In a replay of a conversation growing from 105K to 195K tokens, then
  compacted, this stalled new surfaces for 1.3–6.1 s, seven times in six
  minutes. Each stall began 2–3 s after a turn ended, and they grew with the
  conversation. GPU time, drawing into existing surfaces, WindowServer
  round trips, page faults and disk writes stayed under 40 ms throughout.
- Splash issue [#220](https://github.com/incoai/splash/issues/220) reports the
  same family of stall at >180K context, severe enough that WindowServer's
  watchdog logged users out. Its proposed patch covers KV unmapping; here
  Splash reports no unmaps, only the residency release.

Splash has no setting for how long it keeps the cache locked. So the chat
proxy keeps it locked for Splash: from each generation request until
`SPLASH_KEEP_RESIDENT_MIN` minutes (20 by default) after the last one, it
sends Splash a one-token request every 0.7 s, skipping while a request runs.
Replaying the same growing conversation with it on, no app waited more than
9 ms, and turns took as long as before.

What it costs:

- Each ping is about 60 ms of GPU, roughly 8% of the GPU while the window is
  open.
- 54 GiB stays wired for those 20 minutes instead of 35.
- The pings show up in Splash's request counts and its TTFT percentiles.
- When the window closes, Splash gives the cache back once, and that can
  stall apps once.

`0` turns it off. `always` is not offered: the GPU would never idle.

## Thinking

Splash ignores `chat_template_kwargs.enable_thinking`, which llama.cpp and MTPLX
honour. It turns thinking off only with `reasoning_effort: "none"`.

The chat proxy knows which engine is behind it (`CHAT_BACKEND_ENGINE`, from the
slot registry). On endpoints with thinking off, it sends `"none"` to Splash
only. The Qwen template raises on that value, so the other engines never see it.

A client that calls Splash directly has to send `reasoning_effort` itself, or
rely on the slot's default.

## Known limits

- **Young software.** It was released 2026-09-18 (1.1.0).
- **Output limit.** A single response is limited to 32k tokens (Splash issue
  #221).
- **Concurrent load.** A crash was reported under concurrent GPU load (#220).
  It did not reproduce here.
- **Health checks.** `/health` answers before the model has loaded, so the
  stack's health checks use `/ready`.
