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

1. Install it (1.2.1 or newer; the Studio runs 1.3.0 since 2026-10-07):

   ```bash
   brew install incoai/tap/splash
   ```

   To upgrade, stop the slot first (the server and engine must be the same
   version), then `brew update && brew upgrade incoai/tap/splash`, then start
   it again.

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
  Splash's SSD tier for evicted conversations is off unless
  `LLM_A_SPLASH_MAX_CACHE_DISK_GB` sizes it. See [Memory](#memory).
- **Vision** (images and PDFs) is on by default; Splash downloads the repo's
  `mmproj` (~0.9 GB) on first start. `LLM_A_SPLASH_VISION=off` serves text
  only and saves that memory.
- **Memory stays locked** in RAM for `LLM_A_SPLASH_IDLE_RELEASE` after the last
  request (`--idle-release`; blank is 20m, `off` never releases). See
  [Stalls](#stalls).
- **Neural Engine prefill is off** (`--disable-ane`) unless
  `LLM_A_SPLASH_ANE=on`. Splash 1.3 can run part of a long prompt's FFN on the
  Neural Engine in W8A8, which is not lossless. On the Studio it calibrated a
  15% share and made a 103K-token prefill 3.5% faster (85 s against 88).
  Smaller Macs gain more: Splash measured 1.3x on an M5 Pro and 1.8x on an
  M4 Max.
- **Not carried over:** llama.cpp placement, cache types, draft settings (Splash
  picks its own drafter) and `CUSTOM_ARGS_JSON`. Splash's own flags go in
  `LLM_A_SPLASH_ARGS_JSON`.

## Sampling and repetition

Splash 1.1.0 sampled with temperature, top_p and top_k only, and refused with
HTTP 400 ("the requested logits or output transformation is not supported")
any request whose `presence_penalty`, `frequency_penalty` or `min_p` was not 0.
Since 1.2.0 it accepts `presence_penalty`, `frequency_penalty`,
`repetition_penalty` and `min_p` (checked on 1.3.0, 2026-10-07); only a
non-empty `logit_bias` is still refused. The proxy still sends 0 for all of
them, so the usual Qwen cure for endless repetition (`*_PRESENCE_PENALTY=1.5`)
is now available but untried here.

What is available is the model card's own sampling, which the proxy sets per
persona: thinking at `temperature=1.0, top_p=0.95, top_k=20`
(`THINK_TEMP`, `CODE_TEMP`) and instruct at `temperature=0.7, top_p=0.80`
(`NOTHINK_*`). On 2026-10-01 the Studio still ran `think` at 0.7 and `code` at
0.6, and the uncensored Q8_0 had six thinking loops in about 840 requests:
replies that ran to `THINK_MAX_TOKENS` (16,384) at 120-170 tok/s, far above
the usual 65-90, because the drafter accepts almost every repeated token. The
stock Q8_0 had one in about 560. Both temperatures went to 1.0 that morning.

Speed tells a loop from long work. The same afternoon an xhigh turn (cleaning
an 11k-token meeting transcript) hit the 16,384 cap five times at a normal
65-88 tok/s, and its thinking had no repeated lines: real work cut short, not a
loop. So `THINK_MAX_TOKENS` went to 0 like `CODE_`/`NOTHINK_`: the proxy sends
no limit of its own. Hermes sends `max_tokens: 16384` itself
(`~/.hermes/config.yaml`). A client that sends none got 32,768 from Splash
1.1; since 1.2 it may use all the context the prompt leaves. Don't set a fixed
cap above about 60K instead: on `/v1/chat/completions` Splash refuses (400
`context_length_exceeded`) any request whose prompt plus `max_tokens` passes
the 256K context, and Hermes only compacts at 200K.

To list long replies and their speed (a loop runs at 120+ tok/s):

```bash
grep -E 'output (16,384|32,768)' logs/llm-a.stdout.log
```

### The loop guard

Without a cap, a true loop would run to the client's limit (Hermes: 16,384) or,
with none, to the end of the context, many minutes of the only
slot. So the proxy watches for loops itself (`LOOP_GUARD=on`, on the Studio
since 2026-10-01; `LOOP_GUARD_*` in `config/llm-stack.env.example`). It sees
every chat reply in full, reasoning included, although `reasoning_stream=hidden`
keeps the reasoning out of the content clients read.

- **What counts as a loop.** Reasoning and content are watched separately.
  Every 256 new characters the newest 128 (~32 tokens) are looked for earlier
  in the last 32,000; each earlier copy proposes a period. It is a loop when
  the last 6 blocks of that period are identical and cover at least 1,500
  characters. Back to back and exact is the point: long code, tables and
  reasoning that re-quotes one source line repeat fragments, with different
  text between them.
- **Tuned on Hermes's `state.db`** (2026-10-01). The 09-29 loop (32,768 tokens
  at 153 tok/s; a 304-character paragraph) trips 1,950 characters (~490
  tokens, 3 s) after the exact repetition begins, a seventh of the way into the
  reply. None of the other 8,252 stored replies trips, including session
  `20261001_134609_56722c` (long xhigh thinking), and neither does any of
  ~10,000 tool results, which are dense repetitive JSON. Counting how often a window recurs, the other obvious rule,
  scored legitimate reasoning that re-quotes a transcript line at 7 and JSON at
  48, so it was not used. `tests/test_loop_guard.py` re-runs these checks
  whenever `state.db` is on the machine.
- **On a loop** the proxy closes the backend connection (Splash logs the
  request `Cancelled`) and writes one line to `logs/llm-a-proxy.stdout.log`:

  ```
  loop-guard port=think field=reasoning period=304 repeats=6 after=18,944 chars attempt=1/2: cancelled; retrying at temperature 1.1
  ```

  It retries once (`LOOP_GUARD_RETRIES`) at temperature + 0.1, and with the
  client's seed + 1 if it sent one; otherwise Splash draws a fresh seed for
  every request anyway. The client's stream carries on: the reasoning gets a
  `[llm-chat-proxy: …restarted]` marker and then the new attempt, under the
  same chat id. Hermes doesn't send reasoning back to this endpoint, so the
  cut loop doesn't seed later turns.
- **No second retry.** If the retry loops too, or the loop is in content the
  client has already received, the reply ends with `finish_reason: "length"`.
  When the loop was in the reasoning, the looping text becomes the content.
  That is what the model produced, and it is what makes Hermes stop with
  "Response Stopped — Repetition Detected" instead of asking the model to
  continue (`agent/repetition_guard.py`).
- **Non-streamed requests** (scripts, the llms fallback) are streamed from the
  backend all the same (`LOOP_GUARD_NONSTREAM=on`), and the client gets the one
  `chat.completion` JSON it asked for, built from the final attempt: content,
  reasoning, tool calls, usage and timings. A request with `n` > 1 or log
  probabilities is relayed as before, unguarded.
- **With `PROXY_STREAM_PASSTHROUGH=on`** (llms) streams are still watched, but
  reach the client as the backend sent them; only the guard's marker and its
  ending are added, and a retry keeps the backend's second chat id.
- **Not watched:** `/v1/responses`, `/v1/completions` and tool-call arguments.
  Repeats that change between copies (numbered, or a counter) aren't caught
  either. A reply that was *asked* to repeat a long paragraph back to back
  would be cut, but when asked, the 27B numbers its copies.

Tried live on 2026-10-01 through a scratch proxy: asked to write a paragraph
12 times, the reply was cut at the seventh copy and Splash logged
`Cancelled · output 376 · 166.7 tok/s`. The retry and both endings are tested
against a fake streaming backend in `tests/test_loop_guard.py`.

To see what the guard has done:

```bash
grep loop-guard logs/llm-a-proxy.stdout.log
```

## Memory

Left to itself (`--max-memory auto`), Splash may use Metal's recommended
working set: 85% of RAM, 78 GiB on the 96 GB Studio. It keeps the KV cache and
recurrent state of every finished conversation, and evicts only when that limit
is reached. A benchmark run never gets there. A day of Hermes sessions does:
70 kept conversations held 35 GiB of KV and 13 GiB of state on top of the
weights, 80 GiB in all. With macOS and the other services on top, the Mac ran
out of swap and froze.

The stack therefore passes a cap: half of RAM, 48 GiB on the Studio. The 27B
Q8_0 plan is about 33 GiB of weights, drafter, vision and buffers. That leaves
about 15 GiB for KV and state: one bf16 conversation of about 230K tokens.
Past that, Splash evicts the oldest instead of growing. `auto` restores
Splash's own limit.

Everything Splash allocates (weights, KV, state) stays locked into RAM until
the idle release (see [Stalls](#stalls)), and locked memory cannot be
compressed or swapped. At 48 GiB, 55 GiB of the Studio's 96 was wired. The Studio's config sets 56
(`LLM_A_SPLASH_MAX_MEMORY_GB`), for about 64 GiB wired and ~410K tokens of
conversations. The default stays at half of RAM for smaller Macs.

### The SSD tier

`--max-cache-disk` moves evicted conversations to the SSD instead of dropping
them. It is off unless `LLM_A_SPLASH_MAX_CACHE_DISK_GB` sizes it, because it
did not pay for its writes under an agent. With 40 GiB of tier and a 48 GiB cap
on 2026-09-30 (2.7M prompt tokens, mostly stress tests):

- It wrote 337 GB in 2 h 40 min, about 121 KiB per prompt token: 153 GB of KV
  pages and ~180 GB of 187 MiB recurrent-state snapshots.
- It read back 49 GB, 15%, and served 57K tokens of KV from disk.
- A compacted conversation is never resumed, yet each compaction wrote its
  ~12 GiB of KV to the tier.

A normal heavy day (~800K prompt tokens) would write ~100 GB. That is small
against an SSD's rated endurance, but it bought little. The tier writes its
files as `$TMPDIR/splash-cache-*`.

Splash reports its plan and live use on `/status` (`memory_plan.budget`,
`memory_actual`) and `/metrics` (`splash_memory_current_bytes`,
`splash_state_evictions_total`). Since 1.2 the weights are ordinary Metal
buffers rather than mapped cache files, and the manager's memory panel and
`footprint` agree (31 GB just after start on 1.3.0). 1.1's prepared-weight
cache, `~/Library/Caches/Splash/weights` (55 GB on the Studio), is no longer
read and can be deleted.

## Stalls

With Splash serving an agent, the Mac froze for 1–30 s after nearly every
turn. The pointer and window dragging kept working, but apps could not
navigate, Chrome could not switch tabs, and windows sometimes went blank. It
was worst when one conversation grew toward 200K tokens and was compacted
again and again.

The cause, measured on 2026-09-30 with a probe that times what apps ask the
system for:

- Splash locks its whole KV cache into RAM (Metal residency) while it serves a
  request: wired memory rose from 35 to 54 GiB at the 48 GiB cap then in use.
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
  watchdog logged users out. Its proposed patch covers KV unmapping. Here
  Splash reported no unmaps on 09-30, but by 10-07, after six days up, 1.1
  had done 564, the longest taking 1.4 s.

Splash 1.1 had no setting for how long it kept the cache locked. So the chat
proxy kept it locked (commit f03c5ed): until 20 minutes after the last
request, it sent Splash a one-token request every 0.7 s. No app waited more
than 9 ms, but the pings cost ~8% of the GPU and showed up in Splash's request
counts.

### Since 1.2

Splash 1.2 rebuilt both paths. Every buffer joins one residency set that stays
wired between requests until `--idle-release` passes without one. KV is held
in ordinary shared buffers, so nothing is mapped or unmapped while it serves.
The stack passes `LLM_A_SPLASH_IDLE_RELEASE`, 20m unless set, and the proxy no
longer pings. Measured on 1.3.0 on 2026-10-07 with the same IOSurface probe and
a 2-minute release:

- A 103K-token prompt, then 2 minutes idle: wired memory held at 46 GiB
  throughout, and no new surface took more than 1.1 ms.
- The release itself unlocked 37 GiB at once (46 to 9 GiB wired), and no
  surface took more than 0.6 ms. That is the event that froze apps for
  1–6 s on 1.1.
- The next request re-locked it without a stall (0.7 ms), restored the
  weights in 3.7 s, and found the 103K-token conversation still cached: the
  whole turn took 3.9 s.

So the release costs a few seconds on the next request and nothing else.
`off` keeps the memory locked for as long as Splash runs.

## Thinking

Splash 1.1 ignored `chat_template_kwargs.enable_thinking`, which llama.cpp and
MTPLX honour, and turned thinking off only with `reasoning_effort: "none"`.
Since 1.2 it reads `enable_thinking` too (a boolean overrides the effort), but
`"none"` still works and the proxy keeps sending it.

The chat proxy knows which engine is behind it (`CHAT_BACKEND_ENGINE`, from the
slot registry). On endpoints with thinking off, it sends `"none"` to Splash
only. The Qwen template raises on that value, so the other engines never see it.

A client that calls Splash directly has to send `reasoning_effort` itself, or
rely on the slot's default.

## Known limits

- **Young software.** It was released 2026-09-18; 1.3.0 came out 2026-10-06.
- **Output limit.** Since 1.2 a response may run to the end of the context
  (1.1 stopped at 32k, issue #221). `--max-new-tokens` is gone.
- **Monitoring names changed in 1.2** (status schema 6): `splash_ttft_seconds`
  is `splash_http_ttft_seconds`, and `/status` `latency.ttft` is
  `latency.http_ttft`. Nothing in the stack read them.
- **Mac sleep.** Since 1.2.1 Splash keeps the Mac awake while a request runs
  (the display may still sleep), and a request survives a sleep.
- **Concurrent load.** A crash was reported under concurrent GPU load (#220).
  It did not reproduce here.
- **Health checks.** `/health` answers before the model has loaded, so the
  stack's health checks use `/ready`.
