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

- **Carried over:** the alias, context size (capped at Splash's 256K), the
  reasoning level (as `--default-reasoning-effort`), and the SSD-offload RAM
  budget (as `--max-memory`).
- **Not carried over:** llama.cpp placement, cache types, draft settings (Splash
  picks its own drafter) and `CUSTOM_ARGS_JSON`. Splash's own flags go in
  `LLM_A_SPLASH_ARGS_JSON`.

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
