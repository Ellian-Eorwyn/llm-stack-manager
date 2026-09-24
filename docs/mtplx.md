# MTPLX: a chat slot on Apple silicon without llama.cpp

Point a chat slot's model at an MTPLX pack and it is served by
[MTPLX](https://github.com/youssofal/mtplx) instead of llama-server; point it
back at a GGUF and llama-server serves it again. MTPLX is
an MLX server that drafts from the model's own multi-token-prediction head and
verifies the draft in one batched pass. llama.cpp's `draft-mtp` does the same
thing; MTPLX does it in a runtime written for Apple silicon, and on the Studio
that is the difference below.

Linux is untouched: its models are GGUFs, and the engine refuses to build a
command on anything but macOS.

## Measured on the Studio (M5 Ultra, 96 GB)

Qwen3.8 27B both ways, same prompts, same sampling (temp 1.0, top-p 0.95,
top-k 20), one request at a time, thinking on. Not quite the same weights: the
llama.cpp columns are an abliterated fine-tune at Q6_K (6.6 bits/weight), the
MTPLX columns are stock Qwen at 5.8 and 8. Draft acceptance, and so speed, depends on
how well the MTP head matches the trunk it drafts for.

| | llama.cpp `7e4c0a9`, Q6_K, `draft-mtp` n-max 3 | llama.cpp master (Sep 24), same | MTPLX 2.12, Optimized Speed | MTPLX 2.12, Optimized Quality |
|---|---|---|---|---|
| Code, 1024 tokens | 53.4 tok/s | 53.6 tok/s | **84.6 tok/s** | 73.0 tok/s |
| Prose, 768 tokens | 44.6 tok/s | 44.3 tok/s | **76.0 tok/s** | 64.2 tok/s |
| Prefill, 15k-token prompt | 885 tok/s | 944 tok/s | 1,524 tok/s | **1,528 tok/s** |
| Decode at ~100k context | 23.3 tok/s | 30.4 tok/s | **67.9 tok/s** | 61.5 tok/s |
| First token, ~100k prompt | 251 s | 191 s | **80 s** | 83 s |

The two packs trade 9-16% of speed for fidelity. Measured by their publisher
against the bf16 original: Optimized Speed (5.8 bits/weight, 25 GiB peak)
agrees on the top token 96.0% of the time, KL 0.022 on a coding set;
Optimized Quality (8-bit, 33 GiB peak) 99.3%, KL 0.001.

The last two rows are the ones that matter for long agent sessions: the
llama.cpp logs on this machine show requests running at ~160k tokens of
context and ~19 tok/s.

Newer llama.cpp does not close the gap, and cannot be dropped in: upstream
removed `--no-mmap` (now `--load-mode none`), which every llama.cpp slot here
is launched with.

## What changes when a slot moves

- **The model.** MTPLX serves a *pack*: a directory of MLX safetensors with an
  `mtplx_runtime.json`, not a GGUF. The pack above is the stock
  `Qwen/Qwen3.8-27B`, 5.8 bits/weight, 19 GiB on disk, 25 GiB peak, with its
  vision tower inside (so `*_MMPROJ_PATH` is not read). A fine-tune needs its
  own pack, built from its safetensors with `mtplx forge build`.
- **Settings that carry over:** model alias (served as `--model-id`), context
  size, temperature / top-p / top-k, reasoning effort, preserve-thinking,
  `CACHE_RAM`, which caps MTPLX's warm-conversation cache
  (`MTPLX_SESSION_BANK_MAX_BYTES`; left alone it takes half the memory the
  weights leave, ~37 GB on this machine).
- **Settings that do not:** placement, batch sizes, the `SPEC_*` draft
  settings, `CUSTOM_ARGS_JSON` (llama-server flags), and the KV cache types
  (below). MTPLX's own flags go in `LLM_A_MTPLX_ARGS_JSON`, same format.

## KV cache quantization

Its own setting, `LLM_A_MTPLX_KV_QUANT` (`off`, `q8`, `q4`; LLM A MTPLX KV
Cache in the UI), off by default -- deliberately not `CACHE_TYPE_K/V`. The
27B's cache is 64 KiB a token unquantized, 16 GiB at 262k context, and q8
halves it. But measured on the Studio with the Speed pack:

| | off | q8 |
|---|---|---|
| Code, short context | 84.6 tok/s | 80.0 tok/s |
| Decode at ~100k context | 67.9 tok/s | 22.2 tok/s |

Past a context threshold MTPLX 2.12 verifies a quantized cache on a slow,
uncompiled path (its per-request stats say `compiled_verify.fallback_reasons:
context_above_threshold` on every step), so q8 turns long sessions back into
llama.cpp speeds. Carrying the GGUF slot's q8_0 over would have done that
silently. Worth trying again when MTPLX says the long-context lane is fixed.

What sets memory at long context instead: the conversation itself -- the
cache is paged and grows with it, 64 KiB a token, so `LLM_A_CTX_SIZE` is the
ceiling on how far it can grow -- the pack (Speed is 8.6 GB lighter than
Quality), and `CACHE_RAM`, the warm-conversation cache.
- **The proxy is unchanged.** think / nothink / code, reasoning separated into
  `reasoning_content`, and streamed tool calls all work through it as they do
  against llama-server.
- **Health** reads MTPLX's `/health` (`"ok": true`) instead of `/props`.
  `validate.sh` does the same.
- **Telemetry** has less to say: the slot, context and Prometheus panels are
  llama-server's, and are not asked of MTPLX. MTPLX's own dashboard is at the
  slot's port, `http://127.0.0.1:8010/`.
- **Binds loopback only.** MTPLX demands an API key on any other address, and
  the proxy sends none.

## Setting it up

Once per Mac, the runtime:

```bash
bash scripts/install-mtplx-runtime.sh
```

It creates `deps/mtplx-venv` -- its own, because MTPLX pins mlx 0.32 and
transformers < 5.15, which the MLX runtime's mlx-audio has moved past -- and
downloads the Optimized Speed pack at a pinned revision into `models/mlx/`
(`--no-pack` skips that; `--pack REPO@REVISION` picks another).

## Getting and switching packs

The same way as GGUFs:

- **Download** from the Models page. Give it a pack repo, such as
  `Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality`, and it offers the whole
  pack as one model. It lands in `models/mlx/<repo name>/`, pinned to the
  commit the repo was at when the download started. It is staged in
  `<name>.part/` until every file is in, so the pickers never offer half a
  pack, and an interrupted download resumes.
- **Switch** by choosing the pack wherever a GGUF would be chosen: LLM A's or
  LLM B's model path in Configuration (packs are marked MTPLX, and offered
  only there), or a custom model, then restart the slot. No reinstall.

The slot's engine setting, `LLM_A_ENGINE` (LLM A Server in the UI), is `auto`
by default: a pack runs on MTPLX and a GGUF on llama.cpp, so the model is the
whole switch. Setting it to `llamacpp` or `mtplx` forces one, and a model that
contradicts a forced engine is refused at start with the reason.
