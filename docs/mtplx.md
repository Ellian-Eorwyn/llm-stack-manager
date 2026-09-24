# MTPLX: a chat slot on Apple silicon without llama.cpp

`LLM_A_ENGINE=mtplx` (or `LLM_B_ENGINE`) serves that chat slot with
[MTPLX](https://github.com/youssofal/mtplx) instead of llama-server. MTPLX is
an MLX server that drafts from the model's own multi-token-prediction head and
verifies the draft in one batched pass. llama.cpp's `draft-mtp` does the same
thing; MTPLX does it in a runtime written for Apple silicon, and on the Studio
that is the difference below.

Linux is untouched. The engine is opt-in per slot, defaults to `llamacpp`, and
refuses to build a command on anything but macOS.

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
  size, temperature / top-p / top-k, reasoning effort, preserve-thinking, and
  `CACHE_RAM`, which caps MTPLX's warm-conversation cache
  (`MTPLX_SESSION_BANK_MAX_BYTES`). Left alone that cache takes half the memory
  the weights leave -- about 37 GB on this machine.
- **Settings that do not:** placement, KV cache types, batch sizes, the
  `SPEC_*` draft settings, `CUSTOM_ARGS_JSON` (llama-server flags). MTPLX's
  own flags go in `LLM_A_MTPLX_ARGS_JSON`, same format.
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

```bash
bash scripts/install-mtplx-runtime.sh
```

Creates `deps/mtplx-venv` (its own, because MTPLX pins mlx 0.32 and
transformers < 5.15, which the MLX runtime's mlx-audio has moved past) and
downloads the pack at a pinned revision into `models/mlx/`. Then, in
`config/llm-stack.env` or under Apple Silicon (MLX) in the UI:

```sh
LLM_A_ENGINE=mtplx
LLM_A_MODEL_PATH=${STACK_DIR}/models/mlx/Qwen3.8-27B-MTPLX-Optimized-Speed
```

and restart LLM A. No reinstall: `start-backend.sh` chooses the engine each
time the slot starts. To go back, save the current config first (Saved
Configs), or set the engine to `llamacpp` and the model path to the GGUF.
Switching llm-a to a custom model does this for you: an MTPLX pack sets the
engine to `mtplx`, a GGUF sets it back.
