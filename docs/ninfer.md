# NInfer: Qwen3.8-27B on one RTX 3090

NInfer is a C++/CUDA server built for Qwen3.8-27B and Qwen3.6 only. The stack runs
[ashalliants/ninfer-3090](https://github.com/ashalliants/ninfer-3090), the maintained RTX 3090
fork of [Neroued/ninfer](https://github.com/Neroued/ninfer). It is a chat-slot engine next to
llama.cpp, MTPLX and Splash, Linux only.

Why it exists here: on `llms`, LLM A ran Qwen3.8-27B Q4_K_M on llama.cpp at about 41 tok/s
decode and 650–1,100 tok/s prefill, on GPU1. NInfer's paged INT8 KV, MTP speculation and
CUDA-graph decode are built for exactly that card and model.

## Which fork, and why

- **Don-Chad/ninfer-3090**, the first 3090 port, only compile-checks Linux. Its notes say "a
  real-artifact Linux generation and Linux performance qualification remain open".
- **ashalliants/ninfer-3090** continues it:
  - Linux release archives and a native Ubuntu 24.04 build.
  - Vision "overlay" residency: the vision tower waits in host RAM and borrows the card per
    image, so vision costs no context.
  - Structured JSON output, llama.cpp-compatible `/slots` and `/metrics`, and model files in
    container v3.
- Others run it on Linux too: [practicorecoza/ninfer](https://github.com/practicorecoza/ninfer)
  measured 103 tok/s single-stream (MTP3, INT8) on rented RTX 3090s.

## Pins

| What | Pin |
|---|---|
| Engine | `ashalliants/ninfer-3090` v0.16.0 @ `50a337e19f` (2026-10-14), `dependencies.json` entry `ninfer-3090` |
| Model | `WarlaxZ/Qwen3.8-27B-NInfer-3090` @ `d47f2732d369acaec76dc44228f67c72b081df2f`, `qwen3_8_27b.ninfer` (saved on llms as `qwen3_8_27b-v013.ninfer`) |
| Model size | 18,982,458,624 bytes, sha256 `7f2a0086a071ee932c66106e1552d118a85cfec8ddef13c67a54cd64abb39706` |

Until 2026-10-18 the stack ran `da0f15c25d` (v0.11.0+138) with neroued's original artifact
(`neroued/Qwen3.8-27B-NInfer` @ `1cbd84e7`, 20.4 GB). The fork's v0.13.0 re-converted the
same stock weights with an imatrix-weighted scale search: 1.46 GB smaller and, by its
KL measurements against Unsloth's Q8_0, 15% closer to it. That artifact needs v0.13 or
newer binaries and no longer takes the `--embedding-q4` load-time transcode.

The fork's `scripts/download-model.sh qwen38-27b` fetches exactly this revision and checks size
and hash before keeping the file:

```bash
NINFER_MODEL_DIR=models deps/ninfer-3090/scripts/download-model.sh qwen38-27b
```

## Building

`NINFER_ENABLED=on` in `config/llm-stack.env` makes install/update do two things:

- `scripts/install-system-dependencies.sh` adds GCC 13 and the FFmpeg and curl headers.
- `scripts/install-dependencies.py` builds the pinned engine for sm_86 with the newest CUDA
  toolkit under `/usr/local`. It does not use the `nvcc` on PATH, which on Ubuntu is often an
  older one.

The build takes about an hour on a 6-core machine. By hand:

```bash
scripts/install-dependencies.py --only ninfer-3090
```

## Using it

```
LLM_A_ENGINE=ninfer          # or auto: a .ninfer model path picks it
LLM_A_MODEL_PATH=/…/models/qwen3_8_27b.ninfer
LLM_A_CTX_SIZE=131072
LLM_A_GPU_VISIBLE_DEVICES=1  # the card; NInfer is told --device 0
```

What carries over from the slot:

- **Alias:** becomes `--model-id`, the only name NInfer accepts. The proxy renames requests to
  it.
- **Context size:** both the per-request ceiling and the KV pool.
- **Parallel slots:** `--max-concurrency`.
- **Preserve thinking.**

The engine's own settings are in the "NVIDIA (NInfer)" section:

| Setting | Default | What it trades |
|---|---|---|
| `LLM_*_NINFER_KV_DTYPE` | `int8` | Perplexity indistinguishable from bf16. `rk8v4` is 23% smaller for +0.09%; `rk4v4` is half the size for +0.21% and lower draft acceptance. |
| `LLM_*_NINFER_SPEC` | `mtp` | The model's own draft heads, 3 tokens. `dflash2` (7 tokens, a separate drafter) is faster at one stream, but its weights cost ~65K tokens of context on 24 GB. |
| `LLM_*_NINFER_VISION` | `overlay` | Images without giving up context. `resident` costs ~24K tokens; `off` answers images with a 400. |
| `LLM_*_NINFER_PREFILL_CUBLAS` | `off` | `on`: 1.6–1.8× prompt processing for +0.156% perplexity. |
| `LLM_*_NINFER_KV_CAPACITY` | the context | `auto` fills the card with 1 GiB spare. |
| `LLM_*_NINFER_HOST_KV_MIB` | 4096 | Pinned host RAM for cached conversations that no longer fit on the card (`--host-context-mib`). |
| `LLM_*_NINFER_NGRAM` | on with `dflash2` at 1–2 lanes | N-gram copy drafting (`--ngram-draft-tokens 15`): text the reply copies from its prompt is drafted from the prompt. Lossless; the fork measured +42% decode at one lane on agent traffic. |

The defaults leave out the fork's lossy memory trades (`--embedding-q4`, `--lm-head-q6`,
`--gdn-state-fp16`).

### v0.16 changes the launcher had to follow

v0.16.0 replaced the context cache, and `ninfer-serve` now stops at startup with
`unknown argument` on any knob that sized the old one. The stack used to pass four:
`--max-private-continuations 8 --max-shared-prefixes 8 --auto-prefix-grid --host-kv-mib N`.
It now passes `--host-context-mib N` (same setting, same 4 GiB default). Also gone in v0.16:
the `/slots` endpoint and the `ninfer:*` metric names (now `ninfer_*`); the `llamacpp:*`
series are unchanged.

## What the chat proxy changes

NInfer refuses what it cannot honour instead of ignoring it. For an NInfer slot the proxy
(`_adapt_for_ninfer`) makes these changes:

- Renames the model to the slot alias.
- On Chat Completions:
  - drops a non-neutral `repetition_penalty`, nonzero `logit_bias` and requested log
    probabilities;
  - caps `top_k` at 20;
  - turns `tool_choice: "required"` into `"auto"`.
- On Responses, keeps only the top-level fields that route accepts. The reasoning level moves to
  `reasoning.effort`, and thinking-off becomes `"none"`.

Each drop is logged as `ninfer-adapt … dropped=…`.

## Health, gpu-lease

`/health` answers 503 until the engine has loaded and 200 `{"status":"ok"}` after. v0.16
dropped `/slots`, so `scripts/gpu-lease.py` reads `llamacpp:requests_processing` from
`/metrics` when `/slots` answers 404, to wait for an in-flight request, or for the reload
after ComfyUI is done.

## Measured on llms

Measured 2026-09-29 on llms, GPU1 (RTX 3090, driver 615.71, CUDA 13.3), with
`scripts/bench-offload.py`. The prompts were identical across engines, thinking was off, and
temperature was 0.7. "Prose" is a free-text reply and "code" a reply that copies much of its
context. All figures are tok/s.

| | llama.cpp Q4_K_M | A: MTP3, int8 | B: DFlash2, rk8v4 |
|---|---:|---:|---:|
| Prose, short | 47 | 91 | **101** |
| Prose, 32K | 42 | **83** | 79 |
| Prose, 100K | 34 | **68** | 66 |
| Code, short | 129 | 143 | **234** |
| Code, 32K | 111 | 130 | **206** |
| Code, 100K | 84 | 106 | **171** |
| Prefill, 8K | 1,329 | **1,672** | — |
| Prefill, 32K | 1,184 | **1,556** | 1,526 |
| Prefill, 100K | 877 | **1,153** | 1,132 |

Profile settings:

- **A** (131,072 context, int8 KV, MTP3 + draft head, vision overlay) boots with 1.83 GiB of the
  card free.
- **B** (the same with `NINFER_SPEC=dflash2`, `NINFER_KV_DTYPE=rk8v4` and
  `NINFER_ARGS_JSON=["--embedding-q4"]`) boots with 1.82 GiB free.

About the two profiles:

- DFlash2 drafts seven tokens at a time. On text it can predict, that makes it 60% faster than
  MTP3. On free prose it lands only 20–25% of its drafts, which is about even with MTP3 and
  3–5% behind it deep in a long context.
- B is what LLM A runs: faster overall, for +0.08% perplexity from rk8v4. The 4-bit embedding
  costs nothing measurable.

Both profiles passed these checks:

- a 1920×1080 image;
- a tool call and the turn after its result;
- reasoning effort none, low, medium and high;
- JSON mode;
- a mid-conversation system message;
- a 126,406-token prompt that found a planted code, read at 1.02K tok/s, with no change in
  VRAM.

Results are in `benchmarks/2026092*-llms-*.json`.
