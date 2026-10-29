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
| Engine | `ashalliants/ninfer-3090` v0.15.1 @ `d2dd8e70e5` (2026-10-09), `dependencies.json` entry `ninfer-3090` |
| Engine on llms | v0.16.2 @ `8426065f` (2026-10-24), built by hand in `deps/ninfer-3090-v0162` and chosen with `NINFER_BIN`; blank `NINFER_BIN` is the v0.15.1 rollback |
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
| `LLM_*_NINFER_NGRAM` | on with `dflash2` at 1–2 lanes, v0.16+ | N-gram copy drafting (`--ngram-draft-tokens 15`): text the reply copies from its prompt is drafted from the prompt. Lossless; the fork measured +42% decode at one lane on agent traffic. |

The defaults leave out the fork's lossy memory trades (`--embedding-q4`, `--lm-head-q6`,
`--gdn-state-fp16`).

### v0.16.0's structured output, fixed in v0.16.1

v0.16.0 (2026-10-14) breaks structured output on llms. With `response_format`
`json_object` or `json_schema` the replies are valid JSON with mangled content:
`{".city":".Tokyo"}`, `{"city":": ","country":": "}`, values wrapped in arrays. It does so
with DFlash2, MTP and no speculation alike, and with both 27B artifacts; unconstrained
replies are fine. v0.11 and v0.15.1 answer the same requests correctly (tested 2026-10-18,
`{"city": "Tokyo", "country": "Japan"}` in all six cases). v0.16 also refuses a
`json_object` that carries a `schema` ("json_object format has no options"), which the chat
proxy adds to every bare `json_object`. Hermes's helpers and the memory graph ask llms for
JSON, so the pin stays on v0.15.1 until a release fixes it. Reported upstream as
[ashalliants/ninfer-3090#331](https://github.com/ashalliants/ninfer-3090/issues/331), with a
temperature-0 reproduction (11 of 12 constrained replies wrong on v0.16.0, none on v0.15.1).

v0.16.1 fixed both (#335): v0.16.0 had compiled the JSON grammar without whitespace, which
pushed the model off its own tokens. The constrained reply now keeps the model's layout,
indentation included, and `json_object` takes `"schema":{"type":"object"}`. Tested on
2026-10-24 with v0.16.2: in all five configurations (both artifacts, DFlash2 and none,
v0.15.1 as baseline) every constrained reply was the unconstrained text with the same token
count, and the same on the live slot with n-gram drafting. The test kit is
`ninfer-work/run-331.sh` on llms. `json_object` with any narrower schema is still refused,
so the proxy sends that as `json_schema` (below). Prose on the live slot: 103 tok/s against
105 on v0.15.1 (thinking off, temperature 0, 600 tokens).

The launcher is ready for v0.16 all the same. It asks the binary (`--help`) which cache
flags it takes: v0.16 replaced `--max-private-continuations 8 --max-shared-prefixes 8
--auto-prefix-grid --host-kv-mib N` with `--host-context-mib N`, and each release stops at
startup on the other's. N-gram copy drafting is passed only to a build that has it (v0.16).
v0.16 also drops the `/slots` endpoint, which gpu-lease falls back from, and renames the
`ninfer:*` metrics to `ninfer_*`.

Measured on llms with v0.15.1 and the v0.13 artifact against v0.11 with neroued's, same
flags otherwise (DFlash2, rk8v4, one lane, thinking off, temperature 0.7): prose 113–116
tok/s against 110, a reply that copies a 2.6K-token file back 294–296 against 280.

## A second 27B on GPU 0 (2026-10-29)

llms runs the 27B twice: LLM A on GPU 1 and LLM B on GPU 0. Same artifact, same build, same flags
except the port (8020), the model id (`chat-dense-b`) and a 4 GiB host cache
(`LLM_B_NINFER_HOST_KV_MIB=4096`; two of LLM A's 24 GiB would not fit llms's 62 GB). To make room
the embedding and reranker models moved to the Studio and the 9B task model was retired.

| GPU 0, GiB | Before | Now |
|---|---:|---:|
| Frigate | 0.5 | 0.5 |
| Embedding 4B, awake / asleep | 4.3 / 0.3 | on the Studio |
| Reranker 0.6B, awake / asleep | 1.9 / 0.3 | on the Studio |
| Task 9B, awake / asleep | ~9.5 / 0.3 | retired |
| 27B, 131,072 context | — | 21.1 (1.93 free) |

- **Task calls** (`llms:8007`, model `task`) reach LLM B's no-think port: `NOTHINK2_PORT=8007`.
  Hermes's config did not change. Titles come back in 0.14–0.32 s; JSON mode and streaming work as
  on LLM A.
- **LLM B's proxy:** think 8103, chat 8007, code 8108, aggregate 8112 (`think`/`chat`/`code`, as on
  8012). It binds 127.0.0.1 (`LLM_B_LISTEN_HOST`) because Tailscale Serve already publishes those
  ports on the tailnet address.
- **A ComfyUI lease leaves it alone.** Only llm-a-proxy follows the GPU 1 lease
  (`follows_gpu_lease` in `web/backends/proxies.py`). LLM B falls back to the Studio only when its
  own backend is down.
- **Embeddings and reranking come from the Studio:** the same Qwen3-Embedding-4B Q6_K file
  (sha256 `7e7693eb…`) and 0.6B reranker, with llms's settings (q8_0 KV, 2048 batches, 4 rerank
  slots, 900 s idle unload). `llms:8005` and `llms:8006` still answer, so no caller changed:
  `systemd/llm-forward.service` is an nginx instance of its own on llms that passes
  127.0.0.1:8005/8006 to the Studio and rewrites `Host`, which the Studio's Tailscale Serve routes
  by. Studio vectors match llms's at cosine 0.99988 or better; Lens's probe check reads 0.9999.
- **What that costs the Studio:** an idle embedder costs no measurable speed and 3.8 GiB. A busy one
  (bulk, back to back, 4,272 tok/s) took Splash's decode from 76 to 44 tok/s short and from 70 to
  41 at 16K, and its prefill down 13%. Embedding ran ~12M tokens a day on 10-28 and 10-29.
- **voice-tts no longer fits on GPU 0** (5.7 GB awake). It is stopped and disabled.

Measured 2026-10-29 with `scripts/bench-offload.py`, 400 tokens, thinking off, tok/s:

| | LLM B alone | LLM B, LLM A busy | LLM A, same window |
|---|---:|---:|---:|
| Prose, short | 94.9 | 88.9 | 88.9 |
| Prose, 16K | 103.1 | 97.9 | 95.0 |
| Code, short | 296.5 | 286.9 | 296.0 |
| Code, 16K | 277.7 | 277.0 | 279.5 |
| Prefill, 16K | 1,762 | 1,751 | 1,788 |

To undo it on llms: stop `llm-b-proxy` and `llm-b` from the manager, restore
`config/llm-stack.env.bak-before-llm-b-ninfer-20261029`, run `sudo systemctl disable --now
llm-forward`, and start `embed`, `rerank` and `task`. On the Studio, stop `embed` and `rerank`
(its previous settings are in `config/llm-stack.env.bak-before-embed-rerank-move-20261029`).

## What the chat proxy changes

NInfer refuses what it cannot honour instead of ignoring it. For an NInfer slot the proxy
(`_adapt_for_ninfer`) makes these changes:

- Renames the model to the slot alias.
- On Chat Completions:
  - drops a non-neutral `repetition_penalty`, nonzero `logit_bias` and requested log
    probabilities;
  - caps `top_k` at 20;
  - turns `tool_choice: "required"` into `"auto"`;
  - sends a `json_object` carrying a schema other than `{"type":"object"}` as `json_schema`,
    which v0.16 requires and v0.15.1 also takes.
- On Responses, keeps only the top-level fields that route accepts. The reasoning level moves to
  `reasoning.effort`, and thinking-off becomes `"none"`.

Each drop is logged as `ninfer-adapt … dropped=…`.

## Health, gpu-lease

`/health` answers 503 until the engine has loaded and 200 `{"status":"ok"}` after. v0.16
drops `/slots`, so `scripts/gpu-lease.py` reads `llamacpp:requests_processing` from
`/metrics` when `/slots` answers 404, to wait for an in-flight request, or for the reload
after ComfyUI is done.

## Prefix reuse

An agent resends its whole conversation every turn, so what matters is whether the next
turn prefills only what is new. `/metrics` says: `ninfer_prompt_tokens_total` counts every
prompt token and `ninfer_prompt_tokens_cached_total` the ones served from cache. The
llama.cpp-compatible `llamacpp:prompt_tokens_total` beside them counts only the tokens
prefilled, so the two must not be added together.

`scripts/ninfer_prefix_test.py` measures it. It sends one synthetic agent request of about 36k tokens
(system prompt, 25 tool schemas, a research session of tool calls and long results; a fresh
nonce each run), then the same conversation with its reply and one more turn, and prints JSON:
both TTFTs, the cached tokens of each, and PASS when the second request's cached tokens are at
least 90% of the prefix and its TTFT is under 10 s. It goes through a proxy with the
fallback-hop header (`--url`, default `http://127.0.0.1:8012`) and reads the backend's own
`/metrics` (`--metrics-url`, default `http://127.0.0.1:8010/metrics`). It skips (exit 2)
while the backend is busy. `--interleave N` sends N short, unrelated conversations between
the two, so the first one's state has to leave the card and come back from the host cache
(`--host-context-mib`, `LLM_*_NINFER_HOST_KV_MIB`), as it must when agent turns are
interleaved with other jobs; `--shape replace` swaps the last user turn instead of adding one.

First run on llms, 2026-10-26, v0.16.2, host cache 4,096 MiB, no interleaving: the second
request reused 35,706 of 35,715 prefix tokens, and TTFT fell from 26.5 s to 0.21 s. The
template puts the tool schemas first, so even the cold request found 2,110 tokens cached. The
host cache was full (4.11 of 4.29 GB) and each request added more
`ninfer_host_context_shortage_failures_total`, so a conversation whose state has left the
card's two state slots cannot come back; `--interleave 2` is the test for that.

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
