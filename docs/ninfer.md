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
| `LLM_*_NINFER_TOOL_CONSTRAINTS` | blank | The proxy sends this as `tool_constraints` on Chat Completions with tools. Read on the next request, with no restart. See "Fine-tunes". |
| `LLM_*_NINFER_THINK_EFFORT` | blank | Every request that thinks runs at this level, whatever level it asked for. Read on the next request, with no restart. |

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

## Two lanes (2026-10-29)

Both 27Bs take two requests at once, so Hermes's quick calls (titles, approvals and MCP on
`llms:8007`) run beside a batch job instead of queueing behind it.

```
LLM_A_N_PARALLEL=2                                      # --max-concurrency 2
LLM_A_NINFER_ARGS_JSON='["--device-state-slots","0"]'
LLM_B_N_PARALLEL=2
LLM_B_NINFER_ARGS_JSON='["--device-state-slots","0"]'
```

What a second lane is in NInfer (its `docs/serving.md`):

- **The context isn't divided.** `--max-context` stays the per-request ceiling (131,072) and the KV
  pool is shared, so a lone request still gets the whole window. Only two long requests at once can
  fill the pool; then one pauses into host memory and resumes. The manager's context panel and the
  state API still divide by the lane count, as for llama.cpp (65,536 "per slot", 262,144 total, and
  an `n_ctx_per_slot` drift entry). That is display only.
- **No extra throughput.** Speculation runs only while one request decodes; while both decode the
  rounds are plain. N-gram copy drafting stays on (NInfer's launcher default at one or two lanes). So
  batch pools keep one worker per server, and the second lane is for interactive calls.
- **Device states.** `--device-state-slots`, the StateImages beyond one per lane, defaults to the
  lane count. At 0 the card holds two, one per lane: as many as one lane plus one extra did before.
  Retained conversations wait in the host cache.

Memory, from the boot `capacity` line and `nvidia-smi`:

| | runtime | free at boot | `nvidia-smi` free | process |
|---|---:|---:|---:|---:|
| LLM B, one lane, 1 extra state | 4.70 GiB | 1.93 GiB | 1,991 MiB | 21,614 MiB |
| LLM B, two lanes, 1 extra state | 5.82 GiB | 1.64 GiB | 1,681 MiB | — |
| LLM B, two lanes, no extra state | 5.64 GiB | 1.84 GiB | 1,884 MiB | 21,726 MiB |
| LLM A, one lane, 1 extra state | 4.70 GiB | 1.98 GiB | 2,025 MiB | 21,616 MiB |
| LLM A, two lanes, no extra state | 5.64 GiB | 1.87 GiB | 1,915 MiB | 21,724 MiB |

- The extra state cost 203 MiB on GPU 0, more than the 147 MiB NInfer's notes give for one
  StateImage, and left 1,681 MiB free. The wildlife scan needs 1,800 MiB free on GPU 0 to start
  (Hermes `50de5b9`; it peaked at 1,432 MiB), so LLM B runs without it, at 1,884 MiB free. During a
  two-lane run GPU 0's free memory moved between 1,855 and 1,902 MiB; that was the desktop's
  processes, and NInfer's own allocation didn't change.
- GPU 1: ComfyUI idles at 458 MiB beside LLM A. After the 10-26 lease releases LLM A booted with
  2.01–2.02 GiB free, as on any boot, so ComfyUI gives its memory back and the two-lane 27B (110 MiB
  more) still fits after a lease.

Measured 2026-10-29:

- **A title beside a long call.** A streaming `:8112` call (model `chat`, 8,135 tokens in 84.9 s) and
  four title calls to `:8007` (model `task`), 20 s apart: each came back in 0.49–0.54 s while the
  long one streamed, and the log shows `running 2`. On LLM A, `:8012` (6,305 tokens in 63.5 s) beside
  `:8004`: 0.46–0.52 s. Both went through the chat proxies, so the proxies pass both lanes through.
- **One stream** (`scripts/bench-offload.py`, 400 tokens, thinking off, tok/s; prose three runs each,
  lowest–highest):

  | | LLM A, one lane | LLM A, two lanes | LLM B, two lanes |
  |---|---:|---:|---:|
  | Prose, short | 92.7–105.7 | 96.2–102.7 | 88.5–95.8 |
  | Prose, 16K | 84.9–94.3 | 91.0–95.3 | 91.5–96.7 |
  | Code, short | 297.3 | 297.2 | 294.4 |
  | Code, 16K | 279.2 | 278.5 | 275.3 |
  | Prefill, 16K | 1,761–1,788 | 1,741–1,763 | 1,731–1,750 |

  Within noise, except prefill: about 1.5% slower at 16K, and a 185-token prompt reads 737 → 608
  tok/s, about 50 ms more to the first token.
- **Prefix reuse** (`scripts/ninfer_prefix_test.py --interleave 2`) passes on both. LLM B reused
  35,706 of 35,715 tokens (TTFT 29.1 → 0.27 s), LLM A 35,707 (27.1 → 0.29 s). LLM A's first run read
  FAIL only because its 24-token reply was a tool call cut at the limit, so the stream carried no
  text to time; the server logged TTFT 291 ms with 99.8% cached. LLM B's 4 GiB host cache stays full
  (`ninfer_host_context_shortage_failures_total` 5 → 13 over the test), as LLM A's was at 4 GiB.
- **Hermes's doctor:** `llms-task` and `llms-model` pass.

gpu-lease still drains LLM A correctly. `backend_busy()` reads `llamacpp:requests_processing`, which
counts both lanes, so `acquire` waits (up to `GPU_LEASE_DRAIN_SECONDS`, 600) until neither is
generating; the proxy has sent new requests to the Studio since the lease file was written.

To undo it on llms: restore `config/llm-stack.env.bak-before-two-lanes-20261029` (or set both
`N_PARALLEL` back to 1 and both `NINFER_ARGS_JSON` to `"[]"`), then restart `llm-b` and `llm-a` from
the manager.

## The uncensored 27B (2026-10-30)

`models/qwen3_8_27b_uncensored-v013.ninfer` is JonathanColetti's `Qwen3.8-27B-Uncensored` converted with the same
recipe as the stock v0.13 artifact. An older conversion doesn't fit GPU 1 beside ComfyUI, and v0.13 artifacts refuse
`--embedding-q4` and `--lm-head-q6`, so the fix is to reconvert. The result is 768 bytes smaller than the stock
artifact, and boots with the same runtime (5.64 GiB) and the same 1.87 GiB free.

The `qwen3_8_27b` recipe needs an importance matrix, and two packages the convert venv lacked. The imatrix is
Unsloth's, made on the stock weights; the tokenizer and preprocessor files are the official repo's, because the
abliterated repo's `tokenizer.json` has an older regex (no `\p{M}`) that NInfer refuses.

```
# from deps/ninfer-3090-v0162, with M=<models folder> and R=$M/hf/qwen38-official-resources
../ninfer-convert-venv/bin/python -m pip install safetensors gguf
curl -fsSL -o $R/imatrix_unsloth.gguf \
  https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/imatrix_unsloth.gguf
../ninfer-convert-venv/bin/python -m tools.convert.imatrix --gguf $R/imatrix_unsloth.gguf \
  --config $M/hf/Qwen3.8-27B-Uncensored/config.json --out $R/qwen3_8_27b.imatrix.safetensors
nice -n 10 ionice -c3 ../ninfer-convert-venv/bin/python -m tools.convert \
  --model $M/hf/Qwen3.8-27B-Uncensored --recipe qwen3_8_27b \
  --source imatrix=$R/qwen3_8_27b.imatrix.safetensors --source dflash2=$M/hf/Qwen3.8-27B-DFlash2 \
  --components text,vision,mtp,dflash2 --proposal --name qwen3.8-27b-uncensored \
  --resource tokenizer.json=$R/tokenizer.json --resource tokenizer_config.json=$R/tokenizer_config.json \
  --device cpu --out $M/qwen3_8_27b_uncensored-v013.ninfer
```

It took 2 h 36 min on six CPU cores (9,364 s) while both 27Bs served, using under 0.5 GB of RAM. The older recipe took
about 6 minutes; `grouped_search` searches each group of weights in a first pass of 192 parameters, about 1.3 minutes a
layer, and a second pass then handles the other 992 at about 10 a minute. Judge the time from both passes.

Serve it with the stock artifact's `--device-state-slots 0`, as in "Two lanes".

## Converting on the GPU

The same recipe took 17 minutes on GPU 1 (1,019 and 1,023 s with two running at once), about nine times faster than on
the CPU. `tools.convert` defaults to CUDA, but the convert venv has CPU torch, so run it with ComfyUI's python, which has
CUDA torch, numpy, safetensors and gguf: nothing needs installing. Each conversion used about 400 MiB of VRAM. They ran
with the 27B stopped under a lease (`gpu-lease.py acquire --holder ninfer-convert --max-minutes 120`), and running beside
the 27B is untested.

```
# from deps/ninfer-3090-v0162, with M and R as above
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 /home/ellie/AI/venv/bin/python -m tools.convert \
  --model $M/hf/<source> --recipe qwen3_8_27b \
  --source imatrix=$R/qwen3_8_27b.imatrix.safetensors --source dflash2=$M/hf/Qwen3.8-27B-DFlash2 \
  --components text,vision,mtp,dflash2 --proposal --name <name> \
  --resource tokenizer.json=$R/tokenizer.json --resource tokenizer_config.json=$R/tokenizer_config.json \
  --device cuda --out $M/<file>-v013.ninfer
```

A model that ships no vision tower needs one if the slot serves with `--vision`. Hemmingway-1.5's source folder is a
merge: its own text shards by symlink, the stock tower's 333 `model.visual.*` tensors in a separate file, and a config and
index that name both (`models/hf/hemmingway_merged.py` on llms).

## Fine-tunes: the drafter and the thinking close

- **The DFlash2 drafter is trained on stock weights.** It suits a model close to stock (Hemmingway-1.5, 70.9% of drafts
  accepted) and not one far from it (Turbo Fusion, 17.1%, against 82.5% on its own MTP). Check
  `ninfer_spec_decode_accepted_tokens_total` over `…_draft_tokens_total` on `/metrics`, and set `NINFER_SPEC=mtp` for a
  model far from stock. MTP also leaves about 1.8 GB more of the card free.
- **Strict tool mode needs the exact close.** With `tool_constraints: "basic"`, the default, only `\n</think>\n\n` ends
  thinking, and the grammar refuses a stop before it. Turbo Fusion sometimes opens its reply with `</think>`, or writes
  one line break after it. The server then counts everything after as thinking, refuses the stop, and the model writes
  tool calls until the output limit, with nothing returned as a call. Set `LLM_*_NINFER_TOOL_CONSTRAINTS=auto` for such
  a model. It's a request field with no server flag, so the proxy sends it, and v0.17.1 has the same rule. The Responses
  route refuses the field (`unknown_parameter`), so a Responses client still gets the strict grammar.
- **A fine-tune can need more effort.** Turbo Fusion thinks little at any level and takes the first plausible answer:
  `LLM_*_NINFER_THINK_EFFORT=xhigh` took its chains from 8 of 10 to 10 of 10 at little cost.
- **Request settings are read live.** The proxy re-reads both keys from `config/llm-stack.env` when the file changes, so
  they follow a model switch, a lease's return or an edit here without a proxy restart. A request that bypasses the proxy
  (a direct `:8010`) doesn't get them.

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
    which v0.16 requires and v0.15.1 also takes;
  - adds `tool_constraints` to a request with tools when `LLM_*_NINFER_TOOL_CONSTRAINTS` is set
    and the request didn't name its own.
- On Responses, keeps only the top-level fields that route accepts. The reasoning level moves to
  `reasoning.effort`, and thinking-off becomes `"none"`.
- On both, `LLM_*_NINFER_THINK_EFFORT` replaces the level of a request that thinks. A request with
  thinking off stays off.

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
