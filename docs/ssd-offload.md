# SSD offload: a model bigger than the RAM it may use

Set a chat slot's **Memory Mode** to `ssd-offload` (Config → SSD Offload) and
its model may live partly on the SSD. Leave it at `resident`, the default, and
nothing changes: every offload setting is inert, and the command line is the
one the slot always ran, down to the byte (`tests/test_launchers.py`).

It exists for Qwen 3.8 Flash-Next on the Studio (M5 Ultra, 96 GB). The target
there is 262k context, 50–64 GB resident, and ≥50 tok/s at short-to-mid
depth. None of it is specific to that model.

## The settings

Each one exists for LLM A (`LLM_A_*`) and LLM B (`LLM_B_*`). Changing any of
them restarts that slot.

| Setting | Values | MTPLX | llama.cpp |
|---|---|---|---|
| `MEMORY_MODE` | `resident` / `ssd-offload` | the switch | the switch |
| `RAM_BUDGET_GB` | GiB, blank = engine default | `MTPLX_MEMORY_LIMIT_BYTES`, which MTPLX treats as its whole engine budget. A budget below the pack's floor is refused at load, with that reason | not passed. The budget report prices against it (below) |
| `NGRAM_PREWARM` | `auto` / `off` / `all` / GiB | `--ngram-prewarm` | not read |
| `EXPERT_CACHE_GB` | GiB, blank = the build chooses | not read: MTPLX keeps every expert resident | `--moe-stream-cache`, on a build that has it |
| `EXPERT_STREAM_IO_THREADS` | count, default 8 | not read | `--moe-stream-io-threads` |
| `LLAMA_SERVER_BIN` | path, blank = `LLAMA_SERVER_BIN` | not read | this slot's binary |

A setting an engine cannot honour is ignored, not refused. A slot moves
between a GGUF and a pack by choosing the model alone, and a setting chosen for
one engine must not stop the other from starting.

### What an offloaded llama.cpp slot runs

In place of `--no-mmap` and `--mlock`, which it never passes:

- `--load-mode mmap`, on a build that knows `--load-mode`. The pinned build
  and current master both do. An older fork gets nothing, because mmap is
  already its default.
- `--override-tensor per_layer_token_embd=CPU`. This keeps the n-gram table (stored in a GGUF as the per-layer embedding) in host memory,
  where the mmap leaves it pageable; in a Metal buffer it would be wired
  whole. It is skipped when your custom arguments already carry `-ot`.
- `--moe-stream`, the cache, the thread count and `--moe-stream-direct`, but
  only when `llama-server --help` lists them. The launcher asks the binary
  rather than assuming, because passing an unknown flag fails the backend at
  start, and that failure shows up as a restart loop.

The launcher also skips `metal_keep_resident` for an offloaded slot. A wired
page is one macOS cannot send back to the SSD, and wiring everything is the
opposite of offload.

### Budget report

`web/budget.py` prices an offloaded GGUF at what it holds, not at its size on
disk. The weights get whatever the RAM budget leaves after KV, compute and
overhead; the rest is reported as `on_ssd_mib`. Without a budget, the weights
are priced whole, and the report says so. For an MTPLX pack, the report
separates the held weights from the n-gram table streamed from the SSD.

## Qwen 3.8 Flash-Next

This is a 125B-parameter MoE with about 6B active per token:

- 512 experts per layer; each token uses 10 routed experts plus 1 shared.
- 48 layers: 36 Gated DeltaNet and 12 QSA sparse-attention.
- A 51B-parameter **n-gram embedding table**, of which each token gathers a
  handful of rows. At 4-bit the table is about 32 GB, and it is built to live
  on the SSD.
- A 4B MTP head.

The context is 262,144 tokens natively. The KV cache is small because only
the 12 QSA layers keep one: about 6.4 GB at fp16 at 262k (3.4 GB at q8), plus
about 112 MB of fixed DeltaNet state. Decode still slows with depth, even when
everything is resident: the indexer scans every cached token (llama.cpp
#28734). So "≥50 tok/s" is a target for ≲32k of context, not for 262k.

### Candidates

| | Build | Engine | Resident | Notes |
|---|---|---|---|---|
| A | MTPLX **Bare Speed** pack (`Youssofal/Qwen3.8-Flash-Next-MTPLX-Bare-Speed`, 106 GB download) | MTPLX 2.12 | ~78 GiB peak | Only the n-gram table is on the SSD. The ceiling to compare against, but over the RAM target. |
| B | **3-bit** MTPLX pack, forged locally from `config/forge-recipes/` | MTPLX 2.12 | ~51 (g64) to ~58 (g32) GB of weights, plus KV and MTP | Everything resident but the n-gram table. The best candidate for ≥50 tok/s under 64 GB. |
| B′ | unsloth `UD-IQ3_XXS` GGUF (82 GB) | `llama.cpp-next` | ~55 GB | The quickest look at 3-bit quality, with no conversion. |
| C | unsloth `UD-Q4_K_XL` GGUF (111 GB), or the format the fork's docs name | `llama.cpp-moe-stream` | the budget | Full 4-bit quality, with experts paged through a cache. Measured at 18 tok/s (27.6 with MTP) on a 64 GB M5 Pro; expect more on the Ultra, probably not 50. |

The 3-bit size estimate works like this: MLX affine quantization stores a
16-bit scale and bias per group. So 4-bit/g32 costs 5 bits per weight,
3-bit/g32 costs 4, and 3-bit/g64 costs 3.5. Scaled from Bare Speed's 72.6 GB,
that gives about 58 and 51 GB. The 8-bit attention and gates don't shrink, so
expect a little more.

### Why expert streaming is unlikely to reach 50 on its own

tok/s ≈ 1 / (T_compute + (1 − hit rate) × expert bytes per token / SSD bandwidth),
multiplied by what speculation accepts.

At 4-bit, a token touches about 1.3 GB of routed experts. Reported LRU hit
rates for this model are about 81% with a quarter of the experts cached and
about 90% with half. The SSD delivers roughly 14 GB/s for expert-sized reads,
and each miss costs a read that compute cannot overlap. A 55 GB cache holds
about 80% of the experts on this machine, which puts decode around 30–45 tok/s
with MTP. The 3-bit resident pack pays no SSD cost beyond n-gram rows.

### N-gram speculation

N-gram speculation is a different thing from the n-gram table: it drafts
tokens by matching text already in the context (`SPEC_METHOD=ngram-*`). The
reported gains are 1.1–1.6× on turns that copy earlier text (code edits,
agent tool output) and roughly nothing on chat. Stacked with MTP it measured
no faster than MTP alone (#23184). On an SSD-offloaded MoE, every extra
drafted token is more experts to read, so pick one. The benchmark's `code`
task is the one that shows it.

## Running the comparison

Nothing below runs by itself. Each step is a download or build of tens to
hundreds of GB, or a restart. Run them when the Studio is not serving anything
you need, and stop LLM A first if it holds the 27B: none of the candidates fit
beside it.

```bash
# A: download Youssofal/Qwen3.8-Flash-Next-MTPLX-Bare-Speed from the Models
#    tab, which puts it under models/mlx/ where the pickers find packs.

# B: forge 3-bit packs. --model-root receives both the 354 GB BF16 source and
#    the finished pack; delete the source once both recipes are built.
deps/mtplx-venv/bin/mtplx forge build --repo Qwen/Qwen3.8-Flash-Next \
    --recipe config/forge-recipes/flash-next-3bit-g32.json \
    --out downloads/forge --run-id flash-next-3bit-g32 \
    --branded-name Qwen3.8-Flash-Next-MTPLX-3bit-g32 --model-root models/mlx

# B′ and C: the optional llama.cpp builds. --only leaves the pinned build alone.
scripts/install-dependencies.py --only llama.cpp-next
scripts/install-dependencies.py --only llama.cpp-moe-stream
```

Then, for each candidate, in LLM B (or LLM A):

| | Model | Memory Mode | RAM Budget | Other |
|---|---|---|---|---|
| A | the Bare Speed pack | `ssd-offload` | 80 | Pre-read `auto` |
| B | the 3-bit pack | `ssd-offload` | 64 | Pre-read `auto`, then `off` to see what the page cache is worth |
| B′ | `UD-IQ3_XXS` | `ssd-offload` | 64 | Binary `deps/llama.cpp-next/build/bin/llama-server`, `SPEC_METHOD=draft-mtp` |
| C | `UD-Q4_K_XL` | `ssd-offload` | 64 | Binary `deps/llama.cpp-moe-stream/build/bin/llama-server`, Expert Cache 40–52, `SPEC_METHOD=draft-mtp`, draft n-max 3 |

Then measure each candidate (or run them all with `scripts/eval-run.py`, [model-evals.md](model-evals.md)):

```bash
scripts/bench-offload.py --slot llm-b --variant flash-next-B-3bit-g32
```

That runs chat and code prompts at 0, 8k, 32k, 128k and 262k tokens of
context. It records prefill and decode tok/s, the backend's peak memory, and
the bytes it read from the SSD while decoding, and writes
`benchmarks/<date>-<variant>.json`. Save each working configuration as a
profile (Config → Saved), so switching between them is one click.

### Caveats

- llama.cpp's Flash-Next support (#27742, b10665+) runs QSA as dense masking.
  The PR says Metal is not explicitly tested, and the n-gram conv state is
  only exact from position 0 until the chunked-prefill fix lands.
- The moe-stream fork's measurements are one author's, on an M5 Pro, at 98k
  context. 262k is untested there.
- A 3-bit body may lower MTP acceptance. The `code` task's decode rate is
  where that shows.
