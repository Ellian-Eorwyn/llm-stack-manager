# LoRA adapters

A LoRA adapter is a small set of low-rank deltas trained on top of a frozen base
model. The stack applies them at runtime rather than merging them into the
weights, which is what makes swapping between fine-tunes cheap:

* one base GGUF serves every fine-tune of it — the 17 GiB file stays loaded and
  each adapter costs tens of megabytes on top;
* several adapters can be resident at once, and switching between them is a
  scale change on a model that is already in VRAM, not a reload;
* merging would mean producing a ~54 GB fp16 checkpoint and requantising it,
  which needs more RAM than this class of machine has.

Adapters are available on the `llm-a`, `llm-b` and `task` slots.

## The loop

```
train  ->  export adapter  ->  convert to GGUF  ->  models/loras/  ->  attach  ->  swap
                                                                    (config)   (services)
```

### 1. Train and export

Any PEFT-compatible trainer works; the stack only cares about the output being a
directory with `adapter_config.json` and `adapter_model.safetensors`. Export the
**adapter alone**, not a merged model.

Two things to get right, because they are invisible until inference is wrong:

* **Render the chat template the way the backend serves it.** If the template
  takes a `reasoning_effort`, train with the value the slot runs
  (`LLM_A_REASONING_EFFORT`); a mismatch means every training example carried a
  system preamble the served model never sees.
* **Target only the standard projections** — `q_proj`, `k_proj`, `v_proj`,
  `o_proj`, `gate_proj`, `up_proj`, `down_proj`. On hybrid-attention models such
  as the Qwen3.5/3.8 family the linear-attention layers use `in_proj_a`,
  `in_proj_b`, `in_proj_qkv` and `out_proj` instead, and llama.cpp's converter
  permutes exactly those tensors when it writes the GGUF (V heads grouped ->
  tiled). An adapter that touches them is converted through a reorder it was not
  trained under.

### 2. Convert

```bash
scripts/import-lora-adapter.sh --base <hf-snapshot-dir> <adapter-dir> [name]
```

The converter needs the base model's `config.json`, not its weights, to know the
tensor layout it is writing against. Use `deps/llama.cpp/convert_lora_to_gguf.py`
— the same tree `llama-server` is built from — because an adapter has to be
written by the converter that matches the loader.

The script verifies the result really is an adapter (`general.type = adapter`,
`adapter.type = lora`) before leaving it in `models/loras/`. A converted file
that is not one would be offered by the config UI and then refused by
llama-server after exec, which reads as a restart loop rather than an error.

### 3. Attach

**Config -> the slot -> LoRA Adapters.**

| key | meaning |
|---|---|
| `*_LORA_PATHS` | comma-separated; a bare name resolves under `models/loras/`, an absolute path is used as given |
| `*_LORA_SCALES` | positional, one per adapter; a missing entry means 1.0 |
| `*_LORA_INIT_WITHOUT_APPLY` | `on` (default): load everything at scale 0. `off`: apply at the configured scales from startup |

These are launch flags, so changing them restarts the slot.

`*_LORA_INIT_WITHOUT_APPLY` defaults to `on` deliberately. Adapters passed with
`--lora-scaled` all start applied and **stack**, so a slot with two fine-tunes
configured would serve both blended together on its first request. With it on
they start at zero and you choose. The configured scale then means "the strength
to restore when this one is switched on", not "the strength at boot".

**How that is actually implemented, and why it is not just the flag.** The
llama.cpp server README says adapters loaded with `--lora-init-without-apply`
"start at scale 0.0". In llama-server they do not. `common.cpp` only skips the
one-time `common_set_adapter_lora` at startup; `server-context.cpp` then sets
`slot.lora = params_base.lora_adapters` for every task and re-applies it per
batch, so the first request restores each adapter's configured scale. Measured
on build b10434: a backend launched with the flag and `:1.0` answered its very
first completion in the adapter's voice.

So the launcher emits `--lora-scaled <path>:0` when preloading is on — a
configured zero is what actually holds — and passes the flag as well, since
skipping the startup application is real and harmless. With `:0` the same
backend answers byte-identically to the base model until a scale is posted.

### 4. Swap

**Services -> the backend's card.** A slot with adapters loaded grows an
*Adapters* block, one row per adapter: a dot to switch it on or off, and a 0–1
slider for anything finer. The dot raises the adapter to its **configured**
scale, which is why `/api/backends/<slot>/lora` returns `configured_scale`
beside the live one — with preloading on, everything boots at 0 and the live
scale cannot say how strongly it was meant to apply.

Either control POSTs to `/api/backends/<slot>/lora`, which llama-server applies
to the resident model — no restart, and the next request uses the new blend. A
row is dim at 0 and lights up when applied, so the card answers "which fine-tune
am I serving" at a glance.

Scales are not exclusive. Two adapters at 0.5 each is a legitimate blend, and
dialling one down to ~0.6 is the usual remedy if a style adapter is crowding out
the base model's reasoning.

## Under the hood

| piece | where |
|---|---|
| flag emission | `web/backends/options.py` — `Lora`, and `lora_pairs` shared with the router |
| slot wiring | `web/backends/slots.py` — last entry of `_large_model_tail()` |
| fields | `web/config_fields.py` — `CHAT_LORA_*` (becomes `LLM_A_*`), `LLM_B_LORA_*`, `TASK_LORA_*` |
| discovery | `web/models.py` — `list_lora_adapters()`, by GGUF metadata rather than filename |
| live control | `web/routes/models.py` — `/api/backends/<slot>/lora`; `web/static/js/lora.js` |
| pooled models | `scripts/render-models-ini.py` — one computed `lora-scaled` list |

`Lora` is emitted last in the tail so that a slot with no adapters configured
produces exactly the command line it produced before adapters existed — which is
why `tests/launcher-argv.golden.json` needed no regeneration.

## When it does not work

* **Adapter listed but no `--lora-scaled` in the journal.** The file was not
  found or was rejected; the launcher says which and why. Those notes reach the
  journal only since the `said` list stopped being discarded — see
  `SaidPropagationTests` in `tests/test_backends.py`.
* **Adapter loads but changes nothing.** Check the scale is above 0 — preloading
  is on by default, so a freshly restarted backend starts with everything at 0.
  `GET /api/backends/<slot>/lora` reports the live scales.
* **`llama-server` refuses the adapter at load.** It was converted against a
  different base than the GGUF being served. The adapter's `general.architecture`
  must match the model's.
* **A path with a colon in it.** `--lora-scaled` splits on the last colon, so
  such a path cannot be expressed; the launcher refuses it rather than passing a
  mangled scale.
