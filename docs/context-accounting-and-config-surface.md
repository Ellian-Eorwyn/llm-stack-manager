# Context accounting and the config surface

Two things this repo got wrong for a long time, what they cost, and what
replaced them.

## 1. `--ctx-size` is a total, not a per-request limit

llama.cpp divides `--ctx-size` by `--parallel` and gives each slot the
quotient. A request is measured against the quotient. The UI showed the total.

The bill arrived as this, on a backend the config form called `262144`:

```
request (155751 tokens) exceeds the available context size (131072 tokens)
```

Nothing was misconfigured. `--ctx-size 262144 --parallel 2` is 131072 tokens per
slot, and 155751 does not fit in 131072. The only place that computed the
quotient was `cache-aware-scheduling.js`, and it did not lead with it.

**Now:** the per-slot figure is shown wherever a total is, and it leads.

| Surface | Where |
| --- | --- |
| Config form | `.per-slot-hint` under each backend's identity panel, live as you type |
| Cache-aware panel | Headline reads *N per slot × slots = total* |
| Services panel | `.card-ctx` on each model service, from `/api/status` → `contexts` |
| Telemetry | `props.n_ctx_per_slot` against `n_ctx_total` |
| Pre-flight | Every priced backend reports `per_slot_context` |
| Aggregate `/v1/models` | Already correct — reports `n_ctx: 131072` |

`backend_context_summary()` in `web/app.py` is the server-side source: it reads
the env rather than the running backend, so the number is there while a service
is stopped — which is when it is being chosen.

## 2. The recommended preset was a constant

`cache-aware-scheduling.js` used to hardcode:

```js
CTX_SIZE: "262144", CACHE_RAM: "8192", CTX_CHECKPOINTS: "32"
```

On the box those numbers were written for, that configuration thrashed. This is
a hybrid-attention model: every context checkpoint carries ~150 MiB of recurrent
state before a single token is stored, so 32 checkpoints × 2 slots wanted
roughly 27 GiB against the 8 GiB budget it was given. Measured over three days:
827 evictions across 1522 slot launches, ~2.4 TiB of prompt cache churned, p90
of 2.0s of pure scheduling delay, on a host already 6.8 GB into swap.

**Now:** `GET /api/backend/budget/recommend` derives the profile from detected
VRAM, host RAM and the model's own GGUF geometry (`budget.recommend`). The
constants survive only as `FALLBACK` in the JS module, for a host that cannot be
measured — and they were lowered to values that do not thrash.

Two properties the recommendation holds, both tested:

- It never recommends past the model's trained context.
- It prices the backend **as it will actually launch** — projector, draft head,
  micro-batch, tensor split — via `base_settings`. Priced without them, this box
  recommended 524288 tokens for a configuration that carries an 885 MiB
  projector worth several gigabytes of encoder working set.

A recommendation that trips the pre-flight is not a recommendation.
`RecommendTest.test_the_recommendation_passes_its_own_verdict` enforces that.

## 3. Pre-flight on save

`POST /api/config` prices the proposed configuration through `budget.evaluate`
before writing it, and returns `409` with the issue list when the prediction says
it cannot allocate. `?force=1` overrides — the model is a prediction and the
operator holds the hardware — but the refusal is the default, so the failure
surfaces at the form rather than in a restart loop.

Warnings never block. They are returned on success and rendered under the save
bar. The ones worth knowing about:

| Code | What it means |
| --- | --- |
| `vram_overcommit` | **error** — predicted peak exceeds the GPU |
| `cache_ram_shortfall` | Checkpoints want more than `--cache-ram`; expect eviction on most requests |
| `swa_full_unsupported` | `--swa-full` on a model with no sliding-window attention. llama-server logs `swa_full is not supported by this model` and ignores it |
| `swa_full_expensive` | `--swa-full` on a model that *does* have sliding-window attention, where it stops the window layers being windows |
| `fit_ctx_without_fit` | `--fit-ctx` set with auto-fit off, so it does nothing |
| `cache_reuse_with_multimodal` | llama-server disables `--cache-reuse` when a projector is loaded |
| `context_above_trained` | Total context above what the model was trained for |

Only backends the update actually touches are priced — reading GGUF metadata
means touching the model file, and a port change should not pay for it.

Applying a **saved profile** reports its pre-flight but does not enforce it.
That path also runs at startup, where refusing would leave the stack with no
configuration at all.

`POST /api/config/preflight` runs the same check without writing.

## 3b. Not every layer holds a KV cache, and not every one that does grows

Three architectures answer "which layers hold KV" three different ways, and the
budget model has to know all three or it is wrong by an order of magnitude.

| Metadata | Meaning | Example |
| --- | --- | --- |
| `recurrent_layer_arr` / `full_attention_interval` | Hybrid attention. Recurrent layers hold fixed state, not KV | Qwen3.6-27B: 48 recurrent, 16 full |
| `attention.sliding_window_pattern` | Interleaved local/global. Window layers hold only their window | Gemma4-31B: 50 window, 10 global |
| neither | Every layer is a full-attention layer | a conventional dense model |

Missing the middle row cost a 7.6× overestimate. Gemma4-31B was predicted to
need 284,145 MiB and refused as unfittable, on a box where it runs in 37,236
MiB. Three compounding mistakes, all in the same direction:

1. **Per-layer head counts collapsed with `max()`.** Gemma publishes
   `head_count_kv` per layer — 16 for the window layers, 4 for the global ones.
   Taking the maximum charged 16 heads to every layer, including the only ten
   that actually scale with context.
2. **Window layers priced at full context.** 50 of 60 layers hold 1024 tokens,
   not 255,998.
3. **`key_length_swa` / `value_length_swa` ignored.** Window layers use 256, not
   512.

The same bug ran through the checkpoint term. A checkpoint saves only state that
cannot be rebuilt by reprocessing the prompt: llama.cpp writes it with
`LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY`, and both `llama_kv_cache_iswa::state_write`
and `llama_memory_hybrid::state_write` skip the base attention cache under that
flag. So a checkpoint is the recurrent state, or the window, and for a plain
full-attention model it does not exist at all — the `n_swa > 0` guard on
`do_checkpoint` in `server-context.cpp` means none are created. Charging the
full-attention KV predicted 254,998 MiB per checkpoint for a backend whose
checkpoints are about 425 MiB.

`--swa-full` deserves its own note. On a model with no sliding-window attention
it is inert. On one that has it, it is the flag that makes the window layers
stop being windows: on Gemma4-31B it takes the KV cache from 11,050 MiB to
roughly 117,000 MiB. `recommend()` used to turn it *on* wherever the model
supported it, which was exactly backwards. It is now off everywhere, for those
two different reasons.

Validated with `budget.py --validate` against the running backend: 37,558 MiB
predicted against 37,236 MiB observed, −0.9%. `LiveSlidingWindowModelTest`
guards it against the real file.

## 4. The dual env-key naming, and how it ends

The backend slots were once named for what they held — `CHAT_DENSE_*` for a
dense model, `CHAT_MOE_*` for a mixture-of-experts one — and before that for the
specific models themselves (`CHAT_MODEL_27B_PATH`). Both schemes described the
contents rather than the slot, so both went stale the moment a slot's model
changed. `CHAT_PRIMARY_*` / `CHAT2_*` name the slot, and are canonical.

The staged path out, so no stage can break a saved config:

1. **Done.** Canonical on write, legacy backfilled on read.
   `normalize_env_keys` fills the canonical key from its legacy twin;
   `normalize_config_updates` rewrites legacy keys to canonical before writing.
   Nothing writes a legacy key any more.
2. **Done.** `GET /api/config/deprecations` names every legacy key still on
   disk, what replaces it, and whether its canonical twin is already present.
   It reports saved profiles separately.
3. **Done, opt-in.** `POST /api/config/deprecations/migrate` rewrites
   `llm-stack.env` onto the canonical names. `update_env_values` already
   collapses a canonical key's legacy aliases onto one line when it writes, so
   migrating is writing each canonical key with the value the configuration
   already resolves to — a rename, not a change.
4. **Not taken.** Once a report comes back empty on a host, drop the legacy
   names from `allowed_config_keys` so they stop being writable at all, keeping
   `LEGACY_ENV_KEY_MAP` for read-side backfill of old profiles.

Step 4 is deliberately left undone: it is only safe once step 3 has run and the
report is empty, and that is an operator action rather than a code change.

**Saved profiles are never rewritten.** They are user data, and a profile
carrying only `CHAT_DENSE_MODEL_PATH` would lose its model if the key were
dropped. Step 1 keeps reading them correctly for as long as they exist.
