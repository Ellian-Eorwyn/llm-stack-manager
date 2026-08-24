# GPU split modes

`Split Mode` decides how a model's weights are placed across the visible GPUs.
llama-server advertises four values; **this build can act on two of them**, and
the third is conditional on the model. The launcher checks before it execs and
falls back to `layer` with a reason in the journal, because both refusals
otherwise surface as a core dump in a systemd restart loop rather than as an
error message.

| Mode | Status here |
| --- | --- |
| `none` | Works. Whole model on the GPU named by `*_MAIN_GPU`. |
| `layer` | Works. Splits by layer across GPUs in the `*_TENSOR_SPLIT` ratio. The dependable multi-GPU choice, and the default. |
| `row` | **Cannot work.** Not offered in the UI; refused by the launcher. |
| `tensor` | Experimental, and only for non-hybrid architectures. |

## Why `row` cannot work

Upstream commit `74976e1ae` ("CUDA: remove -sm row, refactor cuBLAS") deleted
the CUDA split-buffer implementation. `make_gpu_buft_list`
(`src/llama-model.cpp`) asks the backend for `ggml_backend_split_buffer_type`
and throws when it is absent:

```
llama_model_load: error loading model: device CUDA0 does not support split buffers
```

`ggml_backend_cuda_reg_get_proc_address` no longer answers that name — only the
SYCL backend still implements it — and `nm -D libggml-cuda.so` confirms no such
symbol is built. The throw happens on the first device, so this fails even with
one GPU, with any model, whatever `--tensor-split` and `--main-gpu` say. There
is no configuration that recovers it; a SYCL build would be a different stack.

## Why `tensor` refuses hybrid models

`tensor` is genuine tensor parallelism: it shards weights *and* KV, and folds
every visible GPU into a single "Meta device". llama.cpp gates it on
architecture in `llm_arch_supports_sm_tensor` (`src/llama-arch.cpp`), which is a
blacklist.

Cross-referenced against `llm_arch_is_hybrid` in the same file, every hybrid
architecture is blacklisted — `JAMBA`, `FALCON_H1`, `PLAMO2`, `GRANITE_HYBRID`,
`LFM2`, `NEMOTRON_H`, `KIMI_LINEAR`, `DEEPSEEK4` — **except `QWEN3NEXT`,
`QWEN35` and `QWEN35MOE`**, which were added to `llm_arch_is_hybrid` and missed
in the blacklist. Those three therefore pass a gate they should have failed and
abort later, in the meta backend, during the warmup decode:

```
ggml/src/ggml.c: GGML_ASSERT(obj_new) failed
ggml_new_object: not enough space in the context's memory pool (needed 754032, available 753664)
  #4 ggml_backend_meta_buffer_init_tensor_impl(...)
  #5 ggml_gallocr_alloc_graph
  #8 llama_context::decode
```

The shortfall is one `ggml_tensor_overhead()` — the pool is sized for 2048
tensors (`compute_headroom = 16` in `ggml/src/ggml-backend-meta.cpp`) and a
2049th is asked for. That is a symptom of the missing blacklist entry, not a
tuning problem.

**Every `Qwen3.5-*` and `Qwen3.8-*` GGUF in `models/` reports
`general.architecture = qwen35`**, so the whole family — including the primary
backend's model — is excluded. `gemma4`, `llama` and `glm4` models are not.

Check any model with:

```
python3 web/budget.py --model models/<file>.gguf --field geometry.architecture
python3 web/budget.py --model models/<file>.gguf --field geometry.is_hybrid
```

## What `tensor` changes about the other settings

Because the GPUs become one device:

- **`*_TENSOR_SPLIT` is dropped.** There is no ratio between devices that no
  longer exist separately.
- **`*_MAIN_GPU` is dropped.** There is no main GPU to choose.
- **Auto-fit does nothing.** `common/fit.cpp` refuses `SPLIT_MODE_TENSOR` and
  downgrades to a log warning, so context size and layer counts are used exactly
  as configured — nothing is trimmed to fit VRAM.
- **Flash attention is mandatory.** `llama_context` errors out without it, so
  the launcher refuses `tensor` when flash attention is off.

## Is `tensor` faster?

Unmeasured on this box, and not obviously yes. Tensor parallelism computes on
both cards at once instead of pipelining, which helps single-stream latency, but
it all-reduces after every layer. On this machine that traffic crosses PCIe Gen3
x8 via host memory: `nvidia-smi topo -p2p rw` reports peer access unsupported
between the two 3090s, there is no NVLink, and the build logs

```
NCCL not compiled in; falling back to internal AllReduce.
Recompile with -DGGML_CUDA_NCCL=ON for best multi-GPU performance.
```

Whether the parallelism outruns the interconnect is an empirical question, and
it cannot be asked of the primary model at all while `qwen35` is excluded.

## Where the checks live

- `resolve_split_opts` in `scripts/lib/backend-preflight.sh` — vets the mode
  against the model and builds the flags. All nine llama.cpp launchers use it.
- `_resolve_split_mode` in `scripts/render-models-ini.py` — the same two
  refusals for pooled router models, which never touch the start scripts.
- `LLAMA_SPLIT_MODE_OPTIONS` in `web/config_fields.py` — what the UI offers.
- `_reject_unsupported_split_modes` in `web/config_env.py` — stops an
  unsupported value being persisted by the config API.
