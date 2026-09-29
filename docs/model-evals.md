# Evaluating a model before it replaces one

When a new model or quantization comes out, the question is whether it beats
what LLM A runs now, on this machine, for this work. That means answering four
questions with measurements, not guesses:

- **Speed.** Decode and prefill speed at the depths the work actually reaches.
- **Memory.** How much it holds resident, and how deep a context it serves
  before it runs out.
- **General quality.** Math, code and knowledge.
- **Workload quality.** Long documents, coding passages against a scheme,
  classification, staying grounded, prose. This is the one that decides.

Three suites answer these. A runner serves each candidate in turn, and a report
puts them side by side. Everything is local, and nothing edits
`config/llm-stack.env`.

## One-time setup

The workload suite reads Parquet, so it needs its own small environment:

```bash
python3 -m venv deps/eval-venv
```

```bash
deps/eval-venv/bin/pip install pyarrow huggingface_hub
```

```bash
deps/eval-venv/bin/python scripts/eval-workload.py --prepare
```

The last command fetches the question sets once into `benchmarks/eval-data/`.
Each set is pinned to a dataset revision and sampled with a fixed seed, so every
model gets the same questions, run after run.

## Running an evaluation

1. **Describe the candidates.** Copy `config/eval-candidates.example.json` to
   `config/eval-candidates.json` and add an entry for the new model:

   ```json
   {"name": "my-new-model",
    "settings": {"MODEL_PATH": "models/mlx/My-New-Model", "MEMORY_MODE": "resident",
                 "CACHE_RAM": "2048", "MTPLX_SESSION_TTL": "5"}}
   ```

   - `settings` are the slot's `{PREFIX}_*` keys without the prefix, the same
     ones the Config page writes.
   - `env` sets extra environment variables.
   - `args` adds extra command-line arguments.
   - Relative paths are relative to the stack.
   - Keep the incumbent, `qwen3.8-27b-mtplx-speed`, in the file, so there is
     always a baseline.

2. **Check what would run.** This builds every launch command with the stack's
   own command builder and starts nothing:

   ```bash
   scripts/eval-run.py plan
   ```

   An MTPLX pack should show `mtplx serve`, and a GGUF `llama-server`. A path
   that did not resolve shows up here rather than two hours in.

3. **Free the slot.** The runner borrows LLM B's port (8020) and refuses if
   something it did not start is listening there. A large model needs the
   memory LLM A holds, too, so stop both chat slots first. The runner never
   stops a backend itself.

4. **Run the suites.** The full set takes roughly 45–90 minutes per model:

   ```bash
   scripts/eval-run.py run --only my-new-model,qwen3.8-27b-mtplx-speed
   ```

   `--suites` picks some of `workload,quality,speed`. `--suite-args` passes
   options through, e.g. `--suite-args "--depths 0,8192,32768"` for a shorter
   speed sweep.

5. **Read the report:**

   ```bash
   scripts/eval-report.py
   ```

   Results are under `benchmarks/<model>/`. They are not committed.

## The suites

### Speed (`scripts/bench-offload.py`)

Chat and code prompts at 0, 8k, 32k, 128k and 262k tokens of context. It records:

- decode and prefill tok/s;
- the backend's peak memory: phys_footprint plus resident pages of mapped
  weight files, the same measure the manager's memory panel uses;
- bytes read from the SSD while decoding;
- the worst memory-pressure level seen.

"Code" output mostly copies its input, so it shows what MTP and n-gram
drafting buy. "Chat" shows the floor.

### General quality (`scripts/eval-quality.py`)

- GSM8K math: 100 problems.
- HumanEval code: 164 problems, run under `sandbox-exec` with no network and
  no writes.
- MMLU-Pro knowledge: 140 questions across its subjects.

Thinking is off and decoding is greedy, so the comparison is between weights.
The scores are not comparable with published ones.

### Workload (`scripts/eval-workload.py`)

| Set | Tests | Source |
|---|---|---|
| `longctx` | Synthesis: multiple choice over English documents of 13k–106k tokens | LongBench v2 (40) |
| `oolong` | Coding at scale: classify every line of a 32k–128k document by a given scheme, then count, compare or trace the labels | Oolong-synth (48; official scoring, partial credit on counts) |
| `faith` | Grounding: answer from a context that contradicts common knowledge (60), and say "unknown" when the answer was removed (60) | FaithEval |
| `classify` | 77 intent codes, each defined by an example in the prompt | Banking77 via LongICLBench (100) |
| `prose` | 300-word syntheses at Qwen's recommended sampling; scored for degenerate repetition, texts saved in the result for reading | LongBench v2 documents (5) |

## Reading the results

- **Noise.** With 40–160 questions per set, differences of a few points are
  noise. Treat 5 or more as a signal, and 10 or more on a workload set as a real
  loss.
- **Weight the sets to the work.** Rank them `longctx`, `oolong` and `faith`
  first; then `classify`; then the general suite.
- **Errors (⚠ in the report)** are requests the backend refused or failed, not
  wrong answers.
  - The runner restarts a backend that starts refusing and asks again. A
    residue of errors means the model genuinely could not serve that request,
    usually for memory at long context.
  - Read that together with the speed table's "deepest answered" column.
- **Quantization.** The literature and these results agree on the pattern:
  - Long-context recall and faithfulness erode first.
  - Short classification and prose last.
  - Mixture-of-experts models tolerate low-bit experts much better than dense
    models, if attention, DeltaNet, router and shared experts stay at 8-bit.
  - The failure mode for weak 3-bit builds is repetition, which the `prose`
    set catches.
  - The recipes in `config/forge-recipes/` follow this. `docs/ssd-offload.md`
    has the Flash-Next specifics.

## Known quirks

- **MTPLX memory creep.** MTPLX's memory grows by about 0.2 GiB per distinct
  request, until its guard refuses everything with a 507. Two things keep
  results valid:
  - The suites restart the backend on a 507 and retry.
  - The candidates use a 5-second session TTL (`MTPLX_SESSION_TTL`) so
    finished conversations do not pile up.
  For daily use it matters too: a long unattended batch on MTPLX needs a
  restart every few hundred requests until the cause is fixed.
- **Runtime.** Long-context sets dominate the run time: a 128k-token prompt
  takes 45–60 s to prefill on MTPLX and about 3.5 minutes on the
  expert-streaming build.
