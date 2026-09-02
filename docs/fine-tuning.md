# Fine-tuning: a folder of documents to a swappable adapter

`scripts/finetune/` turns a directory of source material into a validated SFT
dataset, trains a LoRA on it, and hands the result to
[docs/lora-adapters.md](lora-adapters.md), which covers everything from the
converted `.gguf` onwards.

The pipeline exists because the first adapter built here was built by hand, and
the decisions that mattered most left no trace in the result: a chat template
whose default reasoning effort prepended a paragraph the served model never
sees, a loss mask that had to start after the think block, and eight rows out of
177 that were quietly poisonous. Each of those is now a check that fails loudly
rather than a judgement someone has to remember to make.

## The loop

```
ingest  ->  build  ->  validate  ->  report  ->  [ train  ->  import  ->  compare ]
 format     shape      the gate     one call      the deliberate half
```

The first four are cheap, repeatable and safe. Training is separate on purpose:
it takes a GPU for twenty minutes or more, and nothing before it commits anything.

## Where things live

Code is in the repo; runs are not. A run holds someone's writing, a dataset and
a build product, none of which belong in version control.

```
scripts/finetune/          common.py ingest.py shapes.py corpus.py train.py compare.py
/mnt/LLMs/unsloth/runs/<name>/
  run.yaml                 the recipe — source, shape, base model, hyperparameters
  documents/               normalised text, one file per source
    manifest.jsonl         provenance: path, sha256, converter, words, warnings
  dataset.jsonl            the built rows
  eval.jsonl               held out, never trained on
  reports/                 build-report.json, validate-report.json
  outputs/<name>/          the PEFT adapter
```

`FINETUNE_RUNS_ROOT` moves the root; `FINETUNE_UNSLOTH_PYTHON` moves the
training interpreter.

## 1. Ingest

```bash
python3 scripts/finetune/corpus.py ingest ~/Documents/SomeCorpus --run my-voice
```

| Input | Converter |
|---|---|
| `.md` `.txt` | read directly |
| `.vtt` `.srt` | built in — timecodes dropped, speaker labels kept |
| `.pdf` | pymupdf if the interpreter has it, else `pdftotext -layout` |
| `.docx` `.odt` `.rtf` `.epub` `.html` | pandoc |

It prints which converters this host actually has before it starts, and every
file it could not read is listed with the reason. A PDF with no text layer is
named as probably scanned, with a pointer at the stack's own GLM-OCR service on
port 5002 — which can read it when pdftotext cannot.

Nothing is dropped silently. A corpus that quietly lost a third of its sources
looks exactly like a corpus that was always that size.

## 2. Build

```bash
python3 scripts/finetune/corpus.py build --run my-voice --shape style \
    --system "You are ..."
```

| Shape | What it teaches | Wants |
|---|---|---|
| `style` | to write like the source | prose with headings |
| `dialogue` | to answer like one person | a transcript with speaker labels |
| `qa` | the content, as questions and answers | prose, plus the task model |
| `raw` | a domain's vocabulary, no instructions | anything |

`dialogue` needs `--assistant-speaker`. It refuses an unattributed transcript
rather than guessing: which side of a conversation the model is being taught to
be is the entire content of the dataset.

**Instructions come from headings first.** A heading becomes an instruction
through a deterministic template, with repairs for the accidents that showed up
in the first corpus — a leading "the" that produced "Write the the repair shop",
quote marks that made the slot unreadable, and sentence-length headings that do
not fit a short-title grammar. Only a chunk with no usable heading goes to the
task model on port 8007. Every row records which method wrote it, so a dataset
that quietly went mostly model-written is visible in the report rather than
inferred later.

## 3. Validate — the gate

```bash
python3 scripts/finetune/corpus.py validate --run my-voice --max-seq-length 2048
```

Exits non-zero, and says which rows and why, on:

- an assistant turn that ends mid-clause — a chunker cut it, and training on it
  teaches the model to trail off;
- a reference list, bibliography or section outline;
- a duplicate, matched on an alphanumeric fingerprint so a resubmitted draft
  that differs by a footnote marker is still caught;
- a row over `max_seq_length` once really tokenised.

It also measures the token distribution with the base model's own tokenizer and
suggests the smallest window that fits. On the first corpus that was 1024, not
the 2048 originally assumed — and the difference is activation memory on a card
that has none to spare.

Real tokenisation needs `transformers` from the training venv. Without it the
numbers degrade to a character estimate **and say so**, rather than reporting a
confident guess.

## 4. Report — the one to read

```bash
python3 scripts/finetune/corpus.py report --run my-voice
```

Counts first, then what was dropped and why, the token distribution, the
instruction-source split, one row rendered through the real chat template
exactly as the backend will see it, and sample rows. It is one call because the
alternative is an agent walking the corpus a file at a time.

The rendered prompt is the most valuable line in it. It is what catches a
train/serve mismatch before a training run instead of after.

**How much is enough:** a voice adapter wants 200-500 rows; 1000-3000 is
comfortable. A 6,000-word document yields roughly eight. Under about 100, the
answer is more sources, not more epochs.

## 5. Train

Run under the training interpreter, on the card that is not serving:

```bash
CUDA_VISIBLE_DEVICES=1 HF_HOME=/mnt/LLMs/unsloth/hf-cache \
  /mnt/LLMs/unsloth/unsloth_studio/bin/python scripts/finetune/train.py \
  train --run my-voice
```

Add `--dry-run` to run every check, print the loss mask and train nothing.

Four refusals, all before twenty gigabytes are loaded:

- `CUDA_VISIBLE_DEVICES` unset — GPU 0 serves `llm-a` and has no room;
- the run was never validated;
- the visible card cannot hold the checkpoint, sized against the real 4-bit
  weights (20.8 GiB for the 27B) rather than its GGUF (16.7 GiB) — a third
  understated, which is a run that OOMs twenty minutes in;
- the response marker matches nothing in the rendered rows.

Three defaults are load-bearing rather than tasteful:

- `target_modules` is spelled out — `q,k,v,o_proj` and `gate,up,down_proj`.
  Qwen3.5/3.8's linear-attention layers use `in_proj_*` instead, and llama.cpp's
  GGUF converter permutes exactly those tensors. An adapter that touches them is
  converted through a reorder it never trained under.
- `--response-part` defaults to `</think>\n\n`, so loss starts after the think
  block. A corpus with no reasoning traces would otherwise teach the model to
  emit an empty think block and stop reasoning.
- `--reasoning-effort medium`, matching how the stack serves. The template's own
  default is `xhigh`, which prepends a paragraph to every example.

## 6. Import and compare

```bash
scripts/import-lora-adapter.sh --base <hf-snapshot> \
    /mnt/LLMs/unsloth/runs/my-voice/outputs/my-voice my-voice

python3 scripts/finetune/compare.py --run my-voice --slot llm-a
```

`compare.py` is the only test that separates "trained" from "trained and
actually doing something": the same prompt and seed with the adapter at 0 and at
1, printed side by side. Identical output means it is loaded but inert.

It restores the scales it found, so running it against a live backend leaves
nothing changed. If either side spends its whole budget reasoning and never
reaches an answer it reports the comparison as **inconclusive** — an empty
answer is not evidence about the adapter.

From here, [docs/lora-adapters.md](lora-adapters.md) covers attaching the
adapter to a slot and swapping between fine-tunes without a restart.

## Pitfalls

- **The training checkpoint is not the GGUF.** 20.8 GiB against 16.7 for the
  same 27B: bitsandbytes leaves `lm_head`, embeddings, the vision tower and the
  linear-attention projections in fp16. Sizing a run off the GGUF understates it
  by a third.
- **`build` and `ingest` are stdlib**, so they run under any Python on the box.
  `validate`'s tokenizer half and `train` need the training venv.
- **A converted file that yields no rows** usually means its export format
  slipped past the cleaner, not that it was empty. `report` names those files.
- **The cleaners are per-export-format**, and the artefacts differ by converter
  rather than by subject. Obsidian leaves wikilinks and `[^n]` footnotes; a
  PDF export leaves running headers and publisher boilerplate; a docx export
  leaves bare `####` markers and `^1^` footnotes. Inspect a couple of documents
  before assuming one cleaner covers a new corpus.
