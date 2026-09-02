#!/usr/bin/env python3
"""QLoRA a validated run onto a base model.

Run under the training interpreter, not the system one:

    /mnt/LLMs/unsloth/unsloth_studio/bin/python train.py --run <name>

Three defaults here are load-bearing rather than tasteful, and each has a
comment saying which failure it prevents. They were established by the
academic-voice run on 2026-09-02.

This refuses before loading twenty gigabytes rather than warning and
proceeding: a warning the operator reads ten minutes later, after the GPU is
already committed, is not a control.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
from common import RunError, open_run, read_jsonl  # noqa: E402

#: LoRA goes on the standard projections only. Qwen3.5/3.8's linear-attention
#: layers use in_proj_a/in_proj_b/in_proj_qkv/out_proj instead, and llama.cpp's
#: GGUF converter permutes exactly those tensors (V heads grouped -> tiled).
#: An adapter that touches them is converted through a reorder it never trained
#: under. Naming these seven also keeps it off the modules bitsandbytes leaves
#: in fp16, which costs nothing for a style adapter.
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

#: Loss starts after the think block closes, not at the assistant turn. At
#: serve time the model is primed with `<|im_start|>assistant\n<think>\n`; a
#: corpus with no reasoning traces would otherwise teach it to emit an empty
#: think block and stop reasoning.
DEFAULT_RESPONSE_PART = common.DEFAULT_RESPONSE_PART
DEFAULT_INSTRUCTION_PART = "<|im_start|>user\n"

#: Bytes per parameter is the wrong model for this: half the 4-bit checkpoint
#: is fp16 that bitsandbytes will not quantise (lm_head, embeddings, the vision
#: tower, the linear-attention projections). Measure the file instead.
VRAM_HEADROOM_GIB = 1.5

#: Floor used when the checkpoint's real size cannot be read. No 4-bit base
#: worth fine-tuning loads into less than this, so a card below it is occupied.
MIN_FREE_GIB = 6.0


def cmd_train(args) -> int:
    run = open_run(args.run)
    run.require("shape")
    rows = read_jsonl(run.dataset)
    if not rows:
        return common.fail(f"run {run.name!r} has no dataset; "
                           f"run `corpus.py build` then `corpus.py validate`")
    if not run.report_path("validate").is_file():
        return common.fail(
            f"run {run.name!r} has not been validated. `corpus.py validate --run "
            f"{run.name}` first — it is the step that catches truncated rows, "
            f"bibliographies and over-length examples.")

    base_model = args.base_model or run.recipe.get("base_model") or common.DEFAULT_BASE_MODEL
    max_seq_length = args.max_seq_length or run.recipe.get("suggested_max_seq_length") or 1024
    response_part = args.response_part

    device = os.environ.get("CUDA_VISIBLE_DEVICES")
    if device is None:
        return common.fail(
            "CUDA_VISIBLE_DEVICES is not set. Set it explicitly — GPU 0 serves "
            "llm-a and has no room for a training run. Use "
            "CUDA_VISIBLE_DEVICES=1.")

    print(f"# Training {run.name}")
    print(f"  base            {base_model}")
    print(f"  rows            {len(rows)}")
    print(f"  max_seq_length  {max_seq_length}")
    print(f"  CUDA devices    {device}")
    print()

    problem = _check_vram(base_model, args.skip_vram_check)
    if problem:
        return common.fail(problem)

    from unsloth import FastLanguageModel  # noqa: PLC0415

    import torch  # noqa: PLC0415
    from datasets import Dataset  # noqa: PLC0415
    from trl import SFTConfig, SFTTrainer  # noqa: PLC0415
    from unsloth.chat_templates import train_on_responses_only  # noqa: PLC0415

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model, max_seq_length=max_seq_length,
        load_in_4bit=True, full_finetuning=False)

    texts = _render(tokenizer, rows, args.reasoning_effort)
    missing = sum(1 for t in texts if response_part not in t)
    if missing:
        return common.fail(
            f"the response marker {response_part!r} appears in none of "
            f"{missing} of {len(texts)} rendered rows. Loss would cover the whole "
            f"assistant turn, which for a corpus with no reasoning traces teaches "
            f"the model to stop reasoning. Pass --response-part to match this "
            f"template.")

    model = FastLanguageModel.get_peft_model(
        model, r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0, bias="none",
        target_modules=TARGET_MODULES,
        use_gradient_checkpointing="unsloth", random_state=args.seed)

    attached = sorted({name.split(".")[-3] for name, _ in model.named_modules()
                       if name.endswith("lora_A.default")})
    if not attached:
        return common.fail("no LoRA adapters attached — target_modules matched "
                           "nothing in this architecture")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  LoRA on         {', '.join(attached)}")
    print(f"  trainable       {trainable:,}")

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=Dataset.from_dict({"text": texts}).shuffle(seed=args.seed),
        args=SFTConfig(
            dataset_text_field="text", max_length=max_seq_length, packing=False,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs, learning_rate=args.lr,
            warmup_steps=5, lr_scheduler_type="cosine", optim="adamw_8bit",
            weight_decay=0.01, logging_steps=1, save_strategy="no",
            seed=args.seed, output_dir=str(run.outputs / f"{run.name}-trainer"),
            report_to="none"),
    )
    trainer = train_on_responses_only(
        trainer, instruction_part=args.instruction_part, response_part=response_part)

    sample = trainer.train_dataset[0]
    masked = sum(1 for label in sample["labels"] if label == -100)
    trained = len(sample["labels"]) - masked
    print(f"  loss mask       {masked} masked / {trained} trained tokens")
    if trained == 0:
        return common.fail("the loss mask covers every token — nothing would train")

    if args.dry_run:
        print("Dry run: everything checked, stopping before training.")
        return 0

    free, total = torch.cuda.mem_get_info()
    print(f"  VRAM before     {(total - free) / 2**30:.2f} / {total / 2**30:.2f} GiB")
    print()

    stats = trainer.train()

    adapter = run.outputs / run.name
    adapter.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter))
    tokenizer.save_pretrained(str(adapter))

    run.update(base_model=base_model, max_seq_length=max_seq_length,
               rank=args.rank, epochs=args.epochs, lr=args.lr,
               trained_loss=round(stats.metrics["train_loss"], 4),
               adapter=str(adapter))

    print()
    print(f"  peak VRAM       {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB")
    print(f"  runtime         {stats.metrics['train_runtime']:.0f}s")
    print(f"  final loss      {stats.metrics['train_loss']:.4f}")
    print(f"Adapter saved to {adapter}")
    print(f"Convert it with: scripts/import-lora-adapter.sh --base <hf-snapshot> "
          f"{adapter} {run.name}")
    return 0


def _render(tokenizer, rows, effort: str) -> list[str]:
    """Rows through the real chat template, at the effort the backend serves.

    The template's own default is xhigh, which prepends a paragraph to every
    system message that llm-stack never sends. Training on that teaches a
    prompt the model will not see.
    """
    texts = []
    for row in rows:
        if "text" in row:
            texts.append(row["text"])
            continue
        texts.append(tokenizer.apply_chat_template(
            row["messages"], tokenize=False, reasoning_effort=effort))
    return texts


def _check_vram(base_model: str, skip: bool) -> str | None:
    """Refuse a run the visible card cannot hold.

    Sized against the checkpoint on disk, because a 27B 4-bit training
    checkpoint is 20.8 GiB while its Q4 GGUF is 16.7 -- sizing off the GGUF
    understates it by a third and OOMs twenty minutes in.
    """
    if skip:
        return None
    try:
        import torch  # noqa: PLC0415
        if not torch.cuda.is_available():
            return "no CUDA device is visible"
        free, total = torch.cuda.mem_get_info()
    except Exception as exc:
        return f"could not read GPU memory: {exc}"

    size = _checkpoint_bytes(base_model)
    if size is None:
        # The checkpoint is not in the local cache, so its real size is unknown.
        # Still refuse an obviously occupied card: skipping the check here is
        # how a run reaches the loader and dies in bitsandbytes twenty seconds
        # later with a message about device_map that names no cause.
        if free / 2**30 < MIN_FREE_GIB:
            return (f"the visible device has only {free / 2**30:.1f} GiB free, and "
                    f"no 4-bit base worth training fits in that. {base_model} is "
                    f"not in the local cache so its exact size is unknown — free "
                    f"the card, or point CUDA_VISIBLE_DEVICES at an idle one.")
        print(f"  vram check      {base_model} is not cached; "
              f"checked against the {MIN_FREE_GIB} GiB floor only "
              f"({free / 2**30:.1f} GiB free)")
        return None
    need = size / 2**30 + VRAM_HEADROOM_GIB
    if free / 2**30 < need:
        return (f"{base_model} needs about {need:.1f} GiB "
                f"({size / 2**30:.1f} GiB of weights plus {VRAM_HEADROOM_GIB} for "
                f"activations and optimiser state) and the visible device has "
                f"{free / 2**30:.1f} GiB free. Free the card, or point "
                f"CUDA_VISIBLE_DEVICES at one that is idle.")
    return None


def _checkpoint_bytes(base_model: str) -> int | None:
    try:
        from huggingface_hub import snapshot_download  # noqa: PLC0415
        path = Path(snapshot_download(base_model, local_files_only=True))
    except Exception:
        return None
    weights = list(path.glob("*.safetensors")) or list(path.glob("*.bin"))
    return sum(f.stat().st_size for f in weights) or None


def cmd_status(args) -> int:
    """What each run has reached. Always prints something."""
    names = [args.run] if args.run else common.list_runs()
    if not names:
        print(f"No runs under {common.RUNS_ROOT}.")
        return 0
    print(f"{'run':28} {'shape':10} {'rows':>6}  state")
    for name in names:
        try:
            run = open_run(name)
        except RunError as exc:
            print(f"{name:28} UNAVAILABLE: {exc}")
            continue
        recipe = run.recipe
        if recipe.get("adapter") and Path(recipe["adapter"]).is_dir():
            state = f"trained, loss {recipe.get('trained_loss', '?')}"
        elif run.report_path("validate").is_file():
            state = "validated — ready to train"
        elif run.dataset.is_file():
            state = "built — not validated"
        elif run.manifest.is_file():
            state = "ingested — not built"
        else:
            state = "empty"
        print(f"{name:28} {str(recipe.get('shape', '-')):10} "
              f"{recipe.get('rows', 0):>6}  {state}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="QLoRA the run's dataset")
    p.add_argument("--run", required=True)
    p.add_argument("--base-model")
    p.add_argument("--max-seq-length", type=int)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--reasoning-effort", default=common.DEFAULT_REASONING_EFFORT,
                   help="must match how the backend serves this model")
    p.add_argument("--response-part", default=DEFAULT_RESPONSE_PART)
    p.add_argument("--instruction-part", default=DEFAULT_INSTRUCTION_PART)
    p.add_argument("--dry-run", action="store_true",
                   help="run every check, print the loss mask, train nothing")
    p.add_argument("--skip-vram-check", action="store_true")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("status", help="what each run has reached")
    p.add_argument("--run")
    p.set_defaults(func=cmd_status)

    args = parser.parse_args()
    try:
        return args.func(args)
    except RunError as exc:
        return common.fail(str(exc))


if __name__ == "__main__":
    sys.exit(main())
