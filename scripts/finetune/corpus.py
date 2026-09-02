#!/usr/bin/env python3
"""Build a fine-tuning dataset from a folder of documents.

    corpus.py ingest <source-dir> --run <name>
    corpus.py build  --run <name> --shape style|dialogue|qa|raw
    corpus.py validate --run <name>
    corpus.py report --run <name>
    corpus.py sample --run <name> [--n 3]

`report` is the one to read: a single call that prints everything a decision
needs, counts first. The stages are separate because each is a place a corpus
can be wrong in a different way, and a single command would hide which.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

import common
import ingest as ingest_mod
import shapes
from common import RunError, open_run, read_jsonl, write_jsonl

#: The small model on the stack. Used only to caption chunks with no heading,
#: and to write questions for the `qa` shape.
TASK_ENDPOINT = "http://127.0.0.1:8007/v1/chat/completions"
TASK_MODEL = "task"

DEFAULT_SYSTEM = ("You are a careful writer working in the register and voice of "
                  "the source material.")

#: An assistant turn ending anywhere else was cut mid-clause by a chunker.
TERMINALS = (".", "!", "?", '"', "”", ")", ":", "’", "'")


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

def cmd_ingest(args) -> int:
    source = Path(args.source).expanduser().resolve()
    if not source.exists():
        return common.fail(f"no such source: {source}")

    run = open_run(args.run, create=True)
    run.documents.mkdir(parents=True, exist_ok=True)

    print("Converters available on this host:")
    for line in ingest_mod.describe_converters():
        print(line)
    print()

    paths = ingest_mod.walk(source, recursive=not args.no_recursive)
    candidates = [p for p in paths if p.suffix.lower() in ingest_mod.SUPPORTED]
    ignored = len(paths) - len(candidates)

    records, kept = [], 0
    for path in candidates:
        doc = ingest_mod.convert(path)
        record = doc.record()
        if doc.ok:
            name = _document_name(path, source)
            (run.documents / name).write_text(doc.text)
            record["stored"] = name
            kept += 1
        records.append(record)

    write_jsonl(run.manifest, records)
    run.update(source=str(source), documents=kept,
               ingested_files=len(candidates))

    skipped = [r for r in records if r["skipped"]]
    print(f"Counts")
    print(f"  files seen         {len(paths)}")
    print(f"  candidates         {len(candidates)}   ({ignored} ignored by extension)")
    print(f"  converted          {kept}")
    print(f"  skipped            {len(skipped)}")
    print(f"  words              {sum(r['words'] for r in records if not r['skipped']):,}")
    print()
    if skipped:
        print("Skipped, with reasons — nothing here was dropped silently:")
        for record in skipped:
            print(f"  {record['name']}: {record['skipped']}")
        print()
    by_converter = Counter(r["converter"] for r in records if not r["skipped"])
    if by_converter:
        print("By converter: " + ", ".join(f"{k} {v}" for k, v in by_converter.most_common()))
    print(f"Ingested into {run.documents}")
    return 0


def _document_name(path: Path, source: Path) -> str:
    """A flat, unique, readable filename for a nested source path."""
    try:
        relative = path.relative_to(source)
    except ValueError:
        relative = Path(path.name)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", str(relative.with_suffix("")))
    return f"{stem}.txt"


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def cmd_build(args) -> int:
    run = open_run(args.run)
    run.require("source")
    documents = _load_documents(run)
    if not documents:
        return common.fail(f"run {run.name!r} has no ingested documents; "
                           f"run `corpus.py ingest` first")

    system = args.system or run.recipe.get("system") or DEFAULT_SYSTEM
    shape = args.shape or run.recipe.get("shape")
    if shape not in common.SHAPES:
        return common.fail(f"--shape must be one of {', '.join(common.SHAPES)}")

    caption_state = {"asked": 0, "failed": 0, "unavailable": ""}

    def caption(chunk: str) -> str | None:
        return _ask_task_model(
            "Write one short instruction that would prompt a writer to produce the "
            "passage below. Reply with the instruction only.\n\n" + chunk[:1500],
            caption_state)

    def ask_question(chunk: str, heading: str) -> str | None:
        return _ask_task_model(
            "Write one question that the passage below answers. The question must be "
            "answerable from the passage alone. Reply with the question only.\n\n"
            + chunk[:1500], caption_state)

    if shape == "style":
        result = shapes.build_style(documents, system, args.min_words, args.max_words,
                                    args.min_prose_ratio, args.seed, caption=caption)
    elif shape == "raw":
        result = shapes.build_raw(documents, args.min_words, args.max_words,
                                  args.min_prose_ratio)
    elif shape == "dialogue":
        if not args.assistant_speaker:
            return common.fail(
                "--assistant-speaker is required for --shape dialogue: which side "
                "of the conversation the model is being taught to be is the whole "
                "content of the dataset, and guessing it would be a coin flip")
        result = shapes.build_dialogue(documents, system, args.assistant_speaker,
                                       args.max_words, args.seed)
    else:  # qa
        probe = _ask_task_model("Reply with the word ready.", caption_state)
        if probe is None:
            return common.fail(
                f"--shape qa needs the task model on {TASK_ENDPOINT} and it is not "
                f"answering ({caption_state['unavailable']}). Start the `task` "
                f"service, or build a different shape.")
        result = shapes.build_qa(documents, system, args.min_words, args.max_words,
                                 args.min_prose_ratio, args.seed, ask_question)

    if not result.rows:
        for note in result.notes:
            print(f"  {note}")
        return common.fail(f"the {shape} shape produced no rows from "
                           f"{len(documents)} documents")

    written = write_jsonl(run.dataset, result.rows)
    run.update(shape=shape, system=system, rows=written,
               min_words=args.min_words, max_words=args.max_words,
               seed=args.seed)
    (run.reports / "build-report.json").write_text(json.dumps({
        "shape": shape, "rows": written,
        "dropped": dict(result.dropped), "methods": dict(result.methods),
        "per_source": dict(result.per_source), "notes": result.notes,
        "task_model": caption_state,
    }, indent=2))

    print("Counts")
    print(f"  documents          {len(documents)}")
    print(f"  rows built         {written}")
    print(f"  dropped            {sum(result.dropped.values())}")
    print()
    if result.dropped:
        print("Dropped, by reason:")
        for reason, count in result.dropped.most_common():
            print(f"  {count:5d}  {reason}")
        print()
    print("Instruction source: " +
          ", ".join(f"{k} {v}" for k, v in result.methods.most_common()))
    if caption_state["unavailable"]:
        print(f"UNAVAILABLE: task model — {caption_state['unavailable']}")
    for note in result.notes:
        print(f"  {note}")
    print(f"Wrote {run.dataset}")
    return 0


def _load_documents(run) -> list[dict]:
    documents = []
    for record in read_jsonl(run.manifest):
        stored = record.get("stored")
        if not stored:
            continue
        path = run.documents / stored
        if path.is_file():
            documents.append({"name": record["name"], "text": path.read_text()})
    return documents


def _ask_task_model(prompt: str, state: dict) -> str | None:
    """One completion from the local task model, or None with a reason kept.

    Failures are counted rather than raised: a backend that goes away mid-build
    should cost the rows it was asked for, not the whole run.
    """
    if state.get("unavailable"):
        return None
    state["asked"] += 1
    body = json.dumps({
        "model": TASK_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7, "max_tokens": 120,
    }).encode()
    request = urllib.request.Request(
        TASK_ENDPOINT, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read())
    except (urllib.error.URLError, OSError, ValueError) as exc:
        state["failed"] += 1
        if state["failed"] >= 3 and not state.get("unavailable"):
            state["unavailable"] = f"{type(exc).__name__}: {exc}"
        return None
    text = (payload.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    text = re.sub(r"^[\"'\s]+|[\"'\s]+$", "", text.strip().split("\n")[0])
    return text or None


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

def cmd_validate(args) -> int:
    run = open_run(args.run)
    rows = read_jsonl(run.dataset)
    if not rows:
        return common.fail(f"run {run.name!r} has no dataset; run `corpus.py build`")

    kept, refused = [], Counter()
    examples: dict[str, str] = {}
    seen: set[str] = set()

    for row in rows:
        assistant = _assistant_text(row)
        reason = _refuse(assistant, seen)
        if reason:
            refused[reason] += 1
            examples.setdefault(reason, assistant[-110:].replace("\n", " "))
            continue
        seen.add(shapes.normalize_sig(assistant))
        kept.append(row)

    smells = [smell for smell in shapes.GRAMMAR_SMELLS
              if any(smell in _user_text(row).lower() for row in kept)]

    tokens, tokenizer_note = _token_lengths(run, kept, args.base_model)
    over = [t for t in tokens if t > args.max_seq_length] if tokens else []

    holdout = min(args.holdout, max(0, len(kept) // 10))
    train_rows = kept[holdout:] if holdout else kept
    eval_rows = kept[:holdout]

    report = {
        "rows_in": len(rows), "rows_kept": len(kept),
        "refused": dict(refused), "grammar_smells": smells,
        "tokens": _describe(tokens), "tokenizer": tokenizer_note,
        "over_max_seq_length": len(over), "max_seq_length": args.max_seq_length,
        "train_rows": len(train_rows), "eval_rows": len(eval_rows),
    }
    (run.reports / "validate-report.json").write_text(json.dumps(report, indent=2))

    print("Counts")
    print(f"  rows in            {len(rows)}")
    print(f"  rows kept          {len(kept)}")
    print(f"  refused            {sum(refused.values())}")
    print(f"  over {args.max_seq_length} tokens    {len(over)}")
    print()
    if refused:
        print("Refused, by reason:")
        for reason, count in refused.most_common():
            print(f"  {count:5d}  {reason}")
            print(f"         e.g. ...{examples[reason]}")
        print()
    if tokens:
        print(f"Tokens ({tokenizer_note}): "
              f"min {min(tokens)}  p50 {int(statistics.median(tokens))}  "
              f"p90 {sorted(tokens)[int(len(tokens) * 0.9)]}  max {max(tokens)}")
    else:
        print(f"UNAVAILABLE: token lengths — {tokenizer_note}")
    if smells:
        print(f"Instruction grammar problems: {', '.join(repr(s) for s in smells)}")

    if not kept:
        return common.fail("validation kept no rows")

    write_jsonl(run.dataset, train_rows)
    if eval_rows:
        write_jsonl(run.evalset, eval_rows)
    run.update(rows=len(train_rows), eval_rows=len(eval_rows),
               max_seq_length=args.max_seq_length,
               suggested_max_seq_length=_suggest_length(tokens, args.max_seq_length),
               base_model=args.base_model)

    print(f"Split: {len(train_rows)} train, {len(eval_rows)} held out")
    if over:
        return common.fail(
            f"{len(over)} rows exceed max_seq_length {args.max_seq_length}; "
            f"rebuild with a smaller --max-words")
    print("Validation passed.")
    return 0


def _assistant_text(row: dict) -> str:
    if "text" in row:
        return row["text"]
    for message in reversed(row.get("messages", [])):
        if message.get("role") == "assistant":
            return message.get("content", "")
    return ""


def _user_text(row: dict) -> str:
    for message in row.get("messages", []):
        if message.get("role") == "user":
            return message.get("content", "")
    return ""


def _refuse(assistant: str, seen: set[str]) -> str | None:
    text = assistant.strip()
    if not text:
        return "empty assistant turn"
    if not text.endswith(TERMINALS):
        return "assistant turn truncated mid-sentence"
    if shapes.PSEUDO_HEADING.match(text.split("\n")[0].strip()):
        return "reference list or notes section"
    lines = [l for l in text.split("\n") if l.strip()]
    if lines and sum(1 for l in lines if shapes.NUM_REF.match(l)) / len(lines) >= 0.5:
        return "reference list or notes section"
    if len(re.findall(r"\*Section \d", text)) >= 3:
        return "section outline, not prose"
    italics = len(re.findall(r"\*[^*\n]{5,}\*", text))
    years = len(re.findall(r"\b(?:19|20)\d\d\b", text))
    if italics >= 20 and years >= 20:
        return "bibliography"
    if shapes.normalize_sig(text) in seen:
        return "duplicate of an earlier row"
    return None


def _token_lengths(run, rows, base_model: str) -> tuple[list[int], str]:
    """Real token counts, or a character estimate that says it is one."""
    python = common.unsloth_python()
    if python is None:
        return ([len(_assistant_text(r)) // 4 for r in rows],
                "characters/4 — estimated, the training venv was not found")
    payload = json.dumps({"model": base_model,
                          "rows": [r.get("messages") or r.get("text") for r in rows]})
    code, out, err = _pipe(python, _TOKENIZE_SCRIPT, payload)
    if code != 0:
        return ([len(_assistant_text(r)) // 4 for r in rows],
                f"characters/4 — estimated; tokenizer failed: {err.strip()[:120]}")
    try:
        return json.loads(out)["lengths"], f"{base_model} tokenizer"
    except (ValueError, KeyError):
        return ([len(_assistant_text(r)) // 4 for r in rows],
                "characters/4 — estimated; tokenizer returned nothing usable")


def _pipe(python: Path, script: str, payload: str) -> tuple[int, str, str]:
    import subprocess  # noqa: PLC0415
    try:
        done = subprocess.run([str(python), "-c", script], input=payload,
                              capture_output=True, text=True, timeout=600)
        return done.returncode, done.stdout, done.stderr
    except Exception as exc:
        return 1, "", str(exc)


_TOKENIZE_SCRIPT = r"""
import json, sys
data = json.load(sys.stdin)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(data["model"])
lengths = []
for row in data["rows"]:
    if isinstance(row, str):
        lengths.append(len(tok(row)["input_ids"]))
        continue
    enc = tok.apply_chat_template(row, tokenize=True, reasoning_effort="medium")
    ids = enc["input_ids"] if hasattr(enc, "keys") else enc
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    lengths.append(len(ids))
print(json.dumps({"lengths": lengths}))
"""


def _describe(values: list[int]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    return {"min": ordered[0], "p50": int(statistics.median(ordered)),
            "p90": ordered[int(len(ordered) * 0.9)], "max": ordered[-1],
            "total": sum(ordered)}


def _suggest_length(tokens: list[int], ceiling: int) -> int:
    """The smallest power-of-two window that still holds every row.

    Training at 2048 when the longest row is 827 buys nothing and costs
    activation memory on a card that has none to spare.
    """
    if not tokens:
        return ceiling
    window = 512
    while window < max(tokens) and window < ceiling:
        window *= 2
    return min(window, ceiling)


# ---------------------------------------------------------------------------
# report / sample
# ---------------------------------------------------------------------------

def cmd_report(args) -> int:
    """Everything a decision needs, in one call. Counts first."""
    run = open_run(args.run)
    rows = read_jsonl(run.dataset)
    manifest = read_jsonl(run.manifest)
    build = _read_report(run, "build")
    validate = _read_report(run, "validate")

    print(f"# Run: {run.name}")
    print()
    print("Counts")
    print(f"  source             {run.recipe.get('source', 'UNAVAILABLE: not ingested')}")
    print(f"  shape              {run.recipe.get('shape', 'UNAVAILABLE: not built')}")
    print(f"  documents          {sum(1 for r in manifest if not r['skipped'])}"
          f" of {len(manifest)} files")
    print(f"  rows               {len(rows)}")
    print(f"  held out           {run.recipe.get('eval_rows', 0)}")
    validated = "yes" if validate else "NO — run `corpus.py validate`"
    print(f"  validated          {validated}")
    print()

    skipped = [r for r in manifest if r["skipped"]]
    if skipped:
        print(f"Files not converted ({len(skipped)}):")
        for record in skipped[:10]:
            print(f"  {record['name']}: {record['skipped']}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")
        print()

    if build:
        if build.get("dropped"):
            print("Dropped at build, by reason:")
            for reason, count in sorted(build["dropped"].items(), key=lambda kv: -kv[1]):
                print(f"  {count:5d}  {reason}")
            print()
        methods = build.get("methods") or {}
        if methods:
            total = sum(methods.values()) or 1
            print("Instruction source: " + ", ".join(
                f"{k} {v} ({100 * v / total:.0f}%)" for k, v in
                sorted(methods.items(), key=lambda kv: -kv[1])))
            if methods.get("llm", 0) / total > 0.5:
                print("  NOTE: most instructions were written by the local model. "
                      "A dataset mostly captioned by a model teaches the model to "
                      "imitate itself.")
            print()

    if validate:
        if validate.get("refused"):
            print("Refused at validation, by reason:")
            for reason, count in sorted(validate["refused"].items(), key=lambda kv: -kv[1]):
                print(f"  {count:5d}  {reason}")
            print()
        tokens = validate.get("tokens") or {}
        if tokens:
            print(f"Tokens ({validate.get('tokenizer')}): min {tokens['min']}  "
                  f"p50 {tokens['p50']}  p90 {tokens['p90']}  max {tokens['max']}  "
                  f"total {tokens['total']:,}")
            print(f"  suggested max_seq_length: "
                  f"{run.recipe.get('suggested_max_seq_length', '?')}")
        if validate.get("grammar_smells"):
            print(f"  instruction grammar problems: {validate['grammar_smells']}")
        print()

    per_source = (build or {}).get("per_source") or {}
    if per_source:
        ordered = sorted(per_source.items(), key=lambda kv: -kv[1])
        print("Rows per source (top 10):")
        for name, count in ordered[:10]:
            print(f"  {count:5d}  {name}")
        zero = [r["name"] for r in manifest
                if not r["skipped"] and r["name"] not in per_source]
        if zero:
            print(f"  Yielded nothing: {', '.join(zero[:5])}"
                  + (f" and {len(zero) - 5} more" if len(zero) > 5 else ""))
            print("  A converted file yielding no rows usually means its export "
                  "format slipped past the cleaner.")
        print()

    _print_render(run, rows)
    _print_samples(rows, args.n)

    print(f"How much is enough: a voice adapter wants 200-500 rows, and 1000-3000 "
          f"is comfortable. This run has {len(rows)}.")
    return 0


def _read_report(run, stage: str) -> dict | None:
    path = run.report_path(stage)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except ValueError:
        return None


def _print_render(run, rows) -> None:
    """One row through the real chat template, with the serving arguments.

    This is the check that catches a train/serve mismatch before a run rather
    than after it: the template's own default reasoning effort prepends a
    paragraph to every example that the served model never sees.
    """
    if not rows or "messages" not in rows[0]:
        return
    python = common.unsloth_python()
    if python is None:
        print("UNAVAILABLE: rendered prompt — the training venv was not found, "
              "so the chat template could not be applied.")
        print()
        return
    payload = json.dumps({"model": run.recipe.get("base_model", common.DEFAULT_BASE_MODEL),
                          "messages": rows[0]["messages"],
                          "effort": common.DEFAULT_REASONING_EFFORT})
    code, out, err = _pipe(python, _RENDER_SCRIPT, payload)
    if code != 0:
        print(f"UNAVAILABLE: rendered prompt — {err.strip()[:160]}")
        print()
        return
    print("Rendered prompt (first row, as the backend will see it):")
    print("-" * 68)
    print(out.strip()[:1200])
    print("-" * 68)
    print()


_RENDER_SCRIPT = r"""
import json, sys
data = json.load(sys.stdin)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(data["model"])
print(tok.apply_chat_template(data["messages"], tokenize=False,
                              reasoning_effort=data["effort"]))
"""


def _print_samples(rows, count: int) -> None:
    print(f"Sample rows ({min(count, len(rows))} of {len(rows)}):")
    step = max(1, len(rows) // max(1, count))
    for row in rows[::step][:count]:
        print()
        if "text" in row:
            print(f"  [raw] {row['text'][:300]}...")
            continue
        for message in row["messages"]:
            body = message["content"].replace("\n", " ")
            limit = 400 if message["role"] == "assistant" else 200
            print(f"  [{message['role']}] {body[:limit]}"
                  + ("..." if len(body) > limit else ""))
    print()


def cmd_sample(args) -> int:
    run = open_run(args.run)
    rows = read_jsonl(run.dataset)
    if not rows:
        return common.fail(f"run {run.name!r} has no dataset")
    _print_samples(rows, args.n)
    print(f"{len(rows)} rows in {run.dataset}")
    return 0


# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="convert a folder of documents to text")
    p.add_argument("source")
    p.add_argument("--run", required=True)
    p.add_argument("--no-recursive", action="store_true")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("build", help="turn documents into a dataset")
    p.add_argument("--run", required=True)
    p.add_argument("--shape", choices=common.SHAPES)
    p.add_argument("--system", help="the system prompt every row carries")
    p.add_argument("--assistant-speaker", help="dialogue: whose turns to learn")
    p.add_argument("--min-words", type=int, default=150)
    p.add_argument("--max-words", type=int, default=500)
    p.add_argument("--min-prose-ratio", type=float, default=0.6,
                   help="drop a section with fewer prose lines than this")
    p.add_argument("--seed", type=int, default=3407)
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("validate", help="the quality gate")
    p.add_argument("--run", required=True)
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--holdout", type=int, default=8)
    p.add_argument("--base-model", default=common.DEFAULT_BASE_MODEL)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("report", help="everything a decision needs, in one call")
    p.add_argument("--run", required=True)
    p.add_argument("--n", type=int, default=3)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("sample", help="print rows")
    p.add_argument("--run", required=True)
    p.add_argument("--n", type=int, default=3)
    p.set_defaults(func=cmd_sample)

    args = parser.parse_args()
    try:
        return args.func(args)
    except RunError as exc:
        return common.fail(str(exc))


if __name__ == "__main__":
    sys.exit(main())
