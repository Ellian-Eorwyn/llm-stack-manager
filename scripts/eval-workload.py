#!/usr/bin/env python3
"""Workload eval: the research tasks, rather than trivia and code.

    scripts/eval-workload.py --slot llm-b --variant flash-next-experts3-rest8
    scripts/eval-workload.py --url http://127.0.0.1:8020 --sets faith,classify

What this stack is used for is reading long material and working only from
it -- synthesizing it, coding passages against a scheme, classifying, writing
it up. Those are the abilities quantization is known to erode first (long-
context recall, faithfulness, abstention), and exactly the ones knowledge and
coding benchmarks do not measure. docs/model-evals.md has the background.

  longctx   LongBench v2 (zai-org/LongBench-v2): multiple choice over English
            documents up to ~110k tokens, excluding code. Synthesis.
  oolong    Oolong-synth (oolongbench/oolong-synth): 32k-128k documents in
            which every line must be classified and the labels aggregated --
            counts, comparisons, timelines. Qualitative coding at scale.
            Official scoring: exact match, numeric partial credit 0.75^|err|.
  faith     FaithEval (Salesforce): counterfactual contexts that contradict
            common knowledge (answer from the text), and contexts with the
            answer removed (say it is unknown). Grounding and abstention.
  classify  Banking77 via LongICLBench: 77 intent codes, each defined by an
            example in the prompt. Classification against a given scheme.
  prose     Five grounded writing tasks at the model's recommended sampling.
            Scored for degenerate repetition, the known 3-bit failure; the
            texts are saved for reading side by side.

Every set is a fixed, seeded sample, so each model gets the same questions.
The data is fetched once into benchmarks/eval-data/. Parquet needs pyarrow:
the script re-runs itself under deps/eval-venv when that exists (see
docs/model-evals.md to create it).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import random
import re
import subprocess
import sys
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarks" / "eval-data"
EVAL_VENV = ROOT / "deps" / "eval-venv" / "bin" / "python"
SLOT_PORTS = {"llm-a": 8010, "llm-b": 8020}
SEED = 20260928

#: Dataset revisions, so a later upstream edit cannot change the questions.
REVISIONS = {
    "zai-org/LongBench-v2": "2b48e494",
    "oolongbench/oolong-synth": "f0d59eaf",
    "Salesforce/FaithEval-counterfactual-v1.0": "e655f7c8",
    "Salesforce/FaithEval-unanswerable-v1.0": "4a14e0e9",
    "TIGER-Lab/LongICLBench": "fc3f5082",
}
#: Oolong-synth test files holding 32k-128k windows, one per source dataset.
OOLONG_FILES = ("data/test-00009-of-00041.parquet",   # formality
                "data/test-00012-of-00041.parquet",   # imdb sentiment
                "data/test-00017-of-00041.parquet",   # app reviews
                "data/test-00033-of-00041.parquet")   # metaphors


# -- data --------------------------------------------------------------------

def _fetch(repo: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, filename, repo_type="dataset", revision=REVISIONS[repo])


def _parquet(repo: str, filename: str, columns=None) -> list[dict]:
    import pyarrow.parquet as pq
    return pq.read_table(_fetch(repo, filename), columns=columns).to_pylist()


def _english(text: str) -> bool:
    sample = text[:20000]
    return sum(ch.isascii() for ch in sample) / max(len(sample), 1) > 0.97


def build(name: str) -> list[dict]:
    rng = random.Random(f"{SEED}-{name}")
    if name == "longctx":
        data = json.load(open(_fetch("zai-org/LongBench-v2", "data.json")))
        pool = [d for d in data
                if d["domain"] != "Code Repository Understanding"
                and d["length"] in ("short", "medium")
                and _english(d["context"]) and len(d["context"]) / 4 <= 110_000]
        return rng.sample(pool, min(40, len(pool)))
    if name == "oolong":
        items = []
        for filename in OOLONG_FILES:
            rows = [r for r in _parquet("oolongbench/oolong-synth", filename)
                    if r["context_len"] in (32768, 65536, 131072)]
            picked = rng.sample(rows, min(12, len(rows)))
            items += [{k: r[k] for k in ("id", "context_len", "dataset", "context_window_text",
                                         "question", "task", "answer", "answer_type")}
                      for r in picked]
        return items
    if name == "faith":
        counter = _parquet("Salesforce/FaithEval-counterfactual-v1.0", "data/test-00000-of-00001.parquet")
        unans = _parquet("Salesforce/FaithEval-unanswerable-v1.0", "data/test-00000-of-00001.parquet")
        return ([dict(r, kind="counterfactual") for r in rng.sample(counter, 60)]
                + [dict(r, kind="unanswerable") for r in rng.sample(unans, 60)])
    if name == "classify":
        rows = _parquet("TIGER-Lab/LongICLBench", "data/BANKING77-00000-of-00001.parquet",
                        columns=["1 Round Prompt", "label"])
        return [{"prompt": r["1 Round Prompt"], "label": r["label"]} for r in rng.sample(rows, 100)]
    if name == "prose":
        data = json.load(open(_fetch("zai-org/LongBench-v2", "data.json")))
        pool = [d for d in data if d["domain"] == "Single-Document QA" and d["length"] == "short"
                and _english(d["context"])]
        return [{"context": d["context"][:24000], "id": d["_id"]} for d in rng.sample(pool, 5)]
    raise SystemExit(f"unknown set {name}")


def load(name: str) -> list[dict]:
    DATA.mkdir(parents=True, exist_ok=True)
    cache = DATA / f"workload-{name}.json"
    if not cache.exists():
        cache.write_text(json.dumps(build(name), default=str))
    return json.loads(cache.read_text())


# -- asking ------------------------------------------------------------------

def ask(url: str, model: str, prompt: str, max_tokens: int, timeout: float,
        sampling: dict | None = None) -> tuple[str, bool]:
    """(reply, whether it hit max_tokens). A cut-off reply is reported apart:
    it measures the budget, not the model."""
    body = {"model": model, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "chat_template_kwargs": {"enable_thinking": False}, "enable_thinking": False}
    body.update(sampling or {"temperature": 0})
    request = urllib.request.Request(f"{url}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.load(response)
    choice = data["choices"][0]
    text = choice["message"].get("content") or ""
    return (re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip(),
            choice.get("finish_reason") == "length")


def _is_ready(url: str) -> bool:
    """`/ready` where the server has one (Splash: `/health` answers before the
    model has loaded), else `/health`."""
    import urllib.error
    try:
        with urllib.request.urlopen(f"{url}/ready", timeout=3):
            return True
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            return False
    except Exception:
        return False
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=3):
            return True
    except Exception:
        return False


def wait_healthy(url: str, timeout: float = 900) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _is_ready(url):
            return
        time.sleep(5)


# -- prompts and grading -----------------------------------------------------

def prompt_for(name: str, item: dict) -> tuple[str, int, dict | None]:
    if name == "longctx":
        choices = "\n".join(f"({c}) {item['choice_' + c]}" for c in "ABCD")
        return (f"Please read the following text and answer the question below.\n\n<text>\n"
                f"{item['context']}\n</text>\n\nWhat is the correct answer to this question: "
                f"{item['question']}\nChoices:\n{choices}\n\nFormat your response as follows: "
                "\"The correct answer is (insert answer here)\".", 2048, None)
    if name == "oolong":
        # Room to work through the lines, which is how a model solves these:
        # at 1,024 tokens most replies were cut off mid-count.
        return (f"{item['context_window_text']}\n{item['question']}", 4096, None)
    if name == "faith" and item["kind"] == "counterfactual":
        options = "\n".join(f"{l}. {t}" for l, t in zip(item["choices"]["label"], item["choices"]["text"]))
        return (f"Context:\n{item['context']}\n\nAnswer the question using only the context above, "
                f"even where it disagrees with what you believe.\n\nQuestion: {item['question']}\n"
                f"{options}\n\nFinish with a line of the form 'Answer: <letter>'.", 768, None)
    if name == "faith":
        return (f"Context:\n{item['context']}\n\nAnswer the question using only the context above. "
                "If the context does not contain the answer, reply exactly 'unknown'.\n\n"
                f"Question: {item['question']}\nAnswer briefly.", 256, None)
    if name == "classify":
        return (item["prompt"].rstrip() + "\n\nReply with the intent label only, exactly as written "
                "in the examples.", 32, None)
    return (f"Here is a document.\n\n<document>\n{item['context']}\n</document>\n\nWrite about 300 "
            "words of polished prose for an informed reader, synthesizing the document's central "
            "argument and its most important supporting evidence. Use only the document. No bullet "
            "points or headings.", 700,
            # Qwen's recommended non-thinking sampling; greedy decoding is
            # what makes a weak quantization loop.
            {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5})


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


def _after(reply: str, keys: tuple[str, ...]) -> str:
    found = None
    for match in re.finditer(r"(?im)^\W*(%s)\s*:\s*(.+)$" % "|".join(keys), reply):
        found = match.group(2)
    return (found or reply.strip().splitlines()[-1] if reply.strip() else "").strip(" .*'\"[]")


def _oolong_score(item: dict, reply: str) -> float:
    try:
        gold = eval(item["answer"], {"datetime": datetime})[0]  # noqa: S307 -- the dataset's own repr
    except Exception:
        gold = item["answer"].strip("[]'\"")
    kind = item["answer_type"].split(".")[-1]
    got = _after(reply, ("Answer", "Label", "User", "Date"))
    if kind == "NUMERIC":
        numbers = re.findall(r"-?\d+", got)
        return 0.75 ** abs(int(numbers[-1]) - int(gold)) if numbers else 0.0
    if kind == "DATE":
        try:
            parsed = datetime.datetime.strptime(re.findall(r"\d{1,2}/\d{1,2}/\d{4}", got)[-1], "%m/%d/%Y").date()
            return float(parsed == gold)
        except (IndexError, ValueError):
            return 0.0
    if kind == "COMPARISON":
        for phrase in ("more common than", "less common than", "same frequency as"):
            if phrase in got.lower():
                return float(phrase == str(gold).lower())
        return 0.0
    return float(_norm(str(gold)) == _norm(got) or _norm(str(gold)) in _norm(got).split(" "))


_UNKNOWN = ("unknown", "unanswerable", "no answer", "no information", "not mentioned",
            "not provided", "not specified", "not stated", "does not say", "cannot be determined")


def score(name: str, item: dict, reply: str) -> float:
    if name == "longctx":
        found = re.findall(r"correct answer is \(?([A-D])\)?", reply) or re.findall(r"\(([A-D])\)", reply)
        return float(bool(found) and found[-1] == item["answer"])
    if name == "oolong":
        return _oolong_score(item, reply)
    if name == "faith" and item["kind"] == "counterfactual":
        found = re.findall(r"Answer:\s*\(?([A-J])\)?", reply)
        return float(bool(found) and found[-1] == item["answerKey"])
    if name == "faith":
        return float(any(term in reply.lower() for term in _UNKNOWN))
    if name == "classify":
        return float(_norm(reply.splitlines()[0] if reply else "") == _norm(item["label"].replace("_", " "))
                     or _norm(reply.splitlines()[0] if reply else "").replace(" ", "_") == item["label"])
    words = reply.split()
    grams = [tuple(words[i:i + 4]) for i in range(max(len(words) - 3, 0))]
    repeated = 1 - len(set(grams)) / max(len(grams), 1)
    item["_repetition"] = round(repeated, 3)
    item["_words"] = len(words)
    # Clean prose repeats almost no 4-word sequence; a looping model repeats most.
    return float(repeated < 0.08 and len(words) >= 150)


def main(argv: list[str] | None = None) -> int:
    try:
        import pyarrow  # noqa: F401
        import huggingface_hub  # noqa: F401
    except ImportError:
        # By prefix, not by interpreter path: a venv's python is a symlink to
        # the one this may already be running.
        if EVAL_VENV.exists() and pathlib.Path(sys.prefix).resolve() != EVAL_VENV.parents[1].resolve():
            os.execv(str(EVAL_VENV), [str(EVAL_VENV), __file__, *(argv or sys.argv[1:])])
        raise SystemExit("needs pyarrow and huggingface_hub; see docs/model-evals.md (deps/eval-venv)")

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--slot", choices=sorted(SLOT_PORTS), default="llm-b")
    parser.add_argument("--url")
    parser.add_argument("--model", default="")
    parser.add_argument("--variant", default="")
    parser.add_argument("--sets", default="classify,faith,longctx,oolong,prose")
    parser.add_argument("--limit", type=int, default=0, help="questions per set; 0 = all")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--restart-cmd", default="")
    parser.add_argument("--prepare", action="store_true", help="fetch and cache the data, then exit")
    parser.add_argument("--out", default=str(ROOT / "benchmarks"))
    args = parser.parse_args(argv)

    # Names meant for another suite (the runner passes one --sets to all) are skipped.
    known = ("longctx", "oolong", "faith", "classify", "prose")
    sets = [s for s in args.sets.split(",") if s in known]
    if args.prepare:
        for name in sets:
            print(f"{name}: {len(load(name))} items")
        return 0

    url = (args.url or f"http://127.0.0.1:{SLOT_PORTS[args.slot]}").rstrip("/")
    model = args.model
    if not model:
        with urllib.request.urlopen(f"{url}/v1/models", timeout=30) as response:
            model = json.load(response)["data"][0]["id"]

    results: dict[str, dict] = {}
    for name in sets:
        items = load(name)[: args.limit or None]
        total, restarts, failures, samples, started = 0.0, 0, [], [], time.monotonic()
        truncated = 0
        for index, item in enumerate(items):
            prompt, max_tokens, sampling = prompt_for(name, item)
            for attempt in range(2):
                try:
                    reply, cut = ask(url, model, prompt, max_tokens, args.timeout, sampling)
                    truncated += cut
                    got = score(name, item, reply)
                    break
                except Exception as exc:
                    reply, got = f"ERROR {type(exc).__name__}: {exc}", 0.0
                    # 507: MTPLX's memory guard. Refused/reset: the server is gone.
                    dead = any(m in str(exc) for m in ("507", "Connection refused",
                                                       "Connection reset", "Remote end closed"))
                    if not (args.restart_cmd and dead and attempt == 0):
                        break
                    restarts += 1
                    subprocess.run(args.restart_cmd, shell=True, check=False)
                    wait_healthy(url)
            total += got
            if name == "prose":
                samples.append({"id": item["id"], "text": reply, "repetition": item.get("_repetition"),
                                "words": item.get("_words")})
            if got < 1:
                failures.append({"index": index, "score": got, "reply": reply[-500:],
                                 **({"context_len": item["context_len"], "task": item["task"]}
                                    if name == "oolong" else {})})
            print(f"\r{name}: {index + 1}/{len(items)}  {100 * total / (index + 1):.1f}%", end="",
                  file=sys.stderr, flush=True)
        print(file=sys.stderr)
        results[name] = {"score": round(100 * total / max(len(items), 1), 1),
                         "correct": round(total, 2), "total": len(items), "restarts": restarts,
                         "truncated": truncated,
                         "seconds": round(time.monotonic() - started), "failures": failures,
                         **({"samples": samples} if samples else {})}

    print(f"\n{args.variant or model}")
    for name, r in results.items():
        print(f"  {name:9} {r['score']:5.1f}%  ({r['total']} items, {r['seconds']} s, "
              f"{r['restarts']} restarts, {r['truncated']} cut off)")
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    path = out / f"{stamp}-workload-{args.variant or model}.json".replace("/", "_")
    path.write_text(json.dumps({"variant": args.variant, "model": model, "url": url,
                                "suite": "workload", "results": results}, indent=2, default=str) + "\n")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
