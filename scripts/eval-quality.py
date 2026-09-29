#!/usr/bin/env python3
"""Answer-quality check for a chat backend: math, code and knowledge.

    scripts/eval-quality.py --slot llm-b --variant flash-next-3bit-g64

The question SSD offload keeps raising is what a smaller quantization costs,
and a speed benchmark cannot answer it. This asks the same fixed questions of
every variant, greedily, and grades the answers mechanically:

  gsm8k      grade-school math word problems; the final number must match
  humaneval  Python functions; the completion must pass the problem's tests
  mmlu-pro   ten-option multiple choice across 14 subjects

Run every variant with the same arguments and compare the scores; the
absolute numbers depend on the prompts here (thinking off, short answers) and
are not comparable with published ones. At ~100 questions a difference of a
few points is within noise.

Model-written code runs under `sandbox-exec` with no network and no writes
outside a temporary directory, with a timeout. The questions come from the
Hugging Face datasets API and are cached under benchmarks/eval-data/.
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import re
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarks" / "eval-data"
SLOT_PORTS = {"llm-a": 8010, "llm-b": 8020}
ROWS_API = "https://datasets-server.huggingface.co"



def _get(url: str) -> dict:
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.load(response)
        except Exception:
            if attempt == 4:
                raise
            time.sleep(5 * (attempt + 1))
    raise AssertionError


def _rows(dataset: str, config: str, split: str, offset: int, length: int,
          where: str = "") -> list[dict]:
    params = {"dataset": dataset, "config": config, "split": split,
              "offset": offset, "length": length}
    endpoint = "rows"
    if where:
        params["where"] = where
        endpoint = "filter"
    data = _get(f"{ROWS_API}/{endpoint}?{urllib.parse.urlencode(params)}")
    return [r["row"] for r in data["rows"]]


def load(name: str, limit: int) -> list[dict]:
    """The first `limit` questions of a set, fetched once and cached."""
    DATA.mkdir(parents=True, exist_ok=True)
    cache = DATA / f"{name}.json"
    if cache.exists():
        items = json.loads(cache.read_text())
    else:
        if name == "gsm8k":
            items = [r for off in (0, 100) for r in _rows("openai/gsm8k", "main", "test", off, 100)]
        elif name == "humaneval":
            items = [r for off in (0, 100) for r in
                     _rows("openai/openai_humaneval", "openai_humaneval", "test", off, 100)]
        elif name == "mmlu-pro":
            # Ten from each of fourteen evenly spaced points in the 12,032-row
            # test split, which is grouped by subject. The API's filter
            # endpoint, which could pick per subject, did not answer.
            items = []
            for block in range(14):
                items += _rows("TIGER-Lab/MMLU-Pro", "default", "test", block * 859, 10)
        else:
            raise SystemExit(f"unknown set {name}")
        cache.write_text(json.dumps(items))
    return items[:limit]


# -- asking ------------------------------------------------------------------

def ask(url: str, model: str, prompt: str, max_tokens: int, timeout: float) -> tuple[str, int]:
    body = json.dumps({
        "model": model, "max_tokens": max_tokens, "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
        # Off for every variant alike: the comparison is between weights, and
        # a thinking budget would mostly measure how long each one deliberates.
        "chat_template_kwargs": {"enable_thinking": False},
        "enable_thinking": False,
    }).encode()
    request = urllib.request.Request(f"{url}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.load(response)
    message = data["choices"][0]["message"]
    text = message.get("content") or ""
    # A server that thinks anyway puts it here; the answer is still in content.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return text, int((data.get("usage") or {}).get("completion_tokens") or 0)


# -- grading -----------------------------------------------------------------

_NUMBER = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


def _to_number(text: str) -> float | None:
    try:
        return float(text.replace(",", "").replace("$", ""))
    except ValueError:
        return None


def grade_gsm8k(item: dict, reply: str) -> bool:
    gold = _to_number(item["answer"].split("####")[-1].strip())
    # The first number after the last "Answer:", else the last number at all.
    if "Answer:" in reply:
        numbers = _NUMBER.findall(reply.rsplit("Answer:", 1)[1])[:1]
    else:
        numbers = _NUMBER.findall(reply)[-1:]
    return bool(numbers) and gold is not None and _to_number(numbers[0]) == gold


_SANDBOX = """(version 1)
(allow default)
(deny network*)
(deny file-write*)
(allow file-write* (subpath "{tmp}") (subpath "/private/var/folders") (literal "/dev/null"))
"""


def grade_humaneval(item: dict, reply: str) -> bool:
    blocks = re.findall(r"```(?:python)?\n(.*?)```", reply, flags=re.S)
    code = max(blocks, key=len) if blocks else reply
    if f"def {item['entry_point']}" not in code:
        code = item["prompt"] + code
    program = f"{code}\n\n{item['test']}\n\ncheck({item['entry_point']})\n"
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "solution.py"
        path.write_text(program)
        profile = _SANDBOX.format(tmp=tmp)
        try:
            done = subprocess.run(["sandbox-exec", "-p", profile, sys.executable, str(path)],
                                  cwd=tmp, capture_output=True, timeout=20)
        except subprocess.TimeoutExpired:
            return False
    return done.returncode == 0


def grade_mmlu_pro(item: dict, reply: str) -> bool:
    found = re.findall(r"Answer:\s*\(?([A-J])\)?", reply)
    if not found:
        found = re.findall(r"\b([A-J])\b", reply[-40:])
    return bool(found) and found[-1] == item["answer"]


def prompt_for(name: str, item: dict) -> tuple[str, int]:
    if name == "gsm8k":
        return (f"{item['question']}\n\nSolve it step by step, briefly. "
                "Finish with a line of the form 'Answer: <number>'.", 512)
    if name == "humaneval":
        return ("Complete this Python function. Reply with the complete function, "
                f"including its signature, in one ```python block.\n\n```python\n{item['prompt']}```",
                768)
    letters = "ABCDEFGHIJ"
    options = "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(item["options"]))
    return (f"{item['question']}\n\n{options}\n\nReason in at most 150 words, then finish "
            "with a line of the form 'Answer: <letter>'.", 1024)


GRADERS = {"gsm8k": grade_gsm8k, "humaneval": grade_humaneval, "mmlu-pro": grade_mmlu_pro}


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


def wait_healthy(url: str, timeout: float = 600) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _is_ready(url):
            return
        time.sleep(5)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--slot", choices=sorted(SLOT_PORTS), default="llm-b")
    parser.add_argument("--url")
    parser.add_argument("--model", default="")
    parser.add_argument("--variant", default="")
    parser.add_argument("--sets", default="gsm8k,humaneval,mmlu-pro")
    parser.add_argument("--limit", type=int, default=0,
                        help="questions per set; 0 = gsm8k 100, humaneval 164, mmlu-pro 140")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--restart-cmd", default="",
                        help="shell command that restarts the backend; run, and the question "
                             "retried, when it refuses with 507 (MTPLX's memory guard)")
    parser.add_argument("--out", default=str(ROOT / "benchmarks"))
    args = parser.parse_args(argv)

    url = (args.url or f"http://127.0.0.1:{SLOT_PORTS[args.slot]}").rstrip("/")
    model = args.model
    if not model:
        with urllib.request.urlopen(f"{url}/v1/models", timeout=30) as response:
            model = json.load(response)["data"][0]["id"]
    defaults = {"gsm8k": 100, "humaneval": 164, "mmlu-pro": 140}

    results: dict[str, dict] = {}
    for name in [s for s in args.sets.split(",") if s in GRADERS]:
        items = load(name, args.limit or defaults[name])
        correct, tokens, failures, started = 0, 0, [], time.monotonic()
        restarts = 0
        for index, item in enumerate(items):
            prompt, max_tokens = prompt_for(name, item)
            for attempt in range(2):
                try:
                    reply, used = ask(url, model, prompt, max_tokens, args.timeout)
                    ok = GRADERS[name](item, reply)
                    break
                except Exception as exc:
                    reply, used, ok = f"ERROR {type(exc).__name__}: {exc}", 0, False
                    # 507: MTPLX's memory guard. Refused/reset: the server is gone.
                    dead = any(m in str(exc) for m in ("507", "Connection refused",
                                                       "Connection reset", "Remote end closed"))
                    if not (args.restart_cmd and dead and attempt == 0):
                        break
                    restarts += 1
                    subprocess.run(args.restart_cmd, shell=True, check=False)
                    wait_healthy(url)
            correct += ok
            tokens += used
            if not ok:
                failures.append({"index": index, "reply": reply[-600:]})
            print(f"\r{name}: {index + 1}/{len(items)}  {correct} correct", end="",
                  file=sys.stderr, flush=True)
        print(file=sys.stderr)
        results[name] = {"correct": correct, "total": len(items),
                         "score": round(100 * correct / max(len(items), 1), 1),
                         "tokens": tokens, "seconds": round(time.monotonic() - started),
                         "restarts": restarts,
                         "failures": failures}

    print(f"\n{args.variant or model}")
    for name, r in results.items():
        print(f"  {name:10} {r['score']:5.1f}%  ({r['correct']}/{r['total']}, {r['seconds']} s, "
              f"{r['restarts']} restarts)")

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    path = out / f"{stamp}-quality-{args.variant or model}.json".replace("/", "_")
    path.write_text(json.dumps({"variant": args.variant, "model": model, "url": url,
                                "results": results}, indent=2) + "\n")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
