#!/usr/bin/env python3
"""Decode speed, memory and SSD reads for a chat backend, at several depths.

    scripts/bench-offload.py --slot llm-b --variant flash-next-3bit
    scripts/bench-offload.py --slot llm-a --depths 0,8192,32768 --tasks code

Built to compare the ways of running a model bigger than the RAM it may use
(docs/ssd-offload.md): the same prompts at the same depths against each, so
the numbers differ only by the configuration. It talks to a backend that is
already serving and never starts, stops or reconfigures one.

Per depth and task it records prefill and decode tok/s, the backend's peak
memory (phys_footprint plus the resident pages of mapped weight files -- what
the manager's memory panel shows), the bytes it read from disk while
decoding, and the worst memory-pressure level seen. Each prompt opens with a
fresh nonce so no prompt cache can answer for the prefill.

Writes benchmarks/<date>-<variant>.json and prints a table.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import random
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "web"))

#: The backend ports, which docs/pi-forge-scheduling-contract.md freezes.
SLOT_PORTS = {"llm-a": 8010, "llm-b": 8020}

#: Characters per token for the synthetic code context. Only a first guess at
#: the prompt length; the table reports what the server counted.
CHARS_PER_TOKEN = 3.3

#: `kern.memorystatus_vm_pressure_level`.
PRESSURE = {1: "normal", 2: "warn", 4: "critical"}


def synthetic_code(tokens: int, seed: int) -> str:
    """Plausible Python, deterministic for a seed, about `tokens` long."""
    rng = random.Random(seed)
    words = ("cache", "slot", "budget", "token", "expert", "layer", "page", "stream",
             "prefetch", "route", "weight", "block", "shard", "table", "window")
    parts, size, target = [], 0, int(tokens * CHARS_PER_TOKEN)
    n = 0
    while size < target:
        a, b, c = rng.sample(words, 3)
        body = (f"def {a}_{b}_{n}({c}, limit={rng.randint(2, 512)}):\n"
                f"    \"\"\"Return the {b} of each {c} under the {a} limit.\"\"\"\n"
                f"    out = []\n"
                f"    for i, item in enumerate({c}):\n"
                f"        if i >= limit:\n"
                f"            break\n"
                f"        out.append(item.{b} * {rng.randint(1, 9)} + {rng.randint(0, 99)})\n"
                f"    return out\n\n")
        parts.append(body)
        size += len(body)
        n += 1
    return "".join(parts)


TARGET_FUNCTION = '''def merge_expert_cache(slots, incoming, capacity=64):
    """Merge newly routed experts into the per-layer slot cache."""
    evicted = []
    for layer, experts in incoming.items():
        cache = slots.setdefault(layer, [])
        for expert in experts:
            if expert in cache:
                cache.remove(expert)
            cache.append(expert)
            if len(cache) > capacity:
                evicted.append((layer, cache.pop(0)))
    return evicted
'''

TASKS = {
    # Mostly new text: what decode costs when nothing can be copied.
    "chat": ("Without writing code, explain in plain prose how an operating system "
             "decides which memory pages to evict under pressure. About 300 words."),
    # Mostly copied text: where n-gram and MTP drafting pay off.
    "code": ("Rewrite merge_expert_cache from the code above so that `capacity` is "
             "renamed `max_slots` and evicted entries are returned as dicts with keys "
             "'layer' and 'expert'. Output only the complete function."),
}


def backend_pid(port: int) -> int | None:
    try:
        out = subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
                             capture_output=True, text=True, timeout=10).stdout.split()
    except (OSError, subprocess.SubprocessError):
        return None
    return int(out[0]) if out else None


def pressure_level() -> int:
    try:
        return int(subprocess.run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
                                  capture_output=True, text=True, timeout=5).stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0


def process_tree(pid: int) -> list[int]:
    """`pid` and its descendants: a server may keep the model in a worker
    process (Splash does), and the launcher alone would read as near zero."""
    out, todo = [], [pid]
    while todo:
        current = todo.pop()
        out.append(current)
        children = subprocess.run(["pgrep", "-P", str(current)], capture_output=True,
                                  text=True).stdout.split()
        todo += [int(c) for c in children]
    return out


class Sampler(threading.Thread):
    """Peak memory and pressure while a request runs."""

    def __init__(self, pid: int | None, every: float = 0.5):
        super().__init__(daemon=True)
        self.pid, self.every = pid, every
        self.peak_mib, self.pressure = 0, 0
        self._done = threading.Event()

    def run(self):
        from platforms import darwin
        while not self._done.is_set():
            if self.pid:
                total = sum(darwin._process_memory_mib(p) or 0 for p in process_tree(self.pid))
                self.peak_mib = max(self.peak_mib, total)
            self.pressure = max(self.pressure, pressure_level())
            self._done.wait(self.every)

    def stop(self):
        self._done.set()
        self.join(timeout=5)


def disk_read(pid: int | None) -> int | None:
    if not pid:
        return None
    from platforms import darwin
    reads = [darwin._process_disk_read_bytes(p) for p in process_tree(pid)]
    return sum(r for r in reads if r is not None) if any(r is not None for r in reads) else None


def run_one(url: str, model: str, prompt: str, max_tokens: int, pid: int | None,
            timeout: float) -> dict:
    body = json.dumps({
        "model": model, "stream": True, "max_tokens": max_tokens, "temperature": 0.7,
        "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": prompt}],
        # Thinking is measured as decode like anything else, but a reply that
        # spends its budget deliberating says less about copyable output.
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    request = urllib.request.Request(f"{url}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
    sampler = Sampler(pid)
    sampler.start()
    read_start = disk_read(pid)
    start = time.monotonic()
    first = last = None
    read_at_first = None
    pieces, usage, timings = 0, {}, {}
    refused = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[5:])
                if chunk.get("error"):
                    refused = chunk["error"]
                usage = chunk.get("usage") or usage
                timings = chunk.get("timings") or timings
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("content") or delta.get("reasoning_content"):
                        now = time.monotonic()
                        if first is None:
                            first, read_at_first = now, disk_read(pid)
                        last = now
                        pieces += 1
    finally:
        sampler.stop()
    read_end = disk_read(pid)

    if first is None:
        # A server that turns a request away -- MTPLX's memory guard does, at
        # admission -- may still answer 200 with an empty stream.
        detail = refused.get("message") if isinstance(refused, dict) else refused
        raise RuntimeError(f"no output: {detail or 'the stream ended empty'}")
    completion = int(usage.get("completion_tokens") or pieces)
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    decode_s = (last - first) if first and last and last > first else 0.0
    result = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion,
        "ttft_s": round((first or time.monotonic()) - start, 2),
        "prefill_tps": round(prompt_tokens / ((first or start) - start), 1)
                       if first and prompt_tokens else None,
        "decode_tps": round((completion - 1) / decode_s, 1) if decode_s and completion > 1 else None,
        "peak_mib": sampler.peak_mib or None,
        "pressure": PRESSURE.get(sampler.pressure, str(sampler.pressure)),
        "disk_read_decode_mib": round((read_end - read_at_first) / 2**20)
                                if read_end is not None and read_at_first is not None else None,
        "disk_read_total_mib": round((read_end - read_start) / 2**20)
                               if read_end is not None and read_start is not None else None,
    }
    # llama-server reports its own rates, which exclude streaming overhead,
    # and how often a draft was accepted.
    if timings:
        result["server_decode_tps"] = round(float(timings.get("predicted_per_second") or 0), 1) or None
        result["server_prefill_tps"] = round(float(timings.get("prompt_per_second") or 0), 1) or None
        if timings.get("draft_n"):
            result["draft_accept"] = round(timings.get("draft_n_accepted", 0) / timings["draft_n"], 2)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--slot", choices=sorted(SLOT_PORTS), default="llm-b")
    parser.add_argument("--url", help="Backend base URL; default is the slot's loopback port")
    parser.add_argument("--model", default="", help="Model id to send; default asks /v1/models")
    parser.add_argument("--variant", default="", help="Name for the results file")
    parser.add_argument("--pid", type=int, default=0,
                        help="Backend process for the memory and disk columns; default "
                             "finds whatever listens on the URL's port")
    parser.add_argument("--depths", default="0,8192,32768,131072,262144",
                        help="Comma-separated context depths in tokens")
    parser.add_argument("--tasks", default="chat,code", help=f"Any of {','.join(TASKS)}")
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--timeout", type=float, default=3600,
                        help="Seconds per request; a cold 262k prefill takes minutes")
    parser.add_argument("--pause", type=float, default=0,
                        help="seconds to wait between requests, e.g. to let a server release "
                             "a finished conversation before the next one")
    parser.add_argument("--out", default=str(ROOT / "benchmarks"))
    args = parser.parse_args(argv)

    port = SLOT_PORTS[args.slot]
    url = (args.url or f"http://127.0.0.1:{port}").rstrip("/")
    model = args.model
    if not model:
        with urllib.request.urlopen(f"{url}/v1/models", timeout=30) as response:
            model = json.load(response)["data"][0]["id"]
    if args.url:
        port = int(urllib.parse.urlsplit(url).port or port)
    pid = args.pid or backend_pid(port)
    if pid is None:
        print("note: backend process not found; memory and disk columns stay empty",
              file=sys.stderr)

    depths = [int(d) for d in args.depths.split(",") if d.strip()]
    tasks = [t for t in args.tasks.split(",") if t in TASKS]
    rows = []
    for depth in depths:
        for task in tasks:
            nonce = random.randrange(1 << 30)
            context = synthetic_code(depth, seed=depth) if depth else ""
            if task == "code":
                context += "\n" + TARGET_FUNCTION
            prompt = (f"[run {nonce}]\n" + (f"```python\n{context}\n```\n\n" if context else "")
                      + TASKS[task])
            print(f"· {task} at ~{depth} tokens ...", file=sys.stderr, flush=True)
            try:
                row = run_one(url, model, prompt, args.max_tokens, pid, args.timeout)
            except Exception as exc:  # one failed depth must not lose the rest
                row = {"error": f"{type(exc).__name__}: {exc}"}
            rows.append({"depth": depth, "task": task, **row})
            if args.pause:
                time.sleep(args.pause)

    header = ("depth", "task", "prompt", "prefill/s", "decode/s", "peak GiB", "SSD MiB", "pressure")
    print("\n" + "  ".join(f"{h:>10}" for h in header))
    for row in rows:
        if "error" in row:
            print(f"{row['depth']:>10}  {row['task']:>10}  {row['error']}")
            continue
        decode = row.get("server_decode_tps") or row.get("decode_tps")
        peak = f"{row['peak_mib'] / 1024:.1f}" if row.get("peak_mib") else "-"
        cells = (row["depth"], row["task"], row["prompt_tokens"],
                 row.get("server_prefill_tps") or row.get("prefill_tps") or "-",
                 decode or "-", peak, row.get("disk_read_decode_mib", "-"), row["pressure"])
        print("  ".join(f"{str(c):>10}" for c in cells))

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    name = f"{stamp}-{args.variant or model}".replace("/", "_")
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps({"variant": args.variant, "model": model, "slot": args.slot,
                                "url": url, "max_tokens": args.max_tokens, "rows": rows,
                                "host": os.uname().nodename}, indent=2) + "\n")
    print(f"\nwrote {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
