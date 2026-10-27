#!/usr/bin/env python3
"""Does a chat backend reuse a long prompt's prefix? One agent-shaped replay.

    scripts/ninfer_prefix_test.py
    scripts/ninfer_prefix_test.py --url http://other-host:8012 \\
        --metrics-url http://other-host:8010/metrics --out before.json

An agent resends its whole conversation every turn: the system prompt, the tool
schemas, every earlier message, then something new. A backend that keeps the
state of the last prompt answers the next turn after prefilling only the new
part; one that does not prefills all of it again. NInfer reports which it did
in `/metrics` (`prompt_tokens_cached_total`), so this sends one synthetic
~30k-token agent request, then the same conversation with one more turn, and
reads the counters around each.

- **The prefix** is synthetic but shaped like an agent's: a long system prompt
  with a skills index, ~25 tool schemas, and a research session of tool calls
  and long tool results (about 36k tokens as Qwen counts them). Its system
  prompt opens with a fresh nonce, so the first request is cold whatever an
  earlier run left cached, except for the tool schemas, which the Qwen
  template renders first (~2k tokens).
- **The second request** is the first one, its reply, and a new user turn
  (`--shape continue`, what an agent sends every turn), or the first one with
  its last user turn replaced (`--shape replace`).
- **`--interleave N`** sends N short, unrelated conversations between the two,
  so the first one's state must leave the card and come back from the host
  cache, as it does when an agent's turns are interleaved with other jobs.
- **TTFT** is the time from sending a request to its first streamed token of
  content or reasoning.
- **PASS** when the second request's cached tokens are at least
  `--min-cached-share` (0.9) of the prefix and its TTFT is under `--max-ttft`
  (10 s). The prefix is the first request's whole prompt for `continue`, and
  that less an estimate of its last turn for `replace`.

The request goes through a chat proxy by default and carries the proxy's
fallback-hop header, so a proxy whose backend is down answers 503 rather than
handing the test to another machine. It is a read-only request: nothing is
configured, started or stopped. It refuses to run while the backend is busy
(`requests_running` or `requests_waiting` above 0) unless `--force`, since
another request would both wait behind it and spoil its counters.

Prints one JSON object. Exit status: 0 PASS, 1 FAIL, 2 skipped or broken.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import secrets
import sys
import time
import urllib.error
import urllib.request

#: Characters per token for the synthetic prefix: a first guess at its length.
#: The result reports what the backend counted.
CHARS_PER_TOKEN = 3.6

FALLBACK_HOP_HEADER = "X-LLM-Fallback-Hop"

_WORDS = {
    "subject": ["the survey", "the city", "the pilot program", "each district", "the utility",
                "the review panel", "the field team", "the county", "the working group", "the vendor",
                "the dataset", "the second phase", "the regional office", "the audit", "the model"],
    "verb": ["reported", "measured", "estimated", "compared", "projected", "found", "recorded",
             "summarised", "flagged", "revised", "published", "documented", "tracked", "sampled"],
    "object": ["canopy cover", "peak surface temperature", "water use per household",
               "compost diversion rates", "energy savings", "maintenance cost per tree",
               "survival of new plantings", "permit processing times", "storm runoff volume",
               "heat-related emergency visits", "bus ridership", "air quality readings",
               "volunteer hours", "repair backlog", "grant spending"],
    "qualifier": ["across the study period", "in the hottest month", "after the second year",
                  "relative to the baseline", "in low-income tracts", "on a per-capita basis",
                  "for the median block", "within the margin of error", "in the revised estimate",
                  "before the policy change", "after adjusting for season", "in the pilot area"],
    "topic": ["urban heat", "street trees", "stormwater", "composting", "transit", "air quality",
              "building retrofits", "water conservation", "wildfire smoke", "community gardens"],
}


def _sentence(rng: random.Random) -> str:
    w = _WORDS
    n = rng.randint(2, 97)
    return (f"{rng.choice(w['subject']).capitalize()} {rng.choice(w['verb'])} "
            f"{rng.choice(w['object'])} of {n}% {rng.choice(w['qualifier'])}, "
            f"and {rng.choice(w['subject'])} {rng.choice(w['verb'])} {rng.choice(w['object'])} "
            f"{rng.choice(w['qualifier'])}.")


def _paragraph(rng: random.Random, sentences: int) -> str:
    return " ".join(_sentence(rng) for _ in range(sentences))


def _article(rng: random.Random, chars: int) -> str:
    """A long markdown page, as a page reader returns one."""
    topic = rng.choice(_WORDS["topic"])
    parts = [f"# {topic.title()} in mid-sized cities: findings from {rng.randint(3, 40)} programs\n",
             f"Source: https://example.org/reports/{topic.replace(' ', '-')}-{rng.randint(1000, 9999)}\n"]
    size = sum(len(p) for p in parts)
    section = 1
    while size < chars:
        heading = f"\n## {section}. {rng.choice(_WORDS['object']).capitalize()}\n\n"
        body = "\n\n".join(_paragraph(rng, rng.randint(3, 6)) for _ in range(rng.randint(2, 4)))
        table = ""
        if section % 3 == 0:
            rows = "\n".join(f"| {rng.choice(_WORDS['subject'])} | {rng.randint(10, 999)} | "
                             f"{rng.uniform(0, 100):.1f}% | {rng.randint(2015, 2026)} |"
                             for _ in range(rng.randint(5, 12)))
            table = f"\n\n| Program | Sites | Change | Year |\n|---|---:|---:|---:|\n{rows}"
        parts.append(heading + body + table + "\n")
        size += len(parts[-1])
        section += 1
    return "".join(parts)


def _search_results(rng: random.Random, query: str, n: int = 8) -> str:
    results = []
    for i in range(n):
        topic = rng.choice(_WORDS["topic"])
        results.append({
            "title": f"{topic.title()}: {rng.choice(_WORDS['object'])} {rng.choice(_WORDS['qualifier'])}",
            "url": f"https://example.org/{topic.replace(' ', '-')}/{rng.randint(10000, 99999)}",
            "snippet": _paragraph(rng, 2),
            "published": f"20{rng.randint(15, 26)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
            "rank": i + 1,
        })
    return json.dumps({"query": query, "results": results}, indent=2)


def _csv(rng: random.Random, rows: int) -> str:
    lines = ["date,site,tmax_c,tmin_c,precip_mm,canopy_pct,notes"]
    for i in range(rows):
        lines.append(f"2026-{rng.randint(5, 9):02d}-{rng.randint(1, 28):02d},site-{rng.randint(1, 40):02d},"
                     f"{rng.uniform(18, 41):.1f},{rng.uniform(8, 22):.1f},{rng.uniform(0, 12):.1f},"
                     f"{rng.uniform(5, 60):.1f},{rng.choice(['ok', 'sensor gap', 'estimated', 'revised'])}")
    return "\n".join(lines)


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required},
    }}


def _tools() -> list[dict]:
    s = {"type": "string"}
    i = {"type": "integer"}
    specs = [
        ("web_search", "Search the web. Returns titles, links and snippets; read a page with web_extract before citing it.",
         {"query": {**s, "description": "What to search for."}, "max_results": {**i, "description": "1-20."}}, ["query"]),
        ("web_extract", "Read one or more web pages as markdown. Every quotation must come from text this returns.",
         {"urls": {"type": "array", "items": s, "description": "Up to 5 links."}}, ["urls"]),
        ("read_file", "Read a text file with line numbers. Use offset and limit for long files.",
         {"path": s, "offset": i, "limit": i}, ["path"]),
        ("write_file", "Write a text file, replacing it. Prefer patch for small changes.",
         {"path": s, "content": s}, ["path", "content"]),
        ("patch", "Replace one exact string in a file. Fails when the string is missing or not unique.",
         {"path": s, "old_string": s, "new_string": s}, ["path", "old_string", "new_string"]),
        ("search_files", "Find files by name or content under a folder.",
         {"pattern": s, "path": s, "target": {"type": "string", "enum": ["content", "files"]}}, ["pattern"]),
        ("terminal", "Run a shell command and return its output. Long jobs go to the background.",
         {"command": s, "timeout": i, "background": {"type": "boolean"}}, ["command"]),
        ("notes_search", "Search the user's notes by keyword and meaning.",
         {"query": s, "folder": s, "limit": i}, ["query"]),
        ("notes_read", "Read one note in full, with its metadata.", {"title": s}, ["title"]),
        ("notes_propose", "Propose a change to a note. It waits for the user's approval.",
         {"title": s, "content": s, "reason": s}, ["title", "content", "reason"]),
        ("calendar_list", "List calendar events between two dates.", {"start": s, "end": s}, ["start", "end"]),
        ("weather", "The forecast and recent observations for a place.", {"place": s, "days": i}, ["place"]),
        ("image_search", "Find images on the web, with their source pages.", {"query": s, "limit": i}, ["query"]),
        ("vision_analyze", "Describe an image or answer a question about it.", {"image_url": s, "question": s}, ["image_url"]),
        ("memory", "Add, replace or remove one durable fact about the user or the environment.",
         {"action": {"type": "string", "enum": ["add", "replace", "remove"]}, "target": s, "content": s}, ["action"]),
        ("todo", "Keep the task list for this session.",
         {"todos": {"type": "array", "items": {"type": "object", "properties": {"content": s, "status": s}}}}, ["todos"]),
        ("delegate_task", "Hand a self-contained task to a helper with its own context.",
         {"goal": s, "context": s, "toolsets": {"type": "array", "items": s}}, ["goal"]),
        ("session_search", "Search earlier conversations.", {"query": s, "limit": i}, ["query"]),
        ("skill_view", "Load a skill's instructions.", {"name": s}, ["name"]),
        ("skills_list", "List the available skills with one line each.", {"category": s}, []),
        ("send_message", "Send a message to a channel. Only when the user asked for it.",
         {"target": s, "message": s}, ["target", "message"]),
        ("clarify", "Ask the user one question when the request is ambiguous.",
         {"question": s, "choices": {"type": "array", "items": s}}, ["question"]),
        ("execute_code", "Run a short Python script with the tools available as functions.", {"code": s}, ["code"]),
        ("text_to_speech", "Speak a short text aloud.", {"text": s, "voice": s}, ["text"]),
        ("cronjob", "Create, list or remove a scheduled job.",
         {"action": s, "schedule": s, "prompt": s, "job_id": s}, ["action"]),
    ]
    return [_tool(*spec) for spec in specs]


def _system_prompt(rng: random.Random, nonce: str) -> str:
    rules = [
        "Find sources before answering anything factual, and say when something comes from memory instead.",
        "Quote only text a tool returned, word for word, with its link.",
        "Prefer the user's own notes and documents to the web when both answer the question.",
        "Never change a note directly: propose the change and say what it is.",
        "Keep replies short. Lead with the answer, then the evidence, then any caveats.",
        "When a tool fails, say which and why in one line, then carry on with what you have.",
        "Ask before sending a message, booking anything, or deleting anything.",
        "Use metric units, and local time with the time zone.",
    ]
    skills = "\n".join(
        f"- {rng.choice(_WORDS['topic']).replace(' ', '-')}-{k}: {_sentence(rng)}" for k in range(60)
    )
    memory = "\n".join(f"- {_sentence(rng)}" for _ in range(25))
    return (
        f"Session {nonce}.\n\n"
        "You are a careful research assistant running on the user's own machines. You can search the web, "
        "read pages, read and search the user's notes and files, and run commands.\n\n"
        "## Rules\n" + "\n".join(f"{n}. {r}" for n, r in enumerate(rules, 1)) + "\n\n"
        "## Environment\n- Platform: a Unix workstation\n- Date: 2026-10-01, 09:00 local time\n\n"
        "## Skills (load one with skill_view before following it)\n" + skills + "\n\n"
        "## What you remember\n" + memory + "\n"
    )


def build_prefix(target_tokens: int, nonce: str, seed: int = 20261026) -> tuple[list[dict], list[dict]]:
    """Messages and tools of about `target_tokens`, opening with `nonce`."""
    rng = random.Random(seed)
    tools = _tools()
    messages: list[dict] = [
        {"role": "system", "content": _system_prompt(rng, nonce)},
        {"role": "user", "content": "I'm writing a short brief on cooling cities with street trees. "
                                    "Find what mid-sized cities have measured, read the best two or three "
                                    "sources, and compare them with the site data in my project folder."},
    ]
    target_chars = int(target_tokens * CHARS_PER_TOKEN)

    def size() -> int:
        return len(json.dumps(messages)) + len(json.dumps(tools))

    call = 0
    while size() < target_chars:
        call += 1
        step = call % 4
        if step == 1:
            name, args = "web_search", {"query": f"{rng.choice(_WORDS['topic'])} {rng.choice(_WORDS['object'])} study"}
            result = _search_results(rng, args["query"])
        elif step == 2:
            name, args = "web_extract", {"urls": [f"https://example.org/report/{rng.randint(100, 999)}"]}
            result = _article(rng, min(14000, max(2000, target_chars - size())))
        elif step == 3:
            name, args = "read_file", {"path": f"projects/cooling-brief/data/sites-{call:02d}.csv"}
            result = _csv(rng, 120)
        else:
            name, args = "notes_search", {"query": rng.choice(_WORDS["object"])}
            result = json.dumps({"hits": [{"title": f"Note {rng.randint(1, 400)}", "excerpt": _paragraph(rng, 2)}
                                          for _ in range(6)]}, indent=2)
        call_id = f"call_{call:03d}"
        messages.append({"role": "assistant", "content": "", "tool_calls": [{
            "id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]})
        messages.append({"role": "tool", "tool_call_id": call_id, "content": result})
    messages.append({"role": "assistant", "content":
                     "I have three usable sources and your site data. " + _paragraph(rng, 4)})
    return messages, tools


# -- metrics -----------------------------------------------------------------


def read_metrics(url: str, timeout: float = 10.0) -> dict[str, float]:
    """Prometheus text as {short name: value}, label sets summed.

    NInfer exports its own names (`ninfer_prompt_tokens_total`: every prompt
    token, cached or not) and llama.cpp-compatible ones beside them
    (`llamacpp:prompt_tokens_total`: only the tokens it prefilled). They mean
    different things, so each short name takes the first family present, in
    `_FAMILIES` order, and never a sum of two.
    """
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        text = resp.read().decode("utf-8", "replace")
    full: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[0].split("{", 1)[0]
        try:
            full[name] = full.get(name, 0.0) + float(parts[1])
        except ValueError:
            continue
    values: dict[str, float] = {}
    for family in _FAMILIES:
        for name, value in full.items():
            if name.startswith(family):
                values.setdefault(name[len(family):], value)
    return values


#: Metric name prefixes, most specific first: NInfer v0.16, NInfer v0.15, llama.cpp.
_FAMILIES = ("ninfer_", "ninfer:", "llamacpp:")


def busy(metrics: dict[str, float]) -> float:
    if "requests_running" in metrics:
        return metrics["requests_running"] + metrics.get("requests_waiting", 0.0)
    return metrics.get("requests_processing", 0.0) + metrics.get("requests_deferred", 0.0)


# -- requests ----------------------------------------------------------------


def stream_chat(url: str, payload: dict, *, token: str, hop: bool, timeout: float) -> dict:
    headers = {"Content-Type": "application/json"}
    if hop:
        headers[FALLBACK_HOP_HEADER] = "1"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = json.dumps(payload).encode()
    req = urllib.request.Request(f"{url.rstrip('/')}/v1/chat/completions", data=body, headers=headers, method="POST")
    started = time.monotonic()
    ttft = None
    content: list[str] = []
    usage = None
    finish = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        served_by = resp.headers.get("X-LLM-Served-By") or "backend"
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                text = delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning") or ""
                if (text or delta.get("tool_calls")) and ttft is None:
                    ttft = time.monotonic() - started
                if delta.get("content"):
                    content.append(delta["content"])
                finish = choice.get("finish_reason") or finish
    return {
        "ttft_s": round(ttft, 3) if ttft is not None else None,
        "total_s": round(time.monotonic() - started, 3),
        "content": "".join(content),
        "usage": usage,
        "finish_reason": finish,
        "served_by": served_by,
        "body_chars": len(body),
    }


def _delta(after: dict[str, float], before: dict[str, float], name: str) -> int:
    return int(after.get(name, 0.0) - before.get(name, 0.0))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8012", help="OpenAI-compatible base (a chat proxy or the backend)")
    ap.add_argument("--metrics-url", default="http://127.0.0.1:8010/metrics", help="the backend's own /metrics")
    ap.add_argument("--model", default="chat", help="alias to ask for (default chat: thinking off)")
    ap.add_argument("--prefix-tokens", type=int, default=30000)
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--shape", choices=("continue", "replace"), default="continue")
    ap.add_argument("--interleave", type=int, default=0,
                    help="other short conversations sent between the two requests (default 0)")
    ap.add_argument("--min-cached-share", type=float, default=0.9)
    ap.add_argument("--max-ttft", type=float, default=10.0)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--token", default=os.environ.get("PREFIX_TEST_TOKEN", ""), help="bearer token, if the URL needs one")
    ap.add_argument("--allow-fallback", action="store_true", help="omit the fallback-hop header")
    ap.add_argument("--force", action="store_true", help="run even if the backend is busy")
    ap.add_argument("--out", help="also write the JSON here")
    args = ap.parse_args(argv)

    result: dict = {
        "test": "ninfer-prefix", "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "url": args.url, "metrics_url": args.metrics_url, "model": args.model, "shape": args.shape,
    }

    def finish(code: int) -> int:
        text = json.dumps(result, indent=2)
        print(text)
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(text + "\n")
        return code

    try:
        m0 = read_metrics(args.metrics_url)
    except (OSError, urllib.error.URLError) as exc:
        result.update(result="SKIP", reason=f"metrics unreadable: {exc}")
        return finish(2)
    if "prompt_tokens_cached_total" not in m0:
        result.update(result="SKIP", reason="the backend's /metrics has no prompt_tokens_cached_total")
        return finish(2)
    if busy(m0) and not args.force:
        result.update(result="SKIP", reason=f"backend busy ({busy(m0):.0f} requests running or waiting)")
        return finish(2)
    result["backend_before"] = {k: m0[k] for k in (
        "prompt_tokens_total", "prompt_tokens_cached_total", "host_context_used_bytes",
        "host_context_capacity_bytes", "host_context_shortage_failures_total", "max_context_tokens",
    ) if k in m0}

    nonce = secrets.token_hex(8)
    prefix, tools = build_prefix(args.prefix_tokens, nonce)
    turn_a = {"role": "user", "content": "Before the comparison: in one short sentence, which source is strongest?"}
    turn_b = {"role": "user", "content": "Now in one short sentence: which of my sites ran hottest?"}
    base = {"model": args.model, "stream": True, "max_tokens": args.max_tokens, "temperature": 0, "tools": tools}

    try:
        first = stream_chat(args.url, {**base, "messages": prefix + [turn_a]},
                            token=args.token, hop=not args.allow_fallback, timeout=args.timeout)
        m1 = read_metrics(args.metrics_url)
        others = []
        for k in range(args.interleave):
            other_prefix, _ = build_prefix(1500, secrets.token_hex(8), seed=k + 1)
            others.append(stream_chat(args.url, {**base, "max_tokens": 8, "messages": other_prefix[:2]},
                                      token=args.token, hop=not args.allow_fallback, timeout=args.timeout)["ttft_s"])
        m1b = read_metrics(args.metrics_url) if others else m1
        if args.shape == "continue":
            second_messages = prefix + [turn_a, {"role": "assistant", "content": first["content"]}, turn_b]
        else:
            second_messages = prefix + [turn_b]
        second = stream_chat(args.url, {**base, "messages": second_messages},
                             token=args.token, hop=not args.allow_fallback, timeout=args.timeout)
        m2 = read_metrics(args.metrics_url)
    except urllib.error.HTTPError as exc:
        result.update(result="SKIP", reason=f"HTTP {exc.code}: {exc.read()[:300].decode('utf-8', 'replace')}")
        return finish(2)
    except (OSError, urllib.error.URLError) as exc:
        result.update(result="SKIP", reason=f"request failed: {exc}")
        return finish(2)

    first_prompt = _delta(m1, m0, "prompt_tokens_total")
    second_prompt = _delta(m2, m1b, "prompt_tokens_total")
    cached_first = _delta(m1, m0, "prompt_tokens_cached_total")
    cached_second = _delta(m2, m1b, "prompt_tokens_cached_total")
    prefix_tokens = first_prompt
    if args.shape == "replace":
        prefix_tokens = max(0, first_prompt - round(len(turn_a["content"]) / 3) - 8)
    other_requests = ((_delta(m2, m0, "requests_total") - 2 - len(others))
                      if "requests_total" in m2 else None)
    share = cached_second / prefix_tokens if prefix_tokens else 0.0
    ttft_ok = second["ttft_s"] is not None and second["ttft_s"] < args.max_ttft
    passed = share >= args.min_cached_share and ttft_ok

    for r in (first, second):
        r.pop("content", None)
    result.update({
        "result": "PASS" if passed else "FAIL",
        "prefix_tokens": prefix_tokens,
        "first": {**first, "prompt_tokens": first_prompt, "cached_tokens": cached_first},
        "second": {**second, "prompt_tokens": second_prompt, "cached_tokens": cached_second},
        "interleaved": {"conversations": len(others), "ttft_s": others} if others else None,
        "cached_delta": cached_second,
        "cached_share": round(share, 4),
        "ttft_speedup": (round(first["ttft_s"] / second["ttft_s"], 2)
                         if first["ttft_s"] and second["ttft_s"] else None),
        "criteria": {"min_cached_share": args.min_cached_share, "max_ttft_s": args.max_ttft},
        "other_requests_during_test": other_requests,
        "backend_after": {k: m2[k] for k in (
            "prompt_tokens_cached_total", "host_context_used_bytes",
            "host_context_shortage_failures_total", "snapshot_restores_total",
        ) if k in m2},
    })
    if other_requests:
        result["warning"] = "other requests ran during the test, so the token counts include them"
    return finish(0 if passed else 1)


if __name__ == "__main__":
    sys.exit(main())
