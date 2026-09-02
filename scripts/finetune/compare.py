#!/usr/bin/env python3
"""Prove an adapter is doing something: same prompt, adapter off then on.

    compare.py --run <name> --slot llm-a

A trained adapter and a working adapter are different claims. An adapter can
convert cleanly, load cleanly, appear in `GET /lora-adapters`, and change
nothing at all — because its scale is zero, because it was built against a
different base, or because the tensors it names do not exist in the served
model. The only test that separates those is generating the same prompt twice
with the scale moved, and reading both.

Identical output is the failure this exists to catch.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
from common import RunError, open_run, read_jsonl  # noqa: E402

SLOT_PORTS = {"llm-a": "8010", "llm-b": "8020", "task": "8007"}

DEFAULT_PROMPT = ("Write a short section on the moral economy of repair, "
                  "in your usual register.")


def _post(url: str, payload: dict, timeout: int = 300):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
    return json.loads(body) if body else {}


def _get(url: str, timeout: int = 15):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run")
    parser.add_argument("--slot", default="llm-a", choices=sorted(SLOT_PORTS))
    parser.add_argument("--base-url", help="overrides --slot")
    parser.add_argument("--adapter", help="match by name; default is the run's")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--prompt")
    parser.add_argument("--system")
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    base_url = args.base_url or f"http://127.0.0.1:{SLOT_PORTS[args.slot]}"
    system = args.system
    prompt = args.prompt or DEFAULT_PROMPT

    if args.run:
        run = open_run(args.run)
        system = system or run.recipe.get("system")
        if not args.prompt:
            held = read_jsonl(run.evalset)
            for row in held:
                for message in row.get("messages", []):
                    if message["role"] == "user":
                        prompt = message["content"]
                        break
                break

    try:
        adapters = _get(f"{base_url}/lora-adapters")
    except (urllib.error.URLError, OSError) as exc:
        return common.fail(f"{args.slot} is not answering on {base_url}: {exc}")
    if not adapters:
        return common.fail(
            f"{args.slot} has no adapters loaded. Set its LoRA Adapters field "
            f"and restart it — an adapter has to be named at launch to be "
            f"switchable at runtime.")

    wanted = args.adapter or (args.run or "")
    chosen = next((a for a in adapters if wanted and wanted in str(a.get("path", ""))),
                  adapters[0])
    print(f"Adapter: {chosen['path']}  (id {chosen['id']}, currently at "
          f"scale {chosen['scale']})")
    if len(adapters) > 1:
        print(f"  {len(adapters)} loaded; the others are held at 0 for this test.")
    print(f"Backend: {base_url}    prompt: {prompt[:80]}...")
    print()

    messages = ([{"role": "system", "content": system}] if system else [])
    messages.append({"role": "user", "content": prompt})

    # This runs against a backend someone may be using. Whatever the scales
    # were before the test, they are what they go back to after it -- a
    # diagnostic that leaves production in a different state than it found it
    # is a change, not a measurement.
    original = [{"id": a["id"], "scale": a["scale"]} for a in adapters]

    def restore() -> None:
        try:
            _post(f"{base_url}/lora-adapters", original)
        except (urllib.error.URLError, OSError):
            print(f"WARNING: could not restore the original scales {original}. "
                  f"Set them from the Services page.")

    outputs, reasoning = {}, {}
    for label, scale in (("off", 0.0), ("on", args.scale)):
        _post(f"{base_url}/lora-adapters",
              [{"id": a["id"], "scale": scale if a["id"] == chosen["id"] else 0.0}
               for a in adapters])
        try:
            answer = _post(f"{base_url}/v1/chat/completions", {
                "messages": messages, "temperature": 0, "seed": args.seed,
                "max_tokens": args.max_tokens})
        except (urllib.error.URLError, OSError) as exc:
            restore()
            return common.fail(f"generation failed at scale {scale}: {exc}")
        message = (answer.get("choices") or [{}])[0].get("message", {})
        outputs[label] = (message.get("content") or "").strip()
        reasoning[label] = len((message.get("reasoning_content") or "").strip())

    restore()

    for label, scale in (("off", 0.0), ("on", args.scale)):
        print(f"===== adapter {label} (scale {scale}) " + "=" * 30)
        print(outputs[label][:900] or
              f"(empty — the model spent all {args.max_tokens} tokens on "
              f"{reasoning[label]} characters of reasoning and never reached an "
              f"answer)")
        print()

    # An empty side makes the comparison meaningless, and calling it "inert"
    # would be a false accusation against a working adapter. A thinking model
    # can spend the whole budget reasoning, and it does so more often with the
    # adapter off, which is exactly when this would misfire.
    empty = [label for label in ("off", "on") if not outputs[label]]
    if empty:
        return common.fail(
            f"inconclusive: the {' and '.join(empty)} side produced no answer "
            f"within --max-tokens {args.max_tokens}, because the reasoning block "
            f"used the whole budget. Re-run with a larger --max-tokens; this is "
            f"not evidence about the adapter either way.")

    if outputs["off"] == outputs["on"]:
        return common.fail(
            "the two outputs are identical, so the adapter changed nothing. It "
            "is loaded but inert: check it was converted against the base this "
            "backend serves, and that the scale actually moved.")
    print("The outputs differ — the adapter is applied and changing generation.")
    print(f"Scales restored to what they were: "
          + ", ".join(f"id {a['id']} at {a['scale']}" for a in original))
    print("Change them from the Services page, or POST /lora-adapters.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RunError as exc:
        sys.exit(common.fail(str(exc)))
