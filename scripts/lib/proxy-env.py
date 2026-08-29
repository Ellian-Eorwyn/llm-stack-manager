#!/usr/bin/env python3
"""The environment one proxy should run with, as shell assignments.

The twin of `slot-facts.py`, and for the same reason: two hand-copied launchers
resolved the same settings independently and had already drifted -- one of them
exported three aliases the other did not, and neither exported
`THINK_REASONING_EFFORT` at all.

    eval "$(proxy-env.py llm-a-proxy)"

`web/backends/proxies.py` decides; this only formats.
"""

from __future__ import annotations

import os
import shlex
import sys

sys.path.insert(0, os.path.join(os.environ.get("STACK_DIR", "."), "web"))

from backends.proxies import PROXIES, resolve  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] not in PROXIES:
        print(f"usage: {argv[0]} <{'|'.join(PROXIES)}>", file=sys.stderr)
        return 2
    for key, value in resolve(PROXIES[argv[1]], dict(os.environ)).items():
        print(f"export {key}={shlex.quote(value)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
