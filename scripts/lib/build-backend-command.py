#!/usr/bin/env python3
"""Print a slot's argv, NUL-separated, for the launcher to exec.

A file rather than a heredoc inside start-backend.sh, so it can be imported and
tested directly, and so a mistake in it is a Python error with a line number
rather than a shell quoting puzzle.

Reads the slot from LLM_BACKEND_SLOT and the settings from the environment,
which the launcher has already populated by sourcing the env file. Extra
arguments are appended, preserving the `"$@"` passthrough every launcher had.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.environ.get("STACK_DIR", "."), "web"))

import backends  # noqa: E402


def main() -> int:
    slot = os.environ.get("LLM_BACKEND_SLOT", "")
    if slot not in backends.SLOTS:
        print(f"unknown backend slot {slot!r}; "
              f"expected one of {', '.join(sorted(backends.SLOTS))}", file=sys.stderr)
        return 2
    argv = backends.build_command(slot, os.environ, sys.argv[1:])
    # Every element is NUL-*terminated*, not NUL-separated: bash's
    # `read -r -d ''` needs the delimiter after the last token too, and joining
    # instead of terminating silently drops the final argument.
    for argument in argv:
        sys.stdout.write(argument + "\0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
