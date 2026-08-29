#!/usr/bin/env python3
"""The pooled member list, for the shell half.

`web/backends/router.py` decides; this only prints. Four shell scripts carried
their own copy of the default string, which is four chances to disagree about
whether a model is in the pool.

    MEMBERS="$(router-members.py)"        # EMBED,OCR,RERANK,TASK
    router-members.py --list             # one per line
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.environ.get("STACK_DIR", "."), "web"))

from backends.router import members_string, pooled_members  # noqa: E402


def main(argv: list[str]) -> int:
    env = dict(os.environ)
    if "--list" in argv:
        for member in pooled_members(env):
            print(member)
    else:
        print(members_string(env))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
