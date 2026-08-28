#!/usr/bin/env python3
r"""What the launcher needs to know about a slot, as shell assignments.

`start-backend.sh` used to ask twelve separate questions with
`eval "printf '%s' \"\${${PREFIX}_X:-}\""`, which reads exactly one prefix. The
primary chat slot has two, and the five keys once spelled `CHAT_DENSE_*` have
three, so those expansions would have resolved the placement and the memory-fit
report differently from the command they describe.

So the registry answers instead, and the shell evals the result. One reader,
one set of fallback chains: the shell and the command builder cannot disagree.

    eval "$(slot-facts.py embed)"
    eval "$(slot-facts.py embed --tensor-split 1,1 --devices 2)"

The second form fills in the two report settings the launcher resolves rather
than reads.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys

sys.path.insert(0, os.path.join(os.environ.get("STACK_DIR", "."), "web"))

from backends.slots import COMMON_FLAGS, SLOTS  # noqa: E402
from backends.spec import lookup  # noqa: E402

#: The report settings that are not `COMMON_FLAGS` entries, with the defaults
#: the launchers applied to them. Everything else takes its default from the
#: flag it describes, so there is one place a default lives.
_EXTRA_DEFAULTS = {"SWA_FULL": "off", "SPEC_METHOD": "off", "FIT_CTX": ""}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("slot")
    parser.add_argument("--tensor-split", default="")
    parser.add_argument("--devices", default="")
    args = parser.parse_args()

    slot = SLOTS.get(args.slot)
    if slot is None:
        return 2
    env = os.environ
    prefixes = slot.prefixes
    resolved = {"tensor_split": args.tensor_split, "devices": args.devices}

    def read(keys, default="", empty_is_set=False):
        return lookup(env, keys, prefixes, empty_is_set) or default

    facts = {
        "FACT_PREFIX": slot.prefix,
        "FACT_ENGINE": slot.engine(env) or "llamacpp",
        "FACT_MODEL": read(slot.model_keys),
        "FACT_MMPROJ": read(slot.mmproj_keys, empty_is_set=True),
        "FACT_SPLIT_MODE": read(("SPLIT_MODE",), "layer"),
        "FACT_TENSOR_SPLIT": read(("TENSOR_SPLIT",)),
        "FACT_MAIN_GPU": read(("MAIN_GPU",)),
        "FACT_FLASH_ATTN": read(("FLASH_ATTN",), "on"),
        "FACT_VISIBLE": read(("GPU_VISIBLE_DEVICES",)),
        "FACT_SWA_FULL": read(("SWA_FULL",), "off"),
        "FACT_BUDGET_NAME": slot.budget,
    }
    for key, value in facts.items():
        print(f"{key}={shlex.quote(value)}")

    defaults = {flag.keys[0]: (flag.default or "") for flag in COMMON_FLAGS}
    defaults.update(_EXTRA_DEFAULTS)
    defaults.update(slot.defaults)

    settings = []
    for label, suffix in slot.preflight_fields:
        value = resolved[label] if not suffix else read((suffix,), defaults.get(suffix, ""))
        settings.append(f"{label}={value}")
    print("FACT_PREFLIGHT=({})".format(" ".join(shlex.quote(s) for s in settings)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
