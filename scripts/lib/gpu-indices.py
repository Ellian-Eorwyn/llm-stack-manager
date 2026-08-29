#!/usr/bin/env python3
"""Renumbered GPU indices, and what it takes to stop renumbering.

`start-backend.sh` exports `CUDA_VISIBLE_DEVICES` from each slot's own
`<PREFIX>_GPU_VISIBLE_DEVICES`, so llama.cpp sees a *renumbered* device list and
`MAIN_GPU` indexes into that rather than into the machine. A slot with
`GPU_VISIBLE_DEVICES=1` and `MAIN_GPU=0` runs on physical GPU 1; the same
`MAIN_GPU=0` under the router, which sets its own visible list, runs on physical
GPU 0. One number, two meanings, decided by which unit happens to start it.

Absolute indices fix that, and cannot simply be switched on: the stored values
were written in renumbered space, so changing what they mean moves models
between cards. On the host this was written for, `rerank` and `ocr` would both
have jumped from GPU 1 to GPU 0 without anything in the config changing.

So the switch is `LLM_ABSOLUTE_GPU_INDICES`, off by default, and this reports
what flipping it would move and what the values become:

    gpu-indices.py --report     # what moves, per slot
    gpu-indices.py --migrate    # the absolute values to write

Nothing here writes config. The operator does, having read the report.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "web"))

import backends  # noqa: E402
import config_env  # noqa: E402


def _visible(env: dict, prefix: str) -> list[int]:
    raw = str(env.get(f"{prefix}_GPU_VISIBLE_DEVICES", "") or "").strip()
    if not raw:
        return []
    out = []
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            out.append(int(part))
    return out


def translate(env: dict) -> list[dict]:
    """Per slot: the physical GPU it uses now, and the absolute values for it.

    `main_gpu` is an index into the visible list, so the physical card is
    `visible[main_gpu]`. `tensor_split` is one weight per visible device, and
    absolute space has one per device on the machine -- so a split written for a
    one-device view has to be widened, with zeros for the cards it never used.
    """
    rows = []
    for name, slot in backends.SLOTS.items():
        prefix = slot.prefix
        visible = _visible(env, prefix)
        if not visible:
            # Nothing renumbered, so nothing to translate.
            rows.append({"slot": name, "prefix": prefix, "visible": [],
                         "moves": False, "main_gpu": None, "tensor_split": None})
            continue
        raw_main = str(env.get(f"{prefix}_MAIN_GPU", "") or "0").strip()
        index = int(raw_main) if raw_main.lstrip("-").isdigit() else 0
        physical = visible[index] if 0 <= index < len(visible) else visible[0]

        split_raw = str(env.get(f"{prefix}_TENSOR_SPLIT", "") or "").strip()
        absolute_split = None
        if split_raw and split_raw != "auto":
            weights = [w.strip() for w in split_raw.split(",") if w.strip()]
            width = max(visible) + 1
            widened = ["0"] * width
            for slot_index, weight in enumerate(weights):
                if slot_index < len(visible):
                    widened[visible[slot_index]] = weight
            absolute_split = ",".join(widened)

        rows.append({
            "slot": name, "prefix": prefix, "visible": visible,
            "moves": physical != index,
            "main_gpu": physical, "tensor_split": absolute_split,
        })
    return rows


def main(argv: list[str]) -> int:
    env = config_env.read_env()
    rows = translate(env)
    migrate = "--migrate" in argv
    if migrate:
        for row in rows:
            if not row["visible"]:
                continue
            print(f"{row['prefix']}_MAIN_GPU={row['main_gpu']}")
            if row["tensor_split"] is not None:
                print(f"{row['prefix']}_TENSOR_SPLIT={row['tensor_split']}")
            print(f"{row['prefix']}_GPU_VISIBLE_DEVICES=")
        return 0

    moving = [r for r in rows if r["moves"]]
    print("Slot        visible   main_gpu now -> absolute   moves?")
    for row in rows:
        if not row["visible"]:
            print(f"  {row['slot']:<10} (not renumbered)")
            continue
        print(f"  {row['slot']:<10} {','.join(map(str, row['visible'])):<9} "
              f"{row['main_gpu']:<24} {'YES' if row['moves'] else 'no'}")
    print()
    if moving:
        print(f"{len(moving)} slot(s) would change card unless migrated: "
              f"{', '.join(r['slot'] for r in moving)}")
        print("Run with --migrate to print the absolute values that keep them put.")
    else:
        print("No slot changes card; every visible list is already the identity.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
