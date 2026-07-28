#!/usr/bin/env python3
"""Render /api/backend/telemetry as a terminal summary for `llm-stack-manager status`.

Reads the JSON payload on stdin. Anything missing is skipped rather than
guessed, so a partially-degraded stack still prints what it does know.
"""

import json
import sys


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0

    host = data.get("host") or {}
    if host.get("mem_total_mib"):
        line = f"Host RAM {host.get('mem_used_pct')}% of {host['mem_total_mib']} MiB"
        if host.get("swap_total_mib"):
            line += f" | swap {host.get('swap_used_pct')}% of {host['swap_total_mib']} MiB"
        print(line)

    for gpu in data.get("gpus") or []:
        total = gpu.get("mem_total") or 0
        used = gpu.get("mem_used") or 0
        print(f"GPU {gpu.get('index')}    {used}/{total} MiB used, "
              f"{total - used} MiB free, {gpu.get('util')}% util")

    for backend in data.get("backends") or []:
        if not backend.get("active"):
            continue
        props = backend.get("props") or {}
        stats = backend.get("stats") or {}
        generation = (stats.get("throughput") or {}).get("generation_tps") or {}
        cache = stats.get("cache") or {}

        print()
        print(f"{backend.get('label')} ({backend.get('unit') or '?'})")
        per_slot = props.get("n_ctx_per_slot")
        if per_slot:
            # Per slot, not total: this is the limit a single request hits.
            print(f"  context   {per_slot:,} per slot x {props.get('total_slots', 1)}")
        if generation.get("p50") is not None:
            print(f"  gen tok/s p50 {generation['p50']}  p90 {generation.get('p90')}")
        if cache.get("evictions_per_launch") is not None:
            print(f"  cache     {cache['evictions_per_launch']} evictions/launch "
                  f"over {cache.get('launches', 0)} launches")

    warnings = data.get("warnings") or []
    if warnings:
        print()
        for warning in warnings:
            print(f"  [{warning.get('level')}] {warning.get('text')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
