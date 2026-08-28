#!/usr/bin/env python3
"""NVIDIA driver and toolkit probing.

Split out of `linux.py` so the two pure functions -- parsing `nvidia-smi`'s
table and choosing a toolkit against a driver's reported CUDA version -- can be
tested without a GPU, which is what `tests/test_setup_engine.py` has always
done. `setup_engine` re-exports them under their original names.
"""

from __future__ import annotations

import re

# Newest first: the driver reports the highest CUDA it supports, and anything
# at or below that works.
SUPPORTED_TOOLKITS = ("13.3", "13.0", "12.8")


def parse_gpus(text: str) -> list[dict]:
    gpus: list[dict] = []
    for line in (text or "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            compute_cap = parts[2]
            gpus.append({
                "index": int(parts[0]),
                "name": parts[1],
                "compute_capability": compute_cap,
                # The build reads this: nvcc wants "86", not "8.6".
                "cmake_architecture": compute_cap.replace(".", ""),
                "memory_total_mib": int(float(parts[3])),
                "memory_free_mib": int(float(parts[4])),
            })
        except ValueError:
            continue
    return gpus


def driver_cuda_version(text: str) -> str:
    match = re.search(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)?)", text or "")
    return match.group(1) if match else ""


def choose_toolkit(driver_cuda: str, supported: tuple[str, ...] = SUPPORTED_TOOLKITS) -> str:
    """The newest supported toolkit the driver can run, or "".

    A driver reports the highest CUDA version it supports; installing a newer
    toolkit than that produces binaries the driver refuses to load.
    """
    try:
        maximum = tuple(int(part) for part in (driver_cuda or "").split("."))
    except ValueError:
        return ""
    if not maximum:
        return ""
    for candidate in supported:
        parsed = tuple(int(part) for part in candidate.split("."))
        if parsed <= maximum:
            return candidate
    return ""


def probe(platform) -> tuple[list[dict], str, str]:
    """(gpus, driver CUDA version, error) from `nvidia-smi`."""
    try:
        result = platform.run_cmd([
            "nvidia-smi",
            "--query-gpu=index,name,compute_cap,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ], timeout=10)
        gpus = parse_gpus(result.stdout) if result.returncode == 0 else []
        summary = platform.run_cmd(["nvidia-smi"], timeout=10)
        return gpus, driver_cuda_version(summary.stdout + summary.stderr), (result.stderr or "").strip()
    except Exception as exc:
        return [], "", str(exc)
