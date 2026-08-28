#!/usr/bin/env python3
"""Turning a slot description into an MLX server command line.

Much shorter than the llama.cpp builder, and not because it is unfinished: the
MLX servers take a host and a port and read everything else from the
environment they are launched with. There is no flag surface to assemble.
"""

from __future__ import annotations

from .spec import Slot, lookup

#: Slot -> the server script that serves it under MLX.
SERVERS = {
    "embed": "mlx-embedding-server.py",
}


def build(slot: Slot, env: dict, extra: list[str] | None = None) -> list[str]:
    script = SERVERS.get(slot.name)
    if script is None:
        raise SystemExit(
            f"{slot.name}: the mlx engine has no server for this slot "
            f"(have: {', '.join(sorted(SERVERS))})")

    stack = str(env.get("STACK_DIR") or ".")
    venv = str(env.get("MLX_RUNTIME_VENV") or f"{stack}/deps/mlx-runtime-venv")
    host = str(env.get("LISTEN_HOST") or "127.0.0.1")
    port = lookup(env, slot.port_keys, slot.prefixes) or slot.port_default
    return [f"{venv}/bin/python", f"{stack}/scripts/{script}",
            "--host", host, "--port", port, *(extra or [])]
