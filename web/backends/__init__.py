#!/usr/bin/env python3
"""Backend engines, and the slots they serve.

`slots.SLOTS` describes each servable position in the stack as data; an engine
module turns that description into a command line. The split exists because the
point where llama.cpp and MLX diverge is exactly the flag assembly, which was
previously duplicated across ten launcher scripts.
"""

from . import llamacpp, mlx, options, slots, spec, speculative
from .slots import SLOTS
from .spec import Flag, Slot, Toggle

ENGINES = {
    "llamacpp": llamacpp,
    "mlx": mlx,
}


def build_command(slot_name: str, env: dict, extra=None, said=None) -> list[str]:
    """The argv for a slot, from whichever engine is configured to serve it.

    `said` collects the messages the launcher would have echoed. Only the
    llama.cpp engine has any.
    """
    slot = SLOTS[slot_name]
    engine_name = slot.engine(env)
    engine = ENGINES.get(engine_name)
    if engine is None:
        raise SystemExit(
            f"{slot_name}: unknown engine {engine_name!r}; "
            f"expected one of {', '.join(sorted(ENGINES))}")
    if engine is llamacpp:
        return engine.build(slot, env, extra, said)
    return engine.build(slot, env, extra)


__all__ = ["ENGINES", "Flag", "SLOTS", "Slot", "Toggle", "build_command",
           "llamacpp", "mlx", "options", "slots", "spec", "speculative"]
