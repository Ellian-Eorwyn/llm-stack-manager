#!/usr/bin/env python3
"""What a backend slot is, independently of which engine serves it.

Ten launcher scripts used to answer this question by example. They were between
66 and 325 lines each, all of the same shape -- source the env file, resolve a
chain of fallbacks, assemble an `OPTS=()` array, exec llama-server -- and
between them they duplicated the same twenty-odd flags with the same defaults
under different prefixes.

That duplication is also what blocked a second engine, because the point where
llama.cpp and MLX diverge *is* the flag assembly. So the slots are described
here as data, and each engine turns that description into a command line.

The description is deliberately literal about the env fallback chains rather
than tidying them. `${EMBED_UBATCH_SIZE:-${CHAT_UBATCH_SIZE:-512}}` is a real
chain a real deployment relies on, and "the embedding slot falls back to the
chat slot's micro-batch" is a fact about the configuration surface, not an
accident to be normalised away.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Flag:
    """A `--flag value` pair, resolved from the first env key that is set.

    `keys` are tried in order and are *suffixes* unless they start with "!",
    which marks an absolute key that is not prefixed by the slot. The empty
    string as a default means the flag is omitted when nothing is set, which is
    different from passing an empty value: llama.cpp reads `--tensor-split ""`
    as an explicit empty split and refuses it.
    """

    name: str
    keys: tuple[str, ...]
    default: str | None = None

    def resolve(self, env: dict, prefix: str) -> list[str]:
        for key in self.keys:
            absolute = key.startswith("!")
            name = key[1:] if absolute else f"{prefix}_{key}"
            value = env.get(name)
            if value not in (None, ""):
                return [self.name, str(value)]
        if self.default in (None, ""):
            return []
        return [self.name, self.default]


@dataclass(frozen=True)
class Toggle:
    """A valueless flag, emitted when its setting matches `when`.

    `otherwise` covers the pairs llama.cpp states both ways round --
    `--kv-offload` / `--no-kv-offload` -- where leaving the flag off is not the
    same as passing its negation.
    """

    name: str
    key: str
    when: str = "on"
    default: str = "off"
    otherwise: str | None = None

    def resolve(self, env: dict, prefix: str) -> list[str]:
        value = str(env.get(f"{prefix}_{self.key}") or self.default).strip()
        if value == self.when:
            return [self.name]
        return [self.otherwise] if self.otherwise else []


@dataclass(frozen=True)
class Slot:
    """One servable position in the stack, and how it is configured."""

    name: str
    prefix: str
    #: Absolute env keys holding the model path, in fallback order.
    model_keys: tuple[str, ...]
    alias_default: str
    port_default: str = ""
    host_keys: tuple[str, ...] = ("!LISTEN_HOST",)
    #: Which engine serves it. The env key defaults to `{PREFIX}_ENGINE`, so
    #: every slot is switchable without having to be listed here; set it
    #: explicitly only where the key does not follow the prefix.
    engine_key: str = ""
    engine_default: str = "llamacpp"
    #: Suffix -> default, overriding COMMON_FLAGS' defaults for this slot.
    defaults: dict[str, str] = field(default_factory=dict)
    #: Common flags this slot does not take at all.
    omit: frozenset[str] = frozenset()
    #: Emitted verbatim, after the common flags.
    literals: tuple[str, ...] = ()
    extra_toggles: tuple[Toggle, ...] = ()
    #: JSON list of extra arguments, appended last.
    custom_args_key: str = ""

    def engine(self, env: dict) -> str:
        key = self.engine_key or f"{self.prefix}_ENGINE"
        return str(env.get(key) or self.engine_default).strip()
