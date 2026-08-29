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


def lookup(env: dict, keys: tuple[str, ...], prefixes: tuple[str, ...],
           empty_is_set: bool = False) -> str | None:
    """The first of `keys` that is set, or None.

    A key starting with "!" is absolute. Anything else is a *suffix*, tried
    under each of the slot's prefixes in order -- which is how the primary chat
    slot reads `LLM_A_TEMP` and then `CHAT_TEMP` from one entry.

    `empty_is_set` is the difference between `${X:-d}` and `${X-d}`, and it is
    not a nicety. The launchers use the first for required settings, so an
    empty value lands on a working default, and the second for the paths and
    tuning knobs where "unset" is a legitimate choice, so clearing the new key
    means cleared rather than inheriting whatever the legacy key still holds.
    Getting that backwards is why `--fit-ctx` kept being passed alongside
    `--fit off` long after it had been cleared in the UI.
    """
    for key in keys:
        names = [key[1:]] if key.startswith("!") else [f"{p}_{key}" for p in prefixes]
        for name in names:
            value = env.get(name)
            if value is None:
                continue
            if value == "" and not empty_is_set:
                continue
            return str(value)
    return None


@dataclass(frozen=True)
class Flag:
    """A `--flag value` pair, resolved from the first env key that is set.

    The empty string as a default means the flag is omitted when nothing is
    set, which is different from passing an empty value: llama.cpp reads
    `--tensor-split ""` as an explicit empty split and refuses it.
    """

    name: str
    keys: tuple[str, ...]
    default: str | None = None
    #: Whether an explicitly emptied key means "cleared" rather than "unset".
    #: See `lookup`.
    empty_is_set: bool = False

    def resolve(self, env: dict, prefixes: tuple[str, ...], ctx=None) -> list[str]:
        value = lookup(env, self.keys, prefixes, self.empty_is_set)
        if value is None:
            value = self.default
        if value in (None, ""):
            return []
        return [self.name, str(value)]


@dataclass(frozen=True)
class Toggle:
    """A valueless flag, emitted when its setting matches `when`.

    `otherwise` covers the pairs llama.cpp states both ways round --
    `--kv-offload` / `--no-kv-offload` -- where leaving the flag off is not the
    same as passing its negation.

    One key, no chain: every toggle in the tree resolves `${NEW:-${OLD:-d}}`
    over the slot's prefixes and nothing else needs saying. Give it a `keys`
    tuple when something actually needs one.
    """

    name: str
    key: str
    when: str = "on"
    default: str = "off"
    otherwise: str | None = None

    def resolve(self, env: dict, prefixes: tuple[str, ...], ctx=None) -> list[str]:
        value = str(lookup(env, (self.key,), prefixes) or self.default).strip()
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
    #: Prefixes this slot still answers to, tried after `prefix`. The primary
    #: chat slot is `LLM_A` and reads `CHAT_*` behind it, which is what
    #: makes a config written before the rename keep working.
    legacy_prefixes: tuple[str, ...] = ()
    host_keys: tuple[str, ...] = ("!LISTEN_HOST",)
    port_keys: tuple[str, ...] = ("PORT",)
    alias_keys: tuple[str, ...] = ("MODEL_NAME",)
    mmproj_keys: tuple[str, ...] = ("MMPROJ_PATH",)
    #: Which engine serves it. The env key defaults to `{PREFIX}_ENGINE`, so
    #: every slot is switchable without having to be listed here; set it
    #: explicitly only where the key does not follow the prefix.
    engine_key: str = ""
    engine_default: str = "llamacpp"
    #: Suffix -> default, overriding COMMON_FLAGS' defaults for this slot.
    defaults: dict[str, str] = field(default_factory=dict)
    #: Suffix -> the whole key chain, for the few flags whose fallbacks this
    #: slot does not share with the rest. Only the five keys that were once
    #: spelled `CHAT_DENSE_*` need one, so listing them beats giving every
    #: flag on the slot a lookup that cannot match. Write these absolute: a
    #: relative key is tried under every prefix before the next entry, so a
    #: mixed chain puts the legacy prefix ahead of the key after it.
    key_chains: dict[str, tuple[str, ...]] = field(default_factory=dict)
    #: Common flags this slot does not take at all.
    omit: frozenset[str] = frozenset()
    #: Emitted verbatim, after the common flags.
    literals: tuple[str, ...] = ()
    #: Everything after the common toggles, in order. Toggles, flags, and the
    #: decided arguments in `backends/options.py` -- anything with a
    #: `resolve(env, prefixes, ctx)`. The order here is the order on the
    #: command line, which is why it is a tuple and not a set of features.
    tail: tuple = ()
    #: JSON list of extra arguments, appended last. Empty means the slot
    #: offers none.
    custom_args_keys: tuple[str, ...] = ()
    #: What `budget.py` calls this slot. Not always the slot's own name: the
    #: budget model knows `llm-a` where the unit is `llm-a`.
    budget_name: str = ""
    #: The settings the memory-fit report carries, in order. A pair is
    #: (label, key suffix); an empty suffix marks one the launcher supplies,
    #: because it is resolved rather than read -- the tensor split after `auto`
    #: has been expanded, and the number of visible devices.
    preflight_fields: tuple[tuple[str, str], ...] = ()

    # -- how the rest of the stack refers to this slot -----------------------
    #
    # Four vocabularies name the same thing, and no single one of them will do:
    # the unit is `llm-a`, the budget model calls it
    # `llm-a`, the setup wizard calls it `primary`, and its settings
    # live under `LLM_A_`. Each of those was written out independently
    # in a different module, which is why renaming a slot used to mean editing
    # twelve files.

    #: What the setup wizard calls it. Empty for a slot the wizard cannot
    #: install on its own.
    component: str = ""
    #: "chat" or "auxiliary", for the services panel's two groups.
    group: str = "auxiliary"
    label: str = ""
    desc: str = ""
    config_section: str = ""
    #: The human port string the services panel shows. Not machine-readable --
    #: `health` derives its port map from `port_keys` for that reason.
    ports_display: str = ""
    #: Where telemetry and health dial it, which is not always where it binds:
    #: a slot may listen on `LISTEN_HOST` (0.0.0.0) and still be probed on
    #: loopback. Empty means loopback.
    probe_host_keys: tuple[str, ...] = ()

    @property
    def prefixes(self) -> tuple[str, ...]:
        return (self.prefix, *self.legacy_prefixes)

    @property
    def budget(self) -> str:
        return self.budget_name or self.name

    @property
    def port_key(self) -> str:
        """The single env key holding this slot's port, for the tables that
        want a name rather than a chain."""
        keys = self.absolute(self.port_keys)
        return keys[0] if keys else ""

    @property
    def probe_host_key(self) -> str:
        keys = self.absolute(self.probe_host_keys)
        return keys[0] if keys else ""

    def absolute(self, keys: tuple[str, ...]) -> tuple[str, ...]:
        """`keys` as whole env names, expanding relative ones over the prefixes."""
        out: list[str] = []
        for key in keys:
            if key.startswith("!"):
                out.append(key[1:])
            else:
                out.extend(f"{prefix}_{key}" for prefix in self.prefixes)
        return tuple(out)

    def engine(self, env: dict) -> str:
        key = self.engine_key or f"{self.prefix}_ENGINE"
        return str(env.get(key) or self.engine_default).strip()
