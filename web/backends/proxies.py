#!/usr/bin/env python3
"""The proxy registry: which trio of personas fronts which slot.

`start-chat-proxy2.sh` was a hand copy of `start-chat-proxy.sh`. It differed in
exactly three ways -- seven env overrides, three alias overrides, and the 19
memory-gateway exports replaced by a hardcoded `MEMORY_GATEWAY_ENABLED=off` --
and the other 33 persona exports were byte-for-byte identical. Which meant both
proxies read the same `THINK_TEMP` and the same `CODE_REASONING_EFFORT`: the two
slots could serve different models and could not be tuned differently, because
there was no `THINK2_TEMP` for anyone to set.

So the differences become data and the script becomes one script. What each
proxy overrides is the table below; everything else it inherits.

**Per-slot persona keys, shared keys as the fallback.** `LLM_B_THINK_TEMP` wins
over `THINK_TEMP` for slot B and nothing else changes -- a host that has only
ever set `THINK_TEMP` keeps both proxies behaving exactly as they did. That is
the same shape `llm-chat-proxy.py` already uses internally, where `THINK_TEMP`
falls back to `CHAT_TEMP`; this adds one level in front of it rather than a
second mechanism.

**Ports and aliases keep their old key names.** `THINK2_PORT` and
`THINK2_MODEL_NAME` carry the contract `docs/pi-forge-scheduling-contract.md`
pins, and section 2.1 of the handoff freezes ports, aliases and persona
semantics. Renaming the keys that hold them buys nothing and risks the one thing
that must not move.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Proxy:
    """One think/chat/code trio, and the slot it fronts."""

    #: Unit name.
    name: str
    #: The backend slot this proxy dials.
    slot: str
    #: Prefix for this proxy's own persona overrides, e.g. `LLM_A`.
    prefix: str
    #: Where the backend is. Absolute keys: both are the port contract.
    backend_host_key: str
    backend_port_key: str
    backend_port_default: str
    #: persona -> (port key, default). The keys are frozen by the contract.
    ports: dict[str, tuple[str, str]]
    #: persona -> (alias key, default). Frozen likewise.
    aliases: dict[str, tuple[str, str]]
    #: The model-routed aggregate endpoint.
    aggregate_enabled_key: str
    aggregate_port_key: str
    aggregate_port_default: str
    #: Whether this proxy runs the memory gateway. Only one may: the gateway
    #: binds its own listener, and a second one collides on the port.
    memory_gateway: bool = False
    label: str = ""
    component: str = ""


#: The 33 persona settings both proxies read, per persona. Each resolves
#: `{PREFIX}_{PERSONA}_{SUFFIX}` first and `{PERSONA}_{SUFFIX}` behind it.
PERSONA_SUFFIXES = (
    "PRESERVE_THINKING", "JINJA", "TEMP", "TOP_P", "TOP_K", "MIN_P",
    "PRESENCE_PENALTY", "REPEAT_PENALTY", "REASONING_FORMAT", "MAX_TOKENS",
    "REASONING_STREAM_MODE", "REASONING_EFFORT",
)

#: `CODE_THINKING` exists and `THINK_THINKING` would be nonsense: the thinking
#: persona is defined by having it on. Kept as a per-persona exception rather
#: than given every persona a key two of them ignore.
PERSONA_EXTRA_SUFFIXES = {"CODE": ("THINKING",)}

PERSONAS = ("THINK", "NOTHINK", "CODE")

#: Read by the proxy and exported by neither script before this. Latent, because
#: systemd's `EnvironmentFile` supplies the whole file to the unit -- but
#: `start-chat-proxy.sh` sourced the env bare, so running it by hand dropped
#: both and the two personas silently lost their reasoning effort.
PREVIOUSLY_UNEXPORTED = ("THINK_REASONING_EFFORT", "CODE_REASONING_EFFORT")

#: Everything that is the same for every proxy and carries no per-slot override.
SHARED_KEYS = (
    "BACKEND_CONNECT_TIMEOUT_SEC", "BACKEND_READ_TIMEOUT_SEC",
    "LISTEN_HOST", "EMBED_PORT", "EMBED_MODEL_NAME", "EMBED_BACKEND_HOST",
    "PROXY_STREAM_PASSTHROUGH",
    "UPSTREAM_400_CAPTURE_ENABLED", "UPSTREAM_400_CAPTURE_DIR",
    "UPSTREAM_400_CAPTURE_KEEP", "UPSTREAM_400_CAPTURE_MAX_BYTES",
    "STACK_DIR",
)

#: The memory gateway's own settings, exported only by the proxy that runs it.
MEMORY_KEYS = (
    "GRAPHITI_PORT", "GRAPHITI_PUBLIC_URL",
    "MEMORY_GATEWAY_ENABLED", "MEMORY_ENABLE_THINK", "MEMORY_ENABLE_NOTHINK",
    "MEMORY_ENABLE_CODE", "MEMORY_GRAPHITI_BASE_URL",
    "MEMORY_GRAPHITI_TIMEOUT_SEC", "MEMORY_GRAPHITI_COOLDOWN_SEC",
    "MEMORY_INJECTION_MODE", "MEMORY_MAX_FACTS", "MEMORY_MAX_FACT_CHARS",
    "MEMORY_MAX_BLOCK_CHARS", "MEMORY_MAX_QUERY_MESSAGES",
    "MEMORY_INCLUDE_SYSTEM_IN_QUERY", "MEMORY_MAX_INGEST_CHARS",
    "MEMORY_GROUP_HEADER_PRIORITY", "MEMORY_GROUP_FALLBACK_SALT",
    "MEMORY_FAIL_OPEN",
)


PROXIES = {
    "llm-a-proxy": Proxy(
        name="llm-a-proxy",
        slot="llm-a",
        prefix="LLM_A",
        label="LLM A Proxy",
        component="llm-a",
        backend_host_key="CHAT_BACKEND_HOST",
        backend_port_key="CHAT_BACKEND_PORT",
        backend_port_default="8010",
        ports={"THINK": ("THINK_PORT", "8003"),
               "NOTHINK": ("NOTHINK_PORT", "8004"),
               "CODE": ("CODE_PORT", "8008")},
        aliases={"THINK": ("THINK_MODEL_NAME", "think"),
                 "NOTHINK": ("NOTHINK_MODEL_NAME", "chat"),
                 "CODE": ("CODE_MODEL_NAME", "code")},
        aggregate_enabled_key="AGGREGATE_ENABLED",
        aggregate_port_key="AGGREGATE_PORT",
        aggregate_port_default="8012",
        memory_gateway=True,
    ),
    "llm-b-proxy": Proxy(
        name="llm-b-proxy",
        slot="llm-b",
        prefix="LLM_B",
        label="LLM B Proxy",
        component="llm-b",
        backend_host_key="CHAT2_BACKEND_HOST",
        backend_port_key="CHAT2_BACKEND_PORT",
        backend_port_default="8020",
        ports={"THINK": ("THINK2_PORT", "8103"),
               "NOTHINK": ("NOTHINK2_PORT", "8104"),
               "CODE": ("CODE2_PORT", "8108")},
        aliases={"THINK": ("THINK2_MODEL_NAME", "think2"),
                 "NOTHINK": ("NOTHINK2_MODEL_NAME", "chat2"),
                 "CODE": ("CODE2_MODEL_NAME", "code2")},
        aggregate_enabled_key="AGGREGATE2_ENABLED",
        aggregate_port_key="AGGREGATE2_PORT",
        aggregate_port_default="8112",
        # Off, and not because slot B deserves less: the gateway binds a
        # listener of its own and two of them collide.
        memory_gateway=False,
    ),
}

#: Proxy unit for a slot, for the places that reason in slots.
PROXY_BY_SLOT = {proxy.slot: proxy for proxy in PROXIES.values()}


def persona_keys(persona: str) -> tuple[str, ...]:
    """Every setting suffix this persona takes."""
    return PERSONA_SUFFIXES + PERSONA_EXTRA_SUFFIXES.get(persona, ())


def resolve(proxy: Proxy, env: dict) -> dict[str, str]:
    """The environment `llm-chat-proxy.py` should see for this proxy.

    Every value is written under the name the proxy already reads, so the Python
    half needs no notion of which slot it is serving. That is deliberate: the
    proxy's job is one trio of personas against one backend, and teaching it to
    be two would be a second thing that can disagree with this table.
    """
    out: dict[str, str] = {}

    def pick(*names: str) -> str | None:
        for name in names:
            if name in env and str(env[name]) != "":
                return str(env[name])
        return None

    host = pick(proxy.backend_host_key, "CHAT_BACKEND_HOST") or "127.0.0.1"
    out["CHAT_BACKEND_HOST"] = host
    out["CHAT_BACKEND_PORT"] = pick(proxy.backend_port_key) or proxy.backend_port_default

    for persona in PERSONAS:
        port_key, port_default = proxy.ports[persona]
        out[f"{persona}_PORT"] = pick(port_key) or port_default
        alias_key, alias_default = proxy.aliases[persona]
        out[f"{persona}_MODEL_NAME"] = pick(alias_key) or alias_default
        for suffix in persona_keys(persona):
            # The whole point: this proxy's own key first, the shared one behind
            # it. A host that only ever set the shared key is unaffected.
            value = pick(f"{proxy.prefix}_{persona}_{suffix}", f"{persona}_{suffix}")
            if value is not None:
                out[f"{persona}_{suffix}"] = value

    out["AGGREGATE_ENABLED"] = pick(proxy.aggregate_enabled_key, "AGGREGATE_ENABLED") or "on"
    out["AGGREGATE_PORT"] = pick(proxy.aggregate_port_key) or proxy.aggregate_port_default

    for key in SHARED_KEYS:
        value = pick(key)
        if value is not None:
            out[key] = value

    if proxy.memory_gateway:
        for key in MEMORY_KEYS:
            value = pick(key)
            if value is not None:
                out[key] = value
    else:
        out["MEMORY_GATEWAY_ENABLED"] = "off"

    return out
