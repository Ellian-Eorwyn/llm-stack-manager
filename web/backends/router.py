#!/usr/bin/env python3
"""Which auxiliary models the router pools, and the one place that decides.

`MODEL_ROUTER_MEMBERS` is a comma-separated string, and its default was written
out in thirteen files -- four shell scripts, five Python modules, the config
field's own hint, and the docs. Thirteen copies of a default is twelve chances
to add a member to the pool and have half the stack disagree about whether it is
in it. `ASR` is the one that proved it: opt-in, so it is in the member *table*
and not in the default *string*, a distinction every copy had to get right.

**Pooled-or-dedicated becomes a per-member switch.** `<MEMBER>_POOLED` says
whether the router owns that model or whether it runs as its own unit, which is
the question an operator actually has -- "should embed hold VRAM permanently or
load on demand" -- rather than editing a list and remembering the syntax.

`MODEL_ROUTER_MEMBERS` stays readable forever and is still what the string-shaped
consumers see, so a host that has only ever set it keeps working. It is the
fallback, not the source: a `_POOLED` key set anywhere means the switches decide.
"""

from __future__ import annotations

#: Every model the router *can* pool, in the order it renders them. Not the same
#: as the ones it does: see `DEFAULT_POOLED`.
MEMBER_PREFIXES = ("EMBED", "OCR", "RERANK", "TASK", "ASR")

#: Pooled unless someone says otherwise. `ASR` is deliberately absent -- pooling
#: the audio model is opt-in, because its only caller is the transcription
#: sidecar and it competes for VRAM with models that serve interactive traffic.
DEFAULT_POOLED = ("EMBED", "OCR", "RERANK", "TASK")

#: The string form, for the consumers that still read one. This is the value the
#: config example ships and the last resort when nothing else answers.
DEFAULT_MEMBERS = ",".join(DEFAULT_POOLED)


def pooled_key(prefix: str) -> str:
    return f"{prefix.upper()}_POOLED"


def _switches(env: dict) -> dict[str, bool]:
    """The `_POOLED` switches that are actually set, by prefix."""
    out = {}
    for prefix in MEMBER_PREFIXES:
        raw = str(env.get(pooled_key(prefix), "") or "").strip().lower()
        # `inherit` is the explicit form of unset. A select cannot render "no
        # value" without showing its first option instead, which would say `on`
        # for a member nobody had configured.
        if raw in {"", "inherit"}:
            continue
        if raw in {"on", "true", "1", "yes"}:
            out[prefix] = True
        elif raw in {"off", "false", "0", "no"}:
            out[prefix] = False
    return out


def pooled_members(env: dict, warn=None) -> list[str]:
    """The members the router owns, in table order.

    Authority, most specific first:

    1. The `<MEMBER>_POOLED` switches, if any is set. A host that has adopted
       them has adopted them for every member: a mixture, where some members are
       switches and the rest come from a stale string, is the ambiguity this
       replaces rather than a feature.
    2. `MODEL_ROUTER_MEMBERS`, for a host that has only ever had the string.
    3. `DEFAULT_POOLED`.
    """
    switches = _switches(env)
    if switches:
        return [p for p in MEMBER_PREFIXES if switches.get(p, p in DEFAULT_POOLED)]

    raw = str(env.get("MODEL_ROUTER_MEMBERS", "") or "").strip()
    if not raw:
        return list(DEFAULT_POOLED)
    named = {part.strip().upper() for part in raw.split(",") if part.strip()}
    # A name nothing serves is reported rather than dropped: it is almost always
    # a typo, and a pool quietly one model smaller than the operator asked for is
    # the kind of thing found weeks later.
    if warn:
        for unknown in sorted(named - set(MEMBER_PREFIXES)):
            warn(f"unknown model router member {unknown!r}; "
                 f"known members are {', '.join(MEMBER_PREFIXES)}")
    # Ordered by the table rather than by the string, so two hosts that list the
    # same members in a different order render the same preset.
    return [p for p in MEMBER_PREFIXES if p in named]


def members_string(env: dict) -> str:
    """`pooled_members` as the comma-separated form the shell half reads."""
    return ",".join(pooled_members(env))
