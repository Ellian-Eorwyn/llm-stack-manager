#!/usr/bin/env python3
"""Speculative decoding arguments, for any slot that offers them.

Three launchers carried this: `start-chat-backend-dense.sh`,
`start-chat-backend2.sh` and `start-task.sh`, ninety-three lines each and the
same ninety-three lines with the prefix swapped. Five pre-built arrays, a
branch on the method, and four membership tests -- copied twice, so a fix to
one was a fix to a third of the deployments.

It is not `Flag` data, because the arguments are not a flat set: which groups
appear depends on the method, and two of the branches refuse to build a command
at all. So it is a small dispatch, transcribed rather than redesigned.

**The method is not validated against a closed list**, deliberately. The shell
passed any unrecognised value straight to `--spec-type` and let llama-server
decide, and `ngram-cache` -- an option the UI offers today -- takes no group of
its own. A table of known methods would have to be right about a list that
belongs to llama.cpp, and being wrong about it means refusing a method the
build supports. What *is* enumerated is the two things the shell enumerated:
which methods need a draft model, and which name a group of tuning arguments.
"""

from __future__ import annotations

import os

from .spec import Flag, lookup

#: The one spelling the UI never wrote but configs still carry.
ALIASES = {"mtp": "draft-mtp"}

#: Methods that name a draft model without being `draft-model` itself.
#:
#: `draft-mtp` is deliberately absent: with no draft model given,
#: `common_speculative_init_result` creates the MTP draft context against the
#: *target* model (common/speculative.cpp, the `else if (spec_mtp)` branch),
#: which is how a GGUF carrying its own blk.N.nextn.* head runs MTP with no
#: sidecar at all. Requiring a draft path here would refuse to start exactly
#: that configuration.
NEEDS_DRAFT_MODEL = ("draft-simple", "draft-eagle3", "draft-dflash")

DRAFT_CACHE = (
    Flag("--spec-draft-type-k", ("SPEC_DRAFT_TYPE_K",), "f16"),
    Flag("--spec-draft-type-v", ("SPEC_DRAFT_TYPE_V",), "f16"),
)

COMMON = (
    Flag("--spec-draft-n-max",   ("SPEC_DRAFT_N_MAX",),   "6"),
    Flag("--spec-draft-n-min",   ("SPEC_DRAFT_N_MIN",),   "0"),
    Flag("--spec-draft-p-min",   ("SPEC_DRAFT_P_MIN",),   "0.75"),
    Flag("--spec-draft-p-split", ("SPEC_DRAFT_P_SPLIT",), "0.10"),
)

#: Method token -> the arguments it adds, appended in this order. The three
#: n-gram families share the same three settings under different flag names.
NGRAM_GROUPS = (
    ("ngram-mod", (
        Flag("--spec-ngram-mod-n-match", ("SPEC_NGRAM_MOD_N_MATCH",), "24"),
        Flag("--spec-ngram-mod-n-min",   ("SPEC_NGRAM_MOD_N_MIN",),   "48"),
        Flag("--spec-ngram-mod-n-max",   ("SPEC_NGRAM_MOD_N_MAX",),   "64"),
    )),
    ("ngram-simple", (
        Flag("--spec-ngram-simple-size-n",    ("SPEC_NGRAM_SIZE_N",),    "12"),
        Flag("--spec-ngram-simple-size-m",    ("SPEC_NGRAM_SIZE_M",),    "48"),
        Flag("--spec-ngram-simple-min-hits",  ("SPEC_NGRAM_MIN_HITS",),  "1"),
    )),
    ("ngram-map-k", (
        Flag("--spec-ngram-map-k-size-n",   ("SPEC_NGRAM_SIZE_N",),   "12"),
        Flag("--spec-ngram-map-k-size-m",   ("SPEC_NGRAM_SIZE_M",),   "48"),
        Flag("--spec-ngram-map-k-min-hits", ("SPEC_NGRAM_MIN_HITS",), "1"),
    )),
    ("ngram-map-k4v", (
        Flag("--spec-ngram-map-k4v-size-n",   ("SPEC_NGRAM_SIZE_N",),   "12"),
        Flag("--spec-ngram-map-k4v-size-m",   ("SPEC_NGRAM_SIZE_M",),   "48"),
        Flag("--spec-ngram-map-k4v-min-hits", ("SPEC_NGRAM_MIN_HITS",), "1"),
    )),
)

DRAFT_MODEL_KEYS = ("SPEC_DRAFT_MODEL_PATH",)
DRAFT_DEVICES_KEYS = ("SPEC_DRAFT_DEVICES",)
DRAFT_NGL_KEYS = ("SPEC_DRAFT_N_GPU_LAYERS",)


def method(env: dict, prefixes: tuple[str, ...]) -> str:
    chosen = lookup(env, ("SPEC_METHOD",), prefixes) or "off"
    return ALIASES.get(chosen, chosen)


def _names(spec_type: str) -> list[str]:
    """The method's tokens, the way the shell tested membership.

    `[[ ",${M}," == *,ngram-mod,* ]]` is a comma-delimited membership test, not
    a substring one -- which is what keeps `ngram-map-k4v` from matching
    `ngram-map-k`. A method may be a comma-joined list, which is how the hint
    on `SPEC_NGRAM_MOD` describes `draft-mtp,ngram-mod`.
    """
    return [part for part in spec_type.split(",") if part]


def _flags(group, env, prefixes) -> list[str]:
    out: list[str] = []
    for flag in group:
        out += flag.resolve(env, prefixes)
    return out


def _draft_model(env, prefixes, spec_type, label) -> str:
    path = lookup(env, DRAFT_MODEL_KEYS, prefixes, empty_is_set=True) or ""
    key = f"{prefixes[0]}_SPEC_DRAFT_MODEL_PATH"
    if not path:
        raise SystemExit(
            f"{label} {spec_type} is enabled, but {key} is empty.")
    if not os.path.isfile(path):
        raise SystemExit(f"{label} Draft model not found: {path}")
    return path


def _draft_args(env, prefixes, path) -> list[str]:
    ngl = lookup(env, DRAFT_NGL_KEYS, prefixes) or "auto"
    args = ["--spec-draft-model", path, "--spec-draft-ngl", ngl]
    return args


def build(env: dict, prefixes: tuple[str, ...], label: str = "") -> tuple[list[str], list[str]]:
    """(arguments, things worth saying) for this slot's speculative settings.

    Raises `SystemExit` when a method needs a draft model and there is not one.
    That is command construction, and the rule there is that it must be right
    or not run -- unlike the helpers in `backend-preflight.sh`, which degrade to
    permissive because a helper must never stop a backend from starting.
    """
    spec_type = method(env, prefixes)
    if spec_type == "off":
        return [], []

    args: list[str] = []
    said: list[str] = []
    devices = lookup(env, DRAFT_DEVICES_KEYS, prefixes, empty_is_set=True) or ""

    if spec_type == "draft-model":
        path = _draft_model(env, prefixes, spec_type, label)
        args += _draft_args(env, prefixes, path)
        args += _flags(DRAFT_CACHE, env, prefixes)
        args += _flags(COMMON, env, prefixes)
        if devices:
            args += ["--spec-draft-device", devices]
        # Read back out of the arguments rather than out of the env a second
        # time, so the banner describes the process that is about to start.
        # Re-deriving a setting independently is how --fit-ctx came to be
        # reported as set while being passed as cleared.
        said_of = dict(zip(args, args[1:]))
        said += [
            f"Draft model:      {path}",
            f"Draft GPU layers: {said_of['--spec-draft-ngl']}",
            f"Draft devices:    {devices or 'auto'}",
            f"Draft n-max/min:  {said_of['--spec-draft-n-max']}/{said_of['--spec-draft-n-min']}",
            f"Draft p-min/split:{said_of['--spec-draft-p-min']}/{said_of['--spec-draft-p-split']}",
        ]
        return args, said

    args += ["--spec-type", spec_type]
    args += _flags(DRAFT_CACHE, env, prefixes)
    args += _flags(COMMON, env, prefixes)

    if spec_type in NEEDS_DRAFT_MODEL:
        args += _draft_args(env, prefixes, _draft_model(env, prefixes, spec_type, label))
        if devices:
            args += ["--spec-draft-device", devices]

    names = _names(spec_type)
    for token, group in NGRAM_GROUPS:
        if token not in names:
            continue
        resolved = _flags(group, env, prefixes)
        args += resolved
        if token == "ngram-mod":
            values = dict(zip(resolved, resolved[1:]))
            said.append(
                f"N-gram mod:       match={values['--spec-ngram-mod-n-match']} "
                f"min={values['--spec-ngram-mod-n-min']} "
                f"max={values['--spec-ngram-mod-n-max']}")

    if "ngram-mod" not in names and (lookup(env, ("SPEC_NGRAM_MOD",), prefixes) or "off") == "on":
        said.append(
            "N-gram mod assist requested, but this llama-server build only "
            f"accepts ngram-mod as a standalone --spec-type; leaving "
            f"--spec-type={spec_type}.")

    return args, said
