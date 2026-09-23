#!/usr/bin/env python3
"""The arguments that are decided rather than looked up.

`Flag` and `Toggle` cover a setting that becomes an argument. These cover the
dozen that do not: `--device` is dropped on a Metal build, `--fit-ctx` is inert
when auto-fit is off, `--jinja` steps aside if the operator passed their own,
and a chat template id that names no file refuses to build a command at all.

Every one of them was a helper in `scripts/lib/backend-preflight.sh` or an
inline branch repeated across three launchers. They are transcribed here, with
the reason each exists kept next to it -- those reasons are the record of a
failure that arrived after exec, as a restart loop rather than an error.

Each takes the same shape as `Flag.resolve`, plus a `Context` carrying what a
decision needs but a lookup does not: the slot, the operator's already-parsed
custom arguments, and somewhere to put what is worth saying.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import platforms

from . import speculative
from .spec import Slot, Toggle, lookup

#: A template id names a file under config/chat-templates/, so it may not
#: contain a path separator or anything else that could leave that directory.
TEMPLATE_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")

#: The three levels Qwen 3.8's template accepts. It raises on anything else,
#: which would fail every request against the backend rather than degrade, so
#: an unrecognised level is dropped with a note instead of passed through.
REASONING_EFFORTS = ("xhigh", "medium", "low")


@dataclass
class Context:
    """What a decision needs and a lookup does not."""

    slot: Slot
    #: The operator's custom arguments, already shell-split.
    custom: tuple[str, ...] = ()
    stack_dir: str = ""
    said: list[str] = field(default_factory=list)

    def custom_has(self, *flags: str) -> bool:
        return any(flag in self.custom for flag in flags)

    def say(self, message: str) -> None:
        self.said.append(message)


@dataclass(frozen=True)
class Device:
    """`--device`, unless this is a Metal build and the name is a CUDA one.

    The shipped default is `CUDA0`, so an Apple silicon host that never edited
    it would pass `--device CUDA0` to a Metal build, which fails at load with a
    device-not-found rather than falling back -- arriving after exec and looking
    like a crash loop.

    Dropped rather than translated: on a single-device Metal machine there is
    nothing for `--device` to choose between, and silently rewriting an
    operator's CUDA0 into MTL0 would hide a configuration that is wrong for the
    host. The name is `MTL0`, not `Metal0`; `--list-devices` prints
    "MTL0: Apple M1 Pro", and matching on "Metal" refuses the correct name.
    """

    keys: tuple[str, ...] = ("DEVICE",)

    def resolve(self, env, prefixes, ctx=None):
        device = (lookup(env, self.keys, prefixes, empty_is_set=True) or "").strip()
        if not device:
            return []
        if platforms.active().name == "darwin" and not device.startswith("MTL"):
            if ctx:
                ctx.say(f"Ignoring Device '{device}': this is a Metal build, which "
                        f"enumerates MTL0. Letting llama-server choose.")
            return []
        return ["--device", device]


@dataclass(frozen=True)
class SwaFull:
    """`--swa-full`, when the model has sliding-window attention.

    Whether it does is a fact about the GGUF, which only `budget.py` can read,
    so `backend-preflight.sh` answers it and hands the verdict over the same way
    it hands over placement. Unknown counts as supported, which keeps the
    operator's setting on a host where the model cannot be read -- a helper must
    never stop a backend from starting.
    """

    keys: tuple[str, ...] = ("SWA_FULL",)
    #: Where the shell leaves its answer.
    vetted_key: str = "LLM_BACKEND_SWA_FULL"

    def resolve(self, env, prefixes, ctx=None):
        if (lookup(env, self.keys, prefixes) or "off").strip() != "on":
            return []
        vetted = env.get(self.vetted_key)
        if vetted is not None and str(vetted).strip() != "on":
            if ctx:
                ctx.say("Ignoring Full SWA KV Cache: this model has no sliding-window "
                        "attention, so --swa-full has no effect.")
            return []
        return ["--swa-full"]


@dataclass(frozen=True)
class FitCtx:
    """`--fit-ctx`, only when auto-fit is on to act on it."""

    keys: tuple[str, ...] = ("FIT_CTX",)
    fit_keys: tuple[str, ...] = ("FIT",)

    def resolve(self, env, prefixes, ctx=None):
        value = (lookup(env, self.keys, prefixes, empty_is_set=True) or "").strip()
        if not value or value == "0":
            return []
        if (lookup(env, self.fit_keys, prefixes) or "on").strip() == "off":
            if ctx:
                ctx.say(f"Ignoring Minimum Fit Context {value}: auto-fit is off, "
                        f"so --fit-ctx has no effect.")
            return []
        return ["--fit-ctx", value]


@dataclass(frozen=True)
class CacheReuse:
    """`--cache-reuse N`, where zero means the same as unset."""

    keys: tuple[str, ...] = ("CACHE_REUSE",)

    def resolve(self, env, prefixes, ctx=None):
        value = (lookup(env, self.keys, prefixes) or "").strip()
        return [] if not value or value == "0" else ["--cache-reuse", value]


@dataclass(frozen=True)
class MMProj:
    """`--mmproj`, when the slot names one and the file is there.

    Emptied means cleared: a slot whose own key has been cleared must not
    inherit the legacy one. A projector left behind by a model switch is traded
    for the new model's own; see `budget.matching_projector`.
    """

    def resolve(self, env, prefixes, ctx=None):
        keys = ctx.slot.mmproj_keys if ctx else ("MMPROJ_PATH",)
        path = (lookup(env, keys, prefixes, empty_is_set=True) or "").strip()
        if not path or not os.path.isfile(path):
            return []
        model = (lookup(env, ctx.slot.model_keys, prefixes) or "").strip() if ctx else ""
        if model and os.path.isfile(model):
            import budget  # deferred: budget imports backends
            path, why = budget.matching_projector(model, path)
            if why:
                ctx.say(why)
        return ["--mmproj", path]


@dataclass(frozen=True)
class Jinja:
    """`--jinja`, unless the operator already passed it themselves."""

    keys: tuple[str, ...] = ("JINJA",)

    def resolve(self, env, prefixes, ctx=None):
        if ctx and ctx.custom_has("--jinja"):
            return []
        return Toggle("--jinja", self.keys[0], when="on", default="off").resolve(env, prefixes)


@dataclass(frozen=True)
class TemplateKwargs:
    """`--chat-template-kwargs`, built from the thinking settings.

    Two shapes, because the two slots ask the template different questions. The
    chat slots say `preserve_thinking` -- keep the reasoning that is produced --
    and the task slot says `enable_thinking`, which turns it on and off. Both
    ride a `reasoning_effort` when one is set and valid.

    Only meaningful under a template that reads them: Qwen 3.8 reads all three,
    older Qwen templates read the first two, and a template that reads none is
    unaffected either way.
    """

    #: "preserve" (chat) or "enable" (task).
    style: str = "preserve"
    thinking_keys: tuple[str, ...] = ("PRESERVE_THINKING",)
    effort_keys: tuple[str, ...] = ("REASONING_EFFORT",)

    def resolve(self, env, prefixes, ctx=None):
        if ctx and ctx.custom_has("--chat-template-kwargs"):
            return []
        thinking = (lookup(env, self.thinking_keys, prefixes) or
                    ("on" if self.style == "preserve" else "off")).strip()
        effort = (lookup(env, self.effort_keys, prefixes, empty_is_set=True) or "").strip()
        if effort and effort not in REASONING_EFFORTS:
            if ctx:
                ctx.say(f"Ignoring Reasoning Effort '{effort}': expected "
                        f"{', '.join(REASONING_EFFORTS[:-1])}, or {REASONING_EFFORTS[-1]}.")
            effort = ""

        if self.style == "enable":
            # Written exactly as the launcher wrote it, spacing included: this
            # string reaches llama-server verbatim and the golden compares it.
            if thinking != "on":
                return ["--chat-template-kwargs", '{"enable_thinking":false}']
            if effort:
                return ["--chat-template-kwargs",
                        '{"enable_thinking":true, "reasoning_effort": "%s"}' % effort]
            return ["--chat-template-kwargs", '{"enable_thinking":true}']

        pairs = []
        if thinking == "on":
            pairs.append('"preserve_thinking": true')
        if effort:
            pairs.append('"reasoning_effort": "%s"' % effort)
        if not pairs:
            return []
        return ["--chat-template-kwargs", "{%s}" % ", ".join(pairs)]


@dataclass(frozen=True)
class TemplateFile:
    """`--chat-template-file`, resolved from an id under config/chat-templates/.

    A refusal here is deliberate. The helpers that read the model degrade to
    permissive, because a helper must never stop a backend from starting; this
    is command construction, where the rule is the opposite -- an id naming a
    file that is not there would start a backend serving a template nobody
    chose.
    """

    keys: tuple[str, ...] = ("TEMPLATE_ID",)

    def resolve(self, env, prefixes, ctx=None):
        if ctx and ctx.custom_has("--chat-template", "--chat-template-file"):
            return []
        template_id = (lookup(env, self.keys, prefixes, empty_is_set=True) or "").strip()
        if not template_id:
            return []
        label = f"[{ctx.slot.name}]" if ctx else ""
        key = f"{prefixes[0]}_{self.keys[0].lstrip('!')}"
        if not TEMPLATE_ID_RE.match(template_id):
            raise SystemExit(f"{label} Invalid {key}: {template_id}".strip())
        stack = (ctx.stack_dir if ctx else "") or str(env.get("STACK_DIR") or ".")
        path = os.path.join(stack, "config", "chat-templates", f"{template_id}.jinja")
        if not os.path.isfile(path):
            raise SystemExit(f"{label} Chat template not found: {path}".strip())
        return ["--chat-template-file", path]


def lora_pairs(paths_raw: str, scales_raw: str, stack_dir: str):
    """Pair adapter paths with their scales, resolving each path.

    Shared with `scripts/render-models-ini.py`: a pooled model reaches
    llama.cpp through a preset file rather than an argv, but the pairing and
    the refusals are the same, and two copies of them would drift.

    Yields `(resolved_path, scale, problem)`. `problem` is None when the pair
    is usable and a sentence explaining the refusal otherwise -- the caller
    decides whether that becomes a launcher note or a preset warning.
    """
    paths = [p.strip() for p in (paths_raw or "").split(",") if p.strip()]
    scales = [s.strip() for s in (scales_raw or "").split(",")]
    for index, path in enumerate(paths):
        resolved = path if os.path.isabs(path) else os.path.join(
            stack_dir or ".", "models", "loras", path)
        if not os.path.isfile(resolved):
            yield resolved, "", f"Ignoring LoRA adapter '{path}': no such file at {resolved}."
            continue
        # `--lora-scaled` splits on the last colon, so a path containing one
        # would be read as part of the scale.
        if ":" in resolved:
            yield resolved, "", (f"Ignoring LoRA adapter '{path}': --lora-scaled cannot "
                                 f"express a path containing a colon.")
            continue
        raw = scales[index].strip() if index < len(scales) else ""
        try:
            scale = "%g" % float(raw) if raw else "1.0"
        except ValueError:
            yield resolved, "1.0", (f"Ignoring LoRA scale '{raw}' for '{path}': not a "
                                    f"number. Using 1.0.")
            continue
        yield resolved, scale, None


@dataclass(frozen=True)
class Lora:
    """The `--lora-scaled` block, plus `--lora-init-without-apply`.

    A LoRA adapter is applied at runtime rather than merged into the weights,
    so one 17 GB base serves every fine-tune of it and swapping between them
    costs nothing -- `POST /lora-adapters` changes a scale without a reload.
    That is the whole reason this is a launch flag at all: what gets *loaded*
    is fixed at exec, what is *applied* is not.

    Preload-unapplied is emitted as `:0` on each adapter, **not** by relying on
    `--lora-init-without-apply` alone. The server README says adapters loaded
    with that flag "start at scale 0.0", and in llama-server they do not:
    `common.cpp` only skips the one-time `common_set_adapter_lora` at startup,
    while `server-context.cpp` assigns `slot.lora = params_base.lora_adapters`
    for every task and re-applies it per batch -- so the first request restores
    each adapter's configured scale. Measured on build b10434: launched with the
    flag and `:1.0`, the very first completion came back in the adapter's voice.
    A configured `:0` is what actually holds.

    The flag is still passed, because skipping the startup application is real
    and harmless; it is just not sufficient on its own.

    This matters because adapters *stack*. A slot with two fine-tunes listed
    would otherwise serve both blended together on its first request.

    A path that names no file is dropped with a note rather than passed through:
    llama-server exits on a missing adapter, which would arrive as a restart
    loop instead of an error.
    """

    path_keys: tuple[str, ...] = ("LORA_PATHS",)
    scale_keys: tuple[str, ...] = ("LORA_SCALES",)
    init_keys: tuple[str, ...] = ("LORA_INIT_WITHOUT_APPLY",)

    def resolve(self, env, prefixes, ctx=None):
        if ctx and ctx.custom_has("--lora", "--lora-scaled"):
            return []
        raw = (lookup(env, self.path_keys, prefixes, empty_is_set=True) or "").strip()
        if not raw:
            return []
        stack = (ctx.stack_dir if ctx else "") or str(env.get("STACK_DIR") or ".")
        scales_raw = lookup(env, self.scale_keys, prefixes, empty_is_set=True) or ""

        unapplied = (lookup(env, self.init_keys, prefixes) or "on").strip() == "on"

        args: list[str] = []
        for resolved, scale, problem in lora_pairs(raw, scales_raw, stack):
            if problem and ctx:
                ctx.say(problem)
            if not scale:
                continue
            args += ["--lora-scaled", f"{resolved}:{'0' if unapplied else scale}"]

        if args and unapplied:
            args.append("--lora-init-without-apply")
        return args


@dataclass(frozen=True)
class Speculative:
    """The `--spec-*` block. See `backends/speculative.py`."""

    def resolve(self, env, prefixes, ctx=None):
        label = f"[{ctx.slot.name}]" if ctx else ""
        args, said = speculative.build(env, prefixes, label)
        if ctx:
            for message in said:
                ctx.say(message)
        return args
