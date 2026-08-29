#!/usr/bin/env python3
"""Which units the boot path starts, and on whose authority.

Prints one unit name per line, in start order, for `restore-active-stack.sh`.

Authority, most specific first:

1. ``config/service-expectations.json`` -- what an operator last said about a
   named unit. This is the file the manager writes when someone stops a service
   on purpose, and the one ``/api/v1/snapshot`` reports as ``expected``. ``off``
   is final: nothing below can start a unit that was switched off deliberately,
   and ``on`` starts a unit whose component was never selected.
2. ``LLM_STACK_SELECTED_COMPONENTS``, else the wizard's selection in
   ``config/install-state.json``, else the same literal
   ``activate-selected-stack.sh`` falls back to. This decides the units nobody
   has expressed an expectation about.

The boot path had none of this. An unset ``LLM_STACK_SELECTED_COMPONENTS`` meant
"start everything", and ``llm-stack-restore.service`` sets no variable, so every
boot took that branch. On a host whose GPU 0 already holds a 27B primary that
meant starting ``chat-backend2`` -- a second 27B -- onto it every time;
``service-expectations.json`` had recorded ``off`` for that unit since June and
nothing on the boot path read the file. It was survivable only because someone
stopped it again by hand after each reboot.

The component-to-unit map is ``setup_engine.COMPONENT_SERVICES`` rather than a
third copy of it. ``tests/test_slot_registry.py`` records that
``restore-active-stack.sh`` and ``activate-selected-stack.sh`` each wrote that
map out in shell, and says collapsing it "needs Python on the install path and
is a change of its own". This is that change, for the boot path.

A model the router pools is never started as a unit: it is a child of
``llama-router``, loads on demand, and starting it would fight nginx for its
port.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "web"))

import config_env  # noqa: E402
import health  # noqa: E402
import setup_engine  # noqa: E402

#: What `activate-selected-stack.sh:6` falls back to. Kept identical on purpose:
#: two boot paths that disagree about the default are how a host comes back from
#: a reboot in a state nobody chose.
FALLBACK_COMPONENTS = "primary,embedding,task,ocr,glmocr-sdk,searxng,playwright"

#: Unit -> the `MODEL_ROUTER_MEMBERS` name that would make it a router child.
#: Only these four can be pooled; everything else is always its own unit.
ROUTER_MEMBER_BY_UNIT = {
    "embed": "EMBED",
    "ocr": "OCR",
    "rerank": "RERANK",
    "task": "TASK",
}

#: Units that answer to a feature switch as well as to selection. A recorded
#: `on` from when the feature was enabled must not outlive the switch being
#: turned off -- `llama-router` especially, because starting it against
#: `MODEL_ROUTER_ENABLED=off` gives the pooled models two owners.
FEATURE_SWITCH = {
    "llama-router": "MODEL_ROUTER_ENABLED",
    "transcript-backend": "TRANSCRIPT_ENABLED",
}

#: Start order. A backend comes up before the proxy in front of it so the
#: proxy's first upstream probe has something to reach, and `llm-manager` is
#: never here because the caller keeps it running throughout.
START_ORDER = (
    "chat-backend-dense",
    "chat-proxy",
    "chat-backend2",
    "chat-proxy2",
    "llama-router",
    "embed",
    "rerank",
    "task",
    "ocr",
    "glmocr-sdk",
    "playwright-server",
    "transcript-backend",
)


def _text(env: dict, key: str, default: str = "") -> str:
    return str(env.get(key, default) or "").strip()


def _is_on(env: dict, key: str, default: str = "off") -> bool:
    return _text(env, key, default).lower() == "on"


def selected_components(env: dict) -> list[str]:
    """The component selection, from the most specific source that answers."""
    raw = _text(env, "LLM_STACK_SELECTED_COMPONENTS")
    if not raw:
        try:
            stored = setup_engine.load_state().get("selection", {}).get("components", [])
        except Exception:
            # An unreadable or half-written state file must not decide that this
            # host runs nothing. Fall through to the literal.
            stored = []
        raw = ",".join(str(item) for item in stored if item)
    if not raw:
        raw = FALLBACK_COMPONENTS
    return setup_engine.resolve_components(
        [part.strip() for part in raw.split(",") if part.strip()])


def router_owns(env: dict, unit: str) -> bool:
    if not _is_on(env, "MODEL_ROUTER_ENABLED"):
        return False
    member = ROUTER_MEMBER_BY_UNIT.get(unit)
    if member is None:
        return False
    members = _text(env, "MODEL_ROUTER_MEMBERS", "EMBED,OCR,RERANK,TASK")
    return member in {part.strip().upper() for part in members.split(",")}


def boot_units(env: dict, expectations: dict | None = None,
               chat_backend: str = "chat-backend-dense") -> list[str]:
    """The units to start, in start order.

    `chat_backend` is resolved by the caller because a saved profile may name a
    different unit for the primary slot, and that resolution reads a marker file
    the shell already owns.
    """
    if expectations is None:
        expectations = health.read_expectations()

    def expected(unit: str) -> str:
        entry = expectations.get(unit)
        return str((entry or {}).get("expected") or "unspecified")

    wanted: set[str] = set()
    for component in selected_components(env):
        wanted.update(setup_engine.COMPONENT_SERVICES.get(component, []))
    # An operator who switched a unit on outranks a selection that never
    # mentioned it -- the expectation is the more recent statement of intent.
    # This is what keeps `transcript-backend` coming back on a host whose
    # wizard selection predates the transcription sidecar.
    wanted.update(unit for unit in expectations if expected(unit) == "on")

    if _is_on(env, "MODEL_ROUTER_ENABLED"):
        wanted.add("llama-router")

    units = []
    for unit in START_ORDER:
        candidate = chat_backend if unit == "chat-backend-dense" else unit
        if candidate not in wanted and unit not in wanted:
            continue
        if expected(candidate) == "off":
            continue
        if router_owns(env, candidate):
            continue
        # Gated by its own switch as well as by selection, because these units
        # exist on hosts that have the component installed but turned off, and a
        # recorded `on` from when it was enabled must not outlive the switch.
        if candidate in FEATURE_SWITCH and not _is_on(env, FEATURE_SWITCH[candidate]):
            continue
        units.append(candidate)
    return units


def caller_env() -> dict:
    """The config file, with the caller's environment allowed to override it.

    `restore-active-stack.sh` sources `llm-stack.env` without `set -a`, so those
    values are shell variables and never reach a child process -- reading
    `os.environ` alone would see `MODEL_ROUTER_ENABLED` unset on a host that has
    it on, and start `embed` as a unit against the router holding the same port.
    The config file is read through `config_env` for the same reason the manager
    does: it applies `LEGACY_ENV_KEY_MAP`, so a host that still spells a key the
    old way is understood rather than silently defaulted.

    `os.environ` still wins where it says anything, which is what makes
    `LLM_STACK_SELECTED_COMPONENTS=... restore-active-stack.sh` an override.
    """
    env = dict(config_env.read_env())
    env.update({k: v for k, v in os.environ.items() if v})
    return env


def main(argv: list[str]) -> int:
    chat_backend = argv[1] if len(argv) > 1 and argv[1] else "chat-backend-dense"
    for unit in boot_units(caller_env(), chat_backend=chat_backend):
        print(unit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
