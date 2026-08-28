#!/usr/bin/env python3
"""Which platform this manager is running on, and how to ask it things.

Reach behaviour through the module -- `platforms.active().meminfo()` -- rather
than binding `active` or an adapter method to a local name. Binding captures the
object as it was at import time and makes it unsubstitutable, which is the same
trap `ModuleBoundaryTests` guards against for imports and the reason
`docs/repo-layout-and-deploy-drift.md` states the rule.

Nothing in this package may import `app`.
"""

from __future__ import annotations

import sys

from . import base
from .darwin import DarwinPlatform
from .linux import LinuxPlatform

Platform = base.Platform

_ACTIVE: base.Platform | None = None


def detect() -> base.Platform:
    """A fresh adapter for this host, without touching the cached one."""
    if sys.platform == "darwin":
        return DarwinPlatform()
    return LinuxPlatform()


def active() -> base.Platform:
    """The adapter for this host, created once and reused.

    Cached because the Darwin adapter carries state -- the synthesised restart
    counts, which are only meaningful as a series of observations by one object.
    Building a new adapter per call would reset them to zero on every poll and
    silently reproduce the fixed-zero `n_restarts` this package exists to fix.
    """
    global _ACTIVE
    if _ACTIVE is None:
        _ACTIVE = detect()
    return _ACTIVE


def set_active(platform: base.Platform | None) -> base.Platform | None:
    """Install a specific adapter, or `None` to fall back to detection.

    For tests: the Linux implementations must stay exercisable on a macOS
    developer machine and in macOS CI, or the platform this project is mostly
    deployed on would only ever be tested on one runner. Returns the previous
    value so a caller can restore it.
    """
    global _ACTIVE
    previous = _ACTIVE
    _ACTIVE = platform
    return previous
