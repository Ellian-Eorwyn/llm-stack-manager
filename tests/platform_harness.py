"""Running a platform's code path on a host that is not that platform.

Without this, half of this project is only testable on half of its runners: the
systemd and nvidia-smi paths would be exercised on Linux only, the launchd and
ioreg paths on macOS only, and each would be free to rot on the other -- which
is precisely how the macOS support that already existed came to be broken in
four separate places without a single failing test.

Not named `test_*`, so `unittest discover` does not try to collect it.
"""

import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "web"))

import core
import platforms
from platforms.darwin import DarwinPlatform
from platforms.linux import LinuxPlatform


@contextmanager
def as_platform(platform):
    """Run the body with `platforms.active()` returning `platform`.

    `core.ttl_cache` windows are collapsed for the duration, because swapping
    the adapter invalidates every cached platform-derived answer and those
    caches are deliberately zero-argument -- there is no key for them to miss
    on. `app.get_gpu_info` is the one that bit: a 2s window, and
    `CrossPlatformTests` runs both directions in 0.4s, so the Linux spoke's
    reading was still cached when the Darwin spoke asked. The Darwin adapter
    was active and correct; it was simply never called.

    That was invisible on both CI runners, which have no GPU: the leaked list
    is empty there and an empty list asserts nothing. It fails on a host with
    real GPUs, which is the machine this package exists to keep testable from
    the other one.
    """
    previous = platforms.set_active(platform)
    previous_ttl = core.CACHE_TTL_SECONDS
    core.CACHE_TTL_SECONDS = 0
    try:
        yield platform
    finally:
        core.CACHE_TTL_SECONDS = previous_ttl
        platforms.set_active(previous)


@contextmanager
def as_linux(run_cmd=None):
    """The Linux adapter, optionally with its subprocess runner stubbed.

    `run_cmd` receives `(cmd, timeout=...)` and returns a
    `subprocess.CompletedProcess`, so a test can hand back canned `systemctl` or
    `nvidia-smi` output and assert on what the adapter makes of it.
    """
    platform = LinuxPlatform()
    with as_platform(platform):
        if run_cmd is None:
            yield platform
        else:
            with patch.object(LinuxPlatform, "run_cmd", staticmethod(run_cmd)):
                yield platform


@contextmanager
def as_darwin(run_cmd=None):
    """The Darwin adapter, optionally with its subprocess runner stubbed."""
    platform = DarwinPlatform()
    with as_platform(platform):
        if run_cmd is None:
            yield platform
        else:
            with patch.object(DarwinPlatform, "run_cmd", staticmethod(run_cmd)):
                yield platform
