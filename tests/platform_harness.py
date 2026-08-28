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

import platforms
from platforms.darwin import DarwinPlatform
from platforms.linux import LinuxPlatform


@contextmanager
def as_platform(platform):
    """Run the body with `platforms.active()` returning `platform`."""
    previous = platforms.set_active(platform)
    try:
        yield platform
    finally:
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
