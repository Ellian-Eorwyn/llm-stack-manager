"""validate.sh and the launchd half of scripts/cross-platform.sh.

Both are run for real, on either platform, against stub `uname`, `dscl`,
`launchctl` and `curl` on PATH -- the same idea as `platform_harness`, applied
to the shell side. On macOS the script used to crash under `set -u` before
checking anything, and once past that it asked `systemctl`, which a Mac does
not have, so every service would have been skipped as "not running".
"""

from __future__ import annotations

import os
import pathlib
import shutil
import stat
import subprocess
import tempfile
import textwrap
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: What `launchctl list <label>` really prints: an OpenStep plist, not JSON.
LAUNCHCTL_LIST = textwrap.dedent('''\
    {
    \t"Label" = "com.llmstack.%(name)s";
    \t"LastExitStatus" = 0;
    \t"PID" = 4242;
    \t"ProgramArguments" = (
    \t\t"/stack/scripts/launchd-wrapper-%(name)s.sh";
    \t);
    };
''')


class _FakeMac:
    """A temp dir laid out as a macOS host with some launchd services running."""

    def __init__(self, running: tuple[str, ...], installed: tuple[str, ...]):
        self.dir = pathlib.Path(tempfile.mkdtemp())
        self.home = self.dir / "home"
        agents = self.home / "Library" / "LaunchAgents"
        agents.mkdir(parents=True)
        for name in installed:
            (agents / f"com.llmstack.{name}.plist").write_text("<plist/>")
        self.bin = self.dir / "bin"
        self.bin.mkdir()
        self._stub("uname", 'echo Darwin')
        self._stub("dscl", f'echo "NFSHomeDirectory: {self.home}"')
        cases = "\n".join(
            f'    com.llmstack.{name}) cat <<"EOF"\n{LAUNCHCTL_LIST % {"name": name}}EOF\n    ;;'
            for name in running)
        self._stub("launchctl", textwrap.dedent('''\
            [ "$1" = list ] || exit 1
            case "$2" in
            %s
                *) echo "Could not find service \\"$2\\" in domain" >&2; exit 113 ;;
            esac
        ''') % cases)
        self._stub("curl", 'echo \'{"total_slots":4,"object":"list","content":"ok"}\'')

    def _stub(self, name: str, body: str) -> None:
        path = self.bin / name
        path.write_text("#!/usr/bin/env bash\n" + body + "\n")
        path.chmod(path.stat().st_mode | stat.S_IEXEC)

    def env(self) -> dict:
        env = dict(os.environ)
        env["PATH"] = f"{self.bin}{os.pathsep}{env['PATH']}"
        env.pop("LLM_LAUNCHD_DOMAIN", None)
        env.pop("SERVICE_USER", None)
        return env

    def cleanup(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)


class SvcIsActiveOnDarwinTests(unittest.TestCase):
    def setUp(self):
        self.mac = _FakeMac(running=("llm-a",), installed=("llm-a", "llm-b"))
        self.addCleanup(self.mac.cleanup)

    def _active(self, name: str) -> bool:
        script = f'source "{ROOT}/scripts/cross-platform.sh"; svc_is_active {name}'
        return subprocess.run(["bash", "-uc", script], env=self.mac.env(),
                              capture_output=True).returncode == 0

    def test_a_running_launchd_service_is_active(self):
        # `launchctl list` output was parsed as JSON, which always failed, so
        # this read as inactive on every Mac.
        self.assertTrue(self._active("llm-a"))

    def test_an_installed_but_unloaded_service_is_not(self):
        self.assertFalse(self._active("llm-b"))

    def test_a_service_with_no_plist_is_not(self):
        self.assertFalse(self._active("transcript-backend"))


class ValidateOnDarwinTests(unittest.TestCase):
    """validate.sh finds its config and helpers next to itself, so it is copied
    into a scratch tree with a minimal config."""

    def setUp(self):
        self.mac = _FakeMac(running=("llm-manager", "llm-a", "llm-a-proxy"),
                            installed=("llm-manager", "llm-a", "llm-a-proxy"))
        self.addCleanup(self.mac.cleanup)
        self.tree = self.mac.dir / "stack"
        (self.tree / "scripts").mkdir(parents=True)
        (self.tree / "config").mkdir()
        shutil.copy(ROOT / "validate.sh", self.tree / "validate.sh")
        shutil.copy(ROOT / "scripts" / "cross-platform.sh",
                    self.tree / "scripts" / "cross-platform.sh")
        (self.tree / "config" / "llm-stack.env").write_text(textwrap.dedent('''\
            THINK_PORT=8003
            NOTHINK_PORT=8004
            CODE_PORT=8008
            EMBED_PORT=8005
            RERANK_PORT=8006
            TASK_PORT=8007
            CHAT_BACKEND_PORT=8010
            TRANSCRIPT_ENABLED=off
        '''))

    def _run(self) -> subprocess.CompletedProcess:
        return subprocess.run(["bash", str(self.tree / "validate.sh")],
                              env=self.mac.env(), capture_output=True, text=True,
                              timeout=60)

    def test_runs_to_the_end_under_set_u(self):
        result = self._run()
        self.assertNotIn("unbound variable", result.stderr)
        self.assertIn("Results:", result.stdout)

    def test_launchd_services_are_checked_not_skipped(self):
        result = self._run()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("[PASS] GET :8010/props returns slot geometry", result.stdout)
        for port in (8003, 8004, 8008):
            self.assertIn(f"[PASS] GET :{port}/v1/models returns JSON", result.stdout)
        # Not installed and not running: skipped, not failed.
        self.assertIn("[SKIP] embed /v1/models", result.stdout)
        self.assertIn("[SKIP] Transcription sidecar", result.stdout)
        self.assertIn(" 0 failed", result.stdout)

    def test_a_down_backend_is_skipped_when_expected_off(self):
        (self.tree / "config" / "service-expectations.json").write_text(
            '{"llm-a": {"expected": "off"}}')
        result = self._run()
        self.assertIn("[SKIP] Primary backend", result.stdout)


if __name__ == "__main__":
    unittest.main()
