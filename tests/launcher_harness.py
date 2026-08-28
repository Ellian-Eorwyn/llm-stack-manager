"""Capture exactly what a launcher execs, without running llama-server.

Every launcher ends in `exec "${LLAMA_SERVER_BIN}" --model ... --alias ...`,
and that argv is the whole contract: it is what the backend actually runs with,
and the only thing a consolidation of these scripts has to preserve.

So the launchers are run in a sandbox whose `LLAMA_SERVER_BIN` is a script that
prints its arguments and exits. The result is the exact command line, captured
without a GPU, a model file, or a served port -- which is what makes it usable
as an equivalence check in CI on either platform.

The sandbox is a *copy* rather than a symlink farm: every launcher derives
STACK_DIR from `dirname "${BASH_SOURCE[0]}"/..`, and `cd`-ing through a symlink
resolves it back to the real checkout, which would read the real config.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STUB = """#!/usr/bin/env bash
printf '%s\\n' "$@"
"""


class LauncherSandbox:
    """A throwaway stack directory with a stubbed llama-server."""

    def __init__(self, env: dict[str, str] | None = None, platform: str = "Linux"):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.platform = platform

        shutil.copytree(ROOT / "scripts", self.root / "scripts")
        shutil.copytree(ROOT / "web", self.root / "web",
                        ignore=shutil.ignore_patterns(".venv", "__pycache__", "static"))
        (self.root / "config").mkdir()
        (self.root / "models").mkdir()
        (self.root / "logs").mkdir()

        self.bin = self.root / "llama-server-stub"
        self.bin.write_text(STUB)
        self.bin.chmod(0o755)

        self.env_file = self.root / "config" / "llm-stack.env"
        self.write_env(env or {})

    def write_env(self, env: dict[str, str]) -> None:
        """Render the example config, then apply overrides on top of it.

        Built from the shipped example rather than from a hand-written minimal
        set, so a launcher reading a key nobody thought to include fails here
        the way it would in production instead of being silently defaulted.
        """
        text = (ROOT / "config" / "llm-stack.env.example").read_text()
        text = text.replace("@STACK_DIR@", str(self.root)).replace("@SERVICE_USER@", "tester")
        lines = []
        overrides = {"LLAMA_SERVER_BIN": str(self.bin), **env}
        seen = set()
        for line in text.splitlines():
            key = line.split("=", 1)[0].strip() if "=" in line and not line.startswith("#") else None
            if key in overrides:
                lines.append(f'{key}={overrides[key]}')
                seen.add(key)
            else:
                lines.append(line)
        for key, value in overrides.items():
            if key not in seen:
                lines.append(f'{key}={value}')
        self.env_file.write_text("\n".join(lines) + "\n")

    def normalise(self, argv: list[str]) -> list[str]:
        """Replace the sandbox path with a stable token.

        The sandbox lives in a fresh temp directory each run, so its path is the
        one thing in the argv that legitimately differs between two identical
        runs. Everything else must not.
        """
        return [a.replace(str(self.root), "@STACK@") for a in argv]

    def run(self, script: str, *args: str) -> tuple[list[str], str, int]:
        """(argv the launcher would exec, what it said, exit code)."""
        proc = subprocess.run(
            ["bash", str(self.root / "scripts" / script), *args],
            capture_output=True, text=True,
            env={**os.environ, "LLM_STACK_PLATFORM": self.platform},
        )
        argv, said = [], []
        # The stub prints one argument per line after the launcher's own banner;
        # a launcher's messages are distinguishable by their "[slot] " prefix.
        for line in proc.stdout.splitlines():
            (said if line.startswith("[") and "] " in line else argv).append(line)
        return argv, "\n".join(said + proc.stderr.splitlines()), proc.returncode

    def cleanup(self) -> None:
        self._tmp.cleanup()
