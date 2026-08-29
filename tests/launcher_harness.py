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

# The budget model, replaced by something that records the question instead of
# answering it.
#
# `preflight_report` is the only part of a launcher that does not reach argv,
# and it is the part most easily gutted by a consolidation: the backend name and
# the fourteen settings it carries are what `budget.py` weighs the fit against,
# and dropping nine of them still starts the backend and still passes the argv
# golden.
#
# Python, not bash: `preflight_report` and `budget_field` both invoke it as
# `python3 "${BUDGET_PY}"`. `--field` queries exit non-zero, which is what a
# real `budget.py` does against these empty model files anyway, so recording
# does not move the command line.
BUDGET_STUB = """#!/usr/bin/env python3
import sys

if "--field" in sys.argv[1:]:
    raise SystemExit(1)
print("PREFLIGHT " + " ".join(sys.argv[1:]))
"""


class LauncherSandbox:
    """A throwaway stack directory with a stubbed llama-server."""

    def __init__(self, env: dict[str, str] | None = None, platform: str = "Linux",
                 record_budget: bool = False):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.platform = platform
        self.record_budget = record_budget

        shutil.copytree(ROOT / "scripts", self.root / "scripts")
        shutil.copytree(ROOT / "web", self.root / "web",
                        ignore=shutil.ignore_patterns(".venv", "__pycache__", "static"))
        (self.root / "config").mkdir()
        (self.root / "models").mkdir()
        (self.root / "logs").mkdir()

        self.bin = self.root / "llama-server-stub"
        self.bin.write_text(STUB)
        self.bin.chmod(0o755)

        self.budget = self.root / "budget-stub.py"
        self.budget.write_text(BUDGET_STUB)
        self.budget.chmod(0o755)

        self.env_file = self.root / "config" / "llm-stack.env"
        self.write_env(env or {})
        if self.record_budget:
            self._create_model_files()

    def _create_model_files(self) -> None:
        """`preflight_report` returns early unless the model file exists.

        Only `*_MODEL_PATH` is created. That is not the same as "no mmproj
        appears": the task slot's `TASK_MMPROJ_PATH` is the same file as its
        model, so `--mmproj` shows up there and nowhere else -- which is what
        happens on a real host, and is a branch the empty-`models/` golden
        cannot reach at all.
        """
        for line in self.env_file.read_text().splitlines():
            key, _, value = line.partition("=")
            if not key.endswith("_MODEL_PATH") or not value:
                continue
            path = Path(value)
            if path.is_absolute() and str(path).startswith(str(self.root)):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()

    def write_env(self, env: dict[str, str]) -> None:
        """Render the example config, then apply overrides on top of it.

        Built from the shipped example rather than from a hand-written minimal
        set, so a launcher reading a key nobody thought to include fails here
        the way it would in production instead of being silently defaulted.

        Passing `None` for a value removes the line instead of emptying it.
        The launchers distinguish `${X:-d}` from `${X-d}` deliberately, so
        "absent" and "present but empty" are two different configurations and a
        test has to be able to write either one.
        """
        text = (ROOT / "config" / "llm-stack.env.example").read_text()
        text = text.replace("@STACK_DIR@", str(self.root)).replace("@SERVICE_USER@", "tester")
        lines = []
        overrides = {"LLAMA_SERVER_BIN": str(self.bin), **env}
        seen = set()
        for line in text.splitlines():
            key = line.split("=", 1)[0].strip() if "=" in line and not line.startswith("#") else None
            if key in overrides:
                seen.add(key)
                if overrides[key] is None:
                    continue
                lines.append(f'{key}={overrides[key]}')
            else:
                lines.append(line)
        for key, value in overrides.items():
            if key not in seen and value is not None:
                lines.append(f'{key}={value}')
        self.env_file.write_text("\n".join(lines) + "\n")

    def exported(self) -> dict[str, str]:
        """The env file as `EnvironmentFile=` would present it.

        Raw, not expanded: systemd does not expand `${LISTEN_HOST}` in an
        environment file, and neither does this. The launcher's own `source`
        expands it for the launcher's own variables.
        """
        out = {}
        for line in self.env_file.read_text().splitlines():
            if line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            out[key.strip()] = value
        return out

    def normalise(self, argv: list[str]) -> list[str]:
        """Replace the sandbox path with a stable token.

        The sandbox lives in a fresh temp directory each run, so its path is the
        one thing in the argv that legitimately differs between two identical
        runs. Everything else must not.
        """
        return [a.replace(str(self.root), "@STACK@") for a in argv]

    def run(self, script: str, *args: str) -> tuple[list[str], str, int]:
        """(argv the launcher would exec, what it said, exit code)."""
        # systemd gives every unit `EnvironmentFile=config/llm-stack.env` and the
        # launchd wrapper sources it under `set -a`, so a launcher starts with
        # the file's values already in its environment as well as sourcing it
        # itself. That is not a detail: the custom-argument blocks read
        # `os.environ` from a Python heredoc, and without this they would see
        # nothing here while seeing everything in production.
        environ = {**os.environ, **self.exported(), "LLM_STACK_PLATFORM": self.platform}
        if self.record_budget:
            environ["BUDGET_PY"] = str(self.budget)
        proc = subprocess.run(
            ["bash", str(self.root / "scripts" / script), *args],
            capture_output=True, text=True, env=environ,
        )
        argv, said = [], []
        # The stub prints one argument per line after the launcher's own banner;
        # a launcher's messages are distinguishable by their "[slot] " prefix.
        for line in proc.stdout.splitlines():
            (said if line.startswith("[") and "] " in line else argv).append(line)
        return argv, "\n".join(said + proc.stderr.splitlines()), proc.returncode

    def preflight_call(self, said: str) -> list[str]:
        """The arguments the launcher handed `budget.py`, from what it said.

        `preflight_report` prefixes every line of the report with the slot's
        own "[name] " banner marker, so the recorded call arrives through the
        same channel as the launcher's other messages.
        """
        for line in said.splitlines():
            _, marker, rest = line.partition("] PREFLIGHT ")
            if marker:
                return self.normalise(rest.split(" "))
        return []

    def cleanup(self) -> None:
        self._tmp.cleanup()
