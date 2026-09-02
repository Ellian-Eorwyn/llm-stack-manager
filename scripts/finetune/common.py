#!/usr/bin/env python3
"""Run layout, recipe I/O, and interpreter resolution.

A "run" is one corpus turned into one adapter. Everything about it lives under
`RUNS_ROOT/<name>/`, outside this repository: the documents are someone's
writing, the datasets are large, and the adapters are build products. What the
repo holds is the code that produces them.

`run.yaml` is the reproducibility artefact. A run that cannot say which folder
it read, which shape it built, and which base it trained against is a fine-tune
nobody can repeat, and six months later that is the same as not having it.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

#: Runs live beside the venv and the HF cache they need, not in the repo.
RUNS_ROOT = Path(os.environ.get("FINETUNE_RUNS_ROOT", "/mnt/LLMs/unsloth/runs"))

#: The interpreter that has torch, transformers and unsloth. Only `train.py` and
#: the tokenizer half of `validate` need it; ingest and build are stdlib.
UNSLOTH_PYTHON = Path(os.environ.get(
    "FINETUNE_UNSLOTH_PYTHON", "/mnt/LLMs/unsloth/unsloth_studio/bin/python"))

#: Where a converted adapter is dropped for llama-server to load.
STACK_DIR = Path(__file__).resolve().parents[2]

DEFAULT_BASE_MODEL = "unsloth/Qwen3.8-27B-unsloth-bnb-4bit"

#: Matches how llm-stack serves: LLM_A_REASONING_EFFORT is medium, which is the
#: template's neutral level. Training at the template's own default (xhigh)
#: prepends a preamble to every example that the served model never sees.
DEFAULT_REASONING_EFFORT = "medium"

#: Loss starts here, not at the assistant turn. A thinking model is primed with
#: `<|im_start|>assistant\n<think>\n` at serve time; a corpus with no reasoning
#: traces would otherwise teach it to emit an empty think block and stop
#: reasoning altogether.
DEFAULT_RESPONSE_PART = "</think>\n\n"

SHAPES = ("style", "dialogue", "qa", "raw")


class RunError(Exception):
    """Something the operator can fix. Printed without a traceback."""


@dataclass
class Run:
    """One corpus, one dataset, one adapter."""

    name: str
    root: Path
    recipe: dict = field(default_factory=dict)

    # -- layout ----------------------------------------------------------
    @property
    def documents(self) -> Path:
        return self.root / "documents"

    @property
    def manifest(self) -> Path:
        return self.documents / "manifest.jsonl"

    @property
    def dataset(self) -> Path:
        return self.root / "dataset.jsonl"

    @property
    def evalset(self) -> Path:
        return self.root / "eval.jsonl"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    @property
    def recipe_path(self) -> Path:
        return self.root / "run.yaml"

    def report_path(self, stage: str) -> Path:
        return self.reports / f"{stage}-report.json"

    # -- recipe ----------------------------------------------------------
    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.reports.mkdir(exist_ok=True)
        self.logs.mkdir(exist_ok=True)
        self.recipe_path.write_text(_dump_yaml(self.recipe))

    def update(self, **values) -> None:
        self.recipe.update({k: v for k, v in values.items() if v is not None})
        self.save()

    def require(self, *keys: str) -> None:
        missing = [k for k in keys if not self.recipe.get(k)]
        if missing:
            raise RunError(
                f"run {self.name!r} has no {', '.join(missing)} in run.yaml — "
                f"the earlier stage has not been run")


def open_run(name: str, create: bool = False) -> Run:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", name):
        raise RunError(f"invalid run name {name!r}: use letters, digits, . _ -")
    root = RUNS_ROOT / name
    if not root.is_dir():
        if not create:
            raise RunError(
                f"no run named {name!r} under {RUNS_ROOT}. "
                f"Start one with: corpus.py ingest <source-dir> --run {name}")
        root.mkdir(parents=True)
    run = Run(name=name, root=root)
    if run.recipe_path.is_file():
        run.recipe = _load_yaml(run.recipe_path.read_text())
    return run


def list_runs() -> list[str]:
    if not RUNS_ROOT.is_dir():
        return []
    return sorted(p.name for p in RUNS_ROOT.iterdir() if (p / "run.yaml").is_file())


# ---------------------------------------------------------------------------
# jsonl
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RunError(f"{path}:{number}: {exc}") from None
    return rows


def write_jsonl(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


# ---------------------------------------------------------------------------
# A minimal YAML subset
# ---------------------------------------------------------------------------
# The recipe is a flat map of scalars and short string lists, and the repo's
# other scripts are stdlib-only so they run under any interpreter on the box.
# Pulling in PyYAML for eleven keys would make `ingest` depend on a venv it
# otherwise does not need.

def _dump_yaml(data: dict) -> str:
    lines = ["# Written by scripts/finetune. Edit to change the recipe, then re-run the stage.\n"]
    for key, value in data.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            lines.extend(f"  - {_scalar(item)}" for item in value)
        else:
            lines.append(f"{key}: {_scalar(value)}")
    return "\n".join(lines) + "\n"


def _scalar(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if text == "" or text.strip() != text or any(c in text for c in ':#\n"\''):
        return json.dumps(text)
    return text


def _load_yaml(text: str) -> dict:
    data: dict = {}
    key = None
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if raw.startswith("  - ") and key is not None:
            data.setdefault(key, [])
            if isinstance(data[key], list):
                data[key].append(_parse_scalar(raw[4:].strip()))
            continue
        if ":" not in raw:
            continue
        key, _, value = raw.partition(":")
        key = key.strip()
        value = value.strip()
        data[key] = [] if value == "" else _parse_scalar(value)
    return data


def _parse_scalar(text: str):
    if text.startswith('"'):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text.strip('"')
    if text == "true":
        return True
    if text == "false":
        return False
    if text == "null":
        return None
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------

def unsloth_python() -> Path | None:
    """The training interpreter, or None with the reason left to the caller.

    Returning None rather than raising is deliberate: `validate` degrades to a
    character estimate without it, and only `train` genuinely cannot proceed.
    """
    return UNSLOTH_PYTHON if UNSLOTH_PYTHON.is_file() else None


def have(command: str) -> bool:
    return shutil.which(command) is not None


def run_command(argv: list[str], timeout: int = 120) -> tuple[int, str, str]:
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return done.returncode, done.stdout, done.stderr
    except FileNotFoundError:
        return 127, "", f"{argv[0]}: not found"
    except subprocess.TimeoutExpired:
        return 124, "", f"{argv[0]}: timed out after {timeout}s"


def fail(message: str) -> int:
    """Print an operator-facing error and return an exit code.

    Never silent: a stage that refuses must say so on stdout as well, because
    a caller reading only stdout would otherwise see an empty report and read
    it as "nothing to do".
    """
    print(f"ERROR: {message}")
    print(f"ERROR: {message}", file=sys.stderr)
    return 1
