#!/usr/bin/env python3
"""
The substrate every part of the manager stands on: where things are, how to run
them, and how to talk to them.

Nothing in here knows what a configuration key is or what a service does. It is
the layer underneath that — filesystem locations derived from the install root,
the systemd wrapper, subprocess and HTTP helpers, and the small parsers that
several modules would otherwise each write badly.

**Reach these through the module, not through a bound import.** Write
`core.read_meminfo()`, not `from core import read_meminfo`. The difference only
shows up under test: a bound name is resolved once at import time, so
substituting `core.read_meminfo` afterwards leaves every importer still calling
the original. Since these are exactly the things tests substitute — the paths a
temp directory stands in for, the systemd calls that must not actually run —
the module-attribute form is what keeps one patch point working for all of
them. The tables in `config_fields` are bound directly instead, because those
are read and never replaced.

Paths hang off `STACK_DIR`, which is derived from this file's own location
rather than configured, so the stack works from wherever it is installed — the
manager runs from `/mnt/LLMs/llamacpp/llm-stack-git` on this box and from a
developer checkout in tests, with no setting to keep in sync.
"""

import functools
import json
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from urllib import error as urlerror
from urllib import request as urlrequest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import platforms
import setup_engine


# ---------------------------------------------------------------------------
# Where everything lives
# ---------------------------------------------------------------------------

STACK_DIR   = Path(__file__).resolve().parent.parent
CONFIG_FILE = STACK_DIR / "config" / "llm-stack.env"
SCRIPTS_DIR = STACK_DIR / "scripts"
MODELS_DIR  = STACK_DIR / "models"
TRANSCRIPTION_MODELS_DIR = MODELS_DIR / "transcription"
CUSTOM_MODELS_FILE = STACK_DIR / "config" / "custom-models.json"
CUSTOM_MODEL_ARG_PRESETS_FILE = STACK_DIR / "config" / "custom-model-arg-presets.json"
SAVED_CONFIGS_DIR  = STACK_DIR / "config" / "saved"
DEFAULT_SAVED_CONFIG_FILE = STACK_DIR / "config" / "default-saved-config"
CHAT_TEMPLATES_DIR = STACK_DIR / "config" / "chat-templates"
CHAT_TEMPLATES_META_FILE = CHAT_TEMPLATES_DIR / "templates.json"
TTS_CONFIG_FILE    = STACK_DIR / "config" / "tts-backends.json"
TTS_STATE_FILE     = STACK_DIR / "config" / "tts-state.json"
LOGS_DIR           = STACK_DIR / "logs"
GRAPHITI_EXPORTS_DIR = STACK_DIR / "exports" / "graphiti"

# ---------------------------------------------------------------------------
# Short-lived memoisation
# ---------------------------------------------------------------------------

def ttl_cache(seconds: float):
    """Memoise a zero-argument function for `seconds`, across threads.

    The status poll asks the same expensive questions several times per request
    — `nvidia-smi` runs twice and `systemctl show` once per unit — and now has
    external API clients polling alongside it. A window shorter than the UI's
    own five-second interval keeps every reading fresh enough to act on while
    collapsing the duplicates within one poll, and the duplicates are the whole
    cost: two clients a second apart still see two real samples.

    Deliberately zero-argument. A cache keyed on a config dict would either need
    that dict to be hashable or would have to ignore it, and ignoring an
    argument is how a cache comes to answer a question nobody asked.

    Set `CACHE_TTL_SECONDS = 0` or call `.cache_clear()` to disable it in tests.
    """
    def decorate(func):
        lock = threading.Lock()
        state: dict = {"at": 0.0, "value": None, "filled": False}

        @functools.wraps(func)
        def wrapper():
            ttl = CACHE_TTL_SECONDS if CACHE_TTL_SECONDS is not None else seconds
            with lock:
                if state["filled"] and ttl > 0 and (time.monotonic() - state["at"]) < ttl:
                    return state["value"]
            # Computed outside the lock: these call subprocesses, and holding a
            # lock across one would serialise every poller behind the slowest.
            value = func()
            with lock:
                state.update(at=time.monotonic(), value=value, filled=True)
            return value

        def cache_clear():
            with lock:
                state.update(at=0.0, value=None, filled=False)

        wrapper.cache_clear = cache_clear
        wrapper.__wrapped__ = func
        return wrapper
    return decorate


# Overrides every `ttl_cache` window at once. `None` means each decorator keeps
# the interval it asked for; tests set this to 0 to make caching a no-op.
CACHE_TTL_SECONDS: float | None = None


# ---------------------------------------------------------------------------
# systemd
# ---------------------------------------------------------------------------

class ServiceManager:
    """The service manager, whichever one this host has.

    Kept as a class of classmethods because roughly forty call sites across the
    app, the routes and the tests already reach it that way. The behaviour now
    lives in `platforms`, and every method here is a one-line delegation -- the
    `if IS_MAC:` branches that used to be inline are gone, along with the two
    bugs they were hiding: `launchctl list` was parsed as JSON (it emits an
    OpenStep plist, so every service read as inactive on macOS, always) and
    `n_restarts` was hardcoded to 0 (which reads as healthy, not as unknown, and
    silently disabled flap detection).
    """

    #: Retained because existing call sites and tests read it. Prefer asking
    #: `platforms.active().name`, which follows a substituted platform.
    IS_MAC = sys.platform == 'darwin'

    @classmethod
    def run_cmd(cls, cmd, timeout=30):
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    @classmethod
    def state(cls, name: str) -> dict:
        return platforms.active().service_state(name)

    @classmethod
    def is_active(cls, name: str) -> bool:
        return cls.state(name)["active"]

    @classmethod
    def start(cls, name: str, timeout=30) -> subprocess.CompletedProcess:
        return platforms.active().service_start(name, timeout=timeout)

    @classmethod
    def stop(cls, name: str, timeout=30) -> subprocess.CompletedProcess:
        return platforms.active().service_stop(name, timeout=timeout)

    @classmethod
    def restart(cls, name: str, timeout=120) -> tuple[int, str]:
        return platforms.active().service_restart(name, timeout=timeout)

    @classmethod
    def action(cls, act: str, name: str, timeout=30) -> tuple[int, str]:
        if act == "start":
            r = cls.start(name, timeout=timeout)
            return r.returncode, (r.stdout + r.stderr).strip()
        elif act == "stop":
            r = cls.stop(name, timeout=timeout)
            return r.returncode, (r.stdout + r.stderr).strip()
        elif act == "restart":
            return cls.restart(name, timeout=timeout)
        else:
            return 1, "unsupported action"

    @classmethod
    def get_pid(cls, name: str) -> int:
        return platforms.active().service_pid(name)

    @classmethod
    def is_installed(cls, name: str) -> bool:
        return cls.state(name)["installed"]


SETUP_RUNNER = setup_engine.SetupRunner()

# ---------------------------------------------------------------------------
# Running things, and reading what the host says about itself
# ---------------------------------------------------------------------------

def run_script(script_name: str, *args) -> tuple:
    script = SCRIPTS_DIR / script_name
    try:
        r = subprocess.run(
            ['bash', str(script)] + list(args),
            capture_output=True, text=True, timeout=60,
        )
        return r.returncode == 0, (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return False, 'Script timed out'
    except Exception as e:
        return False, str(e)


def run_command(cmd: list[str], cwd: Path | None = None, timeout: int = 1200) -> tuple[int, str]:
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(cwd) if cwd else None,
        )
        return r.returncode, (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return 124, f"Command timed out after {timeout}s: {' '.join(cmd)}"
    except Exception as exc:
        return 1, str(exc)


def append_command_log(lines: list[str], cmd: list[str], rc: int, out: str):
    lines.append(f"$ {' '.join(cmd)}")
    lines.append(f"[exit {rc}]")
    if out:
        lines.append(out)


def find_git_repo_root(start: Path) -> Path | None:
    current = start.resolve(strict=False)
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def read_meminfo() -> dict[str, int]:
    """Host memory in `/proc/meminfo`'s key names and KiB units, on any platform.

    The name and the shape are kept because every consumer already reads them
    and the tests already patch them; only the source is now platform-dependent.
    """
    return platforms.active().meminfo()


def format_kib_as_gib(value_kib: int) -> str:
    return f"{value_kib / 1024 / 1024:.1f} GiB"

def load_json_file(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default

# ---------------------------------------------------------------------------
# Talking to the services, and parsing what they say back
# ---------------------------------------------------------------------------

def http_json(url: str, method: str = "GET", payload=None, timeout: int = 30, headers=None):
    data = None
    request_headers = dict(headers or {})
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    req = urlrequest.Request(url, data=data, method=method, headers=request_headers)
    with urlrequest.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
        if not body:
            return {}
        return json.loads(body.decode("utf-8"))


def http_multipart(url: str, fields: dict | None = None, files: dict | None = None,
                   timeout: int = 300, headers=None) -> tuple[int, str]:
    """POST multipart/form-data and return (status, body).

    Hand-rolled because the stack has no `requests` in the manager's own
    environment — `web/requirements.txt` is one line, and keeping it that way is
    deliberate. `files` maps field name to (filename, bytes, content_type).

    Returns the status rather than raising on 4xx/5xx: the caller is proxying
    another service's answer, and that service's error body is the useful part.
    """
    boundary = f"----llmstack{uuid.uuid4().hex}"
    body = bytearray()
    for name, value in (fields or {}).items():
        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        body += f"{value}\r\n".encode()
    for name, (filename, content, content_type) in (files or {}).items():
        body += f"--{boundary}\r\n".encode()
        body += (f'Content-Disposition: form-data; name="{name}"; '
                 f'filename="{filename}"\r\n').encode()
        body += f"Content-Type: {content_type}\r\n\r\n".encode()
        body += content if isinstance(content, bytes) else str(content).encode()
        body += b"\r\n"
    body += f"--{boundary}--\r\n".encode()

    request_headers = dict(headers or {})
    request_headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    req = urlrequest.Request(url, data=bytes(body), method="POST", headers=request_headers)
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urlerror.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


def http_bytes(url: str, method: str = "GET", payload=None, timeout: int = 300):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urlrequest.Request(url, data=data, method=method, headers=headers)
    with urlrequest.urlopen(req, timeout=timeout) as resp:
        return resp.read(), resp.headers.get_content_type()


def parse_int(value, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        out = default
    if minimum is not None:
        out = max(minimum, out)
    if maximum is not None:
        out = min(maximum, out)
    return out


def parse_iso_datetime(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        # Validate only; we keep the original string for query params.
        datetime.fromisoformat(value.replace('Z', '+00:00'))
        return value
    except ValueError:
        return None


def truncate_text(value: str | None, limit: int = 220) -> str:
    if not isinstance(value, str):
        return ''
    if len(value) <= limit:
        return value
    if limit <= 3:
        return value[:limit]
    return value[: limit - 3] + '...'
