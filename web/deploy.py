#!/usr/bin/env python3
"""
Whether the installed stack is running the code it was supposed to be running.

The manager does not run from the checkout somebody edits. `llm-manager.service`
sets `WorkingDirectory=/mnt/LLMs/llamacpp/llm-stack-git`, and that tree is
updated by `update.sh` — which means it advances only when an operator asks it
to. Nothing anywhere reported how far behind it had fallen, so the normal
failure was reading a bug report against code that had already been fixed, on a
box whose UI gave no hint that it was serving an older tree.

Two details make the naive version of this check wrong.

**A local comparison lies.** `git rev-list --left-right --count origin/main...HEAD`
compares against the *cached* remote ref, which is only as fresh as the last
fetch. Measured on this host: the deployed tree sat at `340a2b5` while the
remote was at `59116d6`, and because its `origin/main` ref predated both
commits, the local comparison answered `0 0` — "up to date" — while the operator
looked at a bug that had been fixed twice. So the remote ref has to be refreshed
before it is believed, and `remote_checked_at` is reported so a stale answer can
be recognised as one.

**The check runs as root against somebody else's tree.** The manager is root;
the checkout is owned by the user who installed it. Git refuses to operate
across that boundary unless told the directory is trusted, so every command goes
through `git_cmd`, which sets `safe.directory` the same way the llama.cpp
updater in `app.py` already does.

Fetching is done on a background interval rather than inside the request, for
the reason `health.py` gives about its probes: the UI polls every five seconds,
and a network round trip on that path is exactly the thing that must not block
it. `/api/deploy/status` therefore always serves a cached snapshot, and never
reaches the network itself.

What the report is *for* is deciding what an update costs. `update.sh` already
distinguishes a cheap restart from one that reloads tens of GB of weights and
discards a warm prompt cache; `BACKEND_SENSITIVE_PATHS` is that distinction, and
it lives here so both sides read the same list.
"""

from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

DEFAULT_CHECK_INTERVAL_SECONDS = 900
GIT_TIMEOUT_SECONDS = 120
FETCH_TIMEOUT_SECONDS = 180

# Commit subjects are the operator-facing summary of what an update would bring
# in, so enough of them to be useful and not so many that the badge becomes a
# changelog.
MAX_PENDING_COMMITS = 25

# Changes under these paths mean a model backend is running stale launcher code
# and only a restart — a multi-GB model reload — will pick them up. `update.sh`
# keeps a bash array with the same contents for its own post-update report;
# `tests/test_deploy.py` asserts the two agree, because a list that drifts would
# quietly stop reporting the expensive case.
BACKEND_SENSITIVE_PATHS = (
    "scripts/start-chat-backend",
    "scripts/start-embed",
    "scripts/start-rerank",
    "scripts/start-task",
    "scripts/start-ocr",
    # Sourced by every launcher, and it decides which flags reach llama-server.
    "scripts/lib/",
    # The launchers consult the budget model at startup to skip flags the loaded
    # model cannot act on, so a change here changes the next launch.
    "web/budget.py",
)

REMEDY = "sudo llm-stack-manager update"


def git_cmd(stack_dir) -> list[str]:
    """A git invocation that works when root runs it against a user-owned tree.

    Without `safe.directory` every command fails with "detected dubious
    ownership", which is the manager's normal situation rather than an edge
    case: the service runs as root and the checkout belongs to whoever installed
    it.
    """
    path = str(Path(stack_dir))
    return ["git", "-c", f"safe.directory={path}", "-C", path]


def _run(cmd: list[str], timeout: int = GIT_TIMEOUT_SECONDS) -> tuple[int, str]:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return 1, str(exc)
    return result.returncode, ((result.stdout or "") + (result.stderr or "")).strip()


def is_git_checkout(stack_dir) -> bool:
    return (Path(stack_dir) / ".git").exists()


def _head_details(git: list[str]) -> dict:
    rc, out = _run([*git, "log", "-1", "--pretty=%H%x1f%h%x1f%s%x1f%cI"])
    if rc != 0 or "\x1f" not in out:
        return {"head": "", "head_short": "", "subject": "", "committed_at": ""}
    head, short, subject, committed = (out.split("\x1f") + ["", "", "", ""])[:4]
    return {"head": head, "head_short": short, "subject": subject,
            "committed_at": committed}


def _upstream_ref(git: list[str], branch: str) -> str:
    """What this checkout compares itself against.

    `@{upstream}` is the honest answer when the branch tracks something. A
    detached HEAD — which `update.sh --release` produces — has no upstream at
    all, so fall back to the remote branch by name rather than reporting nothing.
    """
    rc, out = _run([*git, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"])
    if rc == 0 and out and "@{upstream}" not in out:
        return out
    candidates = [f"origin/{branch}"] if branch else []
    candidates += ["origin/main", "origin/master"]
    for candidate in candidates:
        rc, _ = _run([*git, "rev-parse", "--verify", "--quiet", candidate])
        if rc == 0:
            return candidate
    return ""


def local_state(stack_dir) -> dict:
    """Everything knowable about the deployed tree without touching the network."""
    if not is_git_checkout(stack_dir):
        return {"ok": False, "is_git_checkout": False, "stack_dir": str(stack_dir),
                "error": "the installed stack is not a git checkout"}

    git = git_cmd(stack_dir)
    state = {"ok": True, "is_git_checkout": True, "stack_dir": str(stack_dir)}
    state.update(_head_details(git))

    rc, branch = _run([*git, "symbolic-ref", "--quiet", "--short", "HEAD"])
    state["branch"] = branch if rc == 0 else ""
    state["detached"] = rc != 0
    state["upstream"] = _upstream_ref(git, state["branch"])

    rc, remote = _run([*git, "remote", "get-url", "origin"])
    state["remote_url"] = remote if rc == 0 else ""

    # Same condition `update.sh` gates on, untracked files included — it refuses
    # to run on any of them, so any of them belongs in the report. `-uall` only
    # changes granularity: bare `--porcelain` collapses a wholly-untracked
    # directory to `config/`, and naming the file is what makes the report
    # actionable.
    rc, porcelain = _run([*git, "status", "--porcelain", "--untracked-files=all"])
    # Split on the status column rather than slicing a fixed offset: the leading
    # space of an unstaged ` M path` does not survive being read back as text.
    dirty_paths = [
        parts[1] for parts in
        (line.split(maxsplit=1) for line in porcelain.splitlines() if line.strip())
        if len(parts) == 2
    ] if rc == 0 else []
    state["dirty"] = bool(dirty_paths)
    # `update.sh` refuses to run at all with a dirty tree, so the paths are the
    # actionable part of the report, not decoration.
    state["dirty_paths"] = sorted(dirty_paths)[:MAX_PENDING_COMMITS]
    return state


def _pending_commits(git: list[str], upstream: str) -> list[dict]:
    rc, out = _run([*git, "log", f"HEAD..{upstream}",
                    f"--max-count={MAX_PENDING_COMMITS}", "--pretty=%h%x1f%s"])
    if rc != 0 or not out:
        return []
    commits = []
    for line in out.splitlines():
        if "\x1f" not in line:
            continue
        short, subject = line.split("\x1f", 1)
        commits.append({"sha": short, "subject": subject})
    return commits


def backend_sensitive_changes(stack_dir, upstream: str) -> list[str]:
    """Which pending changes mean a model backend is on stale launcher code.

    Reported rather than acted on, matching `update.sh`: restarting a backend
    reloads the weights and throws away a warm prompt cache, and when to pay
    that is the operator's call.
    """
    if not upstream:
        return []
    git = git_cmd(stack_dir)
    rc, out = _run([*git, "diff", "--name-only", "HEAD", upstream, "--",
                    *(f"{path}*" for path in BACKEND_SENSITIVE_PATHS)])
    if rc != 0 or not out:
        return []
    return sorted({line.strip() for line in out.splitlines() if line.strip()})


def refresh_remote(stack_dir) -> dict:
    """Fetch, then compare. The whole snapshot, network included.

    Never raises: a box that cannot reach GitHub should report that it could not
    check, not lose the local half of the answer too.
    """
    state = local_state(stack_dir)
    state["remote_checked_at"] = time.time()
    state["fetch_error"] = ""
    state["behind"] = 0
    state["ahead"] = 0
    state["pending_commits"] = []
    state["backend_sensitive_changes"] = []
    state["remedy"] = REMEDY
    if not state.get("ok"):
        return state

    git = git_cmd(stack_dir)
    rc, out = _run([*git, "fetch", "--prune", "origin"], timeout=FETCH_TIMEOUT_SECONDS)
    if rc != 0:
        state["fetch_error"] = out or "git fetch failed"

    # The upstream ref may only exist after the fetch, so resolve it again.
    upstream = _upstream_ref(git, state.get("branch", ""))
    state["upstream"] = upstream
    if not upstream:
        state["fetch_error"] = state["fetch_error"] or "no upstream branch to compare against"
        return state

    rc, counts = _run([*git, "rev-list", "--left-right", "--count", f"HEAD...{upstream}"])
    if rc == 0:
        parts = counts.split()
        if len(parts) == 2 and all(p.isdigit() for p in parts):
            state["ahead"], state["behind"] = int(parts[0]), int(parts[1])

    if state["behind"]:
        state["pending_commits"] = _pending_commits(git, upstream)
        state["backend_sensitive_changes"] = backend_sensitive_changes(stack_dir, upstream)
    return state


def summarize(snapshot: dict) -> dict:
    """The one-line verdict the header badge renders.

    `dirty` outranks `behind` because a dirty tree is not just untidy: it is the
    condition under which `update.sh` refuses to run, so telling the operator to
    update without telling them that would send them into a failure.
    """
    if not snapshot or not snapshot.get("ok"):
        return {"state": "unknown",
                "message": snapshot.get("error", "deployment state unknown") if snapshot
                else "deployment state not checked yet"}

    behind = snapshot.get("behind", 0)
    behind_phrase = ""
    if behind:
        plural = "commit" if behind == 1 else "commits"
        behind_phrase = f"{behind} {plural} behind {snapshot.get('upstream') or 'upstream'}"
        if snapshot.get("backend_sensitive_changes"):
            behind_phrase += "; includes model backend launcher changes"

    if snapshot.get("dirty"):
        # Both facts matter and neither replaces the other: being behind is what
        # the operator wants to fix, and being dirty is why the fix will fail.
        blocked = ("the installed tree has local modifications, so update.sh "
                   "will refuse to run")
        return {"state": "dirty",
                "message": f"{behind_phrase} — {blocked}" if behind_phrase else blocked}
    if behind_phrase:
        return {"state": "behind", "message": behind_phrase}
    if snapshot.get("fetch_error"):
        return {"state": "unknown",
                "message": f"could not check for updates: {snapshot['fetch_error']}"}
    return {"state": "current", "message": "running the latest committed code"}


def check_interval(env: dict | None = None) -> float:
    """Seconds between background checks. `0` disables them entirely."""
    raw = str((env or {}).get("LLM_MANAGER_DEPLOY_CHECK_INTERVAL", "")).strip()
    if not raw:
        return float(DEFAULT_CHECK_INTERVAL_SECONDS)
    try:
        value = float(raw)
    except ValueError:
        return float(DEFAULT_CHECK_INTERVAL_SECONDS)
    if value <= 0:
        return 0.0
    # A minute is already far more often than a deployed tree can change.
    return max(60.0, value)


class DriftWatcher:
    """Keeps a cached deployment snapshot fresh, off the request path."""

    def __init__(self, stack_dir, interval: float = DEFAULT_CHECK_INTERVAL_SECONDS):
        self.stack_dir = stack_dir
        self.interval = interval
        self._lock = threading.Lock()
        self._snapshot: dict = {}
        self._thread: threading.Thread | None = None

    def check(self) -> dict:
        """Refresh now. Synchronous, so the button and the tests can call it."""
        try:
            snapshot = refresh_remote(self.stack_dir)
        except Exception as exc:
            snapshot = {"ok": False, "error": f"deployment check failed: {exc}",
                        "remote_checked_at": time.time()}
        with self._lock:
            self._snapshot = snapshot
        return snapshot

    def snapshot(self) -> dict:
        """The cached answer, never a network call."""
        with self._lock:
            if self._snapshot:
                return dict(self._snapshot)
        # Before the first background check completes, the local half is still
        # worth serving — it is what renders the version in the header.
        try:
            local = local_state(self.stack_dir)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        local["remote_checked_at"] = None
        return local

    def start(self, env_reader=None) -> None:
        """Idempotent, and a no-op when the interval is zero."""
        if self._thread is not None and self._thread.is_alive():
            return
        interval = self.interval
        if env_reader is not None:
            try:
                interval = check_interval(env_reader())
            except Exception:
                pass
        if interval <= 0:
            return
        self.interval = interval
        self._thread = threading.Thread(target=self._loop, name="deploy-drift", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while True:
            # A check that raises must not end the thread; a dead watcher would
            # freeze the badge on whatever it last said, which is worse than
            # saying nothing.
            self.check()
            time.sleep(self.interval)


__all__ = [
    "BACKEND_SENSITIVE_PATHS", "DEFAULT_CHECK_INTERVAL_SECONDS", "DriftWatcher",
    "REMEDY", "backend_sensitive_changes", "check_interval", "git_cmd",
    "is_git_checkout", "local_state", "refresh_remote", "summarize",
]
