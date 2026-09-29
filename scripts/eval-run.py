#!/usr/bin/env python3
"""Serve model candidates one at a time and run the eval suites against each.

    scripts/eval-run.py run  --candidates config/eval-candidates.json
    scripts/eval-run.py run  --only flash-next-3bit-g64 --suites speed,quality
    scripts/eval-run.py serve flash-next-3bit-g64     # start one, leave it up
    scripts/eval-run.py stop
    scripts/eval-run.py plan                          # print each launch command, start nothing
    scripts/eval-report.py                            # compare everything run so far

A candidate is a model plus the slot settings it should run with -- the same
`{PREFIX}_*` keys the Config page writes (MODEL_PATH, MEMORY_MODE, RAM_BUDGET_GB,
SPEC_METHOD, ...), given without the prefix. It is launched the way the stack
launches a slot: the env file is sourced, the candidate's settings are laid
over it, and `scripts/lib/build-backend-command.py` builds the command. So a
candidate measures exactly what the slot would run, offload and all, and
config/llm-stack.env is never written.

It serves on one slot's backend port (llm-b's 8020 by default) and refuses if
something it did not start is listening there: stop that slot first. Suites:

  speed     scripts/bench-offload.py  decode/prefill tok/s, memory, SSD reads by depth
  quality   scripts/eval-quality.py   math, code, multiple-choice knowledge
  workload  scripts/eval-workload.py  long-context synthesis, grounding,
                                      classification, qualitative coding, prose samples

Results go to benchmarks/<candidate>/<suite>-<date>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "web"))

from backends.slots import SLOTS  # noqa: E402

STATE = ROOT / "benchmarks" / ".eval-server.json"
SUITES = {
    "speed": "bench-offload.py",
    "quality": "eval-quality.py",
    "workload": "eval-workload.py",
}
#: The two slots a candidate can borrow: they hold a conversation, sit behind
#: the chat proxy, and take either engine.
PORTS = {"llm-a": ("CHAT_BACKEND_PORT", "CHAT_BACKEND_HOST", "8010"),
         "llm-b": ("CHAT2_BACKEND_PORT", "CHAT2_BACKEND_HOST", "8020")}


def sourced_env(env_file: pathlib.Path) -> dict[str, str]:
    """The env file as the launcher sees it: sourced by bash, so ${VAR} expands."""
    if not env_file.is_file():
        return dict(os.environ)
    out = subprocess.run(["bash", "-c", 'set -a; source "$1" >/dev/null 2>&1; env -0', "_",
                          str(env_file)], capture_output=True, check=True).stdout
    return dict(item.split("=", 1) for item in out.decode().split("\0") if "=" in item)


def load_candidates(path: pathlib.Path) -> tuple[str, list[dict]]:
    data = json.loads(path.read_text())
    slot = data.get("slot", "llm-b")
    if slot not in PORTS:
        raise SystemExit(f"slot must be one of {', '.join(PORTS)}")
    return slot, data["candidates"]


#: Where models and runtimes live: the env file's STACK_DIR. The same as ROOT
#: for an installed stack; different when this runs from a second checkout.
DATA_ROOTS: list[pathlib.Path] = [ROOT]


def _resolve(value: str) -> str:
    """Relative paths in a candidates file are relative to the stack."""
    if value and not value.startswith("/") and "/" in value:
        for root in DATA_ROOTS:
            if (root / value).exists():
                return str(root / value)
    return value


def candidate_env(slot_name: str, candidate: dict, base: dict[str, str]) -> dict[str, str]:
    slot = SLOTS[slot_name]
    port_key, host_key, default_port = PORTS[slot_name]
    env = dict(base)
    data_root = base.get("STACK_DIR") or str(ROOT)
    env.setdefault("MTPLX_VENV", f"{data_root}/deps/mtplx-venv")
    env.update({
        "STACK_DIR": str(ROOT),
        "LLM_BACKEND_SLOT": slot_name,
        host_key: "127.0.0.1",
        port_key: env.get(port_key) or default_port,
        # The alias is the candidate's name, so every result names its model.
        f"{slot.prefix}_MODEL_NAME": candidate["name"],
        # Neither leaks in from the slot's own saved configuration: a
        # projector for a different model, or llama.cpp flags meant for it.
        f"{slot.prefix}_MMPROJ_PATH": "",
        f"{slot.prefix}_CUSTOM_ARGS_JSON": "[]",
    })
    for key, value in (candidate.get("settings") or {}).items():
        env[f"{slot.prefix}_{key}"] = _resolve(str(value))
    for key, value in (candidate.get("env") or {}).items():
        env[key] = str(value)
    if env.get(f"{slot.prefix}_MEMORY_MODE") == "ssd-offload":
        env.pop("GGML_METAL_RESIDENCY_KEEP_ALIVE_S", None)
    else:
        env.setdefault("GGML_METAL_RESIDENCY_KEEP_ALIVE_S", "8640000")
    binary = env.get(f"{slot.prefix}_LLAMA_SERVER_BIN") or env.get("LLAMA_SERVER_BIN") or ""
    if binary:
        env["DYLD_LIBRARY_PATH"] = str(pathlib.Path(binary).parent)
    return env


def _port_owner(port: str) -> int | None:
    out = subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
                         capture_output=True, text=True).stdout.split()
    return int(out[0]) if out else None


def _state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except (OSError, ValueError):
        return {}


def _alive(pid: int) -> bool:
    """Whether `pid` still runs. A zombie is not running, and neither is a
    process macOS will not let us signal: a server started by an earlier
    runner is a zombie until *that* runner reaps it, and signalling its group
    from here answers EPERM rather than ESRCH."""
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    stat = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)],
                          capture_output=True, text=True).stdout.strip()
    return bool(stat) and not stat.startswith("Z")


def _signal(pid: int, sig: int) -> None:
    for send in (lambda: os.killpg(pid, sig), lambda: os.kill(pid, sig)):
        try:
            send()
            return
        except (ProcessLookupError, PermissionError):
            continue


def stop() -> None:
    state = _state()
    pid = state.get("pid")
    if pid:
        _signal(pid, signal.SIGTERM)
        for _ in range(60):
            if not _alive(pid):
                break
            time.sleep(1)
        else:
            _signal(pid, signal.SIGKILL)
    # Whatever else is left on the port was ours: the state names its URL.
    port = (state.get("url") or "").rsplit(":", 1)[-1]
    if port.isdigit():
        for _ in range(30):
            owner = _port_owner(port)
            if not owner:
                break
            _signal(owner, signal.SIGTERM)
            time.sleep(1)
    STATE.unlink(missing_ok=True)


def _is_ready(url: str) -> bool:
    """`/ready` where the server has one (Splash: `/health` answers before the
    model has loaded), else `/health`."""
    import urllib.error
    try:
        with urllib.request.urlopen(f"{url}/ready", timeout=3):
            return True
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            return False
    except Exception:
        return False
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=3):
            return True
    except Exception:
        return False


def wait_healthy(url: str, pid: int, timeout: float = 900) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pid and not _alive(pid):
            return False
        if _is_ready(url):
            return True
        time.sleep(3)
    return False


def build(slot_name: str, candidate: dict, env: dict[str, str]) -> list[str]:
    built = subprocess.run([sys.executable, str(ROOT / "scripts/lib/build-backend-command.py"),
                            *[_resolve(str(a)) for a in candidate.get("args", [])]],
                           env=env, capture_output=True)
    if built.returncode != 0:
        raise SystemExit(f"{candidate['name']}: {built.stderr.decode().strip()}")
    return [a for a in built.stdout.decode().split("\0") if a]


def serve(slot_name: str, candidate: dict, env_file: pathlib.Path) -> str:
    """Start one candidate, detached; return its base URL once healthy."""
    stop()
    env = candidate_env(slot_name, candidate, sourced_env(env_file))
    port = env[PORTS[slot_name][0]]
    owner = _port_owner(port)
    if owner:
        raise SystemExit(f"port {port} is in use by pid {owner}, which this did not start; "
                         f"stop {slot_name} first")
    argv = build(slot_name, candidate, env)
    logs = ROOT / "benchmarks" / candidate["name"]
    logs.mkdir(parents=True, exist_ok=True)
    log = open(logs / f"server-{time.strftime('%Y%m%d-%H%M%S')}.log", "w")
    log.write(" ".join(argv) + "\n\n")
    log.flush()
    process = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    url = f"http://127.0.0.1:{port}"
    STATE.write_text(json.dumps({"pid": process.pid, "candidate": candidate["name"],
                                 "url": url, "slot": slot_name}))
    if not wait_healthy(url, process.pid):
        stop()
        raise SystemExit(f"{candidate['name']}: did not become healthy; see {log.name}")
    return url


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("run", "serve", "plan"):
        p = sub.add_parser(name)
        p.add_argument("--candidates", default=str(ROOT / "config/eval-candidates.json"))
        p.add_argument("--env-file", default=str(ROOT / "config/llm-stack.env"))
        if name == "run":
            p.add_argument("--only", default="", help="comma-separated candidate names")
            p.add_argument("--suites", default="speed,quality,workload")
            p.add_argument("--suite-args", default="",
                           help="extra arguments passed to every suite, e.g. '--depths 0,8192'")
        elif name == "serve":
            p.add_argument("name")
    sub.add_parser("stop")
    args = parser.parse_args(argv)

    if args.action == "stop":
        stop()
        return 0

    slot_name, candidates = load_candidates(pathlib.Path(args.candidates))
    data_root = sourced_env(pathlib.Path(args.env_file)).get("STACK_DIR")
    if data_root and pathlib.Path(data_root) != ROOT:
        DATA_ROOTS.insert(0, pathlib.Path(data_root))
    by_name = {c["name"]: c for c in candidates}
    env_file = pathlib.Path(args.env_file)

    if args.action == "plan":
        base = sourced_env(env_file)
        for candidate in candidates:
            try:
                argv = build(slot_name, candidate, candidate_env(slot_name, candidate, base))
                print(f"{candidate['name']}:\n  " + " ".join(argv) + "\n")
            except SystemExit as exc:
                print(f"{candidate['name']}: CANNOT BUILD -- {exc}\n")
        return 0

    if args.action == "serve":
        if args.name not in by_name:
            raise SystemExit(f"no candidate {args.name!r}")
        print(serve(slot_name, by_name[args.name], env_file))
        return 0

    chosen = [by_name[n] for n in args.only.split(",") if n] if args.only else candidates
    suites = [s for s in args.suites.split(",") if s in SUITES]
    restart = (f"{sys.executable} {ROOT / 'scripts/eval-run.py'} serve "
               f"--candidates {args.candidates} --env-file {args.env_file}")
    try:
        for candidate in chosen:
            print(f"== {candidate['name']} {time.strftime('%H:%M:%S')}", flush=True)
            try:
                url = serve(slot_name, candidate, env_file)
            except SystemExit as exc:
                print(f"   skipped: {exc}", flush=True)
                continue
            out = ROOT / "benchmarks" / candidate["name"]
            for suite in suites:
                if suite != "speed":
                    # A backend that starts refusing (MTPLX's memory guard) is
                    # restarted and the question asked again, not scored wrong.
                    extra = ["--restart-cmd", f"{restart} {candidate['name']}"]
                else:
                    extra = ["--pid", str(_state().get("pid") or 0)]
                cmd = [sys.executable, str(ROOT / "scripts" / SUITES[suite]), "--url", url,
                       "--variant", candidate["name"], "--out", str(out), *extra,
                       *args.suite_args.split()]
                print(f"   {suite} ...", flush=True)
                done = subprocess.run(cmd, capture_output=True, text=True)
                tail = (done.stdout or done.stderr).strip().splitlines()[-6:]
                print("\n".join(f"   {line}" for line in tail), flush=True)
                # A suite that restarted the backend left it up under a new pid;
                # one that crashed it did not, and the next suite needs one.
                if not wait_healthy(url, _state().get("pid") or 0, timeout=5):
                    url = serve(slot_name, candidate, env_file)
    finally:
        stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
