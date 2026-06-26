#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "dependencies.json"


def run(cmd: list[str], *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("$ " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True)
    if check and result.returncode != 0:
        raise SystemExit(result.returncode)
    return result


def capture(cmd: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError((result.stdout + result.stderr).strip() or f"command failed: {' '.join(cmd)}")
    return result.stdout.strip()


def require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"Missing required command: {name}")


def node_version_key(path: Path) -> tuple[int, int, int]:
    name = path.name.removeprefix("v")
    parts = name.split(".")
    try:
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        patch = int(parts[2]) if len(parts) > 2 else 0
    except (IndexError, ValueError):
        return (0, 0, 0)
    return (major, minor, patch)


def prepend_nvm_node() -> None:
    if truthy_env("LLM_STACK_USE_SYSTEM_NODE"):
        return

    version_dirs: list[Path] = []
    nvm_dir = os.environ.get("NVM_DIR")
    if nvm_dir:
        version_dirs.extend((Path(nvm_dir) / "versions" / "node").glob("v*"))
    version_dirs.extend((Path.home() / ".nvm" / "versions" / "node").glob("v*"))

    seen: set[Path] = set()
    for node_dir in sorted(version_dirs, key=node_version_key, reverse=True):
        if node_dir in seen:
            continue
        seen.add(node_dir)
        bin_dir = node_dir / "bin"
        node = bin_dir / "node"
        npm = bin_dir / "npm"
        major, _, _ = node_version_key(node_dir)
        if major < 20 or not node.exists() or not npm.exists():
            continue
        current_path = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{bin_dir}{os.pathsep}{current_path}" if current_path else str(bin_dir)
        print(f"Using Node.js from {bin_dir}", flush=True)
        return


def find_or_bootstrap_uv() -> str:
    existing = shutil.which("uv")
    if existing:
        return existing

    venv_dir = ROOT / "deps" / "uv-venv"
    uv_bin = venv_dir / "bin" / "uv"
    if uv_bin.exists():
        return str(uv_bin)

    print(f"Bootstrapping local uv into {venv_dir}", flush=True)
    run([sys.executable, "-m", "venv", str(venv_dir)])
    pip = venv_dir / "bin" / "pip"
    run([str(pip), "install", "--upgrade", "pip"])
    run([str(pip), "install", "uv>=0.5.0"])
    if not uv_bin.exists():
        raise SystemExit(f"uv bootstrap did not create expected binary: {uv_bin}")
    return str(uv_bin)


def has_dirty_worktree(path: Path) -> bool:
    out = capture(["git", "status", "--porcelain"], cwd=path)
    return bool(out.strip())


def checkout_ref(path: Path, ref: str) -> None:
    run(["git", "fetch", "origin", "--prune"], cwd=path)
    remote_ref = f"origin/{ref}"
    result = subprocess.run(["git", "show-ref", "--verify", f"refs/remotes/{remote_ref}"], cwd=path)
    if result.returncode == 0:
        run(["git", "checkout", ref], cwd=path, check=False)
        current_branch = subprocess.run(["git", "branch", "--show-current"], cwd=path, capture_output=True, text=True).stdout.strip()
        if current_branch != ref:
            run(["git", "checkout", "-B", ref, remote_ref], cwd=path)
        run(["git", "pull", "--ff-only", "origin", ref], cwd=path)
    else:
        run(["git", "checkout", ref], cwd=path)


def truthy_env(name: str) -> bool:
    value = os.environ.get(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def build_cmake(dep: dict, jobs: int) -> None:
    source = ROOT / dep["path"]
    build_dir = ROOT / dep.get("build_dir", f"{dep['path']}/build")
    target = dep.get("target", "")
    cmake_args = [str(x) for x in dep.get("cmake_args", [])]
    build_dir.mkdir(parents=True, exist_ok=True)
    run(["cmake", "-S", str(source), "-B", str(build_dir), *cmake_args])
    build_cmd = ["cmake", "--build", str(build_dir)]
    if target:
        build_cmd.extend(["--target", target])
    build_cmd.extend(["--parallel", str(jobs)])
    run(build_cmd)
    binary = ROOT / dep.get("binary", "")
    if dep.get("binary") and not binary.exists():
        raise SystemExit(f"Expected dependency binary was not built: {binary}")
    if dep.get("require_gpu"):
        verify_gpu_binary(binary)


def verify_gpu_binary(binary: Path) -> None:
    probe_model = ROOT / ".llama-cuda-probe-missing-model.gguf"
    try:
        result = subprocess.run(
            [
                str(binary),
                "--model",
                str(probe_model),
                "--host",
                "127.0.0.1",
                "--port",
                "65534",
                "--n-gpu-layers",
                "1",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:
        raise SystemExit(f"Unable to verify GPU support for {binary}: {exc}") from exc
    output = (result.stdout + result.stderr).strip()
    lowered = output.lower()
    cpu_only_markers = (
        "compiled without support for gpu offload",
        "no usable gpu found",
        "ggml_cuda: not found",
    )
    if "cuda" not in lowered or any(marker in lowered for marker in cpu_only_markers):
        raise SystemExit(
            "Built llama-server does not appear to have CUDA GPU offload support. "
            "Refusing to install a CPU-only backend because large models can exhaust RAM. "
            "Probe output was:\n"
            f"{output}"
        )

def install_git_checkout(dep: dict, *, update: bool, force: bool) -> Path:
    path = ROOT / dep["path"]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", dep["repo"], str(path)])
        checkout_ref(path, dep.get("ref", "master"))
    else:
        if not (path / ".git").exists():
            raise SystemExit(f"Dependency path exists but is not a git checkout: {path}")
        if has_dirty_worktree(path) and not force:
            raise SystemExit(f"Refusing to update dirty dependency checkout: {path}")
        if update:
            checkout_ref(path, dep.get("ref", "master"))
    return path


def sync_uv(dep: dict, path: Path) -> None:
    uv = find_or_bootstrap_uv()
    uv_args = [str(x) for x in dep.get("uv_args", ["sync", "--frozen", "--no-group", "dev"])]
    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", str(ROOT / "deps" / ".uv-cache"))
    print("$ " + " ".join([uv, *uv_args]), flush=True)
    result = subprocess.run([uv, *uv_args], cwd=str(path), text=True, env=env)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def install_dependency(dep: dict, *, update: bool, force: bool, jobs: int) -> None:
    name = dep["name"]
    dep_type = dep.get("type")
    enabled_env = dep.get("enabled_env")
    if enabled_env and not truthy_env(enabled_env):
        print(f"\n== {name} ==\nSkipping because {enabled_env} is not enabled.", flush=True)
        return

    print(f"\n== {name} ==", flush=True)
    path = install_git_checkout(dep, update=update, force=force)
    if dep_type == "git-cmake":
        build_cmake(dep, jobs)
    elif dep_type == "git-uv":
        sync_uv(dep, path)
    else:
        raise SystemExit(f"Unsupported dependency type for {name}: {dep_type}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Install/update llm-stack external dependencies")
    parser.add_argument("--update", action="store_true", help="fetch and fast-forward existing dependency checkouts")
    parser.add_argument("--force", action="store_true", help="allow updating dirty dependency checkouts")
    parser.add_argument("--jobs", type=int, default=max(1, min((os.cpu_count() or 2) - 1, 4)))
    args = parser.parse_args()
    if args.jobs < 1:
        raise SystemExit("--jobs must be >= 1")
    prepend_nvm_node()
    require_tool("git")
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if any(dep.get("type") == "git-cmake" for dep in data.get("dependencies", [])):
        require_tool("cmake")
    for dep in data.get("dependencies", []):
        install_dependency(dep, update=args.update, force=args.force, jobs=args.jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
