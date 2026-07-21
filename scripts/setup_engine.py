#!/usr/bin/env python3
"""Durable fresh-machine setup engine used by the web UI and recovery CLI."""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import platform
import pwd
import re
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import uuid
from urllib import request as urlrequest
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = ROOT / "config" / "llm-stack.env"
STATE_FILE = ROOT / "config" / "install-state.json"
MODELS_DIR = ROOT / "models"
LOG_DIR = ROOT / "logs" / "setup"

CORE_DEFAULTS = ["primary", "embedding", "task", "ocr", "glmocr-sdk", "searxng", "playwright"]
OPTIONAL_COMPONENTS = ["secondary", "embedding2", "reranker", "honcho"]
ALL_COMPONENTS = CORE_DEFAULTS + OPTIONAL_COMPONENTS
MODEL_COMPONENTS = ["primary", "secondary", "embedding", "embedding2", "task", "ocr", "reranker"]
COMPONENT_DEPENDENCIES = {
    "glmocr-sdk": ["ocr"],
    "primary": ["chat-proxy"],
    "secondary": ["chat-proxy2"],
    "honcho": ["primary", "embedding"],
}
COMPONENT_SERVICES = {
    "primary": ["chat-backend-dense", "chat-proxy"],
    "secondary": ["chat-backend2", "chat-proxy2"],
    "embedding": ["embed"],
    "embedding2": ["embed2"],
    "task": ["task"],
    "ocr": ["ocr"],
    "glmocr-sdk": ["glmocr-sdk"],
    "reranker": ["rerank"],
    "playwright": ["playwright-server"],
    "honcho": ["honcho-api", "honcho-deriver"],
}
COMPONENT_PORTS = {
    "primary": [8003, 8004, 8008], "secondary": [8103, 8104, 8108],
    "embedding": [8005], "embedding2": [8011], "reranker": [8006],
    "task": [8007], "ocr": [8009], "glmocr-sdk": [5002],
    "searxng": [80], "playwright": [80], "honcho": [8090],
}
MODEL_ENV_KEYS = {
    "primary": ("CHAT_PRIMARY_MODEL_PATH", "CHAT_PRIMARY_MMPROJ_PATH"),
    "secondary": ("CHAT2_MODEL_PATH", "CHAT2_MMPROJ_PATH"),
    "embedding": ("EMBEDDING_MODEL_PATH", ""),
    "embedding2": ("EMBED2_MODEL_PATH", ""),
    "task": ("TASK_MODEL_PATH", "TASK_MMPROJ_PATH"),
    "ocr": ("OCR_MODEL_PATH", "OCR_MMPROJ_PATH"),
    "reranker": ("RERANKER_MODEL_PATH", ""),
}


def default_state() -> dict[str, Any]:
    return {
        "version": 1,
        "status": "new",
        "selection": {
            "components": list(CORE_DEFAULTS),
            "models": {},
            "allow_vram_override": False,
        },
        "preflight": {},
        "gpu_assignments": {},
        "component_versions": {},
        "completed_stages": [],
        "jobs": {},
        "last_validation": {},
        "updated_at": int(time.time()),
    }


def atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def load_state(path: Path = STATE_FILE) -> dict[str, Any]:
    state = default_state()
    if path.exists():
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(stored, dict):
                state.update(stored)
                state["selection"] = {**default_state()["selection"], **stored.get("selection", {})}
        except (OSError, ValueError):
            pass
    return state


def save_state(state: dict[str, Any], path: Path = STATE_FILE) -> dict[str, Any]:
    state["updated_at"] = int(time.time())
    atomic_json_write(path, state)
    return state


def resolve_components(values: list[str]) -> list[str]:
    selected = {item for item in values if item in ALL_COMPONENTS}
    changed = True
    while changed:
        changed = False
        for component in tuple(selected):
            for dependency in COMPONENT_DEPENDENCIES.get(component, []):
                if dependency in ALL_COMPONENTS and dependency not in selected:
                    selected.add(dependency)
                    changed = True
    return [component for component in ALL_COMPONENTS if component in selected]


def selected_ports(components: list[str]) -> list[int]:
    ports = {8077}
    for component in resolve_components(components):
        ports.update(COMPONENT_PORTS.get(component, []))
    return sorted(ports)


def firewall_rules(components: list[str], cidr: str) -> list[list[str]]:
    network = ipaddress.ip_network(cidr, strict=False)
    if not network.is_private:
        raise ValueError("Firewall source must be a private network")
    return [["ufw", "allow", "from", str(network), "to", "any", "port", str(port), "proto", "tcp"] for port in selected_ports(components)]


def run_capture(command: list[str], timeout: int = 20) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout)


def parse_os_release(path: Path = Path("/etc/os-release")) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip('"')
    return values


def parse_nvidia_gpus(text: str) -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            compute_cap = parts[2]
            gpus.append({
                "index": int(parts[0]),
                "name": parts[1],
                "compute_capability": compute_cap,
                "cmake_architecture": compute_cap.replace(".", ""),
                "memory_total_mib": int(float(parts[3])),
                "memory_free_mib": int(float(parts[4])),
            })
        except ValueError:
            continue
    return gpus


def driver_cuda_version(text: str) -> str:
    match = re.search(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)?)", text)
    return match.group(1) if match else ""


def choose_cuda_toolkit(driver_cuda: str, supported: tuple[str, ...] = ("13.3", "13.0", "12.8")) -> str:
    try:
        maximum = tuple(int(part) for part in driver_cuda.split(".")[:2])
    except ValueError:
        return ""
    for candidate in supported:
        parsed = tuple(int(part) for part in candidate.split("."))
        if parsed <= maximum:
            return candidate
    return ""


def detect_private_network() -> dict[str, str]:
    try:
        route = run_capture(["ip", "-j", "route", "show", "default"], timeout=5)
        entries = json.loads(route.stdout or "[]")
        interface = entries[0].get("dev", "") if entries else ""
        if not interface:
            return {}
        addr = run_capture(["ip", "-j", "address", "show", "dev", interface], timeout=5)
        for info in json.loads(addr.stdout or "[]"):
            for item in info.get("addr_info", []):
                if item.get("family") != "inet":
                    continue
                ip = ipaddress.ip_address(item["local"])
                if not ip.is_private:
                    continue
                network = ipaddress.ip_network(f"{ip}/{item['prefixlen']}", strict=False)
                return {"interface": interface, "address": str(ip), "cidr": str(network)}
    except Exception:
        pass
    return {}


def collect_preflight() -> dict[str, Any]:
    os_release = parse_os_release()
    checks: dict[str, dict[str, Any]] = {}
    checks["os"] = {
        "ok": os_release.get("ID") == "ubuntu" and os_release.get("VERSION_ID") == "24.04",
        "value": f"{os_release.get('ID', platform.system())} {os_release.get('VERSION_ID', platform.release())}".strip(),
        "required": "Ubuntu 24.04",
    }
    machine = platform.machine()
    checks["architecture"] = {"ok": machine in {"x86_64", "amd64"}, "value": machine, "required": "x86_64"}
    checks["systemd"] = {"ok": Path("/run/systemd/system").exists(), "value": "systemd"}
    checks["sudo"] = {"ok": os.geteuid() == 0 or shutil.which("sudo") is not None, "value": "available" if shutil.which("sudo") else "root-only"}
    usage = shutil.disk_usage(ROOT)
    checks["disk"] = {"ok": usage.free >= 20 * 1024**3, "free_bytes": usage.free, "required_bytes": 20 * 1024**3}
    internet_error = ""
    try:
        with socket.create_connection(("github.com", 443), timeout=5):
            pass
        internet_ok = True
    except Exception as exc:
        internet_ok = False
        internet_error = str(exc)
    checks["internet"] = {"ok": internet_ok, "target": "github.com:443", "error": internet_error}

    gpus: list[dict[str, Any]] = []
    cuda_version = ""
    error = ""
    try:
        result = run_capture([
            "nvidia-smi",
            "--query-gpu=index,name,compute_cap,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ], timeout=10)
        gpus = parse_nvidia_gpus(result.stdout) if result.returncode == 0 else []
        summary = run_capture(["nvidia-smi"], timeout=10)
        cuda_version = driver_cuda_version(summary.stdout + summary.stderr)
        error = (result.stderr or "").strip()
    except Exception as exc:
        error = str(exc)
    checks["nvidia_driver"] = {"ok": bool(gpus), "gpu_count": len(gpus), "error": error}
    toolkit = choose_cuda_toolkit(cuda_version)
    checks["cuda_compatibility"] = {
        "ok": bool(toolkit),
        "driver_cuda": cuda_version,
        "selected_toolkit": toolkit,
        "error": "Driver does not report compatibility with a supported CUDA toolkit" if not toolkit else "",
    }
    network = detect_private_network()
    checks["private_network"] = {"ok": bool(network), **network}
    ufw_active = False
    if shutil.which("ufw"):
        try:
            ufw_active = "Status: active" in run_capture(["ufw", "status"], timeout=5).stdout
        except Exception:
            pass
    checks["firewall"] = {"ok": ufw_active, "active": ufw_active, "warning": "No active UFW firewall detected" if not ufw_active else ""}
    required = ["os", "architecture", "systemd", "sudo", "disk", "internet", "nvidia_driver", "cuda_compatibility", "private_network"]
    return {
        "ok": all(checks[name]["ok"] for name in required),
        "checks": checks,
        "gpus": gpus,
        "cuda_toolkit": toolkit,
        "network": network,
        "trusted_lan_warning": "The manager is unauthenticated. Never expose it to the public internet.",
        "timestamp": int(time.time()),
    }


def validate_gguf(path: Path, *, minimum_size: int = 32) -> dict[str, Any]:
    result = {"ok": False, "path": str(path), "size": 0, "sha256": "", "error": ""}
    if not path.is_file():
        result["error"] = "File does not exist"
        return result
    size = path.stat().st_size
    result["size"] = size
    if size < minimum_size:
        result["error"] = "File is too small to be a GGUF model"
        return result
    with path.open("rb") as handle:
        magic = handle.read(4)
    if magic != b"GGUF":
        result["error"] = "Invalid GGUF header; HTML/XML/error downloads are rejected"
        return result
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    result.update(ok=True, sha256=digest.hexdigest())
    return result


def estimate_model_mib(model: dict[str, Any]) -> int:
    size = int(model.get("size") or 0)
    context_mib = int(model.get("context_mib") or 1024)
    return int((size / 1024**2) * 1.10) + context_mib


def plan_gpu_placement(gpus: list[dict[str, Any]], models: dict[str, dict[str, Any]], allow_override: bool = False) -> dict[str, Any]:
    if not gpus:
        return {"ok": False, "assignments": {}, "error": "No NVIDIA GPUs detected"}
    remaining = {int(gpu["index"]): int(gpu.get("memory_free_mib", gpu["memory_total_mib"]) * 0.90) for gpu in gpus}
    total = sum(remaining.values())
    required = sum(estimate_model_mib(model) for model in models.values())
    if required > total and not allow_override:
        return {"ok": False, "assignments": {}, "required_mib": required, "usable_mib": total, "error": "Selected models exceed 90% of available aggregate VRAM"}
    assignments: dict[str, Any] = {}
    ordered = sorted(gpus, key=lambda gpu: (remaining[int(gpu["index"])], -int(gpu["index"])), reverse=True)
    for component in ("primary", "secondary", "embedding", "embedding2", "task", "ocr", "reranker"):
        model = models.get(component)
        if not model:
            continue
        need = estimate_model_mib(model)
        if component in {"primary", "secondary", "ocr"} and max(remaining.values()) < need:
            candidates = [gpu for gpu in ordered if remaining[int(gpu["index"])] > 0]
        else:
            candidates = [max(ordered, key=lambda gpu: remaining[int(gpu["index"])] )]
        indices = [int(gpu["index"]) for gpu in candidates]
        capacities = [max(1, remaining[index]) for index in indices]
        cap_total = sum(capacities)
        splits = [round(capacity / cap_total, 4) for capacity in capacities]
        for index, split in zip(indices, splits):
            remaining[index] -= int(need * split)
        assignments[component] = {
            "gpu_indices": indices,
            "visible_devices": ",".join(str(index) for index in indices),
            "main_gpu": indices[0],
            "tensor_split": ",".join(str(max(1, capacity)) for capacity in capacities),
            "estimated_mib": need,
        }
        ordered = sorted(gpus, key=lambda gpu: (remaining[int(gpu["index"])], -int(gpu["index"])), reverse=True)
    layout_gpu = max(ordered, key=lambda gpu: remaining[int(gpu["index"])])
    assignments["glmocr-sdk"] = {"gpu_indices": [int(layout_gpu["index"])], "visible_devices": str(layout_gpu["index"])}
    return {"ok": True, "assignments": assignments, "required_mib": required, "usable_mib": total, "remaining_mib": remaining}


def quote_env(value: Any) -> str:
    text = str(value)
    if text == "":
        return '""'
    if re.fullmatch(r"[A-Za-z0-9_./,:@%+${}-]+", text):
        return text
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def update_env(updates: dict[str, Any], path: Path = CONFIG_FILE) -> None:
    content = path.read_text(encoding="utf-8") if path.exists() else ""
    for key, value in updates.items():
        rendered = quote_env(value)
        pattern = re.compile(rf"^{re.escape(key)}=.*$", re.MULTILINE)
        if pattern.search(content):
            content = pattern.sub(f"{key}={rendered}", content, count=1)
        else:
            content += f"\n{key}={rendered}\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(re.sub(r"\n{3,}", "\n\n", content).lstrip("\n"), encoding="utf-8")


def read_env(path: Path = CONFIG_FILE) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", line):
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip('"')
    return values


def probe_json(url: str, payload: dict[str, Any] | None = None, timeout: int = 120, attempts: int = 1) -> tuple[bool, str]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    last_error = ""
    for attempt in range(attempts):
        try:
            req = urlrequest.Request(url, data=body, headers=headers, method="POST" if payload is not None else "GET")
            with urlrequest.urlopen(req, timeout=timeout) as response:
                parsed = json.loads(response.read().decode("utf-8", errors="replace"))
            return True, json.dumps(parsed)[:500]
        except Exception as exc:
            last_error = str(exc)
            if attempt + 1 < attempts:
                time.sleep(5)
    return False, last_error


def placement_env(assignments: dict[str, Any]) -> dict[str, str]:
    mapping = {
        "primary": "CHAT_PRIMARY",
        "secondary": "CHAT2",
        "embedding": "EMBED",
        "embedding2": "EMBED2",
        "task": "TASK",
        "ocr": "OCR",
        "reranker": "RERANK",
    }
    updates: dict[str, str] = {}
    for component, prefix in mapping.items():
        item = assignments.get(component)
        if not item:
            continue
        updates[f"{prefix}_GPU_VISIBLE_DEVICES"] = item["visible_devices"]
        updates[f"{prefix}_MAIN_GPU"] = "0"
        updates[f"{prefix}_TENSOR_SPLIT"] = item["tensor_split"]
        if component in {"primary", "secondary"}:
            updates[f"{prefix}_DEVICE"] = ",".join(f"CUDA{i}" for i in range(len(item["gpu_indices"])))
    layout = assignments.get("glmocr-sdk")
    if layout:
        updates["GLMOCR_LAYOUT_CUDA_VISIBLE_DEVICES"] = layout["visible_devices"]
        updates["GLMOCR_LAYOUT_DEVICE"] = "cuda:0"
    return updates


class SetupRunner:
    def __init__(self, state_path: Path = STATE_FILE):
        self.state_path = state_path
        self.lock = threading.Lock()
        state = load_state(state_path)
        changed = False
        for job in state.get("jobs", {}).values():
            if job.get("status") in {"queued", "running"}:
                job.update(status="interrupted", error="Manager restarted while this job was active", updated_at=int(time.time()))
                changed = True
        if changed:
            save_state(state, state_path)

    def _mutate(self, callback: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        with self.lock:
            state = load_state(self.state_path)
            callback(state)
            return save_state(state, self.state_path)

    def create_job(self, retry_of: str = "") -> dict[str, Any]:
        job_id = uuid.uuid4().hex[:12]
        job = {"id": job_id, "status": "queued", "stage": "queued", "progress": 0, "error": "", "log": [], "retry_of": retry_of, "created_at": int(time.time()), "updated_at": int(time.time())}
        self._mutate(lambda state: state["jobs"].__setitem__(job_id, job))
        return job

    def job(self, job_id: str) -> dict[str, Any] | None:
        return load_state(self.state_path).get("jobs", {}).get(job_id)

    def _update_job(self, job_id: str, **updates: Any) -> None:
        def apply(state: dict[str, Any]) -> None:
            job = state["jobs"][job_id]
            job.update(updates)
            job["updated_at"] = int(time.time())
        self._mutate(apply)

    def _log(self, job_id: str, message: str) -> None:
        def apply(state: dict[str, Any]) -> None:
            job = state["jobs"][job_id]
            job.setdefault("log", []).append(message)
            job["log"] = job["log"][-500:]
            job["updated_at"] = int(time.time())
        self._mutate(apply)

    def _stage_done(self, stage: str) -> bool:
        return stage in load_state(self.state_path).get("completed_stages", [])

    def _complete_stage(self, stage: str) -> None:
        def apply(state: dict[str, Any]) -> None:
            completed = state.setdefault("completed_stages", [])
            if stage not in completed:
                completed.append(stage)
        self._mutate(apply)

    def _command(self, job_id: str, command: list[str], timeout: int = 3600, env: dict[str, str] | None = None) -> None:
        self._log(job_id, "$ " + " ".join(command))
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=timeout, env=env)
        output = (result.stdout + result.stderr).strip()
        if output:
            for line in output.splitlines()[-100:]:
                self._log(job_id, line)
        if result.returncode != 0:
            raise RuntimeError(output or f"Command failed with exit code {result.returncode}")

    def _owner_command(self, command: list[str]) -> list[str]:
        if os.geteuid() != 0:
            return command
        owner = pwd.getpwuid(ROOT.stat().st_uid).pw_name
        if owner == "root":
            return command
        return ["sudo", "-H", "-u", owner, *command]

    def run(self, job_id: str) -> None:
        try:
            self._update_job(job_id, status="running", stage="preflight", progress=5)
            preflight = collect_preflight()
            self._mutate(lambda state: state.__setitem__("preflight", preflight))
            if not preflight["ok"]:
                raise RuntimeError("Required preflight checks failed")
            self._complete_stage("preflight")
            state = load_state(self.state_path)
            selection = state["selection"]
            components = resolve_components(selection.get("components", []))
            self._complete_stage("components")
            models = selection.get("models", {})
            for component in components:
                if component not in MODEL_COMPONENTS:
                    continue
                model_path = Path(str(models.get(component, {}).get("path", "")))
                validation = validate_gguf(model_path)
                if not validation["ok"]:
                    raise RuntimeError(f"{component}: {validation['error']}")
                models[component].update(validation)
            placement = plan_gpu_placement(preflight["gpus"], {key: value for key, value in models.items() if key in components}, bool(selection.get("allow_vram_override")))
            if not placement["ok"]:
                raise RuntimeError(placement["error"])
            update_env(placement_env(placement["assignments"]))
            model_updates: dict[str, str] = {}
            for component, model in models.items():
                if component not in components or component not in MODEL_ENV_KEYS:
                    continue
                model_key, mmproj_key = MODEL_ENV_KEYS[component]
                model_updates[model_key] = model["path"]
                if mmproj_key:
                    model_updates[mmproj_key] = model.get("mmproj_path", "")
            model_updates.update({
                "SEARXNG_ENABLED": "on" if "searxng" in components else "off",
                "PLAYWRIGHT_ENABLED": "on" if "playwright" in components else "off",
                "GLMOCR_SDK_ENABLED": "on" if "glmocr-sdk" in components else "off",
                "HONCHO_ENABLED": "on" if "honcho" in components else "off",
                "LLM_STACK_SELECTED_COMPONENTS": ",".join(components),
            })
            update_env(model_updates)
            self._mutate(lambda current: current.update(gpu_assignments=placement["assignments"], selection={**selection, "components": components, "models": models}))
            self._complete_stage("models")
            self._complete_stage("gpu-placement")

            self._update_job(job_id, stage="dependencies", progress=20)
            if not self._stage_done("dependencies"):
                self._command(job_id, ["bash", str(ROOT / "scripts" / "install-system-dependencies.sh"), "--full"], timeout=3600)
                if "searxng" in components or "playwright" in components:
                    self._command(job_id, ["bash", str(ROOT / "scripts" / "install-nginx-stack.sh")], timeout=120)
                if any(component in components for component in MODEL_COMPONENTS):
                    dependency_command = ["env", f"HONCHO_ENABLED={'on' if 'honcho' in components else 'off'}", str(ROOT / "scripts" / "install-dependencies.py"), "--update"]
                    self._command(job_id, self._owner_command(dependency_command), timeout=7200)
                self._complete_stage("dependencies")
            else:
                self._log(job_id, "Skipping completed dependency stage")
            if "searxng" in components:
                self._update_job(job_id, stage="searxng", progress=40)
                if not self._stage_done("searxng"):
                    self._command(job_id, ["bash", str(ROOT / "scripts" / "install-searxng.sh")], timeout=1800)
                    self._complete_stage("searxng")
            if "playwright" in components:
                self._update_job(job_id, stage="playwright", progress=50)
                if not self._stage_done("playwright"):
                    self._command(job_id, ["bash", str(ROOT / "scripts" / "install-playwright.sh")], timeout=1800)
                    self._complete_stage("playwright")
            if "glmocr-sdk" in components:
                self._update_job(job_id, stage="glmocr-sdk", progress=60)
                if not self._stage_done("glmocr-sdk"):
                    self._command(job_id, self._owner_command(["bash", str(ROOT / "scripts" / "install-glmocr-sdk.sh")]), timeout=3600)
                    self._complete_stage("glmocr-sdk")

            self._update_job(job_id, stage="services", progress=75)
            if not self._stage_done("services"):
                env = os.environ.copy()
                env.update({"LLM_STACK_SKIP_DEP_UPDATE": "1", "LLM_STACK_SKIP_EXTERNAL_INSTALL": "1", "LLM_STACK_SETUP_COMPONENTS": ",".join(components)})
                self._command(job_id, ["bash", str(ROOT / "install.sh"), "--configure-services"], timeout=1800, env=env)
                self._command(job_id, ["bash", str(ROOT / "scripts" / "activate-selected-stack.sh")], timeout=900)
                self._complete_stage("services")
                self._complete_stage("network")
                self._complete_stage("install")

            validation = validate_installation(components)
            versions = collect_component_versions(components)
            self._complete_stage("validation")
            self._complete_stage("completion")
            self._mutate(lambda current: current.update(status="complete" if validation["ok"] else "needs_attention", component_versions=versions, last_validation=validation))
            self._update_job(job_id, status="complete" if validation["ok"] else "needs_attention", stage="complete", progress=100)
        except Exception as exc:
            self._log(job_id, f"ERROR: {exc}")
            self._update_job(job_id, status="failed", error=str(exc))
            self._mutate(lambda state: state.__setitem__("status", "failed"))


def validate_installation(components: list[str] | None = None) -> dict[str, Any]:
    components = resolve_components(components or load_state().get("selection", {}).get("components", []))
    checks: dict[str, Any] = {}
    for component in components:
        services = COMPONENT_SERVICES.get(component, [])
        if component == "searxng":
            services = ["uwsgi", "nginx"]
        if not services:
            continue
        component_checks = []
        for service in services:
            result = run_capture(["systemctl", "is-active", service], timeout=5)
            component_checks.append({"service": service, "ok": result.stdout.strip() == "active", "status": result.stdout.strip() or result.stderr.strip()})
        checks[component] = {"ok": all(item["ok"] for item in component_checks), "services": component_checks}
    env = read_env()
    endpoint_checks: dict[str, Any] = {}
    if "primary" in components:
        payload = {"model": env.get("NOTHINK_MODEL_NAME", "chat"), "messages": [{"role": "user", "content": "Reply with exactly OK"}], "max_tokens": 8, "temperature": 0}
        ok, detail = probe_json(f"http://127.0.0.1:{env.get('NOTHINK_PORT', '8004')}/v1/chat/completions", payload, attempts=60)
        endpoint_checks["primary_chat"] = {"ok": ok, "detail": detail}
    if "embedding" in components:
        payload = {"model": env.get("EMBED_MODEL_NAME", "embed"), "input": "setup validation"}
        ok, detail = probe_json(f"http://127.0.0.1:{env.get('EMBED_PORT', '8005')}/v1/embeddings", payload, attempts=30)
        endpoint_checks["embedding"] = {"ok": ok and "embedding" in detail, "detail": detail}
    if "task" in components:
        payload = {"model": env.get("TASK_MODEL_NAME", "task"), "messages": [{"role": "user", "content": "Reply OK"}], "max_tokens": 8}
        ok, detail = probe_json(f"http://127.0.0.1:{env.get('TASK_PORT', '8007')}/v1/chat/completions", payload, attempts=30)
        endpoint_checks["task"] = {"ok": ok, "detail": detail}
    if "ocr" in components:
        pixel = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
        payload = {"model": env.get("OCR_MODEL_NAME", "ocr"), "messages": [{"role": "user", "content": [{"type": "text", "text": "OCR"}, {"type": "image_url", "image_url": {"url": pixel}}]}], "max_tokens": 8}
        ok, detail = probe_json(f"http://127.0.0.1:{env.get('OCR_PORT', '8009')}/v1/chat/completions", payload, attempts=30)
        endpoint_checks["ocr"] = {"ok": ok, "detail": detail}
    if "glmocr-sdk" in components:
        ok, detail = probe_json(f"http://127.0.0.1:{env.get('GLMOCR_SDK_PORT', '5002')}/health", attempts=30)
        endpoint_checks["glmocr_sdk"] = {"ok": ok, "detail": detail}
    if "searxng" in components:
        ok, detail = probe_json(f"http://127.0.0.1{env.get('SEARXNG_URL_PATH', '/searxng')}/search?q=setup&format=json", attempts=12)
        endpoint_checks["searxng"] = {"ok": ok and "results" in detail, "detail": detail}
    if "playwright" in components:
        try:
            result = subprocess.run(["node", str(ROOT / "playwright" / "test-remote.js"), "--host", "127.0.0.1", "--port", env.get("PLAYWRIGHT_PORT", "3001")], cwd=ROOT / "playwright", capture_output=True, text=True, timeout=90)
            endpoint_checks["playwright"] = {"ok": result.returncode == 0, "detail": (result.stdout + result.stderr)[-1000:]}
        except Exception as exc:
            endpoint_checks["playwright"] = {"ok": False, "detail": str(exc)}
    services_ok = bool(checks) and all(item["ok"] for item in checks.values())
    endpoints_ok = all(item["ok"] for item in endpoint_checks.values())
    return {"ok": services_ok and endpoints_ok, "checks": checks, "endpoint_checks": endpoint_checks, "timestamp": int(time.time())}


def collect_component_versions(components: list[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    try:
        manifest = json.loads((ROOT / "dependencies.json").read_text(encoding="utf-8"))
        llama = next(item for item in manifest.get("dependencies", []) if item.get("name") == "llama.cpp")
        versions["llama.cpp"] = str(llama.get("ref", ""))
    except Exception:
        pass
    commands = {"node": ["node", "--version"], "cuda": ["nvcc", "--version"]}
    for name, command in commands.items():
        try:
            result = run_capture(command, timeout=10)
            versions[name] = (result.stdout + result.stderr).strip().splitlines()[-1]
        except Exception:
            pass
    if "glmocr-sdk" in components:
        versions["glmocr"] = read_env().get("GLMOCR_SDK_VERSION", "0.1.5")
    if "playwright" in components:
        try:
            versions["playwright"] = json.loads((ROOT / "playwright" / "package-lock.json").read_text(encoding="utf-8"))["packages"]["node_modules/playwright"]["version"]
        except Exception:
            pass
    return versions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    sub.add_parser("state")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--wait", action="store_true")
    select_parser = sub.add_parser("select")
    select_parser.add_argument("--components", required=True, help="comma-separated component ids")
    select_parser.add_argument("--model", action="append", default=[], help="component=/absolute/model.gguf")
    retry_parser = sub.add_parser("retry")
    retry_parser.add_argument("job_id")
    sub.add_parser("validate")
    args = parser.parse_args()
    if args.command == "preflight":
        print(json.dumps(collect_preflight(), indent=2))
    elif args.command == "state":
        print(json.dumps(load_state(), indent=2))
    elif args.command == "validate":
        print(json.dumps(validate_installation(), indent=2))
    elif args.command == "select":
        state = load_state()
        components = resolve_components([item.strip() for item in args.components.split(",")])
        models = dict(state["selection"].get("models", {}))
        for value in args.model:
            component, separator, path = value.partition("=")
            if not separator or component not in MODEL_COMPONENTS or not path.startswith("/"):
                raise SystemExit(f"Invalid --model value: {value}")
            models[component] = {"path": path, "size": Path(path).stat().st_size if Path(path).exists() else 0}
        state["selection"] = {"components": components, "models": models, "allow_vram_override": False}
        state["completed_stages"] = [stage for stage in state.get("completed_stages", []) if stage == "preflight"]
        state["status"] = "selection_changed"
        save_state(state)
        print(json.dumps(state["selection"], indent=2))
    else:
        runner = SetupRunner()
        job = runner.create_job(getattr(args, "job_id", ""))
        if getattr(args, "wait", False):
            runner.run(job["id"])
            job = runner.job(job["id"]) or job
        else:
            thread = threading.Thread(target=runner.run, args=(job["id"],), daemon=False)
            thread.start()
        print(json.dumps(job, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
