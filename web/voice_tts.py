"""Hermes's voice (scripts/voice-tts-server.py), as the manager sees it.

The server's settings are in `llm-stack.env` like every other service's, but
which voices exist and which one is the default live in its voices file, because
the server re-reads that file's `default` on every request -- changing it there
is the one setting that needs no restart. So the page edits that one field in
place and leaves the rest of the file to the voice tools that write it.

The manager runs as root while the service runs as the stack's owner, so "~"
here means the owner's home, not root's: that is where the server looks.
"""

from __future__ import annotations

import json
import os
import pwd
import shutil
from pathlib import Path

import core

UNIT = "voice-tts"
DEFAULT_PORT = "8016"


def service_home() -> Path:
    """The home directory of the user the unit runs as (the checkout's owner)."""
    try:
        return Path(pwd.getpwuid(core.STACK_DIR.stat().st_uid).pw_dir)
    except (KeyError, OSError):
        return Path.home()


def _path(value: str, default: Path) -> Path:
    value = str(value or "").strip()
    if not value:
        return default
    if value.startswith("~/"):
        return service_home() / value[2:]
    return Path(value)


def config(env: dict) -> dict:
    home = service_home()
    host = str(env.get("VOICE_TTS_HOST") or "127.0.0.1").strip()
    port = str(env.get("VOICE_TTS_PORT") or DEFAULT_PORT).strip()
    local_host = "127.0.0.1" if host in {"0.0.0.0", "::", ""} else host
    local_url = f"http://{local_host}:{port}"
    return {
        "enabled": str(env.get("VOICE_TTS_ENABLED") or "off").strip(),
        "host": host,
        "port": port,
        "gpu": str(env.get("VOICE_TTS_GPU") or "0").strip(),
        "local_url": local_url,
        "public_url": str(env.get("VOICE_TTS_PUBLIC_URL") or "").strip() or local_url,
        "python": str(_path(env.get("VOICE_TTS_PYTHON"), home / "AI/voice-tts/venv/bin/python")),
        "voices_file": str(_path(env.get("VOICE_TTS_VOICES"), home / "AI/voice-tts/voices.json")),
        "clone_model": str(env.get("VOICE_TTS_CLONE_MODEL") or "").strip() or "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "preset_model": str(env.get("VOICE_TTS_MODEL") or "").strip() or "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "language": str(env.get("VOICE_TTS_LANGUAGE") or "").strip() or "English",
        "service_unit": f"/etc/systemd/system/{UNIT}.service",
        "start_script": str(core.SCRIPTS_DIR / "start-voice-tts.sh"),
    }


def read_voices(path: str | Path) -> dict:
    """{"default": name, "voices": {name: "clone" | "preset"}} from the voices file."""
    data = json.loads(Path(path).read_text())
    voices = data.get("voices") or {}
    return {
        "default": str(data.get("default") or "").lower(),
        "voices": {str(name).lower(): ("clone" if "ref_audio" in (spec or {}) else "preset")
                   for name, spec in voices.items()},
    }


def set_default_voice(path: str | Path, voice: str) -> str:
    """Point the voices file's `default` at `voice`, keeping a `.bak` beside it.

    Written in place rather than replaced, so the file keeps its owner: the
    manager is root and the voice tools that also edit it are not. Returns the
    previous default.
    """
    path = Path(path)
    voice = str(voice or "").strip().lower()
    data = json.loads(path.read_text())
    names = {str(name).lower() for name in (data.get("voices") or {})}
    if voice not in names:
        raise ValueError(f"{voice!r} is not in {path} (voices: {', '.join(sorted(names)) or 'none'})")
    previous = str(data.get("default") or "")
    backup = path.with_name(path.name + ".bak")
    shutil.copyfile(path, backup)
    try:
        st = path.stat()
        os.chown(backup, st.st_uid, st.st_gid)
    except OSError:
        pass
    data["default"] = voice
    with open(path, "r+", encoding="utf-8") as fh:
        fh.write(json.dumps(data, indent=2) + "\n")
        fh.truncate()
    return previous
