#!/usr/bin/env python3
"""
The model catalogue: what is on disk, what can be fetched, and how it is launched.

Five related things live here because they are the same question asked from
different directions — "which model, and with what arguments":

  * **GGUF files** on disk, and the classifier that tells a model apart from its
    multimodal projector. Size matters as well as name: a projector is a few
    hundred MB beside a model of tens of GB, and files are named inconsistently
    enough that the name alone is not decisive.
  * **Chat templates** — Jinja files passed to llama-server with `--chat-template-file`,
    stored with a metadata sidecar so the UI can show a name rather than a path.
  * **Custom models** — user-defined launch presets, each carrying its own extra
    llama.cpp arguments. `resolve_custom_args_for_model` is why adding a Qwen3.6
    model picks up `--jinja` automatically: arguments are matched by inferred
    model family, not typed out per model.
  * **Transcription models**, which are directories of weights rather than single
    files, and so are catalogued by scanning rather than globbing.
  * **HuggingFace downloads**, streamed to disk on a background thread with a
    job registry the UI polls. Downloads are validated as they land — a
    truncated GGUF is otherwise discovered at launch, minutes later, as an
    unhelpful llama.cpp error.

`HF_ALLOWED_HOSTS` bounds where a download may come from. The repo reference
arrives from the browser as free text, so without it a crafted value would make
the manager — running as root — fetch from anywhere.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import threading
import time
import uuid
from pathlib import Path
from urllib import error as urlerror
from urllib import request as urlrequest
from urllib.parse import quote, unquote, urlparse

import jinja2

import backends
import budget
import config_env
import core
import setup_engine
from config_fields import (
        TRANSCRIPTION_ENGINE_BY_ID,
    TRANSCRIPTION_ENGINES,
    WHISPER_MODEL_PRESETS,
)


HF_ALLOWED_HOSTS = {"huggingface.co", "www.huggingface.co", "hf.co"}
HF_DOWNLOAD_JOBS: dict[str, dict] = {}
HF_DOWNLOAD_JOBS_LOCK = threading.Lock()
# The engine registry lives in `config_fields` because `config_env` has to loop
# it to set per-engine defaults and cannot import this module — `models` imports
# `config_env`, so the dependency only runs one way. Re-exported here because
# this is where callers have always found it.
TRANSCRIPTION_MODEL_PRESETS = WHISPER_MODEL_PRESETS

BUILTIN_CUSTOM_MODEL_ARG_PRESETS = {
    "qwen3.6": [
        "--chat-template-kwargs '{\"preserve_thinking\": true}'",
        "--jinja",
    ],
}



def chat_template_id_from_name(name: str) -> str:
    base = re.sub(r"[^A-Za-z0-9._-]+", "-", (name or "").strip()).strip("-._")
    return base[:80] or f"template-{int(time.time())}"


def validate_chat_template_id(template_id: str) -> str:
    template_id = (template_id or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,80}", template_id):
        raise ValueError("Template id may only contain letters, numbers, dot, underscore, and dash")
    return template_id


def validate_chat_template_content(content: str) -> str:
    """Reject template text that llama.cpp would load but silently mis-render.

    A Jinja parse check alone is not enough. A chat template pasted from a
    model card's web page carries the surrounding page chrome — "Hugging Face's
    logo", "Files", "Community", the file size — and every line of that is
    valid Jinja *literal text*, so it compiles clean and then prepends itself
    to the prompt of every request. That is how `Qwen3.6.jinja` shipped with 56
    lines of scraped page above the template and went unnoticed while serving.

    Every real chat template starts with a Jinja tag, so requiring the first
    non-whitespace character to open one catches the paste at the point it is
    saved rather than in the model's output.
    """
    if not content.lstrip().startswith("{"):
        raise ValueError(
            "Template must begin with a Jinja tag ('{%' or '{{'). Text before the "
            "first tag is emitted into every prompt — this usually means the model "
            "card's page was copied along with the template."
        )
    try:
        jinja2.Environment().parse(content)
    except jinja2.TemplateSyntaxError as exc:
        raise ValueError(f"Jinja syntax error on line {exc.lineno}: {exc.message}") from exc
    return content


def load_chat_template_meta() -> dict:
    try:
        data = json.loads(core.CHAT_TEMPLATES_META_FILE.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_chat_template_meta(meta: dict):
    core.CHAT_TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    core.CHAT_TEMPLATES_META_FILE.write_text(json.dumps(meta, indent=2, sort_keys=True))


def chat_template_path(template_id: str) -> Path:
    return core.CHAT_TEMPLATES_DIR / f"{validate_chat_template_id(template_id)}.jinja"


def list_chat_templates() -> list[dict]:
    meta = load_chat_template_meta()
    templates = [{
        "id": "",
        "name": "Model default",
        "description": "Use the chat template embedded in the GGUF/model metadata.",
        "builtin": True,
        "updated_at": 0,
    }]
    if core.CHAT_TEMPLATES_DIR.is_dir():
        for path in sorted(core.CHAT_TEMPLATES_DIR.glob("*.jinja")):
            template_id = path.stem
            item = meta.get(template_id, {}) if isinstance(meta.get(template_id), dict) else {}
            templates.append({
                "id": template_id,
                "name": item.get("name") or template_id,
                "description": item.get("description", ""),
                "builtin": False,
                "updated_at": item.get("updated_at", int(path.stat().st_mtime)),
            })
    return templates


def list_gguf_files() -> list:
    """Return all .gguf files in the models directory."""
    files = []
    if core.MODELS_DIR.is_dir():
        for f in sorted(core.MODELS_DIR.rglob("*.gguf")):
            if f.name.startswith('.'):
                continue  # skip macOS resource forks and hidden files
            if LORAS_SUBDIR in f.relative_to(core.MODELS_DIR).parts:
                continue  # adapters have their own picker; they are not models
            files.append({
                "path": str(f),
                "name": f.name,
                "size_gb": round(f.stat().st_size / (1024**3), 2),
                "relative": str(f.relative_to(core.MODELS_DIR)),
                "is_mmproj": is_mmproj_gguf(f.name, f.stat().st_size),
                "kind": "gguf",
            })
    return files


#: Where a downloaded MTPLX pack goes, beside the other MLX models.
MTPLX_PACKS_SUBDIR = "mlx"


def list_mtplx_packs() -> list:
    """MTPLX packs under the models directory, as `list_gguf_files` entries.

    A pack is a directory -- whatever holds an `mtplx_runtime.json` -- so it is
    found by that file rather than by a suffix. A download still in progress is
    staged in `<name>.part/` and is not offered until it is renamed into place.
    """
    packs = []
    if not core.MODELS_DIR.is_dir():
        return packs
    for marker in sorted(core.MODELS_DIR.rglob(backends.spec.MTPLX_PACK_MARKER)):
        pack = marker.parent
        relative = pack.relative_to(core.MODELS_DIR)
        if any(part.startswith('.') or part.endswith('.part') for part in relative.parts):
            continue
        size = sum(f.stat().st_size for f in pack.rglob('*')
                   if f.is_file() and not f.relative_to(pack).parts[0].startswith('.'))
        packs.append({
            "path": str(pack),
            "name": pack.name,
            "size_gb": round(size / (1024**3), 2),
            "relative": str(relative),
            "is_mmproj": False,
            "kind": "mtplx",
        })
    return packs


#: Adapters live in their own directory so the base-model pickers do not have to
#: tell a 40 MB LoRA apart from a 17 GB model by its name.
LORAS_SUBDIR = "loras"


def loras_dir():
    return core.MODELS_DIR / LORAS_SUBDIR


def list_lora_adapters() -> list:
    """Return the LoRA adapter GGUFs under models/loras/.

    An adapter is identified by its metadata -- `general.type` is "adapter" and
    `adapter.type` is "lora" -- not by its filename. A converted adapter is an
    ordinary .gguf and nothing in its name distinguishes it from a base model,
    so a name heuristic would offer the operator a 17 GB model as an adapter and
    llama-server would refuse it after exec.

    A file that cannot be read is listed with `ok: False` and the reason rather
    than dropped: a failed conversion sitting in the directory is something the
    operator needs to see, not something to hide.
    """
    adapters = []
    directory = loras_dir()
    if not directory.is_dir():
        return adapters
    for path in sorted(directory.rglob("*.gguf")):
        if path.name.startswith("."):
            continue
        entry = {
            "path": str(path),
            "name": path.name,
            "relative": str(path.relative_to(directory)),
            "size_mb": round(path.stat().st_size / (1024**2), 1),
            "ok": False,
            "error": "",
            "base_arch": "",
            "alpha": None,
        }
        try:
            metadata = budget.read_gguf_metadata(path)
        except Exception as exc:
            entry["error"] = str(exc)
            adapters.append(entry)
            continue
        if str(metadata.get("general.type", "")).lower() != "adapter":
            entry["error"] = "not an adapter (general.type is %r)" % metadata.get("general.type", "")
        elif str(metadata.get("adapter.type", "")).lower() != "lora":
            entry["error"] = "unsupported adapter type %r" % metadata.get("adapter.type", "")
        else:
            entry["ok"] = True
            entry["base_arch"] = str(metadata.get("general.architecture", ""))
            entry["alpha"] = metadata.get("adapter.lora.alpha")
        adapters.append(entry)
    return adapters


def configured_lora_scales(env: dict, slot_name: str) -> dict:
    """Adapter path -> the scale configured for it, for one slot.

    Keyed by the path llama-server reports, which is the path the launcher
    passed it, so the two line up without re-deriving either.

    Needed because "preload unapplied" boots every adapter at 0: the live scale
    then says nothing about how strongly the operator wanted it applied, and
    switching one on from the UI has to raise it to something.
    """
    import backends  # local: web/backends imports platforms, which app.py owns

    slot = backends.SLOTS.get(slot_name)
    if slot is None:
        return {}
    prefixes = slot.prefixes
    paths = backends.spec.lookup(env, ("LORA_PATHS",), prefixes, empty_is_set=True) or ""
    scales = backends.spec.lookup(env, ("LORA_SCALES",), prefixes, empty_is_set=True) or ""
    return {resolved: float(scale)
            for resolved, scale, problem in backends.options.lora_pairs(
                paths, scales, str(core.STACK_DIR))
            if scale and not problem}


def is_mmproj_gguf(filename: str, size_bytes: int | None = None) -> bool:
    name = Path(filename).name.lower()
    path_text = str(filename).lower()
    if any(token in path_text for token in ("mmproj", "mm_project", "projector")):
        return True
    if ("clip" in name or "vision" in name) and size_bytes and size_bytes < 2 * 1024**3:
        return True
    return False


def slugify_repo_ref(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-") or "model"


def validate_transcription_engine_id(engine_id: str) -> dict:
    engine = TRANSCRIPTION_ENGINE_BY_ID.get((engine_id or "").strip())
    if not engine:
        raise ValueError("Unknown transcription engine")
    return engine


def transcription_engine_models_dir(engine_id: str) -> Path:
    """Where an engine's downloaded weights live.

    Falls back to a `legacy_ids` directory when the canonical one does not exist
    yet, so renaming an engine does not orphan models already on disk.
    """
    engine = validate_transcription_engine_id(engine_id)
    canonical = core.TRANSCRIPTION_MODELS_DIR / engine_id
    if not canonical.is_dir():
        for legacy_id in engine.get("legacy_ids", []):
            legacy = core.TRANSCRIPTION_MODELS_DIR / legacy_id
            if legacy.is_dir():
                return legacy
    return canonical


def transcription_model_storage_dir(engine_id: str, repo_ref: dict) -> Path:
    repo_slug = slugify_repo_ref(repo_ref["repo_id"].replace("/", "--"))
    revision = repo_ref.get("revision") or "main"
    base_dir = transcription_engine_models_dir(engine_id)
    if revision == "main":
        return base_dir / repo_slug
    return base_dir / f"{repo_slug}--{slugify_repo_ref(revision)}"


def transcription_model_dir_info(path: Path) -> dict | None:
    """Identify which runtime can load the weights in `path`, if any.

    Order matters. A CTranslate2 export and a stock transformers checkpoint both
    carry `config.json`, `tokenizer.json` and `preprocessor_config.json`, so
    those markers cannot tell them apart — only CTranslate2 writes `model.bin`,
    and only transformers writes safetensors/`pytorch_model.bin`. Testing the
    ambiguous markers first is what would file every downloaded HF model as
    faster-whisper and then fail to load it.
    """
    if not path.is_dir():
        return None
    nemo_files = sorted(path.glob("*.nemo"))
    if nemo_files:
        return {
            "runtime": "nemo",
            "format": "nemo",
            "supported_local": True,
            "primary_file": nemo_files[0].name,
        }
    if (path / "model.bin").exists():
        return {
            "runtime": "faster-whisper",
            "format": "ctranslate2",
            "supported_local": True,
        }
    hf_weight_markers = ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json")
    if (path / "config.json").exists() and (
        any((path / name).exists() for name in hf_weight_markers) or any(path.glob("*.safetensors"))
    ):
        return {
            "runtime": "hf",
            "format": "transformers",
            "supported_local": True,
        }
    return None


def format_transcription_model_value(kind: str, value: str) -> str:
    return f"{kind}:{value}"


def parse_transcription_model_value(value: str) -> dict[str, str]:
    raw = (value or "").strip()
    if not raw:
        return {"kind": "", "value": ""}
    if raw.startswith("preset:"):
        return {"kind": "preset", "value": raw.split(":", 1)[1]}
    if raw.startswith("local:"):
        return {"kind": "local", "value": raw.split(":", 1)[1]}
    return {"kind": "legacy", "value": raw}


def list_transcription_models(engine_id: str) -> list[dict]:
    engine = validate_transcription_engine_id(engine_id)
    items = []
    seen_values = set()
    base_dir = transcription_engine_models_dir(engine_id)

    # Per-engine rather than one global list: NeMo names full repo ids and the
    # router names a served alias, so Whisper's size names are meaningless there.
    for preset in engine.get("presets", []):
        value = format_transcription_model_value("preset", preset)
        items.append({
            "value": value,
            "label": preset,
            "kind": "preset",
            "path": "",
            "relative": "",
            "source": f"{engine['label']} preset",
        })
        seen_values.add(value)

    if base_dir.is_dir():
        for path in sorted(base_dir.rglob("*")):
            info = transcription_model_dir_info(path)
            if not info:
                continue
            value = format_transcription_model_value("local", str(path))
            if value in seen_values:
                continue
            seen_values.add(value)
            try:
                relative = str(path.relative_to(base_dir))
            except ValueError:
                relative = path.name
            items.append({
                "value": value,
                "label": relative,
                "kind": "local",
                "path": str(path),
                "relative": relative,
                "format": info["format"],
                "runtime": info["runtime"],
                "supported_local": info["supported_local"],
                "primary_file": info.get("primary_file", ""),
                "source": f"{engine['label']} local folder",
            })
    return items


def transcript_engine_capabilities(env: dict | None = None) -> dict[str, dict[str, bool]]:
    env = env or config_env.read_env()
    result = {}
    for engine in TRANSCRIPTION_ENGINES:
        prefix = engine["env_prefix"]
        backend_type = (env.get(f"{prefix}_BACKEND_TYPE", "upstream") or "upstream").strip().lower()
        local_info = None
        local_value = (env.get(f"{prefix}_LOCAL_MODEL", "") or "").strip()
        if local_value.startswith("local:"):
            local_path = Path(local_value.split(":", 1)[1])
            local_info = transcription_model_dir_info(local_path) if local_path.is_dir() else (
                {"runtime": "nemo", "format": "nemo", "supported_local": True, "primary_file": local_path.name}
                if local_path.is_file() and local_path.suffix.lower() == ".nemo" else None
            )
        runtime = engine.get("runtime", "")
        result[engine["id"]] = {
            "id": engine["id"],
            "label": engine["label"],
            "runtime": runtime,
            "install_extra": engine.get("install_extra", ""),
            "backend_type": backend_type,
            "supports_streaming": (
                env.get(f"{prefix}_SUPPORTS_STREAMING", "").strip().lower() in {"1", "true", "yes", "on"}
                # Keyed on the runtime, not the engine id: adding a second NeMo
                # engine should inherit this rather than need the test extended.
                or (runtime == "nemo" and backend_type == "local"
                    and bool(local_info and local_info.get("runtime") == "nemo"))
            ),
            "supports_speaker_detection": (env.get(f"{prefix}_SUPPORTS_SPEAKER_DETECTION", "off").strip().lower() in {"1", "true", "yes", "on"}),
            # The router forwards to llama.cpp, whose transcription endpoint
            # returns prose with no timeline at all — see docs/transcription.md.
            "supports_word_timestamps": runtime in {"faster-whisper", "nemo", "hf"},
            "supports_segments": runtime != "router",
            "supports_translate": runtime in {"faster-whisper", "hf"} or engine["id"] == "canary-qwen",
            "supports_diarization": False,
        }
    return result


def normalize_custom_arg_entries(values) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized = []
    for value in values:
        text = str(value).strip()
        if text:
            normalized.append(text)
    return normalized


def validate_custom_arg_entries(values: list[str]):
    for value in values:
        shlex.split(value)


def infer_model_arg_family(*texts) -> str:
    merged = " ".join(str(text or "") for text in texts).lower()
    normalized = re.sub(r"[^a-z0-9]+", "", merged)
    if re.search(r"qwen[\s._-]*3[\s._-]*8", merged) or "qwen38" in normalized:
        return "qwen3.8"
    if re.search(r"qwen[\s._-]*3[\s._-]*6", merged) or "qwen36" in normalized:
        return "qwen3.6"
    if re.search(r"gemma[\s._-]*4\b", merged) or "gemma4" in normalized:
        return "gemma4"
    return ""


def format_model_arg_family_label(family: str) -> str:
    if family == "qwen3.8":
        return "Qwen 3.8"
    if family == "qwen3.6":
        return "Qwen 3.6"
    if family == "gemma4":
        return "Gemma 4"
    return family or "Custom"


def load_custom_model_arg_presets() -> dict[str, list[str]]:
    if core.CUSTOM_MODEL_ARG_PRESETS_FILE.exists():
        try:
            data = json.loads(core.CUSTOM_MODEL_ARG_PRESETS_FILE.read_text())
            if isinstance(data, dict):
                return {
                    str(key): normalize_custom_arg_entries(value)
                    for key, value in data.items()
                }
        except Exception:
            return {}
    return {}


def save_custom_model_arg_presets(presets: dict[str, list[str]]):
    payload = {
        str(key): normalize_custom_arg_entries(value)
        for key, value in presets.items()
    }
    core.CUSTOM_MODEL_ARG_PRESETS_FILE.write_text(json.dumps(payload, indent=2))


def resolve_custom_args_for_family(family: str) -> tuple[list[str], str]:
    if not family:
        return [], "none"
    presets = load_custom_model_arg_presets()
    if family in presets:
        return presets[family], "family"
    if family in BUILTIN_CUSTOM_MODEL_ARG_PRESETS:
        return BUILTIN_CUSTOM_MODEL_ARG_PRESETS[family], "builtin"
    return [], "none"


def resolve_custom_args_for_model(model: dict) -> tuple[list[str], str, str]:
    family = model.get("arg_family", "")
    if not family:
        family = infer_model_arg_family(
            model.get("display_name", ""),
            model.get("model_name", ""),
            model.get("model_path", ""),
        )
    family_args, source = resolve_custom_args_for_family(family)
    if family_args:
        return family_args, family, source
    model_args = normalize_custom_arg_entries(model.get("custom_args", []))
    if model_args:
        return model_args, family, "model"
    return [], family, "none"


def normalize_custom_model(model: dict) -> dict:
    item = dict(model)
    item["custom_args"] = normalize_custom_arg_entries(item.get("custom_args", []))
    item["arg_family"] = item.get("arg_family") or infer_model_arg_family(
        item.get("display_name", ""),
        item.get("model_name", ""),
        item.get("model_path", ""),
    )
    resolved_args, family, source = resolve_custom_args_for_model(item)
    item["arg_family"] = family
    item["arg_family_label"] = format_model_arg_family_label(family) if family else ""
    item["resolved_custom_args"] = resolved_args
    item["custom_arg_source"] = source
    return item


def display_name_from_model_path(model_path: str) -> str:
    name = Path(model_path or "").name
    if name.lower().endswith(".gguf"):
        name = name[:-5]
    return name or "Custom Model"


def model_name_from_display_name(display_name: str) -> str:
    slug = re.sub(r"[^a-z0-9._-]+", "-", (display_name or "").strip().lower())
    slug = re.sub(r"-+", "-", slug).strip("-._")
    return slug or "custom-model"


def load_custom_models() -> list:
    """Load custom model definitions from JSON file."""
    if core.CUSTOM_MODELS_FILE.exists():
        try:
            data = json.loads(core.CUSTOM_MODELS_FILE.read_text())
            if isinstance(data, list):
                return [normalize_custom_model(item) for item in data if isinstance(item, dict)]
        except Exception:
            return []
    return []


def save_custom_models_file(models: list):
    """Write custom model definitions to JSON file."""
    payload = []
    for model in models:
        item = dict(model)
        item["custom_args"] = normalize_custom_arg_entries(item.get("custom_args", []))
        item["arg_family"] = item.get("arg_family") or infer_model_arg_family(
            item.get("display_name", ""),
            item.get("model_name", ""),
            item.get("model_path", ""),
        )
        item.pop("resolved_custom_args", None)
        item.pop("custom_arg_source", None)
        item.pop("arg_family_label", None)
        payload.append(item)
    core.CUSTOM_MODELS_FILE.write_text(json.dumps(payload, indent=2))


def huggingface_headers() -> dict[str, str]:
    headers = {"User-Agent": "llm-stack-manager/1.0"}
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def parse_huggingface_repo_ref(value: str) -> dict:
    raw = (value or "").strip()
    if not raw:
        raise ValueError("Hugging Face repo URL or repo id is required")

    if re.fullmatch(r"[\w.-]+/[\w.-]+", raw):
        return {
            "repo_id": raw,
            "revision": "main",
            "repo_url": f"https://huggingface.co/{raw}",
        }

    parsed = urlparse(raw)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Hugging Face URL must start with http:// or https://")
    host = (parsed.netloc or "").lower()
    if host not in HF_ALLOWED_HOSTS:
        raise ValueError("URL must point to huggingface.co")

    parts = [unquote(part) for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        raise ValueError("Could not determine repo id from Hugging Face URL")

    repo_id = f"{parts[0]}/{parts[1]}"
    revision = "main"
    if len(parts) >= 4 and parts[2] in {"tree", "blob", "resolve"}:
        revision = parts[3] or "main"

    return {
        "repo_id": repo_id,
        "revision": revision,
        "repo_url": f"https://huggingface.co/{repo_id}",
    }


def list_huggingface_repo_files(repo_ref: dict) -> list[dict]:
    repo_id = repo_ref["repo_id"]
    revision = repo_ref.get("revision") or "main"
    api_url = f"https://huggingface.co/api/models/{quote(repo_id, safe='/')}"
    if revision and revision != "main":
        api_url += f"/revision/{quote(revision, safe='')}"
    req = urlrequest.Request(api_url, headers=huggingface_headers())
    with urlrequest.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    siblings = payload.get("siblings") or []
    files = []
    for item in siblings:
        filename = item.get("rfilename") or item.get("path") or item.get("name")
        if not filename:
            continue
        files.append({
            "path": filename,
            "name": Path(filename).name,
            "size": item.get("size") or (item.get("lfs") or {}).get("size"),
            "sha256": (item.get("lfs") or {}).get("sha256") or "",
        })
    return files


def huggingface_repo_metadata(repo_ref: dict) -> dict:
    repo_id = repo_ref["repo_id"]
    api_url = f"https://huggingface.co/api/models/{quote(repo_id, safe='/')}"
    req = urlrequest.Request(api_url, headers=huggingface_headers())
    with urlrequest.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    card = payload.get("cardData") if isinstance(payload.get("cardData"), dict) else {}
    return {
        "license": card.get("license") or payload.get("license") or "not declared",
        "gated": bool(payload.get("gated")),
        "private": bool(payload.get("private")),
        "revision_sha": payload.get("sha") or "",
    }


def score_mmproj_match(model_name: str, mmproj_name: str) -> tuple[int, int]:
    def tokens(text: str) -> set[str]:
        parts = re.split(r"[^a-z0-9]+", text.lower())
        ignored = {
            "gguf", "mmproj", "model", "q2", "q3", "q4", "q5", "q6", "q8",
            "k", "m", "s", "xs", "l", "xl", "xxl", "f16", "bf16",
        }
        return {part for part in parts if part and part not in ignored}

    model_tokens = tokens(Path(model_name).stem)
    mmproj_tokens = tokens(Path(mmproj_name).stem)
    overlap = len(model_tokens & mmproj_tokens)
    return overlap, -len(mmproj_name)


def choose_matching_mmproj_file(model_file: str, repo_files: list[dict]) -> str:
    mmproj_candidates = [
        item["path"]
        for item in repo_files
        if item["name"].lower().endswith(".gguf") and is_mmproj_gguf(item["path"], item.get("size"))
    ]
    if not mmproj_candidates:
        return ""
    if len(mmproj_candidates) == 1:
        return mmproj_candidates[0]
    return max(mmproj_candidates, key=lambda candidate: score_mmproj_match(model_file, candidate))


def derive_mmproj_target_name(model_filename: str) -> str:
    model_path = Path(model_filename)
    return f"{model_path.stem}.mmproj{model_path.suffix}"


def build_huggingface_download_url(repo_ref: dict, repo_file: str) -> str:
    repo_id = quote(repo_ref["repo_id"], safe="/")
    revision = quote(repo_ref.get("revision") or "main", safe="")
    file_path = quote(repo_file, safe="/")
    return f"https://huggingface.co/{repo_id}/resolve/{revision}/{file_path}?download=true"


def update_hf_download_job(job_id: str, **changes):
    with HF_DOWNLOAD_JOBS_LOCK:
        job = HF_DOWNLOAD_JOBS.get(job_id)
        if not job:
            return
        job.update(changes)
        job["updated_at"] = int(time.time())


def stream_download_to_path(url: str, dest_path: Path, job_id: str | None = None, label: str = "", validate_gguf_file: bool = False, expected_sha256: str = ""):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_name(dest_path.name + ".part")
    free_bytes = shutil.disk_usage(dest_path.parent).free
    headers = huggingface_headers()
    downloaded = tmp_path.stat().st_size if tmp_path.exists() else 0
    if downloaded:
        headers["Range"] = f"bytes={downloaded}-"
    req = urlrequest.Request(url, headers=headers)
    total = None
    with urlrequest.urlopen(req, timeout=300) as resp:
        partial = getattr(resp, "status", 200) == 206 and downloaded > 0
        if not partial:
            downloaded = 0
        total_header = resp.headers.get("Content-Length")
        try:
            remaining = int(total_header) if total_header else None
            total = (downloaded + remaining) if remaining is not None else None
        except ValueError:
            total = None
        if total and total - downloaded > free_bytes:
            raise RuntimeError(f"Not enough disk space for {dest_path.name}: need {total - downloaded} bytes")
        with tmp_path.open("ab" if partial else "wb") as fh:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
                downloaded += len(chunk)
                if job_id:
                    progress = round((downloaded / total) * 100, 1) if total else None
                    update_hf_download_job(
                        job_id,
                        current_file=label or dest_path.name,
                        current_bytes=downloaded,
                        total_bytes=total,
                        progress=progress,
                    )
            fh.flush()
            os.fsync(fh.fileno())
    validation = {"ok": True, "sha256": ""}
    if validate_gguf_file:
        validation = setup_engine.validate_gguf(tmp_path)
        if not validation["ok"]:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(f"Rejected download {dest_path.name}: {validation['error']}")
        if expected_sha256 and validation.get("sha256", "").lower() != expected_sha256.lower():
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(f"Rejected download {dest_path.name}: published SHA-256 does not match")
    tmp_path.replace(dest_path)
    return validation


def download_huggingface_model_bundle(repo_ref: dict, model_file: str, mmproj_file: str = "", job_id: str | None = None, model_sha256: str = "", mmproj_sha256: str = "") -> dict:
    model_target = core.MODELS_DIR / Path(model_file).name
    model_validation = stream_download_to_path(
        build_huggingface_download_url(repo_ref, model_file),
        model_target,
        job_id=job_id,
        label=Path(model_file).name,
        validate_gguf_file=True,
        expected_sha256=model_sha256,
    )

    mmproj_target = None
    if mmproj_file:
        mmproj_target = core.MODELS_DIR / derive_mmproj_target_name(Path(model_file).name)
        mmproj_validation = stream_download_to_path(
            build_huggingface_download_url(repo_ref, mmproj_file),
            mmproj_target,
            job_id=job_id,
            label=Path(mmproj_file).name,
            validate_gguf_file=True,
            expected_sha256=mmproj_sha256,
        )

    return {
        "model_path": str(model_target),
        "mmproj_path": str(mmproj_target) if mmproj_target else "",
        "model_name": Path(model_file).name,
        "mmproj_name": mmproj_target.name if mmproj_target else "",
        "model_sha256": model_validation.get("sha256", ""),
        "mmproj_sha256": mmproj_validation.get("sha256", "") if mmproj_target else "",
    }


def run_hf_download_job(job_id: str, repo_ref: dict, model_file: str, mmproj_file: str, model_sha256: str = "", mmproj_sha256: str = ""):
    try:
        update_hf_download_job(job_id, status="running", stage="Downloading model bundle")
        result = download_huggingface_model_bundle(repo_ref, model_file, mmproj_file, job_id=job_id, model_sha256=model_sha256, mmproj_sha256=mmproj_sha256)
        update_hf_download_job(
            job_id,
            ok=True,
            status="done",
            stage="Completed",
            progress=100.0,
            current_file="",
            result=result,
        )
    except Exception as exc:
        update_hf_download_job(
            job_id,
            ok=False,
            status="error",
            error=str(exc),
            stage="Failed",
        )


def mtplx_pack_files(files: list[dict]) -> list[dict]:
    """The files of an MTPLX pack repo, or [] when the repo is not one."""
    if not any(item.get("path") == backends.spec.MTPLX_PACK_MARKER for item in files):
        return []
    return [item for item in files
            if item.get("path") and not item["path"].startswith(".") and "/." not in item["path"]]


def download_mtplx_pack(repo_ref: dict, job_id: str | None = None) -> dict:
    """Every file of an MTPLX pack repo, into models/mlx/<repo name>/.

    Staged in `<name>.part/` and renamed into place at the end: the pickers
    offer any directory holding `mtplx_runtime.json`, and that file arrives
    long before the weights. A failed download resumes where it stopped -- a
    file already complete is skipped, and a partial one keeps its own `.part`.

    Pinned to the commit the repo was at when the download began, so a pack
    cannot change revision halfway through a 20 GB transfer.
    """
    if (repo_ref.get("revision") or "main") == "main":
        sha = huggingface_repo_metadata(repo_ref).get("revision_sha")
        if sha:
            repo_ref = dict(repo_ref, revision=sha)
    files = mtplx_pack_files(list_huggingface_repo_files(repo_ref))
    if not files:
        raise ValueError(f"{repo_ref['repo_id']} is not an MTPLX pack "
                         f"(it has no {backends.spec.MTPLX_PACK_MARKER})")

    name = repo_ref["repo_id"].split("/")[-1]
    target = core.MODELS_DIR / MTPLX_PACKS_SUBDIR / name
    result = {"model_path": str(target), "model_name": name, "mmproj_path": "",
              "mmproj_name": "", "kind": "mtplx", "repo_id": repo_ref["repo_id"],
              "revision": repo_ref["revision"], "file_count": len(files)}
    if backends.spec.is_mtplx_pack(str(target)):
        return dict(result, already_present=True)

    staging = target.with_name(name + ".part")
    remaining = sum((item.get("size") or 0) for item in files
                    if not (staging / item["path"]).is_file())
    staging.mkdir(parents=True, exist_ok=True)
    if remaining > shutil.disk_usage(staging).free:
        raise RuntimeError(f"Not enough disk space for {name}: need {remaining} bytes")
    for index, item in enumerate(files, start=1):
        dest = staging / item["path"]
        if dest.is_file() and item.get("size") and dest.stat().st_size == item["size"]:
            continue
        update_hf_download_job(job_id, stage=f"Downloading file {index}/{len(files)}",
                               current_file=item["path"])
        stream_download_to_path(build_huggingface_download_url(repo_ref, item["path"]),
                                dest, job_id=job_id, label=item["path"])
    staging.rename(target)
    return result


def run_hf_mtplx_download_job(job_id: str, repo_ref: dict):
    try:
        update_hf_download_job(job_id, status="running", stage="Downloading MTPLX pack")
        result = download_mtplx_pack(repo_ref, job_id=job_id)
        update_hf_download_job(
            job_id,
            ok=True,
            status="done",
            stage="Already downloaded" if result.get("already_present") else "Completed",
            progress=100.0,
            current_file="",
            result=result,
        )
    except Exception as exc:
        update_hf_download_job(
            job_id,
            ok=False,
            status="error",
            error=str(exc),
            stage="Failed",
        )


def list_transcription_repo_download_files(repo_ref: dict) -> list[dict]:
    files = []
    for item in list_huggingface_repo_files(repo_ref):
        path = item.get("path") or ""
        name = item.get("name") or ""
        if not path or not name or name.startswith("."):
            continue
        if path.startswith(".") or "/." in path:
            continue
        if name in {".gitattributes", "README.md"}:
            continue
        files.append(item)
    if not files:
        raise ValueError("No downloadable transcription model files were found in this repo")
    return files


def download_huggingface_transcription_model(engine_id: str, repo_ref: dict, job_id: str | None = None) -> dict:
    engine = validate_transcription_engine_id(engine_id)
    target_dir = transcription_model_storage_dir(engine_id, repo_ref)
    repo_files = list_transcription_repo_download_files(repo_ref)
    total_files = len(repo_files)
    for index, item in enumerate(repo_files, start=1):
        relative_path = item["path"]
        update_hf_download_job(
            job_id,
            stage=f"Downloading file {index}/{total_files}",
            current_file=relative_path,
        )
        stream_download_to_path(
            build_huggingface_download_url(repo_ref, relative_path),
            target_dir / relative_path,
            job_id=job_id,
            label=relative_path,
        )
    return {
        "engine_id": engine_id,
        "engine_label": engine["label"],
        "model_dir": str(target_dir),
        "model_value": format_transcription_model_value("local", str(target_dir)),
        "model_label": str(target_dir.relative_to(transcription_engine_models_dir(engine_id))) if target_dir.exists() else target_dir.name,
        "file_count": total_files,
        "repo_id": repo_ref["repo_id"],
        "revision": repo_ref["revision"],
    }


def run_hf_transcription_download_job(job_id: str, engine_id: str, repo_ref: dict):
    try:
        update_hf_download_job(job_id, status="running", stage="Downloading transcription model")
        result = download_huggingface_transcription_model(engine_id, repo_ref, job_id=job_id)
        update_hf_download_job(
            job_id,
            ok=True,
            status="done",
            stage="Completed",
            progress=100.0,
            current_file="",
            result=result,
        )
    except Exception as exc:
        update_hf_download_job(
            job_id,
            ok=False,
            status="error",
            error=str(exc),
            stage="Failed",
        )

