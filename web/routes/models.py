#!/usr/bin/env python3
"""
HTTP routes for the model catalogue: listing, downloading, and chat templates.

Wrappers over `models.py`. The two that do more than translate are the download
endpoints, which start a background thread and hand back a job id — a 20 GB
fetch cannot be a request — and the chat-template writes, which validate the
template id before it becomes a filename.

Registered without a `url_prefix`, so the rules are exactly what they were in
`app.py`.
"""

from __future__ import annotations

import re
import threading
import time
import uuid
from urllib import error as urlerror

from flask import Blueprint, jsonify, request

import config_env
import core
import models

bp = Blueprint("models", __name__)


@bp.route('/api/gguf-files')
def api_gguf_files():
    """List all .gguf files in the models directory."""
    return jsonify(models.list_gguf_files())


@bp.route('/api/transcription-models/<engine_id>')
def api_transcription_models(engine_id):
    try:
        engine = models.validate_transcription_engine_id(engine_id)
    except ValueError as exc:
        return jsonify(ok=False, error=str(exc)), 404
    return jsonify({
        "ok": True,
        "engine_id": engine_id,
        "engine_label": engine["label"],
        "directory": str(models.transcription_engine_models_dir(engine_id)),
        "models": models.list_transcription_models(engine_id),
    })


@bp.route('/api/transcription-capabilities')
def api_transcription_capabilities():
    return jsonify({
        "ok": True,
        "engines": models.transcript_engine_capabilities(),
    })


@bp.route('/api/huggingface/repo-files', methods=['POST'])
def api_huggingface_repo_files():
    data = request.json or {}
    try:
        repo_ref = models.parse_huggingface_repo_ref(data.get('repo_url', ''))
        files = models.list_huggingface_repo_files(repo_ref)
    except ValueError as exc:
        return jsonify(ok=False, error=str(exc)), 400
    except urlerror.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore").strip()
        return jsonify(ok=False, error=detail or f"Hugging Face request failed: HTTP {exc.code}"), 502
    except Exception as exc:
        return jsonify(ok=False, error=str(exc)), 500

    gguf_files = [
        file for file in files
        if file["name"].lower().endswith(".gguf")
    ]
    model_files = [
        file for file in gguf_files
        if not models.is_mmproj_gguf(file["path"], file.get("size"))
    ]
    mmproj_files = [
        file for file in gguf_files
        if models.is_mmproj_gguf(file["path"], file.get("size"))
    ]
    for file in model_files:
        matched_mmproj = models.choose_matching_mmproj_file(file["path"], files)
        file["matched_mmproj"] = matched_mmproj
        file["renamed_mmproj"] = models.derive_mmproj_target_name(file["name"]) if matched_mmproj else ""

    return jsonify({
        "ok": True,
        "repo_id": repo_ref["repo_id"],
        "revision": repo_ref["revision"],
        "repo_url": repo_ref["repo_url"],
        "model_files": model_files,
        "mmproj_files": mmproj_files,
    })


@bp.route('/api/huggingface/downloads', methods=['GET'])
def api_huggingface_downloads_list():
    with models.HF_DOWNLOAD_JOBS_LOCK:
        jobs = sorted(
            (dict(job) for job in models.HF_DOWNLOAD_JOBS.values()),
            key=lambda item: item.get('created_at', 0),
        )
    return jsonify(ok=True, jobs=jobs)


@bp.route('/api/huggingface/downloads', methods=['POST'])
def api_huggingface_download_create():
    data = request.json or {}
    model_file = (data.get('model_file') or '').strip()
    if not model_file:
        return jsonify(ok=False, error='model_file is required'), 400

    try:
        repo_ref = models.parse_huggingface_repo_ref(data.get('repo_url', ''))
        requested_revision = str(data.get('revision') or '').strip()
        if requested_revision:
            if not re.fullmatch(r'[A-Za-z0-9._-]{1,128}', requested_revision):
                return jsonify(ok=False, error='Invalid Hugging Face revision'), 400
            repo_ref['revision'] = requested_revision
    except ValueError as exc:
        return jsonify(ok=False, error=str(exc)), 400

    job_id = uuid.uuid4().hex[:10]
    job = {
        "id": job_id,
        "ok": False,
        "status": "queued",
        "stage": "Queued",
        "progress": 0.0,
        "repo_id": repo_ref["repo_id"],
        "revision": repo_ref["revision"],
        "model_file": model_file,
        "mmproj_file": (data.get('mmproj_file') or '').strip(),
        "expected_model_sha256": (data.get('model_sha256') or '').strip(),
        "expected_mmproj_sha256": (data.get('mmproj_sha256') or '').strip(),
        "current_file": "",
        "current_bytes": 0,
        "total_bytes": None,
        "created_at": int(time.time()),
        "updated_at": int(time.time()),
        "result": {},
    }
    with models.HF_DOWNLOAD_JOBS_LOCK:
        models.HF_DOWNLOAD_JOBS[job_id] = job

    thread = threading.Thread(
        target=models.run_hf_download_job,
        args=(job_id, repo_ref, job["model_file"], job["mmproj_file"], job["expected_model_sha256"], job["expected_mmproj_sha256"]),
        daemon=True,
    )
    thread.start()
    return jsonify(ok=True, job=job)


@bp.route('/api/huggingface/downloads/<job_id>', methods=['GET'])
def api_huggingface_download_status(job_id):
    with models.HF_DOWNLOAD_JOBS_LOCK:
        job = models.HF_DOWNLOAD_JOBS.get(job_id)
        if not job:
            return jsonify(ok=False, error='Download job not found'), 404
        return jsonify(ok=True, job=job)


@bp.route('/api/huggingface/transcription-repo-files', methods=['POST'])
def api_huggingface_transcription_repo_files():
    data = request.json or {}
    try:
        engine = models.validate_transcription_engine_id(data.get('engine_id', ''))
        repo_ref = models.parse_huggingface_repo_ref(data.get('repo_url', ''))
        files = models.list_transcription_repo_download_files(repo_ref)
    except ValueError as exc:
        return jsonify(ok=False, error=str(exc)), 400
    except urlerror.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore").strip()
        return jsonify(ok=False, error=detail or f"Hugging Face request failed: HTTP {exc.code}"), 502
    except Exception as exc:
        return jsonify(ok=False, error=str(exc)), 500

    return jsonify({
        "ok": True,
        "engine_id": engine["id"],
        "engine_label": engine["label"],
        "repo_id": repo_ref["repo_id"],
        "revision": repo_ref["revision"],
        "repo_url": repo_ref["repo_url"],
        "file_count": len(files),
        "target_dir": str(models.transcription_model_storage_dir(engine["id"], repo_ref)),
        "sample_files": [item["path"] for item in files[:8]],
    })


@bp.route('/api/huggingface/transcription-downloads', methods=['POST'])
def api_huggingface_transcription_download_create():
    data = request.json or {}
    try:
        engine = models.validate_transcription_engine_id(data.get('engine_id', ''))
        repo_ref = models.parse_huggingface_repo_ref(data.get('repo_url', ''))
    except ValueError as exc:
        return jsonify(ok=False, error=str(exc)), 400

    job_id = uuid.uuid4().hex[:10]
    job = {
        "id": job_id,
        "ok": False,
        "status": "queued",
        "stage": "Queued",
        "progress": 0.0,
        "repo_id": repo_ref["repo_id"],
        "revision": repo_ref["revision"],
        "current_file": "",
        "current_bytes": 0,
        "total_bytes": None,
        "created_at": int(time.time()),
        "updated_at": int(time.time()),
        "result": {},
        "kind": "transcription-model",
        "engine_id": engine["id"],
    }
    with models.HF_DOWNLOAD_JOBS_LOCK:
        models.HF_DOWNLOAD_JOBS[job_id] = job

    thread = threading.Thread(
        target=models.run_hf_transcription_download_job,
        args=(job_id, engine["id"], repo_ref),
        daemon=True,
    )
    thread.start()
    return jsonify(ok=True, job=job)


@bp.route('/api/custom-models', methods=['GET'])
def api_custom_models_list():
    return jsonify(models.load_custom_models())


@bp.route('/api/custom-models', methods=['POST'])
def api_custom_models_add():
    data = request.json or {}
    model_path = (data.get('model_path') or '').strip()
    if not model_path:
        return jsonify(ok=False, error='model_path is required'), 400
    display_name = (data.get('display_name') or '').strip() or models.display_name_from_model_path(model_path)
    model_name = (data.get('model_name') or '').strip() or models.model_name_from_display_name(display_name)
    custom_args_supplied = 'custom_args' in data
    custom_args = models.normalize_custom_arg_entries(data.get('custom_args', []))
    try:
        models.validate_custom_arg_entries(custom_args)
    except ValueError as exc:
        return jsonify(ok=False, error=f'invalid custom argument: {exc}'), 400
    family = models.infer_model_arg_family(
        display_name,
        model_name,
        model_path,
    )
    if not custom_args_supplied and family:
        custom_args, _ = models.resolve_custom_args_for_family(family)
    model = {
        'id': str(uuid.uuid4())[:8],
        'display_name': display_name,
        'model_name': model_name,
        'model_path': model_path,
        'mmproj_path': data.get('mmproj_path', ''),
        'ctx_size': str(data.get('ctx_size', '32768')),
        'custom_args': custom_args,
        'arg_family': family,
        'created': int(time.time()),
    }
    if family and custom_args_supplied:
        presets = models.load_custom_model_arg_presets()
        presets[family] = custom_args
        models.save_custom_model_arg_presets(presets)
    catalogue = models.load_custom_models()
    catalogue.append(model)
    models.save_custom_models_file(catalogue)
    return jsonify(ok=True, model=models.normalize_custom_model(model))


@bp.route('/api/custom-models/<model_id>', methods=['PUT'])
def api_custom_models_update(model_id):
    data = request.json or {}
    custom_args_supplied = 'custom_args' in (data or {})
    custom_args = models.normalize_custom_arg_entries((data or {}).get('custom_args', []))
    if custom_args_supplied:
        try:
            models.validate_custom_arg_entries(custom_args)
        except ValueError as exc:
            return jsonify(ok=False, error=f'invalid custom argument: {exc}'), 400
    catalogue = models.load_custom_models()
    for m in catalogue:
        if m['id'] == model_id:
            for k in ('display_name', 'model_name', 'model_path',
                       'mmproj_path', 'ctx_size'):
                if k in data:
                    m[k] = data[k]
            family = models.infer_model_arg_family(
                m.get('display_name', ''),
                m.get('model_name', ''),
                m.get('model_path', ''),
            )
            m['arg_family'] = family
            if custom_args_supplied:
                m['custom_args'] = custom_args
            elif family and not models.normalize_custom_arg_entries(m.get('custom_args', [])):
                family_args, _ = models.resolve_custom_args_for_family(family)
                if family_args:
                    m['custom_args'] = family_args
            if family and custom_args_supplied:
                presets = models.load_custom_model_arg_presets()
                presets[family] = models.normalize_custom_arg_entries(m.get('custom_args', []))
                models.save_custom_model_arg_presets(presets)
            models.save_custom_models_file(catalogue)
            return jsonify(ok=True, model=models.normalize_custom_model(m))
    return jsonify(ok=False, error='Model not found'), 404


@bp.route('/api/custom-model-arg-presets/match', methods=['POST'])
def api_custom_model_arg_preset_match():
    data = request.json or {}
    family = models.infer_model_arg_family(
        data.get('display_name', ''),
        data.get('model_name', ''),
        data.get('model_path', ''),
    )
    args, source = models.resolve_custom_args_for_family(family)
    return jsonify({
        'family': family,
        'family_label': models.format_model_arg_family_label(family) if family else '',
        'args': args,
        'source': source,
    })


@bp.route('/api/custom-models/<model_id>', methods=['DELETE'])
def api_custom_models_delete(model_id):
    catalogue = models.load_custom_models()
    catalogue = [m for m in catalogue if m['id'] != model_id]
    models.save_custom_models_file(catalogue)
    return jsonify(ok=True)

@bp.route('/api/chat-templates', methods=['GET'])
def api_chat_templates_list():
    return jsonify(models.list_chat_templates())


@bp.route('/api/chat-templates', methods=['POST'])
def api_chat_templates_create():
    data = request.get_json(silent=True) or {}
    name = (data.get('name') or '').strip()
    content = data.get('content')
    if not name:
        return jsonify(ok=False, error='Name is required'), 400
    if not isinstance(content, str) or not content.strip():
        return jsonify(ok=False, error='Template content is required'), 400
    template_id = models.chat_template_id_from_name(data.get('id') or name)
    existing_ids = {item['id'] for item in models.list_chat_templates() if item.get('id')}
    base_id = template_id
    suffix = 2
    while template_id in existing_ids:
        template_id = f'{base_id}-{suffix}'[:80]
        suffix += 1
    try:
        path = models.chat_template_path(template_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        meta = models.load_chat_template_meta()
        meta[template_id] = {
            'name': name,
            'description': (data.get('description') or '').strip(),
            'updated_at': int(time.time()),
        }
        models.save_chat_template_meta(meta)
    except Exception as exc:
        return jsonify(ok=False, error=f'Could not save chat template: {exc}'), 500
    return jsonify(ok=True, id=template_id)


@bp.route('/api/chat-templates/<template_id>', methods=['GET'])
def api_chat_templates_get(template_id):
    try:
        path = models.chat_template_path(template_id)
    except ValueError as exc:
        return jsonify(ok=False, error=str(exc)), 400
    if not path.exists():
        return jsonify(ok=False, error='Template not found'), 404
    item = models.load_chat_template_meta().get(template_id, {})
    return jsonify({
        'ok': True,
        'id': template_id,
        'name': item.get('name') or template_id,
        'description': item.get('description', ''),
        'content': path.read_text(),
        'updated_at': item.get('updated_at', int(path.stat().st_mtime)),
    })


@bp.route('/api/chat-templates/<template_id>', methods=['PUT'])
def api_chat_templates_update(template_id):
    data = request.get_json(silent=True) or {}
    try:
        path = models.chat_template_path(template_id)
    except ValueError as exc:
        return jsonify(ok=False, error=str(exc)), 400
    if not path.exists():
        return jsonify(ok=False, error='Template not found'), 404
    content = data.get('content')
    if not isinstance(content, str) or not content.strip():
        return jsonify(ok=False, error='Template content is required'), 400
    try:
        path.write_text(content)
        meta = models.load_chat_template_meta()
        current = meta.get(template_id, {}) if isinstance(meta.get(template_id), dict) else {}
        current.update({
            'name': (data.get('name') or current.get('name') or template_id).strip(),
            'description': (data.get('description') or '').strip(),
            'updated_at': int(time.time()),
        })
        meta[template_id] = current
        models.save_chat_template_meta(meta)
    except Exception as exc:
        return jsonify(ok=False, error=f'Could not update chat template: {exc}'), 500
    return jsonify(ok=True, id=template_id)


@bp.route('/api/chat-templates/<template_id>', methods=['DELETE'])
def api_chat_templates_delete(template_id):
    try:
        path = models.chat_template_path(template_id)
    except ValueError as exc:
        return jsonify(ok=False, error=str(exc)), 400
    if path.exists():
        path.unlink()
    meta = models.load_chat_template_meta()
    if template_id in meta:
        del meta[template_id]
        models.save_chat_template_meta(meta)
    return jsonify(ok=True)
