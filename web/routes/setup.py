#!/usr/bin/env python3
"""
HTTP routes for the first-run setup wizard.

The wizard is the one part of the manager that runs before there is a working
stack to manage: it picks components, resolves models — from disk or from
HuggingFace — writes the env file, generates systemd units and starts them.
`setup_engine` does the work; these routes are its HTTP surface, plus the job
polling the UI needs because a full setup takes long enough that no browser
would wait on one request.

Registered without a `url_prefix`, so the rules are exactly what they were in
`app.py`.
"""

from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path

from flask import Blueprint, jsonify, request

import config_env
import core
import models
import setup_engine

bp = Blueprint("setup", __name__)


@bp.route('/api/setup/preflight')
def api_setup_preflight():
    result = setup_engine.collect_preflight()
    state = setup_engine.load_state()
    state["preflight"] = result
    setup_engine.save_state(state)
    return jsonify(result)


@bp.route('/api/setup/selection', methods=['GET', 'PUT'])
def api_setup_selection():
    state = setup_engine.load_state()
    if request.method == 'GET':
        env = config_env.read_env()
        configured_models = [
            env.get('CHAT_PRIMARY_MODEL_PATH') or env.get('CHAT_DENSE_MODEL_PATH'),
            env.get('EMBEDDING_MODEL_PATH'), env.get('TASK_MODEL_PATH'), env.get('OCR_MODEL_PATH'),
        ]
        setup_required = not setup_engine.STATE_FILE.exists() and not any(Path(path).is_file() for path in configured_models if path)
        return jsonify(ok=True, selection=state['selection'], state_status=state.get('status', 'new'), setup_required=setup_required)
    data = request.get_json(silent=True) or {}
    components = setup_engine.resolve_components(data.get('components', state['selection'].get('components', [])))
    models = data.get('models', state['selection'].get('models', {}))
    if not isinstance(models, dict):
        return jsonify(ok=False, error='models must be an object'), 400
    new_selection = {
        'components': components,
        'models': models,
        'allow_vram_override': bool(data.get('allow_vram_override', False)),
    }
    if new_selection != state.get('selection'):
        state['completed_stages'] = [stage for stage in state.get('completed_stages', []) if stage == 'preflight']
        state['status'] = 'selection_changed'
    state['selection'] = new_selection
    setup_engine.save_state(state)
    return jsonify(ok=True, selection=state['selection'])


@bp.route('/api/setup/models/inspect', methods=['POST'])
def api_setup_model_inspect():
    data = request.get_json(silent=True) or {}
    component = str(data.get('component') or '').strip()
    if component not in setup_engine.MODEL_COMPONENTS:
        return jsonify(ok=False, error='Unknown model component'), 400
    local_path = str(data.get('path') or '').strip()
    if local_path:
        result = setup_engine.validate_gguf(Path(local_path))
        return jsonify(ok=result['ok'], component=component, validation=result), (200 if result['ok'] else 400)
    try:
        repo_ref = models.parse_huggingface_repo_ref(data.get('repo_url', ''))
        files = models.list_huggingface_repo_files(repo_ref)
        metadata = models.huggingface_repo_metadata(repo_ref)
    except ValueError as exc:
        return jsonify(ok=False, error=str(exc)), 400
    except Exception as exc:
        return jsonify(ok=False, error=str(exc)), 502
    model_files = []
    mmproj_files = []
    for item in files:
        if not item['name'].lower().endswith('.gguf'):
            continue
        target = mmproj_files if models.is_mmproj_gguf(item['path'], item.get('size')) else model_files
        target.append(item)
    return jsonify(ok=True, component=component, repo=repo_ref, metadata=metadata, model_files=model_files, mmproj_files=mmproj_files)


def _start_setup_job(retry_of=''):
    job = core.SETUP_RUNNER.create_job(retry_of)
    threading.Thread(target=core.SETUP_RUNNER.run, args=(job['id'],), daemon=True).start()
    return job


@bp.route('/api/setup/run', methods=['POST'])
def api_setup_run():
    if os.geteuid() != 0 and not core.ServiceManager.IS_MAC:
        return jsonify(ok=False, error='The manager must run as root to install system services'), 403
    job = _start_setup_job()
    return jsonify(ok=True, job=job), 202


@bp.route('/api/setup/jobs/<job_id>')
def api_setup_job(job_id):
    job = core.SETUP_RUNNER.job(job_id)
    if not job:
        return jsonify(ok=False, error='Setup job not found'), 404
    return jsonify(ok=True, job=job)


@bp.route('/api/setup/jobs/<job_id>/retry', methods=['POST'])
def api_setup_job_retry(job_id):
    prior = core.SETUP_RUNNER.job(job_id)
    if not prior:
        return jsonify(ok=False, error='Setup job not found'), 404
    if prior.get('status') not in {'failed', 'needs_attention', 'interrupted'}:
        return jsonify(ok=False, error='Only failed or interrupted jobs can be retried'), 409
    job = _start_setup_job(job_id)
    return jsonify(ok=True, job=job), 202


@bp.route('/api/setup/validation')
def api_setup_validation():
    result = setup_engine.validate_installation()
    state = setup_engine.load_state()
    state['last_validation'] = result
    setup_engine.save_state(state)
    return jsonify(result)


@bp.route('/api/setup/placement')
def api_setup_placement():
    state = setup_engine.load_state()
    preflight = state.get('preflight') or setup_engine.collect_preflight()
    components = setup_engine.resolve_components(state['selection'].get('components', []))
    models = {}
    for component, model in state['selection'].get('models', {}).items():
        if component not in components:
            continue
        item = dict(model)
        path = Path(str(item.get('path') or ''))
        if path.is_file():
            item['size'] = path.stat().st_size
        models[component] = item
    result = setup_engine.plan_gpu_placement(preflight.get('gpus', []), models, bool(state['selection'].get('allow_vram_override')))
    return jsonify(result), (200 if result.get('ok') else 400)


@bp.route('/api/setup/repair', methods=['POST'])
def api_setup_repair():
    if os.geteuid() != 0 and not core.ServiceManager.IS_MAC:
        return jsonify(ok=False, error='Repair requires a root-run manager'), 403
    action = str((request.get_json(silent=True) or {}).get('action') or '')
    commands = {
        'packages': ['bash', str(core.SCRIPTS_DIR / 'install-system-dependencies.sh'), '--full'],
        'dependencies': [str(core.SCRIPTS_DIR / 'install-dependencies.py'), '--update', '--force'],
        'services': ['bash', str(core.STACK_DIR / 'install.sh'), '--configure-services'],
    }
    command = commands.get(action)
    if not command:
        return jsonify(ok=False, error='Unknown repair action'), 400
    try:
        if action == 'dependencies':
            command = core.SETUP_RUNNER._owner_command(command)
        env = os.environ.copy()
        if action == 'services':
            selection = setup_engine.load_state()['selection']['components']
            env.update(LLM_STACK_SKIP_DEP_UPDATE='1', LLM_STACK_SKIP_EXTERNAL_INSTALL='1', LLM_STACK_SETUP_COMPONENTS=','.join(selection))
        result = subprocess.run(command, cwd=core.STACK_DIR, capture_output=True, text=True, timeout=7200, env=env)
        output = (result.stdout + result.stderr).strip()
        if result.returncode == 0 and action == 'services':
            activation = subprocess.run(['bash', str(core.SCRIPTS_DIR / 'activate-selected-stack.sh')], cwd=core.STACK_DIR, capture_output=True, text=True, timeout=900)
            output += '\n' + (activation.stdout + activation.stderr).strip()
            result = activation
        return jsonify(ok=result.returncode == 0, output=output)
    except Exception as exc:
        return jsonify(ok=False, error=str(exc)), 500


@bp.route('/api/setup/uninstall', methods=['POST'])
def api_setup_uninstall():
    if os.geteuid() != 0 and not core.ServiceManager.IS_MAC:
        return jsonify(ok=False, error='Uninstall requires a root-run manager'), 403
    data = request.get_json(silent=True) or {}
    scope = str(data.get('scope') or '')
    state = setup_engine.load_state()
    if scope == 'model':
        component = str(data.get('component') or '')
        model = state['selection'].get('models', {}).get(component)
        if not model:
            return jsonify(ok=False, error='No selected model for that component'), 404
        removed = []
        models_root = core.MODELS_DIR.resolve()
        for key in ('path', 'mmproj_path'):
            value = model.get(key)
            if not value:
                continue
            path = Path(value).resolve()
            if not path.is_relative_to(models_root):
                return jsonify(ok=False, error='Refusing to remove a model outside the managed models directory'), 400
            if path.exists():
                path.unlink()
                removed.append(str(path))
        state['selection']['models'].pop(component, None)
        setup_engine.save_state(state)
        return jsonify(ok=True, removed=removed)
    if scope == 'services':
        removed = []
        core.ServiceManager.run_cmd(['systemctl', 'disable', '--now', 'llm-stack-restore'], timeout=30)
        restore_unit = Path('/etc/systemd/system/llm-stack-restore.service')
        if restore_unit.exists():
            restore_unit.unlink()
            removed.append(str(restore_unit))
        for services in setup_engine.COMPONENT_SERVICES.values():
            for service in services:
                core.ServiceManager.run_cmd(['systemctl', 'disable', '--now', service], timeout=30)
                unit = Path('/etc/systemd/system') / f'{service}.service'
                if unit.exists():
                    unit.unlink()
                    removed.append(str(unit))
        managed_links = [
            Path('/etc/nginx/sites-enabled/llm-stack-manager'),
            Path('/etc/nginx/sites-available/llm-stack-manager'),
            Path('/etc/nginx/default.d/playwright.conf'),
            Path('/etc/nginx/default.d/searxng.conf'),
            Path('/etc/uwsgi/apps-enabled/searxng.ini'),
        ]
        for path in managed_links:
            if path.exists() or path.is_symlink():
                path.unlink()
                removed.append(str(path))
        default_site = Path('/etc/nginx/sites-available/default')
        default_link = Path('/etc/nginx/sites-enabled/default')
        if default_site.exists() and not default_link.exists():
            default_link.symlink_to(default_site)
        core.ServiceManager.run_cmd(['nginx', '-t'], timeout=15)
        core.ServiceManager.run_cmd(['systemctl', 'reload', 'nginx'], timeout=30)
        core.ServiceManager.run_cmd(['systemctl', 'daemon-reload'], timeout=30)
        state['status'] = 'services_removed'
        setup_engine.save_state(state)
        return jsonify(ok=True, removed=removed, models_preserved=True)
    return jsonify(ok=False, error='Unknown uninstall scope'), 400
