// setup.js
//
// The first-run wizard, plus the SearXNG and Playwright installers that share
// its shape: check state, run an install, poll until it settles.

// -- setup wizard --
const setupModelLabels = {
  primary: 'Primary backend', secondary: 'Secondary backend', embedding: 'Embedding',
  embedding2: 'Embedding 2', task: 'Task', ocr: 'GLM-OCR model', reranker: 'Reranker'
};

async function initSetupWizard() {
  if (setupLoaded) return;
  setupLoaded = true;
  try {
    const d = await fetchJSON('/api/setup/selection');
    setupSelection = d.selection || setupSelection;
    document.querySelectorAll('#setup-component-list input[type="checkbox"]').forEach(input => {
      input.checked = (setupSelection.components || []).includes(input.value);
      input.addEventListener('change', renderSetupModels);
    });
    document.getElementById('setup-vram-override').checked = !!setupSelection.allow_vram_override;
    renderSetupModels();
    await setupLoadPreflight();
  } catch (e) {
    toast('Setup wizard failed to load: ' + e, 'err');
  }
}

function selectedSetupComponents() {
  return [...document.querySelectorAll('#setup-component-list input:checked')].map(input => input.value);
}

function renderSetupModels() {
  const selected = selectedSetupComponents();
  const root = document.getElementById('setup-model-list');
  root.innerHTML = Object.entries(setupModelLabels)
    .filter(([component]) => selected.includes(component))
    .map(([component, label]) => {
      const saved = setupSelection.models?.[component] || {};
      return `<div class="cfg-panel setup-model-card" data-component="${component}" style="margin:12px 0;">
        <h4>${escapeHtml(label)}</h4>
        <div class="cfg-grid tight">
          <div class="field wide"><label>Hugging Face repository</label><input class="setup-repo mono" value="${escapeHtml(saved.repo_url || '')}" placeholder="owner/repo"></div>
          <div class="field"><label>Model GGUF</label><select class="setup-model-file"><option value="">-- Inspect repository first --</option></select></div>
          <div class="field"><label>MMProj GGUF (optional)</label><select class="setup-mmproj-file"><option value="">-- None --</option></select></div>
          <div class="field wide"><label>Installed path</label><input class="setup-model-path mono" value="${escapeHtml(saved.path || '')}" readonly></div>
        </div>
        <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
          <button class="btn btn-primary btn-sm" onclick="setupInspectModel('${component}',this)">Inspect</button>
          <button class="btn btn-success btn-sm" onclick="setupDownloadModel('${component}',this)">Download Selected GGUF</button>
          ${saved.path ? `<button class="btn btn-danger btn-sm" onclick="setupRemoveModel('${component}',this)">Remove Model</button>` : ''}
          <span class="hf-status setup-model-status">${saved.sha256 ? 'Validated · SHA-256 ' + escapeHtml(saved.sha256.slice(0,12)) : ''}</span>
        </div>
      </div>`;
    }).join('') || '<p>No model-backed components selected.</p>';
}

async function setupLoadPreflight(btn) {
  if (btn) { btn.disabled = true; btn.textContent = 'Checking...'; }
  try {
    const d = await fetchJSON('/api/setup/preflight');
    const root = document.getElementById('setup-preflight-results');
    root.innerHTML = Object.entries(d.checks || {}).map(([name, item]) =>
      `<div class="svc-card" data-status="${item.ok ? 'active' : (name === 'firewall' ? 'unknown' : 'failed')}">
        <div class="card-top"><span class="card-name">${escapeHtml(name.replaceAll('_',' '))}</span><span class="status-pill ${item.ok ? 'active' : (name === 'firewall' ? 'unknown' : 'failed')}">${item.ok ? 'ready' : (name === 'firewall' ? 'warning' : 'blocked')}</span></div>
        <div class="card-desc">${escapeHtml(item.value || item.warning || item.error || item.cidr || '')}</div>
      </div>`).join('');
    if (!d.ok) toast('Required setup preflight checks need attention', 'err');
  } catch (e) { toast('Preflight failed: ' + e, 'err'); }
  finally { if (btn) { btn.disabled = false; btn.textContent = 'Run Preflight'; } }
}

async function setupInspectModel(component, btn) {
  const card = btn.closest('.setup-model-card');
  const repoUrl = card.querySelector('.setup-repo').value.trim();
  const status = card.querySelector('.setup-model-status');
  btn.disabled = true; status.textContent = 'Inspecting repository...';
  try {
    const d = await fetchJSON('/api/setup/models/inspect', 'POST', {component, repo_url: repoUrl});
    setupRepoFiles[component] = d;
    const modelSelect = card.querySelector('.setup-model-file');
    const mmprojSelect = card.querySelector('.setup-mmproj-file');
    modelSelect.innerHTML = '<option value="">-- Select GGUF --</option>' + (d.model_files || []).map(item => `<option value="${escapeHtml(item.path)}" data-size="${item.size || 0}">${escapeHtml(item.path)}${item.size ? ' · ' + formatBytes(item.size) : ''}</option>`).join('');
    mmprojSelect.innerHTML = '<option value="">-- None --</option>' + (d.mmproj_files || []).map(item => `<option value="${escapeHtml(item.path)}">${escapeHtml(item.path)}</option>`).join('');
    status.textContent = `${(d.model_files || []).length} model files · ${(d.mmproj_files || []).length} projectors · license: ${d.metadata?.license || 'not declared'}${d.metadata?.gated ? ' · gated' : ''}`;
  } catch (e) { status.textContent = 'Inspect failed: ' + e; toast(status.textContent, 'err'); }
  finally { btn.disabled = false; }
}

async function setupWaitForDownload(jobId, status) {
  while (true) {
    const d = await fetchJSON(`/api/huggingface/downloads/${jobId}`);
    const job = d.job;
    status.textContent = `${job.stage || job.status}${job.progress != null ? ' · ' + job.progress + '%' : ''}`;
    if (job.status === 'done') return job;
    if (job.status === 'failed') throw new Error(job.error || 'Download failed');
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
}

async function setupDownloadModel(component, btn) {
  const card = btn.closest('.setup-model-card');
  const repoUrl = card.querySelector('.setup-repo').value.trim();
  const modelSelect = card.querySelector('.setup-model-file');
  const modelFile = modelSelect.value;
  const mmprojFile = card.querySelector('.setup-mmproj-file').value;
  const status = card.querySelector('.setup-model-status');
  if (!repoUrl || !modelFile) { toast('Inspect a repository and choose a GGUF first', 'err'); return; }
  btn.disabled = true;
  try {
    const modelMeta = (setupRepoFiles[component]?.model_files || []).find(item => item.path === modelFile) || {};
    const mmprojMeta = (setupRepoFiles[component]?.mmproj_files || []).find(item => item.path === mmprojFile) || {};
    const d = await fetchJSON('/api/huggingface/downloads', 'POST', {
      repo_url: repoUrl, model_file: modelFile, mmproj_file: mmprojFile,
      revision: setupRepoFiles[component]?.metadata?.revision_sha || setupRepoFiles[component]?.repo?.revision || 'main',
      model_sha256: modelMeta.sha256 || '', mmproj_sha256: mmprojMeta.sha256 || ''
    });
    const job = await setupWaitForDownload(d.job.id, status);
    const chosen = modelSelect.selectedOptions[0];
    setupSelection.models ||= {};
    setupSelection.models[component] = {
      repo_url: repoUrl, revision: setupRepoFiles[component]?.repo?.revision || 'main',
      resolved_revision_sha: setupRepoFiles[component]?.metadata?.revision_sha || '',
      source_file: modelFile, path: job.result.model_path, mmproj_path: job.result.mmproj_path || '',
      size: Number(chosen?.dataset.size || 0), sha256: job.result.model_sha256 || ''
    };
    card.querySelector('.setup-model-path').value = job.result.model_path;
    status.textContent = `Validated · SHA-256 ${(job.result.model_sha256 || '').slice(0,12)}`;
    toast(`${setupModelLabels[component]} downloaded and validated`, 'ok');
  } catch (e) { status.textContent = 'Download failed: ' + e; toast(status.textContent, 'err'); }
  finally { btn.disabled = false; }
}

async function setupSaveSelection(btn) {
  if (btn) btn.disabled = true;
  try {
    const payload = {
      components: selectedSetupComponents(), models: setupSelection.models || {},
      allow_vram_override: document.getElementById('setup-vram-override').checked
    };
    const d = await fetchJSON('/api/setup/selection', 'PUT', payload);
    setupSelection = d.selection;
    document.querySelectorAll('#setup-component-list input').forEach(input => input.checked = setupSelection.components.includes(input.value));
    renderSetupModels();
    toast('Setup selection saved', 'ok');
    return true;
  } catch (e) { toast('Could not save setup selection: ' + e, 'err'); return false; }
  finally { if (btn) btn.disabled = false; }
}

async function setupStartInstall(btn) {
  if (!confirm('Install the selected stack now? This will install system packages, build llama.cpp, write systemd units, and start LAN services.')) return;
  if (!(await setupSaveSelection())) return;
  btn.disabled = true;
  try {
    const d = await fetchJSON('/api/setup/run', 'POST');
    await setupPollJob(d.job.id);
  } catch (e) { toast('Setup failed to start: ' + e, 'err'); btn.disabled = false; }
}

async function setupPreviewPlacement(btn) {
  if (!(await setupSaveSelection())) return;
  btn.disabled = true;
  try {
    const d = await fetchJSON('/api/setup/placement');
    const panel = document.getElementById('setup-placement');
    panel.style.display = 'block';
    panel.innerHTML = `<h4>Generated GPU Placement</h4><p>Estimated ${d.required_mib || 0} MiB of ${d.usable_mib || 0} MiB usable VRAM.</p>` +
      Object.entries(d.assignments || {}).map(([name,item]) => `<div><strong>${escapeHtml(name)}</strong>: GPU ${escapeHtml((item.gpu_indices || []).join(','))}${item.tensor_split ? ' · split ' + escapeHtml(item.tensor_split) : ''}${item.estimated_mib ? ' · estimated ' + item.estimated_mib + ' MiB' : ''}</div>`).join('');
  } catch (e) { toast('GPU placement preview failed: ' + e, 'err'); }
  finally { btn.disabled = false; }
}

async function setupPollJob(jobId) {
  const panel = document.getElementById('setup-job-status');
  const log = document.getElementById('setup-job-log');
  panel.style.display = 'block';
  while (true) {
    const d = await fetchJSON(`/api/setup/jobs/${jobId}`);
    const job = d.job;
    panel.innerHTML = `<h4>${escapeHtml(job.stage || job.status)} · ${job.progress || 0}%</h4><p>${escapeHtml(job.error || '')}</p>`;
    log.textContent = (job.log || []).join('\n'); log.scrollTop = log.scrollHeight;
    if (['complete','needs_attention','failed'].includes(job.status)) {
      document.getElementById('setup-install-btn').disabled = false;
      toast(job.status === 'complete' ? 'Stack installation complete' : `Setup ${job.status}`, job.status === 'complete' ? 'ok' : 'err');
      if (job.status === 'complete') setupShowCompletion();
      else panel.innerHTML += `<button class="btn btn-primary btn-sm" onclick="setupRetryJob('${jobId}')">Retry From Saved State</button>`;
      return;
    }
    await new Promise(resolve => setTimeout(resolve, 1500));
  }
}

async function setupRetryJob(jobId) {
  const d = await fetchJSON(`/api/setup/jobs/${jobId}/retry`, 'POST');
  await setupPollJob(d.job.id);
}

async function setupValidate(btn) {
  if (btn) btn.disabled = true;
  try {
    const d = await fetchJSON('/api/setup/validation');
    toast(d.ok ? 'All selected services are active' : 'Some selected services need attention', d.ok ? 'ok' : 'err');
    document.getElementById('setup-job-status').style.display = 'block';
    document.getElementById('setup-job-status').innerHTML = Object.entries(d.checks || {}).map(([name, check]) => `<div>${escapeHtml(name)}: ${check.ok ? 'ready' : 'failed'}</div>`).join('');
    if (d.ok) setupShowCompletion();
  } catch (e) { toast('Validation failed: ' + e, 'err'); }
  finally { if (btn) btn.disabled = false; }
}

async function setupShowCompletion() {
  const host = location.hostname;
  const selected = selectedSetupComponents();
  const endpoints = [];
  if (selected.includes('primary')) endpoints.push(['Chat / code', `http://${host}:8008/v1`]);
  if (selected.includes('embedding')) endpoints.push(['Embeddings', `http://${host}:8005/v1`]);
  if (selected.includes('task')) endpoints.push(['Task', `http://${host}:8007/v1`]);
  if (selected.includes('ocr')) endpoints.push(['OCR model', `http://${host}:8009/v1`]);
  if (selected.includes('glmocr-sdk')) endpoints.push(['GLM-OCR SDK', `http://${host}:5002/glmocr/parse`]);
  if (selected.includes('searxng')) endpoints.push(['SearXNG JSON', `http://${host}/searxng/search?q=test&format=json`]);
  if (selected.includes('playwright')) endpoints.push(['Playwright remote protocol', `ws://${host}/playwright/`]);
  const modelSummary = Object.entries(setupSelection.models || {}).filter(([name]) => selected.includes(name)).map(([name,model]) => `<div><strong>${escapeHtml(setupModelLabels[name] || name)}</strong>: ${escapeHtml((model.path || '').split('/').pop())}</div>`).join('');
  let placementSummary = '';
  try {
    const placement = await fetchJSON('/api/setup/placement');
    placementSummary = Object.entries(placement.assignments || {}).map(([name,item]) => `<div><strong>${escapeHtml(name)}</strong>: GPU ${escapeHtml((item.gpu_indices || []).join(','))}</div>`).join('');
  } catch (_) {}
  document.getElementById('setup-completion').innerHTML = `<h3>LAN Stack Endpoints</h3><div class="endpoint-list">${endpoints.map(([label,url]) => `<div><strong>${escapeHtml(label)}</strong><code>${escapeHtml(url)}</code></div>`).join('')}</div><h4>Models</h4>${modelSummary}<h4>GPU Placement</h4>${placementSummary}<p>Recovery: <code>sudo ${escapeHtml(window.__STACK__.modelsDir.replace('/models',''))}/scripts/setup_engine.py validate</code></p>`;
}

async function setupRepair(action, btn) {
  if (!confirm(`Run ${action} repair now? Existing models and configuration will be preserved.`)) return;
  btn.disabled = true;
  try {
    const d = await fetchJSON('/api/setup/repair', 'POST', {action});
    document.getElementById('setup-job-log').textContent = d.output || d.error || '';
    toast(d.ok ? 'Repair completed' : 'Repair failed', d.ok ? 'ok' : 'err');
  } catch (e) { toast('Repair failed: ' + e, 'err'); }
  finally { btn.disabled = false; }
}

async function setupRemoveServices(btn) {
  if (!confirm('Remove generated runtime service units? The manager, downloaded models, and configuration will be preserved.')) return;
  btn.disabled = true;
  try {
    const d = await fetchJSON('/api/setup/uninstall', 'POST', {scope: 'services'});
    toast(d.ok ? 'Stack services removed; models preserved' : 'Service removal failed', d.ok ? 'ok' : 'err');
  } catch (e) { toast('Service removal failed: ' + e, 'err'); }
  finally { btn.disabled = false; }
}

async function setupRemoveModel(component, btn) {
  if (!confirm(`Remove the managed ${setupModelLabels[component]} GGUF and its MMProj file?`)) return;
  btn.disabled = true;
  try {
    const d = await fetchJSON('/api/setup/uninstall', 'POST', {scope: 'model', component});
    delete setupSelection.models[component];
    renderSetupModels();
    toast(`Removed ${d.removed.length} model file(s)`, 'ok');
  } catch (e) { toast('Model removal failed: ' + e, 'err'); btn.disabled = false; }
}

// -- searxng --
function renderEndpointList(targetId, endpoints = {}) {
  const el = document.getElementById(targetId);
  if (!el) return;
  el.innerHTML = Object.entries(endpoints).map(([name, value]) =>
    `<span class="meta-chip"><strong>${escapeHtml(name)}:</strong> <code>${escapeHtml(value)}</code></span>`
  ).join('') || '<span class="meta-chip">No endpoints configured</span>';
}

async function initSearxngTab() {
  if (searxngLoaded) return;
  searxngLoaded = true;
  await loadSearxngStatus();
}

async function loadSearxngStatus() {
  try {
    const d = await fetchJSON('/api/searxng/status');
    const cfg = d.config || {};
    document.getElementById('searxng-last-refresh').textContent = `Last refresh: ${fmtEpoch(d.last_refresh)}`;
    document.getElementById('searxng-service-status').textContent = d.service_status || '-';
    document.getElementById('searxng-public-url').textContent = cfg.public_url || '-';
    const link = document.getElementById('searxng-open-link');
    const frame = document.getElementById('searxng-frame');
    if (cfg.public_url) {
      link.href = cfg.public_url;
      frame.src = cfg.public_url;
    }
    document.getElementById('searxng-meta-list').innerHTML = [
      `Enabled: ${cfg.enabled || '-'}`,
      `Formats: ${cfg.formats || '-'}`,
      `Path: ${cfg.url_path || '-'}`,
      `Base URL: ${cfg.base_url || '-'}`
    ].map(item => `<span class="meta-chip">${escapeHtml(item)}</span>`).join('');
    document.getElementById('searxng-path-list').innerHTML = [
      `Home: ${cfg.home || '-'}`,
      `Settings: ${cfg.settings_path || '-'}`,
      `uWSGI: ${cfg.uwsgi_ini || '-'}`,
      `Socket: ${cfg.uwsgi_socket || '-'}`,
      `Nginx: ${cfg.nginx_conf || '-'}`
    ].map(item => `<span class="meta-chip">${escapeHtml(item)}</span>`).join('');
    renderEndpointList('searxng-endpoint-list', cfg.endpoints || {});

    const checks = d.checks || {};
    document.getElementById('searxng-status-cards').innerHTML = Object.entries(checks).map(([key, v]) => {
      const ok = v?.ok === true;
      const status = ok ? 'active' : 'failed';
      const label = key.replaceAll('_', ' ');
      const detail = v?.status || v?.path || v?.error || (ok ? 'ok' : 'not ready');
      const count = typeof v?.result_count === 'number' ? `<span class="meta-chip">results: ${v.result_count}</span>` : '';
      return `
        <div class="svc-card" data-status="${status}">
          <div class="card-top">
            <span class="card-name">${escapeHtml(label)}</span>
            <span class="status-pill ${status}">${ok ? 'ok' : 'down'}</span>
          </div>
          <div class="card-desc">${escapeHtml(detail)}</div>
          ${count ? `<div class="meta-list">${count}</div>` : ''}
        </div>`;
    }).join('');
  } catch (e) {
    toast('SearXNG status failed: ' + e, 'err');
  }
}

async function installSearxng(btn) {
  if (!confirm('Install or update SearXNG now? This can install apt packages, pull SearXNG, update its virtualenv, and reload nginx/uWSGI.')) return;
  const orig = btn.textContent;
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Installing...';
  try {
    const d = await fetchJSON('/api/searxng/install', 'POST');
    toast(d.ok ? 'SearXNG install/update complete' : (d.output || d.error || 'SearXNG install failed'), d.ok ? 'ok' : 'err');
    await loadSearxngStatus();
    await poll();
  } catch (e) {
    toast('SearXNG install failed: ' + e, 'err');
  } finally {
    btn.disabled = false;
    btn.textContent = orig;
  }
}

// -- playwright --
async function initPlaywrightTab() {
  if (playwrightLoaded) return;
  playwrightLoaded = true;
  await loadPlaywrightStatus();
}

async function loadPlaywrightStatus() {
  try {
    const d = await fetchJSON('/api/playwright/status');
    const cfg = d.config || {};
    document.getElementById('playwright-last-refresh').textContent = `Last refresh: ${fmtEpoch(d.last_refresh)}`;
    document.getElementById('playwright-service-status').textContent = d.service_status || '-';
    document.getElementById('playwright-ws-url').textContent = cfg.public_ws_url || '-';
    document.getElementById('playwright-meta-list').innerHTML = [
      `Enabled: ${cfg.enabled || '-'}`,
      `Browser: ${cfg.browser || '-'}`,
      `Host: ${cfg.host || '-'}`,
      `Port: ${cfg.port || '-'}`,
      `Internal: ${cfg.upstream_port || '-'}`,
      `Path: ${cfg.url_path || '-'}`
    ].map(item => `<span class="meta-chip">${escapeHtml(item)}</span>`).join('');
    document.getElementById('playwright-path-list').innerHTML = [
      `Server dir: ${cfg.server_dir || '-'}`,
      `package.json: ${cfg.package_json || '-'}`,
      `node_modules: ${cfg.node_modules || '-'}`,
      `Browsers: ${cfg.browsers_path || '-'}`,
      `Unit: ${cfg.service_unit || '-'}`,
      `Nginx: ${cfg.nginx_conf || '-'}`
    ].map(item => `<span class="meta-chip">${escapeHtml(item)}</span>`).join('');
    renderEndpointList('playwright-endpoint-list', cfg.endpoints || {});

    const checks = d.checks || {};
    document.getElementById('playwright-status-cards').innerHTML = Object.entries(checks).map(([key, v]) => {
      const ok = v?.ok === true;
      const status = ok ? 'active' : 'failed';
      const label = key.replaceAll('_', ' ');
      const detail = v?.status || v?.path || v?.endpoint || v?.error || (ok ? 'ok' : 'not ready');
      return `
        <div class="svc-card" data-status="${status}">
          <div class="card-top">
            <span class="card-name">${escapeHtml(label)}</span>
            <span class="status-pill ${status}">${ok ? 'ok' : 'down'}</span>
          </div>
          <div class="card-desc">${escapeHtml(detail)}</div>
        </div>`;
    }).join('');
  } catch (e) {
    toast('Playwright status failed: ' + e, 'err');
  }
}

async function installPlaywright(btn) {
  if (!confirm('Install or update Playwright now? This can run npm install, download browser binaries, and write the systemd unit.')) return;
  const orig = btn.textContent;
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Installing...';
  try {
    const d = await fetchJSON('/api/playwright/install', 'POST');
    toast(d.ok ? 'Playwright install/update complete' : (d.output || d.error || 'Playwright install failed'), d.ok ? 'ok' : 'err');
    await loadPlaywrightStatus();
    await poll();
  } catch (e) {
    toast('Playwright install failed: ' + e, 'err');
  } finally {
    btn.disabled = false;
    btn.textContent = orig;
  }
}
