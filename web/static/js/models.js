// models.js
//
// The model catalogue: which model is loaded, what is on disk, HuggingFace
// downloads, and the custom-model editor with its per-family launch arguments.

// -- active model tracking --
async function pollActiveModel() {
  const d = await fetchJSON('/api/active-chat-model');
  activeModel = d.variant;
  const label = document.getElementById('active-model-label');
  // Update label
  if (builtInChatVariantMap[d.variant]) label.textContent = 'Active: ' + (d.label || builtInChatVariantMap[d.variant].label);
  else if (d.custom_model) label.textContent = 'Active: ' + d.custom_model.display_name;
  else if (d.variant === 'generic') label.textContent = 'Active: Custom';
  else label.textContent = 'No backend active';

  // Highlight active switch button
  document.querySelectorAll('.switch-model-btn').forEach(btn => {
    if (btn.dataset.variant === d.variant) {
      btn.classList.add('btn-active-model');
      btn.classList.remove('btn-ghost');
    } else {
      btn.classList.remove('btn-active-model');
      btn.classList.add('btn-ghost');
    }
  });

  // Update custom model card statuses
  updateCustomModelCardStatuses(d);
}

function updateCustomModelCardStatuses(activeData) {
  customModels.forEach(m => {
    const card = document.getElementById('custom-card-' + m.id);
    if (!card) return;
    const pill = card.querySelector('.status-pill');
    if (activeData.variant === m.id) {
      card.dataset.status = 'active';
      pill.className = 'status-pill active';
      pill.textContent = 'active';
    } else {
      card.dataset.status = 'inactive';
      pill.className = 'status-pill inactive';
      pill.textContent = 'inactive';
    }
  });
}

function escapeHtml(s) {
  return String(s ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function prettyJson(v) {
  return JSON.stringify(v, null, 2);
}

function fmtEpoch(epoch) {
  if (!epoch) return '-';
  try { return new Date(epoch * 1000).toLocaleString(); } catch { return '-'; }
}

function formatBytes(value) {
  const n = Number(value || 0);
  if (!Number.isFinite(n) || n <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let size = n;
  let unitIndex = 0;
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex += 1;
  }
  return `${size.toFixed(size >= 10 || unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

function slugFromFilename(filename = '') {
  return filename
    .replace(/\.gguf$/i, '')
    .replace(/[_\s]+/g, '-')
    .replace(/[^a-zA-Z0-9.-]+/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '')
    .toLowerCase();
}

function filenameFromPath(path = '') {
  return String(path || '').split('/').filter(Boolean).pop() || '';
}

function displayNameFromFilename(filename = '') {
  return filenameFromPath(filename).replace(/\.gguf$/i, '');
}

function isLikelyMmprojGguf(file) {
  if (file && typeof file.is_mmproj === 'boolean') return file.is_mmproj;
  const name = String(file?.name || filenameFromPath(file?.path || '')).toLowerCase();
  const relative = String(file?.relative || file?.path || '').toLowerCase();
  if (name.includes('mmproj') || relative.includes('mmproj')) return true;
  if (name.includes('projector') || relative.includes('projector')) return true;
  if (name.includes('mm_project') || relative.includes('mm_project')) return true;
  if ((name.includes('clip') || name.includes('vision')) && Number(file?.size_gb || 0) > 0 && Number(file.size_gb) < 2) return true;
  return false;
}

function ggufFilesForTarget(targetId = '') {
  const key = String(targetId || '').replace(/^cfg-/, '').toUpperCase();
  const wantsMmproj = key.includes('MMPROJ');
  return ggufFiles.filter(file => wantsMmproj ? isLikelyMmprojGguf(file) : !isLikelyMmprojGguf(file));
}

function fillCustomModelIdentityFromPath(path = '', overwrite = false) {
  const filename = filenameFromPath(path);
  if (!filename) return;
  const display = document.getElementById('new-display-name');
  const model = document.getElementById('new-model-name');
  if (display && (overwrite || !display.value.trim())) {
    display.value = displayNameFromFilename(filename);
  }
  if (model && (overwrite || !model.value.trim())) {
    model.value = slugFromFilename(filename);
  }
  scheduleCustomArgPresetRefresh();
}

function ensureCustomModelIdentity() {
  const selectedHfPath = document.getElementById('hf-model-file-select')?.value || '';
  const modelPath = document.getElementById('new-model-path')?.value || '';
  const identitySource = modelPath || selectedHfPath;
  if (identitySource) {
    fillCustomModelIdentityFromPath(identitySource);
  }
  const display = document.getElementById('new-display-name');
  const model = document.getElementById('new-model-name');
  if (!display.value.trim() && identitySource) {
    display.value = displayNameFromFilename(identitySource);
  }
  if (!model.value.trim() && display.value.trim()) {
    model.value = slugFromFilename(display.value.trim());
  }
  return {
    displayName: display.value.trim(),
    modelName: model.value.trim(),
  };
}

async function initGraphitiTab() {
  if (graphitiLoaded) return;
  graphitiLoaded = true;
  await refreshGraphitiAll();
}

async function refreshGraphitiAll() {
  await loadGraphitiStatus();
  await loadGraphitiStats();
  await refreshGraphitiRecents();
  await loadGraphitiExports();
}

async function loadGraphitiStatus() {
  try {
    const d = await fetchJSON('/api/graphiti/status');
    document.getElementById('graphiti-last-refresh').textContent = `Last refresh: ${fmtEpoch(d.last_refresh)}`;
    const checks = d.checks || {};
    const cards = [
      ['Graphiti API', checks.graphiti_api],
      ['Neo4j', checks.neo4j],
      ['LLM Endpoint', checks.llm_endpoint],
      ['Embedding Endpoint', checks.embedding_endpoint],
      ['Reranker Endpoint', checks.reranker_endpoint],
      ['Ingestion Worker', checks.ingestion_worker],
    ];
    document.getElementById('graphiti-status-cards').innerHTML = cards.map(([label, v]) => {
      const ok = v?.ok === true;
      const unknown = v?.ok === null || typeof v?.ok === 'undefined';
      const status = ok ? 'active' : (unknown ? 'unknown' : 'failed');
      const text = ok ? 'ok' : (unknown ? 'unknown' : 'down');
      const err = escapeHtml(v?.error || '');
      return `
        <div class="svc-card" data-status="${status}">
          <div class="card-top">
            <span class="card-name">${escapeHtml(label)}</span>
            <span class="status-pill ${status}">${text}</span>
          </div>
          <div class="card-desc">${err || 'reachable'}</div>
        </div>
      `;
    }).join('');
  } catch (e) {
    toast('Graphiti status failed: ' + e, 'err');
  }
}

async function loadGraphitiStats() {
  try {
    const d = await fetchJSON('/api/graphiti/stats');
    if (!d.ok) throw new Error(d.error || 'stats failed');
    const t = d.totals || {};
    document.getElementById('graphiti-totals').innerHTML = [
      `Episodes: ${t.episodes ?? 0}`,
      `Entities: ${t.entities ?? 0}`,
      `Relationships: ${t.relationships ?? 0}`,
      `Refreshed: ${fmtEpoch(d.last_refresh)}`
    ].map(x => `<span class="meta-chip">${escapeHtml(x)}</span>`).join('');

    const groupsBody = document.querySelector('#graphiti-top-groups-table tbody');
    groupsBody.innerHTML = (d.top_groups || []).map(r =>
      `<tr><td>${escapeHtml(r.group_id || '(none)')}</td><td>${escapeHtml(r.c)}</td></tr>`
    ).join('') || `<tr><td colspan="2">No data</td></tr>`;

    const entsBody = document.querySelector('#graphiti-top-entities-table tbody');
    entsBody.innerHTML = (d.top_entities || []).map(r =>
      `<tr><td>${escapeHtml(r.name || r.uuid || '')}</td><td>${escapeHtml(r.group_id || '')}</td><td>${escapeHtml(r.degree)}</td></tr>`
    ).join('') || `<tr><td colspan="3">No data</td></tr>`;
  } catch (e) {
    toast('Graphiti stats failed: ' + e, 'err');
  }
}

function graphitiFilters() {
  return {
    group_id: document.getElementById('graphiti-filter-group').value.trim(),
    q: document.getElementById('graphiti-filter-q').value.trim(),
    start_time: document.getElementById('graphiti-filter-start').value.trim(),
    end_time: document.getElementById('graphiti-filter-end').value.trim(),
    page_size: 25,
    page: 1,
  };
}

async function refreshGraphitiRecents() {
  const f = graphitiFilters();
  const qs = new URLSearchParams(f).toString();
  try {
    const [episodes, entities, rels] = await Promise.all([
      fetchJSON('/api/graphiti/recent/episodes?' + qs),
      fetchJSON('/api/graphiti/recent/entities?' + qs),
      fetchJSON('/api/graphiti/recent/relationships?' + qs),
    ]);

    const epBody = document.querySelector('#graphiti-episodes-table tbody');
    epBody.innerHTML = (episodes.items || []).map(e => `
      <tr>
        <td>${escapeHtml(e.created_at || '')}</td>
        <td>${escapeHtml(e.group_id || '')}</td>
        <td>${escapeHtml(e.source || '')}</td>
        <td>${escapeHtml(e.content_snippet || '')}</td>
        <td><button class="btn btn-ghost btn-sm" onclick="showEpisodeDetail('${escapeHtml(e.uuid)}')">Open</button></td>
      </tr>
    `).join('') || `<tr><td colspan="5">No episodes found</td></tr>`;

    const enBody = document.querySelector('#graphiti-entities-table tbody');
    enBody.innerHTML = (entities.items || []).map(e => `
      <tr>
        <td>${escapeHtml(e.created_at || '')}</td>
        <td>${escapeHtml(e.name || '')}</td>
        <td>${escapeHtml(e.group_id || '')}</td>
        <td>${escapeHtml((e.labels || []).join(', '))}</td>
        <td>${escapeHtml(e.degree || 0)}</td>
        <td><button class="btn btn-ghost btn-sm" onclick="showEntityDetail('${escapeHtml(e.uuid)}')">Open</button></td>
      </tr>
    `).join('') || `<tr><td colspan="6">No entities found</td></tr>`;

    const relBody = document.querySelector('#graphiti-relationships-table tbody');
    relBody.innerHTML = (rels.items || []).map(r => `
      <tr>
        <td>${escapeHtml(r.created_at || '')}</td>
        <td>${escapeHtml(r.relation_name || '')}</td>
        <td>${escapeHtml(r.source_name || '')}</td>
        <td>${escapeHtml(r.target_name || '')}</td>
        <td>${escapeHtml(r.fact_snippet || '')}</td>
        <td><button class="btn btn-ghost btn-sm" onclick="showRelationshipDetail('${escapeHtml(r.uuid)}')">Open</button></td>
      </tr>
    `).join('') || `<tr><td colspan="6">No relationships found</td></tr>`;
  } catch (e) {
    toast('Graphiti recents failed: ' + e, 'err');
  }
}

async function showEpisodeDetail(uuid) {
  const d = await fetchJSON(`/api/graphiti/detail/episode/${encodeURIComponent(uuid)}`);
  if (!d.ok) { toast(d.error || 'episode detail failed', 'err'); return; }
  document.getElementById('graphiti-detail-output').textContent = prettyJson(d.item);
}

async function showEntityDetail(uuid) {
  const d = await fetchJSON(`/api/graphiti/detail/entity/${encodeURIComponent(uuid)}`);
  if (!d.ok) { toast(d.error || 'entity detail failed', 'err'); return; }
  document.getElementById('graphiti-detail-output').textContent = prettyJson(d.item);
}

async function showRelationshipDetail(uuid) {
  const d = await fetchJSON(`/api/graphiti/detail/relationship/${encodeURIComponent(uuid)}`);
  if (!d.ok) { toast(d.error || 'relationship detail failed', 'err'); return; }
  document.getElementById('graphiti-detail-output').textContent = prettyJson(d.item);
}

async function runGraphitiMemorySearch() {
  const query = document.getElementById('graphiti-search-query').value.trim();
  const group_id = document.getElementById('graphiti-search-group').value.trim();
  const max_facts = Number(document.getElementById('graphiti-search-max-facts').value || '10');
  if (!query) { toast('Enter memory search text', 'err'); return; }
  const d = await fetchJSON('/api/graphiti/search/memory', 'POST', { query, group_id, max_facts });
  if (!d.ok) { toast(d.error || 'search failed', 'err'); return; }
  document.getElementById('graphiti-search-output').textContent = prettyJson(d.result || {});
}

async function inspectGraphitiGroup() {
  const group = document.getElementById('graphiti-inspect-group').value.trim();
  const last_n = Number(document.getElementById('graphiti-inspect-lastn').value || '30');
  if (!group) { toast('Enter a group_id', 'err'); return; }
  const d = await fetchJSON(`/api/graphiti/search/group/${encodeURIComponent(group)}?last_n=${last_n}`);
  if (!d.ok) { toast(d.error || 'group inspect failed', 'err'); return; }
  document.getElementById('graphiti-search-output').textContent = prettyJson(d);
}

async function inspectGraphitiNeighborhood() {
  const uuid = document.getElementById('graphiti-inspect-entity').value.trim();
  const limit = Number(document.getElementById('graphiti-inspect-limit').value || '50');
  if (!uuid) { toast('Enter an entity uuid', 'err'); return; }
  const d = await fetchJSON(`/api/graphiti/neighborhood/${encodeURIComponent(uuid)}?limit=${limit}`);
  if (!d.ok) { toast(d.error || 'neighborhood inspect failed', 'err'); return; }
  document.getElementById('graphiti-search-output').textContent = prettyJson(d.item || {});
}

async function createGraphitiExport() {
  const export_type = document.getElementById('graphiti-export-type').value;
  const format = document.getElementById('graphiti-export-format').value;
  const limit = Number(document.getElementById('graphiti-export-limit').value || '200');
  const group_id = document.getElementById('graphiti-export-group').value.trim();
  const entity_uuid = document.getElementById('graphiti-export-entity').value.trim();
  const start_time = document.getElementById('graphiti-export-start').value.trim();
  const end_time = document.getElementById('graphiti-export-end').value.trim();
  const payload = { export_type, format, limit, group_id, entity_uuid, start_time, end_time };
  const d = await fetchJSON('/api/graphiti/export', 'POST', payload);
  if (!d.ok) { toast(d.error || 'export failed', 'err'); return; }
  toast(`Export created: ${d.file.filename}`, 'ok');
  await loadGraphitiExports();
}

async function loadGraphitiExports() {
  try {
    const d = await fetchJSON('/api/graphiti/exports');
    if (!d.ok) throw new Error(d.error || 'exports list failed');
    document.getElementById('graphiti-export-dir').innerHTML =
      `<span class="meta-chip">Export dir: ${escapeHtml(d.directory || '-')}</span>`;
    const body = document.querySelector('#graphiti-exports-table tbody');
    body.innerHTML = (d.items || []).map(f => `
      <tr>
        <td>${escapeHtml(f.filename)}</td>
        <td>${escapeHtml(f.size_bytes)}</td>
        <td>${escapeHtml(fmtEpoch(f.modified_at))}</td>
        <td><a class="btn btn-ghost btn-sm" href="${escapeHtml(f.download_url)}">Download</a></td>
      </tr>
    `).join('') || `<tr><td colspan="4">No exports yet</td></tr>`;
  } catch (e) {
    toast('Failed to load export files: ' + e, 'err');
  }
}

// -- GGUF file management --
async function loadGgufFiles() {
  try {
    ggufFiles = await fetchJSON('/api/gguf-files');
    populateGgufSelects();
  } catch (e) { console.error('Failed to load GGUF files:', e); }
}

async function loadTranscriptionModels(engineId) {
  try {
    const d = await fetchJSON(`/api/transcription-models/${engineId}`);
    transcriptionModelsByEngine[engineId] = d.models || [];
    populateTranscriptionModelSelects();
  } catch (e) {
    console.error('Failed to load transcription models:', e);
  }
}

async function loadTranscriptionCapabilities() {
  try {
    const d = await fetchJSON('/api/transcription-capabilities');
    transcriptionCapabilities = d.engines || {};
    applyTranscriptionCapabilities();
  } catch (e) {
    console.error('Failed to load transcription capabilities:', e);
  }
}

function populateGgufSelects() {
  // All selects with class gguf-path-select (config tab) and add-model selects
  const allSelects = document.querySelectorAll('.gguf-path-select');
  allSelects.forEach(sel => {
    const targetId = sel.dataset.target;
    const targetInput = document.getElementById(targetId);
    const currentVal = targetInput ? targetInput.value : '';

    // Keep first option, clear rest
    while (sel.options.length > 1) sel.remove(1);

    ggufFilesForTarget(targetId).forEach(f => {
      const opt = document.createElement('option');
      opt.value = f.path;
      opt.textContent = `${f.name} (${f.size_gb} GB)`;
      if (f.path === currentVal) opt.selected = true;
      sel.appendChild(opt);
    });
  });

  // Add-model selects
  populateAddModelSelects();
}

function populateTranscriptionModelSelects() {
  document.querySelectorAll('.transcript-model-select').forEach(sel => {
    const targetId = sel.dataset.target;
    const targetInput = document.getElementById(targetId);
    const currentVal = targetInput ? targetInput.value : '';
    const engineId = sel.dataset.engineId || '';
    const models = transcriptionModelsByEngine[engineId] || [];

    while (sel.options.length > 1) sel.remove(1);

    models.forEach(item => {
      const opt = document.createElement('option');
      opt.value = item.value;
      opt.textContent = item.kind === 'local'
        ? `${item.label} · ${item.format || 'folder'}${item.supported_local === false ? ' · unsupported' : ''}`
        : `${item.label} · preset`;
      if (item.value === currentVal) opt.selected = true;
      sel.appendChild(opt);
    });
  });
}

function populateAddModelSelects() {
  const modelSel = document.getElementById('new-model-gguf-select');
  const mmprojSel = document.getElementById('new-mmproj-gguf-select');
  if (!modelSel || !mmprojSel) return;

  // Clear and repopulate model select
  while (modelSel.options.length > 1) modelSel.remove(1);
  while (mmprojSel.options.length > 1) mmprojSel.remove(1);

  ggufFiles.filter(f => !isLikelyMmprojGguf(f)).forEach(f => {
    const opt1 = document.createElement('option');
    opt1.value = f.path;
    opt1.textContent = `${f.name} (${f.size_gb} GB)`;
    modelSel.appendChild(opt1);
  });

  ggufFiles.filter(f => isLikelyMmprojGguf(f)).forEach(f => {
    const opt2 = document.createElement('option');
    opt2.value = f.path;
    opt2.textContent = `${f.name} (${f.size_gb} GB)`;
    mmprojSel.appendChild(opt2);
  });
}

function ggufSelectChanged(sel, targetId) {
  const input = document.getElementById(targetId);
  if (input && sel.value) input.value = sel.value;
  if (targetId === 'new-model-path' && sel.value) {
    fillCustomModelIdentityFromPath(sel.value);
  }
}

function transcriptModelSelectChanged(sel, targetId) {
  const input = document.getElementById(targetId);
  if (input) input.value = sel.value;
}

function applyTranscriptionCapabilities() {
  document.querySelectorAll('.transcript-capability-panel').forEach(panel => {
    const engineId = panel.dataset.engineId;
    const capability = panel.dataset.capability;
    const supported = !!transcriptionCapabilities?.[engineId]?.[capability];
    panel.style.display = supported ? '' : 'none';
  });
}

function parseArgJson(value) {
  if (!value) return [];
  try {
    const parsed = JSON.parse(value);
    return Array.isArray(parsed) ? parsed.filter(v => String(v || '').trim()).map(v => String(v)) : [];
  } catch {
    return [];
  }
}

function syncConfigArgInput(targetId, silent = false) {
  const input = document.getElementById(targetId);
  if (!input) return;
  const rows = [...document.querySelectorAll(`.custom-args-list[data-config-target="${targetId}"] .custom-arg-input`)];
  const values = rows.map(el => el.value.trim()).filter(Boolean);
  input.value = JSON.stringify(values);
  if (!silent) {
    const section = input.closest('.cfg-section')?.id?.replace(/^cfgsec-/, '').replace(/-/g, ' ') || 'Shared Backend';
    markDirty(section);
  }
  if (targetId === 'cfg-CHAT_PRIMARY_CUSTOM_ARGS_JSON') refreshCacheAwareScheduling('CHAT_PRIMARY');
  if (targetId === 'cfg-CHAT2_CUSTOM_ARGS_JSON') refreshCacheAwareScheduling('CHAT2');
}

function renderConfigArgRows(targetId, values = ['']) {
  const list = document.querySelector(`.custom-args-list[data-config-target="${targetId}"]`);
  if (!list) return;
  const rows = (values && values.length ? values : ['']).map((value, index) => `
    <div class="custom-arg-row">
      <input type="text" class="custom-arg-input mono" value="${escapeHtml(value)}" placeholder="--chat-template-kwargs '{&quot;preserve_thinking&quot;: true}'" oninput="syncConfigArgInput('${targetId}')">
      <button class="btn btn-ghost btn-sm" type="button" onclick="addConfigArgRow('${targetId}')">+</button>
      <button class="btn btn-ghost btn-sm" type="button" onclick="removeConfigArgRow(this, '${targetId}')" ${index === 0 && values.length <= 1 ? 'disabled' : ''}>-</button>
    </div>
  `).join('');
  list.innerHTML = rows;
  syncConfigArgInput(targetId, true);
}

function addConfigArgRow(targetId, value = '') {
  const input = document.getElementById(targetId);
  const current = parseArgJson(input?.value || '[]');
  current.push(value);
  renderConfigArgRows(targetId, current);
}

function removeConfigArgRow(btn, targetId) {
  const input = document.getElementById(targetId);
  const current = parseArgJson(input?.value || '[]');
  const row = btn.closest('.custom-arg-row');
  const rows = [...row.parentElement.querySelectorAll('.custom-arg-row')];
  const index = rows.indexOf(row);
  if (current.length <= 1) {
    renderConfigArgRows(targetId, ['']);
    return;
  }
  current.splice(index, 1);
  renderConfigArgRows(targetId, current);
}

function initConfigArgEditors() {
  document.querySelectorAll('.custom-args-list[data-config-target]').forEach(list => {
    const targetId = list.dataset.configTarget;
    const input = document.getElementById(targetId);
    renderConfigArgRows(targetId, parseArgJson(input?.value || '[]'));
  });
}

function setChatTemplateStatus(message = '') {
  const el = document.getElementById('chat-template-status');
  if (el) el.textContent = message;
}

function populateChatTemplateSelect(select, currentValue = '') {
  if (!select) return;
  select.innerHTML = '';
  chatTemplates.forEach(t => {
    const opt = document.createElement('option');
    opt.value = t.id || '';
    opt.textContent = t.id ? `${t.name || t.id} (${t.id})` : (t.name || 'Model default');
    select.appendChild(opt);
  });
  select.value = currentValue || '';
}

function refreshChatTemplateSelects() {
  document.querySelectorAll('.chat-template-select[name]').forEach(select => {
    populateChatTemplateSelect(select, select.value);
  });

  const list = document.getElementById('chat-template-list');
  if (list) {
    list.innerHTML = '<option value="">-- Select template --</option>';
    chatTemplates.filter(t => t.id).forEach(t => {
      const opt = document.createElement('option');
      opt.value = t.id;
      opt.textContent = `${t.name || t.id} (${t.id})`;
      list.appendChild(opt);
    });
    list.value = editingChatTemplateId || '';
  }
}

async function loadChatTemplates() {
  try {
    chatTemplates = await fetchJSON('/api/chat-templates');
    refreshChatTemplateSelects();
    setChatTemplateStatus(`${Math.max(chatTemplates.length - 1, 0)} saved templates`);
  } catch (e) {
    setChatTemplateStatus('Could not load templates');
    toast('Could not load chat templates: ' + e, 'err');
  }
}

function newChatTemplate() {
  editingChatTemplateId = '';
  const list = document.getElementById('chat-template-list');
  if (list) list.value = '';
  document.getElementById('chat-template-name').value = '';
  document.getElementById('chat-template-description').value = '';
  document.getElementById('chat-template-content').value = '';
  setChatTemplateStatus('New template');
}

async function selectChatTemplateForEdit(templateId) {
  editingChatTemplateId = templateId || '';
  if (!editingChatTemplateId) {
    newChatTemplate();
    return;
  }
  try {
    const d = await fetchJSON(`/api/chat-templates/${encodeURIComponent(editingChatTemplateId)}`);
    if (!d.ok) {
      toast(d.error || 'Template not found', 'err');
      return;
    }
    document.getElementById('chat-template-name').value = d.name || editingChatTemplateId;
    document.getElementById('chat-template-description').value = d.description || '';
    document.getElementById('chat-template-content').value = d.content || '';
    setChatTemplateStatus(`Editing ${editingChatTemplateId}`);
  } catch (e) {
    toast('Could not load template: ' + e, 'err');
  }
}

async function saveChatTemplate(btn) {
  const name = document.getElementById('chat-template-name').value.trim();
  const description = document.getElementById('chat-template-description').value.trim();
  const content = document.getElementById('chat-template-content').value;
  if (!name || !content.trim()) {
    toast('Template name and content are required', 'err');
    return;
  }
  const orig = btn?.textContent || 'Save Template';
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Saving...';
  }
  try {
    const method = editingChatTemplateId ? 'PUT' : 'POST';
    const url = editingChatTemplateId
      ? `/api/chat-templates/${encodeURIComponent(editingChatTemplateId)}`
      : '/api/chat-templates';
    const d = await fetchJSON(url, method, { name, description, content });
    if (!d.ok) {
      toast(d.error || 'Could not save template', 'err');
      return;
    }
    editingChatTemplateId = d.id || editingChatTemplateId;
    await loadChatTemplates();
    const list = document.getElementById('chat-template-list');
    if (list) list.value = editingChatTemplateId;
    setChatTemplateStatus(`Saved ${editingChatTemplateId}`);
    toast('Chat template saved', 'ok');
  } catch (e) {
    toast('Could not save template: ' + e, 'err');
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.textContent = orig;
    }
  }
}

async function deleteChatTemplate(btn) {
  if (!editingChatTemplateId) {
    toast('Select a saved template first', 'err');
    return;
  }
  if (!confirm(`Delete chat template "${editingChatTemplateId}"?`)) return;
  const orig = btn?.textContent || 'Delete Template';
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Deleting...';
  }
  try {
    const d = await fetchJSON(`/api/chat-templates/${encodeURIComponent(editingChatTemplateId)}`, 'DELETE');
    if (!d.ok) {
      toast(d.error || 'Could not delete template', 'err');
      return;
    }
    newChatTemplate();
    await loadChatTemplates();
    toast('Chat template deleted', 'ok');
  } catch (e) {
    toast('Could not delete template: ' + e, 'err');
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.textContent = orig;
    }
  }
}

// -- custom models --
function setCustomArgFamilyBadge(familyLabel = '', source = '') {
  const badge = document.getElementById('custom-arg-family-badge');
  if (!badge) return;
  if (!familyLabel) {
    badge.style.display = 'none';
    badge.textContent = '';
    return;
  }
  const suffix = source === 'builtin' ? ' default'
    : source === 'family' ? ' preset'
    : '';
  badge.textContent = `${familyLabel}${suffix}`;
  badge.style.display = '';
}

function getCustomArgValues() {
  return [...document.querySelectorAll('.custom-arg-input')]
    .map(input => input.value.trim())
    .filter(Boolean);
}

function markCustomArgsDirty() {
  customArgsDirty = true;
}

function renderCustomArgRows(values = ['']) {
  const list = document.getElementById('custom-args-list');
  const rows = (values && values.length ? values : ['']).map((value, index) => `
    <div class="custom-arg-row">
      <input type="text" class="custom-arg-input mono" value="${escapeHtml(value)}" placeholder="--jinja or --chat-template-kwargs '{&quot;preserve_thinking&quot;: true}'" oninput="markCustomArgsDirty()">
      <button class="btn btn-ghost btn-sm" type="button" onclick="addCustomArgRow()">+</button>
      <button class="btn btn-ghost btn-sm" type="button" onclick="removeCustomArgRow(this)" ${index === 0 && values.length <= 1 ? 'disabled' : ''}>-</button>
    </div>
  `).join('');
  list.innerHTML = rows;
}

function addCustomArgRow(value = '') {
  const current = getCustomArgValues();
  current.push(value);
  renderCustomArgRows(current);
  markCustomArgsDirty();
}

function removeCustomArgRow(btn) {
  const rows = [...document.querySelectorAll('.custom-arg-row')];
  if (rows.length <= 1) {
    rows[0]?.querySelector('.custom-arg-input')?.focus();
    return;
  }
  btn.closest('.custom-arg-row')?.remove();
  if (document.querySelectorAll('.custom-arg-row').length === 0) {
    renderCustomArgRows(['']);
  } else {
    renderCustomArgRows(getCustomArgValues());
  }
  markCustomArgsDirty();
}

function resetCustomModelForm() {
  document.getElementById('new-display-name').value = '';
  document.getElementById('new-model-name').value = '';
  document.getElementById('new-model-path').value = '';
  document.getElementById('new-mmproj-path').value = '';
  document.getElementById('new-ctx-size').value = '32768';
  document.getElementById('new-model-gguf-select').selectedIndex = 0;
  document.getElementById('new-mmproj-gguf-select').selectedIndex = 0;
  document.getElementById('add-model-title-text').textContent = 'Add Chat Model';
  const addBtn = document.getElementById('custom-model-submit-btn');
  if (addBtn) {
    addBtn.textContent = 'Add Model';
    addBtn.onclick = addModel;
  }
  renderCustomArgRows(['']);
  customArgsDirty = false;
  setCustomArgFamilyBadge();
  resetHuggingFacePicker();
}

async function maybeLoadCustomArgPreset(force = false) {
  const displayName = document.getElementById('new-display-name').value.trim();
  const modelName = document.getElementById('new-model-name').value.trim();
  const modelPath = document.getElementById('new-model-path').value.trim();
  if (!displayName && !modelName && !modelPath) {
    if (!customArgsDirty) {
      renderCustomArgRows(['']);
      setCustomArgFamilyBadge();
    }
    return;
  }
  try {
    const d = await fetchJSON('/api/custom-model-arg-presets/match', 'POST', {
      display_name: displayName,
      model_name: modelName,
      model_path: modelPath,
    });
    setCustomArgFamilyBadge(d.family_label, d.source);
    if (force || !customArgsDirty || getCustomArgValues().length === 0) {
      renderCustomArgRows(d.args && d.args.length ? d.args : ['']);
      customArgsDirty = false;
    }
  } catch (e) {
    console.error('Failed to load custom arg preset:', e);
  }
}

function scheduleCustomArgPresetRefresh() {
  if (customArgPresetTimer) clearTimeout(customArgPresetTimer);
  customArgPresetTimer = setTimeout(() => maybeLoadCustomArgPreset(false), 180);
}

function initCustomModelForm() {
  resetCustomModelForm();
  ['new-display-name', 'new-model-name', 'new-model-path'].forEach(id => {
    const el = document.getElementById(id);
    el?.addEventListener('input', scheduleCustomArgPresetRefresh);
  });
  document.getElementById('hf-model-file-select')?.addEventListener('change', handleHuggingFaceModelSelection);
}

function setHfStatus(message = '', type = '') {
  const el = document.getElementById('hf-status-text');
  if (!el) return;
  el.textContent = message;
  el.className = `hf-status${type ? ' ' + type : ''}`;
}

function resetHuggingFacePicker() {
  huggingFaceRepoFiles = [];
  huggingFaceDownloadJobId = null;
  const sel = document.getElementById('hf-model-file-select');
  if (sel) {
    sel.innerHTML = '<option value="">-- Select model file --</option>';
  }
  const mmprojMatch = document.getElementById('hf-mmproj-match');
  const mmprojRename = document.getElementById('hf-mmproj-rename');
  const downloadBtn = document.getElementById('hf-download-btn');
  const meta = document.getElementById('hf-download-meta');
  if (mmprojMatch) mmprojMatch.value = '';
  if (mmprojRename) mmprojRename.value = '';
  if (downloadBtn) downloadBtn.disabled = true;
  if (meta) meta.innerHTML = '';
  setHfStatus('');
}

async function loadHuggingFaceRepoFiles(btn) {
  const repoUrl = document.getElementById('hf-repo-url').value.trim();
  if (!repoUrl) {
    setHfStatus('Enter a Hugging Face repo URL first', 'err');
    return;
  }
  const orig = btn.textContent;
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>';
  resetHuggingFacePicker();
  try {
    const d = await fetchJSON('/api/huggingface/repo-files', 'POST', { repo_url: repoUrl });
    if (!d.ok) {
      setHfStatus(d.error || 'Failed to load repo files', 'err');
      return;
    }
    populateHuggingFaceRepoFiles(d);
    setHfStatus(huggingFaceRepoFiles.length ? 'Select a GGUF file to download' : 'No non-MMProj GGUF files found in this repo', huggingFaceRepoFiles.length ? 'ok' : 'err');
  } catch (e) {
    setHfStatus(`Failed to load repo files: ${e}`, 'err');
  } finally {
    btn.disabled = false;
    btn.textContent = orig;
  }
}

function populateHuggingFaceRepoFiles(d) {
  huggingFaceRepoFiles = d.model_files || [];
  const sel = document.getElementById('hf-model-file-select');
  sel.innerHTML = '<option value="">-- Select model file --</option>';
  huggingFaceRepoFiles.forEach(file => {
    const opt = document.createElement('option');
    opt.value = file.path;
    const sizePart = file.size ? ` · ${formatBytes(file.size)}` : '';
    opt.textContent = `${file.name}${sizePart}`;
    sel.appendChild(opt);
  });
  document.getElementById('hf-download-meta').innerHTML = [
    `<span class="meta-chip">Repo: ${escapeHtml(d.repo_id || '-')}</span>`,
    `<span class="meta-chip">Revision: ${escapeHtml(d.revision || 'main')}</span>`,
    `<span class="meta-chip">Models: ${escapeHtml(huggingFaceRepoFiles.length)}</span>`,
  ].join('');
}

async function loadHuggingFaceRepoFilesForAdd() {
  const repoUrl = document.getElementById('hf-repo-url').value.trim();
  if (!repoUrl) return false;
  setHfStatus('Loading Hugging Face repo files...', '');
  const d = await fetchJSON('/api/huggingface/repo-files', 'POST', { repo_url: repoUrl });
  if (!d.ok) {
    setHfStatus(d.error || 'Failed to load repo files', 'err');
    return false;
  }
  populateHuggingFaceRepoFiles(d);
  if (huggingFaceRepoFiles.length === 1) {
    const sel = document.getElementById('hf-model-file-select');
    sel.value = huggingFaceRepoFiles[0].path;
    handleHuggingFaceModelSelection();
    return true;
  }
  setHfStatus(huggingFaceRepoFiles.length ? 'Choose a GGUF file, then click Add Model again' : 'No non-MMProj GGUF files found in this repo', huggingFaceRepoFiles.length ? '' : 'err');
  return false;
}

function handleHuggingFaceModelSelection() {
  const selected = document.getElementById('hf-model-file-select').value;
  const file = huggingFaceRepoFiles.find(item => item.path === selected);
  const mmprojMatch = document.getElementById('hf-mmproj-match');
  const mmprojRename = document.getElementById('hf-mmproj-rename');
  const downloadBtn = document.getElementById('hf-download-btn');
  if (!file) {
    if (mmprojMatch) mmprojMatch.value = '';
    if (mmprojRename) mmprojRename.value = '';
    if (downloadBtn) downloadBtn.disabled = true;
    return;
  }
  if (mmprojMatch) mmprojMatch.value = file.matched_mmproj || '';
  if (mmprojRename) mmprojRename.value = file.renamed_mmproj || '';
  if (downloadBtn) downloadBtn.disabled = false;
  fillCustomModelIdentityFromPath(file.name || file.path);
  scheduleCustomArgPresetRefresh();
  setHfStatus(file.matched_mmproj ? 'Model and MMProj ready to download' : 'Model ready to download; no MMProj found', 'ok');
}

function setTranscriptHfStatus(engineId, message = '', type = '') {
  const el = document.getElementById(`transcript-hf-status-text-${engineId}`);
  if (!el) return;
  el.textContent = message;
  el.className = `hf-status${type ? ' ' + type : ''}`;
}

function resetTranscriptHuggingFacePicker(engineId) {
  transcriptRepoInfoByEngine[engineId] = null;
  transcriptDownloadJobIdByEngine[engineId] = null;
  const targetDir = document.getElementById(`transcript-hf-target-dir-${engineId}`);
  const fileCount = document.getElementById(`transcript-hf-file-count-${engineId}`);
  const preview = document.getElementById(`transcript-hf-preview-${engineId}`);
  const downloadBtn = document.getElementById(`transcript-hf-download-btn-${engineId}`);
  const meta = document.getElementById(`transcript-hf-download-meta-${engineId}`);
  if (targetDir) targetDir.value = '';
  if (fileCount) fileCount.value = '';
  if (preview) preview.value = '';
  if (downloadBtn) downloadBtn.disabled = true;
  if (meta) meta.innerHTML = '';
  setTranscriptHfStatus(engineId, '');
}

async function loadTranscriptHuggingFaceRepoInfo(btn, engineId) {
  const repoUrl = document.getElementById(`transcript-hf-repo-url-${engineId}`).value.trim();
  if (!repoUrl) {
    setTranscriptHfStatus(engineId, 'Enter a Hugging Face repo URL first', 'err');
    return;
  }
  const orig = btn.textContent;
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>';
  resetTranscriptHuggingFacePicker(engineId);
  try {
    const d = await fetchJSON('/api/huggingface/transcription-repo-files', 'POST', { repo_url: repoUrl, engine_id: engineId });
    if (!d.ok) {
      setTranscriptHfStatus(engineId, d.error || 'Failed to inspect repo', 'err');
      return;
    }
    transcriptRepoInfoByEngine[engineId] = d;
    document.getElementById(`transcript-hf-target-dir-${engineId}`).value = d.target_dir || '';
    document.getElementById(`transcript-hf-file-count-${engineId}`).value = `${d.file_count || 0} files`;
    document.getElementById(`transcript-hf-preview-${engineId}`).value = (d.sample_files || []).join(', ');
    document.getElementById(`transcript-hf-download-meta-${engineId}`).innerHTML = [
      `<span class="meta-chip">Repo: ${escapeHtml(d.repo_id || '-')}</span>`,
      `<span class="meta-chip">Revision: ${escapeHtml(d.revision || 'main')}</span>`,
      `<span class="meta-chip">Files: ${escapeHtml(d.file_count || 0)}</span>`,
    ].join('');
    document.getElementById(`transcript-hf-download-btn-${engineId}`).disabled = !(d.file_count > 0);
    setTranscriptHfStatus(engineId, d.file_count > 0 ? 'Repo ready to download' : 'No usable files found in this repo', d.file_count > 0 ? 'ok' : 'err');
  } catch (e) {
    setTranscriptHfStatus(engineId, `Failed to inspect repo: ${e}`, 'err');
  } finally {
    btn.disabled = false;
    btn.textContent = orig;
  }
}

async function pollTranscriptHuggingFaceDownload(engineId, jobId) {
  while (transcriptDownloadJobIdByEngine[engineId] === jobId) {
    const d = await fetchJSON(`/api/huggingface/downloads/${jobId}`);
    if (!d.ok || !d.job) {
      setTranscriptHfStatus(engineId, d.error || 'Download job disappeared', 'err');
      transcriptDownloadJobIdByEngine[engineId] = null;
      return null;
    }
    const job = d.job;
    const meta = document.getElementById(`transcript-hf-download-meta-${engineId}`);
    const fileLabel = job.current_file ? escapeHtml(job.current_file) : '-';
    const progressLabel = job.total_bytes
      ? `${formatBytes(job.current_bytes)} / ${formatBytes(job.total_bytes)}`
      : formatBytes(job.current_bytes);
    meta.innerHTML = [
      `<span class="meta-chip">Status: ${escapeHtml(job.status || '-')}</span>`,
      `<span class="meta-chip">Stage: ${escapeHtml(job.stage || '-')}</span>`,
      `<span class="meta-chip">File: ${fileLabel}</span>`,
      `<span class="meta-chip">Transferred: ${escapeHtml(progressLabel)}</span>`,
      job.progress != null ? `<span class="meta-chip">Progress: ${escapeHtml(job.progress)}%</span>` : '',
    ].join('');
    if (job.status === 'done') {
      transcriptDownloadJobIdByEngine[engineId] = null;
      return job;
    }
    if (job.status === 'error') {
      transcriptDownloadJobIdByEngine[engineId] = null;
      setTranscriptHfStatus(engineId, job.error || 'Download failed', 'err');
      return null;
    }
    setTranscriptHfStatus(engineId, `${job.stage || 'Downloading'}${job.current_file ? ` · ${job.current_file}` : ''}`, '');
    await new Promise(resolve => setTimeout(resolve, 1500));
  }
  return null;
}

async function downloadTranscriptHuggingFaceModel(btn, engineId) {
  const repoUrl = document.getElementById(`transcript-hf-repo-url-${engineId}`).value.trim();
  if (!repoUrl || !transcriptRepoInfoByEngine[engineId]) {
    setTranscriptHfStatus(engineId, 'Inspect a repo first', 'err');
    return;
  }
  const orig = btn.textContent;
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Downloading...';
  setTranscriptHfStatus(engineId, 'Starting download...', '');
  try {
    const d = await fetchJSON('/api/huggingface/transcription-downloads', 'POST', { repo_url: repoUrl, engine_id: engineId });
    if (!d.ok || !d.job) {
      setTranscriptHfStatus(engineId, d.error || 'Failed to start download', 'err');
      return;
    }
    transcriptDownloadJobIdByEngine[engineId] = d.job.id;
    const job = await pollTranscriptHuggingFaceDownload(engineId, d.job.id);
    if (!job || !job.result) return;
    await loadTranscriptionModels(engineId);
    const downloadedValue = job.result.model_value || '';
    const inputId = engineId === 'parakeet-v3' ? 'cfg-PARAKEET_V3_LOCAL_MODEL' : 'cfg-WHISPERKIT_LARGE_V3_LOCAL_MODEL';
    const localInput = document.getElementById(inputId);
    const localSelect = document.querySelector(`.transcript-model-select[data-target="${inputId}"]`);
    if (localInput) localInput.value = downloadedValue;
    if (localSelect) localSelect.value = downloadedValue;
    markDirty('Transcription');
    setTranscriptHfStatus(engineId, 'Download complete. This engine model list was refreshed and selected.', 'ok');
    toast(`Downloaded ${job.result.engine_label || 'transcription'} model ${job.result.model_label || ''}`.trim(), 'ok');
  } catch (e) {
    setTranscriptHfStatus(engineId, `Download failed: ${e}`, 'err');
  } finally {
    btn.disabled = false;
    btn.textContent = orig;
  }
}

async function pollHuggingFaceDownload(jobId) {
  while (huggingFaceDownloadJobId === jobId) {
    const d = await fetchJSON(`/api/huggingface/downloads/${jobId}`);
    if (!d.ok || !d.job) {
      setHfStatus(d.error || 'Download job disappeared', 'err');
      huggingFaceDownloadJobId = null;
      return null;
    }
    const job = d.job;
    const meta = document.getElementById('hf-download-meta');
    const fileLabel = job.current_file ? escapeHtml(job.current_file) : escapeHtml(job.model_file || '');
    const progressLabel = job.total_bytes
      ? `${formatBytes(job.current_bytes)} / ${formatBytes(job.total_bytes)}`
      : formatBytes(job.current_bytes);
    meta.innerHTML = [
      `<span class="meta-chip">Status: ${escapeHtml(job.status || '-')}</span>`,
      `<span class="meta-chip">Stage: ${escapeHtml(job.stage || '-')}</span>`,
      `<span class="meta-chip">File: ${fileLabel || '-'}</span>`,
      `<span class="meta-chip">Transferred: ${escapeHtml(progressLabel)}</span>`,
      job.progress != null ? `<span class="meta-chip">Progress: ${escapeHtml(job.progress)}%</span>` : '',
    ].join('');
    if (job.status === 'done') {
      huggingFaceDownloadJobId = null;
      return job;
    }
    if (job.status === 'error') {
      huggingFaceDownloadJobId = null;
      setHfStatus(job.error || 'Download failed', 'err');
      return null;
    }
    setHfStatus(`${job.stage || 'Downloading'}${job.current_file ? ` · ${job.current_file}` : ''}`, '');
    await new Promise(resolve => setTimeout(resolve, 1500));
  }
  return null;
}

async function downloadSelectedHuggingFaceModel(btn) {
  const repoUrl = document.getElementById('hf-repo-url').value.trim();
  const modelFile = document.getElementById('hf-model-file-select').value;
  const match = huggingFaceRepoFiles.find(item => item.path === modelFile);
  if (!repoUrl || !modelFile || !match) {
    setHfStatus('Load a repo and select a model file first', 'err');
    return;
  }
  const orig = btn.textContent;
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Downloading...';
  setHfStatus('Starting download...', '');
  try {
    const d = await fetchJSON('/api/huggingface/downloads', 'POST', {
      repo_url: repoUrl,
      model_file: modelFile,
      mmproj_file: match.matched_mmproj || '',
    });
    if (!d.ok || !d.job) {
      setHfStatus(d.error || 'Failed to start download', 'err');
      return;
    }
    huggingFaceDownloadJobId = d.job.id;
    const job = await pollHuggingFaceDownload(d.job.id);
    if (!job || !job.result) return;

    await loadGgufFiles();

    const downloadedModelPath = job.result.model_path || '';
    const downloadedMmprojPath = job.result.mmproj_path || '';
    const downloadedName = job.result.model_name || filenameFromPath(downloadedModelPath);

    document.getElementById('new-model-path').value = downloadedModelPath;
    document.getElementById('new-mmproj-path').value = downloadedMmprojPath;
    fillCustomModelIdentityFromPath(downloadedName || downloadedModelPath);
    selectGgufOption('new-model-gguf-select', downloadedModelPath);
    selectGgufOption('new-mmproj-gguf-select', downloadedMmprojPath);
    setHfStatus('Download complete. The form below was populated with the new local paths.', 'ok');
    toast(`Downloaded ${downloadedName}`, 'ok');
  } catch (e) {
    setHfStatus(`Download failed: ${e}`, 'err');
  } finally {
    btn.disabled = false;
    btn.textContent = orig;
  }
}

async function ensureSelectedHuggingFaceModelDownloaded() {
  const existingPath = document.getElementById('new-model-path').value.trim();
  if (existingPath) {
    ensureCustomModelIdentity();
    return true;
  }

  const repoUrl = document.getElementById('hf-repo-url').value.trim();
  let modelFile = document.getElementById('hf-model-file-select').value;
  if (repoUrl && (!huggingFaceRepoFiles.length || !modelFile)) {
    const ready = await loadHuggingFaceRepoFilesForAdd();
    modelFile = document.getElementById('hf-model-file-select').value;
    if (!ready && !modelFile) {
      return false;
    }
  }
  const match = huggingFaceRepoFiles.find(item => item.path === modelFile);
  if (!repoUrl || !modelFile || !match) {
    return false;
  }

  ensureCustomModelIdentity();
  setHfStatus('Downloading selected model before adding it...', '');
  const d = await fetchJSON('/api/huggingface/downloads', 'POST', {
    repo_url: repoUrl,
    model_file: modelFile,
    mmproj_file: match.matched_mmproj || '',
  });
  if (!d.ok || !d.job) {
    throw new Error(d.error || 'Failed to start download');
  }

  huggingFaceDownloadJobId = d.job.id;
  const job = await pollHuggingFaceDownload(d.job.id);
  if (!job || !job.result) {
    throw new Error('Download did not complete');
  }

  await loadGgufFiles();

  const downloadedModelPath = job.result.model_path || '';
  const downloadedMmprojPath = job.result.mmproj_path || '';
  const downloadedName = job.result.model_name || filenameFromPath(downloadedModelPath);

  document.getElementById('new-model-path').value = downloadedModelPath;
  document.getElementById('new-mmproj-path').value = downloadedMmprojPath;
  fillCustomModelIdentityFromPath(downloadedName || downloadedModelPath);
  selectGgufOption('new-model-gguf-select', downloadedModelPath);
  selectGgufOption('new-mmproj-gguf-select', downloadedMmprojPath);
  await maybeLoadCustomArgPreset(false);
  setHfStatus('Download complete. Adding model...', 'ok');
  return true;
}

async function loadCustomModels() {
  try {
    customModels = await fetchJSON('/api/custom-models');
    renderCustomModels();
  } catch (e) { console.error('Failed to load custom models:', e); }
}

function renderCustomModels() {
  const group = document.getElementById('group-custom-models');
  const container = document.getElementById('custom-models-cards');
  const switchContainer = document.getElementById('custom-switch-btns');

  // Render switch buttons in quick bar
  switchContainer.innerHTML = customModels.map(m =>
    `<button class="btn btn-ghost btn-sm switch-model-btn" data-variant="${m.id}" onclick="switchModel(this,'${m.id}')">Switch to ${m.display_name}</button>`
  ).join('');

  if (customModels.length === 0) {
    group.style.display = 'none';
    return;
  }
  group.style.display = '';

  container.innerHTML = customModels.map(m => `
    <div class="svc-card" id="custom-card-${m.id}" data-status="inactive">
      <div class="card-top">
        <span class="card-drag-handle" title="Drag to reorder card">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="8" y1="6" x2="21" y2="6"></line><line x1="8" y1="12" x2="21" y2="12"></line><line x1="8" y1="18" x2="21" y2="18"></line><line x1="3" y1="6" x2="3.01" y2="6"></line><line x1="3" y1="12" x2="3.01" y2="12"></line><line x1="3" y1="18" x2="3.01" y2="18"></line></svg>
        </span>
        <span class="card-name">${m.display_name}</span>
        <span class="status-pill inactive">inactive</span>
      </div>
      <div class="card-desc">${m.model_name} | ctx ${m.ctx_size}</div>
      <div class="card-port" title="${m.model_path}">${m.model_path.split('/').pop()}</div>
      <div class="meta-list">
        ${m.arg_family_label ? `<span class="meta-chip">family: ${escapeHtml(m.arg_family_label)}</span>` : ''}
        ${(m.resolved_custom_args || []).length ? `<span class="meta-chip">args: ${m.resolved_custom_args.length}</span>` : '<span class="meta-chip">args: none</span>'}
      </div>
      <div class="card-acts">
        <button class="btn btn-success btn-sm" onclick="switchModel(this,'${m.id}')">Switch To</button>
        <button class="btn btn-ghost btn-sm" onclick="editCustomModel('${m.id}')">Edit</button>
        <button class="btn btn-danger btn-sm" onclick="removeCustomModel('${m.id}')">Remove</button>
      </div>
    </div>
  `).join('');
}

function toggleAddModel() {
  const panel = document.getElementById('add-model-panel');
  panel.classList.toggle('open');
  if (panel.classList.contains('open')) {
    panel.classList.remove('collapsed');
    loadGgufFiles();
    maybeLoadCustomArgPreset(false);
  } else {
    resetCustomModelForm();
  }
}

async function addModel() {
  const addBtn = document.getElementById('custom-model-submit-btn');
  const orig = addBtn?.textContent || 'Add Model';
  if (addBtn) {
    addBtn.disabled = true;
    addBtn.innerHTML = '<span class="spinner"></span> Adding...';
  }

  try {
    const hasModelPath = await ensureSelectedHuggingFaceModelDownloaded();
    if (!hasModelPath) {
      toast('Select a local GGUF or load a Hugging Face repo and choose a model file', 'err');
      return;
    }

    const identity = ensureCustomModelIdentity();
    const displayName = identity.displayName;
    const modelName = identity.modelName;
    const modelPath = document.getElementById('new-model-path').value.trim();
    const mmprojPath = document.getElementById('new-mmproj-path').value.trim();
    const ctxSize = document.getElementById('new-ctx-size').value.trim();
    const customArgs = getCustomArgValues();

    if (!modelPath) {
      toast('Model GGUF is required', 'err');
      return;
    }

    const d = await fetchJSON('/api/custom-models', 'POST', {
      display_name: displayName,
      model_name: modelName || undefined,
      model_path: modelPath,
      mmproj_path: mmprojPath,
      ctx_size: ctxSize || '32768',
      custom_args: customArgs,
    });
    if (d.ok) {
      toast(`Added model: ${displayName}`, 'ok');
      toggleAddModel();
      await loadCustomModels();
    } else {
      toast('Failed: ' + (d.error || 'unknown'), 'err');
    }
  } catch (e) {
    setHfStatus(`Download/add failed: ${e}`, 'err');
    toast('Error: ' + e, 'err');
  } finally {
    if (addBtn) {
      addBtn.disabled = false;
      addBtn.textContent = orig;
    }
  }
}

async function removeCustomModel(id) {
  const m = customModels.find(m => m.id === id);
  if (!confirm(`Remove "${m ? m.display_name : id}"?`)) return;
  try {
    await fetchJSON(`/api/custom-models/${id}`, 'DELETE');
    toast('Model removed', 'ok');
    await loadCustomModels();
  } catch (e) { toast('Error: ' + e, 'err'); }
}

function editCustomModel(id) {
  const m = customModels.find(m => m.id === id);
  if (!m) return;

  // Populate the add-model form with existing values for editing
  document.getElementById('new-display-name').value = m.display_name;
  document.getElementById('new-model-name').value = m.model_name;
  document.getElementById('new-model-path').value = m.model_path;
  document.getElementById('new-mmproj-path').value = m.mmproj_path || '';
  document.getElementById('new-ctx-size').value = m.ctx_size || '32768';
  renderCustomArgRows((m.resolved_custom_args || m.custom_args || []).length ? (m.resolved_custom_args || m.custom_args) : ['']);
  customArgsDirty = false;
  setCustomArgFamilyBadge(m.arg_family_label, m.custom_arg_source);

  // Open panel and change button to update
  const panel = document.getElementById('add-model-panel');
  panel.classList.add('open');
  panel.classList.remove('collapsed');
  document.getElementById('add-model-title-text').textContent = 'Edit Chat Model';
  loadGgufFiles().then(() => {
    // Select matching options
    selectGgufOption('new-model-gguf-select', m.model_path);
    selectGgufOption('new-mmproj-gguf-select', m.mmproj_path);
  });

  // Replace Add button with Update
  const addBtn = document.getElementById('custom-model-submit-btn');
  addBtn.textContent = 'Update Model';
  addBtn.onclick = async function() {
    const data = {
      display_name: document.getElementById('new-display-name').value.trim(),
      model_name: document.getElementById('new-model-name').value.trim(),
      model_path: document.getElementById('new-model-path').value.trim(),
      mmproj_path: document.getElementById('new-mmproj-path').value.trim(),
      ctx_size: document.getElementById('new-ctx-size').value.trim() || '32768',
      custom_args: getCustomArgValues(),
    };
    try {
      const d = await fetchJSON(`/api/custom-models/${id}`, 'PUT', data);
      if (d.ok) {
        toast('Model updated', 'ok');
        toggleAddModel();
        // Reset button
        addBtn.textContent = 'Add Model';
        addBtn.onclick = addModel;
        document.getElementById('add-model-title-text').textContent = 'Add Chat Model';
        await loadCustomModels();
      } else {
        toast('Failed: ' + (d.error || 'unknown'), 'err');
      }
    } catch (e) { toast('Error: ' + e, 'err'); }
  };
}

function selectGgufOption(selectId, path) {
  const sel = document.getElementById(selectId);
  if (!sel || !path) return;
  for (let i = 0; i < sel.options.length; i++) {
    if (sel.options[i].value === path) {
      sel.selectedIndex = i;
      return;
    }
  }
}
