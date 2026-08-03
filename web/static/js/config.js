// config.js
//
// The configuration form: loading it, tracking which sections are dirty, the
// per-slot context read-out, the pre-flight budget check, and saved profiles.

// -- config --
async function loadConfig() {
  try {
    if (!ggufFiles.length) await loadGgufFiles();
    if (!chatTemplates.length) await loadChatTemplates();
    if (!Object.keys(transcriptionModelsByEngine).length) {
      await loadTranscriptionModels('parakeet-v3');
      await loadTranscriptionModels('whisperkit-large-v3');
    }
    if (!Object.keys(transcriptionCapabilities).length) await loadTranscriptionCapabilities();
    cfgCurrent = await fetchJSON('/api/config');
    for (const [key, val] of Object.entries(cfgCurrent)) {
      const el = document.getElementById('cfg-' + key);
      if (el) {
        el.value = val;
        // Also update the corresponding GGUF dropdown
        const sel = document.querySelector(`.gguf-path-select[data-target="cfg-${key}"]`);
        if (sel) selectGgufOption(sel.id || '', val);
      }
    }
    refreshChatTemplateSelects();
    document.querySelectorAll('.chat-template-select[name]').forEach(sel => {
      const key = sel.name;
      sel.value = cfgCurrent[key] || '';
    });
    // Sync gguf selects with current values
    document.querySelectorAll('.gguf-path-select').forEach(sel => {
      const targetId = sel.dataset.target;
      const input = document.getElementById(targetId);
      if (input) {
        for (let i = 0; i < sel.options.length; i++) {
          if (sel.options[i].value === input.value) {
            sel.selectedIndex = i;
            break;
          }
        }
      }
    });
    document.querySelectorAll('.transcript-model-select').forEach(sel => {
      const targetId = sel.dataset.target;
      const input = document.getElementById(targetId);
      if (input) {
        for (let i = 0; i < sel.options.length; i++) {
          if (sel.options[i].value === input.value) {
            sel.selectedIndex = i;
            break;
          }
        }
      }
    });
    initConfigArgEditors();
    initCacheAwareScheduling();
    refreshAllCacheAwareScheduling();
    initPerSlotHints();
    syncQuickGpuPlacementSelects();
    refreshShadowedAliasHints();
  } catch (e) { toast('Could not load config: ' + e, 'err'); }
}

// A legacy env key sitting in the file next to its replacement is read by
// nothing — `normalize_env_keys` only backfills when the canonical key is
// absent, and every start script resolves the canonical name first. But
// `/api/config` reports both, with their independent on-disk values, and
// nothing distinguished the live one from the dead one. On this host the dead
// `CHAT_DENSE_MODEL_PATH` names a different model file than the running
// `CHAT_PRIMARY_MODEL_PATH`, and `CHAT_DENSE_CTX_SIZE=32768` sits beside a live
// 262144 — a reader of the config page had no way to tell.
async function refreshShadowedAliasHints() {
  document.querySelectorAll('[data-shadowed-key]').forEach(hint => {
    hint.hidden = true;
    hint.textContent = '';
  });
  let deprecations;
  try {
    deprecations = await fetchJSON('/api/config/deprecations');
  } catch (e) {
    return; // A missing annotation must never take the config form with it.
  }
  const entries = deprecations?.env_keys || [];
  for (const entry of entries) {
    // Without a canonical twin the legacy key is still the live value, backfilled
    // on read. That is worth saying too, but it is not the same warning.
    const target = entry.canonical_present ? entry.replacement : entry.key;
    const hint = document.querySelector(`[data-shadowed-key="${CSS.escape(target)}"]`);
    if (!hint) continue;
    hint.hidden = false;
    hint.innerHTML = entry.canonical_present
      ? `Ignored: <code>${escapeHtml(entry.key)}=${escapeHtml(entry.value)}</code> is still in the env file but this field wins.`
      : `Read from the legacy <code>${escapeHtml(entry.key)}</code>; saving this field writes <code>${escapeHtml(entry.replacement)}</code>.`;
  }
  renderShadowedAliasBanner(entries, deprecations?.saved_configs || []);
}

// Not every legacy key has a field to sit under — the CHAT_SECONDARY_* controls
// are never generated (config_fields.py builds the "primary" variant only), so
// five of the ten on this host would otherwise stay invisible.
function renderShadowedAliasBanner(entries, savedConfigs) {
  const banner = document.getElementById('shadowed-alias-banner');
  if (!banner) return;
  if (!entries.length) { banner.hidden = true; banner.innerHTML = ''; return; }
  const rows = entries.map(entry => {
    const fate = entry.canonical_present
      ? `ignored — <code>${escapeHtml(entry.replacement)}</code> wins`
      : `still live, read in place of <code>${escapeHtml(entry.replacement)}</code>`;
    return `<li><code>${escapeHtml(entry.key)}=${escapeHtml(entry.value)}</code> — ${fate}</li>`;
  }).join('');
  const profiles = savedConfigs.length
    ? `<p>Saved profiles also carry these and are left alone: ${
        savedConfigs.map(p => `<code>${escapeHtml(p.name)}</code>`).join(', ')}.</p>`
    : '';
  banner.innerHTML =
    `<strong>${entries.length} legacy config key${entries.length === 1 ? '' : 's'} in the env file</strong>` +
    `<ul>${rows}</ul>${profiles}` +
    `<button class="btn btn-ghost btn-sm" onclick="migrateShadowedAliases()">Rewrite to canonical keys</button>`;
  banner.hidden = false;
}

async function migrateShadowedAliases() {
  try {
    // fetchJSON resolves on a 500 too, so the body is the only error signal.
    const result = await fetchJSON('/api/config/deprecations/migrate', 'POST');
    if (result.ok === false) { toast('Could not migrate legacy keys: ' + (result.error || 'unknown'), 'err'); return; }
    const migrated = result.migrated?.length ?? 0;
    toast(`Rewrote ${migrated} key${migrated === 1 ? '' : 's'} to canonical names`, 'ok');
    await loadConfig();
  } catch (e) { toast('Could not migrate legacy keys: ' + e, 'err'); }
}

function markDirty(section) { cfgDirty[section] = true; }

const CACHE_AWARE_SUFFIXES = [
  'N_PARALLEL',
  'CTX_SIZE',
  'CACHE_RAM',
  'CTX_CHECKPOINTS',
  'CACHE_IDLE_SLOTS',
  'FIT',
  'CUSTOM_ARGS_JSON',
];

function cacheAwareValues(prefix) {
  return Object.fromEntries(CACHE_AWARE_SUFFIXES.map(suffix => {
    const input = document.getElementById(`cfg-${prefix}_${suffix}`);
    return [suffix, input?.value || ''];
  }));
}

// Host measurements from /api/backend/budget/recommend, keyed by config
// prefix. The recommended profile is derived from detected VRAM, host RAM and
// the selected model's geometry rather than from a constant, so it has to be
// fetched; until it arrives the panel reports compatibility only.
const cacheAwareRecommendations = {};
const CACHE_AWARE_BACKENDS = { CHAT_PRIMARY: 'chat-primary', CHAT2: 'chat-secondary' };

async function loadCacheAwareRecommendation(prefix) {
  const backend = CACHE_AWARE_BACKENDS[prefix];
  if (!backend) return null;
  try {
    const payload = await fetchJSON(`/api/backend/budget/recommend?backend=${backend}`);
    cacheAwareRecommendations[prefix] = payload && !payload.error ? payload : null;
  } catch (_e) {
    cacheAwareRecommendations[prefix] = null;
  }
  return cacheAwareRecommendations[prefix];
}

function refreshCacheAwareScheduling(prefix) {
  const status = document.getElementById(`cache-aware-status-${prefix}`);
  if (!status || typeof CacheAwareScheduling === 'undefined') return;
  const recommendation = cacheAwareRecommendations[prefix];
  const result = CacheAwareScheduling.evaluate(cacheAwareValues(prefix), recommendation);
  const total = result.totalContext.toLocaleString('en-US');
  const perSlot = result.perSlotContext.toLocaleString('en-US');
  let headline = '';

  status.classList.remove('compatible', 'warning', 'incompatible');
  if (!result.compatible) {
    status.classList.add('incompatible');
    headline = '<strong>Incompatible with cache-aware scheduling.</strong>';
  } else if (!result.hasRecommendation) {
    status.classList.add(result.conflicts.length ? 'warning' : 'compatible');
    headline = '<strong>Compatible.</strong> This host has not been measured — the model file could not be read, so no context profile is recommended.';
  } else if (result.recommended && !result.notes.length) {
    status.classList.add(result.conflicts.length ? 'warning' : 'compatible');
    headline = '<strong>Compatible — the profile measured to fit this host is configured.</strong>';
  } else {
    status.classList.add('warning');
    headline = `<strong>Compatible — this host was measured to fit ${result.recommendedTotalContext.toLocaleString('en-US')} total tokens.</strong>`;
  }

  // The per-slot figure is the one a request is measured against, so it leads.
  const contextSummary = `<div><strong>${perSlot}</strong> tokens per slot × ${result.slots || 0} slots = ${total} total. A request above the per-slot figure is rejected.</div>`;
  const issues = result.issues.length
    ? `<ul>${result.issues.map(issue => `<li>${escapeHtml(issue)}</li>`).join('')}</ul>`
    : '';
  const notes = result.notes.map(note => `<div class="cache-aware-note">${escapeHtml(note)}</div>`).join('');
  const conflicts = result.conflicts.length
    ? `<div style="margin-top:6px;"><strong>Custom argument warning:</strong> these later arguments may override the structured settings: ${result.conflicts.map(escapeHtml).join(', ')}</div>`
    : '';
  status.innerHTML = `${headline}${contextSummary}${issues}${notes}${conflicts}`;
}

function refreshAllCacheAwareScheduling() {
  refreshCacheAwareScheduling('CHAT_PRIMARY');
  refreshCacheAwareScheduling('CHAT2');
  // Measuring the host means reading GGUF metadata off disk, so it trails the
  // form rather than blocking it.
  for (const prefix of Object.keys(CACHE_AWARE_BACKENDS)) {
    loadCacheAwareRecommendation(prefix).then(() => refreshCacheAwareScheduling(prefix));
  }
  // Checks the running backend rather than the form, so it trails too.
  refreshSchedulingVerify();
}

function initCacheAwareScheduling() {
  for (const prefix of ['CHAT_PRIMARY', 'CHAT2']) {
    for (const suffix of CACHE_AWARE_SUFFIXES) {
      const input = document.getElementById(`cfg-${prefix}_${suffix}`);
      if (!input || input.dataset.cacheAwareListener === '1') continue;
      input.dataset.cacheAwareListener = '1';
      input.addEventListener('input', () => refreshCacheAwareScheduling(prefix));
      input.addEventListener('change', () => refreshCacheAwareScheduling(prefix));
    }
  }
}

async function applyCacheAwarePreset(prefix, section) {
  if (typeof CacheAwareScheduling === 'undefined') return;
  const recommendation = cacheAwareRecommendations[prefix] ?? await loadCacheAwareRecommendation(prefix);
  const updates = CacheAwareScheduling.presetValues(prefix, recommendation);
  for (const [key, value] of Object.entries(updates)) {
    const input = document.getElementById(`cfg-${key}`);
    if (input) input.value = value;
  }
  markDirty(section);
  refreshCacheAwareScheduling(prefix);
  refreshPerSlotHints();
  const perSlot = (Number(recommendation?.per_slot_context) || 0).toLocaleString('en-US');
  toast(recommendation
    ? `Applied the profile measured for this host: ${perSlot} tokens per slot. Save and restart the backend to activate it.`
    : `Applied fallback scheduling values to ${section} — this host could not be measured. Save and restart the backend to activate it.`,
    recommendation ? 'ok' : 'info');
}

// -- per-slot context and pre-flight --
// Every field whose value llama.cpp divides by --parallel gets a live per-slot
// readout beside it. The total is what is configured; the per-slot figure is
// what a request is measured against, and only one of the two used to be shown.
function refreshPerSlotHints() {
  document.querySelectorAll('[data-per-slot-prefix]').forEach(hint => {
    const prefix = hint.dataset.perSlotPrefix;
    const total = Number.parseInt(document.getElementById(`cfg-${prefix}_CTX_SIZE`)?.value || '', 10);
    const slots = Number.parseInt(document.getElementById(`cfg-${prefix}_N_PARALLEL`)?.value || '', 10);
    if (!Number.isFinite(total) || total <= 0) { hint.textContent = ''; return; }
    const divisor = Number.isFinite(slots) && slots > 0 ? slots : 1;
    hint.innerHTML = `llama.cpp divides this across slots: <strong>${Math.floor(total / divisor).toLocaleString('en-US')}</strong> tokens per slot × ${divisor}. A request larger than the per-slot figure is rejected.`;
  });
}

function initPerSlotHints() {
  for (const prefix of Object.keys(CACHE_AWARE_BACKENDS)) {
    for (const suffix of ['CTX_SIZE', 'N_PARALLEL']) {
      const input = document.getElementById(`cfg-${prefix}_${suffix}`);
      if (!input || input.dataset.perSlotListener === '1') continue;
      input.dataset.perSlotListener = '1';
      input.addEventListener('input', refreshPerSlotHints);
      input.addEventListener('change', refreshPerSlotHints);
    }
  }
  refreshPerSlotHints();
}

function showPreflight(section, preflight) {
  const panel = document.getElementById('preflight-' + section.replace(/ /g, '-'));
  if (!panel) return;
  if (!preflight || !(preflight.backends || []).length) {
    panel.style.display = 'none';
    panel.innerHTML = '';
    return;
  }

  const parts = [];
  for (const backend of preflight.backends) {
    if (backend.error) {
      parts.push(`<div class="preflight-issue info">${escapeHtml(backend.prefix)}: ${escapeHtml(backend.error)} — the configuration could not be priced.</div>`);
      continue;
    }
    if (Number.isFinite(backend.per_slot_context)) {
      parts.push(`<div class="preflight-ctx">${escapeHtml(backend.prefix)}: ${backend.per_slot_context.toLocaleString('en-US')} tokens per slot × ${backend.slots} = ${backend.total_context.toLocaleString('en-US')} total · predicted peak ${(backend.vram_upper_mib || 0).toLocaleString('en-US')} MiB VRAM</div>`);
    }
    for (const issue of backend.issues || []) {
      parts.push(`<div class="preflight-issue ${escapeHtml(issue.level)}">${escapeHtml(issue.text)}</div>`);
    }
  }
  panel.innerHTML = parts.join('');
  panel.style.display = parts.length ? 'grid' : 'none';
}

// -- saved configurations --
async function patchSelectedSavedConfig(section, updates) {
  const sel = document.getElementById('saved-config-select');
  const name = sel?.value || '';
  if (!name) return;
  try {
    const d = await fetchJSON(`/api/saved-configs/${name}/patch`, 'POST', { updates });
    if (d.ok) {
      await loadSavedConfigsList();
      toast(`Updated saved profile "${name}" with ${section}`, 'ok');
    } else if (d.error !== 'Config not found') {
      toast(`Saved ${section}, but profile update failed: ${d.error || 'unknown'}`, 'err');
    }
  } catch (e) {
    toast(`Saved ${section}, but profile update failed: ${e}`, 'err');
  }
}

function savedConfigModelLabel(config) {
  const slots = config?.active_backend_slots || {};
  const labels = [];
  if (slots.primary?.variant && slots.primary?.label) labels.push(`primary: ${slots.primary.label}`);
  if (slots.secondary?.variant && slots.secondary?.label) labels.push(`secondary: ${slots.secondary.label}`);
  if (labels.length) return labels.join(', ');
  const active = config?.active_chat_model || {};
  if (!active || !active.variant) return '';
  if (active.label) return active.label;
  if (active.kind === 'custom') return `Custom: ${active.variant}`;
  return active.variant;
}

function updateSavedConfigMeta() {
  const sel = document.getElementById('saved-config-select');
  const meta = document.getElementById('saved-config-meta');
  if (!sel || !meta) return;
  const item = savedConfigs.find(c => c.name === sel.value);
  if (!item) {
    meta.textContent = '';
    return;
  }
  const bits = [];
  if (item.is_default) bits.push('default');
  const model = savedConfigModelLabel(item);
  if (model) bits.push(`model: ${model}`);
  meta.textContent = bits.join(' | ');
}

async function loadSavedConfigsList() {
  try {
    savedConfigs = await fetchJSON('/api/saved-configs');
    const sel = document.getElementById('saved-config-select');
    const previous = sel.value;
    while (sel.options.length > 1) sel.remove(1);
    savedConfigs.forEach(c => {
      const opt = document.createElement('option');
      opt.value = c.name;
      const model = savedConfigModelLabel(c);
      opt.textContent = `${c.is_default ? '* ' : ''}${c.display_name}${model ? ` [${model}]` : ''}${c.description ? ` - ${c.description}` : ''}`;
      sel.appendChild(opt);
    });
    if (previous && savedConfigs.some(c => c.name === previous)) sel.value = previous;
    updateSavedConfigMeta();
  } catch (e) { console.error('Failed to load saved configs:', e); }
}

async function saveCurrentConfig() {
  const name = document.getElementById('save-config-name').value.trim();
  if (!name) { toast('Enter a config name first', 'err'); return; }
  try {
    const d = await fetchJSON('/api/saved-configs', 'POST', {
      name,
      config: collectConfigFormValues(),
    });
    if (d.ok) {
      toast(`Saved config: ${name}`, 'ok');
      document.getElementById('save-config-name').value = '';
      await loadSavedConfigsList();
      document.getElementById('saved-config-select').value = d.name || name;
      updateSavedConfigMeta();
    } else {
      toast('Failed: ' + (d.error || 'unknown'), 'err');
    }
  } catch (e) { toast('Error: ' + e, 'err'); }
}

function collectConfigFormValues() {
  document.querySelectorAll('.custom-args-list[data-config-target]').forEach(list => {
    syncConfigArgInput(list.dataset.configTarget, true);
  });
  const values = {};
  document.querySelectorAll('#tab-config [name]').forEach(el => {
    values[el.name] = el.value;
  });
  return values;
}

async function loadSavedConfig(launch = false) {
  const sel = document.getElementById('saved-config-select');
  const name = sel.value;
  if (!name) { toast('Select a config to load', 'err'); return; }

  const action = launch ? 'apply and launch' : 'apply';
  if (!confirm(`${action.charAt(0).toUpperCase() + action.slice(1)} saved config "${name}"? This will overwrite current settings.`)) return;

  try {
    const d = await fetchJSON(`/api/saved-configs/${name}/apply`, 'POST', { launch });
    if (d.ok) {
      toast(launch ? 'Config applied and backend launch requested.' : 'Config applied. Reloading values...', 'ok');
      await loadConfig();
      await pollActiveModel();
      await loadSavedConfigsList();
      if (d.restart_needed && d.restart_needed.length) {
        toast('Restart needed: ' + d.restart_needed.join(', '), 'info');
      }
    } else {
      toast('Failed: ' + (d.error || 'unknown'), 'err');
    }
  } catch (e) { toast('Error: ' + e, 'err'); }
}

async function setDefaultSavedConfig() {
  const sel = document.getElementById('saved-config-select');
  const name = sel.value;
  if (!name) { toast('Select a config first', 'err'); return; }
  try {
    const d = await fetchJSON(`/api/saved-configs/${name}/default`, 'POST');
    if (d.ok) {
      toast(`Default config set: ${name}`, 'ok');
      await loadSavedConfigsList();
    } else {
      toast('Failed: ' + (d.error || 'unknown'), 'err');
    }
  } catch (e) { toast('Error: ' + e, 'err'); }
}

async function clearDefaultSavedConfig() {
  const sel = document.getElementById('saved-config-select');
  const name = sel.value;
  if (!name) { toast('Select a config first', 'err'); return; }
  try {
    const d = await fetchJSON(`/api/saved-configs/${name}/default`, 'DELETE');
    if (d.ok) {
      toast('Default config cleared', 'ok');
      await loadSavedConfigsList();
    } else {
      toast('Failed: ' + (d.error || 'unknown'), 'err');
    }
  } catch (e) { toast('Error: ' + e, 'err'); }
}

async function deleteSavedConfig() {
  const sel = document.getElementById('saved-config-select');
  const name = sel.value;
  if (!name) { toast('Select a config to delete', 'err'); return; }
  if (!confirm(`Delete saved config "${name}"?`)) return;

  try {
    await fetchJSON(`/api/saved-configs/${name}`, 'DELETE');
    toast('Config deleted', 'ok');
    await loadSavedConfigsList();
  } catch (e) { toast('Error: ' + e, 'err'); }
}
