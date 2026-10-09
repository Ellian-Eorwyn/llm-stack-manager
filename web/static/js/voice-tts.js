// voice-tts.js
//
// Hermes's voice server: its unit, what its /health reports, the default voice
// (kept in the voices file, live without a restart), and a spoken test.

let voiceTtsStatus = null;

function initVoiceTtsTab() {
  if (!voiceTtsStatus) loadVoiceTtsStatus();
}

async function loadVoiceTtsStatus(silent = false) {
  try {
    voiceTtsStatus = await fetchJSON('/api/voice-tts/status');
    renderVoiceTtsStatus();
  } catch (e) {
    if (!silent) toast('Voice TTS status failed: ' + e, 'err');
  }
}

function voiceTtsChips(items) {
  return items.map(item => `<span class="meta-chip">${escapeHtml(item)}</span>`).join('');
}

function renderVoiceTtsStatus() {
  const d = voiceTtsStatus;
  if (!d) return;
  const cfg = d.config || {};
  const server = d.server || null;
  document.getElementById('voice-tts-last-refresh').textContent = `Last refresh: ${fmtEpoch(d.last_refresh)}`;
  document.getElementById('voice-tts-service-status').textContent = d.service_status || '-';
  document.getElementById('voice-tts-url').textContent = cfg.public_url || '-';
  document.getElementById('voice-tts-meta-list').innerHTML = voiceTtsChips([
    `Enabled: ${cfg.enabled || '-'}`,
    `Host: ${cfg.host || '-'}`,
    `Port: ${cfg.port || '-'}`,
    `GPU: ${cfg.gpu || '-'}`,
    `Language: ${cfg.language || '-'}`,
    `Voices file: ${cfg.voices_file || '-'}`,
  ]);

  const stats = [];
  if (server) {
    const models = Object.entries(server.models || {}).map(([kind, id]) => `${kind}: ${id}`);
    stats.push(...models.map(m => `Model ${m}`));
    if (server.loaded_at) stats.push(`Loaded: ${server.loaded_at}`);
    if (server.gpu_gb != null) stats.push(`GPU memory: ${server.gpu_gb} GB`);
    stats.push(`Served: ${server.served ?? 0}`, `Capped: ${server.capped ?? 0}`, `Cancelled: ${server.cancelled ?? 0}`);
    if (server.busy) stats.push('Busy');
    const last = server.last;
    if (last) stats.push(`Last: ${last.at} · ${last.chars} chars · first audio ${last.first_audio_ms} ms · ${last.audio_s} s`);
  } else {
    stats.push('Server not answering');
  }
  document.getElementById('voice-tts-stats-list').innerHTML = voiceTtsChips(stats);

  // Voices: the file is the source of truth for what can be the default; the
  // running server says which of them actually loaded.
  const fileVoices = (d.voices && d.voices.voices) || {};
  const loaded = new Set((server && server.voices) || []);
  const skipped = (server && server.skipped) || {};
  const fileDefault = (d.voices && d.voices.default) || '';
  const names = Object.keys(fileVoices).sort();
  const select = document.getElementById('voice-tts-default-select');
  select.innerHTML = names.map(n =>
    `<option value="${escapeHtml(n)}" ${n === fileDefault ? 'selected' : ''}>${escapeHtml(n)}</option>`).join('');
  document.getElementById('voice-tts-voice-list').innerHTML = names.length ? names.map(n => {
    let state = server ? (loaded.has(n) ? 'loaded' : (skipped[n] ? `skipped: ${skipped[n]}` : 'not loaded, restart')) : '';
    const parts = [n, fileVoices[n]];
    if (n === fileDefault) parts.push('default');
    if (state) parts.push(state);
    return `<span class="meta-chip">${escapeHtml(parts.join(' · '))}</span>`;
  }).join('') : voiceTtsChips([d.checks?.voices_file?.error || 'No voices file']);

  const testSelect = document.getElementById('voice-tts-test-voice');
  const current = testSelect.value;
  const testVoices = ['default', ...(server ? [...loaded].sort() : names)];
  testSelect.innerHTML = testVoices.map(n =>
    `<option value="${escapeHtml(n)}" ${n === current ? 'selected' : ''}>${escapeHtml(n)}</option>`).join('');

  document.getElementById('voice-tts-status-cards').innerHTML = Object.entries(d.checks || {}).map(([key, v]) => {
    const ok = v?.ok === true;
    const status = ok ? 'active' : 'failed';
    const detail = v?.status || (ok ? (v?.path || v?.endpoint) : (v?.error || v?.path || v?.endpoint)) || (ok ? 'ok' : 'not ready');
    return `
      <div class="svc-card" data-status="${status}">
        <div class="card-top">
          <span class="card-name">${escapeHtml(key.replaceAll('_', ' '))}</span>
          <span class="status-pill ${status}">${ok ? 'ok' : 'down'}</span>
        </div>
        <div class="card-desc">${escapeHtml(detail)}</div>
      </div>`;
  }).join('');
}

async function voiceTtsAction(action, btn) {
  await svcAction('voice-tts', action, btn);
  await loadVoiceTtsStatus(true);
}

async function setVoiceTtsDefault(btn) {
  const voice = document.getElementById('voice-tts-default-select').value;
  if (!voice) return;
  const orig = btn.textContent;
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>';
  try {
    const d = await fetchJSON('/api/voice-tts/default', 'POST', { voice });
    toast(d.ok ? `Default voice: ${d.default} (was ${d.previous || 'unset'})` : (d.error || 'failed'), d.ok ? 'ok' : 'err');
    await loadVoiceTtsStatus(true);
  } catch (e) {
    toast('Could not set the default voice: ' + e, 'err');
  } finally {
    btn.disabled = false;
    btn.textContent = orig;
  }
}

async function runVoiceTtsTest(btn) {
  const text = document.getElementById('voice-tts-test-input').value.trim();
  if (!text) {
    toast('Enter text to speak', 'err');
    return;
  }
  const orig = btn.textContent;
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>';
  try {
    const res = await fetch('/api/voice-tts/speak', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        input: text,
        voice: document.getElementById('voice-tts-test-voice').value || 'default',
        instructions: document.getElementById('voice-tts-test-style').value.trim(),
      }),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.error || body.detail || `HTTP ${res.status}`);
    }
    const audio = document.getElementById('voice-tts-audio');
    audio.src = URL.createObjectURL(await res.blob());
    audio.style.display = '';
    await audio.play().catch(() => {});
  } catch (e) {
    toast('Voice TTS test failed: ' + e, 'err');
  } finally {
    btn.disabled = false;
    btn.textContent = orig;
  }
}
