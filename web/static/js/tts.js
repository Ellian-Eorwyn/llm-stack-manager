// tts.js
//
// Text-to-speech backend selection and the test-phrase player.

// -- tts --
async function loadTtsOverview(silent = false) {
  try {
    ttsOverview = await fetchJSON('/api/tts/overview');
    renderTtsOverview();
  } catch (e) {
    if (!silent) toast('Could not load TTS overview: ' + e, 'err');
  }
}

function renderTtsOverview() {
  if (!ttsOverview) return;
  document.getElementById('tts-public-endpoint').textContent = ttsOverview.public_endpoint || '-';
  document.getElementById('tts-gateway-status').textContent = ttsOverview.gateway_service_status || 'unknown';
  document.getElementById('tts-active-backend').textContent = ttsOverview.active_backend || 'none';

  const backendSelect = document.getElementById('tts-test-backend');
  const voiceSelect = document.getElementById('tts-test-voice');
  const formatSelect = document.getElementById('tts-test-format');
  backendSelect.innerHTML = ttsOverview.backends.map(b =>
    `<option value="${b.id}" ${b.active ? 'selected' : ''}>${b.label}</option>`
  ).join('');
  const selectedBackend = ttsOverview.backends.find(b => b.id === backendSelect.value) || ttsOverview.backends[0];
  voiceSelect.innerHTML = (selectedBackend?.voices?.length ? selectedBackend.voices : ['default']).map(v =>
    `<option value="${v}">${v}</option>`
  ).join('');
  if (ttsOverview.default_format) formatSelect.value = ttsOverview.default_format;

  document.getElementById('tts-meta-list').innerHTML = [
    `Single active: ${ttsOverview.single_active}`,
    `Backends: ${ttsOverview.backends.length}`,
    ttsOverview.updated_at ? `Last switch: ${ttsOverview.updated_at}` : 'Last switch: -',
    ttsOverview.gateway_error ? `Gateway error: ${ttsOverview.gateway_error}` : 'Gateway health polling available'
  ].map(item => `<span class="meta-chip">${item}</span>`).join('');

  document.getElementById('tts-backend-cards').innerHTML = ttsOverview.backends.map(b => {
    const health = b.health || {};
    const detail = health.detail ? String(health.detail).slice(0, 120) : '';
    return `
      <div class="svc-card" data-status="${b.active ? 'active' : b.service_status || 'inactive'}">
        <div class="card-top">
          <span class="card-name">${b.label}</span>
          <span class="status-pill ${b.active ? 'active' : (b.service_status || 'inactive')}">${b.active ? 'active' : (b.service_status || 'inactive')}</span>
        </div>
        <div class="card-desc">${b.description || b.family || ''}</div>
        <div class="card-port">${b.internal_url || '-'}</div>
        <div class="meta-list">
          <span class="meta-chip">configured: ${b.configured ? 'yes' : 'no'}</span>
          <span class="meta-chip">health: ${health.ok ? 'ok' : 'down'}</span>
          <span class="meta-chip">voices: ${(b.voices || []).length}</span>
        </div>
        ${detail ? `<div class="card-desc" style="margin-top:10px;">${detail}</div>` : ''}
        <div class="card-acts" style="margin-top:12px;">
          <button class="btn btn-success btn-sm" onclick="svcAction('${b.service_name}','start',this)">Start</button>
          <button class="btn btn-danger btn-sm" onclick="svcAction('${b.service_name}','stop',this)">Stop</button>
          <button class="btn btn-primary btn-sm" onclick="activateTtsBackend('${b.id}', this)">Make Active</button>
          <button class="btn btn-ghost btn-sm" onclick="openLogs('${b.service_name}')">Logs</button>
        </div>
      </div>`;
  }).join('');
}

document.getElementById('tts-test-backend').addEventListener('change', () => {
  if (!ttsOverview) return;
  const backend = ttsOverview.backends.find(b => b.id === document.getElementById('tts-test-backend').value);
  const voiceSelect = document.getElementById('tts-test-voice');
  voiceSelect.innerHTML = (backend?.voices?.length ? backend.voices : ['default']).map(v =>
    `<option value="${v}">${v}</option>`
  ).join('');
});

async function activateTtsBackend(backendId, btn) {
  const orig = btn.textContent;
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>';
  try {
    const d = await fetchJSON(`/api/tts/activate/${backendId}`, 'POST');
    toast(d.ok ? `Active TTS backend: ${backendId}` : (d.error || 'failed'), d.ok ? 'ok' : 'err');
    await loadTtsOverview();
    await poll();
  } catch (e) {
    toast('Error: ' + e, 'err');
  } finally {
    btn.disabled = false;
    btn.textContent = orig;
  }
}

async function runTtsTest(btn) {
  const text = document.getElementById('tts-test-input').value.trim();
  if (!text) {
    toast('Enter text to synthesize', 'err');
    return;
  }
  const orig = btn.textContent;
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>';
  try {
    const res = await fetch('/api/tts/test', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: document.getElementById('tts-test-backend').value,
        input: text,
        voice: document.getElementById('tts-test-voice').value,
        response_format: document.getElementById('tts-test-format').value,
        speed: Number(document.getElementById('tts-test-speed').value || '1.0'),
      }),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.error || body.detail || `HTTP ${res.status}`);
    }
    const blob = await res.blob();
    const audio = document.getElementById('tts-audio');
    audio.src = URL.createObjectURL(blob);
    audio.style.display = '';
    await audio.play().catch(() => {});
    toast('Speech generated', 'ok');
  } catch (e) {
    toast('TTS test failed: ' + e, 'err');
  } finally {
    btn.disabled = false;
    btn.textContent = orig;
  }
}
