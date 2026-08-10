// transcribe.js
//
// The transcription sidecar panel: which engines are installed, what is
// resident right now, and a one-file test that proves the whole path works.

let transcribeOverview = null;

function initTranscribeTab() {
  if (!transcribeOverview) loadTranscribeOverview();
}

async function loadTranscribeOverview(silent = false) {
  try {
    transcribeOverview = await fetchJSON('/api/transcribe/overview');
    renderTranscribeOverview();
    document.getElementById('transcribe-last-refresh').textContent =
      'Last refresh: ' + new Date().toLocaleTimeString();
  } catch (e) {
    if (!silent) toast('Could not load transcription overview: ' + e, 'err');
  }
}

function renderTranscribeOverview() {
  const data = transcribeOverview;
  if (!data) return;
  const residentEl = document.getElementById('transcribe-resident');
  const enginesEl = document.getElementById('transcribe-engines');

  if (!data.enabled) {
    residentEl.innerHTML = `<span class="meta-chip">Disabled — set TRANSCRIPT_ENABLED=on</span>`;
    enginesEl.innerHTML = '';
    return;
  }
  if (!data.reachable) {
    residentEl.innerHTML =
      `<span class="meta-chip">Not reachable: ${escapeHtml(data.error || 'unknown error')}</span>`;
    enginesEl.innerHTML = '';
    return;
  }

  const resident = data.resident;
  // An absent model is the designed steady state, so it is reported as a fact
  // rather than styled as a problem.
  residentEl.innerHTML = (resident ? [
    `Engine: ${resident.engine}`,
    `Model: ${resident.model}`,
    `Idle for: ${resident.idle_for}s`,
    resident.unload_in === null ? 'Idle unload: disabled' : `Releases in: ${resident.unload_in}s`,
    `In flight: ${resident.inflight}`,
  ] : [
    'No model resident — VRAM is free',
    `Idle unload: ${data.idle_unload_seconds}s`,
  ]).concat([
    `Device: ${data.device || '-'}`,
    `Default engine: ${data.active_engine || '-'}`,
    `Router: ${data.router?.reachable ? 'reachable' : 'unreachable'} (yield: ${data.router?.yield_mode || 'off'})`,
  ]).map(item => `<span class="meta-chip">${escapeHtml(String(item))}</span>`).join('');

  enginesEl.innerHTML = (data.engines || []).map(eng => {
    const caps = Object.entries(eng.capabilities || {})
      .filter(([, on]) => on)
      .map(([name]) => name.replace(/_/g, ' '))
      .join(', ') || 'none';
    return `<span class="meta-chip">${escapeHtml(eng.id)} · ${escapeHtml(eng.runtime)} · `
      + `${escapeHtml(eng.model || 'no model set')} · ${escapeHtml(caps)}</span>`;
  }).join('');

  const engineSelect = document.getElementById('transcribe-test-engine');
  const current = engineSelect.value;
  engineSelect.innerHTML = '<option value="">Default</option>' + (data.engines || [])
    .map(eng => `<option value="${escapeHtml(eng.id)}">${escapeHtml(eng.id)}</option>`).join('');
  if (current) engineSelect.value = current;
}

async function unloadTranscribeModel(btn) {
  if (btn) btn.disabled = true;
  try {
    const res = await fetchJSON('/api/transcribe/unload', { method: 'POST' });
    toast(res.unloaded ? 'Model released' : 'Nothing was resident', 'ok');
    await loadTranscribeOverview(true);
  } catch (e) {
    toast('Could not release the model: ' + e, 'err');
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function runTranscribeTest(btn) {
  const fileInput = document.getElementById('transcribe-test-file');
  const status = document.getElementById('transcribe-test-status');
  const output = document.getElementById('transcribe-test-output');
  const file = fileInput.files && fileInput.files[0];
  if (!file) {
    toast('Choose an audio file first', 'err');
    return;
  }

  const body = new FormData();
  body.append('file', file);
  const engine = document.getElementById('transcribe-test-engine').value;
  const words = document.getElementById('transcribe-test-words').value;
  if (engine) body.append('engine', engine);
  if (words) body.append('word_timestamps', words);
  body.append('response_format', document.getElementById('transcribe-test-format').value);

  if (btn) btn.disabled = true;
  status.textContent = 'Transcribing — the first request also loads the model…';
  output.textContent = '';
  const started = Date.now();
  try {
    // Not fetchJSON: this posts FormData, and the browser has to set its own
    // multipart boundary.
    const resp = await fetch('/api/transcribe/test', { method: 'POST', body });
    const data = await resp.json();
    const seconds = ((Date.now() - started) / 1000).toFixed(1);
    if (!data.ok) {
      status.textContent = `Failed after ${seconds}s`;
      output.textContent = JSON.stringify(data, null, 2);
      toast('Transcription failed', 'err');
      return;
    }
    status.textContent = `Done in ${seconds}s`;
    output.textContent = data.result !== undefined
      ? JSON.stringify(data.result, null, 2)
      : (data.text || '');
    await loadTranscribeOverview(true);
  } catch (e) {
    status.textContent = 'Failed';
    output.textContent = String(e);
    toast('Transcription failed: ' + e, 'err');
  } finally {
    if (btn) btn.disabled = false;
  }
}
