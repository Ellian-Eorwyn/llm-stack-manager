// telemetry.js
//
// Live backend detail: throughput, speculative-decode acceptance, prompt-cache
// behaviour and per-slot occupancy.
//
// Context is shown per slot, not as the configured total: `--ctx-size` is
// divided by `--parallel`, and reporting the total is how a backend the UI
// called "262144" came to reject a 155751-token request.

// -- backend telemetry --
const TELEMETRY_WINDOW_SECONDS = 3600;

function teleNumber(value, digits = 1) {
  return Number.isFinite(value) ? Number(value).toFixed(digits) : '—';
}

function teleInt(value) {
  return Number.isFinite(value) ? Number(value).toLocaleString('en-US') : '—';
}

function teleMetric(label, value, sub, warn) {
  return `<div class="tele-metric${warn ? ' is-warn' : ''}">
    <div class="tele-metric-label">${escapeHtml(label)}</div>
    <div class="tele-metric-value">${escapeHtml(value)}</div>
    ${sub ? `<div class="tele-metric-sub">${escapeHtml(sub)}</div>` : ''}
  </div>`;
}

function telemetrySlotsHtml(slots) {
  if (!slots?.length) return '<div class="tele-empty">No slot detail available.</div>';
  return slots.map(s => {
    const pct = Number.isFinite(s.ctx_pct) ? s.ctx_pct : 0;
    const state = s.is_processing ? 'busy' : 'idle';
    return `<div class="tele-slot">
      <div class="tele-slot-label">
        <span>slot ${s.id}${s.is_processing ? ' <span class="tele-slot-busy">busy</span>' : ''}</span>
        <span>${teleInt(s.n_prompt_tokens)} / ${teleInt(s.n_ctx)} (${teleNumber(pct)}%)</span>
      </div>
      <div class="bar-wrap" title="slot ${s.id} ${state}">
        <div class="bar-segment" style="width:${Math.min(100, pct)}%; background-color:${s.is_processing ? 'var(--success)' : 'var(--accent)'};"></div>
      </div>
    </div>`;
  }).join('');
}

function telemetryCardHtml(backend) {
  const props = backend.props || {};
  const stats = backend.stats || {};
  const tp = stats.throughput || {};
  const cache = stats.cache || {};
  const sched = stats.scheduling || {};

  if (!backend.active) {
    return `<div class="tele-card">
      <div class="tele-card-head"><span class="tele-name">${escapeHtml(backend.label)}</span>
      <span class="tele-unit">inactive</span></div>
      <div class="tele-empty">Backend is not running.</div>
    </div>`;
  }

  // llama-server divides --ctx-size across slots, so the per-slot figure is the
  // one that actually limits a request. Show both to keep that visible.
  const perSlot = props.n_ctx_per_slot;
  const ctxLine = Number.isFinite(perSlot)
    ? `<strong>${teleInt(perSlot)}</strong> tokens per slot × ${props.total_slots || 1}
       = <strong>${teleInt(props.n_ctx_total)}</strong> total`
    : 'Context unknown';

  const gen = tp.generation_tps || {};
  const draft = tp.draft_acceptance || {};
  const delay = sched.select_to_launch_seconds || {};
  const evicted = cache.evicted_mib || {};
  const perLaunch = cache.evictions_per_launch;
  const methods = sched.select_methods || {};
  const methodText = Object.keys(methods).length
    ? Object.entries(methods).map(([k, v]) => `${k}:${v}`).join(' ')
    : '—';

  const metrics = [
    teleMetric('Gen tok/s',
      teleNumber(tp.last_generation_tps ?? gen.p50),
      `p50 ${teleNumber(gen.p50)} · p90 ${teleNumber(gen.p90)}`),
    teleMetric('Draft accept',
      draft.p50 != null ? `${teleNumber(draft.p50 * 100, 0)}%` : '—',
      `mean len ${teleNumber((tp.draft_mean_len || {}).p50)}`),
    teleMetric('Evict / launch',
      perLaunch != null ? teleNumber(perLaunch, 2) : '—',
      `${teleInt(cache.evictions)} of ${teleInt(cache.launches)}`,
      perLaunch != null && perLaunch >= 0.25),
    teleMetric('Evicted p50',
      evicted.p50 != null ? `${teleNumber(evicted.p50, 0)}M` : '—',
      `max ${evicted.max != null ? teleNumber(evicted.max, 0) + 'M' : '—'}`),
    teleMetric('Slot delay p90',
      delay.p90 != null ? `${teleNumber(delay.p90, 2)}s` : '—',
      `p99 ${delay.p99 != null ? teleNumber(delay.p99, 2) + 's' : '—'}`,
      delay.p90 != null && delay.p90 >= 1),
    teleMetric('Checkpoint p50',
      (cache.checkpoint_mib || {}).p50 != null ? `${teleNumber(cache.checkpoint_mib.p50, 0)}M` : '—',
      `${teleInt(cache.checkpoint_erasures)} erased`),
    teleMetric('Slot select', methodText, 'by id = client-pinned'),
    teleMetric('Prompt tok/s',
      teleNumber((tp.prompt_tps || {}).p50),
      `p90 ${teleNumber((tp.prompt_tps || {}).p90)}`),
  ].join('');

  return `<div class="tele-card">
    <div class="tele-card-head">
      <span class="tele-name">${escapeHtml(backend.label)}</span>
      <span class="tele-unit">${escapeHtml(backend.unit || '')}${backend.metrics_available ? ' · metrics' : ''}</span>
    </div>
    <div class="tele-model" title="${escapeHtml(props.model_path || '')}">
      ${escapeHtml((props.model_path || '').split('/').pop() || 'unknown model')}
      ${props.model_ftype ? '· ' + escapeHtml(props.model_ftype) : ''}
      ${props.build_info ? '· ' + escapeHtml(props.build_info) : ''}
    </div>
    <div class="tele-ctx">${ctxLine}</div>
    ${telemetrySlotsHtml(backend.slots)}
    <div class="tele-metrics">${metrics}</div>
  </div>`;
}

function applyTelemetry(payload) {
  const cards = document.getElementById('telemetry-cards');
  const alerts = document.getElementById('telemetry-alerts');
  const windowLabel = document.getElementById('telemetry-window');
  if (!cards) return;

  const host = payload.host || {};
  if (windowLabel) {
    const minutes = Math.round((payload.window_seconds || 0) / 60);
    const swap = host.swap_used_pct != null
      ? ` · RAM ${host.mem_used_pct}% · swap ${host.swap_used_pct}% (${formatG(host.swap_used_mib)})`
      : '';
    windowLabel.textContent = `last ${minutes}m${swap}`;
  }

  if (alerts) {
    alerts.innerHTML = (payload.warnings || []).map(w =>
      `<div class="tele-alert ${w.level === 'warn' ? 'warn' : 'info'}">${escapeHtml(w.text)}</div>`
    ).join('');
  }

  const active = (payload.backends || []).filter(b => b.active);
  cards.innerHTML = active.length
    ? active.map(telemetryCardHtml).join('')
    : '<div class="tele-empty">No llama.cpp backends are running.</div>';
}

async function pollTelemetry() {
  const payload = await fetchJSON(`/api/backend/telemetry?window=${TELEMETRY_WINDOW_SECONDS}`);
  applyTelemetry(payload);
}

// The unit status says whether the process is running. The health entry says
// whether it is working — its readiness probe answered, and the services it
// depends on are up. Prefer the second where there is one; a card whose probe
// has not run yet still renders from the first rather than going blank.
function applyStatuses(statuses, health) {
  for (const [name, status] of Object.entries(statuses)) {
    const card = document.getElementById('card-' + name);
    const pill = document.getElementById('pill-' + name);
    const note = document.getElementById('health-' + name);
    if (!card) continue;
    const entry = (health || {})[name];
    const state = entry?.state || status;
    card.dataset.status = state;
    if (pill) {
      pill.className = 'status-pill ' + state;
      pill.textContent = state;
    }
    if (note) {
      // A stopped service is not a fault, so its reason reads muted.
      note.className = 'card-health' + (state === 'stopped' ? ' muted' : '');
      note.textContent = state === 'active' ? '' : (entry?.reason || '');
    }
  }
}

// The configured context of each model service, per slot. Read from the env
// rather than the running backend so it is there while the service is stopped,
// which is when the number is being chosen.
function applyServiceContexts(contexts) {
  for (const [name, ctx] of Object.entries(contexts || {})) {
    const el = document.getElementById('ctx-' + name);
    if (!el) continue;
    const perSlot = Number(ctx.per_slot_context || 0).toLocaleString('en-US');
    el.textContent = ctx.slots > 1
      ? `${perSlot} tokens/slot × ${ctx.slots} = ${Number(ctx.total_context).toLocaleString('en-US')} total`
      : `${perSlot} tokens of context`;
  }
}

const PROCESS_PALETTE = [
  '#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#14b8a6', '#f43f5e', '#6366f1'
];
const _processColors = {};
let _processColorIdx = 0;
function getProcessColor(name) {
  if (!_processColors[name]) {
    _processColors[name] = PROCESS_PALETTE[_processColorIdx % PROCESS_PALETTE.length];
    _processColorIdx++;
  }
  return _processColors[name];
}

function formatG(mib) {
  return (Number(mib) / 1000).toFixed(1) + 'G';
}

function applyGpus(gpus) {
  const el = document.getElementById('gpu-stats');
  if (!el) return;
  if (!gpus?.length) {
    el.innerHTML = '<div class="gpu-card"><div class="gpu-process-empty">No GPU stats available</div></div>';
    return;
  }
  el.innerHTML = gpus.map(g => {
    const used = Number(g.mem_used || 0);
    const total = Number(g.mem_total || 0);
    const pct = Math.max(0, Math.min(100, Number(g.mem_pct || 0)));
    
    let segmentsHtml = '';
    let itemsHtml = '';
    
    let knownUsed = 0;
    if ((g.processes || []).length) {
      g.processes.forEach(p => {
        const name = p.name || p.process_name || 'process';
        const color = getProcessColor(name);
        const pUsed = Number(p.used_memory || 0);
        knownUsed += pUsed;
        const pPct = total > 0 ? (pUsed / total * 100) : 0;
        
        segmentsHtml += `<div class="bar-segment" style="width:${pPct}%; background-color:${color};" title="${escapeHtml(name)}: ${formatG(pUsed)}"></div>`;
        itemsHtml += `<div class="gpu-process-item" title="${escapeHtml(name)} pid ${escapeHtml(p.pid || '')}">
          <span class="gpu-process-dot" style="background-color:${color};"></span>
          <span style="color:var(--text);">${escapeHtml(name)}</span>
          <span style="font-family:var(--mono);">${formatG(pUsed)}</span>
        </div>`;
      });
    }
    
    const otherUsed = Math.max(0, used - knownUsed);
    if (otherUsed > 0 && total > 0) {
      const otherPct = (otherUsed / total * 100);
      segmentsHtml += `<div class="bar-segment" style="width:${otherPct}%; background-color:var(--dim);" title="Other: ${formatG(otherUsed)}"></div>`;
      if (itemsHtml) {
        itemsHtml += `<div class="gpu-process-item" title="Other non-llm processes">
          <span class="gpu-process-dot" style="background-color:var(--dim);"></span>
          <span style="color:var(--text);">Other</span>
          <span style="font-family:var(--mono);">${formatG(otherUsed)}</span>
        </div>`;
      }
    }
    
    if (!itemsHtml) {
      itemsHtml = '<div class="gpu-process-empty">No compute processes</div>';
    } else {
      itemsHtml = `<div class="gpu-process-inline-list">${itemsHtml}</div>`;
    }

    return `
      <div class="gpu-card">
        <div class="gpu-card-head">
          <span class="gpu-label">GPU${escapeHtml(g.index)}</span>
          <span class="gpu-name" title="${escapeHtml(g.name || '')}">${escapeHtml(g.name || '')}</span>
        </div>
        <div class="gpu-stat-row">
          <span>${formatG(used)}/${formatG(total)} (${pct}%)</span>
          <span>${escapeHtml(g.util || 0)}% util</span>
          <span>${escapeHtml(g.temp || 0)} C</span>
        </div>
        <div class="bar-wrap" title="${formatG(used)}/${formatG(total)} | ${escapeHtml(g.util || 0)}% | ${escapeHtml(g.temp || 0)} C">
          ${segmentsHtml}
        </div>
        ${itemsHtml}
      </div>
    `;
  }).join('');
}
