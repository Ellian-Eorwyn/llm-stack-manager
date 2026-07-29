// scheduling.js
//
// The pi-forge slot-scheduling contract panel.
//
// Verification is passive by default: the journal already records which slot
// was chosen, so the contract can be shown without sending a request. The
// active probe is opt-in because pinning a probe to slot 0 would displace the
// prompt prefix that slot is holding.

// -- pi-forge scheduling contract --
function schedRow(key, value) {
  return `<div class="sched-row"><span class="k">${escapeHtml(key)}</span><span class="v">${escapeHtml(String(value))}</span></div>`;
}

function renderSchedulingVerify(result) {
  const body = document.getElementById('sched-verify-body');
  if (!body) return;
  if (!result || result.error) {
    body.innerHTML = `<div class="sched-verdict bad">${escapeHtml(result?.error || 'Verification failed.')}</div>`;
    return;
  }

  const parts = [];
  const evidence = result.evidence || {};
  const leases = result.leases || {};
  const counts = leases.counts || {};

  if (!result.unit_active) {
    parts.push('<div class="sched-verdict warn"><strong>No primary backend is running.</strong> Only the saved configuration could be checked.</div>');
  } else if (result.ok && evidence.observed) {
    parts.push('<div class="sched-verdict ok"><strong>Contract holds, and pi-forge is using it.</strong> Both slots have been pinned by id.</div>');
  } else if (result.ok) {
    parts.push('<div class="sched-verdict ok"><strong>Contract holds.</strong> No pinned request has been seen in this window, which an idle session explains.</div>');
  } else {
    parts.push('<div class="sched-verdict bad"><strong>The backend cannot honour the scheduling contract.</strong></div>');
  }

  if (result.issues?.length) {
    parts.push(`<ul>${result.issues.map(i => `<li>${escapeHtml(i)}</li>`).join('')}</ul>`);
  }
  // Drift is its own class of problem: the configuration is fine, the running
  // process is fine, and they are not the same configuration.
  for (const note of result.drift || []) {
    parts.push(`<div class="sched-verdict warn">${escapeHtml(note)}</div>`);
  }

  const runtime = result.runtime || {};
  const configured = result.configured || {};
  parts.push(schedRow('Backend unit', result.unit || 'none'));
  parts.push(schedRow('Slots', `${runtime.total_slots ?? configured.slots ?? '—'} (running) / ${configured.slots ?? '—'} (configured)`));
  parts.push(schedRow('Context per slot', (runtime.n_ctx_per_slot ?? configured.per_slot_context ?? 0).toLocaleString('en-US')));
  const byId = evidence.select_by_id_slots || {};
  parts.push(schedRow(`Pinned to slot ${evidence.interactive_slot ?? 0} (interactive)`, `${byId[String(evidence.interactive_slot ?? 0)] || 0} selections`));
  parts.push(schedRow(`Pinned to slot ${evidence.background_slot ?? 1} (background)`, `${byId[String(evidence.background_slot ?? 1)] || 0} selections`));
  const methods = evidence.select_methods || {};
  parts.push(schedRow('Other slot selections', `${(methods.lcp || 0)} by prefix, ${(methods.lru || 0)} by LRU`));

  const holders = (leases.entries || []).filter(e => e.classification === 'fresh');
  parts.push(schedRow('Active leases', holders.length
    ? holders.map(e => `${e.kind} pid ${e.pid} → slot ${e.slot}`).join(', ')
    : 'none'));
  parts.push(schedRow('Orphaned leases', `${leases.reapable || 0} reapable of ${(counts.fresh || 0) + (counts.stale || 0) + (counts.orphan || 0) + (counts.malformed || 0)}`));
  parts.push(schedRow('Lease directory', leases.directory || 'not found'));

  body.innerHTML = parts.join('');
}

async function refreshSchedulingVerify(btn) {
  const body = document.getElementById('sched-verify-body');
  if (!body) return;
  if (btn) btn.disabled = true;
  try {
    renderSchedulingVerify(await fetchJSON('/api/scheduling/verify'));
  } catch (error) {
    renderSchedulingVerify({ error: String(error) });
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function runSchedulingProbe(btn) {
  if (!confirm('Send one request to each slot?\n\nThis proves the backend honours id_slot, but both slots lose the prompt prefix they are holding, so the next pi-forge turn reprocesses its context.')) return;
  if (btn) btn.disabled = true;
  try {
    const result = await fetchJSON('/api/scheduling/verify', 'POST', { probe: true });
    renderSchedulingVerify(result);
    const probe = result.probe || {};
    toast(probe.honoured
      ? 'The backend served each request from the slot it was pinned to.'
      : `Probe did not confirm pinning: ${probe.error || 'no by-id slot selection appeared in the journal'}`,
      probe.honoured ? 'ok' : 'err');
  } catch (error) {
    toast(`Slot probe failed: ${error}`, 'err');
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function reapPiForgeLeases(btn) {
  if (btn) btn.disabled = true;
  try {
    const result = await fetchJSON('/api/scheduling/leases/reap', 'POST', { dry_run: false });
    const removed = (result.removed || []).length;
    toast(removed
      ? `Removed ${removed} orphaned lease${removed === 1 ? '' : 's'}.`
      : 'No orphaned leases — every lease is fresh or its process is still alive.',
      'ok');
    await refreshSchedulingVerify();
  } catch (error) {
    toast(`Could not reap leases: ${error}`, 'err');
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function copyPiForgeSchedulingSnippet(btn) {
  if (typeof CacheAwareScheduling === 'undefined') return;
  const snippet = CacheAwareScheduling.piForgeSnippet();
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(snippet);
    } else {
      const fallback = document.createElement('textarea');
      fallback.value = snippet;
      fallback.style.position = 'fixed';
      fallback.style.opacity = '0';
      document.body.appendChild(fallback);
      fallback.select();
      document.execCommand('copy');
      fallback.remove();
    }
    const original = btn?.textContent;
    if (btn) btn.textContent = 'Copied';
    toast('Copied pi-forge scheduling merge snippet.', 'ok');
    if (btn) setTimeout(() => { btn.textContent = original; }, 1500);
  } catch (error) {
    toast(`Could not copy pi-forge settings: ${error}`, 'err');
  }
}

function quickGpuValueFor(prefix) {
  const devices = (document.getElementById(`cfg-${prefix}_GPU_VISIBLE_DEVICES`)?.value || '').trim();
  const mainGpu = (document.getElementById(`cfg-${prefix}_MAIN_GPU`)?.value || '').trim();
  const splitMode = (document.getElementById(`cfg-${prefix}_SPLIT_MODE`)?.value || '').trim();
  const tensorSplit = (document.getElementById(`cfg-${prefix}_TENSOR_SPLIT`)?.value || '').trim();

  if (devices === '0' && mainGpu === '0' && splitMode === 'none' && tensorSplit === '') return 'gpu0';
  if (devices === '1' && mainGpu === '0' && splitMode === 'none' && tensorSplit === '') return 'gpu1';
  if (devices === '0,1' && mainGpu === '0' && splitMode === 'layer' && tensorSplit === '1,1') return 'gpu01';
  return 'custom';
}

function syncQuickGpuPlacementSelects() {
  document.querySelectorAll('.quick-gpu-placement[data-prefix]').forEach(sel => {
    sel.value = quickGpuValueFor(sel.dataset.prefix);
  });
}

function applyQuickGpuPlacement(sel) {
  const prefix = sel?.dataset?.prefix;
  if (!prefix || sel.value === 'custom') return;
  const presets = {
    gpu0: { devices: '0', mainGpu: '0', splitMode: 'none', tensorSplit: '' },
    gpu1: { devices: '1', mainGpu: '0', splitMode: 'none', tensorSplit: '' },
    gpu01: { devices: '0,1', mainGpu: '0', splitMode: 'layer', tensorSplit: '1,1' },
  };
  const preset = presets[sel.value];
  if (!preset) return;
  const values = {
    [`${prefix}_GPU_VISIBLE_DEVICES`]: preset.devices,
    [`${prefix}_MAIN_GPU`]: preset.mainGpu,
    [`${prefix}_SPLIT_MODE`]: preset.splitMode,
    [`${prefix}_TENSOR_SPLIT`]: preset.tensorSplit,
  };
  for (const [key, value] of Object.entries(values)) {
    const el = document.getElementById('cfg-' + key);
    if (el) el.value = value;
  }
  const section = sel.closest('.cfg-section')?.dataset?.section || '';
  if (section) markDirty(section);
}

async function saveCfgSection(section, btn) {
  const sec = document.getElementById('cfgsec-' + section.replace(/ /g, '-'));
  document.querySelectorAll('.custom-args-list[data-config-target]').forEach(list => {
    const target = document.getElementById(list.dataset.configTarget);
    if (target && sec.contains(target)) syncConfigArgInput(list.dataset.configTarget, true);
  });
  const updates = {};
  sec.querySelectorAll('[name]').forEach(el => { updates[el.name] = el.value; });

  const orig = btn.textContent;
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Saving...';
  try {
    let d = await fetchJSON('/api/config', 'POST', updates);
    // The budget model refuses configurations it predicts cannot allocate. It
    // is a prediction, so the operator can override it — but they have to be
    // told what they are overriding first.
    if (!d.ok && d.preflight && !d.preflight.ok) {
      const reasons = (d.preflight.errors || []).map(issue => '• ' + issue.text).join('\n\n');
      const proceed = confirm(
        `This configuration is predicted not to fit:\n\n${reasons}\n\n` +
        'Save it anyway?'
      );
      if (!proceed) {
        showPreflight(section, d.preflight);
        toast('Not saved. Adjust the configuration or save again to override.', 'err');
        return;
      }
      d = await fetchJSON('/api/config?force=1', 'POST', updates);
    }
    showPreflight(section, d.preflight);
    if (d.ok) {
      cfgCurrent = { ...cfgCurrent, ...updates };
      refreshBuiltInModelButtons(updates);
      cfgDirty[section] = false;
      const hint = document.getElementById('rhint-' + section.replace(/ /g, '-'));
      const restartNeeded = d.restart_needed || [];
      await patchSelectedSavedConfig(section, updates);
      if (section === 'Transcription' && restartNeeded.includes('transcript-backend')) {
        hint.textContent = 'Restarting transcript-backend...';
        const restart = await fetchJSON('/api/service/transcript-backend/restart', 'POST');
        if (restart.ok) {
          hint.textContent = 'Saved and restarted: transcript-backend';
          toast('Saved Transcription and restarted transcript-backend', 'ok');
          await poll();
        } else {
          hint.textContent = 'Saved, but restart failed: transcript-backend';
          toast(restart.output || restart.error || 'Transcript backend restart failed', 'err');
        }
      } else {
        const restartBtn = document.getElementById('restart-btn-' + section.replace(/ /g, '-'));
        if (restartNeeded.length > 0) {
          hint.textContent = 'Restart needed: ' + restartNeeded.join(', ');
          if (restartBtn) {
            restartBtn.style.display = 'inline-block';
            restartBtn.dataset.services = JSON.stringify(restartNeeded);
          }
        } else {
          hint.textContent = '';
          if (restartBtn) restartBtn.style.display = 'none';
        }
        toast('Saved ' + section, 'ok');
      }
    } else {
      toast('Save failed: ' + (d.error || 'unknown'), 'err');
    }
  } catch (e) { toast('Error: ' + e, 'err'); }
  finally { btn.disabled = false; btn.textContent = orig; }
}

async function restartSectionServices(secId, btn) {
  const orig = btn.textContent;
  const svcsText = btn.dataset.services;
  if (!svcsText) return;
  const svcs = JSON.parse(svcsText);
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Restarting...';
  let successCount = 0;
  for (const svc of svcs) {
    try {
      const d = await fetchJSON(`/api/service/${svc}/restart`, 'POST');
      if (d.ok) successCount++;
    } catch {}
  }
  btn.disabled = false;
  btn.textContent = orig;
  if (successCount > 0) {
    toast(`Restarted ${successCount} service(s)`, 'ok');
    btn.style.display = 'none';
    const hint = document.getElementById('rhint-' + secId);
    if (hint) hint.textContent = 'Restarted successfully.';
    await poll();
  } else {
    toast('Failed to restart services.', 'err');
  }
}
