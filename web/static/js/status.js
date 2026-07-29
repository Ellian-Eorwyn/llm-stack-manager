// status.js
//
// The 5-second poll: service cards, GPU gauges, per-slot context, and the
// actions that start, stop and restart units.

// -- status polling --
function startPolling() {
  poll();
  pollTimer = setInterval(poll, 5000);
  // A backgrounded tab has nothing to show, so stop paying for the polls.
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) poll();
  });
}

async function poll() {
  if (document.hidden) return;
  try {
    const d = await fetchJSON('/api/status');
    applyStatuses(d.services, d.health);
    applyGpus(d.gpus);
    applyServiceContexts(d.contexts);
    applyDeployment(d.deployment);
  } catch {}
  try { await pollTelemetry(); } catch {}
  try { await pollActiveModel(); } catch {}
  try { await loadTtsOverview(true); } catch {}
}

// -- service actions --
async function svcAction(name, action, btn) {
  const orig = btn.textContent;
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>';
  try {
    const d = await fetchJSON(`/api/service/${name}/${action}`, 'POST');
    toast(d.ok ? `${name} - ${action} OK` : (d.output || d.error || 'failed'), d.ok ? 'ok' : 'err');
    await poll();
  } catch (e) { toast('Error: ' + e, 'err'); }
  finally { btn.disabled = false; btn.textContent = orig; }
}

async function bulkAction(action) {
  const names = [...document.querySelectorAll('.svc-card[id^="card-"]')]
    .map(c => c.id.replace('card-',''));
  toast(`${action} all services...`, 'info');
  for (const name of names) {
    try { await fetchJSON(`/api/service/${name}/${action}`, 'POST'); } catch {}
  }
  await poll();
  toast(`${action} all done`, 'ok');
}

async function restartApp() {
  if (!confirm('Restart all stack services now?')) return;
  await bulkAction('restart');
}

async function updateApp(btn) {
  if (!confirm('Update LLM Stack Manager from GitHub now? This will pull the latest changes, install dependencies, and restart the app.')) return;
  const orig = btn.textContent;
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Updating...';
  toast('Updating app in background. Page will reload shortly...', 'info');
  try {
    const d = await fetchJSON('/api/app/update', 'POST');
    if (!d.ok) {
      toast('Update failed to start: ' + (d.error || 'Unknown error'), 'err');
      btn.disabled = false;
      btn.textContent = orig;
      return;
    }
    // Wait a few seconds for the app to go down, then poll until it's back up
    setTimeout(() => {
      let attempts = 0;
      const check = setInterval(async () => {
        try {
          const res = await fetch('/api/status');
          if (res.ok) {
            clearInterval(check);
            window.location.reload();
          }
        } catch (e) {
          // Expected while server is down
        }
        if (++attempts > 60) {
          clearInterval(check);
          toast('Update taking longer than expected. Please refresh manually later.', 'warn');
        }
      }, 2000);
    }, 5000);
  } catch (e) {
    toast('Error starting update', 'err');
    btn.disabled = false;
    btn.textContent = orig;
  }
}

async function updateLlamaCpp(btn) {
  if (!confirm('Update llama.cpp now? This runs git pull + cmake build, then restarts active llama services.')) return;
  const orig = btn.textContent;
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Updating...';
  toast('Updating llama.cpp. This may take a few minutes...', 'info');
  try {
    const d = await fetchJSON('/api/llamacpp/update', 'POST');
    if (d.output) {
      document.getElementById('log-out').textContent = d.output;
    }
    if (d.ok) {
      const restarted = (d.restarted_services || []).join(', ') || 'none';
      toast(`llama.cpp updated. Restarted: ${restarted}`, 'ok');
    } else {
      toast('llama.cpp update failed. See Logs tab output.', 'err');
    }
    await poll();
  } catch (e) {
    toast('Error: ' + e, 'err');
  } finally {
    btn.disabled = false;
    btn.textContent = orig;
  }
}

// -- default mode --
async function startRestoreActiveStack(btn) {
  const orig = btn.textContent;
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Running...';
  toast('Running restore-active-stack.sh...', 'info');
  try {
    const d = await fetchJSON('/api/restore-active-stack', 'POST');
    toast(d.ok ? 'Active stack restored' : (d.output || 'failed'), d.ok ? 'ok' : 'err');
    await poll();
  } catch (e) { toast('Error: ' + e, 'err'); }
  finally { btn.disabled = false; btn.textContent = orig; }
}

// -- switch model (built-in + custom) --
async function switchModel(btn, variant) {
  const orig = btn.textContent;
  const targetLabel = builtInChatVariantMap[variant]?.label
    || customModels.find(m => m.id === variant)?.display_name
    || variant;
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>';
  toast(`Switching to ${targetLabel}...`, 'info');
  try {
    const d = await fetchJSON(`/api/switch/${variant}`, 'POST');
    toast(d.ok ? `Switched to ${targetLabel}` : (d.output || d.error || 'failed'), d.ok ? 'ok' : 'err');
    await poll();
  } catch (e) { toast('Error: ' + e, 'err'); }
  finally { btn.disabled = false; btn.textContent = orig; }
}
