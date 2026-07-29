// deploy.js
//
// The header badge that says when the installed tree is behind its remote.
//
// Rides the status poll rather than adding a timer, and reads a server-side
// cache — the git fetch behind it runs on the manager's own thread.

// -- deployment drift --
// The manager does not run from the checkout anyone edits; systemd starts it
// from the installed tree, which only advances when update.sh is run. This rides
// the /api/status poll rather than adding a timer, and reads a server-side cache
// — the git fetch that fills it happens on the manager's own thread.
let deployState = null;

function applyDeployment(payload) {
  if (!payload) return;
  deployState = payload;
  const badge = document.getElementById('deploy-badge');
  const text = document.getElementById('deploy-pill-text');
  if (!badge || !text) return;

  const summary = payload.summary || {};
  const state = summary.state || 'unknown';
  badge.dataset.state = state;
  // Nothing to say when the deployed tree is current: a permanent green badge
  // is noise, and noise is what stops a real warning from being read.
  badge.classList.toggle('visible', state !== 'current');

  const short = payload.head_short || '';
  if (state === 'behind' || state === 'dirty') {
    const behind = payload.behind || 0;
    text.textContent = behind
      ? `${behind} update${behind === 1 ? '' : 's'} pending`
      : 'local changes';
  } else {
    text.textContent = short ? `Version ${short}` : 'Version unknown';
  }
  document.getElementById('deploy-pill').title = summary.message || '';
  renderDeployDetail(payload);
}

function renderDeployDetail(payload) {
  const detail = document.getElementById('deploy-detail');
  if (!detail || !detail.classList.contains('open')) return;

  const summary = payload.summary || {};
  const commits = payload.pending_commits || [];
  const stale = payload.backend_sensitive_changes || [];
  const dirty = payload.dirty_paths || [];
  const checked = payload.remote_checked_at
    ? new Date(payload.remote_checked_at * 1000).toLocaleTimeString()
    : 'not yet';

  let html = `<h4>Deployment</h4><p>${escapeHtml(summary.message || 'unknown')}</p>`;
  html += `<p>Running <code>${escapeHtml(payload.head_short || '?')}</code>`;
  if (payload.subject) html += ` — ${escapeHtml(payload.subject)}`;
  html += `<br>Installed at <code>${escapeHtml(payload.stack_dir || '')}</code>`;
  html += `<br>Last checked ${escapeHtml(checked)}</p>`;

  if (commits.length) {
    html += `<h4>Pending</h4><ul class="deploy-commits">` + commits.map(c =>
      `<li><span class="deploy-sha">${escapeHtml(c.sha)}</span><span>${escapeHtml(c.subject)}</span></li>`
    ).join('') + `</ul>`;
  }
  if (stale.length) {
    // Restarting a model backend reloads tens of GB of weights and discards a
    // warm prompt cache, so this is flagged rather than done.
    html += `<div class="deploy-warn">Model backend launchers changed: ` +
      stale.map(p => `<code>${escapeHtml(p)}</code>`).join(', ') +
      `. Those backends keep running the previous code until restarted.</div>`;
  }
  if (dirty.length) {
    html += `<h4>Local changes</h4><ul class="deploy-commits">` + dirty.map(p =>
      `<li><span>${escapeHtml(p)}</span></li>`).join('') + `</ul>`;
  }
  if (payload.fetch_error) {
    html += `<div class="deploy-warn">Could not reach the remote: ${escapeHtml(payload.fetch_error)}</div>`;
  }
  html += `<h4>Remedy</h4><code class="deploy-remedy">${escapeHtml(payload.remedy || '')}</code>`;
  html += `<button class="btn btn-ghost btn-sm" onclick="checkDeployNow(this)">Check now</button>`;
  detail.innerHTML = html;
}

function toggleDeployDetail() {
  const detail = document.getElementById('deploy-detail');
  const pill = document.getElementById('deploy-pill');
  if (!detail) return;
  const open = !detail.classList.contains('open');
  detail.classList.toggle('open', open);
  if (pill) pill.setAttribute('aria-expanded', open ? 'true' : 'false');
  if (open && deployState) renderDeployDetail(deployState);
}

function initDeployBadge() {
  document.addEventListener('click', (event) => {
    const badge = document.getElementById('deploy-badge');
    if (badge && !badge.contains(event.target)) {
      document.getElementById('deploy-detail')?.classList.remove('open');
      document.getElementById('deploy-pill')?.setAttribute('aria-expanded', 'false');
    }
  });
}

async function checkDeployNow(btn) {
  const orig = btn.textContent;
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Checking...';
  try {
    applyDeployment(await fetchJSON('/api/deploy/check', 'POST'));
    toast('Deployment checked', 'ok');
  } catch (e) {
    toast('Deployment check failed: ' + e.message, 'err');
  }
  btn.disabled = false;
  btn.textContent = orig;
}
