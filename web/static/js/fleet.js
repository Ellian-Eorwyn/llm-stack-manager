// fleet.js
//
// Which machine this page is looking at.
//
// Loaded second, after util.js and before everything that fetches: `fetchJSON`
// asks this module where a request should go, and every panel that loads on
// start-up has to run after the answer is bound.
//
// The whole mechanism is one function. `fleetPath` rewrites a local API path
// into its proxied form when a remote host is selected, so the seventy-eight
// call sites elsewhere in these modules do not know a fleet exists.

// '' means this machine. Not a boolean, because "which host" is the question.
let fleetHost = '';
let fleetHosts = [];

// The paths the hub actually proxies, mirroring the whitelist in
// `web/routes/fleet.py`. A list of what *is* forwarded rather than what is not,
// for the same reason the server has one: a default of "forward it" on an
// unauthenticated port is how one page bug becomes remote code execution on
// every machine in the fleet.
//
// Reads only, for now. Config saves and service actions against another host
// arrive with the control channel; until then the controls that would send
// them are disabled with a reason rather than pointed at a 404.
const FLEET_PROXIED = ['/api/status'];

function fleetPath(url) {
  if (!fleetHost || typeof url !== 'string') return url;
  const path = url.split('?')[0];
  if (!FLEET_PROXIED.includes(path)) return url;
  return '/api/fleet/' + encodeURIComponent(fleetHost) + url.slice('/api'.length);
}

// -- selection --

// sessionStorage, not localStorage and not the URL. A remote selection that
// survives a restart, or that can be bookmarked and shared, is precisely how
// someone ends up editing the wrong machine believing it is theirs. A new tab
// starts here.
const FLEET_STORAGE_KEY = 'llmStackFleetHost';

function storedFleetHost() {
  try { return sessionStorage.getItem(FLEET_STORAGE_KEY) || ''; } catch { return ''; }
}

function rememberFleetHost(id) {
  try {
    if (id) sessionStorage.setItem(FLEET_STORAGE_KEY, id);
    else sessionStorage.removeItem(FLEET_STORAGE_KEY);
  } catch { /* a private window is not a reason to stop working */ }
}

function currentFleetHost() {
  return fleetHosts.find(h => h.id === fleetHost) || null;
}

function selectFleetHost(id) {
  const known = id && fleetHosts.some(h => h.id === id);
  fleetHost = known ? id : '';
  rememberFleetHost(fleetHost);
  applyFleetMode();
  const select = document.getElementById('fleet-host');
  if (select) select.value = fleetHost;
  poll();
}

// -- the visible difference --
//
// A page editing another machine must never look identical to one editing this
// one. The accent on the shell and the label in the header are the whole of
// that, and they are not decoration.
function applyFleetMode() {
  const remote = Boolean(fleetHost);
  document.body.dataset.fleetHost = fleetHost;
  document.getElementById('app-shell')?.classList.toggle('fleet-remote', remote);

  const banner = document.getElementById('fleet-banner');
  if (banner) {
    const host = currentFleetHost();
    banner.hidden = !remote;
    banner.textContent = remote
      ? `Viewing ${host?.label || fleetHost} — read-only. Configuration and service `
        + `controls act on the machine you are sitting at, so they are disabled here.`
      : '';
  }

  // Everything except the services view is about this machine. A tab is
  // disabled and says why; anything else is hidden outright, because a control
  // that acts on a different machine from the one named in the header is the
  // failure this picker exists to prevent -- `bulkAction('stop')` is not
  // proxied, so a live Stop All here would stop the local stack.
  document.querySelectorAll('[data-local-only]').forEach(el => {
    if (el.classList.contains('tab-btn')) {
      el.classList.toggle('is-local-only-blocked', remote);
      el.disabled = remote;
      el.title = remote ? 'This runs on the machine you are sitting at' : '';
    } else {
      el.toggleAttribute('hidden', remote);
    }
  });
  if (remote && document.querySelector('.tab-btn.active[data-local-only]')) {
    showTab('services');
  }

  document.getElementById('svc-groups-container')?.toggleAttribute('hidden', remote);
  document.getElementById('fleet-services')?.toggleAttribute('hidden', !remote);
}

// -- the remote services view --
//
// Rendered from the peer's snapshot rather than into the cards this page was
// served with. Those cards are this host's `SERVICES`, and another machine does
// not have the same ones: a Mac runs MLX units the Linux box has never heard
// of, and neither has the other's. Reusing them would silently show a service
// that is not there and hide one that is.
function renderRemoteServices(payload) {
  const el = document.getElementById('fleet-services');
  if (!el) return;
  const fleet = payload?.fleet || {};
  const services = Object.entries(payload?.services || {});
  const health = payload?.health || {};
  const contexts = payload?.contexts || {};
  const host = currentFleetHost();

  if (!services.length) {
    el.innerHTML = `<div class="fleet-empty">No answer from ${escapeHtml(host?.label || fleetHost)}`
      + `${fleet.error ? ' — ' + escapeHtml(fleet.error) : ''}.</div>`;
    return;
  }

  const stale = !fleet.ok && fleet.stale_for_seconds
    ? `<div class="fleet-stale">Last answered ${Math.round(fleet.stale_for_seconds)}s ago`
      + `${fleet.error ? ' — ' + escapeHtml(fleet.error) : ''}. Showing the last state it reported.</div>`
    : '';

  const cards = services.map(([name, state]) => {
    const entry = health[name] || {};
    const status = entry.state || state || 'unknown';
    const ctx = contexts[name];
    const perSlot = ctx ? Number(ctx.per_slot_context || 0).toLocaleString('en-US') : '';
    return `<div class="svc-card" data-status="${escapeHtml(status)}">
      <div class="card-top">
        <span class="card-name">${escapeHtml(name)}</span>
        <span class="status-pill ${escapeHtml(status)}">${escapeHtml(status)}</span>
      </div>
      <div class="card-health${status === 'stopped' ? ' muted' : ''}">${escapeHtml(entry.reason || '')}</div>
      ${ctx ? `<div class="card-ctx">${perSlot} tokens/slot${ctx.slots > 1 ? ` × ${ctx.slots}` : ''}</div>` : ''}
    </div>`;
  }).join('');

  el.innerHTML = stale + `<div class="svc-cards">${cards}</div>`;
}

// -- start-up --

async function initFleet() {
  let hosts = [];
  try {
    hosts = (await fetchJSON('/api/fleet/hosts')).hosts || [];
  } catch { return; }
  fleetHosts = hosts.filter(h => h.enabled !== false);

  const picker = document.getElementById('fleet-picker');
  const select = document.getElementById('fleet-host');
  if (!picker || !select) return;
  // A single-machine install sees no change at all.
  if (!fleetHosts.length) { picker.hidden = true; return; }

  picker.hidden = false;
  select.innerHTML = ['<option value="">This machine</option>']
    .concat(fleetHosts.map(h =>
      `<option value="${escapeHtml(h.id)}">${escapeHtml(h.label || h.id)}</option>`))
    .join('');
  selectFleetHost(storedFleetHost());
}
