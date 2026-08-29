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
// The writes are here too now, and they are the reason this list is exact
// rather than a prefix: `/api/saved-configs` is proxied, but
// `/api/saved-configs/<name>/patch` and `/default` are local-only operations
// the control API does not implement. A prefix match would have rewritten them
// into hub routes that do not exist.
// The hub routes this page may reach, by method. Method-aware because the hub's
// routes are: `/api/saved-configs` is a GET there and has no POST, so rewriting
// a save into it produced a 405 whose HTML body made `r.json()` throw. A
// path-only whitelist cannot tell those apart.
//
// A list of what *is* forwarded rather than what is not, for the same reason
// the server has one: a default of "forward it" on an unauthenticated port is
// how one page bug becomes a request to every machine in the fleet.
const FLEET_PROXIED = {
  'GET /api/status': true,
  'GET /api/config': true,
  'POST /api/config': true,
  'POST /api/config/preflight': true,
  'GET /api/saved-configs': true,
};

// The two proxied paths that carry a name in them. Anchored at both ends and
// spelled out segment by segment, so they match exactly the two hub routes and
// nothing that merely starts the same way.
const FLEET_PROXIED_PATTERNS = [
  ['POST', /^\/api\/service\/[^/]+\/[^/]+$/],
  ['POST', /^\/api\/saved-configs\/[^/]+\/apply$/],
];

function fleetProxies(path, method) {
  const verb = String(method || 'GET').toUpperCase();
  return FLEET_PROXIED[`${verb} ${path}`] === true
    || FLEET_PROXIED_PATTERNS.some(([m, rule]) => m === verb && rule.test(path));
}

function fleetPath(url, method) {
  if (!fleetHost || typeof url !== 'string') return url;
  const path = url.split('?')[0];
  if (!fleetProxies(path, method)) return url;
  return '/api/fleet/' + encodeURIComponent(fleetHost) + url.slice('/api'.length);
}

// Whether the selected peer may be written to at all. Two independent gates on
// the hub -- `control` in the registry and a matching API major -- and the page
// must not offer an editable form when either is shut, so it reads the same
// answer the hub would give rather than deciding for itself.
function fleetControllable() {
  return Boolean(fleetHost && currentFleetHost()?.controllable);
}

function fleetControlRefused() {
  return String(currentFleetHost()?.control_refused || '');
}

// Whether an element is blocked for the machine currently selected.
//
// One predicate, because there were two and they disagreed. `applyFleetMode`
// honoured the `data-fleet-control` exception and enabled the Configuration tab
// for a writable peer; `showTab` looked only at `data-local-only` and refused
// it. So the remote config form was unreachable through the UI -- the feature
// was there, tested, and could not be opened.
function fleetBlocks(el) {
  if (!fleetHost || !el || !el.hasAttribute || !el.hasAttribute('data-local-only')) {
    return false;
  }
  return !(fleetControllable() && el.hasAttribute('data-fleet-control'));
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
  const writable = fleetControllable();
  document.body.dataset.fleetHost = fleetHost;
  document.body.dataset.fleetWritable = writable ? '1' : '';
  document.getElementById('app-shell')?.classList.toggle('fleet-remote', remote);

  const banner = document.getElementById('fleet-banner');
  if (banner) {
    const host = currentFleetHost();
    const name = host?.label || fleetHost;
    banner.hidden = !remote;
    // Three different sentences, because "why can I not edit this" has three
    // different answers and a single vague one sends the operator to the wrong
    // machine to look for the cause.
    if (!remote) banner.textContent = '';
    else if (writable) {
      banner.textContent = `Editing ${name}. Saves and service actions go to that `
        + `machine, not this one.`;
    } else if (fleetControlRefused()) {
      banner.textContent = `Viewing ${name} — read-only. ${fleetControlRefused()}`;
    } else {
      banner.textContent = `Viewing ${name} — read-only. Control is off for this host; `
        + `turn it on in the host list to edit it from here.`;
    }
  }

  // Everything except the services view is about this machine. A tab is
  // disabled and says why; anything else is hidden outright, because a control
  // that acts on a different machine from the one named in the header is the
  // failure this picker exists to prevent -- `bulkAction('stop')` is not
  // proxied, so a live Stop All here would stop the local stack.
  //
  // `data-fleet-control` is the exception: a control the hub does proxy, which
  // is available on a peer this hub may write to and blocked on one it may not.
  // It is additive to `data-local-only` rather than a replacement, so a tab
  // stays local-only unless someone decides otherwise -- the default that
  // matters when the next tab is added.
  document.querySelectorAll('[data-local-only]').forEach(el => {
    const blocked = fleetBlocks(el);
    if (el.classList.contains('tab-btn')) {
      el.classList.toggle('is-local-only-blocked', blocked);
      el.disabled = blocked;
      el.title = blocked
        ? (el.hasAttribute('data-fleet-control') && fleetControlRefused())
          || 'This runs on the machine you are sitting at'
        : '';
    } else {
      el.toggleAttribute('hidden', blocked);
    }
  });
  const active = document.querySelector('.tab-btn.active[data-local-only]');
  if (active && active.disabled) {
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

  // `controllable` is the poller's answer, not the registry's: it folds in the
  // version check, which needs a round trip to the peer. `/api/fleet/hosts`
  // knows only that control was requested. Merged here so the page asks one
  // question -- "may I write to this host" -- rather than two that can disagree.
  try {
    const polled = (await fetchJSON('/api/fleet')).hosts || [];
    const byId = new Map(polled.map(h => [h.id, h]));
    fleetHosts.forEach(h => {
      const entry = byId.get(h.id);
      h.controllable = Boolean(entry?.controllable);
      h.control_refused = entry?.control_refused || '';
    });
  } catch { /* an unpolled fleet is read-only, which is the safe default */ }

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
