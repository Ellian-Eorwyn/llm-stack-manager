// util.js
//
// Toast, HTML escaping, and the one fetch wrapper every panel uses.

// -- toast --
let toastT = null;
function toast(msg, type = 'info') {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className = 'show ' + type;
  if (toastT) clearTimeout(toastT);
  toastT = setTimeout(() => el.classList.remove('show'), 4000);
}

// -- fetch helper --
async function fetchJSON(url, method = 'GET', body = null) {
  const opts = { method, headers: {} };
  if (body) { opts.body = JSON.stringify(body); opts.headers['Content-Type'] = 'application/json'; }
  // fleet.js decides which machine a request is for. Called through a typeof
  // guard rather than referenced directly so util.js stays first in the load
  // order and depends on nothing -- and so a page that somehow loaded without
  // fleet.js talks to this host instead of throwing on every fetch.
  const target = (typeof fleetPath === 'function') ? fleetPath(url) : url;
  const r = await fetch(target, opts);
  return r.json();
}
