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
  const r = await fetch(url, opts);
  return r.json();
}
