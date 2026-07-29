// logs.js
//
// The journal viewer and its server-sent-events stream.

// -- logs --
function openLogs(name) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelector('.tab-btn[data-tab="logs"]').classList.add('active');
  document.getElementById('tab-logs').classList.add('active');
  document.getElementById('log-svc').value = name;
  startLogs();
}

function startLogs() {
  stopLogs();
  const svc = document.getElementById('log-svc').value;
  clearLogs();
  document.getElementById('log-start').disabled = true;
  document.getElementById('log-stop').disabled  = false;
  logSrc = new EventSource('/api/logs/' + svc);
  logSrc.onmessage = e => {
    const line = JSON.parse(e.data);
    const el = document.createElement('div');
    const ll = line.toLowerCase();
    if (ll.includes('error') || ll.includes('fail')) el.className = 'lerr';
    else if (ll.includes('warn'))                     el.className = 'lwarn';
    else if (ll.includes('info') || ll.includes('listen') || ll.includes('ready')) el.className = 'linfo';
    el.textContent = line;
    const out = document.getElementById('log-out');
    out.appendChild(el);
    if (document.getElementById('autoscroll').checked) out.scrollTop = out.scrollHeight;
  };
  logSrc.onerror = () => { stopLogs(); toast('Log stream ended', 'info'); };
}

function stopLogs() {
  if (logSrc) { logSrc.close(); logSrc = null; }
  document.getElementById('log-start').disabled = false;
  document.getElementById('log-stop').disabled  = true;
}

function clearLogs() {
  document.getElementById('log-out').textContent = '';
}

function setupCollapsibleSection(root, header) {
  if (!root || !header || root.dataset.collapseReady === '1') return;
  root.dataset.collapseReady = '1';
  header.classList.add('collapsible-trigger');
  const indicator = document.createElement('span');
  indicator.className = header.classList.contains('group-hdr')
    ? 'collapse-indicator'
    : 'panel-collapse-indicator';
  indicator.textContent = '▾';
  header.appendChild(indicator);

  const body = document.createElement('div');
  body.className = 'collapsible-body';
  while (header.nextSibling) {
    body.appendChild(header.nextSibling);
  }
  root.appendChild(body);

  header.addEventListener('click', (e) => {
    if (e.target.closest('button, a, input, select, textarea, label')) return;
    root.classList.toggle('collapsed');
  });
}

function initCollapsibleSections() {
  document.querySelectorAll('.svc-group').forEach(group => {
    setupCollapsibleSection(group, group.firstElementChild);
  });
  document.querySelectorAll('.cfg-shell').forEach(panel => {
    setupCollapsibleSection(panel, panel.firstElementChild);
  });
  const addModelPanel = document.getElementById('add-model-panel');
  if (addModelPanel) {
    setupCollapsibleSection(addModelPanel, addModelPanel.firstElementChild);
  }
}
