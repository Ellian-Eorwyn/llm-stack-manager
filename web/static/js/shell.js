// shell.js
//
// Page state, sidebar, tab switching, and the draggable service layout.
//
// Holds the mutable state the other modules read — `cfgCurrent`, `activeModel`,
// `savedConfigs` and the rest — so it must load before them.

// -- state --
let pollTimer  = null;
let logSrc     = null;
let cfgDirty   = {};
let cfgCurrent = {};
let ggufFiles  = [];
let customModels = [];
let activeModel = null;
let ttsOverview = null;
let graphitiLoaded = false;
let searxngLoaded = false;
let playwrightLoaded = false;
let setupLoaded = false;
let setupSelection = {components: [], models: {}, allow_vram_override: false};
let setupRepoFiles = {};
let customArgsDirty = false;
let customArgPresetTimer = null;
let huggingFaceRepoFiles = [];
let huggingFaceDownloadJobId = null;
let transcriptionModelsByEngine = {};
let transcriptionCapabilities = {};
let transcriptRepoInfoByEngine = {};
let transcriptDownloadJobIdByEngine = {};
const builtInChatVariants = window.__STACK__.builtinChatVariants;
const builtInChatVariantMap = Object.fromEntries(builtInChatVariants.map(v => [v.id, v]));
let savedConfigs = [];
let chatTemplates = [];
let editingChatTemplateId = '';

// -- sidebar --
function preferredSidebarCollapsed() {
  const saved = localStorage.getItem('llmStackSidebarCollapsed');
  if (saved !== null) return saved === '1';
  return window.matchMedia('(max-width: 760px)').matches;
}

function setSidebarCollapsed(collapsed) {
  const shell = document.getElementById('app-shell');
  const btn = document.getElementById('sidebar-toggle');
  if (!shell) return;
  shell.classList.toggle('sidebar-collapsed', collapsed);
  localStorage.setItem('llmStackSidebarCollapsed', collapsed ? '1' : '0');
  if (btn) {
    btn.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
  }
}

function toggleSidebar() {
  const shell = document.getElementById('app-shell');
  setSidebarCollapsed(!shell?.classList.contains('sidebar-collapsed'));
}

function initSidebar() {
  setSidebarCollapsed(preferredSidebarCollapsed());
}

// -- tabs --
function showTab(tab) {
  document.querySelectorAll('.tab-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.tab === tab);
  });
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  const panel = document.getElementById('tab-' + tab);
  if (panel) panel.classList.add('active');
  const configSideNav = document.getElementById('config-side-nav');
  if (configSideNav) configSideNav.classList.toggle('visible', tab === 'config');
  if (tab === 'config' && !Object.keys(cfgCurrent).length) loadConfig();
  if (tab === 'graphiti') initGraphitiTab();
  if (tab === 'searxng') initSearxngTab();
  if (tab === 'playwright') initPlaywrightTab();
  if (tab === 'setup') initSetupWizard();
}

document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => showTab(btn.dataset.tab));
});

function showCfgSection(name) {
  document.querySelectorAll('.cfg-tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.side-cfg-btn').forEach(b => b.classList.toggle('active', b.dataset.section === name));
  document.querySelectorAll('.cfg-section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll(`.cfg-tab-btn[data-section="${name}"]`).forEach(b => b.classList.add('active'));
  const section = document.getElementById('cfgsec-' + name.replace(/ /g, '-'));
  if (section) section.classList.add('active');
}

function openServiceConfig(sectionName) {
  showTab('config');
  if (!Object.keys(cfgCurrent).length) {
    loadConfig().then(() => showCfgSection(sectionName));
  } else {
    showCfgSection(sectionName);
  }
}

function refreshBuiltInModelButtons(updates) {
  const slotMappings = [
    { key: 'CHAT_PRIMARY_LABEL', variant: 'dense' },
  ];
  slotMappings.forEach(({ key, variant }) => {
    if (!(key in updates) || !builtInChatVariantMap[variant]) return;
    builtInChatVariantMap[variant].label = updates[key];
    const btn = document.querySelector(`.builtin-switch-btn[data-variant="${variant}"]`);
    if (btn) {
      btn.dataset.label = updates[key];
      btn.textContent = `Switch to ${updates[key]}`;
    }
  });
}

// -- drag and drop layout --
function applyGroupLayout() {
  const container = document.getElementById('svc-groups-container');
  if (!container) return;

  // Apply custom titles
  const titles = JSON.parse(localStorage.getItem('llmStackTitles') || '{}');
  container.querySelectorAll('.group-title').forEach(el => {
    const gid = el.getAttribute('data-group-id');
    if (titles[gid]) el.textContent = titles[gid];
  });

  // Apply V2 order (groups and cards)
  const layout = JSON.parse(localStorage.getItem('llmStackLayoutV2'));
  if (layout && layout.groups) {
    layout.groups.forEach(groupId => {
      const el = document.getElementById(groupId);
      if (el && el.classList.contains('svc-group')) container.appendChild(el);
    });
    
    if (layout.cards) {
      Object.keys(layout.cards).forEach(groupId => {
        const groupEl = document.getElementById(groupId);
        if (!groupEl) return;
        const cardsContainer = groupEl.querySelector('.svc-cards');
        if (!cardsContainer) return;
        layout.cards[groupId].forEach(cardId => {
          const cardEl = document.getElementById(cardId);
          if (cardEl && cardEl.classList.contains('svc-card')) {
            cardsContainer.appendChild(cardEl);
          }
        });
      });
    }
  } else {
    // Fallback for V1 order (only groups)
    const order = JSON.parse(localStorage.getItem('llmStackLayout') || '[]');
    order.forEach(id => {
      const el = document.getElementById(id);
      if (el && el.classList.contains('svc-group')) container.appendChild(el);
    });
  }
}

function saveGroupLayout() {
  const container = document.getElementById('svc-groups-container');
  if (!container) return;
  
  const layout = { groups: [], cards: {} };
  
  Array.from(container.children).forEach(group => {
    if (group.classList.contains('svc-group')) {
      layout.groups.push(group.id);
      layout.cards[group.id] = [];
      const cardsContainer = group.querySelector('.svc-cards');
      if (cardsContainer) {
        Array.from(cardsContainer.children).forEach(card => {
          if (card.classList.contains('svc-card')) {
            layout.cards[group.id].push(card.id);
          }
        });
      }
    }
  });
  
  localStorage.setItem('llmStackLayoutV2', JSON.stringify(layout));
}

function initDragAndDrop() {
  const container = document.getElementById('svc-groups-container');
  if (!container) return;
  
  let draggedEl = null;
  let draggedType = null; // 'group' or 'card'

  container.addEventListener('mouseover', e => {
    const handle = e.target.closest('.drag-handle, .card-drag-handle');
    if (handle) {
      const isCard = handle.classList.contains('card-drag-handle');
      const target = isCard ? e.target.closest('.svc-card') : e.target.closest('.svc-group');
      if (target) target.setAttribute('draggable', 'true');
    }
  });
  
  container.addEventListener('mouseout', e => {
    const handle = e.target.closest('.drag-handle, .card-drag-handle');
    if (handle) {
      const isCard = handle.classList.contains('card-drag-handle');
      const target = isCard ? e.target.closest('.svc-card') : e.target.closest('.svc-group');
      if (target) target.removeAttribute('draggable');
    }
  });

  container.addEventListener('dragstart', e => {
    const card = e.target.closest('.svc-card');
    const group = e.target.closest('.svc-group');
    
    if (card && card.getAttribute('draggable') === 'true') {
      draggedEl = card;
      draggedType = 'card';
    } else if (group && group.getAttribute('draggable') === 'true') {
      draggedEl = group;
      draggedType = 'group';
    } else {
      return;
    }
    
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/plain', draggedEl.id); 
    draggedEl.classList.add('dragging');
    e.stopPropagation(); // prevent card drag from bubbling to group
  });

  container.addEventListener('dragover', e => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    if (!draggedEl) return;
    
    if (draggedType === 'group') {
      const group = e.target.closest('.svc-group');
      if (group && group !== draggedEl && !draggedEl.contains(group)) {
        const rect = group.getBoundingClientRect();
        if (e.clientY < rect.top + rect.height / 2) {
          group.parentNode.insertBefore(draggedEl, group);
        } else {
          group.parentNode.insertBefore(draggedEl, group.nextSibling);
        }
      }
    } else if (draggedType === 'card') {
      const targetCard = e.target.closest('.svc-card');
      const targetContainer = e.target.closest('.svc-cards');
      
      if (targetCard && targetCard !== draggedEl) {
        const rect = targetCard.getBoundingClientRect();
        if (e.clientY < rect.top + rect.height / 2) {
          targetCard.parentNode.insertBefore(draggedEl, targetCard);
        } else {
          targetCard.parentNode.insertBefore(draggedEl, targetCard.nextSibling);
        }
      } else if (targetContainer && !targetCard && targetContainer !== draggedEl.parentNode) {
        // Dropping into an empty container or at the bottom of a container
        targetContainer.appendChild(draggedEl);
      }
    }
  });

  container.addEventListener('dragend', e => {
    if (draggedEl) {
      draggedEl.classList.remove('dragging');
      draggedEl.removeAttribute('draggable');
      draggedEl = null;
      draggedType = null;
      saveGroupLayout();
    }
  });

  // Save titles on edit
  container.addEventListener('input', e => {
    if (e.target.classList.contains('group-title')) {
      const titles = JSON.parse(localStorage.getItem('llmStackTitles') || '{}');
      titles[e.target.getAttribute('data-group-id')] = e.target.textContent;
      localStorage.setItem('llmStackTitles', JSON.stringify(titles));
    }
  });
}
