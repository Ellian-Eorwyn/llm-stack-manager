// boot.js
//
// Start-up order.
//
// Loaded last: `boot()` runs immediately and calls into every other module.

// -- boot --
async function boot() {
  initSidebar();
  initDeployBadge();
  initCollapsibleSections();
  initCustomModelForm();
  applyGroupLayout();
  initDragAndDrop();
  startPolling();
  try {
    const setupState = await fetchJSON('/api/setup/selection');
    if (setupState.setup_required) showTab('setup');
  } catch (_) {}
  await loadGgufFiles();
  await loadTranscriptionModels('parakeet-v3');
  await loadTranscriptionModels('whisperkit-large-v3');
  await loadTranscriptionCapabilities();
  await loadCustomModels();
  await loadTtsOverview();
  await loadSavedConfigsList();
}
boot();
