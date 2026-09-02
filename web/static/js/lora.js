// Live LoRA adapter control on the Services page.
//
// The config tab decides which adapters a backend *loads* -- that is a launch
// flag and needs a restart. This decides which of the loaded ones are
// *applied*, and how strongly, which llama-server changes on a resident model
// via POST /lora-adapters. That is the whole reason adapters are applied at
// runtime rather than merged: switching fine-tunes costs a request, not a
// 17 GB reload.
//
// Rendered into the service card's own `card-lora` slot, and only when the
// backend reports adapters -- a slot with none stays exactly as it looked
// before adapters existed.

// The slots whose launchers emit --lora-scaled. Kept as a list rather than
// probing every service because the other cards are not llama-server at all.
const LORA_SLOTS = ['llm-a', 'llm-b', 'task'];

// Last known scales per slot, so a refresh mid-drag does not yank the slider
// out from under the pointer.
let loraState = {};
let loraBusy = null;

async function refreshLoraPanels() {
  for (const slot of LORA_SLOTS) {
    const host = document.getElementById('lora-' + slot);
    if (!host) continue;
    // Never redraw the panel the operator is currently working in.
    if (loraBusy === slot || host.contains(document.activeElement)) continue;
    try {
      // fetchJSON resolves on a 503 as readily as on a 200, so `ok` in the body
      // is what says whether the backend answered -- not whether this threw.
      const d = await fetchJSON(`/api/backends/${slot}/lora`);
      loraState[slot] = (d && d.ok && d.adapters) || [];
    } catch {
      // A stopped backend is the ordinary case here, not an error worth a toast.
      loraState[slot] = [];
    }
    renderLoraPanel(slot);
  }
}

function renderLoraPanel(slot) {
  const host = document.getElementById('lora-' + slot);
  if (!host) return;
  const adapters = loraState[slot] || [];
  if (!adapters.length) { host.innerHTML = ''; return; }

  const rows = adapters.map(a => {
    const scale = Number(a.scale) || 0;
    // With preloading on, every adapter boots at 0, so the live scale cannot
    // say how strongly it was meant to apply. The configured scale is the
    // target the on/off button raises it to.
    const target = Number(a.configured_scale) || 1;
    const on = scale > 0;
    return `
      <div class="lora-row${on ? ' on' : ''}">
        <button class="lora-toggle" title="${on ? 'Turn off' : `Apply at ${target.toFixed(2)}`}"
                data-slot="${escapeHtml(slot)}" data-id="${a.id}" data-target="${target}"
                onclick="loraToggle(this)">${on ? '&#9679;' : '&#9675;'}</button>
        <span class="lora-name" title="${escapeHtml(a.path || '')}">${escapeHtml(a.name || ('#' + a.id))}</span>
        <input type="range" min="0" max="1" step="0.05" value="${scale}"
               class="lora-scale" data-slot="${escapeHtml(slot)}" data-id="${a.id}"
               oninput="loraScalePreview(this)"
               onchange="loraScaleCommit(this)">
        <span class="lora-value" id="lora-val-${escapeHtml(slot)}-${a.id}">${scale.toFixed(2)}</span>
      </div>`;
  }).join('');

  host.innerHTML = `<div class="lora-panel">
      <div class="lora-hdr">Adapters</div>${rows}
    </div>`;
}

// One click to swap fine-tunes: off, or on at the strength that was configured
// for it. The slider stays for blending two adapters or dialling one back.
function loraToggle(btn) {
  const row = btn.closest('.lora-row');
  const slider = row?.querySelector('.lora-scale');
  if (!slider) return;
  const on = Number(slider.value) > 0;
  slider.value = on ? 0 : Number(btn.dataset.target) || 1;
  loraScalePreview(slider);
  loraScaleCommit(slider);
}

// Slider drag: number only, no request per pixel.
function loraScalePreview(el) {
  const out = document.getElementById(`lora-val-${el.dataset.slot}-${el.dataset.id}`);
  if (out) out.textContent = Number(el.value).toFixed(2);
  el.closest('.lora-row')?.classList.toggle('on', Number(el.value) > 0);
}

// Release: send the whole set, because llama-server replaces every scale on
// each call rather than merging the one that changed.
async function loraScaleCommit(el) {
  const slot = el.dataset.slot;
  loraBusy = slot;
  const payload = Array.from(
    document.querySelectorAll(`#lora-${CSS.escape(slot)} .lora-scale`)
  ).map(s => ({ id: Number(s.dataset.id), scale: Number(s.value) }));
  try {
    const d = await fetchJSON(`/api/backends/${slot}/lora`, 'POST', payload);
    if (!d || !d.ok) throw new Error((d && d.error) || 'no response');
    loraState[slot] = d.adapters || [];
    const named = loraState[slot].filter(a => Number(a.scale) > 0)
      .map(a => `${a.name} ${Number(a.scale).toFixed(2)}`);
    toast(named.length ? `${slot}: ${named.join(', ')}` : `${slot}: adapters off`, 'ok');
  } catch (e) {
    toast(`Could not change ${slot} adapters: ${e.message || e}`, 'err');
  } finally {
    loraBusy = null;
    renderLoraPanel(slot);
  }
}
