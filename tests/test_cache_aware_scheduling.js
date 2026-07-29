"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const scheduling = require("../web/static/cache-aware-scheduling.js");

function compatibleValues(overrides = {}) {
  return {
    N_PARALLEL: "2",
    CTX_SIZE: "262144",
    CACHE_RAM: "8192",
    CTX_CHECKPOINTS: "8",
    CACHE_IDLE_SLOTS: "on",
    FIT: "off",
    CUSTOM_ARGS_JSON: "[]",
    ...overrides,
  };
}

// Shape of a /api/backend/budget/recommend payload for this box.
function recommendation(overrides = {}) {
  return {
    ctx_size: 262144,
    per_slot_context: 131072,
    parallel: 2,
    ctx_checkpoints: 8,
    cache_ram: 3630,
    cache_type_k: "q8_0",
    cache_type_v: "q8_0",
    cache_reuse: 0,
    swa_full: "off",
    checkpoint_each_mib: 421.6,
    ...overrides,
  };
}

test("the preset carries the host measurement, not a constant", () => {
  const values = scheduling.presetValues("CHAT_PRIMARY", recommendation());
  assert.equal(values.CHAT_PRIMARY_CTX_SIZE, "262144");
  assert.equal(values.CHAT_PRIMARY_CTX_CHECKPOINTS, "8");
  assert.equal(values.CHAT_PRIMARY_CACHE_RAM, "3630");
  // The scheduling contract itself is host-independent.
  assert.equal(values.CHAT_PRIMARY_N_PARALLEL, "2");
  assert.equal(values.CHAT_PRIMARY_CACHE_IDLE_SLOTS, "on");
  assert.equal(values.CHAT_PRIMARY_FIT, "off");
});

test("a different host produces a different preset", () => {
  const small = scheduling.presetValues("CHAT_PRIMARY",
    recommendation({ ctx_size: 65536, ctx_checkpoints: 2, cache_ram: 2048 }));
  assert.equal(small.CHAT_PRIMARY_CTX_SIZE, "65536");
  assert.equal(small.CHAT_PRIMARY_CTX_CHECKPOINTS, "2");
});

test("auto-fit off clears the minimum fit context it would otherwise contradict", () => {
  assert.equal(scheduling.presetValues("CHAT_PRIMARY", recommendation()).CHAT_PRIMARY_FIT_CTX, "");
  assert.equal(scheduling.presetValues("CHAT2").CHAT2_FIT_CTX, "");
});

test("an unmeasurable host falls back rather than asserting a number", () => {
  const values = scheduling.presetValues("CHAT2", null);
  assert.equal(values.CHAT2_CTX_SIZE, scheduling.FALLBACK.CTX_SIZE);
  assert.equal(values.CHAT2_FIT, "off");
});

test("evaluate reports total and per-slot context", () => {
  const result = scheduling.evaluate(compatibleValues(), recommendation());
  assert.equal(result.compatible, true);
  assert.equal(result.perSlotContext, 131072);
  assert.deepEqual(result.issues, []);
});

test("recommended means matching this host's measurement", () => {
  const matching = scheduling.evaluate(compatibleValues(), recommendation());
  assert.equal(matching.recommended, true);
  assert.deepEqual(matching.notes, []);

  const above = scheduling.evaluate(compatibleValues({ CTX_SIZE: "524288" }), recommendation());
  assert.equal(above.compatible, true);
  assert.equal(above.recommended, false);
  assert.match(above.notes.join(" "), /measured to fit 262,144/);
});

test("without a measurement nothing is recommended and nothing is claimed", () => {
  const result = scheduling.evaluate(compatibleValues());
  assert.equal(result.compatible, true);
  assert.equal(result.hasRecommendation, false);
  assert.equal(result.recommended, false);
  assert.deepEqual(result.notes, []);
});

test("checkpoints that overrun the prompt-cache budget are called out", () => {
  // 32 checkpoints x 2 slots x 421.6 MiB is ~27 GiB against 8 GiB of budget:
  // the configuration that was hardcoded, and that thrashed this box.
  const result = scheduling.evaluate(
    compatibleValues({ CTX_CHECKPOINTS: "32" }), recommendation());
  assert.equal(result.compatible, true);
  assert.equal(result.checkpointRamMib, 26982);
  assert.match(result.notes.join(" "), /evict on most requests/);
});

test("the minimum context is a per-slot floor, not a host-specific constant", () => {
  const ok = scheduling.evaluate(compatibleValues({ CTX_SIZE: "65536" }), recommendation());
  assert.equal(ok.compatible, true);
  assert.equal(ok.perSlotContext, 32768);

  const tooSmall = scheduling.evaluate(compatibleValues({ CTX_SIZE: "65535" }), recommendation());
  assert.equal(tooSmall.compatible, false);
  assert.match(tooSmall.issues.join(" "), /total context/i);
});

test("each scheduling requirement produces an incompatibility", () => {
  const cases = [
    ["N_PARALLEL", "1", "parallel slots"],
    ["CTX_SIZE", "1024", "total context"],
    ["CACHE_RAM", "0", "prompt-cache RAM"],
    ["CTX_CHECKPOINTS", "0", "context checkpoints"],
    ["CACHE_IDLE_SLOTS", "off", "idle-slot caching"],
    ["FIT", "on", "auto-fit"],
  ];
  for (const [key, value, message] of cases) {
    const result = scheduling.evaluate(compatibleValues({ [key]: value }), recommendation());
    assert.equal(result.compatible, false, key);
    assert.match(result.issues.join(" "), new RegExp(message, "i"), key);
  }
});

test("critical custom arguments are warned about without changing compatibility", () => {
  const result = scheduling.evaluate(compatibleValues({
    CUSTOM_ARGS_JSON: JSON.stringify(["--temp 0.2", "--parallel=1", "--no-cache-idle-slots"]),
  }), recommendation());
  assert.equal(result.compatible, true);
  assert.deepEqual(result.conflicts, ["--parallel=1", "--no-cache-idle-slots"]);
});

test("pi-forge snippet is the exact merge structure", () => {
  assert.deepEqual(JSON.parse(scheduling.piForgeSnippet()), {
    connectedServices: {
      chat: {
        scheduling: {
          enabled: true,
          interactiveSlot: 0,
          backgroundSlot: 1,
          idleGraceMs: 2000,
          yieldMs: 1000,
          backgroundOutputTokens: 4096,
        },
      },
    },
  });
});
