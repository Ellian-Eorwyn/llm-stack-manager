"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const scheduling = require("../web/static/cache-aware-scheduling.js");

function compatibleValues(overrides = {}) {
  return {
    N_PARALLEL: "2",
    CTX_SIZE: "262144",
    CACHE_RAM: "8192",
    CTX_CHECKPOINTS: "32",
    CACHE_IDLE_SLOTS: "on",
    FIT: "off",
    CUSTOM_ARGS_JSON: "[]",
    ...overrides,
  };
}

test("preset targets either backend prefix with the safe values", () => {
  assert.deepEqual(scheduling.presetValues("CHAT_PRIMARY"), {
    CHAT_PRIMARY_N_PARALLEL: "2",
    CHAT_PRIMARY_CTX_SIZE: "262144",
    CHAT_PRIMARY_CACHE_RAM: "8192",
    CHAT_PRIMARY_CTX_CHECKPOINTS: "32",
    CHAT_PRIMARY_CACHE_IDLE_SLOTS: "on",
    CHAT_PRIMARY_FIT: "off",
  });
  assert.equal(scheduling.presetValues("CHAT2").CHAT2_CTX_SIZE, "262144");
});

test("recommended profile reports total and per-slot context", () => {
  const result = scheduling.evaluate(compatibleValues());
  assert.equal(result.compatible, true);
  assert.equal(result.recommended, true);
  assert.equal(result.perSlotContext, 131072);
  assert.deepEqual(result.issues, []);
});

test("minimum compatible context is distinguished from the recommendation", () => {
  const result = scheduling.evaluate(compatibleValues({ CTX_SIZE: "164000" }));
  assert.equal(result.compatible, true);
  assert.equal(result.recommended, false);
  assert.equal(result.perSlotContext, 82000);
});

test("each scheduling requirement produces an incompatibility", () => {
  const cases = [
    ["N_PARALLEL", "1", "parallel slots"],
    ["CTX_SIZE", "163999", "total context"],
    ["CACHE_RAM", "0", "prompt-cache RAM"],
    ["CTX_CHECKPOINTS", "0", "context checkpoints"],
    ["CACHE_IDLE_SLOTS", "off", "idle-slot caching"],
    ["FIT", "on", "auto-fit"],
  ];
  for (const [key, value, message] of cases) {
    const result = scheduling.evaluate(compatibleValues({ [key]: value }));
    assert.equal(result.compatible, false, key);
    assert.match(result.issues.join(" "), new RegExp(message, "i"), key);
  }
});

test("critical custom arguments are warned about without changing compatibility", () => {
  const result = scheduling.evaluate(compatibleValues({
    CUSTOM_ARGS_JSON: JSON.stringify(["--temp 0.2", "--parallel=1", "--no-cache-idle-slots"]),
  }));
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
