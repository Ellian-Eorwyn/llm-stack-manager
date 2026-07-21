(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.CacheAwareScheduling = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const PRESET = Object.freeze({
    N_PARALLEL: "2",
    CTX_SIZE: "262144",
    CACHE_RAM: "8192",
    CTX_CHECKPOINTS: "32",
    CACHE_IDLE_SLOTS: "on",
    FIT: "off",
  });

  const MINIMUM_TOTAL_CONTEXT = 164000;
  const RECOMMENDED_TOTAL_CONTEXT = 262144;
  const PI_FORGE_SETTINGS = Object.freeze({
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

  const CRITICAL_FLAGS = new Set([
    "-c",
    "--ctx-size",
    "-np",
    "--parallel",
    "-cram",
    "--cache-ram",
    "-ctxcp",
    "--ctx-checkpoints",
    "--swa-checkpoints",
    "--cache-idle-slots",
    "--no-cache-idle-slots",
    "--fit",
  ]);

  function integer(value) {
    const parsed = Number.parseInt(String(value ?? "").trim(), 10);
    return Number.isFinite(parsed) ? parsed : 0;
  }

  function customArguments(value) {
    if (Array.isArray(value)) return value.map(String);
    try {
      const parsed = JSON.parse(value || "[]");
      return Array.isArray(parsed) ? parsed.map(String) : [];
    } catch (_error) {
      return [];
    }
  }

  function firstToken(argument) {
    const trimmed = String(argument || "").trim();
    if (!trimmed) return "";
    return trimmed.split(/[\s=]/, 1)[0];
  }

  function conflictingArguments(value) {
    return customArguments(value).filter((argument) => CRITICAL_FLAGS.has(firstToken(argument)));
  }

  function evaluate(values) {
    const slots = integer(values.N_PARALLEL);
    const totalContext = integer(values.CTX_SIZE);
    const cacheRam = integer(values.CACHE_RAM);
    const checkpoints = integer(values.CTX_CHECKPOINTS);
    const idleCaching = String(values.CACHE_IDLE_SLOTS || "").toLowerCase() === "on";
    const fitDisabled = String(values.FIT || "").toLowerCase() === "off";
    const issues = [];

    if (slots < 2) issues.push("Configure at least 2 parallel slots.");
    if (totalContext < MINIMUM_TOTAL_CONTEXT) {
      issues.push(`Configure at least ${MINIMUM_TOTAL_CONTEXT.toLocaleString("en-US")} tokens of total context.`);
    }
    if (cacheRam <= 0) issues.push("Set prompt-cache RAM to a nonzero value.");
    if (checkpoints <= 0) issues.push("Set context checkpoints to a nonzero value.");
    if (!idleCaching) issues.push("Enable idle-slot caching.");
    if (!fitDisabled) issues.push("Disable auto-fit so the configured context cannot be reduced at launch.");

    const conflicts = conflictingArguments(values.CUSTOM_ARGS_JSON);
    const perSlotContext = slots > 0 ? Math.floor(totalContext / slots) : 0;
    const compatible = issues.length === 0;

    return {
      slots,
      totalContext,
      perSlotContext,
      cacheRam,
      checkpoints,
      compatible,
      recommended: compatible && totalContext >= RECOMMENDED_TOTAL_CONTEXT,
      issues,
      conflicts,
    };
  }

  function presetValues(prefix) {
    return Object.fromEntries(Object.entries(PRESET).map(([suffix, value]) => [`${prefix}_${suffix}`, value]));
  }

  function piForgeSnippet() {
    return `${JSON.stringify(PI_FORGE_SETTINGS, null, 2)}\n`;
  }

  return Object.freeze({
    PRESET,
    MINIMUM_TOTAL_CONTEXT,
    RECOMMENDED_TOTAL_CONTEXT,
    PI_FORGE_SETTINGS,
    conflictingArguments,
    evaluate,
    presetValues,
    piForgeSnippet,
  });
});
