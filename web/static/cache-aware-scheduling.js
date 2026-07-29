(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.CacheAwareScheduling = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  // The scheduling contract itself: what pi-forge needs from the backend
  // regardless of which model or host it runs on. Two slots so interactive and
  // background work can be pinned apart, idle-slot caching so a yielded slot
  // keeps its prefix, and auto-fit off so the launcher cannot silently shrink
  // the context a pinned slot was sized against.
  const CONTRACT = Object.freeze({
    N_PARALLEL: "2",
    CACHE_IDLE_SLOTS: "on",
    FIT: "off",
    // Auto-fit off leaves --fit-ctx with nothing to act on. llama.cpp accepts
    // both and ignores the second, which reads as configuration but is not.
    FIT_CTX: "",
  });

  // Everything below is a fallback for when the host cannot be measured — no
  // model file to read geometry from, or the budget endpoint is unreachable.
  // These were the hardcoded preset, and on the box they were written for they
  // thrashed: 32 checkpoints of a hybrid model cost ~150 MiB each before a
  // single token, several times the 8 GiB they were given. They are kept only
  // so the panel degrades to something rather than nothing.
  const FALLBACK = Object.freeze({
    ...CONTRACT,
    CTX_SIZE: "131072",
    CACHE_RAM: "4096",
    CTX_CHECKPOINTS: "4",
  });

  // Absolute floor for the contract to mean anything: two slots must each hold
  // a working interactive context. Host-derived limits replace the ceiling, not
  // this floor.
  const MINIMUM_PER_SLOT_CONTEXT = 32768;
  const DEFAULT_MINIMUM_TOTAL_CONTEXT = MINIMUM_PER_SLOT_CONTEXT * 2;

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

  /**
   * Turn a /api/backend/budget/recommend payload into the limits `evaluate`
   * measures against. Without one the thresholds fall back to the floor, which
   * is host-independent by construction — better to say nothing about the
   * ceiling than to assert a number measured on somebody else's hardware.
   */
  function limitsFrom(recommendation) {
    const recommended = integer(recommendation?.ctx_size);
    const slots = Math.max(1, integer(recommendation?.parallel) || 2);
    return {
      minimumTotalContext: DEFAULT_MINIMUM_TOTAL_CONTEXT,
      recommendedTotalContext: recommended > 0 ? recommended : 0,
      recommendedCheckpoints: integer(recommendation?.ctx_checkpoints),
      recommendedCacheRam: integer(recommendation?.cache_ram),
      checkpointEachMib: Number(recommendation?.checkpoint_each_mib) || 0,
      slots,
      hasRecommendation: recommended > 0,
    };
  }

  function evaluate(values, recommendation) {
    const limits = limitsFrom(recommendation);
    const slots = integer(values.N_PARALLEL);
    const totalContext = integer(values.CTX_SIZE);
    const cacheRam = integer(values.CACHE_RAM);
    const checkpoints = integer(values.CTX_CHECKPOINTS);
    const idleCaching = String(values.CACHE_IDLE_SLOTS || "").toLowerCase() === "on";
    const fitDisabled = String(values.FIT || "").toLowerCase() === "off";
    const issues = [];

    if (slots < 2) issues.push("Configure at least 2 parallel slots.");
    if (totalContext < limits.minimumTotalContext) {
      issues.push(
        `Configure at least ${limits.minimumTotalContext.toLocaleString("en-US")} tokens of total context ` +
        `(${MINIMUM_PER_SLOT_CONTEXT.toLocaleString("en-US")} per slot across 2 slots).`
      );
    }
    if (cacheRam <= 0) issues.push("Set prompt-cache RAM to a nonzero value.");
    if (checkpoints <= 0) issues.push("Set context checkpoints to a nonzero value.");
    if (!idleCaching) issues.push("Enable idle-slot caching.");
    if (!fitDisabled) issues.push("Disable auto-fit so the configured context cannot be reduced at launch.");

    const conflicts = conflictingArguments(values.CUSTOM_ARGS_JSON);
    const perSlotContext = slots > 0 ? Math.floor(totalContext / slots) : 0;
    const compatible = issues.length === 0;

    // The prompt cache is the term the old constants got wrong, so it is
    // reported as a distinct number rather than folded into the issue list: the
    // checkpoints are configurable, the per-checkpoint cost is not.
    const checkpointRam = limits.checkpointEachMib > 0
      ? Math.round(limits.checkpointEachMib * checkpoints * Math.max(1, slots))
      : null;
    const notes = [];
    if (limits.hasRecommendation) {
      if (totalContext > limits.recommendedTotalContext) {
        notes.push(
          `This host is measured to fit ${limits.recommendedTotalContext.toLocaleString("en-US")} tokens of ` +
          `total context; ${totalContext.toLocaleString("en-US")} is above that.`
        );
      }
      if (checkpointRam !== null && cacheRam > 0 && checkpointRam > cacheRam) {
        notes.push(
          `${checkpoints} checkpoints x ${slots} slots need about ${checkpointRam.toLocaleString("en-US")} MiB ` +
          `but prompt-cache RAM is ${cacheRam.toLocaleString("en-US")} MiB. The cache will evict on most requests.`
        );
      }
    }

    return {
      slots,
      totalContext,
      perSlotContext,
      cacheRam,
      checkpoints,
      compatible,
      // "Recommended" now means "matches what this host was measured to fit",
      // not "matches a constant". With no measurement available there is
      // nothing to match, so compatibility is all that can be claimed.
      recommended: compatible && limits.hasRecommendation
        && totalContext === limits.recommendedTotalContext,
      hasRecommendation: limits.hasRecommendation,
      recommendedTotalContext: limits.recommendedTotalContext,
      checkpointRamMib: checkpointRam,
      issues,
      notes,
      conflicts,
    };
  }

  /**
   * The values to write into the form, given a host measurement. The contract
   * keys are fixed; the memory keys come from the budget model. Falls back to
   * FALLBACK only when there is no measurement to use.
   */
  function presetValues(prefix, recommendation) {
    const preset = recommendation && integer(recommendation.ctx_size) > 0
      ? {
          ...CONTRACT,
          N_PARALLEL: String(recommendation.parallel ?? CONTRACT.N_PARALLEL),
          CTX_SIZE: String(recommendation.ctx_size),
          CACHE_RAM: String(recommendation.cache_ram),
          CTX_CHECKPOINTS: String(recommendation.ctx_checkpoints),
          CACHE_TYPE_K: String(recommendation.cache_type_k || "q8_0"),
          CACHE_TYPE_V: String(recommendation.cache_type_v || "q8_0"),
          CACHE_REUSE: String(recommendation.cache_reuse ?? 0),
          SWA_FULL: String(recommendation.swa_full || "off"),
        }
      : { ...FALLBACK };
    return Object.fromEntries(Object.entries(preset).map(([suffix, value]) => [`${prefix}_${suffix}`, value]));
  }

  function piForgeSnippet() {
    return `${JSON.stringify(PI_FORGE_SETTINGS, null, 2)}\n`;
  }

  return Object.freeze({
    CONTRACT,
    FALLBACK,
    MINIMUM_PER_SLOT_CONTEXT,
    MINIMUM_TOTAL_CONTEXT: DEFAULT_MINIMUM_TOTAL_CONTEXT,
    PI_FORGE_SETTINGS,
    conflictingArguments,
    evaluate,
    limitsFrom,
    presetValues,
    piForgeSnippet,
  });
});
