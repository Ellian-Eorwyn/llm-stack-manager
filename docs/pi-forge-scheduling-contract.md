# The pi-forge scheduling contract

pi-forge and this stack share one llama-server. They cooperate through a
contract that has always worked and was never written down, checkable, or
tested. This is the contract, and what now enforces each half of it.

## 1. Two slots, pinned by id

The backend is launched with `--parallel 2`, so llama.cpp gives each slot half
of `--ctx-size`. pi-forge then pins work to a specific slot with `id_slot` in
the request body, so a long interactive session keeps its prompt prefix in slot
0 while background skills churn slot 1.

| Half of the contract | Where it lives |
| --- | --- |
| Interactive turns pin `id_slot: 0` and `cache_prompt: true` | pi-forge `extensions/inference-scheduling.ts`, `addInteractiveSlot` |
| Background work pins `id_slot: 1` | pi-forge `lib/forge-llm.mjs` and `lib/forge_llm.py` |
| Only two providers are scheduled at all | `PROVIDER_SERVICES`: `forge-local` → think, `forge-chat-local` → chat |
| The backend is launched so slots exist to pin | `CHAT_PRIMARY_N_PARALLEL`, `_CACHE_IDLE_SLOTS`, `_FIT` |
| `id_slot` survives the proxy | `scripts/llm-chat-proxy.py` — see below |
| Both halves are checkable | `GET /api/scheduling/verify` |

Foreground calls that are not interactive send no `id_slot` at all and let
llama.cpp choose. That is deliberate, and it is why a healthy stack still shows
selections by LCP and LRU alongside the pinned ones.

### `id_slot` passes through the proxy by accident, and is now tested

The proxy never mentions `id_slot`. It works because every mutation the proxy
applies — thinking-mode injection, tool stripping, sampler overrides, memory
injection — edits the parsed payload in place, and `_body_from_json` re-encodes
it with unknown keys intact.

That is a property, not a decision. A rewrite that rebuilt the payload from
known fields would break cooperative scheduling *silently*: every request would
still succeed, just on whichever slot llama.cpp picked, and the only symptom
would be a slow drift back into prompt reprocessing.

`SlotSchedulingPassthroughTests` in `tests/test_llm_chat_proxy.py` asserts it,
including that `id_slot: 0` survives — the interactive slot is zero, so any
`if payload.get("id_slot")` guard anywhere on that path would unpin every
interactive turn while leaving slot 1 working.

## 2. The lease protocol

Slot pinning alone would let a background generation and an interactive turn run
at once. Leases are how they take turns.

| Property | Value |
| --- | --- |
| Directory | `<agent dir>/inference-leases`, agent dir from `PI_FORGE_AGENT_DIR` or `~/.pi-forge/agent` |
| Interactive filename | `<pid>-<uuid>.json` |
| Background filename | `background-<pid>-<n>.json` |
| Contents | `{pid, kind, slot, updatedAtMs}`, written `0600` via temp file and rename |
| Staleness window | 15000 ms, in `forge-llm.mjs`, `forge_llm.py` and `web/scheduling.py` |
| Missing `kind` | treated as `interactive` by every reader |

Two details that a reader has to get right:

**The trailing number in a background filename is not a timestamp.** The
JavaScript worker writes `Date.now()` there; the Python worker writes a thread
id. Age comes from `updatedAtMs` inside the file, never from the name.

**Nobody reaps.** Each writer deletes its own lease in a `finally`, which does
not run when the process is killed. Readers apply the 15-second freshness check,
so an orphan is harmless to correctness — but `activeInteractiveLeases` and
`backgroundLeaseActive` read and parse *every* file on *every* poll, and
`postPreemptible` polls each second. Orphans accumulate and are pure cost.

One was sitting there when this work started: `background-2485955-…json`, PID
long dead, `updatedAtMs` 27.5 hours old against a 15-second window.

### What the manager will and will not delete

That directory belongs to pi-forge, so the rules are conservative:

- A lease is removed only when it is past `LEASE_ORPHAN_SECONDS` (600, forty
  times the staleness window) **and** `/proc/<pid>` is gone.
- A file that will not parse is removed only once its mtime is past the same
  threshold, since it carries no timestamp of its own.
- A fresh lease, or a stale one whose process is alive, is never touched.
- It runs on request — the Reap button, or `POST /api/scheduling/leases/reap` —
  unless `PI_FORGE_LEASE_REAP=on`, which puts it on the health prober's sweep.

## 3. Verifying it

`GET /api/scheduling/verify` is **passive**. It reads:

| Source | What it establishes |
| --- | --- |
| `CHAT_PRIMARY_*` in the env | what the contract is configured to be |
| `/proc/<MainPID>/cmdline` | what the running process was actually launched with |
| `/props`, `/slots` | slot count and per-slot context as the backend sees them |
| journal, via `telemetry.summarize` | which slots have actually been pinned |

The last one is the interesting one. llama.cpp logs `selected slot by id (0)`,
telemetry has parsed that line all along, and `summarize` now keeps the count
*per slot* rather than only in total — a single `id` total cannot tell a stack
where both halves work from one where everything lands on slot 0.

Measured on this box over six hours, with no traffic generated to get it:

```
select_methods:      {"id": 62, "lcp": 22, "lru": 3}
select_by_id_slots:  {"0": 56, "1": 6}
```

56 interactive pins and 6 background pins. The contract is not just configured,
it is in use.

Drift is reported separately from failure, because it is a distinct problem:
the configuration is valid, the running backend is valid, and they are not the
same configuration — someone saved a change and did not restart.

### The active probe, and why it is opt-in

`POST /api/scheduling/verify` with `{"probe": true}` sends one `max_tokens: 1`
request pinned to each slot and reads back from the journal which slot served
it. It is behind a button with a confirmation because a request pinned to slot 0
**replaces the prompt prefix that slot is holding**. That is the eviction this
stack spent its effort removing; the passive report costs nothing and answers
the same question in the normal case.

## 4. The four proxy personas

All four ports are one process fanning into one backend on :8010. What differs
is the request shaping.

| Port | Model | Thinking | Reasoning stream | Temp | Memory gateway | Consumer |
| --- | --- | --- | --- | --- | --- | --- |
| 8003 | `think` | on | `content` | 0.7 | `MEMORY_ENABLE_THINK=on` | none today |
| 8004 | `chat` | off | `hidden` | 0.7 | `MEMORY_ENABLE_NOTHINK=on` | pi-forge `chat` / `forge-chat-local`, open-webui |
| 8008 | `code` | on | `content` | 0.6 | `MEMORY_ENABLE_CODE=off` | pi-forge `think` / `forge-local` |
| 8012 | routed | per profile | per profile | per profile | per profile | nginx → `llms:8010` |

**pi-forge's "think" service points at :8008, not :8003.** That is not a
misconfiguration. pi-forge's `think` role means "the reasoning-capable provider
for code work", and `forge-local` is the `code` persona. `:8003` is a separate
thing with the same English word attached.

**Why :8003 stays.** Its sampler settings have converged with :8008 — the two
now differ only in temperature and model name. What still separates them is the
memory gateway: `think` and `chat` are memory-backed and `code` is not, so
:8003 is the only endpoint that can be thinking-enabled *and* memory-backed.
Note that `MEMORY_GATEWAY_ENABLED=off` on this box today, so that distinction is
currently inert — it is a capability the port reserves, not one it is providing.
Retiring the port would touch `setup_engine.COMPONENT_PORTS`, the installer's
port-conflict map, `scripts/start-think.sh`, and saved profiles in
`config/saved/`, which are user data and are never rewritten. The cost is real
and the gain is a port that is already free to ignore.

**Raw chain-of-thought in `content` on :8008 is configured, not broken.**
`CODE_PRESERVE_THINKING=on` with `CODE_REASONING_STREAM_MODE=content` asks for
exactly that. :8004 looks different because it is `off` and `hidden`.

## 5. What enforces this

| Claim | Test |
| --- | --- |
| The browser and server contracts agree | `ContractParityTests.test_the_python_and_browser_contracts_agree` |
| The 15s window matches pi-forge | `ContractParityTests.test_the_stale_window_matches_pi_forge` |
| `id_slot` survives the proxy, including slot 0 | `SlotSchedulingPassthroughTests` |
| Per-slot pin counts are kept apart | `SummarizeTests.test_by_id_selections_are_counted_per_slot` |
| Only dead-writer leases are reaped | `LeaseTests.test_a_stale_lease_whose_process_lives_is_not_reaped` |
| Both filename shapes parse | `LeaseTests.test_both_writers_filename_shapes_are_read` |
| An idle session is not reported as a fault | `VerifyTests.test_an_idle_session_is_not_reported_as_a_fault` |
