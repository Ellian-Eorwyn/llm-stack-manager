# The code you are running, and where to find it

Two problems that both cost time on every change: the installed stack could be
running old code with nothing saying so, and `web/app.py` was 6,359 lines.

## 1. The manager does not run from the checkout you edit

`llm-manager.service` sets `WorkingDirectory=/mnt/LLMs/llamacpp/llm-stack-git`.
That tree advances only when someone runs `update.sh`. The checkout you commit
in is a different directory, and nothing connected the two.

Measured on 2026-07-29: the deployed tree was at `340a2b5` while `origin/main`
was at `59116d6` — two commits behind, one of which was the fix for a service
that keeps dying being reported as healthy. The UI showed nothing.

**A local comparison would not have caught it.** Run in that tree:

```
$ git rev-list --left-right --count origin/main...HEAD
0	0
```

Zero behind, because `origin/main` is a *cached ref* and that cache predated
both commits. Anything that reads git without fetching first reports "up to
date" for exactly as long as nobody fetches — which, on a box whose git is only
driven by `update.sh`, is the normal state.

So `web/deploy.py` fetches. On a background thread, on an interval
(`LLM_MANAGER_DEPLOY_CHECK_INTERVAL`, default 900s, `0` disables), caching the
result. `/api/deploy/status` only ever reads that cache: it is on the 5-second
UI poll path, and a network round trip there would stall every card on the page.
`POST /api/deploy/check` forces a refresh for the "Check now" button.

Three details worth knowing:

- **Root against a user-owned tree.** The manager runs as root; the checkout
  belongs to whoever installed it. Git refuses that combination with "detected
  dubious ownership" unless the directory is trusted, so every command goes
  through `deploy.git_cmd`, which sets `safe.directory`. This is the normal
  case, not an edge case.

- **Dirty outranks behind, and reports both.** `update.sh` refuses to run on a
  dirty tree. Telling an operator to update without telling them the update will
  be refused sends them into a failure, so `summarize()` reports the pending
  commit count *and* the blocking condition. Untracked files count, because
  `git status --porcelain` lists them and `update.sh` gates on that.

- **Some updates cost a model reload.** Changes under
  `deploy.BACKEND_SENSITIVE_PATHS` — the launchers, `scripts/lib/`, and
  `web/budget.py`, which launchers consult at startup — mean a running backend
  is on stale code, and only a restart picks them up. That restart reloads tens
  of GB of weights and discards a warm prompt cache, so it is reported, never
  done. `update.sh` holds the same list for its own post-update report and
  `tests/test_deploy.py` asserts the two have not drifted apart.

`llm-stack-manager status` reports the same thing from the CLI, against the
cached ref only — the CLI must not block on the network either, and the
manager's background check is what keeps that ref fresh.

## 2. Where the code lives now

`app.py` went from 6,359 lines to 2,424, and `index.html` from 5,508 to 2,177.
Nothing changed behaviour: `RouteInventoryTests` pins all 84 URL rules, and the
UI's 108 inline handlers are checked against the scripts the page loads.

| Module | What belongs in it |
|---|---|
| `core.py` | Paths derived from the install root, `ServiceManager`, subprocess and HTTP helpers, small parsers. Knows nothing about configuration or services. |
| `config_fields.py` | The field registry, restart hints, legacy key names, code↔chat mirrors. Data, no behaviour. |
| `config_env.py` | The env file and the key names in it: generous on read, strict and canonical on write. |
| `models.py` | The model catalogue — GGUF files, chat templates, custom models, transcription models, HuggingFace downloads. |
| `graphiti.py` | Neo4j queries for the memory graph, and the Markdown export. |
| `deploy.py` | The drift check above. |
| `budget.py`, `health.py`, `telemetry.py`, `scheduling.py` | As before — memory budget, service health, backend telemetry, the pi-forge slot contract. |
| `routes/graphiti.py`, `routes/models.py`, `routes/setup.py` | Blueprints, registered with no `url_prefix` so the rules are unchanged. |
| `app.py` | App creation and blueprint registration, the service tables, and the status / services / config / saved-config / OCR / SearXNG / Playwright / logs / TTS routes. |

### The one rule that is not obvious

**Reach behaviour through the module; bind data directly.**

```python
import core
core.read_meminfo()                    # yes
from core import read_meminfo          # no
from config_fields import CONFIG_FIELDS  # yes — a table, read and never replaced
```

A bound name is resolved once, at import time. Substituting `core.read_meminfo`
afterwards changes nothing the importer can see, because it is still holding the
original function object. Since those are exactly the things tests substitute —
the env reader, the paths a temp directory stands in for, the systemd calls that
must not really run — a bound import silently disconnects the substitution while
leaving the code working.

This is not hypothetical. `routes/models.py` was first written with
`from models import parse_huggingface_repo_ref`, and
`HuggingFaceRepoFileTests` began hitting the live HuggingFace API instead of its
stub. `ModuleBoundaryTests` now fails the build on any such import.

The tables in `config_fields` are exempt because they are read and never
reassigned, and `config_fields.CONFIG_FIELDS` at every use site reads worse than
it reads well.

Two other boundaries the same test enforces: no module may `import app` — that
would be a cycle, and would load a second copy under a different name when tests
load `app.py` by path — and no blueprint may carry a `url_prefix`, which would
silently rewrite every rule in its group.

### Why the UI scripts are classic scripts

`web/static/js/*.js` are loaded as plain `<script src>`, not `type="module"`.
The markup wires its buttons with 108 inline `onclick` handlers, and those
resolve against the global scope, which module scripts do not populate.

The cost is that **load order is a dependency order**, since top-level
`let`/`const` bindings are in the temporal dead zone until their script has run.
`util.js` first, `shell.js` (which owns the shared mutable state) before
anything that reads it, `boot.js` last because `boot()` runs on load and calls
into everything. `ScriptLoadOrderTests` holds that order.

Two values still come from the server, and they arrive through one small inline
bootstrap block rather than being interpolated into the middle of the
JavaScript:

```html
<script>
  window.__STACK__ = { builtinChatVariants: …, modelsDir: … };
</script>
```

Script URLs carry `?v=<short HEAD sha>` from `deploy.local_state`. Without it a
browser keeps the previous deploy's modules and runs them against the new
markup, which fails as anything except a caching problem.

## 3. What is still in `app.py`, and why

The TTS routes were left in place. Their helpers (`run_tts_manager`,
`should_use_local_tts_manager`, `wait_for_tts_gateway`) sit in one cluster with
the SearXNG and transcript service-management code, and they call
`get_service_status`, which belongs to the services area that has not been
extracted. Moving only the routes would need an import back into `app.py`, which
is the one thing the module boundaries forbid. Extracting a `services.py` first
would make the TTS routes a clean move; that is the next seam, not this one.
