# Lending the chat GPU to ComfyUI

On llms, ComfyUI lives on GPU 1 permanently (its unit's default), idling at a
couple hundred MB beside the 27B (`llm-a`). GPU 0 holds a second 27B (`llm-b`,
since 2026-10-29; see docs/ninfer.md) and has no room for a generation model. A
generation job therefore evicts the 27B from GPU 1 for a few minutes with
`scripts/gpu-lease.py` — ComfyUI itself never moves (or restarts) in the
normal flow.

```
gpu-lease.py acquire [--holder NAME] [--max-minutes N] [--idle-minutes N] [--force] [--on-release CMD]
gpu-lease.py acquire --sticky [--force]
gpu-lease.py touch [--max-minutes N]
gpu-lease.py release [--reason TEXT] [--holder NAME] [--skip-on-release]
gpu-lease.py status
```

Each prints one JSON object. The exit status is non-zero, with the reason in
`error`, when something is left wrong.

## What happens

1. `acquire` writes `/run/user/<uid>/gpu1-lease.json`. From then on LLM A's proxy
   (only it: `follows_gpu_lease`; LLM B on GPU 0 keeps serving) sends new generation requests to `CHAT_FALLBACK_URL` (the Studio's proxy),
   marked `X-LLM-Served-By: fallback`. It waits for the 27B's in-flight request
   (`/slots`), stops `llm-a` with `systemctl`, and moves ComfyUI onto GPU 1 —
   which it is already on, so normally no restart. It also starts the
   watchdog (below).
2. `release` waits for ComfyUI's queue to empty (up to 5 minutes, then clears
   it), unloads its models (so the 27B fits back on the card), checks ComfyUI
   is on GPU 1, starts `llm-a` and waits for `/health`. Only then does it
   remove the lease file, so the proxy keeps using the fallback while the 27B
   loads. The next generation pays a one-time model load (~25 s), not a
   ComfyUI restart. If the lease has an `--on-release` command, `release` runs
   it before all of this (below).

## A holder's own cleanup: `--on-release`

A holder that puts something of its own on GPU 1, such as a model served
outside the stack with `scripts/eval-run.py serve`, must take it off before
the 27B comes back. Otherwise the 27B fails to load beside it, never answers
`/health`, and the lease file (and with it the fallback) stays for good. If
the holder dies, its `finally` never runs, and the watchdog's release would
start the 27B on a full card. So the holder hands the lease its cleanup:

```
gpu-lease.py acquire --holder bench --max-minutes 420 --force \
    --on-release "/usr/bin/python3 /path/to/llm-stack/scripts/eval-run.py stop"
```

- The lease file keeps the command (`on_release`), and `status` shows it.
- Every release runs it first, whoever releases: the holder's own, the
  watchdog's, and the one after a failed `acquire`. It runs as
  `/bin/sh -c CMD`, as the lease's user, on the lease's host, with at most
  `GPU_LEASE_ON_RELEASE_SECONDS` (default 180) to finish. The watchdog runs
  under systemd with a short `PATH`, so give absolute paths.
- It must be safe to run more than once, and when there is nothing to clean
  up. A holder that cleaned up itself still has it run once more at release.
  (`eval-run.py stop` does nothing, successfully, when nothing is served.)
- Each run is an `on-release` event in `~/.local/state/gpu-lease.jsonl` (the
  command, its exit status, how long it took), and the release's result
  carries it as `on_release`.
- **While it fails** (a non-zero exit, or the timeout) the backend is not
  started: the card may still be in use. The release reports `ok: false`, with
  the command's last output in `error`. The lease stays, so the proxy keeps
  using the fallback. The watchdog stays too, and tries the whole release
  again when the lease is due (a fixed lease: 10 minutes after `expires`),
  then every five minutes.
- **To give up on it** (a command that can never succeed), clean the card by
  hand, then `gpu-lease.py release --skip-on-release`.

It belongs to the lease the `acquire` creates (or, for an idle lease, renews).
An `acquire` that shares ComfyUI mode's sticky lease, or finds it already on,
does not add it.

## Three kinds of lease

| | fixed (default) | idle (`--idle-minutes N`) | sticky (`--sticky`, ComfyUI mode) |
|---|---|---|---|
| For | one job: `gen.py --gpu1`, `edit_bg.py` | a session: SillyTavern mode | working on generation for a while |
| Who releases | the caller, in its `finally` | the watchdog, once ComfyUI has been idle N minutes | only `release` without `--holder` (Ellie turning it off) |
| `--max-minutes` | the lease's length (default 30); the watchdog releases 10 minutes after it | a hard ceiling (default 720) | ignored: no end |
| Quiet window | — | an unforced lease ends after 5 idle minutes (`GPU_LEASE_QUIET_IDLE_MINUTES`) | turning it on needs `--force`; once on, it stays on overnight |
| `acquire` again, same holder | refused (the first caller would release under the second) | renews it | no-op (`"already": true`) |

**ComfyUI mode** (Ellie, 2026-10-02) exists because each job's own release
unloads ComfyUI's models and reloads the 27B (~35 s), so the next generation
pays both the model load and, before that, the eviction. With a sticky lease
ComfyUI keeps its models loaded between jobs. While it is on:

- `acquire --sticky` with another lease held turns that lease sticky in place
  (`"converted_from"` names the old holder); the 27B is already off.
- A job's `release --holder NAME` (the Mac's `lease.py` always passes it)
  leaves a sticky lease held by someone else alone and answers `"kept": true`.
- An idle acquire (SillyTavern mode) shares it (`"shared": true`); a fixed
  acquire is refused as "already held", and the Mac's `gpu1_lease` then
  shares it too.
- `heal` keeps its hands off, as for any lease. Overnight jobs on llms `think`
  are served by the Studio through the proxy fallback for as long as it's on.
- A reboot clears the lease file (tmpfs), which ends ComfyUI mode.

**Idle means ComfyUI did nothing, whoever its client is.** SillyTavern calls
ComfyUI directly, so the lease can't rely on its callers checking in. Each
check, the watchdog takes the newest of:

- the newest status timestamp among the last 5 entries of ComfyUI's
  `/history` (when a job started or finished);
- now, if ComfyUI's queue is not empty;
- the lease's own `last_activity` (set at acquire, by `touch`, and by each
  check, so a ComfyUI restart, which empties `/history`, doesn't reset the
  clock).

**The watchdog** is a transient user service, `gpu1-lease-watchdog`, running
`gpu-lease.py watch`: a loop that checks once a minute
(`GPU_LEASE_WATCH_SECONDS`) and releases when the lease is due. It exits when
the lease is gone. `Restart=on-failure` brings it back after a crash, and a
release that left something wrong (the 27B didn't answer) exits 1, so it is
retried five minutes later. A release by anyone else stops it, unless that
release left the lease in place (the 27B didn't answer, or an `--on-release`
command failed): then the watchdog stays, to try again when the lease is due.
`systemctl --user status gpu1-lease-watchdog` shows it.

**`touch`** marks activity now (and with `--max-minutes N` moves the end to at
least now + N). If the watchdog isn't running, it starts it again.

**`status`** adds, while a lease is held: `idle_minutes` (since the last
activity), `last_activity`, `comfy_busy`, `releases_at` (when the watchdog
will release if nothing else happens) and `watchdog_active`. `expired` is
still `expires` < now: past the length of a fixed lease, or past the ceiling
of an idle one.

## Some details

- The quiet window (`GPU_LEASE_QUIET`, default `23:00-06:30`) refuses a lease
  unless you pass `--force`: the overnight jobs and evals expect the 27B.
- The manager's recorded expectation for `llm-a` is left alone. A reboot
  mid-lease clears the tmpfs lease file, and `llm-stack-restore` brings the 27B
  back.
- A log line per acquire, touch, `on-release` command and release goes to
  `~/.local/state/gpu-lease.jsonl`. A watchdog release's `reason` says why:
  `watchdog: ComfyUI idle 60 min`, `watchdog: hard ceiling` or
  `watchdog: past its end`.

## When the 27B is up but not serving: `heal`

Found 2026-10-01 and 10-02. The lease handed GPU 1 back correctly every time,
but within minutes of a release the 27B answered `/health` with 503
`{"status":"unavailable"}` and stayed that way, process alive, until someone
restarted it by hand. The proxy served everything from the Studio meanwhile,
so the only visible sign was the doctor's `llms-model` line.

The chain:

1. After an image session, Hermes's background skill review sends the whole
   conversation to llms `think`: ~35-40k tokens, 18 tools, the generated
   images as `image_url` parts of tool results, and no `max_tokens`.
2. NInfer gives a request with no `max_tokens` the rest of the context
   (~95k tokens) and reserves it up front. With an image in the prompt that
   fails with `std::bad_alloc` (reproduced 10-02 with the 10-01 request: it
   fails uncapped and passes at 16k and 65k; a short image prompt fails
   intermittently near a full-context cap). Text-only requests at the same
   size pass.
3. Hermes retries a failed call three times. NInfer recovers from a worker
   failure only twice in a row: the third consecutive one (or any failure
   whose cleanup leaves resources behind, e.g. one that lands while a
   cancelled request is torn down) latches the engine `failed_`
   (`engine_core.h` `recover_locked`). systemd sees a running process, so
   `Restart=` never fires.

Two fixes, both on llms:

- **The proxy caps media requests** (`MEDIA_MAX_TOKENS=32768` in
  `config/llm-stack.env`; off by default). A chat or responses request with an
  image, video or audio part and no output cap, or a larger one, is capped and
  logged as `media-cap`. That removes the trigger seen in production; it does
  not make NInfer's bug impossible.
- **`gpu-lease.py heal`**, run each minute by the user timer
  `llm-a-heal.timer` (`systemd/user/`). Hands off while a lease exists, while
  the unit is not `active` (stopped on purpose or failed: systemd's business),
  within 5 minutes of the unit starting (load time), and while another
  gpu-lease command holds the lock. Otherwise, when `/health` fails on two
  checks at least 90 s apart, it restarts `llm-a` and waits for `/health`.
  At most 3 restarts an hour; past that it logs `gave up` once and leaves the
  backend to the fallback. Each restart is a `heal` event in
  `~/.local/state/gpu-lease.jsonl`, and `status` shows `last_heal` (and
  `heal_gave_up`).

Install on llms:

```
cp /mnt/LLMs/llamacpp/llm-stack-git/systemd/user/llm-a-heal.* ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now llm-a-heal.timer
```

Check: `journalctl --user -u llm-a-heal -n 5` (one JSON line per check) and
`grep '"heal"' ~/.local/state/gpu-lease.jsonl`.

## Setup

**Proxy.** In `config/llm-stack.env` on llms:

```
CHAT_FALLBACK_URL=http://studio.tailfad058.ts.net:8012
```

The Studio's proxy listens on 127.0.0.1 only. `tailscale serve` publishes it on
the tailnet as `studio.tailfad058.ts.net:8012` (tailnet only), and llms reaches
it there. Use the MagicDNS name, not the 100.x address: `tailscale serve`
routes by Host header and answers a bare IP with 404.

`AGGREGATE_EXTRA_LISTEN_HOSTS` must stay unset on the Studio. `tailscale serve`
already holds `<tailnet address>:8012`, so a second listener fails to bind.
Behind `tailscale serve` every client looks local (127.0.0.1) to the proxy, so
`PROXY_AUTH_TOKEN` does not restrict tailnet clients there. It only takes effect
for a proxy that listens on a non-loopback address itself.

**ComfyUI** (`~/.config/systemd/user/comfyui.service`) takes its GPU from a
runtime env file that the lease writes:

```
Environment=COMFY_CUDA_DEVICE=1
EnvironmentFile=-%t/comfyui-device.env
ExecStart=/home/ellie/AI/ComfyUI/run-comfyui.sh --listen 100.124.56.11 --port 8188 --cuda-device ${COMFY_CUDA_DEVICE} --preview-method auto
```

**Overrides.** You can override these through the environment:
`GPU_LEASE_BACKEND_UNIT`, `GPU_LEASE_BACKEND_URL`, `GPU_LEASE_COMFY_UNIT`,
`GPU_LEASE_COMFY_URL`, `GPU_LEASE_DEVICE`, `GPU_LEASE_QUIET`,
`GPU_LEASE_QUIET_IDLE_MINUTES`, `GPU_LEASE_HEAL_MAX_PER_HOUR`, and the timeouts and intervals
(`GPU_LEASE_*_SECONDS`, among them `GPU_LEASE_ON_RELEASE_SECONDS` for an
`--on-release` command).
