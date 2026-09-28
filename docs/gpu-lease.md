# Lending the chat GPU to ComfyUI

On llms, GPU 1 is full with the 27B (`llm-a`), and GPU 0 is shared by the task
model, embeddings, the OCR router and ComfyUI, which leaves ComfyUI about 7 GB.
A heavy image job borrows GPU 1 for a few minutes with `scripts/gpu-lease.py`.

```
gpu-lease.py acquire [--holder NAME] [--max-minutes 30] [--force]
gpu-lease.py release [--reason TEXT]
gpu-lease.py status
```

Each prints one JSON object. The exit status is non-zero, with the reason in
`error`, when something is left wrong.

## What happens

1. `acquire` writes `/run/user/<uid>/gpu1-lease.json`. From then on the proxy
   sends new generation requests to `CHAT_FALLBACK_URL` (the Studio's proxy),
   marked `X-LLM-Served-By: fallback`. It waits for the 27B's in-flight request
   (`/slots`), stops `llm-a` with `systemctl`, and restarts ComfyUI on GPU 1.
   It also arms a watchdog (`systemd-run --user`, unit `gpu1-lease-watchdog`,
   at `max-minutes` + 10) that releases the lease if nobody else does.
2. `release` waits for ComfyUI's queue to empty (up to 5 minutes, then clears
   it), unloads its models, restarts it on GPU 0, starts `llm-a` and waits for
   `/health`. Only then does it remove the lease file, so the proxy keeps using
   the fallback while the 27B loads.

Some details:

- The quiet window (`GPU_LEASE_QUIET`, default `23:00-06:30`) refuses a lease
  unless you pass `--force`: the overnight jobs and evals expect the 27B.
- The manager's recorded expectation for `llm-a` is left alone. A reboot
  mid-lease clears the tmpfs lease file, and `llm-stack-restore` brings the 27B
  back.
- A log line per acquire and release goes to `~/.local/state/gpu-lease.jsonl`.

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
Environment=COMFY_CUDA_DEVICE=0
EnvironmentFile=-%t/comfyui-device.env
ExecStart=/home/ellie/AI/ComfyUI/run-comfyui.sh --listen 100.124.56.11 --port 8188 --cuda-device ${COMFY_CUDA_DEVICE} --preview-method auto
```

**Overrides.** You can override these through the environment:
`GPU_LEASE_BACKEND_UNIT`, `GPU_LEASE_BACKEND_URL`, `GPU_LEASE_COMFY_UNIT`,
`GPU_LEASE_COMFY_URL`, `GPU_LEASE_DEVICE`, `GPU_LEASE_QUIET`, and the timeouts
(`GPU_LEASE_*_SECONDS`).
