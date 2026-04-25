# ACTi Platform — Fresh-Machine Bootstrap

This directory brings up the entire ACTi platform on a clean AMD GPU machine.

## Prerequisites

The target machine must already have:

1. **Ubuntu 22.04** (or compatible) with sudo access
2. **AMD GPU** with the kernel `amdgpu` driver loaded (kernel 6.10+ recommended)
3. **miniconda or anaconda** with `conda` on `PATH`
4. **Network egress** to `repo.radeon.com`, `huggingface.co`, `wheels.vllm.ai`, and pypi
5. **Disk:** ≥ 200 GB free for the conda env, ROCm libraries, and model weights

## Quickstart

```bash
git clone https://github.com/NeelM0906/ACTi.git
cd ACTi/startup
cp env.example .env
$EDITOR .env                # fill in ACTI_MODEL_ID, parsers, HF_TOKEN
./00_bootstrap.sh           # runs all steps in order
```

When complete, the platform is reachable on port `8888`:

| URL | What |
|---|---|
| `http://<host>:8888/` | Chat UI (sign up the first user — auto-promoted to admin) |
| `http://<host>:8888/v1` | OpenAI-compatible API base URL |
| `http://<host>:8888/status` | Status page |

The default API key is auto-generated and saved to `/var/lib/acti/api-keys.txt`. The bootstrap prints it once on first run.

## Steps in detail

The orchestrator (`00_bootstrap.sh`) calls each step in order. They can also be run individually if you need to re-run only a portion.

| Step | What it does |
|---|---|
| `01_install_system.sh` | apt deps: nginx, tmux, OpenMPI runtime, build tools |
| `02_install_rocm.sh` | ROCm 7.2.1 userspace + dev headers; cleans up older `/opt/rocm-*` from `ldconfig` |
| `03_install_python_env.sh` | Creates the `acti-inference` conda env, installs ROCm-built torch + the vLLM fallback engine + runtime deps |
| `03b_install_sglang.sh` | Clones the env to `acti-sglang` and builds SGLang from source against ROCm. SGLang is the default engine — set `ACTI_SKIP_SGLANG=1` if you only need the vLLM fallback |
| `04_patch_openwebui.sh` | One-line patch to OpenWebUI's `env.py` to remove the hardcoded `(Open WebUI)` suffix from app strings |
| `05_install_artifacts.sh` | Copies the platform code/config to `/opt/acti/`, sets up `/var/lib/acti/` and `/var/log/acti/`, installs the nginx config |
| `06_download_model.sh` | Downloads model weights into `/var/lib/acti/hf-cache` using your `HF_TOKEN` |
| `07_start_all.sh` | Starts the inference engine (selected by `ACTI_INFERENCE_ENGINE`, default `sglang`), Sohn API proxy, OpenWebUI, and the status collector — each in its own tmux session |

## Required environment variables

These four must be set in `startup/.env` before `00_bootstrap.sh` will run:

| Variable | Description |
|---|---|
| `ACTI_MODEL_ID` | HuggingFace repo id of the underlying base model. **Operator-confidential** — request from the platform team. |
| `ACTI_TOOL_CALL_PARSER` | Tool-call parser name appropriate for your base model. The same name format is used by both engines (e.g. `the tool-call parser`). |
| `ACTI_REASONING_PARSER` | Reasoning parser name that separates chain-of-thought (e.g. `the reasoning parser`). |
| `HF_TOKEN` | HuggingFace access token for weight download |

`ACTI_INFERENCE_ENGINE` (default `sglang`) selects which engine `07_start_all.sh` launches. Use `vllm` to run the dense-model fallback instead.

All other knobs (paths, ports, GPU memory, context length, attention backend) are documented with sensible defaults in `env.example`.

## Verifying the install

After `00_bootstrap.sh` completes, the bootstrap prints a `tmux ls` summary and a `/sohn-health` probe result. You should see four sessions:

```
acti-inference: 1 windows
acti-proxy:     1 windows
acti-ui:        1 windows
acti-status:    1 windows
```

And the health probe should return:

```json
{"status":"ok","model":"Sohn","version":"0.0.1"}
```

To inspect any service: `tmux attach -t acti-<name>` (Ctrl-b, d to detach).

End-to-end smoke test:

```bash
KEY=$(sudo head -1 /var/lib/acti/api-keys.txt)
curl -sS http://localhost:8888/v1/chat/completions \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"Sohn","messages":[{"role":"user","content":"hi"}],"max_tokens":50,
       "chat_template_kwargs":{"enable_thinking":false}}'
```

If you get a JSON `chat.completion` response, you're live.

## Stopping the platform

```bash
bash startup/stop_all.sh
```

Stops all four ACTi services. Does not modify nginx, the database, or the model cache. Re-run `bash startup/07_start_all.sh` to bring them back up.

## Restarting a single service

```bash
tmux kill-session -t acti-<name>
bash /opt/acti/<name>/launch_*.sh &
```

For example, to reload Sohn's system prompt after editing `platform/system_prompts/sohn.txt`:

```bash
sudo cp platform/system_prompts/sohn.txt /opt/acti/system_prompts/sohn.txt
tmux kill-session -t acti-proxy
bash /opt/acti/proxy/launch_proxy.sh &
```

## Switching engines

The default inference engine is **SGLang** (`acti-sglang` conda env, `launch_sglang.sh`), tuned for MoE base models and shared-prefix prompt caching via RadixAttention. The **vLLM** engine (`acti-inference` conda env, `launch_sohn.sh`) remains as a fallback for dense base models or operators who want the MTP speculative-decoding path.

Both engines are launched the same way; only the env vars change. To swap at runtime:

```bash
# point .env at the engine you want, then:
tmux kill-session -t acti-inference
ACTI_INFERENCE_ENGINE=vllm bash startup/07_start_all.sh
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `02_install_rocm.sh` says "package not found" | the apt repo for `$ACTI_ROCM_VERSION` may be unreachable from your network — retry, or override `ACTI_ROCM_VERSION` to a known-good release |
| torch sanity check segfaults during step 3 | older `/opt/rocm-*` libraries are still in `/etc/ld.so.conf.d/` — verify only one ROCm version is exposed via `ldconfig -p \| grep rocm` |
| inference engine fails to load model | double-check `ACTI_MODEL_ID` and `HF_TOKEN`, and that the model is downloaded under `$ACTI_HF_HOME` |
| `/status` returns 403 | nginx user (`www-data`) can't read `/usr/share/nginx/html/acti-status/` — re-run step 5 to fix permissions |
| OpenWebUI shows `(Open WebUI)` suffix | step 4 patched the wrong python env (a version upgrade may have re-installed) — re-run `04_patch_openwebui.sh` |

## Production hardening (not yet automated)

The bootstrap brings up a dev/single-machine deployment using `tmux`. Before exposing publicly you should:

- Replace tmux with `systemd` units for auto-restart and proper boot-time start
- Put a TLS terminator (Caddy / Cloudflare / load balancer) in front of nginx :8888
- Rotate the API keys in `/var/lib/acti/api-keys.txt`; never commit them
- Lock down inbound firewall to TCP/8888 only
- Restrict OpenWebUI signups to invite-only via the admin panel (after first admin signs up)
