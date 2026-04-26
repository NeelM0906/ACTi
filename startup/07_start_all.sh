#!/bin/bash
# Step 7: start every ACTi service in tmux. Idempotent.
set -e
echo "=== [07] start all services ==="

REPO_ROOT="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )"

# Load env (model id, parsers, HF token, optional overrides)
[ -f "$REPO_ROOT/startup/.env" ] || { echo "  startup/.env missing — copy env.example and fill in"; exit 1; }
set -a; . "$REPO_ROOT/startup/.env"; set +a

# Reload nginx to pick up our config
sudo nginx -t && sudo nginx -s reload || sudo nginx

# Helper to (re)start a service in tmux
start_service() {
  local name="$1"; shift
  local cmd="$*"
  tmux kill-session -t "$name" 2>/dev/null || true
  tmux new -d -s "$name" "$cmd; echo EXIT=\$?; exec bash"
  echo "  started: $name"
}

# Provision a default API key if the operator hasn't dropped one in yet
if ! [ -s /var/lib/acti/api-keys.txt ]; then
  KEY="sk-sohn-$(openssl rand -hex 16)"
  echo "$KEY" | sudo tee /var/lib/acti/api-keys.txt > /dev/null
  sudo chmod 600 /var/lib/acti/api-keys.txt
  echo "  provisioned new API key (SAVE THIS): $KEY"
fi

# Inference engine first (longest cold-start).
# ACTI_INFERENCE_ENGINE selects the launcher; defaults to sglang.
#   sglang  → /opt/acti/inference/launch_sglang.sh (MoE base models, default)
#   vllm    → /opt/acti/inference/launch_sohn.sh   (dense base models, fallback)
case "${ACTI_INFERENCE_ENGINE:-sglang}" in
  sglang) ENGINE_LAUNCHER="/opt/acti/inference/launch_sglang.sh" ;;
  vllm)   ENGINE_LAUNCHER="/opt/acti/inference/launch_sohn.sh" ;;
  *)      echo "  unknown ACTI_INFERENCE_ENGINE='${ACTI_INFERENCE_ENGINE}' (expected sglang|vllm)"; exit 1 ;;
esac
start_service acti-inference "$ENGINE_LAUNCHER"

# Wait for inference to bind :8000 before starting downstream services
echo -n "  waiting for inference :${ACTI_INFERENCE_PORT:-8000} "
for i in $(seq 1 60); do
  if ss -tlnp 2>/dev/null | grep -q ":${ACTI_INFERENCE_PORT:-8000}"; then
    echo "OK"
    break
  fi
  echo -n "."
  sleep 5
done

# Wait further for the engine to actually serve requests (model loading)
for i in $(seq 1 90); do
  if curl -sS "http://127.0.0.1:${ACTI_INFERENCE_PORT:-8000}/health" 2>/dev/null | grep -q .; then
    break
  fi
  sleep 5
done

start_service acti-proxy       "/opt/acti/proxy/launch_proxy.sh"
start_service acti-ui          "/opt/acti/ui/launch_owui.sh"
start_service acti-status      "/opt/acti/status/launch_status.sh"
# Mirrors /opt/acti/skills/<name>/SKILL.md into OWUI's `skill` table on a
# polling loop. The proxy hot-reloads the same files in-process, so adding a
# SKILL.md propagates to BOTH the model side and the OWUI UI within seconds.
start_service acti-skill-sync  "/opt/acti/skills/launch_skill_sync.sh"

sleep 4
echo
tmux ls
echo
echo "  health: $(curl -sS http://127.0.0.1:8888/sohn-health || echo unavailable)"
echo
echo "[07] all services started (acti-inference, acti-proxy, acti-ui, acti-status, acti-skill-sync)."
echo "  chat UI :   http://<host>:8888/"
echo "  API     :   http://<host>:8888/v1"
echo "  status  :   http://<host>:8888/status"
