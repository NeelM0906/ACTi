#!/bin/bash
# Step 1: install OS-level dependencies.
# Idempotent — safe to re-run.
set -e
echo "=== [01] system dependencies ==="

apt-get update -qq
apt-get install -y --no-install-recommends \
  curl wget gnupg2 ca-certificates \
  tmux nginx \
  build-essential \
  libopenmpi3 libopenmpi-dev \
  libnuma-dev rdma-core libibverbs1 \
  python3 python3-pip \
  sqlite3

# Verify nginx is reachable
nginx -v
echo "[01] system dependencies installed."
