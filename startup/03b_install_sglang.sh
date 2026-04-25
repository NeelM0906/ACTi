#!/bin/bash
# Step 3b: install SGLang (the default ACTi inference engine) on top of an
# existing ROCm-built torch environment from step 03.
#
# This script clones a fresh `acti-sglang` env from the `acti-inference` env
# created in step 03, builds sgl-kernel from source against ROCm, and installs
# the SGLang python package. The existing `acti-inference` env is left
# untouched so vLLM stays available as a fallback engine.
set -e

ENV_SRC="${ACTI_PYTHON_ENV:-acti-inference}"
ENV_DST="${ACTI_SGLANG_PYTHON_ENV:-acti-sglang}"
SGL_VERSION="${ACTI_SGLANG_VERSION:-v0.5.10}"
ROCM_VERSION="${ACTI_ROCM_VERSION:-7.2.1}"
SRC_ROOT="${ACTI_SGLANG_SRC:-/opt/acti-sglang-src}"

echo "=== [03b] SGLang env: $ENV_DST  (cloned from $ENV_SRC) ==="

command -v conda >/dev/null || { echo "  conda not found on PATH"; exit 1; }
conda env list | awk '{print $1}' | grep -qx "$ENV_SRC" \
  || { echo "  source env '$ENV_SRC' not found — run 03_install_python_env.sh first"; exit 1; }

# 1. Fresh clone of the working ROCm torch env so we inherit torch / aiter.
conda env remove -n "$ENV_DST" -y 2>/dev/null || true
conda create -n "$ENV_DST" --clone "$ENV_SRC" -y

ENV_PY="/opt/conda/envs/$ENV_DST/bin/python"
PIP="$ENV_PY -m pip --disable-pip-version-check"

# 2. Pull the SGLang source matching AMD's day-0 recommendation.
sudo rm -rf "$SRC_ROOT"
sudo mkdir -p "$(dirname "$SRC_ROOT")"
sudo git clone --depth 1 -b "$SGL_VERSION" https://github.com/sgl-project/sglang.git "$SRC_ROOT"
sudo chown -R "$USER" "$SRC_ROOT"

# 3. Build sgl-kernel against the ROCm we just installed.
export ROCM_PATH="/opt/rocm-$ROCM_VERSION"
export HIP_PATH="$ROCM_PATH"
export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}"
echo "  building sgl-kernel for ROCm $ROCM_VERSION (this can take 15-30 min on a clean cache)..."
( cd "$SRC_ROOT/sgl-kernel" && "$ENV_PY" setup_rocm.py install )

# 4. Install the SGLang python package with the ROCm extras.
( cd "$SRC_ROOT" && rm -f python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml )

cat > /tmp/acti-sglang-constraint.txt <<EOF
torch==2.10.0+git8514f05
torchvision==0.24.1+d801a34
triton==3.6.0
EOF

$PIP install -c /tmp/acti-sglang-constraint.txt -e "$SRC_ROOT/python[all_hip]"

# 5. SGLang's JIT clamp_position kernel needs apache-tvm-ffi.
$PIP install apache-tvm-ffi

# 6. Smoke test.
"$ENV_PY" -c "
import sgl_kernel, sglang, tvm_ffi
import torch
assert torch.version.hip, 'torch is not a ROCm build'
assert torch.cuda.is_available(), 'no GPU visible to torch'
print('  OK sgl_kernel + sglang imports clean, torch hip', torch.version.hip)
"

echo "[03b] SGLang env ready."
