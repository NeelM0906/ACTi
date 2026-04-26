#!/bin/bash
# Step 3: create the Python environment, install ROCm-built torch, install the
# inference engine, install the API + UI dependencies.
set -e
ENV_NAME="${ACTI_PYTHON_ENV:-acti-inference}"
ROCM_VERSION="${ACTI_ROCM_VERSION:-7.2.1}"
echo "=== [03] python env: $ENV_NAME ==="

# Conda must already be installed; this script does not bootstrap miniconda.
command -v conda >/dev/null || { echo "  conda not found on PATH — install miniconda first"; exit 1; }

# Fresh env so we get a known-good interpreter and pip
conda env remove -n "$ENV_NAME" -y 2>/dev/null || true
conda create -n "$ENV_NAME" python=3.12 pip setuptools wheel -y

ENV_PY="/opt/conda/envs/$ENV_NAME/bin/python"
PIP="$ENV_PY -m pip --disable-pip-version-check"

# Constraint file pins torch + torchvision so transitive deps cannot silently
# replace the ROCm builds with CUDA wheels from pypi.
cat > /tmp/acti-torch-constraint.txt <<EOF
torch==2.10.0+git8514f05
triton==3.6.0
torchvision==0.24.1+d801a34
EOF

ROCM_INDEX="--index-url https://wheels.vllm.ai/rocm --extra-index-url https://pypi.org/simple"

echo "  installing ROCm torch + triton + amd-aiter ..."
$PIP install $ROCM_INDEX \
  torch==2.10.0+git8514f05 \
  triton==3.6.0 \
  torchvision==0.24.1+d801a34 \
  amd-aiter==0.1.10.post2 \
  amdsmi==26.2.2+e1a6bc5663 \
  flash-attn==2.8.3

echo "  smoke test: torch sees the GPU ..."
export ROCM_PATH="/opt/rocm-$ROCM_VERSION"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}"
$ENV_PY -c "
import torch
assert torch.version.hip, f'torch is not a ROCm build: {torch.version}'
assert torch.cuda.is_available(), 'no GPU visible to torch'
x = torch.randn(4,4,device='cuda') @ torch.randn(4,4,device='cuda')
torch.cuda.synchronize()
print(f'  OK torch={torch.__version__} hip={torch.version.hip}')
"

echo "  installing inference engine + ROCm-pinned deps ..."
$PIP install --no-deps $ROCM_INDEX vllm==0.19.1+rocm721
$PIP install -c /tmp/acti-torch-constraint.txt $ROCM_INDEX triton_kernels==1.0.0

# Hand-pick the runtime deps so we don't drag in CUDA wheels via vllm's transitive list
$PIP install -c /tmp/acti-torch-constraint.txt \
  transformers tokenizers safetensors sentencepiece tiktoken \
  scipy pillow pyyaml requests aiohttp psutil py-cpuinfo \
  msgpack cloudpickle pyzmq protobuf prometheus_client prometheus-fastapi-instrumentator \
  pydantic pydantic-settings fastapi uvicorn httpx httptools uvloop websockets \
  starlette partial-json-parser llguidance==1.3.0 xgrammar outlines_core==0.2.11 \
  openai typer rich jinja2 cachetools blake3 mistral_common setproctitle \
  huggingface_hub hf_transfer filelock tqdm regex numba==0.61.2 llvmlite==0.44.0 \
  lark==1.2.2 diskcache packaging pybase64 python-multipart typing_extensions \
  watchfiles email-validator timm datasets einops numpy pandas msgspec depyf cbor2 ray \
  compressed_tensors gguf
$PIP install -c /tmp/acti-torch-constraint.txt \
  "openai-harmony>=0.0.3" "amd-quark>=0.8.99" "anthropic>=0.71.0" \
  "conch-triton-kernels==1.2.1" \
  "grpcio==1.78.0" "grpcio-reflection==1.78.0" ijson \
  "lm-format-enforcer==0.11.3" mcp \
  "model-hosting-container-standards>=0.1.13,<1.0.0" \
  "opencv-python-headless>=4.13.0" \
  "opentelemetry-api>=1.27.0" "opentelemetry-exporter-otlp>=1.27.0" \
  "opentelemetry-sdk>=1.27.0" "opentelemetry-semantic-conventions-ai>=0.4.1" \
  peft pytest-asyncio python-json-logger \
  "runai-model-streamer==0.15.7" "runai-model-streamer-s3==0.15.7" \
  "setuptools-scm>=8" "tensorizer==2.10.1"
$PIP install $ROCM_INDEX -c /tmp/acti-torch-constraint.txt "torchaudio==2.9.0+eaa9e4e" || echo "  torchaudio optional, skipped"

echo "  installing OpenAI SDK (for tests) ..."
$PIP install openai

# acti-ui — the ACTi chat UI source tree. Lives as a git submodule at
# vendor/acti-ui/. The hatch build hook runs `npm install` + `npm run
# build` during pip install.
ACTI_UI_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )/vendor/acti-ui"
if [ ! -f "$ACTI_UI_DIR/pyproject.toml" ]; then
  echo "  vendor/acti-ui submodule missing — run 'git submodule update --init --recursive' from repo root"
  exit 1
fi
echo "  building acti-ui from $ACTI_UI_DIR ..."
$PIP install "$ACTI_UI_DIR"

# Final smoke
$ENV_PY -c "
import torch, vllm
print(f'  vllm={vllm.__version__}  torch={torch.__version__}  hip={torch.version.hip}')
"

echo "[03] python env ready."
