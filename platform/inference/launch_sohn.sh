#!/bin/bash
# ACTi inference engine launcher.
#
# Reads required model identity + parser config from environment (typically
# sourced from /etc/acti/env or startup/.env at boot time). The model identity
# is operator-confidential and is NOT checked into the public repository.
set -e
set -o pipefail

# --- Required from environment ---
: "${ACTI_MODEL_ID:?ACTI_MODEL_ID must be set (HuggingFace repo id of the underlying base model)}"
: "${ACTI_TOOL_CALL_PARSER:?ACTI_TOOL_CALL_PARSER must be set (vLLM tool-call parser name)}"
: "${ACTI_REASONING_PARSER:?ACTI_REASONING_PARSER must be set (vLLM reasoning parser name)}"

# --- Optional, with sensible defaults ---
ACTI_PYTHON_ENV="${ACTI_PYTHON_ENV:-acti-inference}"
ACTI_ROCM_PATH="${ACTI_ROCM_PATH:-/opt/rocm}"
ACTI_HF_HOME="${ACTI_HF_HOME:-/var/lib/acti/hf-cache}"
ACTI_LOG_DIR="${ACTI_LOG_DIR:-/var/log/acti}"
ACTI_MAX_MODEL_LEN="${ACTI_MAX_MODEL_LEN:-262144}"
ACTI_GPU_MEM_UTIL="${ACTI_GPU_MEM_UTIL:-0.9}"
ACTI_TENSOR_PARALLEL_SIZE="${ACTI_TENSOR_PARALLEL_SIZE:-1}"
ACTI_ATTENTION_BACKEND="${ACTI_ATTENTION_BACKEND:-TRITON_ATTN}"
ACTI_INFERENCE_PORT="${ACTI_INFERENCE_PORT:-8000}"

# --- Activate the inference Python env ---
source /opt/conda/etc/profile.d/conda.sh
conda activate "$ACTI_PYTHON_ENV"

# --- ROCm runtime paths (must match the torch wheel build target) ---
export ROCM_PATH="$ACTI_ROCM_PATH"
export HIP_PATH="$ACTI_ROCM_PATH"
export LD_LIBRARY_PATH="$ACTI_ROCM_PATH/lib:${LD_LIBRARY_PATH:-}"

# --- Engine optimizations (aiter kernel suite + HF cache + transfer) ---
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1
export VLLM_ROCM_USE_AITER_MHA=1
export VLLM_ROCM_USE_AITER_LINEAR=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_ROCM_USE_AITER_RMSNORM=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME="$ACTI_HF_HOME"

mkdir -p "$ACTI_LOG_DIR"
LOG="$ACTI_LOG_DIR/inference.log"
echo "[$(date -Is)] launch_sohn.sh starting (Sohn served as model alias)" | tee -a "$LOG"

exec vllm serve "$ACTI_MODEL_ID" \
  --served-model-name Sohn \
  --tensor-parallel-size "$ACTI_TENSOR_PARALLEL_SIZE" \
  --max-model-len "$ACTI_MAX_MODEL_LEN" \
  --gpu-memory-utilization "$ACTI_GPU_MEM_UTIL" \
  --reasoning-parser "$ACTI_REASONING_PARSER" \
  --enable-auto-tool-choice \
  --tool-call-parser "$ACTI_TOOL_CALL_PARSER" \
  --enable-prefix-caching \
  --attention-backend "$ACTI_ATTENTION_BACKEND" \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port "$ACTI_INFERENCE_PORT" \
  2>&1 | tee -a "$LOG"
