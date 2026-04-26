#!/bin/bash
# ACTi inference engine launcher (SGLang flavor).
#
# Default engine for the ACTi platform. Targets sparse MoE base models with a
# small active-parameter count, run on AMD Instinct MI3xx via the ROCm torch
# wheels. RadixAttention transparently caches the shared ~3.5k-token Sohn
# system prompt across all users — large win on a multi-user deployment.
#
# Reads required base-model identity + parser config from environment
# (typically sourced from /etc/acti/env or startup/.env at boot time). The
# base-model identity is operator-confidential and is NEVER checked into
# the public repository, in code, comments, or example values.
set -e
set -o pipefail

# --- Required from environment ---
: "${ACTI_MODEL_ID:?ACTI_MODEL_ID must be set (HuggingFace repo id or local path of the underlying base model)}"
: "${ACTI_TOOL_CALL_PARSER:?ACTI_TOOL_CALL_PARSER must be set (SGLang tool-call parser name appropriate for your base model)}"
: "${ACTI_REASONING_PARSER:?ACTI_REASONING_PARSER must be set (SGLang reasoning parser name appropriate for your base model)}"

# --- Optional, with sensible defaults ---
ACTI_PYTHON_ENV="${ACTI_PYTHON_ENV:-acti-sglang}"
ACTI_ROCM_PATH="${ACTI_ROCM_PATH:-/opt/rocm}"
ACTI_HF_HOME="${ACTI_HF_HOME:-/var/lib/acti/hf-cache}"
ACTI_LOG_DIR="${ACTI_LOG_DIR:-/var/log/acti}"
ACTI_CONTEXT_LENGTH="${ACTI_CONTEXT_LENGTH:-262144}"
ACTI_MEM_FRACTION_STATIC="${ACTI_MEM_FRACTION_STATIC:-0.85}"
ACTI_TENSOR_PARALLEL_SIZE="${ACTI_TENSOR_PARALLEL_SIZE:-1}"
ACTI_ATTENTION_BACKEND="${ACTI_ATTENTION_BACKEND:-triton}"
ACTI_INFERENCE_PORT="${ACTI_INFERENCE_PORT:-8000}"

# --- Activate the inference Python env ---
source /opt/conda/etc/profile.d/conda.sh
conda activate "$ACTI_PYTHON_ENV"

# --- ROCm runtime paths (must match the torch wheel build target) ---
export ROCM_PATH="$ACTI_ROCM_PATH"
export HIP_PATH="$ACTI_ROCM_PATH"
export LD_LIBRARY_PATH="$ACTI_ROCM_PATH/lib:${LD_LIBRARY_PATH:-}"

# --- Engine optimizations: aiter master switch + HF cache + transfer ---
export SGLANG_USE_AITER=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME="$ACTI_HF_HOME"

mkdir -p "$ACTI_LOG_DIR"
LOG="$ACTI_LOG_DIR/inference.log"
echo "[$(date -Is)] launch_sglang.sh starting (Sohn served as model alias)" | tee -a "$LOG"

exec python -m sglang.launch_server \
  --model-path "$ACTI_MODEL_ID" \
  --served-model-name Sohn \
  --tensor-parallel-size "$ACTI_TENSOR_PARALLEL_SIZE" \
  --context-length "$ACTI_CONTEXT_LENGTH" \
  --mem-fraction-static "$ACTI_MEM_FRACTION_STATIC" \
  --attention-backend "$ACTI_ATTENTION_BACKEND" \
  --tool-call-parser "$ACTI_TOOL_CALL_PARSER" \
  --reasoning-parser "$ACTI_REASONING_PARSER" \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port "$ACTI_INFERENCE_PORT" \
  2>&1 | tee -a "$LOG"
