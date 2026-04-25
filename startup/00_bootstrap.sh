#!/bin/bash
# One-shot bootstrap for a fresh AMD GPU machine.
# Runs steps 01..07 in order. Re-runnable.
#
# Prerequisites:
#   - Ubuntu 22.04 or compatible
#   - AMD GPU with kernel amdgpu driver loaded (kernel 6.10+ recommended)
#   - miniconda or anaconda installed (conda on PATH)
#   - sudo access
#   - startup/.env populated (copy from env.example)
set -e

cd "$( dirname -- "${BASH_SOURCE[0]}" )"

[ -f .env ] || { echo "Missing startup/.env — copy from env.example and fill in"; exit 1; }

bash 01_install_system.sh
bash 02_install_rocm.sh
bash 03_install_python_env.sh
# 03b builds the SGLang engine env (the default); set ACTI_SKIP_SGLANG=1 to skip
# if the operator only wants the vLLM fallback engine.
[ "${ACTI_SKIP_SGLANG:-0}" = "1" ] || bash 03b_install_sglang.sh
bash 04_patch_openwebui.sh
bash 05_install_artifacts.sh
bash 06_download_model.sh
bash 07_start_all.sh

echo
echo "=== ACTi platform is up ==="
echo "  http://$(hostname -I | awk '{print $1}'):8888/"
