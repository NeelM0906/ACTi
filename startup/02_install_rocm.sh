#!/bin/bash
# Step 2: install ROCm 7.2.1 userspace + dev libraries needed by the inference engine.
# Skip this step on machines that already have ROCm 7.2.x installed and matching
# the kernel driver. Verify with: ls /opt/rocm-7.2.1/lib/libamdhip64.so.7
set -e
ROCM_VERSION="${ACTI_ROCM_VERSION:-7.2.1}"
ROCM_REPO_VERSION="${ROCM_VERSION%.*}/${ROCM_VERSION}"
echo "=== [02] ROCm $ROCM_VERSION userspace ==="

if [ -d "/opt/rocm-$ROCM_VERSION" ]; then
  echo "  /opt/rocm-$ROCM_VERSION already exists — skipping repo + install"
else
  mkdir -p /etc/apt/keyrings
  wget -qO- https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor > /etc/apt/keyrings/rocm.gpg
  echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/$ROCM_VERSION/ jammy main" \
    > /etc/apt/sources.list.d/rocm-$ROCM_VERSION.list
  apt-get update -qq
fi

# Runtime libs
apt-get install -y --no-install-recommends \
  rocm-llvm$ROCM_VERSION hip-runtime-amd$ROCM_VERSION rocminfo$ROCM_VERSION \
  hipblas$ROCM_VERSION hipblaslt$ROCM_VERSION hipsparse$ROCM_VERSION hipsparselt$ROCM_VERSION \
  hipsolver$ROCM_VERSION hipfft$ROCM_VERSION hiprand$ROCM_VERSION \
  rocblas$ROCM_VERSION rocsolver$ROCM_VERSION rocsparse$ROCM_VERSION rocfft$ROCM_VERSION \
  rocrand$ROCM_VERSION rccl$ROCM_VERSION miopen-hip$ROCM_VERSION \
  rocprofiler-sdk$ROCM_VERSION hsa-amd-aqlprofile$ROCM_VERSION \
  rocm-device-libs$ROCM_VERSION

# Dev headers (needed by aiter JIT compilation)
apt-get install -y --no-install-recommends \
  hipsparse-dev$ROCM_VERSION hipblas-dev$ROCM_VERSION hipblaslt-dev$ROCM_VERSION \
  rocblas-dev$ROCM_VERSION rocsparse-dev$ROCM_VERSION rocsolver-dev$ROCM_VERSION \
  hipsolver-dev$ROCM_VERSION hipfft-dev$ROCM_VERSION rocfft-dev$ROCM_VERSION \
  hiprand-dev$ROCM_VERSION rocrand-dev$ROCM_VERSION rccl-dev$ROCM_VERSION \
  miopen-hip-dev$ROCM_VERSION hipsparselt-dev$ROCM_VERSION \
  composablekernel-dev$ROCM_VERSION rocprim-dev$ROCM_VERSION hipcub-dev$ROCM_VERSION \
  rocthrust-dev$ROCM_VERSION hsa-rocr-dev$ROCM_VERSION hip-dev$ROCM_VERSION

# Repoint /opt/rocm to the version we just installed (matching torch wheel target)
update-alternatives --install /opt/rocm rocm /opt/rocm-$ROCM_VERSION 100 || true
ln -sfn /opt/rocm-$ROCM_VERSION /etc/alternatives/rocm

# Remove any stale /opt/rocm-OLD entries from ldconfig path so torch doesn't load mixed sonames
for f in /etc/ld.so.conf.d/*.conf; do
  [ -f "$f" ] || continue
  if grep -q "rocm-[0-9]" "$f" && ! grep -q "rocm-$ROCM_VERSION" "$f"; then
    rm -v "$f"
  fi
done
ldconfig

# Sanity
/opt/rocm-$ROCM_VERSION/bin/rocm-smi --showproductname | head -8
echo "[02] ROCm $ROCM_VERSION installed."
