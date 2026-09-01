#!/usr/bin/env bash
# Query CUDA's maximum active blocks/SM for each required pipeline depth.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ARCH="${ARCH:-100f}"
NVCC="${NVCC:-/usr/local/cuda-13.0/bin/nvcc}"

for stages in 2 3 4 6; do
    bin="$(mktemp /tmp/pipeline-occupancy.XXXXXX)"
    trap 'rm -f "$bin"' EXIT
    "$NVCC" -O3 -std=c++17 -DSTAGES="$stages" -DOCCUPANCY_ONLY \
      -gencode arch="compute_${ARCH}",code="sm_${ARCH}" \
      "$SCRIPT_DIR/03_pipeline.cu" -lcublas -lcuda -o "$bin"
    "$bin" 4096 4096 4096
    rm -f "$bin"
    trap - EXIT
done
