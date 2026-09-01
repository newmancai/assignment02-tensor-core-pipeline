#!/usr/bin/env bash
# Profile one 4096^3 launch of the 4.1 kernel. Run this script inside Slurm.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ARCH="${ARCH:-100f}"
NVCC="${NVCC:-/usr/local/cuda-13.0/bin/nvcc}"
NCU="${NCU:-ncu}"
OUT_BASE="${1:-$SCRIPT_DIR/m41-detailed}"
BIN="$(mktemp /tmp/m41-ncu.XXXXXX)"
trap 'rm -f "$BIN"' EXIT

"$NVCC" -O3 -std=c++17 \
  -gencode arch="compute_${ARCH}",code="sm_${ARCH}" \
  "$SCRIPT_DIR/01_tiled.cu" -lcublas -lcuda -o "$BIN"

"$NCU" --set detailed --kernel-name-base function \
  --kernel-name 'regex:gemm_tiled_kernel' --launch-count 1 \
  --force-overwrite -o "$OUT_BASE" "$BIN" 4096 4096 4096 \
  2>&1 | tee "${OUT_BASE}.txt"
