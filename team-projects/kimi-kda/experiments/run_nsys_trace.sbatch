#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --job-name=kda-nsys-trace
#SBATCH --output=kda_nsys_%j.log

set -euo pipefail

repo="${FLASH_KDA_REPO:-/home/lcpu/61681696/FlashKDA}"
out="${FLASH_KDA_TRACE_OUT:-$repo/results/20260901_profile_archive}"
python="$repo/.venv/bin/python"
mkdir -p "$out"
cd "$repo/results/20260821_dispatch_boundary"
export PYTHONPATH="$repo/results/20260821_dispatch_boundary/integrated_dispatch/lib:$repo"

nsys profile \
  --trace=cuda,nvtx \
  --sample=none \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --force-overwrite=true \
  --output="$out/kda_t4096_h12_v128_vs_auto" \
  "$python" "$repo/profile/k2-vsplit-opt/trace_k2_pair.py" \
  --tokens 4096 --heads 12 --iterations 5

nsys stats \
  --report nvtx_gpu_proj_trace \
  --report cuda_gpu_trace \
  --report cuda_gpu_kern_sum \
  --format csv \
  "$out/kda_t4096_h12_v128_vs_auto.nsys-rep" \
  > "$out/kda_t4096_h12_v128_vs_auto_stats.csv"

echo "trace=$out/kda_t4096_h12_v128_vs_auto.nsys-rep"
