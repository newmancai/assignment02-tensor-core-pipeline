#!/bin/bash
#SBATCH --job-name=kda-stage9-vnext
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=<REMOTE_HOME>/kda-stage9-vnext-%j.out

set -euo pipefail

checkout=<REMOTE_HOME>/flashinfer-cake-sm103-20260908
typed_ir=<REMOTE_HOME>/kda-typed-ir-20260908
python_bin=<REMOTE_HOME>/FlashKDA/.venv/bin/python

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$checkout/.runtime-deps/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$typed_ir:$checkout/.runtime-deps:$checkout:$checkout/benchmarks"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

test "$(nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l | tr -d ' ')" = 1
cd "$checkout/benchmarks"

"$python_bin" "$typed_ir/account_b300_run.py" \
  --output "$typed_ir/evidence/b300_stage9_vnext_niche_accounting.json" \
  --experiment stage9-vnext-development-niche-screen \
  -- "$python_bin" "$typed_ir/stage9_vnext_b300.py" \
  --manifest "$typed_ir/evidence/proposal_manifest_v5.json" \
  --development "$typed_ir/evidence/stage9_development_summary.json" \
  --output "$typed_ir/evidence/b300_stage9_vnext_niche_report.json" \
  --state-rotations 256 \
  --dry-run-iters 5 \
  --screen-repeat-iters 25 \
  --qualification-repeat-iters 100
