#!/bin/bash
#SBATCH --job-name=kda-stage6-h12-pair
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=<REMOTE_HOME>/kda-stage6-h12-pair-%j.out

set -euo pipefail

checkout=<REMOTE_HOME>/flashinfer-cake-sm103-20260908
typed_ir=<REMOTE_HOME>/kda-typed-ir-20260908
python_bin=<REMOTE_HOME>/FlashKDA/.venv/bin/python

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$checkout/.runtime-deps/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$checkout/.runtime-deps:$checkout"

cd "$checkout/benchmarks"
"$python_bin" "$typed_ir/account_b300_run.py" \
  --output "$typed_ir/evidence/b300_stage6_h12_profile_pair_accounting.json" \
  --experiment stage6-kimi-k3-tp8-h12-profile-pair \
  -- "$python_bin" bench_flash_kda_h12_profile_pair.py \
  --baseline-cpc 4 --candidate-cpc 9 \
  --state-rotations 256 --dry-run-iters 20 --repeat-iters 100 \
  --json "$typed_ir/evidence/b300_stage6_h12_profile_pair.json"
