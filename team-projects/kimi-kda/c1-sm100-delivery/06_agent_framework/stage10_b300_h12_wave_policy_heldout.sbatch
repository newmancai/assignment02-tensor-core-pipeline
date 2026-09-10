#!/bin/bash
#SBATCH --job-name=kda-stage10-h12-heldout
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --array=0-7%1
#SBATCH --time=01:00:00
#SBATCH --output=<REMOTE_HOME>/kda-stage10-h12-heldout-%A_%a.out

set -euo pipefail

checkout=<REMOTE_HOME>/flashinfer-cake-sm103-20260908
typed_ir=<REMOTE_HOME>/kda-typed-ir-20260908
python_bin=<REMOTE_HOME>/FlashKDA/.venv/bin/python

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$checkout/.runtime-deps/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$checkout/.runtime-deps:$checkout"

cd "$checkout/benchmarks"
epoch_id="$SLURM_ARRAY_TASK_ID"
"$python_bin" "$typed_ir/account_b300_run.py" \
  --output "$typed_ir/evidence/b300_stage10_h12_wave_policy_heldout_epoch${epoch_id}_accounting.json" \
  --experiment "stage10-kimi-k3-tp8-h12-wave-policy-heldout-epoch${epoch_id}" \
  -- "$python_bin" bench_flash_kda_h12_wave_policy_heldout.py \
  --epoch-id "$epoch_id" \
  --state-rotations 256 --dry-run-iters 20 --repeat-iters 100 \
  --json "$typed_ir/evidence/b300_stage10_h12_wave_policy_heldout_epoch${epoch_id}.json"
