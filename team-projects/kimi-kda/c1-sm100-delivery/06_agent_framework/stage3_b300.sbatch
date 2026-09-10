#!/bin/bash
#SBATCH --job-name=kda-stage3-public
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=<REMOTE_HOME>/kda-stage3-public-%j.out

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
  --output "$typed_ir/evidence/b300_stage3_public_correctness_accounting.json" \
  --experiment stage3-public-evolution-correctness-six \
  -- "$python_bin" bench_flash_kda_evolution.py \
  --suite six --correctness-only \
  --json "$typed_ir/evidence/b300_stage3_public_correctness.json"

"$python_bin" "$typed_ir/account_b300_run.py" \
  --output "$typed_ir/evidence/b300_stage3_public_cake_accounting.json" \
  --experiment stage3-public-cake-six \
  -- "$python_bin" bench_recurrent_kda_prefill.py \
  --case-set legacy --backend cake \
  --dry-run-iters 20 --repeat-iters 100 \
  --json "$typed_ir/evidence/b300_stage3_public_cake.json"

"$python_bin" "$typed_ir/account_b300_run.py" \
  --output "$typed_ir/evidence/b300_stage3_public_evolution_accounting.json" \
  --experiment stage3-public-evolution-six \
  -- "$python_bin" bench_recurrent_kda_prefill.py \
  --case-set legacy --backend evolution \
  --dry-run-iters 20 --repeat-iters 100 \
  --json "$typed_ir/evidence/b300_stage3_public_evolution.json"
