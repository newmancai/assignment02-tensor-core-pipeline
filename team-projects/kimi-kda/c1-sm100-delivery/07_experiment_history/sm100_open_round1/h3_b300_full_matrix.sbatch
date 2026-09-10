#!/bin/bash
#SBATCH --job-name=kda-h3-matrix
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=<REMOTE_HOME>/kda-h3-matrix-%j.out

set -euo pipefail

official_dir=<REMOTE_HOME>/FlashKDA-c1-official
h3_dir=<REMOTE_HOME>/FlashKDA-c1-h3-pw
typed_ir_dir=<REMOTE_HOME>/kda-typed-ir-20260908
python_bin=<REMOTE_HOME>/FlashKDA/.venv/bin/python

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:<REMOTE_HOME>/FlashKDA/.venv/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$official_dir:$h3_dir:$typed_ir_dir"

"$python_bin" "$typed_ir_dir/experiments/sm100_open_round1/bench_h3_matched_hmma.py" \
  --official-source-dir "$official_dir" \
  --official-commit 1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b \
  --h3-source-dir "$h3_dir" \
  --variants official_hmma,h3_matched_hmma \
  --scopes public_full \
  --state-rotations 128 \
  --dry-run-iters 20 \
  --repeat-iters 100 \
  --blocks 8 \
  --cold-l2 \
  --balanced-pair-order \
  --schema "$typed_ir_dir/experiments/sm100_open_round1/schema/h3_b300_results.schema.json" \
  --json "$typed_ir_dir/experiments/sm100_open_round1/evidence/h3_b300_full_matrix.json"
