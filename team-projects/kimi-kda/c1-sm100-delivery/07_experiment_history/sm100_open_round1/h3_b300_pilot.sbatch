#!/bin/bash
#SBATCH --job-name=kda-h3-pilot
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=<REMOTE_HOME>/kda-h3-pilot-%j.out

set -euo pipefail

official_dir=<REMOTE_HOME>/FlashKDA-c1-official
h3_dir=<REMOTE_HOME>/FlashKDA-c1-h3-pw
typed_ir_dir=<REMOTE_HOME>/kda-typed-ir-20260908
python_bin=<REMOTE_HOME>/FlashKDA/.venv/bin/python
runner="$typed_ir_dir/experiments/sm100_open_round1/bench_h3_matched_hmma.py"
schema="$typed_ir_dir/experiments/sm100_open_round1/schema/h3_b300_results.schema.json"
result="$typed_ir_dir/experiments/sm100_open_round1/evidence/h3_b300_pilot.json"

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:<REMOTE_HOME>/FlashKDA/.venv/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$official_dir:$h3_dir:$typed_ir_dir"

test -n "$(compgen -G "$official_dir/flash_kda_C*.so")"
test -n "$(compgen -G "$h3_dir/flash_kda_h3_C*.so")"

"$python_bin" "$runner" \
  --official-source-dir "$official_dir" \
  --official-commit 1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b \
  --h3-source-dir "$h3_dir" \
  --pilot \
  --state-rotations 512 \
  --dry-run-iters 20 \
  --repeat-iters 100 \
  --blocks 8 \
  --cold-l2 \
  --balanced-pair-order \
  --schema "$schema" \
  --json "$result"
