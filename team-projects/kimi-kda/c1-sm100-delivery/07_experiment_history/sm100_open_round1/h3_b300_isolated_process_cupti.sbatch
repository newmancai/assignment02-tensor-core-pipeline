#!/bin/bash
#SBATCH --job-name=kda-h3-isolated
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=<REMOTE_HOME>/kda-h3-isolated-%j.out

set -euo pipefail
official=<REMOTE_HOME>/FlashKDA-c1-official
h3=<REMOTE_HOME>/FlashKDA-c1-h3-pw
cake=<REMOTE_HOME>/flashinfer-cake-sm103-20260908
ir=<REMOTE_HOME>/kda-typed-ir-20260908
py=<REMOTE_HOME>/FlashKDA/.venv/bin/python
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$cake/.runtime-deps/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$cake/.runtime-deps/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$cake/.runtime-deps:$cake"
"$py" "$ir/experiments/sm100_open_round1/bench_h3_isolated_process.py" \
 --official-source-dir "$official" --h3-source-dir "$h3" \
 --blocks 4 --state-rotations 128 --dry-run-iters 20 --repeat-iters 100 \
 --json "$ir/experiments/sm100_open_round1/evidence/h3_b300_isolated_process_cupti.json"
