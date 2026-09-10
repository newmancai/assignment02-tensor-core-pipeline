#!/bin/bash
#SBATCH --job-name=kda-cake-isolated
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=<REMOTE_HOME>/kda-cake-isolated-%j.out

set -euo pipefail
cake=<REMOTE_HOME>/flashinfer-cake-sm103-20260908
ir=<REMOTE_HOME>/kda-typed-ir-20260908
py=<REMOTE_HOME>/FlashKDA/.venv/bin/python
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$cake/.runtime-deps/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$cake/.runtime-deps/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$cake/.runtime-deps:$cake"
cd "$cake/benchmarks"
"$py" bench_recurrent_kda_prefill.py --case-set h12 --backend cake \
 --state-rotations 256 --dry-run-iters 20 --repeat-iters 100 \
 --json "$ir/experiments/sm100_open_round1/evidence/b300_stage12_cake_isolated_h12.json"
"$py" bench_recurrent_kda_prefill.py --case-set legacy --backend cake \
 --state-rotations 256 --dry-run-iters 20 --repeat-iters 100 \
 --json "$ir/experiments/sm100_open_round1/evidence/b300_stage12_cake_isolated_legacy.json"
