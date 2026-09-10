#!/bin/bash
#SBATCH --job-name=kda-r8-hmma-l3r
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=<REMOTE_HOME>/kda-r8-hmma-l3r-%j.out
set -euo pipefail

root=<REMOTE_HOME>
work="$root/kda-dual-ir-agent-rounds-20260910"
round8="$work/round8_hmma"
artifact="$work/round8-hmma-l3-job-24948/build-artifact"
result="$work/round8-hmma-l3-repair-job-$SLURM_JOB_ID/results"
cake="$root/flashinfer-cake-sm103-20260908"
python="$root/FlashKDA/.venv/bin/python"

export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$cake/.runtime-deps/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$cake/.runtime-deps/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$cake/.runtime-deps:$cake"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

nvidia-smi --query-gpu=name,uuid,clocks.sm,memory.total --format=csv
test -f "$artifact/build_receipt.json"
sha256sum "$artifact"/build/lib/flash_kda_phase1_l3_C*.so "$round8/bench_hmma_l3.py"

"$python" "$round8/bench_hmma_l3.py" \
  --l3-build-lib "$artifact/build/lib" \
  --blocks 2 --dry-run-iters 20 --repeat-iters 100 --state-rotations 128 \
  --output "$result"
