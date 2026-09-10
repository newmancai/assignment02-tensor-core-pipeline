#!/bin/bash
#SBATCH --job-name=kda-r8-hmma-l3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:55:00
#SBATCH --output=<REMOTE_HOME>/kda-r8-hmma-l3-%j.out
set -euo pipefail

root=<REMOTE_HOME>
work="$root/kda-dual-ir-agent-rounds-20260910"
round8="$work/round8_hmma"
base_source="$root/kda-zero-state-20260905/clean_source"
artifact="$work/round8-hmma-l3-job-$SLURM_JOB_ID"
cake="$root/flashinfer-cake-sm103-20260908"
python="$root/FlashKDA/.venv/bin/python"

export CUDA_HOME=/usr/local/cuda-13.0
export MAX_JOBS=2
export NVCC_THREADS=8
export FLASH_KDA_CUDA_ARCHS=103a
export FLASH_KDA_ENABLE_V16_PREFETCH4=1
export FLASH_KDA_ENABLE_V16_PHASE1_PREFETCH=1
export FLASH_KDA_EXTENSION_NAME=flash_kda_phase1_l3_C
export FLASH_KDA_VERSION_SUFFIX=+phase1l3-20260910
export PATH="$CUDA_HOME/bin:$cake/.runtime-deps/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$cake/.runtime-deps/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$cake/.runtime-deps:$cake"
python_dev="$root/FlashKDA/.deps/python312-dev/usr/include"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include:$python_dev:$python_dev/python3.12:$python_dev/x86_64-linux-gnu/python3.12${CPATH:+:$CPATH}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

nvidia-smi --query-gpu=name,uuid,clocks.sm,memory.total --format=csv

"$python" "$round8/build_hmma_l3.py" \
  --base-source "$base_source" \
  --patch "$round8/l3.patch" \
  --output "$artifact/build-artifact"

"$python" "$round8/bench_hmma_l3.py" \
  --l3-build-lib "$artifact/build-artifact/build/lib" \
  --blocks 2 --dry-run-iters 20 --repeat-iters 100 --state-rotations 128 \
  --output "$artifact/results"
