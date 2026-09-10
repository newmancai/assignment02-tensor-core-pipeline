#!/bin/bash
#SBATCH --job-name=kda-r9-hmma-pair
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=<REMOTE_HOME>/kda-r9-hmma-pair-%j.out
set -euo pipefail

root=<REMOTE_HOME>
round9="$root/kda-stage12-typed-ir/round9_hmma"
artifact="$root/kda-stage12-typed-ir/round9-hmma-build-$SLURM_JOB_ID"
result="$root/kda-stage12-typed-ir/round9-hmma-job-$SLURM_JOB_ID/results"
cake="$root/flashinfer-cake-sm103-20260908"
python="$root/FlashKDA/.venv/bin/python"

export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$cake/.runtime-deps/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$cake/.runtime-deps/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$cake/.runtime-deps:$cake"
python_dev="$root/FlashKDA/.deps/python312-dev/usr/include"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include:$python_dev:$python_dev/python3.12:$python_dev/x86_64-linux-gnu/python3.12${CPATH:+:$CPATH}"
export FLASH_KDA_CUDA_ARCHS=103a
export FLASH_KDA_EXTENSION_NAME=flash_kda_paired_warps_C
export MAX_JOBS=2
export NVCC_THREADS=8
export OMP_NUM_THREADS=8

nvidia-smi --query-gpu=name,uuid,clocks.sm,memory.total --format=csv
mkdir -p "$(dirname "$artifact")" "$result"
"$python" "$round9/build_hmma_paired_warps.py" \
  --base-source "$root/kda-zero-state-20260905/clean_source" \
  --patch "$round9/paired_warps.patch" \
  --output "$artifact"

"$python" "$round9/bench_hmma_paired_warps.py" \
  --paired-build-lib "$artifact/build/lib" \
  --blocks 2 --dry-run-iters 12 --repeat-iters 40 --state-rotations 64 \
  --output "$result"

sha256sum "$round9"/* "$artifact"/build/lib/*.so "$result"/summary.json \
  > "$result/identity.sha256"
