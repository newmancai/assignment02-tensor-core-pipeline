#!/bin/bash
#SBATCH --job-name=kda-ir-smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=<REMOTE_HOME>/kda-ir-smoke-%j.out
set -euo pipefail
cake=<REMOTE_HOME>/flashinfer-cake-sm103-20260908
work=<REMOTE_HOME>/kda-ir-knowledge-smoke-20260910
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$cake/.runtime-deps/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$cake/.runtime-deps/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$cake/.runtime-deps:$cake"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
nvidia-smi --query-gpu=name,uuid,clocks.sm,memory.total --format=csv
<REMOTE_HOME>/FlashKDA/.venv/bin/python "$work/bench_ir_knowledge_smoke.py" --output "$work/job-$SLURM_JOB_ID"
