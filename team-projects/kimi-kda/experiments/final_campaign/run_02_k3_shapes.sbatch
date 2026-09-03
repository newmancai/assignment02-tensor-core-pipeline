#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --job-name=kda-c1-k3-shapes
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --output=/home/lcpu/<USER_ID>/FlashKDA-c1-results/02_k3_shapes_%j.log

set -euo pipefail

repo=/home/lcpu/<USER_ID>/FlashKDA
campaign="$repo/profile/c1-final"
integrated="$repo/results/20260821_dispatch_boundary/integrated_dispatch/lib"
python_bin="$repo/.venv/bin/python"
job_id="${SLURM_JOB_ID:-manual}"

cd "$repo"
export PYTHONPATH="$integrated:$repo"
export PYTHONHASHSEED=0
unset FLASH_KDA_K2_VALUE_SLICE
unset FLASH_KDA_K2_DISPATCH

echo "campaign=02_k3_shapes"
echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "flashkda_commit=$(git rev-parse HEAD)"
echo "worktree_dirty_files=$(git status --short | wc -l)"
nvidia-smi --query-gpu=name,uuid,driver_version,clocks.current.sm,clocks.max.sm,memory.total --format=csv

"$python_bin" "$campaign/benchmark_k3_shapes.py" \
    --heads 12 --warmup 20 --iterations 200 --repeats 5 \
    --output "/home/lcpu/<USER_ID>/FlashKDA-c1-results/02_k3_shapes_${job_id}.csv"

nvidia-smi --query-gpu=name,uuid,clocks.current.sm,temperature.gpu,power.draw --format=csv
echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
