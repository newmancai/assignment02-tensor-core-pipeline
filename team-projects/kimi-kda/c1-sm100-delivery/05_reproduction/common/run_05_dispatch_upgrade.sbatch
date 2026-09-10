#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --job-name=kda-c1-dispatch-v2
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --output=/home/lcpu/<USER_ID>/FlashKDA-c1-results/05_dispatch_upgrade_%j.log

set -euo pipefail

base=/home/lcpu/<USER_ID>/FlashKDA
implementation=/home/lcpu/<USER_ID>/FlashKDA-c1-final
campaign="$base/profile/c1-final"
integrated="$base/results/20260821_dispatch_boundary/integrated_dispatch/lib"
python_bin="$base/.venv/bin/python"
job_id="${SLURM_JOB_ID:-manual}"

cd "$implementation"
export PYTHONPATH="$implementation:$integrated"
export PYTHONHASHSEED=0
unset FLASH_KDA_K2_VALUE_SLICE
unset FLASH_KDA_K2_DISPATCH

echo "campaign=05_dispatch_upgrade"
echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "flashkda_commit=$(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,uuid,driver_version,clocks.current.sm,clocks.max.sm,memory.total --format=csv

"$python_bin" "$campaign/test_single_packed_dispatch.py"
"$python_bin" "$campaign/benchmark_k3_shapes.py" \
    --heads 12 --warmup 20 --iterations 200 --repeats 3 \
    --output "/home/lcpu/<USER_ID>/FlashKDA-c1-results/05_dispatch_upgrade_${job_id}.csv"

nvidia-smi --query-gpu=name,uuid,clocks.current.sm,temperature.gpu,power.draw --format=csv
echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
