#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --job-name=kda-c1-parity
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --output=/home/lcpu/<USER_ID>/FlashKDA-c1-results/02b_baseline_parity_%j.log

set -euo pipefail

repo=/home/lcpu/<USER_ID>/FlashKDA
official=/home/lcpu/<USER_ID>/FlashKDA-c1-official
campaign="$repo/profile/c1-final"
integrated="$repo/results/20260821_dispatch_boundary/integrated_dispatch/lib"
python_bin="$repo/.venv/bin/python"
job_id="${SLURM_JOB_ID:-manual}"
output="/home/lcpu/<USER_ID>/FlashKDA-c1-results/02b_baseline_parity_${job_id}"

mkdir -p "$output"
echo "campaign=02b_baseline_parity"
echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
nvidia-smi --query-gpu=name,uuid,driver_version,clocks.current.sm,clocks.max.sm,memory.total --format=csv

env PYTHONPATH="$official" FLA_FLASH_KDA=0 \
    "$python_bin" "$campaign/baseline_parity.py" run \
    --label official --output "$output" --warmup 20 --iterations 200 --repeats 3

env PYTHONPATH="$integrated:$repo" FLASH_KDA_K2_VALUE_SLICE=128 FLA_FLASH_KDA=0 \
    "$python_bin" "$campaign/baseline_parity.py" run \
    --label patched_v128 --output "$output" --warmup 20 --iterations 200 --repeats 3

env PYTHONPATH="$repo" "$python_bin" "$campaign/baseline_parity.py" compare --output "$output"

echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
