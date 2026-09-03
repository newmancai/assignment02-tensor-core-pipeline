#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --job-name=kda-c1-chunk
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --output=/home/lcpu/<USER_ID>/FlashKDA-c1-results/04_chunk_analysis_%j.log

set -euo pipefail

repo=/home/lcpu/<USER_ID>/FlashKDA
campaign="$repo/profile/c1-final"
python_bin="$repo/.venv/bin/python"
result_dir=/home/lcpu/<USER_ID>/FlashKDA-c1-results
job_id="${SLURM_JOB_ID:-manual}"

cd "$repo"
export PYTHONPATH="$repo"
export PYTHONHASHSEED=0
export CPATH="$repo/.deps/python312-dev/usr/include/python3.12:$repo/.deps/python312-dev/usr/include"
export FLA_FLASH_KDA=0

echo "campaign=04_chunk_analysis"
echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "flashkda_commit=$(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,uuid,driver_version,clocks.current.sm,clocks.max.sm,memory.total --format=csv

"$python_bin" "$campaign/chunk_analysis.py" \
    --head-dim 128 --sequence-length 8192 --heads 12 --lower-bound=-5 \
    --try-fla --fla-sequence-length 128 --fla-heads 1 --fla-head-dim 128 \
    --fla-warmup 5 --fla-iterations 20 \
    --output "$result_dir/04_chunk_analysis_${job_id}.csv"

nvidia-smi --query-gpu=name,uuid,clocks.current.sm,temperature.gpu,power.draw --format=csv
echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
