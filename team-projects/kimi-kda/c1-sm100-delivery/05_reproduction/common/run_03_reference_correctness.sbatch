#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --job-name=kda-c1-reference
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --output=/home/lcpu/<USER_ID>/FlashKDA-c1-results/03_reference_correctness_%j.log

set -euo pipefail

repo=/home/lcpu/<USER_ID>/FlashKDA
campaign="$repo/profile/c1-final"
integrated="$repo/results/20260821_dispatch_boundary/integrated_dispatch/lib"
python_bin="$repo/.venv/bin/python"
results_dir=/home/lcpu/<USER_ID>/FlashKDA-c1-results
job_id="${SLURM_JOB_ID:-manual}"

cd "$repo"
export PYTHONPATH="$integrated:$repo"
export PYTHONHASHSEED=0
# The cluster's Python reports /usr/include/python3.12, but the headers are
# staged per user under .deps.  CPATH lets Triton's generated launcher find
# Python.h without modifying the system installation.
export CPATH="$repo/.deps/python312-dev/usr/include/python3.12:$repo/.deps/python312-dev/usr/include"
# Must be set before Python imports fla.ops.kda.chunk so the reference cannot
# dispatch back to FlashKDA.
export FLA_FLASH_KDA=0
unset FLASH_KDA_K2_VALUE_SLICE
unset FLASH_KDA_K2_DISPATCH

echo "campaign=03_reference_correctness"
echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "flashkda_commit=$(git rev-parse HEAD)"
echo "worktree_dirty_files=$(git status --short | wc -l)"
nvidia-smi --query-gpu=name,uuid,driver_version,clocks.current.sm,clocks.max.sm,memory.total --format=csv

"$python_bin" "$campaign/validate_fla_references.py" \
    --require-reference-hashes \
    --smoke-rel-rmse-limit 0.02 \
    --output "$results_dir/03_reference_correctness_${job_id}.csv"

nvidia-smi --query-gpu=name,uuid,clocks.current.sm,temperature.gpu,power.draw --format=csv
echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
