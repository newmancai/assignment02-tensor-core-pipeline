#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --job-name=kda-phase1-clean
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
set -euo pipefail
ulimit -c 0
experiment=/home/lcpu/YOUR_USER_ID/kda-zero-state-20260905
previous=/home/lcpu/YOUR_USER_ID/kda-mainline-20260905
base=/home/lcpu/YOUR_USER_ID/FlashKDA
export PYTHONPATH="$experiment/clean_build/lib:$previous:$previous/release_build/lib:$base/results/20260821_dispatch_boundary/integrated_dispatch/lib"
export PATH="/usr/local/cuda-13.0/bin:$PATH"
unset FLASH_KDA_K2_VALUE_SLICE FLASH_KDA_K2_DISPATCH
date -u
nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.sm --format=csv
timeout 180 "$base/.venv/bin/python" -u "$previous/test_entry_hardening.py" --extension flash_kda_phase1_C
timeout 420 "$base/.venv/bin/python" -u "$experiment/clean_probe.py" --source "$experiment/clean_source"
for checker in memcheck synccheck; do
    rc=0
    timeout 100 compute-sanitizer --tool "$checker" --error-exitcode 99 \
        "$base/.venv/bin/python" -u "$experiment/clean_probe.py" \
        --source "$experiment/clean_source" --sanitizer \
        > "$experiment/clean_${SLURM_JOB_ID}_${checker}.log" 2>&1 || rc=$?
    printf '{"kind":"sanitizer_exit","tool":"%s","exit_code":%d}\n' "$checker" "$rc"
done
date -u
