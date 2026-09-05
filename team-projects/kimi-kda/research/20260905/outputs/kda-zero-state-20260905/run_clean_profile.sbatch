#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:06:00
#SBATCH --job-name=kda-phase1-profile
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
set -euo pipefail
ulimit -c 0
experiment=/home/lcpu/YOUR_USER_ID/kda-zero-state-20260905
previous=/home/lcpu/YOUR_USER_ID/kda-mainline-20260905
base=/home/lcpu/YOUR_USER_ID/FlashKDA
export PYTHONPATH="$experiment/clean_build/lib:$previous:$previous/release_build/lib:$base/results/20260821_dispatch_boundary/integrated_dispatch/lib"
export PATH="/usr/local/cuda-13.0/bin:$PATH"
unset FLASH_KDA_K2_VALUE_SLICE FLASH_KDA_K2_DISPATCH
date -u
for state in out both; do
    for variant in p4 phase1; do
        rc=0
        timeout 75 ncu --cache-control none --clock-control none \
            --section SpeedOfLight --section SchedulerStats --section WarpStateStats \
            --section LaunchStats --section Occupancy --section MemoryWorkloadAnalysis \
            --kernel-name 'regex:.*_flash_kda_fwd_recurrence.*' --launch-count 1 \
            --export "$experiment/clean_${SLURM_JOB_ID}_${state}_${variant}" \
            "$base/.venv/bin/python" -u "$experiment/clean_probe.py" \
            --source "$experiment/clean_source" --profile "$variant" --state-mode "$state" \
            > "$experiment/clean_${SLURM_JOB_ID}_${state}_${variant}_ncu.log" 2>&1 || rc=$?
        printf '{"kind":"profile_exit","state":"%s","variant":"%s","exit_code":%d}\n' "$state" "$variant" "$rc"
    done
done
date -u
