#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:08:00
#SBATCH --job-name=kda-zero-match
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
set -euo pipefail
ulimit -c 0
experiment=/home/lcpu/YOUR_USER_ID/kda-zero-state-20260905
previous=/home/lcpu/YOUR_USER_ID/kda-mainline-20260905
base=/home/lcpu/YOUR_USER_ID/FlashKDA
export PYTHONPATH="$previous:$previous/release_build/lib:$base/results/20260821_dispatch_boundary/integrated_dispatch/lib"
unset FLASH_KDA_K2_VALUE_SLICE FLASH_KDA_K2_DISPATCH
date -u
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
timeout 260 "$base/.venv/bin/python" -u "$experiment/matched_probe.py" --source "$previous/release_source"
for variant in release_none release_zero; do
    rc=0
    timeout 90 ncu --cache-control none --clock-control none \
        --section LaunchStats --section SchedulerStats --section WarpStateStats \
        --section Occupancy --section SpeedOfLight \
        --kernel-name 'regex:.*_flash_kda_fwd_recurrence.*' --launch-count 1 \
        --export "$experiment/matched_${SLURM_JOB_ID}_${variant}" \
        "$base/.venv/bin/python" -u "$experiment/matched_probe.py" \
        --source "$previous/release_source" --profile "$variant" \
        > "$experiment/matched_${SLURM_JOB_ID}_${variant}_ncu.log" 2>&1 || rc=$?
    printf '{"kind":"profile_exit","variant":"%s","exit_code":%d}\n' "$variant" "$rc"
done
date -u
