#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --job-name=kda-release-0905
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
set -euo pipefail
ulimit -c 0
experiment=/home/lcpu/YOUR_USER_ID/kda-mainline-20260905
base=/home/lcpu/YOUR_USER_ID/FlashKDA
export PYTHONPATH="$experiment/release_build/lib:$base/results/20260821_dispatch_boundary/integrated_dispatch/lib"
export PATH="/usr/local/cuda-13.0/bin:$PATH"
unset FLASH_KDA_K2_VALUE_SLICE FLASH_KDA_K2_DISPATCH
date -u
nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.sm --format=csv
timeout 180 "$base/.venv/bin/python" -u "$experiment/test_entry_hardening.py" --extension flash_kda_release_C
timeout 360 "$base/.venv/bin/python" -u "$experiment/release_probe.py" --source "$experiment/release_source"
for checker in memcheck synccheck; do
    rc=0
    timeout 180 compute-sanitizer --tool "$checker" --error-exitcode 99 \
        "$base/.venv/bin/python" -u "$experiment/release_probe.py" \
        --source "$experiment/release_source" --sanitizer \
        > "$experiment/release_${SLURM_JOB_ID}_${checker}.log" 2>&1 || rc=$?
    printf '{"kind":"sanitizer_exit","tool":"%s","exit_code":%d}\n' "$checker" "$rc"
done
for variant in baseline release; do
    rc=0
    timeout 180 ncu --cache-control none --clock-control none \
        --section SpeedOfLight --section SchedulerStats --section WarpStateStats \
        --section LaunchStats --section Occupancy --section MemoryWorkloadAnalysis \
        --kernel-name 'regex:.*_flash_kda_fwd_recurrence.*' --launch-count 1 \
        --export "$experiment/release_${SLURM_JOB_ID}_${variant}" \
        "$base/.venv/bin/python" -u "$experiment/release_probe.py" \
        --source "$experiment/release_source" --profile "$variant" \
        > "$experiment/release_${SLURM_JOB_ID}_${variant}_ncu.log" 2>&1 || rc=$?
    printf '{"kind":"profile_exit","variant":"%s","exit_code":%d}\n' "$variant" "$rc"
done
date -u
