#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --job-name=kda-review-0905
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
set -euo pipefail
review_dir=/home/lcpu/YOUR_USER_ID/kda-review-20260905
probe=/home/lcpu/YOUR_USER_ID/FlashKDA/profile/c1-final/tcgen05_probe/phase6_probe
base=/home/lcpu/YOUR_USER_ID/FlashKDA
implementation=/home/lcpu/YOUR_USER_ID/FlashKDA-c1-final
export PYTHONPATH="$implementation:$base/results/20260821_dispatch_boundary/integrated_dispatch/lib"
unset FLASH_KDA_K2_VALUE_SLICE FLASH_KDA_K2_DISPATCH
date -u
nvidia-smi --query-gpu=name,driver_version,clocks.current.sm,memory.total --format=csv
sha256sum "$probe" "$implementation/flash_kda/__init__.py" "$implementation/flash_kda/dispatch.py"
timeout 120 "$probe" --validate-only --values 128 --check-grid 12 --check-inners 1,2,4,128
for round in 1 2; do
    timeout 120 "$probe" --benchmark-only --values 128 --grids 12,148 --inners 1,4,16,64,96,128,256,512 --warmup 20 --iters 100 --repeats 5 --csv "$review_dir/tcgen_sweep_${SLURM_JOB_ID}_${round}.csv"
done
timeout 180 "$base/.venv/bin/python" -u "$review_dir/review_probe.py"
date -u
