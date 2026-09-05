#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --job-name=kda-boundary-0905
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
set -euo pipefail
ulimit -c 0
review_dir=/home/lcpu/YOUR_USER_ID/kda-review-20260905
base=/home/lcpu/YOUR_USER_ID/FlashKDA
implementation=/home/lcpu/YOUR_USER_ID/FlashKDA-c1-final
export PYTHONPATH="$implementation:$base/results/20260821_dispatch_boundary/integrated_dispatch/lib"
unset FLASH_KDA_K2_VALUE_SLICE FLASH_KDA_K2_DISPATCH
date -u
nvidia-smi --query-gpu=name,driver_version,clocks.current.sm,memory.total --format=csv
sha256sum "$review_dir/review_probe.py"
timeout 240 "$base/.venv/bin/python" -u "$review_dir/review_probe.py"
date -u
