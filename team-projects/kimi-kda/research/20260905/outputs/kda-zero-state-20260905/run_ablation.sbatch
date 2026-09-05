#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:07:00
#SBATCH --job-name=kda-zero-ablation
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
set -euo pipefail
ulimit -c 0
experiment=/home/lcpu/YOUR_USER_ID/kda-zero-state-20260905
previous=/home/lcpu/YOUR_USER_ID/kda-mainline-20260905
base=/home/lcpu/YOUR_USER_ID/FlashKDA
export PYTHONPATH="$experiment/build/lib:$previous:$previous/release_build/lib:$base/results/20260821_dispatch_boundary/integrated_dispatch/lib"
unset FLASH_KDA_K2_VALUE_SLICE FLASH_KDA_K2_DISPATCH
date -u
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
timeout 380 "$base/.venv/bin/python" -u "$experiment/ablation_probe.py" --source "$previous/release_source"
date -u
