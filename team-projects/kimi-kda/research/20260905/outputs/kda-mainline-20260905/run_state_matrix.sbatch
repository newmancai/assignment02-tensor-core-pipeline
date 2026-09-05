#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --job-name=kda-state-0905
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
set -euo pipefail
ulimit -c 0
experiment=/home/lcpu/YOUR_USER_ID/kda-mainline-20260905
base=/home/lcpu/YOUR_USER_ID/FlashKDA
export PYTHONPATH="$experiment/release_build/lib:$base/results/20260821_dispatch_boundary/integrated_dispatch/lib"
unset FLASH_KDA_K2_VALUE_SLICE FLASH_KDA_K2_DISPATCH
date -u
timeout 260 "$base/.venv/bin/python" -u "$experiment/state_matrix_probe.py" --source "$experiment/release_source"
date -u
