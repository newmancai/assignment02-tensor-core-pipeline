#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --job-name=kda-mainline-0905
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
set -euo pipefail
ulimit -c 0
experiment=/home/lcpu/YOUR_USER_ID/kda-mainline-20260905
base=/home/lcpu/YOUR_USER_ID/FlashKDA
export PYTHONPATH="$experiment/build/lib:$base/results/20260821_dispatch_boundary/integrated_dispatch/lib"
unset FLASH_KDA_K2_VALUE_SLICE FLASH_KDA_K2_DISPATCH
date -u
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
timeout 180 "$base/.venv/bin/python" -u "$experiment/test_entry_hardening.py" --extension flash_kda_mainline_C
timeout 360 "$base/.venv/bin/python" -u "$experiment/mainline_probe.py"
date -u
