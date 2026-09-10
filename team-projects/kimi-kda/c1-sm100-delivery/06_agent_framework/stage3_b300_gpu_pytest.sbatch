#!/bin/bash
#SBATCH --job-name=kda-stage3-gpu-tests
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=<REMOTE_HOME>/kda-stage3-gpu-tests-%j.out

set -euo pipefail

checkout=<REMOTE_HOME>/flashinfer-cake-sm103-20260908
typed_ir=<REMOTE_HOME>/kda-typed-ir-20260908
python_bin=<REMOTE_HOME>/FlashKDA/.venv/bin/python

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$checkout/.runtime-deps/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$checkout/.runtime-deps:$checkout"
python_dev="$checkout/.runtime-deps/python-dev/root/usr/include"
export CPATH="$python_dev/python3.12:$python_dev:${CPATH:-}"

cd "$checkout"
"$python_bin" "$typed_ir/account_b300_run.py" \
  --output "$typed_ir/evidence/b300_stage3_gpu_pytest_accounting.json" \
  --experiment stage3-evolution-gpu-pytest \
  -- "$python_bin" -m pytest \
  tests/kda/test_flashkda_blackwell_evolution.py -q -m gpu
