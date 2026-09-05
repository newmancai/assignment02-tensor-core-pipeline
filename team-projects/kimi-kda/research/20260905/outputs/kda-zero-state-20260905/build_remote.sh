#!/bin/bash
set -euo pipefail
experiment=/home/lcpu/YOUR_USER_ID/kda-zero-state-20260905
export KDA_ZERO_SOURCE="$experiment/source"
export KDA_CUTLASS_SOURCE=/home/lcpu/YOUR_USER_ID/FlashKDA-c1-final/cutlass
export CUDA_HOME=/usr/local/cuda-13.0 MAX_JOBS=2
export PATH="$CUDA_HOME/bin:$PATH"
python_dev=/home/lcpu/YOUR_USER_ID/FlashKDA/.deps/python312-dev/usr/include
export CPATH="$CUDA_HOME/targets/x86_64-linux/include:$python_dev:$python_dev/python3.12:$python_dev/x86_64-linux-gnu/python3.12${CPATH:+:$CPATH}"
cd "$experiment"
date -u
sha256sum source/csrc/flash_kda.cpp source/csrc/smxx/fwd_kernel2.cuh source/csrc/smxx/fwd_launch.cu
/home/lcpu/YOUR_USER_ID/FlashKDA/.venv/bin/python build_experiment.py build_ext --build-lib build/lib --build-temp build/temp --force
date -u
