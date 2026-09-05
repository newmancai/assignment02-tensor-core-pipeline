#!/bin/bash
set -euo pipefail
experiment=/home/lcpu/YOUR_USER_ID/kda-zero-state-20260905
export KDA_ZERO_SOURCE="$experiment/phase1_source"
export KDA_CUTLASS_SOURCE=/home/lcpu/YOUR_USER_ID/FlashKDA-c1-final/cutlass
export CUDA_HOME=/usr/local/cuda-13.0 MAX_JOBS=2
export PATH="$CUDA_HOME/bin:$PATH"
python_dev=/home/lcpu/YOUR_USER_ID/FlashKDA/.deps/python312-dev/usr/include
export CPATH="$CUDA_HOME/targets/x86_64-linux/include:$python_dev:$python_dev/python3.12:$python_dev/x86_64-linux-gnu/python3.12${CPATH:+:$CPATH}"
cd "$experiment"
date -u
sha256sum phase1_source/csrc/flash_kda.cpp phase1_source/csrc/smxx/fwd_kernel2.cuh phase1_source/csrc/smxx/fwd_launch.cu
/home/lcpu/YOUR_USER_ID/FlashKDA/.venv/bin/python build_experiment.py build_ext --build-lib phase1_build/lib --build-temp phase1_build/temp --force
date -u
