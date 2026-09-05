#!/bin/bash
set -euo pipefail
experiment=/home/lcpu/YOUR_USER_ID/kda-zero-state-20260905
export CUDA_HOME=/usr/local/cuda-13.0
export MAX_JOBS=2 NVCC_THREADS=8
export FLASH_KDA_EXTENSION_NAME=flash_kda_phase1_C
export FLASH_KDA_CUDA_ARCHS=103a FLASH_KDA_ENABLE_V16_PREFETCH4=1
export FLASH_KDA_ENABLE_V16_PHASE1_PREFETCH=1
export FLASH_KDA_VERSION_SUFFIX=+phase120260905
unset FLASH_KDA_DISABLE_K1 FLASH_KDA_DISABLE_K2 FLASH_KDA_BUILD_VSPLIT16 FLASH_KDA_K2_VALUE_SLICE
export PATH="$CUDA_HOME/bin:$PATH"
python_dev=/home/lcpu/YOUR_USER_ID/FlashKDA/.deps/python312-dev/usr/include
export CPATH="$CUDA_HOME/targets/x86_64-linux/include:$python_dev:$python_dev/python3.12:$python_dev/x86_64-linux-gnu/python3.12${CPATH:+:$CPATH}"
cd "$experiment/clean_source"
date -u
sha256sum csrc/flash_kda.cpp csrc/smxx/fwd_kernel2.cuh csrc/smxx/fwd_launch.cu setup.py
/home/lcpu/YOUR_USER_ID/FlashKDA/.venv/bin/python setup.py build_ext --build-lib "$experiment/clean_build/lib" --build-temp "$experiment/clean_build/temp" --force
date -u
