#!/usr/bin/env bash
# M4 梯子表:依次构建并运行 4.1/4.2/4.3,打印三行。
# 用法:./judge_ladder.sh [M N K](默认 4096^3)
# 每个进程套 timeout -k:流水写错的典型故障是挂死,SIGTERM 杀不死
# 卡在 CUDA 同步里的进程,必须 -k 强杀,否则僵尸进程占着 GPU。
set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
M4_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ARCH="${ARCH:-100f}"
NVCC="${NVCC:-nvcc}"
SHAPE=${*:-""}
fail=0
for item in "4.1-tiled/01_tiled" "4.2-tma/02_tma" "4.3-pipeline/03_pipeline"; do
    src="$M4_DIR/$item.cu"
    name="${item##*/}"
    bin=$(mktemp "/tmp/$name.XXXXXX")
    "$NVCC" -O3 -std=c++17 -I"$M4_DIR" \
      -gencode arch=compute_${ARCH},code=sm_${ARCH} \
      "$src" -lcublas -lcuda -o "$bin" || { echo "$name: 编译失败"; fail=1; rm -f "$bin"; continue; }
    timeout -k 5 180 "$bin" $SHAPE || fail=1
    rm -f "$bin"
done
[[ $fail == 0 ]] && echo "JUDGE: PASS" || { echo "JUDGE: FAIL"; exit 1; }
