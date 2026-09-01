#!/usr/bin/env bash
# 问题 4.3 的 stages 扫描:S ∈ {2,3,4,6},两个形状(4096^3 与
# M=256 N=4096 K=16384)。用法:./sweep_stages.sh
# 测性能前后建议 nvidia-smi --query-compute-apps=pid,name --format=csv
# 查一下有没有别的进程占卡,有残留数字整体失真。
set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
M4_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ARCH="${ARCH:-100f}"
NVCC="${NVCC:-nvcc}"
for shape in "4096 4096 4096" "256 4096 16384"; do
    echo "== 形状 $shape =="
    for s in 2 3 4 6; do
        bin=$(mktemp /tmp/pipeline.XXXXXX)
        "$NVCC" -O3 -std=c++17 -DSTAGES=$s -I"$M4_DIR" \
          -gencode arch=compute_${ARCH},code=sm_${ARCH} \
          "$SCRIPT_DIR/03_pipeline.cu" -lcublas -o "$bin" || exit 1
        timeout -k 5 180 "$bin" $shape || echo "S=$s: 失败或挂死(超时被杀)"
        rm -f "$bin"
    done
done
