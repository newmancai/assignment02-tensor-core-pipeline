#!/usr/bin/env bash
# 问题 3.3 的判测(带超时:发布态程序会挂死,这是症状的一部分)。
# 修好后 rounds=1 与 rounds=4 多 seed 都应 PASS。
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ARCH="${ARCH:-100f}"
BIN=$(mktemp /tmp/tcgen05_mbar.XXXXXX)
trap 'rm -f "$BIN"' EXIT
nvcc -O3 -std=c++17 -I"$SCRIPT_DIR/.." \
  -gencode arch=compute_${ARCH},code=sm_${ARCH} \
  "$SCRIPT_DIR/03_bug_mbarrier.cu" -o "$BIN"
ok=1
for r in 1 2 4; do
    for s in 42 7; do
        out=$(timeout 30 $BIN $s $r) || { echo "rounds=$r seed=$s: 超时或出错"; ok=0; continue; }
        echo "rounds=$r $out"
        [[ "$out" == PASS* ]] || ok=0
    done
done
[[ $ok == 1 ]] && echo "JUDGE: PASS" || { echo "JUDGE: FAIL"; exit 1; }
