#!/usr/bin/env bash
# 问题 3.2 的多 seed 判测。用法:./judge_tile.sh
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ARCH="${ARCH:-100f}"
BIN=$(mktemp /tmp/tcgen05_tile.XXXXXX)
trap 'rm -f "$BIN"' EXIT
nvcc -O3 -std=c++17 -I"$SCRIPT_DIR/.." \
  -gencode arch=compute_${ARCH},code=sm_${ARCH} \
  "$SCRIPT_DIR/02_single_tile.cu" -o "$BIN"
ok=1
for s in 1 7 42 1234 99999; do
    out=$($BIN $s) || ok=0
    echo "$out"
    [[ "$out" == PASS* ]] || ok=0
done
[[ $ok == 1 ]] && echo "JUDGE: PASS" || { echo "JUDGE: FAIL"; exit 1; }
