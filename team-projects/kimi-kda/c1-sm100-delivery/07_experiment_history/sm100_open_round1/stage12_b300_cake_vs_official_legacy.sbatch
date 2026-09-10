#!/bin/bash
#SBATCH --job-name=kda-cake-legacy
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=<REMOTE_HOME>/kda-cake-legacy-%j.out

set -euo pipefail

official=<REMOTE_HOME>/FlashKDA-c1-official
checkout=<REMOTE_HOME>/flashinfer-cake-sm103-20260908
typed_ir=<REMOTE_HOME>/kda-typed-ir-20260908
python_bin=<REMOTE_HOME>/FlashKDA/.venv/bin/python

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$checkout/.runtime-deps/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$checkout/.runtime-deps/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$official:$checkout/.runtime-deps:$checkout"

test "$(git -C "$official" rev-parse HEAD)" = 1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b
test -z "$(git -C "$official" status --porcelain --untracked-files=no)"

cd "$checkout/benchmarks"
"$python_bin" "$typed_ir/account_b300_run.py" \
  --output "$typed_ir/experiments/sm100_open_round1/evidence/b300_stage12_cake_vs_official_legacy_accounting.json" \
  --experiment stage12-cake-vs-official-sm80-h96-h64 \
  -- "$python_bin" bench_recurrent_kda_prefill.py \
  --case-set legacy --backend cake --state-rotations 256 \
  --dry-run-iters 20 --repeat-iters 100 \
  --flash-kda-peer --flash-kda-source-dir "$official" \
  --json "$typed_ir/experiments/sm100_open_round1/evidence/b300_stage12_cake_vs_official_legacy.json"
