#!/bin/bash
#SBATCH --job-name=kda-stage11-cake-peer
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=<REMOTE_HOME>/kda-stage11-cake-peer-%j.out

set -euo pipefail

official=<REMOTE_HOME>/FlashKDA-c1-official
official_commit=1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b
checkout=<REMOTE_HOME>/flashinfer-cake-sm103-20260908
typed_ir=<REMOTE_HOME>/kda-typed-ir-20260908
python_bin=<REMOTE_HOME>/FlashKDA/.venv/bin/python

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$checkout/.runtime-deps/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

test "$(git -C "$official" rev-parse HEAD)" = "$official_commit"
test -z "$(git -C "$official" status --porcelain --untracked-files=no)"
test -n "$(compgen -G "$official/flash_kda_C*.so")"

export PYTHONPATH="$official:$checkout/.runtime-deps:$checkout"
cd "$checkout/benchmarks"
"$python_bin" "$typed_ir/account_b300_run.py" \
  --output "$typed_ir/evidence/b300_stage11_cake_vs_official_h12_accounting.json" \
  --experiment stage11-cake-cross-phase-tmem-vs-official-sm80-h12 \
  -- "$python_bin" bench_recurrent_kda_prefill.py \
  --case-set h12 --backend cake --state-rotations 256 \
  --dry-run-iters 20 --repeat-iters 100 \
  --flash-kda-peer --flash-kda-source-dir "$official" \
  --json "$typed_ir/evidence/b300_stage11_cake_vs_official_h12.json"
