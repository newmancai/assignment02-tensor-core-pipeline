#!/bin/bash
#SBATCH --job-name=kda-h3-matched-hmma
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=<REMOTE_HOME>/kda-h3-matched-hmma-%j.out

# DESIGN ONLY.  The preflight intentionally refuses to run until the H3 source,
# its separately named extension, and bench_h3_matched_hmma.py are present.
set -euo pipefail

official_dir=<REMOTE_HOME>/FlashKDA-c1-official
official_commit=1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b
h3_dir=<REMOTE_HOME>/FlashKDA-c1-h3-pw
optimized_dir=<REMOTE_HOME>/kda-zero-state-20260905/clean_source
typed_ir_dir=<REMOTE_HOME>/kda-typed-ir-20260908
python_bin=<REMOTE_HOME>/FlashKDA/.venv/bin/python
runner_path="$typed_ir_dir/experiments/sm100_open_round1/bench_h3_matched_hmma.py"
schema_path="$typed_ir_dir/experiments/sm100_open_round1/schema/h3_b300_results.schema.json"
result_path="$typed_ir_dir/experiments/sm100_open_round1/evidence/h3_b300_matched_hmma.json"
accounting_path="$typed_ir_dir/experiments/sm100_open_round1/evidence/h3_b300_matched_hmma_accounting.json"

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$optimized_dir/.runtime-deps/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

test "$(git -C "$official_dir" rev-parse HEAD)" = "$official_commit"
test -z "$(git -C "$official_dir" status --porcelain --untracked-files=no)"
test -n "$(compgen -G "$official_dir/flash_kda_C*.so")"
test -d "$h3_dir"
test -n "$(compgen -G "$h3_dir/flash_kda_h3_C*.so")"
test -d "$optimized_dir"
test -n "$(compgen -G "$optimized_dir/flash_kda_phase1_C*.so")"
test -f "$runner_path"
test -f "$schema_path"

export PYTHONPATH="$official_dir:$h3_dir:$optimized_dir:$typed_ir_dir"

"$python_bin" "$typed_ir_dir/account_b300_run.py" \
  --output "$accounting_path" \
  --experiment sm100-open-h3-matched-hmma-k1-k2-full \
  -- "$python_bin" "$runner_path" \
  --official-source-dir "$official_dir" \
  --official-commit "$official_commit" \
  --h3-source-dir "$h3_dir" \
  --optimized-source-dir "$optimized_dir" \
  --case-set h12,h96-task \
  --variants official_hmma,h3_matched_hmma,optimized_hmma \
  --scopes k1,k2,raw_full,public_full \
  --state-rotations 512 \
  --dry-run-iters 20 \
  --repeat-iters 100 \
  --blocks 8 \
  --cold-l2 \
  --balanced-pair-order \
  --schema "$schema_path" \
  --json "$result_path"
