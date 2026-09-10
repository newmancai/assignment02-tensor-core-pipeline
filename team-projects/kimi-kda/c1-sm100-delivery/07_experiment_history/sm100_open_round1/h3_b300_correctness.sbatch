#!/bin/bash
#SBATCH --job-name=kda-h3-correctness
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=<REMOTE_HOME>/kda-h3-correctness-%j.out

set -euo pipefail

official_dir=<REMOTE_HOME>/FlashKDA-c1-official
h3_dir=<REMOTE_HOME>/FlashKDA-c1-h3-pw
typed_ir_dir=<REMOTE_HOME>/kda-typed-ir-20260908
python_bin=<REMOTE_HOME>/FlashKDA/.venv/bin/python
runner_path="$typed_ir_dir/experiments/sm100_open_round1/check_h3_gpu_correctness.py"
result_path="$typed_ir_dir/experiments/sm100_open_round1/evidence/h3_gpu_correctness.json"

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:<REMOTE_HOME>/FlashKDA/.venv/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CPLUS_INCLUDE_PATH=<REMOTE_HOME>/FlashKDA/.deps/python312-dev/usr/include:<REMOTE_HOME>/FlashKDA/.deps/python312-dev/usr/include/python3.12
export PYTHONPATH="$official_dir:$h3_dir:$typed_ir_dir"

test -n "$(compgen -G "$h3_dir/flash_kda_h3_C*.so")"
test -f "$runner_path"

"$python_bin" "$runner_path" \
  --official-source-dir "$official_dir" \
  --h3-source-dir "$h3_dir" \
  --heads 1 \
  --seeds 2 \
  --json "$result_path"
