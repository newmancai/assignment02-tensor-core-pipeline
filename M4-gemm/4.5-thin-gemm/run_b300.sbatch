#!/usr/bin/env bash
#SBATCH --job-name=m45-thin-gemm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=M4-gemm/4.5-thin-gemm/evidence/slurm-%j.out

set -euo pipefail

# Slurm copies the script to its spool directory, so submit from repository root.
cd "${SLURM_SUBMIT_DIR:?submit this script from the repository root}"
test -f M4-gemm/4.5-thin-gemm/05_thin_gemm.cu
cd M4-gemm/4.5-thin-gemm
mkdir -p build evidence

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "host=$(hostname)"
echo "started=$(date --iso-8601=seconds)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L
"$CUDA_HOME/bin/nvcc" --version

"$CUDA_HOME/bin/nvcc" -std=c++17 -O3 \
  -gencode arch=compute_100f,code=sm_100f \
  -I"$CUDA_HOME/include" -L"$CUDA_HOME/lib64" \
  05_thin_gemm.cu -lcublas -o build/05_thin_gemm

# Three independent process runs make allocator/autotuning effects visible and
# provide a robust median for the report.  The first launch of every shape is
# already discarded by the benchmark before its timed warm-up and loop.
for run in 1 2 3; do
  echo "===== independent_run=${run} ====="
  ./build/05_thin_gemm 2250 8000 | tee "evidence/run-${run}.txt"
done

echo "finished=$(date --iso-8601=seconds)"
