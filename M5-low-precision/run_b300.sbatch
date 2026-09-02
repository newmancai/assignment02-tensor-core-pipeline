#!/usr/bin/env bash
#SBATCH --job-name=m5-low-precision
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=M5-low-precision/evidence/slurm-%j.out

set -euo pipefail

# Submit from the repository root:
#   sbatch M5-low-precision/run_b300.sbatch
cd "${SLURM_SUBMIT_DIR:?submit this script from the repository root}"
test -d M5-low-precision

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mkdir -p build/m5 M5-low-precision/evidence
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"$CUDA_HOME/bin/nvcc" --version

NVCC_FLAGS=(-O3 -std=c++17 -lineinfo
  -gencode arch=compute_100f,code=sm_100f)

"$CUDA_HOME/bin/nvcc" "${NVCC_FLAGS[@]}" \
  M5-low-precision/5.3a-e2m1/03a_encode_check.cu \
  -o build/m5/03a_encode_check
"$CUDA_HOME/bin/nvcc" "${NVCC_FLAGS[@]}" \
  M5-low-precision/5.3b-nvfp4-quant/03b_nvfp4_quant.cu \
  -o build/m5/03b_nvfp4_quant
"$CUDA_HOME/bin/nvcc" "${NVCC_FLAGS[@]}" \
  M5-low-precision/5.3b-nvfp4-quant/test_fp4_gemm.cu \
  -lcublasLt -o build/m5/test_fp4_gemm
"$CUDA_HOME/bin/nvcc" "${NVCC_FLAGS[@]}" \
  M5-low-precision/5.3c-ceiling-probe/03c_ceiling_probe.cu \
  -o build/m5/03c_ceiling_probe
"$CUDA_HOME/bin/nvcc" "${NVCC_FLAGS[@]}" \
  M5-low-precision/5.4-fused-rmsnorm-nvfp4/04_fused_rms_nvfp4.cu \
  -o build/m5/04_fused_rms_nvfp4

./build/m5/03a_encode_check
./build/m5/03b_nvfp4_quant
./build/m5/test_fp4_gemm

# Three independent timing passes make the median and run-to-run spread visible.
for _run in 1 2 3; do
  ./build/m5/03c_ceiling_probe
done
for _run in 1 2 3; do
  ./build/m5/04_fused_rms_nvfp4
done
