#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --job-name=kda-stage11-v16-tcgen
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=stage11_v16_tcgen_%j.log

set -euo pipefail

probe_dir="${STAGE11_PROBE_DIR:?set STAGE11_PROBE_DIR to this directory}"
result_dir="${STAGE11_RESULT_DIR:-${probe_dir}/results}"
cuda_home="${CUDA_HOME:-/usr/local/cuda-13.0}"
arch="${STAGE11_ARCH:-103a}"
job_id="${SLURM_JOB_ID:-manual}"

mkdir -p "$result_dir"
cd "$probe_dir"

echo "campaign=stage11_v16_tcgen_probe"
echo "semantic_shape=D128_CHUNK16_V16_grid96"
echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$cuda_home/bin/nvcc" --version
nvidia-smi --query-gpu=name,uuid,driver_version,clocks.current.sm,memory.total --format=csv

# tcgen05/TMEM is a data-center Blackwell path.  The shared queue can also
# dispatch consumer SM120 GPUs, which cannot assemble this instruction family.
compute_cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d ' ')"
if [[ "$compute_cap" != "10.3" ]]; then
    echo "SKIP,reason=requires_sm103a_b300,observed_compute_cap=$compute_cap"
    exit 3
fi

# Correctness first.  Guardrails deliberately are not used for timing.
make clean
make -j "${SLURM_CPUS_PER_TASK:-8}" \
    CUDA_HOME="$cuda_home" ARCH="$arch" GUARDRAILS=1
timeout 180 ./stage11_v16_tcgen_probe --validate-only

make clean
make -j "${SLURM_CPUS_PER_TASK:-8}" CUDA_HOME="$cuda_home" ARCH="$arch"
make CUDA_HOME="$cuda_home" ARCH="$arch" sass
timeout 180 ./stage11_v16_tcgen_probe --validate-only
timeout 600 ./stage11_v16_tcgen_probe \
    --benchmark-only --warmup 30 --iters 200 --repeats 5 \
    --csv "$result_dir/stage11_v16_tcgen_probe_${job_id}.csv"

nvidia-smi --query-gpu=name,uuid,clocks.current.sm,temperature.gpu,power.draw --format=csv
echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
