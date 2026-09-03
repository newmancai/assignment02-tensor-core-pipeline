#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --job-name=kda-c1-tcgen05
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/lcpu/<USER_ID>/FlashKDA-c1-results/03_tcgen05_probe_%j.log

set -euo pipefail

# Override when the campaign has been copied elsewhere:
#   sbatch --export=ALL,TCGEN05_PROBE_DIR=/path/to/tcgen05_probe ...
probe_dir="${TCGEN05_PROBE_DIR:-/home/lcpu/<USER_ID>/FlashKDA/profile/c1-final/tcgen05_probe}"
result_dir="${TCGEN05_RESULT_DIR:-/home/lcpu/<USER_ID>/FlashKDA-c1-results}"
arch="${TCGEN05_ARCH:-103a}"
cuda_home="${CUDA_HOME:-/usr/local/cuda-13.0}"
job_id="${SLURM_JOB_ID:-manual}"

mkdir -p "$result_dir"
cd "$probe_dir"

echo "campaign=03_tcgen05_probe"
echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "probe_dir=$probe_dir"
echo "arch=$arch"
"$cuda_home/bin/nvcc" --version
nvidia-smi --query-gpu=name,uuid,driver_version,clocks.current.sm,clocks.max.sm,memory.total --format=csv

# 1. Guardrailed, fail-fast correctness.  rounds 1/2/4 exercise both mbarrier
# parities.  Rebuild release-only after it passes; guardrails distort timing.
make clean
make -j "${SLURM_CPUS_PER_TASK:-8}" CUDA_HOME="$cuda_home" ARCH="$arch" GUARDRAILS=1
timeout 180 ./phase6_probe \
    --validate-only --values 16,32,64,128 --check-grid 12 \
    --check-inners 1,2,4

# 2. Full-kernel CUDA-event campaign.  inner=1 exposes launch/staging/TMEM
# allocation cost; inner=64 is a persistent-CTA-style amortized probe.
make clean
make -j "${SLURM_CPUS_PER_TASK:-8}" CUDA_HOME="$cuda_home" ARCH="$arch"
make CUDA_HOME="$cuda_home" ARCH="$arch" sass
timeout 600 ./phase6_probe \
    --benchmark-only --values 16,32,64,128 --grids 12,148 \
    --inners 1,64 --warmup 30 --iters 200 --repeats 5 \
    --csv "$result_dir/03_tcgen05_probe_${job_id}.csv"

echo "sass=$probe_dir/phase6_probe.sass"
nvidia-smi --query-gpu=name,uuid,clocks.current.sm,temperature.gpu,power.draw --format=csv
echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
