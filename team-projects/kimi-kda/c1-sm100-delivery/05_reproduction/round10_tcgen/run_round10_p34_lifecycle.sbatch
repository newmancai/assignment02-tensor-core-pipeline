#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --job-name=kda-r10-p34
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=<REMOTE_HOME>/kda-r10-p34-%j.out
set -euo pipefail

root=<REMOTE_HOME>
base="$root/kda-dual-ir-agent-rounds-20260910/round8_tcgen"
round10="$root/kda-stage12-typed-ir/round10_tcgen"
work="$root/kda-stage12-typed-ir/round10-p34-job-$SLURM_JOB_ID"
result="$work/results"
cuda_home=/usr/local/cuda-13.0
mkdir -p "$work/source" "$result"
cp "$base/phase6_probe_preferred.cu" "$work/source/"
cp "$round10/p34_lifecycle_probe.cu" "$work/source/"

compute_cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d ' ')"
if [[ "$compute_cap" != "10.3" ]]; then
    echo "requires B300/SM103, found $compute_cap"
    exit 3
fi
nvidia-smi --query-gpu=name,uuid,driver_version,clocks.current.sm,memory.total --format=csv

cd "$work/source"
"$cuda_home/bin/nvcc" -O3 -std=c++17 --expt-relaxed-constexpr -lineinfo \
  -gencode arch=compute_103a,code=sm_103a \
  -Xptxas=-v,--warn-on-spills \
  -o p34_lifecycle p34_lifecycle_probe.cu

./p34_lifecycle --check-grid 12 --check-inners 1,2,4 \
  --grids 12,96 --inners 1,64 --warmup 30 --iters 200 --repeats 7 \
  --csv "$result/p34_lifecycle.csv" > "$result/p34_lifecycle.log" 2>&1

sha256sum \
  "$round10/p34_lifecycle_probe.cu" \
  "$round10/run_round10_p34_lifecycle.sbatch" \
  "$round10/README.md" \
  "$work/source/phase6_probe_preferred.cu" \
  "$work/source/p34_lifecycle_probe.cu" \
  "$work/source/p34_lifecycle" \
  "$result/p34_lifecycle.csv" \
  "$result/p34_lifecycle.log" \
  > "$result/identity.sha256"
grep -E '^(META|VALIDATE|CSV_HEADER|RESULT)' "$result/p34_lifecycle.log"
