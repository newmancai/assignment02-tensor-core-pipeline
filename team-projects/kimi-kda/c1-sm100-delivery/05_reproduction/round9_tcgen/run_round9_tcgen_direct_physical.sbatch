#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --job-name=kda-r9-tcgen-layout
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=<REMOTE_HOME>/kda-r9-tcgen-layout-%j.out
set -euo pipefail

root=<REMOTE_HOME>
source_dir="$root/kda-dual-ir-agent-rounds-20260910/round8_tcgen"
round9="$root/kda-stage12-typed-ir/round9_tcgen"
work="$root/kda-stage12-typed-ir/round9-tcgen-job-$SLURM_JOB_ID"
result="$work/results"
cuda_home=/usr/local/cuda-13.0
python="$root/FlashKDA/.venv/bin/python"
mkdir -p "$work/source" "$result"
cp "$source_dir/phase6_probe_preferred.cu" "$work/source/"
cp "$source_dir/round8_v16_preferred.cu" "$work/source/"
patch --batch --forward -p1 -d "$work/source" -i "$round9/direct_physical_b.patch"

compute_cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d ' ')"
if [[ "$compute_cap" != "10.3" ]]; then
    echo "requires B300/SM103, found $compute_cap"
    exit 3
fi
nvidia-smi --query-gpu=name,uuid,driver_version,clocks.current.sm,memory.total --format=csv

flags=(
  -O3 -std=c++17 --expt-relaxed-constexpr -lineinfo
  -gencode arch=compute_103a,code=sm_103a
  -Xptxas=-v,--warn-on-spills
)
cd "$work/source"
"$cuda_home/bin/nvcc" "${flags[@]}" -o logical_b round8_v16_preferred.cu
"$cuda_home/bin/nvcc" "${flags[@]}" -DKDA_B_GLOBAL_PHYSICAL=1 \
  -o physical_b round8_v16_preferred.cu

./logical_b --warmup 30 --iters 200 --repeats 7 \
  --csv "$result/logical_b.csv" > "$result/logical_b.log" 2>&1
./physical_b --warmup 30 --iters 200 --repeats 7 \
  --csv "$result/physical_b.csv" > "$result/physical_b.log" 2>&1

"$python" "$round9/analyze_direct_physical.py" \
  --logical "$result/logical_b.csv" \
  --physical "$result/physical_b.csv" \
  --logical-log "$result/logical_b.log" \
  --physical-log "$result/physical_b.log" \
  --output "$result/summary.json"

sha256sum "$round9"/* "$work/source"/*.cu "$work/source"/logical_b \
  "$work/source"/physical_b "$result"/* > "$result/identity.sha256"
