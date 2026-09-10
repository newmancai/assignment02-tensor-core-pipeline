#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --job-name=c1-targeted-ncu
#SBATCH --output=/home/lcpu/<USER_ID>/FlashKDA-c1-results/05_targeted_ncu_%j.log

set -euo pipefail

main_repo="/home/lcpu/<USER_ID>/FlashKDA"
official_repo="/home/lcpu/<USER_ID>/FlashKDA-c1-official"
campaign="$main_repo/profile/c1-final"
integrated="$main_repo/results/20260821_dispatch_boundary/integrated_dispatch/lib"
python="$main_repo/.venv/bin/python"
out="/home/lcpu/<USER_ID>/FlashKDA-c1-results"
mkdir -p "$out"

job_id="${SLURM_JOB_ID:-manual}"
official_rep="$out/05_official_v128_h12_t8192_${job_id}.ncu-rep"
v16_rep="$out/05_valueslice_v16_h12_t8192_${job_id}.ncu-rep"
long_csv="$out/05_targeted_ncu_metrics_${job_id}.csv"
summary_csv="$out/05_targeted_ncu_summary_${job_id}.csv"
query_log="$out/05_ncu_supported_tensor_metrics_${job_id}.txt"

echo "START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
nvidia-smi --query-gpu=name,uuid,compute_cap --format=csv,noheader
ncu --version

# Ask this exact NCU/GPU pair which tensor-pipe metrics it exposes.  The core
# four sections below are stable; supported raw tensor metrics are added without
# making the whole run fail on an architecture-specific metric rename.
ncu --query-metrics-mode all > "$query_log"
tensor_metrics=()
tensor_candidates=(
  sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active
  sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed
  sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active
  sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed
  smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active
  smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed
)
for metric in "${tensor_candidates[@]}"; do
  if grep -Fq "$metric " "$query_log"; then
    tensor_metrics+=("$metric")
  fi
done
if ((${#tensor_metrics[@]} == 0)); then
  echo "WARNING: no tensor-active candidate was reported; stable section metrics will still run"
  tensor_arg=()
else
  tensor_csv="$(IFS=,; echo "${tensor_metrics[*]}")"
  tensor_arg=(--metrics "$tensor_csv")
  echo "TENSOR_METRICS=$tensor_csv"
fi

common_ncu=(
  --target-processes all
  --profile-from-start off
  --kernel-name-base demangled
  --kernel-name "regex:.*_flash_kda_fwd_recurrence.*"
  --launch-count 1
  --replay-mode kernel
  --section SpeedOfLight
  --section LaunchStats
  --section Occupancy
  --section SchedulerStats
  --force-overwrite
)

echo "RUN official_v128"
env -u FLASH_KDA_K2_VALUE_SLICE -u FLASH_KDA_K2_DISPATCH \
  PYTHONPATH="$official_repo" \
  ncu "${common_ncu[@]}" "${tensor_arg[@]}" \
  --export "$official_rep" \
  "$python" "$campaign/profile_k2_targeted_ncu.py" \
  --label official_v128 --tokens 8192 --heads 12 --warmup 5

echo "RUN valueslice_v16"
env -u FLASH_KDA_K2_DISPATCH \
  PYTHONPATH="$integrated:$main_repo" FLASH_KDA_K2_VALUE_SLICE=16 \
  ncu "${common_ncu[@]}" "${tensor_arg[@]}" \
  --export "$v16_rep" \
  "$python" "$campaign/profile_k2_targeted_ncu.py" \
  --label valueslice_v16 --tokens 8192 --heads 12 --warmup 5

"$python" "$campaign/export_targeted_ncu.py" \
  --long-output "$long_csv" \
  --summary-output "$summary_csv" \
  "official_v128=$official_rep" \
  "valueslice_v16=$v16_rep"

echo "ARTIFACT official_report=$official_rep"
echo "ARTIFACT valueslice_report=$v16_rep"
echo "ARTIFACT metrics_csv=$long_csv"
echo "ARTIFACT summary_csv=$summary_csv"
echo "ARTIFACT tensor_metric_query=$query_log"
echo "END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
