#!/bin/bash
#SBATCH --job-name=kda-stage9-fifo
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=<REMOTE_HOME>/kda-stage9-fifo-%j.out

set -euo pipefail

checkout=<REMOTE_HOME>/flashinfer-cake-sm103-20260908
typed_ir=<REMOTE_HOME>/kda-typed-ir-20260908
python_bin=<REMOTE_HOME>/FlashKDA/.venv/bin/python
benchmark="$typed_ir/stage3_remote/benchmarks/bench_flash_kda_stage9_ablation.py"
phase=${STAGE9_PHASE:-round1-final}
if [[ "$phase" == "round0" ]]; then
  manifest=${STAGE9_POLICY_MANIFEST:-$typed_ir/evidence/stage9_agent_runs_v2/round_0_manifest.json}
else
  manifest=${STAGE9_POLICY_MANIFEST:-$typed_ir/evidence/stage9_agent_runs_v2/proposal_manifest.json}
fi
amendment=${STAGE9_AMENDMENT:-$typed_ir/evidence/stage9_multiagent_ablation_amendment_v2.json}
secret=${STAGE9_HELDOUT_SECRET:-<REMOTE_HOME>/stage9-sealed/heldout_profiles.json}
commitment=${STAGE9_HELDOUT_COMMITMENT:-$typed_ir/evidence/stage9_heldout_commitment.json}
screen_jsonl=${STAGE9_SCREEN_JSONL:-$typed_ir/evidence/b300_stage9_screen.jsonl}
artifact_tag=${STAGE9_ARTIFACT_TAG:-stage9}

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$checkout/.runtime-deps/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$checkout/.runtime-deps:$checkout:$checkout/benchmarks"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

test -f "$manifest"
test -f "$amendment"
test -f "$secret"
test -f "$commitment"
test "$(nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l | tr -d ' ')" = 1

cd "$checkout/benchmarks"

if [[ "$phase" == "round0" ]]; then
"$python_bin" "$typed_ir/account_b300_run.py" \
  --output "$typed_ir/evidence/b300_${artifact_tag}_screen_round0_accounting.json" \
  --experiment "stage9-five-arm-fifo-screen-round0" \
  -- "$python_bin" "$benchmark" \
  --mode screen-round --round-index 0 \
  --amendment "$amendment" \
  --manifest "$manifest" \
  --heldout-commitment "$commitment" \
  --screen-jsonl "$screen_jsonl" \
  --summary-json "$typed_ir/evidence/b300_${artifact_tag}_screen_round0_summary.json" \
  --state-rotations 256 \
  --screen-dry-run-iters 5 --screen-repeat-iters 25
  exit 0
fi

test "$phase" = "round1-final"

"$python_bin" "$typed_ir/account_b300_run.py" \
  --output "$typed_ir/evidence/b300_${artifact_tag}_screen_round1_accounting.json" \
  --experiment "stage9-five-arm-fifo-screen-round1" \
  -- "$python_bin" "$benchmark" \
  --mode screen-round --round-index 1 \
  --amendment "$amendment" \
  --manifest "$manifest" \
  --heldout-commitment "$commitment" \
  --screen-jsonl "$screen_jsonl" \
  --summary-json "$typed_ir/evidence/b300_${artifact_tag}_screen_round1_summary.json" \
  --state-rotations 256 \
  --screen-dry-run-iters 5 --screen-repeat-iters 25

"$python_bin" "$typed_ir/account_b300_run.py" \
  --output "$typed_ir/evidence/b300_${artifact_tag}_qualification_accounting.json" \
  --experiment stage9-five-arm-fifo-qualification \
  -- "$python_bin" "$benchmark" \
  --mode qualification \
  --amendment "$amendment" \
  --manifest "$manifest" \
  --heldout-commitment "$commitment" \
  --screen-jsonl "$screen_jsonl" \
  --qualification-jsonl "$typed_ir/evidence/b300_${artifact_tag}_qualification.jsonl" \
  --winner-freeze-json "$typed_ir/evidence/${artifact_tag}_winner_freeze.json" \
  --summary-json "$typed_ir/evidence/b300_${artifact_tag}_qualification_summary.json" \
  --state-rotations 256 \
  --qualification-dry-run-iters 20 --qualification-repeat-iters 100

# This second process cannot inspect held-out profiles until the complete winner
# set and its digest have been written by the first process.
"$python_bin" "$typed_ir/account_b300_run.py" \
  --output "$typed_ir/evidence/b300_${artifact_tag}_heldout_accounting.json" \
  --experiment stage9-five-arm-fifo-sealed-heldout \
  -- "$python_bin" "$benchmark" \
  --mode heldout \
  --amendment "$amendment" \
  --heldout-secret "$secret" \
  --heldout-commitment "$commitment" \
  --winner-freeze-json "$typed_ir/evidence/${artifact_tag}_winner_freeze.json" \
  --heldout-jsonl "$typed_ir/evidence/b300_${artifact_tag}_heldout.jsonl" \
  --heldout-reveal-json "$typed_ir/evidence/${artifact_tag}_heldout_reveal.json" \
  --summary-json "$typed_ir/evidence/b300_${artifact_tag}_heldout_summary.json" \
  --state-rotations 256 \
  --qualification-dry-run-iters 20 --qualification-repeat-iters 100
