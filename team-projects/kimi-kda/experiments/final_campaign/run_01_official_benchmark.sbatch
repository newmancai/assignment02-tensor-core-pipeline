#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --job-name=kda-c1-official
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --output=/home/lcpu/<USER_ID>/FlashKDA-c1-results/01_official_benchmark_%j.log

set -euo pipefail

official_repo=/home/lcpu/<USER_ID>/FlashKDA-c1-official
python_bin=/home/lcpu/<USER_ID>/FlashKDA/.venv/bin/python

cd "$official_repo"
export PYTHONPATH="$official_repo"
export PYTHONHASHSEED=0
export FLA_FLASH_KDA=0

echo "campaign=01_official_benchmark"
echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "flashkda_commit=$(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,uuid,driver_version,clocks.current.sm,clocks.max.sm,memory.total --format=csv
"$python_bin" -c 'import sys, torch, fla, flash_kda, flash_kda_C; print(f"python={sys.version.split()[0]}"); print(f"torch={torch.__version__}"); print(f"fla={fla.__file__}"); print(f"flash_kda={flash_kda.__file__}"); print(f"extension={flash_kda_C.__file__}")'

"$python_bin" benchmarks/bench_fwd.py \
    --H 96 --D 128 --mode all --warmup 30 --iters 200 --repeats 5
"$python_bin" benchmarks/bench_fwd.py \
    --H 64 --D 128 --mode all --warmup 30 --iters 200 --repeats 5

echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
