# Copyright (c) 2026 by FlashInfer team.
"""Run one profiled fixed-8192 H12 prepare/chain call after route warmup."""

import argparse
import json

import torch

from bench_flash_kda_h12_physical_search import _bt16_prepare_grid
from bench_flash_kda_h12_profile_matrix import PROFILE_CASES
from bench_recurrent_kda_prefill import _make_case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpc", type=int, required=True)
    args = parser.parse_args()
    case = PROFILE_CASES[0]
    with _bt16_prepare_grid(args.cpc, True):
        prepared = _make_case(
            case,
            state_rotations=8,
            candidate_route="dispatcher",
            candidate_backend="cake",
        )
    prepared.reset_state_pools()
    with _bt16_prepare_grid(args.cpc, True):
        prepared.candidate_run()
    torch.cuda.synchronize()
    print(json.dumps({"cpc": args.cpc, "metadata": prepared.metadata}, sort_keys=True))


if __name__ == "__main__":
    main()
