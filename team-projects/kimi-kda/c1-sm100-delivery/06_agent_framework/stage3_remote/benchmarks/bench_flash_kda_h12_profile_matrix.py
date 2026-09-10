# Copyright (c) 2026 by FlashInfer team.
"""Preregistered matched-work H12 study of prepare/chain critical-path mixture."""

import argparse
import json
from pathlib import Path

import torch

from bench_flash_kda_h12_physical_search import (
    _bt16_prepare_grid,
    _correct,
)
from bench_recurrent_kda_prefill import (
    Case,
    _hardware_metadata,
    _make_case,
    _measure,
    _require_cupti,
    _timing_iteration_budget,
)


# Frozen before the first Stage 6 GPU run. Every profile has exactly 512
# 16-token chunks (8192 tokens), but the recurrent critical path varies.
PROFILE_CASES = (
    Case("h12_profile_fixed_8192", 12, (8192,), False, 14000),
    Case("h12_profile_balanced_2", 12, (4000, 4192), True, 14001),
    Case("h12_profile_balanced_4", 12, (1952, 2016, 2080, 2144), True, 14002),
    Case("h12_profile_skew", 12, (64, 64, 64, 8000), True, 14003),
)
CPC_CANDIDATES = (3, 4, 6, 9, 11, 12)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run-iters", type=int, default=20)
    parser.add_argument("--repeat-iters", type=int, default=100)
    parser.add_argument("--state-rotations", type=int, default=256)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    _require_cupti()
    dry_run_iters, repeat_iters = _timing_iteration_budget(
        state_rotation_capacity=args.state_rotations,
        requested_dry_run_iters=args.dry_run_iters,
        requested_repeat_iters=args.repeat_iters,
    )

    rows = []
    for case in PROFILE_CASES:
        with _bt16_prepare_grid(4, True):
            baseline = _make_case(
                case,
                state_rotations=args.state_rotations,
                candidate_route="dispatcher",
                candidate_backend="cake",
            )
        candidates = []
        for cpc in CPC_CANDIDATES:
            candidate = None
            try:
                with _bt16_prepare_grid(cpc, True):
                    candidate = _make_case(
                        case,
                        state_rotations=args.state_rotations,
                        candidate_route="dispatcher",
                        candidate_backend="cake",
                    )
                    correctness = _correct(candidate, baseline)
                    result = {
                        "chunks_per_cta": cpc,
                        "metadata": candidate.metadata,
                        "correctness": correctness,
                    }
                    if correctness["passed"]:
                        candidate.reset_state_pools()
                        median_ms, samples = _measure(
                            candidate.candidate_run,
                            dry_run_iters=dry_run_iters,
                            repeat_iters=repeat_iters,
                        )
                        result.update({"median_ms": median_ms, "samples_ms": samples})
            except Exception as error:
                result = {
                    "chunks_per_cta": cpc,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            candidates.append(result)
            if "median_ms" in result:
                print(
                    f"{case.name:<28} cpc={cpc:<2} "
                    f"{result['median_ms'] * 1000:9.3f} us"
                )
            else:
                print(f"{case.name:<28} cpc={cpc:<2} rejected")
            if candidate is not None:
                del candidate
        routes = {
            item.get("metadata", {}).get("variant")
            for item in candidates
            if "metadata" in item
        }
        rows.append(
            {
                "name": case.name,
                "num_heads": case.num_heads,
                "seq_lens": list(case.seq_lens),
                "layout": "packed" if case.packed else "fixed",
                "total_chunks": sum((length + 15) // 16 for length in case.seq_lens),
                "max_chunks": max((length + 15) // 16 for length in case.seq_lens),
                "route_invariant": len(routes) == 1,
                "candidates": candidates,
            }
        )
        del baseline
        torch.cuda.empty_cache()

    report = {
        "schema_version": 1,
        "analysis": "stage6-h12-matched-work-critical-path-profile-search",
        "preregistered_before_first_gpu_run": True,
        "hypothesis": (
            "At fixed total prepare work, the end-to-end benefit of larger cpc "
            "depends on the recurrent-chain critical path and not total chunks alone."
        ),
        "hardware": _hardware_metadata(torch.device("cuda")),
        "protocol": {
            "timing_backend": "cupti",
            "cold_l2": True,
            "public_api_scope": True,
            "same_initial_state_per_timed_call": True,
            "state_rotations": args.state_rotations,
            "dry_run_iters": dry_run_iters,
            "repeat_iters": repeat_iters,
        },
        "cpc_candidates": list(CPC_CANDIDATES),
        "rows": rows,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
