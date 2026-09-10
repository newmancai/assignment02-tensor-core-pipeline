# Copyright (c) 2026 by FlashInfer team.
"""Preregistered H12 work-scale test of the resident-wave cpc prediction."""

import argparse
import json
from pathlib import Path

import torch

from bench_flash_kda_h12_physical_search import _bt16_prepare_grid, _correct
from bench_recurrent_kda_prefill import (
    Case,
    _hardware_metadata,
    _make_case,
    _measure,
    _require_cupti,
    _timing_iteration_budget,
)


# Frozen before the first Stage 8 GPU run. The two work levels bracket Stage 6's
# 512 chunks. Fixed and four-way profiles separate total work from chain length.
PROFILE_CASES = (
    Case("h12_w256_fixed_4096", 12, (4096,), False, 15000),
    Case("h12_w256_balanced_4", 12, (976, 1008, 1040, 1072), True, 15001),
    Case("h12_w1024_fixed_16384", 12, (16384,), False, 15002),
    Case("h12_w1024_balanced_4", 12, (4000, 4064, 4128, 4192), True, 15003),
)
CPC_CANDIDATES = (3, 4, 5, 6, 8, 9, 12, 16, 17, 18, 20, 24)
B300_SMS = 148
RESIDENT_CTAS_PER_SM = 5


def _predicted_one_wave_cpc(total_chunks: int) -> int:
    """Smallest cpc whose rectangular H12 grid fits 148*5 resident CTAs."""

    capacity = B300_SMS * RESIDENT_CTAS_PER_SM
    return next(
        cpc
        for cpc in range(1, total_chunks + 1)
        if ((total_chunks + cpc - 1) // cpc) * 12 <= capacity
    )


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
        total_chunks = sum((length + 15) // 16 for length in case.seq_lens)
        predicted = _predicted_one_wave_cpc(total_chunks)
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
                        "rectangular_prepare_ctas": ((total_chunks + cpc - 1) // cpc) * 12,
                        "predicted_occupancy_aware_waves": (
                            ((total_chunks + cpc - 1) // cpc)
                            * 12
                            / (B300_SMS * RESIDENT_CTAS_PER_SM)
                        ),
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
                    f"waves={result['predicted_occupancy_aware_waves']:.3f} "
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
        winner = min(
            (item for item in candidates if "median_ms" in item),
            key=lambda item: item["median_ms"],
        )
        rows.append(
            {
                "name": case.name,
                "num_heads": case.num_heads,
                "seq_lens": list(case.seq_lens),
                "layout": "packed" if case.packed else "fixed",
                "total_chunks": total_chunks,
                "max_chunks": max((length + 15) // 16 for length in case.seq_lens),
                "predicted_one_wave_cpc": predicted,
                "screen_winner_cpc": winner["chunks_per_cta"],
                "route_invariant": len(routes) == 1,
                "candidates": candidates,
            }
        )
        del baseline
        torch.cuda.empty_cache()

    report = {
        "schema_version": 1,
        "analysis": "stage8-h12-workscale-resident-wave-screen",
        "preregistered_before_first_gpu_run": True,
        "hypothesis": (
            "If the Stage 7 prepare gain is governed by CTA decomposition near one "
            "occupancy-aware resident wave, the best cpc region should scale with "
            "total chunks: approximately 5 at W=256 and 17 at W=1024."
        ),
        "claim_boundary": (
            "Development screen at two new controlled work levels; winners require "
            "same-object paired qualification and are not deployment policy evidence."
        ),
        "hardware": _hardware_metadata(torch.device("cuda")),
        "physical_model": {
            "sm_count": B300_SMS,
            "resident_ctas_per_sm_inferred_from_stage7_ncu": RESIDENT_CTAS_PER_SM,
            "resident_grid_capacity_ctas": B300_SMS * RESIDENT_CTAS_PER_SM,
        },
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
