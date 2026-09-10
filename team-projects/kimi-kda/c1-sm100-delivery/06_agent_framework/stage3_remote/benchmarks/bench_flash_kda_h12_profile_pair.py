# Copyright (c) 2026 by FlashInfer team.
"""Paired confirmation of cpc=9 on the Stage 6 matched-work profile matrix."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from bench_flash_kda_h12_cpc_pair import _bootstrap_speedup
from bench_flash_kda_h12_physical_search import _bt16_prepare_grid
from bench_flash_kda_h12_profile_matrix import PROFILE_CASES
from bench_recurrent_kda_prefill import (
    _hardware_metadata,
    _make_case,
    _require_cupti,
    _timing_iteration_budget,
)
from flashinfer.testing import bench_gpu_time


def _run(prepared, chunks_per_cta: int) -> None:
    with _bt16_prepare_grid(chunks_per_cta, True):
        prepared.candidate_run()


def _measure(
    prepared, chunks_per_cta: int, *, dry_run_iters: int, repeat_iters: int
) -> list[float]:
    return [
        float(value)
        for value in bench_gpu_time(
            lambda: _run(prepared, chunks_per_cta),
            enable_cupti=True,
            cold_l2_cache=True,
            use_cuda_graph=False,
            dry_run_iters=dry_run_iters,
            repeat_iters=repeat_iters,
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-cpc", type=int, default=4)
    parser.add_argument("--candidate-cpc", type=int, default=9)
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
        with _bt16_prepare_grid(args.baseline_cpc, True):
            prepared = _make_case(
                case,
                state_rotations=args.state_rotations,
                candidate_route="dispatcher",
                candidate_backend="cake",
            )
        prepared.reset_state_pools()
        _run(prepared, args.baseline_cpc)
        torch.cuda.synchronize()
        baseline_output = prepared.candidate_output.clone()
        baseline_state = prepared.candidate_state_pool[0].clone()
        prepared.reset_state_pools()
        _run(prepared, args.candidate_cpc)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            prepared.candidate_output, baseline_output, atol=1e-2, rtol=1e-2
        )
        torch.testing.assert_close(
            prepared.candidate_state_pool[0], baseline_state, atol=1e-2, rtol=1e-2
        )

        samples = {"baseline": [], "candidate": []}
        block_medians = {"baseline": [], "candidate": []}
        for name, cpc in (
            ("baseline", args.baseline_cpc),
            ("candidate", args.candidate_cpc),
            ("candidate", args.candidate_cpc),
            ("baseline", args.baseline_cpc),
        ):
            prepared.reset_state_pools()
            block = _measure(
                prepared,
                cpc,
                dry_run_iters=dry_run_iters,
                repeat_iters=repeat_iters,
            )
            samples[name].extend(block)
            block_medians[name].append(float(np.median(block)))

        speedup, lower, upper = _bootstrap_speedup(
            samples["baseline"], samples["candidate"], seed=case.seed
        )
        rows.append(
            {
                "name": case.name,
                "num_heads": case.num_heads,
                "seq_lens": list(case.seq_lens),
                "layout": "packed" if case.packed else "fixed",
                "total_chunks": sum((length + 15) // 16 for length in case.seq_lens),
                "max_chunks": max((length + 15) // 16 for length in case.seq_lens),
                "route": prepared.metadata["variant"],
                "physical_variants": prepared.metadata["physical_variants"],
                "correct": True,
                "baseline_median_ms": float(np.median(samples["baseline"])),
                "candidate_median_ms": float(np.median(samples["candidate"])),
                "baseline_block_medians_ms": block_medians["baseline"],
                "candidate_block_medians_ms": block_medians["candidate"],
                "baseline_samples_ms": samples["baseline"],
                "candidate_samples_ms": samples["candidate"],
                "speedup": speedup,
                "bootstrap_95": [lower, upper],
                "decision": "supports_cpc9" if lower > 1.0 else "inconclusive",
            }
        )
        print(
            f"{case.name:<28} {speedup:.4f}x "
            f"[{lower:.4f}, {upper:.4f}] {rows[-1]['decision']}"
        )
        del prepared, baseline_output, baseline_state
        torch.cuda.empty_cache()

    report = {
        "schema_version": 1,
        "analysis": "stage6-h12-matched-work-cpc9-paired-confirmation",
        "selection_source": "b300_stage6_h12_profile_matrix.json",
        "claim_boundary": (
            "Same-object paired confirmation on profiles already used in the "
            "Stage 6 cpc search; not an untouched held-out policy test."
        ),
        "hardware": _hardware_metadata(torch.device("cuda")),
        "all_correct": all(row["correct"] for row in rows),
        "scope_identical": True,
        "baseline_cpc": args.baseline_cpc,
        "candidate_cpc": args.candidate_cpc,
        "pair_order": "baseline/candidate/candidate/baseline",
        "protocol": {
            "timing_backend": "cupti",
            "cold_l2": True,
            "same_prepared_object_per_pair": True,
            "same_initial_state_per_timed_call": True,
            "state_rotations": args.state_rotations,
            "blocks_per_arm": 2,
            "dry_run_iters_per_block": dry_run_iters,
            "measured_iterations_per_block": repeat_iters,
            "bootstrap_draws": 10000,
        },
        "rows": rows,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
