# Copyright (c) 2026 by FlashInfer team.
"""Paired public-API CAKE/evolution qualification for the six legacy shapes."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from bench_recurrent_kda_prefill import (
    LEGACY_CASES,
    _hardware_metadata,
    _make_case,
    _require_cupti,
    _timing_iteration_budget,
)
from flashinfer.testing import bench_gpu_time


def _measure(run, *, dry_run_iters: int, repeat_iters: int) -> list[float]:
    return [
        float(value)
        for value in bench_gpu_time(
            run,
            enable_cupti=True,
            cold_l2_cache=True,
            use_cuda_graph=False,
            dry_run_iters=dry_run_iters,
            repeat_iters=repeat_iters,
        )
    ]


def _bootstrap_speedup(
    cake_samples: list[float],
    evolution_samples: list[float],
    *,
    seed: int,
    draws: int = 10_000,
) -> tuple[float, float, float]:
    cake = np.asarray(cake_samples)
    evolution = np.asarray(evolution_samples)
    rng = np.random.default_rng(seed)
    cake_medians = np.median(
        cake[rng.integers(0, cake.size, size=(draws, cake.size))], axis=1
    )
    evolution_medians = np.median(
        evolution[
            rng.integers(0, evolution.size, size=(draws, evolution.size))
        ],
        axis=1,
    )
    speedups = cake_medians / evolution_medians
    return (
        float(np.median(cake) / np.median(evolution)),
        float(np.quantile(speedups, 0.025)),
        float(np.quantile(speedups, 0.975)),
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
    for case in LEGACY_CASES:
        cake = _make_case(
            case,
            state_rotations=args.state_rotations,
            candidate_route="dispatcher",
            candidate_backend="cake",
        )
        evolution = _make_case(
            case,
            state_rotations=args.state_rotations,
            candidate_route="dispatcher",
            candidate_backend="evolution",
        )
        cake.reset_state_pools()
        evolution.reset_state_pools()
        cake.candidate_run()
        evolution.candidate_run()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            evolution.candidate_output, cake.candidate_output, atol=1e-2, rtol=1e-2
        )
        torch.testing.assert_close(
            evolution.candidate_state_pool[0],
            cake.candidate_state_pool[0],
            atol=1e-2,
            rtol=1e-2,
        )

        samples = {"cake": [], "evolution": []}
        block_medians = {"cake": [], "evolution": []}
        for name, prepared in (
            ("cake", cake),
            ("evolution", evolution),
            ("evolution", evolution),
            ("cake", cake),
        ):
            prepared.reset_state_pools()
            block = _measure(
                prepared.candidate_run,
                dry_run_iters=dry_run_iters,
                repeat_iters=repeat_iters,
            )
            samples[name].extend(block)
            block_medians[name].append(float(np.median(block)))

        speedup, lower, upper = _bootstrap_speedup(
            samples["cake"], samples["evolution"], seed=case.seed
        )
        resolved_evolution = evolution.metadata["resolved_backend"] == "evolution"
        decision = "activate" if resolved_evolution and lower > 1.0 else "retain_cake"
        row = {
            "name": case.name,
            "num_heads": case.num_heads,
            "seq_lens": list(case.seq_lens),
            "layout": "packed" if case.packed else "fixed",
            "correct": True,
            "cake": {
                **cake.metadata,
                "median_ms": float(np.median(samples["cake"])),
                "samples_ms": samples["cake"],
                "block_medians_ms": block_medians["cake"],
            },
            "evolution": {
                **evolution.metadata,
                "median_ms": float(np.median(samples["evolution"])),
                "samples_ms": samples["evolution"],
                "block_medians_ms": block_medians["evolution"],
            },
            "speedup": speedup,
            "bootstrap_95": [lower, upper],
            "decision": decision,
            "pair_order": "cake/evolution/evolution/cake",
            "timing_scope": "public_recurrent_kda_with_inplace_state_update",
            "timing_backend": "cupti",
            "cold_l2": True,
            "same_initial_state_per_timed_call": True,
        }
        rows.append(row)
        print(
            f"{case.name:<18} {speedup:.4f}x "
            f"[{lower:.4f}, {upper:.4f}] {decision}"
        )
        del cake, evolution
        torch.cuda.empty_cache()

    report = {
        "schema_version": 1,
        "hardware": _hardware_metadata(torch.device("cuda")),
        "all_correct": all(row["correct"] for row in rows),
        "scope_identical": True,
        "rows": rows,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
