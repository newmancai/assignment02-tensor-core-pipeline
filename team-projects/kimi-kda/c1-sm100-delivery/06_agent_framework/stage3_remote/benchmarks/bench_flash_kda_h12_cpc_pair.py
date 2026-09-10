# Copyright (c) 2026 by FlashInfer team.
"""Paired public H12 qualification of the total-chunk-aware BT16 prepare grid."""

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from bench_recurrent_kda_prefill import (
    H12_CASES,
    _hardware_metadata,
    _make_case,
    _require_cupti,
    _timing_iteration_budget,
)
from flashinfer import kda_prefill
from flashinfer.testing import bench_gpu_time


@contextmanager
def _h12_long_cpc(chunks_per_cta: int):
    original = kda_prefill._bt16_chunks_per_prepare_cta

    def policy(*, num_heads: int, total_chunks: int) -> int:
        if num_heads == 12:
            return 1 if total_chunks <= 128 else chunks_per_cta
        return original(num_heads=num_heads, total_chunks=total_chunks)

    kda_prefill._bt16_chunks_per_prepare_cta = policy
    try:
        yield
    finally:
        kda_prefill._bt16_chunks_per_prepare_cta = original


def _run(prepared, chunks_per_cta: int) -> None:
    with _h12_long_cpc(chunks_per_cta):
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


def _bootstrap_speedup(
    baseline_samples: list[float],
    candidate_samples: list[float],
    *,
    seed: int,
    draws: int = 10_000,
) -> tuple[float, float, float]:
    baseline = np.asarray(baseline_samples)
    candidate = np.asarray(candidate_samples)
    rng = np.random.default_rng(seed)
    baseline_medians = np.median(
        baseline[rng.integers(0, baseline.size, size=(draws, baseline.size))], axis=1
    )
    candidate_medians = np.median(
        candidate[rng.integers(0, candidate.size, size=(draws, candidate.size))], axis=1
    )
    speedups = baseline_medians / candidate_medians
    return (
        float(np.median(baseline) / np.median(candidate)),
        float(np.quantile(speedups, 0.025)),
        float(np.quantile(speedups, 0.975)),
    )


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
    for case in H12_CASES:
        with _h12_long_cpc(args.baseline_cpc):
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
            prepared.candidate_output,
            baseline_output,
            atol=1e-2,
            rtol=1e-2,
        )
        torch.testing.assert_close(
            prepared.candidate_state_pool[0],
            baseline_state,
            atol=1e-2,
            rtol=1e-2,
        )

        samples = {"baseline": [], "candidate": []}
        block_medians = {"baseline": [], "candidate": []}
        for name, chunks_per_cta in (
            ("baseline", args.baseline_cpc),
            ("candidate", args.candidate_cpc),
            ("candidate", args.candidate_cpc),
            ("baseline", args.baseline_cpc),
        ):
            prepared.reset_state_pools()
            block = _measure(
                prepared,
                chunks_per_cta,
                dry_run_iters=dry_run_iters,
                repeat_iters=repeat_iters,
            )
            samples[name].extend(block)
            block_medians[name].append(float(np.median(block)))

        speedup, lower, upper = _bootstrap_speedup(
            samples["baseline"], samples["candidate"], seed=case.seed
        )
        affected = (
            prepared.metadata["variant"] == "bt16_prepare_chain_m64"
            and sum((length + 15) // 16 for length in case.seq_lens) > 128
        )
        decision = "activate" if affected and lower > 1.0 else "retain_baseline"
        row = {
            "name": case.name,
            "num_heads": case.num_heads,
            "seq_lens": list(case.seq_lens),
            "layout": "packed" if case.packed else "fixed",
            "correct": True,
            "affected": affected,
            "baseline": {
                **prepared.metadata,
                "h12_long_chunks_per_prepare_cta": args.baseline_cpc,
                "median_ms": float(np.median(samples["baseline"])),
                "samples_ms": samples["baseline"],
                "block_medians_ms": block_medians["baseline"],
            },
            "candidate": {
                **prepared.metadata,
                "h12_long_chunks_per_prepare_cta": args.candidate_cpc,
                "median_ms": float(np.median(samples["candidate"])),
                "samples_ms": samples["candidate"],
                "block_medians_ms": block_medians["candidate"],
            },
            "speedup": speedup,
            "bootstrap_95": [lower, upper],
            "decision": decision,
            "pair_order": "baseline/candidate/candidate/baseline",
            "timing_scope": "public_recurrent_kda_with_inplace_state_update",
            "timing_backend": "cupti",
            "cold_l2": True,
            "same_initial_state_per_timed_call": True,
        }
        rows.append(row)
        print(
            f"{case.name:<22} {speedup:.4f}x "
            f"[{lower:.4f}, {upper:.4f}] {decision}"
        )
        del prepared, baseline_output, baseline_state
        torch.cuda.empty_cache()

    report = {
        "schema_version": 1,
        "analysis": "kimi-k3-tp8-h12-bt16-cpc-paired-qualification",
        "hardware": _hardware_metadata(torch.device("cuda")),
        "all_correct": all(row["correct"] for row in rows),
        "scope_identical": True,
        "baseline_cpc": args.baseline_cpc,
        "candidate_cpc": args.candidate_cpc,
        "rows": rows,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
