# Copyright (c) 2026 by FlashInfer team.
"""CUPTI attribution of H12 cpc=4 versus cpc=9 to prepare and chain kernels."""

from __future__ import annotations

import argparse
import bisect
from functools import partial
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import torch

from bench_flash_kda_h12_physical_search import _bt16_prepare_grid
from bench_flash_kda_h12_profile_matrix import PROFILE_CASES
from bench_recurrent_kda_prefill import _hardware_metadata, _make_case, _require_cupti
from flashinfer.testing.utils import get_l2_cache_size


def _run(prepared, chunks_per_cta: int) -> None:
    with _bt16_prepare_grid(chunks_per_cta, True):
        prepared.candidate_run()


def _bootstrap_delta(
    baseline: list[float], candidate: list[float], *, seed: int
) -> tuple[float, float, float]:
    """Bootstrap baseline-minus-candidate median in microseconds."""

    rng = np.random.default_rng(seed)
    left = np.asarray(baseline, dtype=np.float64) * 1000.0
    right = np.asarray(candidate, dtype=np.float64) * 1000.0
    draws = np.empty(10000, dtype=np.float64)
    for index in range(draws.size):
        draws[index] = np.median(rng.choice(left, left.size, replace=True)) - np.median(
            rng.choice(right, right.size, replace=True)
        )
    point = float(np.median(left) - np.median(right))
    lower, upper = np.percentile(draws, [2.5, 97.5])
    return point, float(lower), float(upper)


def _measure_activity(
    prepared,
    chunks_per_cta: int,
    *,
    dry_run_iters: int,
    repeat_iters: int,
) -> tuple[list[dict[str, float]], list[str]]:
    """Return per-iteration ordered prepare/chain CUPTI kernel activity."""

    from cupti import cupti

    if int(version("cupti-python").split(".", 1)[0]) < 13:
        raise RuntimeError("cupti-python >= 13 is required")

    flush_bytes = int(get_l2_cache_size(torch.device("cuda"))) * 2
    l2_flush = torch.empty(flush_bytes, dtype=torch.int8, device="cuda")
    prepared.reset_state_pools()
    for _ in range(dry_run_iters):
        l2_flush.zero_()
        _run(prepared, chunks_per_cta)
    torch.cuda.synchronize()

    launches: list[tuple[int, int, int]] = []
    kernels: list[tuple[str, int, int, int]] = []

    def buffer_requested():
        return 8 * 1024 * 1024, 0

    def buffer_completed(activities):
        for activity in activities:
            if activity.kind == cupti.ActivityKind.CONCURRENT_KERNEL:
                kernels.append(
                    (
                        activity.name,
                        activity.start,
                        activity.end,
                        activity.correlation_id,
                    )
                )
            elif activity.kind in (
                cupti.ActivityKind.RUNTIME,
                cupti.ActivityKind.DRIVER,
            ):
                launches.append(
                    (activity.start, activity.end, activity.correlation_id)
                )

    windows: list[tuple[int, int]] = []
    cupti.activity_enable(cupti.ActivityKind.RUNTIME)
    cupti.activity_enable(cupti.ActivityKind.DRIVER)
    cupti.activity_enable(cupti.ActivityKind.CONCURRENT_KERNEL)
    cupti.activity_register_callbacks(
        buffer_requested, partial(buffer_completed)
    )
    prepared.reset_state_pools()
    for _ in range(repeat_iters):
        l2_flush.zero_()
        torch.cuda.synchronize()
        start_cpu = cupti.get_timestamp()
        _run(prepared, chunks_per_cta)
        end_cpu = cupti.get_timestamp()
        torch.cuda.synchronize()
        windows.append((start_cpu, end_cpu))
    cupti.activity_flush_all(0)
    cupti.activity_disable(cupti.ActivityKind.RUNTIME)
    cupti.activity_disable(cupti.ActivityKind.DRIVER)
    cupti.activity_disable(cupti.ActivityKind.CONCURRENT_KERNEL)
    cupti.finalize()

    sorted_launches = sorted(launches, key=lambda item: item[0])
    launch_starts = [item[0] for item in sorted_launches]
    by_correlation: dict[int, list[tuple[str, int, int, int]]] = {}
    for kernel in kernels:
        by_correlation.setdefault(kernel[3], []).append(kernel)

    samples = []
    names: set[str] = set()
    for index, (start_cpu, end_cpu) in enumerate(windows):
        left = bisect.bisect_left(launch_starts, start_cpu)
        right = bisect.bisect_right(launch_starts, end_cpu)
        correlation_ids = {sorted_launches[i][2] for i in range(left, right)}
        iteration_kernels = sorted(
            (
                kernel
                for correlation_id in correlation_ids
                for kernel in by_correlation.get(correlation_id, ())
            ),
            key=lambda item: item[1],
        )
        if len(iteration_kernels) != 2:
            compact = [(item[0], item[1], item[2]) for item in iteration_kernels]
            raise RuntimeError(
                f"expected exactly prepare then chain activity at iteration {index}, "
                f"got {compact}"
            )
        prepare, chain = iteration_kernels
        if prepare[2] > chain[1]:
            raise RuntimeError("prepare and chain activities unexpectedly overlap")
        names.update((prepare[0], chain[0]))
        samples.append(
            {
                "prepare_ms": (prepare[2] - prepare[1]) / 1e6,
                "inter_phase_gap_ms": (chain[1] - prepare[2]) / 1e6,
                "chain_ms": (chain[2] - chain[1]) / 1e6,
                "full_span_ms": (chain[2] - prepare[1]) / 1e6,
            }
        )
    return samples, sorted(names)


def _columns(samples: list[dict[str, float]]) -> dict[str, list[float]]:
    return {key: [row[key] for row in samples] for key in samples[0]}


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

    rows = []
    for case in PROFILE_CASES:
        with _bt16_prepare_grid(args.baseline_cpc, True):
            prepared = _make_case(
                case,
                state_rotations=args.state_rotations,
                candidate_route="dispatcher",
                candidate_backend="cake",
            )
        if len(prepared.metadata["physical_variants"]) != 2:
            raise RuntimeError(
                "phase attribution requires separate prepare and chain modules, got "
                f"{prepared.metadata['physical_variants']}"
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
        kernel_names: set[str] = set()
        for name, cpc in (
            ("baseline", args.baseline_cpc),
            ("candidate", args.candidate_cpc),
            ("candidate", args.candidate_cpc),
            ("baseline", args.baseline_cpc),
        ):
            block, names = _measure_activity(
                prepared,
                cpc,
                dry_run_iters=args.dry_run_iters,
                repeat_iters=args.repeat_iters,
            )
            samples[name].extend(block)
            kernel_names.update(names)

        columns = {name: _columns(values) for name, values in samples.items()}
        deltas = {}
        for phase in ("prepare_ms", "inter_phase_gap_ms", "chain_ms", "full_span_ms"):
            point, lower, upper = _bootstrap_delta(
                columns["baseline"][phase],
                columns["candidate"][phase],
                seed=case.seed + len(deltas),
            )
            deltas[phase] = {
                "baseline_median_us": float(
                    np.median(columns["baseline"][phase]) * 1000.0
                ),
                "candidate_median_us": float(
                    np.median(columns["candidate"][phase]) * 1000.0
                ),
                "baseline_minus_candidate_us": point,
                "bootstrap_95_us": [lower, upper],
            }
        full_delta = deltas["full_span_ms"]["baseline_minus_candidate_us"]
        prepare_delta = deltas["prepare_ms"]["baseline_minus_candidate_us"]
        chain_delta = deltas["chain_ms"]["baseline_minus_candidate_us"]
        rows.append(
            {
                "name": case.name,
                "seq_lens": list(case.seq_lens),
                "total_chunks": sum((length + 15) // 16 for length in case.seq_lens),
                "max_chunks": max((length + 15) // 16 for length in case.seq_lens),
                "route": prepared.metadata["variant"],
                "physical_variants": prepared.metadata["physical_variants"],
                "kernel_names": sorted(kernel_names),
                "correct": True,
                "samples": samples,
                "phases": deltas,
                "prepare_fraction_of_full_delta": (
                    prepare_delta / full_delta if full_delta else None
                ),
                "chain_fraction_of_full_delta": chain_delta / full_delta if full_delta else None,
            }
        )
        print(
            f"{case.name:<28} full={full_delta:7.3f} us "
            f"prepare={prepare_delta:7.3f} us chain={chain_delta:7.3f} us"
        )
        del prepared, baseline_output, baseline_state
        torch.cuda.empty_cache()

    report = {
        "schema_version": 1,
        "analysis": "stage7-h12-cupti-prepare-chain-activity-attribution",
        "claim_boundary": (
            "CUPTI activity attribution on the Stage 6 selection profiles; phase "
            "timing is causal for these two physical launches but is not a new "
            "untouched deployment qualification."
        ),
        "hardware": _hardware_metadata(torch.device("cuda")),
        "baseline_cpc": args.baseline_cpc,
        "candidate_cpc": args.candidate_cpc,
        "pair_order": "baseline/candidate/candidate/baseline",
        "protocol": {
            "timing_backend": "cupti-activity",
            "cold_l2": True,
            "l2_flush_bytes": int(get_l2_cache_size(torch.device("cuda"))) * 2,
            "same_prepared_object_per_pair": True,
            "state_rotations": args.state_rotations,
            "blocks_per_arm": 2,
            "dry_run_iters_per_block": args.dry_run_iters,
            "measured_iterations_per_block": args.repeat_iters,
            "bootstrap_draws": 10000,
        },
        "all_correct": all(row["correct"] for row in rows),
        "scope_identical": True,
        "rows": rows,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
