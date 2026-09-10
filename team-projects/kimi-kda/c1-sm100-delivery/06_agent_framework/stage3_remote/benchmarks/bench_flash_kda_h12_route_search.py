# Copyright (c) 2026 by FlashInfer team.
"""Profile counterfactual physical routes for the Kimi-K3 TP8 H12 preset."""

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
    _measure,
    _require_cupti,
    _timing_iteration_budget,
)
from flashinfer import kda_prefill


ROUTES = (
    kda_prefill._FLASH_KDA_ROUTE_DIRECT_M128,
    kda_prefill._FLASH_KDA_ROUTE_DIRECT_M128_N16,
    kda_prefill._FLASH_KDA_ROUTE_BT16_M64,
)


@contextmanager
def _force_route(route: str):
    original_runtime = kda_prefill._select_bf16_route
    original_public = kda_prefill._select_flash_kda_bf16_route
    kda_prefill._select_bf16_route = lambda **_kwargs: route
    kda_prefill._select_flash_kda_bf16_route = lambda **_kwargs: route
    try:
        yield
    finally:
        kda_prefill._select_bf16_route = original_runtime
        kda_prefill._select_flash_kda_bf16_route = original_public


def _correct(candidate, baseline) -> dict[str, float | bool]:
    candidate.reset_state_pools()
    baseline.reset_state_pools()
    candidate.candidate_run()
    baseline.candidate_run()
    torch.cuda.synchronize()
    output_delta = float(
        (candidate.candidate_output.float() - baseline.candidate_output.float())
        .abs()
        .max()
    )
    state_delta = float(
        (
            candidate.candidate_state_pool[0].float()
            - baseline.candidate_state_pool[0].float()
        )
        .abs()
        .max()
    )
    try:
        torch.testing.assert_close(
            candidate.candidate_output,
            baseline.candidate_output,
            atol=1e-2,
            rtol=1e-2,
        )
        torch.testing.assert_close(
            candidate.candidate_state_pool[0],
            baseline.candidate_state_pool[0],
            atol=1e-2,
            rtol=1e-2,
        )
    except AssertionError:
        passed = False
    else:
        passed = True
    return {
        "passed": passed,
        "output_max_abs": output_delta,
        "final_state_max_abs": state_delta,
    }


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
    for case in H12_CASES:
        baseline = _make_case(
            case,
            state_rotations=args.state_rotations,
            candidate_route="dispatcher",
            candidate_backend="cake",
        )
        baseline.reset_state_pools()
        baseline_ms, baseline_samples = _measure(
            baseline.candidate_run,
            dry_run_iters=dry_run_iters,
            repeat_iters=repeat_iters,
        )
        candidates = []
        for route in ROUTES:
            candidate = None
            try:
                with _force_route(route):
                    candidate = _make_case(
                        case,
                        state_rotations=args.state_rotations,
                        candidate_route="dispatcher",
                        candidate_backend="cake",
                    )
                    correctness = _correct(candidate, baseline)
                    result = {
                        "route": route,
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
                        result.update(
                            {
                                "median_ms": median_ms,
                                "samples_ms": samples,
                                "speedup_vs_dispatcher": baseline_ms / median_ms,
                            }
                        )
            except Exception as error:
                result = {
                    "route": route,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            candidates.append(result)
            if "median_ms" in result:
                print(
                    f"{case.name:<22} {route:<22} "
                    f"{result['median_ms'] * 1000:9.3f} us "
                    f"{result['speedup_vs_dispatcher']:.4f}x"
                )
            else:
                print(f"{case.name:<22} {route:<22} rejected")
            if candidate is not None:
                del candidate
        rows.append(
            {
                "name": case.name,
                "num_heads": case.num_heads,
                "seq_lens": list(case.seq_lens),
                "layout": "packed" if case.packed else "fixed",
                "dispatcher": {
                    "metadata": baseline.metadata,
                    "median_ms": baseline_ms,
                    "samples_ms": baseline_samples,
                },
                "candidates": candidates,
            }
        )
        del baseline
        torch.cuda.empty_cache()

    report = {
        "schema_version": 1,
        "analysis": "kimi-k3-tp8-h12-physical-route-counterfactual",
        "hardware": _hardware_metadata(torch.device("cuda")),
        "protocol": {
            "timing_backend": "cupti",
            "cold_l2": True,
            "public_api_scope": True,
            "same_initial_state_per_timed_call": True,
            "dry_run_iters": dry_run_iters,
            "repeat_iters": repeat_iters,
        },
        "rows": rows,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
