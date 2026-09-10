#!/usr/bin/env python3
"""Certify Stage 8 resident-wave scaling on two new total-work levels."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import ceil, exp, log
from pathlib import Path
from statistics import fmean


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _geomean(values: list[float]) -> float:
    return exp(fmean(log(value) for value in values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, default=Path("evidence/b300_stage8_h12_workscale_matrix.json"))
    parser.add_argument("--pair", type=Path, default=Path("evidence/b300_stage8_h12_workscale_pair.json"))
    parser.add_argument("--phase", type=Path, default=Path("evidence/b300_stage8_h12_workscale_phase.json"))
    parser.add_argument("--stage6-screen", type=Path, default=Path("evidence/b300_stage6_h12_profile_matrix.json"))
    parser.add_argument(
        "--accounting",
        type=Path,
        nargs="*",
        default=[
            Path("evidence/b300_stage8_h12_workscale_matrix_accounting.json"),
            Path("evidence/b300_stage8_h12_workscale_pair_accounting.json"),
            Path("evidence/b300_stage8_h12_workscale_phase_accounting.json"),
        ],
    )
    parser.add_argument("--output", type=Path, default=Path("evidence/b300_stage8_certificate.json"))
    args = parser.parse_args()

    screen = _load(args.screen)
    pair = _load(args.pair)
    phase = _load(args.phase)
    stage6 = _load(args.stage6_screen)
    accounting = [_load(path) for path in args.accounting]
    capacity = screen["physical_model"]["resident_grid_capacity_ctas"]
    heads = screen["rows"][0]["num_heads"]
    chunk_groups = capacity // heads

    predicted = {
        total: ceil(total / chunk_groups) for total in (256, 512, 1024)
    }
    screen_rows = screen["rows"]
    pair_rows = pair["rows"]
    phase_rows = phase["rows"]
    stage6_winners = [
        min(row["candidates"], key=lambda item: item["median_ms"])["chunks_per_cta"]
        for row in stage6["rows"]
    ]
    checks = {
        "all_accounting_exit_zero": all(item["exit_code"] == 0 for item in accounting),
        "screen_preregistered": screen["preregistered_before_first_gpu_run"],
        "screen_correct_and_scope_identical": all(
            row["route_invariant"]
            and all(item["correctness"]["passed"] for item in row["candidates"])
            for row in screen_rows
        ),
        "prediction_hits_all_new_screens": all(
            row["screen_winner_cpc"] == row["predicted_one_wave_cpc"]
            for row in screen_rows
        ),
        "three_level_scaling_law": (
            predicted == {256: 5, 512: 9, 1024: 17}
            and all(value == 9 for value in stage6_winners)
        ),
        "pair_correct_and_scope_identical": pair["all_correct"] and pair["scope_identical"],
        "all_pair_lower_bounds_positive": all(row["bootstrap_95"][0] > 1.0 for row in pair_rows),
        "phase_correct_and_scope_identical": phase["all_correct"] and phase["scope_identical"],
        "all_prepare_deltas_positive_95": all(
            row["phases"]["prepare_ms"]["bootstrap_95_us"][0] > 0.0
            for row in phase_rows
        ),
        "prepare_explains_at_least_95_percent": all(
            row["prepare_fraction_of_full_delta"] >= 0.95 for row in phase_rows
        ),
        "chain_change_under_one_microsecond": all(
            abs(row["phases"]["chain_ms"]["baseline_minus_candidate_us"]) < 1.0
            for row in phase_rows
        ),
    }
    if not all(checks.values()):
        raise SystemExit(f"Stage 8 certification failed: {checks}")

    speedups = [row["speedup"] for row in pair_rows]
    prepare_deltas = [
        row["phases"]["prepare_ms"]["baseline_minus_candidate_us"]
        for row in phase_rows
    ]
    full_deltas = [
        row["phases"]["full_span_ms"]["baseline_minus_candidate_us"]
        for row in phase_rows
    ]
    report = {
        "schema_version": 1,
        "status": "resident_wave_scaling_confirmed_on_two_new_work_levels",
        "checks": checks,
        "physical_policy": {
            "formula": "ceil(total_chunks / floor((sm_count * resident_ctas_per_sm) / num_heads))",
            "conditions": {
                "sm_count": 148,
                "resident_ctas_per_sm": 5,
                "num_heads": 12,
                "unchanged_prepare_resource_footprint": True,
            },
            "predicted_and_screen_winner_cpc": {str(key): value for key, value in predicted.items()},
        },
        "summary": {
            "profiles": len(pair_rows),
            "paired_speedup_range": [min(speedups), max(speedups)],
            "paired_speedup_geomean": _geomean(speedups),
            "minimum_bootstrap_95_lower": min(row["bootstrap_95"][0] for row in pair_rows),
            "prepare_saving_range_us": [min(prepare_deltas), max(prepare_deltas)],
            "full_span_saving_range_us": [min(full_deltas), max(full_deltas)],
            "stage8_attempts": len(accounting),
            "stage8_allocated_gpu_seconds": sum(item["allocated_gpu_seconds"] for item in accounting),
            "stage8_estimated_energy_joules": sum(
                item["telemetry"]["estimated_allocation_energy_joules"] for item in accounting
            ),
        },
        "claim_boundary": (
            "The one-resident-wave cpc formula predicted the screen winner at W=256, "
            "W=512, and W=1024 on one B300 for this unchanged H12 prepare kernel. "
            "Stage 8 paired and phase tests reuse selection profiles; deployment, "
            "other head counts, resource footprints, devices, and multi-agent advantage "
            "require separate qualification."
        ),
        "artifacts": {
            str(path): _digest(path)
            for path in (
                args.screen,
                args.pair,
                args.phase,
                args.stage6_screen,
                *args.accounting,
            )
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
