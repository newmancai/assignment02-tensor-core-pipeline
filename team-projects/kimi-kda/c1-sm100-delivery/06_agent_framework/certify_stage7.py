#!/usr/bin/env python3
"""Certify Stage 7 phase attribution, NCU mechanism, and agent replay."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from math import sqrt
from statistics import fmean, pstdev


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _pearson(left: list[float], right: list[float]) -> float:
    left_mean = fmean(left)
    right_mean = fmean(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right)
    )
    denominator = sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", type=Path, default=Path("evidence/b300_stage7_h12_phase_activity.json")
    )
    parser.add_argument(
        "--accounting",
        type=Path,
        default=Path("evidence/b300_stage7_h12_phase_activity_accounting.json"),
    )
    parser.add_argument(
        "--agent", type=Path, default=Path("evidence/stage7_autonomous_agent_replay.json")
    )
    parser.add_argument(
        "--ncu",
        type=Path,
        default=Path("evidence/b300_stage7_h12_prepare_ncu_summary.json"),
    )
    parser.add_argument(
        "--ncu-accounting",
        type=Path,
        default=Path("evidence/b300_stage7_h12_prepare_ncu_accounting.json"),
    )
    parser.add_argument(
        "--ncu-attempts",
        type=Path,
        nargs="*",
        default=[
            Path("evidence/attempts/20260909070302-23177-stage7-kimi-k3-tp8-h12-prepare-ncu.json"),
            Path("evidence/attempts/20260909071644-23195-stage7-kimi-k3-tp8-h12-prepare-ncu.json"),
        ],
    )
    parser.add_argument(
        "--output", type=Path, default=Path("evidence/b300_stage7_certificate.json")
    )
    args = parser.parse_args()
    phase = json.loads(args.phase.read_text())
    account = json.loads(args.accounting.read_text())
    agent = json.loads(args.agent.read_text())
    ncu = json.loads(args.ncu.read_text())
    ncu_account = json.loads(args.ncu_accounting.read_text())
    ncu_attempts = [json.loads(path.read_text()) for path in args.ncu_attempts]
    rows = phase["rows"]
    baseline = ncu["baseline_cpc4"]
    candidate = ncu["candidate_cpc9"]
    relative = ncu["candidate_relative_to_baseline"]

    checks = {
        "account_exit_zero": account["exit_code"] == 0,
        "correctness": phase["all_correct"],
        "scope_identical": phase["scope_identical"],
        "four_profiles": len(rows) == 4,
        "separate_prepare_chain_modules": all(
            len(row["physical_variants"]) == 2 for row in rows
        ),
        "prepare_delta_positive_95": all(
            row["phases"]["prepare_ms"]["bootstrap_95_us"][0] > 0 for row in rows
        ),
        "prepare_explains_at_least_95_percent": all(
            row["prepare_fraction_of_full_delta"] >= 0.95 for row in rows
        ),
        "chain_delta_under_one_microsecond": all(
            abs(row["phases"]["chain_ms"]["baseline_minus_candidate_us"]) < 1.0
            for row in rows
        ),
        "agent_replay_activates_all_profiles": agent["all_profiles_activated_cpc9"],
        "ncu_account_exit_zero": ncu_account["exit_code"] == 0,
        "ncu_grid_1536_to_684": baseline["grid_size"] == 1536 and candidate["grid_size"] == 684,
        "ncu_waves_2_08_to_0_92": baseline["waves_per_sm"] == 2.08 and candidate["waves_per_sm"] == 0.92,
        "ncu_resource_footprint_unchanged": ncu["interpretation"]["resource_footprint_unchanged"],
        "ncu_instruction_drop_under_10_percent": -0.10 < relative["instructions"] < 0.0,
        "ncu_profiled_duration_decreases": relative["gpu_duration_ns"] < 0.0,
        "ncu_active_period_throughput_increases": all(
            relative[key] > 0.0
            for key in ("sm_throughput_percent", "dram_throughput_percent", "l2_throughput_percent")
        ),
        "ncu_failed_attempt_retained": any(item["exit_code"] != 0 for item in ncu_attempts),
    }
    if not all(checks.values()):
        raise SystemExit(f"Stage 7 certification failed: {checks}")

    prepare_deltas = [
        row["phases"]["prepare_ms"]["baseline_minus_candidate_us"] for row in rows
    ]
    full_deltas = [
        row["phases"]["full_span_ms"]["baseline_minus_candidate_us"] for row in rows
    ]
    chain_times = [row["phases"]["chain_ms"]["baseline_median_us"] for row in rows]
    max_chunks = [row["max_chunks"] for row in rows]
    prepare_fractions = [row["prepare_fraction_of_full_delta"] for row in rows]
    report = {
        "schema_version": 1,
        "status": "prepare_phase_attributed_ncu_mechanism_supported_and_agent_loop_replayed",
        "checks": checks,
        "summary": {
            "prepare_saving_mean_us": fmean(prepare_deltas),
            "prepare_saving_range_us": [min(prepare_deltas), max(prepare_deltas)],
            "prepare_saving_cv": pstdev(prepare_deltas) / fmean(prepare_deltas),
            "full_span_saving_range_us": [min(full_deltas), max(full_deltas)],
            "prepare_fraction_of_full_delta_range": [
                min(prepare_fractions),
                max(prepare_fractions),
            ],
            "max_chunks_vs_chain_time_pearson": _pearson(max_chunks, chain_times),
            "allocated_gpu_seconds": account["allocated_gpu_seconds"],
            "estimated_energy_joules": account["telemetry"][
                "estimated_allocation_energy_joules"
            ],
            "stage7_total_attempts": 1 + len(ncu_attempts),
            "stage7_total_allocated_gpu_seconds": account["allocated_gpu_seconds"]
            + sum(item["allocated_gpu_seconds"] for item in ncu_attempts),
            "stage7_total_estimated_energy_joules": account["telemetry"][
                "estimated_allocation_energy_joules"
            ]
            + sum(item["telemetry"]["estimated_allocation_energy_joules"] for item in ncu_attempts),
            "ncu_grid_size": [baseline["grid_size"], candidate["grid_size"]],
            "ncu_waves_per_sm": [baseline["waves_per_sm"], candidate["waves_per_sm"]],
            "ncu_profiled_duration_relative_change": relative["gpu_duration_ns"],
            "ncu_instruction_relative_change": relative["instructions"],
        },
        "claim_boundary": (
            "The cpc=4 to cpc=9 gain is attributed to the prepare launch on the "
            "four Stage 6 selection profiles. Representative NCU counters support a "
            "CTA decomposition and scheduling-wave mechanism, but do not make NCU "
            "duration an authoritative timing gate, qualify a new deployment policy, "
            "establish cross-device transfer, or prove a multi-agent advantage."
        ),
        "artifacts": {
            str(path): _digest(path)
            for path in (
                args.phase,
                args.accounting,
                args.agent,
                args.ncu,
                args.ncu_accounting,
                *args.ncu_attempts,
            )
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
