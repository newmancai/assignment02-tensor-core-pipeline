#!/usr/bin/env python3
"""Certify the preregistered Stage 10 capacity-policy interpolation test."""

from __future__ import annotations

import argparse
import hashlib
import json
from math import exp, log
from pathlib import Path

import numpy as np


EPOCHS = 8
PREREG_SHA256 = "15c48e28c262c7f5982847507db24f71584403064e12e331d3a099e22421ccc2"
BENCHMARK_SHA256 = "02a92be28bfa5192a94ffb2da2f6f67c932ea9b4cb3c0ff6857aaf8ffa185b54"
SBATCH_SHA256 = "f699e667d7199a88ff01e1738b97459f89e8935cd0af8588d1003eda26f93e7d"
EXPECTED_PREDICTIONS = {384: 7, 768: 13}
EXPECTED_GRIDS = {
    384: {"baseline": 516, "candidate": 660, "previous": 768},
    768: {"baseline": 1032, "candidate": 720, "previous": 768},
}
EXPECTED_PROFILES = {
    "h12_w384_fixed_6144",
    "h12_w384_balanced_4",
    "h12_w768_fixed_12288",
    "h12_w768_balanced_4",
}
EXPECTED_PHYSICAL_VARIANTS = [
    "sm_103a:flashkda_bf16_bt16_prepare_0d8e6c8011",
    "sm_103a:flashkda_bf16_bt16_chain_m64_c68ffebac9",
]


def _load(path: Path) -> dict:
    with path.open() as stream:
        return json.load(stream)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _familywise_interval(log_effects: list[float], seed: int) -> tuple[float, float, float]:
    values = np.asarray(log_effects, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.median(
        values[rng.integers(0, values.size, size=(50_000, values.size))], axis=1
    )
    return (
        exp(float(np.median(values))),
        exp(float(np.quantile(draws, 0.0125))),
        exp(float(np.quantile(draws, 0.9875))),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evidence/b300_stage10_certificate.json"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    evidence = root / "evidence"

    prereg = evidence / "stage10_h12_capacity_policy_preregistration.json"
    benchmark = root / "stage3_remote/benchmarks/bench_flash_kda_h12_wave_policy_heldout.py"
    sbatch = root / "stage10_b300_h12_wave_policy_heldout.sbatch"
    input_digests = {
        str(prereg.relative_to(root)): _sha256(prereg),
        str(benchmark.relative_to(root)): _sha256(benchmark),
        str(sbatch.relative_to(root)): _sha256(sbatch),
    }
    preregistration_sealed = input_digests == {
        "evidence/stage10_h12_capacity_policy_preregistration.json": PREREG_SHA256,
        "stage3_remote/benchmarks/bench_flash_kda_h12_wave_policy_heldout.py": BENCHMARK_SHA256,
        "stage10_b300_h12_wave_policy_heldout.sbatch": SBATCH_SHA256,
    }

    epoch_reports = []
    accounting_reports = []
    artifacts = dict(input_digests)
    for epoch_id in range(EPOCHS):
        result_path = evidence / f"b300_stage10_h12_wave_policy_heldout_epoch{epoch_id}.json"
        accounting_path = evidence / f"b300_stage10_h12_wave_policy_heldout_epoch{epoch_id}_accounting.json"
        epoch_reports.append(_load(result_path))
        accounting_reports.append(_load(accounting_path))
        artifacts[str(result_path.relative_to(root))] = _sha256(result_path)
        artifacts[str(accounting_path.relative_to(root))] = _sha256(accounting_path)

    row_sets_match = all(
        {row["name"] for row in report["rows"]} == EXPECTED_PROFILES
        for report in epoch_reports
    )
    epoch_ids_match = [report["epoch_id"] for report in epoch_reports] == list(range(EPOCHS))
    all_correct = all(
        report["all_correct"] and all(row["correct"] for row in report["rows"])
        for report in epoch_reports
    )
    all_scope_identical = all(report["scope_identical"] for report in epoch_reports)
    all_routes_match = all(
        row["route"] == "bt16_prepare_chain_m64"
        and row["target"] == "sm_103a"
        and row["physical_variants"] == EXPECTED_PHYSICAL_VARIANTS
        for report in epoch_reports
        for row in report["rows"]
    )
    predictions_and_grids_match = all(
        row["predicted_candidate_cpc"] == EXPECTED_PREDICTIONS[row["total_chunks"]]
        and row["baseline_cpc"] == 9
        and row["baseline_grid_ctas"] == EXPECTED_GRIDS[row["total_chunks"]]["baseline"]
        and row["candidate_grid_ctas"] == EXPECTED_GRIDS[row["total_chunks"]]["candidate"]
        and row["candidate_minus_one_grid_ctas"] == EXPECTED_GRIDS[row["total_chunks"]]["previous"]
        and row["resident_grid_capacity_ctas"] == 740
        for report in epoch_reports
        for row in report["rows"]
    )
    pair_orders_alternate = all(
        report["pair_order"]
        == (
            "baseline/candidate/candidate/baseline"
            if report["epoch_id"] % 2 == 0
            else "candidate/baseline/baseline/candidate"
        )
        for report in epoch_reports
    )

    profile_stats = []
    for profile_index, name in enumerate(sorted(EXPECTED_PROFILES)):
        epoch_effects = []
        baseline_medians = []
        candidate_medians = []
        for report in epoch_reports:
            row = next(row for row in report["rows"] if row["name"] == name)
            baseline = float(np.median(row["baseline_samples_ms"]))
            candidate = float(np.median(row["candidate_samples_ms"]))
            baseline_medians.append(baseline)
            candidate_medians.append(candidate)
            epoch_effects.append(log(baseline / candidate))
        speedup, lower, upper = _familywise_interval(epoch_effects, 17000 + profile_index)
        profile_stats.append(
            {
                "name": name,
                "total_chunks": next(
                    row["total_chunks"]
                    for row in epoch_reports[0]["rows"]
                    if row["name"] == name
                ),
                "epoch_speedups": [exp(value) for value in epoch_effects],
                "baseline_epoch_medians_ms": baseline_medians,
                "candidate_epoch_medians_ms": candidate_medians,
                "speedup": speedup,
                "bonferroni_one_sided_98_75_lower": lower,
                "bootstrap_98_75_upper": upper,
                "mechanism_gate": lower > 1.0,
                "deployment_gate_0_5_percent": lower >= 1.005,
            }
        )

    mechanism_gate = all(row["mechanism_gate"] for row in profile_stats)
    deployment_gate = all(row["deployment_gate_0_5_percent"] for row in profile_stats)
    direction_reversal = all(
        (
            row["candidate_grid_ctas"] > row["baseline_grid_ctas"]
            if row["total_chunks"] == 384
            else row["candidate_grid_ctas"] < row["baseline_grid_ctas"]
        )
        for row in epoch_reports[0]["rows"]
    )
    accounting_exit_zero = all(report["exit_code"] == 0 for report in accounting_reports)
    device_names = {
        item["name"]
        for report in accounting_reports
        for item in report["hardware"]
    }
    device_uuids = {
        item["uuid"]
        for report in accounting_reports
        for item in report["hardware"]
    }
    # Use the exact nvidia-smi product string already present in Stage 8
    # accounting, rather than the shortened paper label used in the first pass.
    same_single_b300 = (
        device_names == {"NVIDIA B300 SXM6 AC"}
        and len(device_uuids) == 1
        and all(report["gpu_count"] == 1 for report in accounting_reports)
    )
    checks = {
        "preregistration_sealed": preregistration_sealed,
        "eight_fresh_process_epochs": epoch_ids_match and len(epoch_reports) == EPOCHS,
        "profile_sets_match": row_sets_match,
        "all_correct": all_correct,
        "all_scope_identical": all_scope_identical,
        "all_routes_and_physical_variants_match": all_routes_match,
        "predictions_and_grids_match_preregistration": predictions_and_grids_match,
        "pair_orders_alternate": pair_orders_alternate,
        "direction_reversal_present": direction_reversal,
        "all_accounting_exit_zero": accounting_exit_zero,
        "same_single_b300": same_single_b300,
        "all_four_familywise_mechanism_gates": mechanism_gate,
        "all_four_half_percent_deployment_gates": deployment_gate,
    }
    status = (
        "prospective_capacity_rule_confirmed_and_shadow_activation_qualified"
        if all(checks.values())
        else "prospective_capacity_rule_not_qualified"
    )
    total_gpu_seconds = sum(report["allocated_gpu_seconds"] for report in accounting_reports)
    total_energy_joules = sum(
        report["telemetry"]["estimated_allocation_energy_joules"] or 0.0
        for report in accounting_reports
    )
    certificate = {
        "schema_version": 1,
        "status": status,
        "claim_boundary": (
            "Prospective interpolation on one physical B300 for the unchanged H12 "
            "BT16 prepare resource receipt. This qualifies shadow activation only; "
            "production activation, other work levels, heads, kernels and devices remain out of scope."
        ),
        "checks": checks,
        "physical_policy": _load(prereg)["physical_hypothesis"],
        "statistics": {
            "familywise_alpha": 0.05,
            "comparisons": 4,
            "one_sided_alpha_per_comparison": 0.0125,
            "bootstrap_draws": 50_000,
            "primary_unit": "fresh process epoch",
            "profiles": profile_stats,
        },
        "summary": {
            "profile_count": len(profile_stats),
            "epochs_per_profile": EPOCHS,
            "minimum_familywise_lower": min(
                row["bonferroni_one_sided_98_75_lower"] for row in profile_stats
            ),
            "speedup_range": [
                min(row["speedup"] for row in profile_stats),
                max(row["speedup"] for row in profile_stats),
            ],
            "allocated_gpu_seconds": total_gpu_seconds,
            "estimated_energy_joules": total_energy_joules,
            "device_uuid_count": len(device_uuids),
        },
        "artifacts": artifacts,
    }
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": status, "checks": checks, "summary": certificate["summary"]}, indent=2))
    raise SystemExit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
