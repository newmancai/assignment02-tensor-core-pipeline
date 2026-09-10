#!/usr/bin/env python3
"""Certify post-qualification CUPTI phase attribution for Stage 10."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence"
EXPECTED_PROFILES = {
    "h12_w384_fixed_6144",
    "h12_w384_balanced_4",
    "h12_w768_fixed_12288",
    "h12_w768_balanced_4",
}
EXPECTED_VARIANTS = [
    "sm_103a:flashkda_bf16_bt16_prepare_0d8e6c8011",
    "sm_103a:flashkda_bf16_bt16_chain_m64_c68ffebac9",
]


def _load(path: Path) -> dict:
    with path.open() as stream:
        return json.load(stream)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    qualification_path = EVIDENCE / "b300_stage10_certificate.json"
    phase_path = EVIDENCE / "b300_stage10_h12_capacity_phase.json"
    accounting_path = EVIDENCE / "b300_stage10_h12_capacity_phase_accounting.json"
    qualification = _load(qualification_path)
    phase = _load(phase_path)
    accounting = _load(accounting_path)
    rows = phase["rows"]
    checks = {
        "qualification_passed_before_diagnostic": qualification["status"]
        == "prospective_capacity_rule_confirmed_and_shadow_activation_qualified",
        "four_frozen_profiles": {row["name"] for row in rows} == EXPECTED_PROFILES,
        "all_correct_and_scope_identical": phase["all_correct"]
        and phase["scope_identical"],
        "all_routes_match": all(
            row["route"] == "bt16_prepare_chain_m64"
            and row["target"] == "sm_103a"
            and row["physical_variants"] == EXPECTED_VARIANTS
            for row in rows
        ),
        "all_prepare_savings_positive_95": all(
            row["gates"]["prepare_saving_positive_95"] for row in rows
        ),
        "all_full_span_savings_positive_95": all(
            row["gates"]["full_span_saving_positive_95"] for row in rows
        ),
        "all_chain_effects_inside_frozen_one_percent_equivalence_band": all(
            row["gates"]["chain_equivalent_within_one_percent_95"] for row in rows
        ),
        "prepare_explains_at_least_90_percent_everywhere": all(
            row["gates"]["prepare_explains_at_least_90_percent"] for row in rows
        ),
        "accounting_exit_zero": accounting["exit_code"] == 0,
    }
    full = [row["phases"]["full_span_ms"]["baseline_minus_candidate_us"] for row in rows]
    prepare = [row["phases"]["prepare_ms"]["baseline_minus_candidate_us"] for row in rows]
    fractions = [row["prepare_fraction_of_full_delta"] for row in rows]
    chain_ci_abs = [
        abs(bound)
        for row in rows
        for bound in row["phases"]["chain_ms"]["bootstrap_95_us"]
    ]
    certificate = {
        "schema_version": 1,
        "status": (
            "prepare_capacity_mechanism_confirmed_on_stage10_profiles"
            if all(checks.values())
            else "stage10_mechanism_not_confirmed"
        ),
        "claim_boundary": (
            "Post-qualification CUPTI attribution for the four Stage10 profiles on one "
            "B300. It supports the prepare-grid mechanism but is not an independent "
            "performance qualification or a universal causal proof."
        ),
        "checks": checks,
        "summary": {
            "full_span_saving_range_us": [min(full), max(full)],
            "prepare_saving_range_us": [min(prepare), max(prepare)],
            "prepare_fraction_range": [min(fractions), max(fractions)],
            "largest_absolute_chain_95_bound_us": max(chain_ci_abs),
            "allocated_gpu_seconds": accounting["allocated_gpu_seconds"],
            "estimated_energy_joules": accounting["telemetry"][
                "estimated_allocation_energy_joules"
            ],
        },
        "artifacts": {
            str(qualification_path.relative_to(ROOT)): _sha(qualification_path),
            str(phase_path.relative_to(ROOT)): _sha(phase_path),
            str(accounting_path.relative_to(ROOT)): _sha(accounting_path),
        },
    }
    output = EVIDENCE / "b300_stage10_mechanism_certificate.json"
    output.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n")
    print(json.dumps(certificate, indent=2, sort_keys=True))
    raise SystemExit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
