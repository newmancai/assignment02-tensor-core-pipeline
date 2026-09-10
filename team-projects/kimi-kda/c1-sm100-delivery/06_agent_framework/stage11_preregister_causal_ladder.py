#!/usr/bin/env python3
"""Write and validate the result-free Stage 11 tcgen05 causal ladder."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from kda_ir.stage11_causal_ladder import (
    CausalLadderPreregistration,
    FactorState,
    LadderCandidate,
    PathDefinition,
    ScreenPlan,
    validate_causal_ladder,
)


STRONGEST_BASELINE = (
    "sha256:f8c8471d660821234c0a16c1cc62e5c4472c03efab00792318d1b3741efa4eed"
)
HELD_FIXED = (
    "chunk_tokens_16",
    "token_recurrence_order",
    "archived_bf16_fp16_rounding_dag",
    "public_operator_abi_and_final_state_semantics",
    "sm_103a_target_and_frozen_toolchain",
)
GATES = (
    "static_verifier",
    "compile_sass_and_resource_receipt",
    "site_correctness",
    "public_output_and_final_state_correctness",
    "paired_low_budget_screen",
)
PROFILES = (
    "h12_t8192_fixed",
    "h64_t8192_fixed",
    "h96_t8192_packed_skew",
)


def _screen(*comparators: str) -> ScreenPlan:
    return ScreenPlan(
        ordered_gates=GATES,
        profiles=PROFILES,
        comparators=comparators,
        continue_rule="all_correct_and_any_preregistered_profile_has_positive_paired_screen_direction",
    )


def build_preregistration() -> CausalLadderPreregistration:
    lifecycle_paths = (
        "protocols.tcgen05_residency_group",
        "buffers.cross_phase_tmem_lifetime",
    )
    thin_n_paths = (
        "tiles.phase6.physical_mnk_m128n16k16",
        "launch.value_columns_v16_eight_slices",
        "launch.grid_h12_x_8_equals_96",
    )
    paths = (
        PathDefinition(
            "A_lifecycle",
            "lifecycle",
            lifecycle_paths,
            HELD_FIXED + ("full_n128_work_geometry",),
            "lifecycle_agent",
            "exact_rounding_or_operand_legality_erases_the_expected_transfer_reduction",
        ),
        PathDefinition(
            "B_thin_n",
            "thin_n",
            thin_n_paths,
            HELD_FIXED + ("per_site_tcgen05_allocation_lifecycle",),
            "thin_n_agent",
            "v16_m128n16k16_grid96_does_not_improve_active_work_or_protocol_amortization",
        ),
    )
    common_sources = (
        "candidate_matched_sass",
        "candidate_matched_resource_receipt",
        "candidate_matched_correctness_receipt",
        "candidate_matched_paired_timing",
    )
    common_stops = (
        "stop_on_static_compile_scope_or_correctness_failure",
        "stop_when_the_declared_mechanism_is_absent_from_sass_or_resource_receipts",
        "stop_qualification_when_no_screen_profile_has_positive_paired_direction",
    )
    control = "cell_00_direct_swap"
    a_only = "cell_10_lifecycle"
    b_only = "cell_01_thin_n"
    candidates = (
        LadderCandidate(
            control,
            FactorState.BASELINE,
            FactorState.BASELINE,
            ("instructions.k2_phase6_sm80_to_tcgen05",),
            HELD_FIXED + ("full_n128_work_geometry", "per_site_tcgen05_allocation_lifecycle"),
            STRONGEST_BASELINE,
            common_sources,
            _screen(STRONGEST_BASELINE),
            common_stops,
        ),
        LadderCandidate(
            a_only,
            FactorState.MUTATED,
            FactorState.BASELINE,
            lifecycle_paths,
            HELD_FIXED + ("full_n128_work_geometry",),
            STRONGEST_BASELINE,
            common_sources,
            _screen(control, STRONGEST_BASELINE),
            common_stops + ("stop_if_exact_rounding_preservation_eliminates_transfer_savings",),
        ),
        LadderCandidate(
            b_only,
            FactorState.BASELINE,
            FactorState.MUTATED,
            thin_n_paths,
            HELD_FIXED + ("per_site_tcgen05_allocation_lifecycle",),
            STRONGEST_BASELINE,
            common_sources,
            _screen(control, STRONGEST_BASELINE),
            common_stops + ("stop_if_v16_grid96_does_not_increase_useful_parallel_work",),
        ),
        LadderCandidate(
            "cell_11_lifecycle_thin_n",
            FactorState.MUTATED,
            FactorState.MUTATED,
            lifecycle_paths + thin_n_paths,
            HELD_FIXED,
            STRONGEST_BASELINE,
            common_sources,
            _screen(control, a_only, b_only, STRONGEST_BASELINE),
            common_stops + ("stop_if_interaction_cannot_be_estimated_from_all_four_cells",),
            available_after=(a_only, b_only),
        ),
    )
    return CausalLadderPreregistration(
        schema_version="stage11_tcgen05_causal_ladder_v1",
        experiment_id="stage11_tcgen05_lifecycle_x_thin_n",
        paths=paths,
        candidates=candidates,
        required_held_fixed=HELD_FIXED,
        strongest_baseline=STRONGEST_BASELINE,
        forbidden_attribution_sources=(
            "cake_full_stack_result",
            "cake_vs_flashkda_end_to_end_speedup",
            "unmatched_historical_kernel_timing",
        ),
        interaction_contrast="paired_log_speedup interaction = (AB-A)-(B-control)",
        cross_agent_protocol=(
            "A_and_B_use_independent_agent_contexts",
            "no_raw_trajectory_exchange_before_path_freeze",
            "interaction_authored_only_after_A_and_B_freeze",
            "only_verified_conclusions_cross_the_path_barrier",
        ),
        knowledge_policy=(
            "single_path_decision_grade_negative_is_reopenable_scoped_prior",
            "unresolved_or_noise_band_result_is_attempt_journal_only",
            "A_or_B_negative_does_not_cancel_AB",
            "reusable_negative_requires_two_resolved_independent_paths_and_AB",
            "target_toolchain_profile_or_held_fixed_change_invalidates_reuse",
            "compile_or_correctness_failure_is_not_a_performance_negative",
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evidence/stage11_tcgen05_causal_ladder_preregistration.json"),
    )
    args = parser.parse_args()
    prereg = build_preregistration()
    diagnostics = validate_causal_ladder(prereg)
    if diagnostics:
        raise SystemExit("; ".join(f"{item.code}: {item.message}" for item in diagnostics))
    protocol = asdict(prereg)
    protocol["knowledge_maturity_order"] = [
        "attempt",
        "single_path_observation",
        "multi_path_corroborated",
        "causal_mechanism",
        "transfer_validated",
    ]
    protocol["causal_attribution_rule"] = (
        "CAKE full-stack results may motivate a path but cannot estimate a tcgen05 "
        "main effect or interaction; only matched receipts from these four cells count."
    )
    report = {
        "schema_version": 1,
        "status": "preregistered_not_executed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol,
        "protocol_digest": sha256(
            json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "claim_boundary": "No GPU result or tcgen05 performance conclusion is contained here.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
