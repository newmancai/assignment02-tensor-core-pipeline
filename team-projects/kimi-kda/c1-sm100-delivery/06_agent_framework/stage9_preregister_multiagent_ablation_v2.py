#!/usr/bin/env python3
"""Write the pre-GPU Stage 9 amendment with executable causal controls."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path

from kda_ir import AblationArm, EqualBudgetContract


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode()).hexdigest()


def _is_sha256(value: str | None) -> bool:
    return value is not None and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--original",
        type=Path,
        default=Path("evidence/stage9_multiagent_ablation_preregistration.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evidence/stage9_multiagent_ablation_amendment_v2.json"),
    )
    parser.add_argument(
        "--heldout-seal-sha256",
        help="SHA-256 commitment supplied by the external held-out custodian",
    )
    args = parser.parse_args()
    if args.heldout_seal_sha256 and not _is_sha256(args.heldout_seal_sha256):
        parser.error("--heldout-seal-sha256 must be 64 lowercase hexadecimal characters")

    original_bytes = args.original.read_bytes()
    original = json.loads(original_bytes)
    contract = EqualBudgetContract(
        trajectories=8,
        proposal_tokens_per_arm=500_000,
        proposal_slots_per_arm=64,
        screen_slots_per_arm=64,
        qualification_slots_per_arm=16,
        gpu_seconds_per_arm=160.0,
        gpu_balance_tolerance_fraction=0.05,
    )
    per_trajectory_caps = {
        "proposal_tokens": 62_500,
        "proposal_slots": 8,
        "screen_slots": 8,
        "qualification_slots": 2,
        "gpu_seconds": 20.0,
    }
    candidate_space = {
        "language": "typed_runtime_policy_v2",
        "features": [
            "total_chunks",
            "max_chunks",
            "num_sequences",
            "resident_grid_capacity_ctas",
        ],
        "predicate_operators": ["le", "gt"],
        "mutations": ["set_prepare_chunks_per_cta", "piecewise_profile_predicate"],
        "max_policy_depth": 2,
        "cpc_values": [1, 2, 3, 4, 5, 6, 8, 9, 12, 16, 17, 18, 20, 24, 32],
        "raw_cuda_edits": False,
        "canonicalization": "semantic_fields_only_full_sha256",
    }
    arm_protocols = {
        AblationArm.SINGLE_GENERALIST.value: {
            "backend_instances": 1,
            "roles": ["generalist"],
            "context": "one_trajectory_scoped_logical_context_reconstructed_from_verified_memory_each_round",
            "cross_lane_memory": False,
        },
        AblationArm.SINGLE_ROLEPLAY.value: {
            "backend_instances": 1,
            "roles": ["profile", "explore", "critic", "review"],
            "context": "one_trajectory_scoped_logical_context_with_sequential_roleplay_reconstructed_each_round",
            "cross_lane_memory": False,
        },
        AblationArm.MULTI_HOMOGENEOUS.value: {
            "backend_instances": 4,
            "roles": ["generalist", "generalist", "generalist", "generalist"],
            "context": "four_private_stateless_backend_invocations_per_trajectory_round",
            "cross_lane_memory": False,
        },
        AblationArm.MULTI_ROLE_NO_SHARE.value: {
            "backend_instances": 4,
            "roles": ["capacity", "tail_skew", "resource", "critic_synthesizer"],
            "context": "four_private_stateless_backend_invocations_per_trajectory_round",
            "cross_lane_memory": False,
        },
        AblationArm.MULTI_ROLE_CONCLUSIONS.value: {
            "backend_instances": 4,
            "roles": ["capacity", "tail_skew", "resource", "critic_synthesizer"],
            "context": "four_private_stateless_backend_invocations_per_trajectory_round",
            "cross_lane_memory": "verified_conclusions_only_after_round_barrier",
        },
    }
    heldout = {
        "dataset": "new_unmeasured_stage9_v2_heldout",
        "opaque_profile_ids": [f"s9v2-heldout-{index:02d}" for index in range(1, 9)],
        "profile_mapping_visible_to_agents": False,
        "custody": "external_to_all_agent_sandboxes",
        "seal_sha256": args.heldout_seal_sha256,
        "required_before": "first_agent_backend_call",
        "winner_freeze_before_unseal": True,
        "primary_queries_per_trajectory_per_arm": 1,
        "query_roles": ["frozen_winner"],
        "conditional_diagnostic_queries": "up_to_two_frozen_parent_controls_only_when_multi_role_conclusions_winner_is_compound",
        "feedback_to_search": False,
        "historical_stage5_or_stage10_evidence_reuse": False,
    }
    statistics_protocol = {
        "primary_metric": "uniform_shape_weighted_mean_log_speedup_per_trajectory",
        "paired_unit": "trajectory_seed",
        "kernel_repeats_and_shapes_are_not_independent_replicates": True,
        "primary_treatment": AblationArm.MULTI_ROLE_CONCLUSIONS.value,
        "primary_controls": [
            AblationArm.SINGLE_ROLEPLAY.value,
            AblationArm.MULTI_HOMOGENEOUS.value,
            AblationArm.MULTI_ROLE_NO_SHARE.value,
        ],
        "interval": "paired_trajectory_bootstrap_mean_difference",
        "familywise_confidence": 0.95,
        "multiplicity": "Bonferroni_three_one_sided_contrasts",
        "bootstrap_iterations": 20_000,
        "bootstrap_seed": 9,
        "success": "all_three_simultaneous_lower_bounds_above_zero",
        "compound_log_speedup_margin": math.log1p(0.005),
        "compound_test": "child_minus_each_parent_simultaneous_lower_bound_above_margin",
    }
    budget_protocol = {
        "scope": "per_arm_aggregated_over_eight_paired_trajectories",
        "contract": asdict(contract),
        "per_trajectory_caps": per_trajectory_caps,
        "fairness": "equal_opportunity_caps_not_equal_actual_padding",
        "proposal_charge": "every_backend_call_including_parse_failure_invalid_and_duplicate",
        "gpu_charge": "every_compile_correctness_screen_qualification_failure_timeout_and_retry",
        "screening": "only_verified_semantic_unique_candidates_reach_GPU",
        "no_winner": "score_as_baseline_log_speedup_zero_do_not_drop_trajectory",
    }
    protocol = {
        "arms": arm_protocols,
        "rounds": 2,
        "proposal_opportunities_per_round": 4,
        "gpu_executor": "single_B300_FIFO_round_robin_by_trajectory_round_arm",
        "branching": "deterministic_receipt_ranked_with_candidate_id_tiebreak",
        "memory": "arm_scoped_append_only_hash_chain",
        "candidate_space": candidate_space,
        "budget": budget_protocol,
        "heldout": heldout,
        "statistics": statistics_protocol,
        "authority": "typed_policy_verifier_correctness_scope_paired_measurement_and_certificate",
        "proposal_backend": {
            "runtime": "codex-cli-0.144.6-ephemeral-repo-external-read-only-failclosed-on-tool-event",
            "model": "gpt-5.3-codex-spark",
            "reasoning_effort": "medium",
            "common_prompt_sha256": sha256(Path("STAGE9_AGENT_PROTOCOL.md").read_bytes()).hexdigest(),
            "output_schema_sha256": sha256(Path("stage9_agent_output_schema.json").read_bytes()).hexdigest(),
            "development_summary_sha256": sha256(Path("evidence/stage9_development_summary.json").read_bytes()).hexdigest(),
            "runner_sha256": sha256(Path("run_stage9_agents.py").read_bytes()).hexdigest(),
            "arm_protocol_digests": {
                arm: _digest(specification) for arm, specification in arm_protocols.items()
            },
            "tool_policy": "any observed tool event rejects the complete proposal request",
            "isolation": {
                "working_directory": "fresh_private_tmp_outside_repository",
                "environment": "allowlisted_transport_variables_plus_ephemeral_HOME_and_CODEX_HOME",
                "user_configuration": "disabled",
                "historical_evidence_in_prompt": False,
                "heldout_secret_present_on_agent_host": False,
            },
        },
    }
    sealed = _is_sha256(args.heldout_seal_sha256)
    report = {
        "schema_version": 2,
        "status": "amended_ready_for_execution" if sealed else "amended_waiting_for_new_heldout_seal",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "amendment_timing": "before_first_stage9_agent_call_and_before_first_stage9_GPU_run",
        "amends": {
            "path": str(args.original),
            "file_sha256": sha256(original_bytes).hexdigest(),
            "protocol_digest": original["protocol_digest"],
        },
        "reasons": [
            "make paired trajectory inference executable",
            "separate common prompt envelope from arm-specific protocols",
            "treat budgets as opportunity caps and charge failed attempts",
            "add a specialized no-sharing control to identify conclusions-only communication",
            "replace historical held-out evidence with a new externally sealed unmeasured split",
        ],
        "protocol": protocol,
        "protocol_digest": _digest(protocol),
        "execution_preconditions": [
            "heldout.seal_sha256 is a valid external commitment",
            "typed_runtime_policy_v2 parser canonicalizer verifier and composer pass tests",
            "all five arm protocol prompt digests are recorded",
            "agent working directories are outside the repository; prompts exclude historical heldout evidence; any tool event is fail-closed; the new heldout mapping exists only on the B300 custodian host",
        ],
        "claim_boundary": (
            "This amendment contains no Stage 9 performance result. No agent or GPU "
            "execution is authorized until all execution_preconditions pass."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
