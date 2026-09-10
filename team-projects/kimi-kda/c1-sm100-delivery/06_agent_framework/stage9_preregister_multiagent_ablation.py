#!/usr/bin/env python3
"""Write the executable equal-budget MARPE ablation preregistration."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

from kda_ir import AblationArm, EqualBudgetContract


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode()).hexdigest()


def main() -> None:
    output = Path("evidence/stage9_multiagent_ablation_preregistration.json")
    # Keep the historical v1 artifact reproducible after v2 adds a dedicated
    # no-sharing control to AblationArm.
    legacy_arms = (
        AblationArm.SINGLE_GENERALIST,
        AblationArm.SINGLE_ROLEPLAY,
        AblationArm.MULTI_HOMOGENEOUS,
        AblationArm.MULTI_ROLE_CONCLUSIONS,
    )
    contract = EqualBudgetContract(
        trajectories=3,
        proposal_tokens_per_arm=150000,
        proposal_slots_per_arm=24,
        screen_slots_per_arm=24,
        qualification_slots_per_arm=4,
        gpu_seconds_per_arm=60.0,
        gpu_balance_tolerance_fraction=0.05,
    )
    candidate_space = {
        "language": "typed_runtime_policy_v1",
        "features": [
            "total_chunks",
            "max_chunks",
            "num_sequences",
            "resident_grid_capacity_ctas",
        ],
        "mutations": ["set_prepare_chunks_per_cta", "piecewise_profile_predicate"],
        "max_policy_depth": 2,
        "cpc_values": [1, 2, 3, 4, 5, 6, 8, 9, 12, 16, 17, 18, 20, 24, 32],
        "raw_cuda_edits": False,
    }
    protocol = {
        "arms": [arm.value for arm in legacy_arms],
        "baseline_for_multiagent_claim": [
            AblationArm.SINGLE_ROLEPLAY.value,
            AblationArm.MULTI_HOMOGENEOUS.value,
        ],
        "gpu_executor": "single_B300_FIFO_round_robin_across_arms",
        "deduplication": "duplicates_consume_proposal_slots_but_measure_once",
        "authority": "KIR_verifier_correctness_oracle_paired_CUPTI_gate",
        "memory": "conclusions_only_for_multi_role_arm",
        "heldout": "opaque_ids_sealed_until_policy_freeze",
        "primary_metric": "uniform_shape_weighted_heldout_log_speedup",
        "primary_test": (
            "paired episode difference: multi_role must beat both single_roleplay "
            "and multi_homogeneous with a positive 95% lower bound"
        ),
        "emergence_requirement": (
            "at least one cross-lineage compound typed candidate whose heldout gain "
            "exceeds both parents by the preregistered noise margin"
        ),
        "negative_interpretations": {
            "multi_homogeneous_matches_multi_role": "best_of_N_not_collaboration",
            "single_roleplay_matches_multi_role": "role_workflow_not_multi_agent",
            "no_communication_matches_multi_role": "independent_search_width_only",
        },
    }
    report = {
        "schema_version": 1,
        "status": "preregistered_not_executed",
        "budget_contract": asdict(contract),
        "candidate_space": candidate_space,
        "candidate_space_digest": _digest(candidate_space),
        "protocol": protocol,
        "protocol_digest": _digest(protocol),
        "claim_boundary": (
            "This file freezes the comparison design. It contains no multi-agent "
            "performance result and cannot support an emergence claim."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
