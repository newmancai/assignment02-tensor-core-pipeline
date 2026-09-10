#!/usr/bin/env python3
"""Freeze the small, best-architecture Stage 9 engineering pilot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal-sha256")
    parser.add_argument("--supersedes-protocol-digest")
    parser.add_argument("--output", type=Path,
                        default=Path("evidence/stage9_best_swarm_pilot_protocol.json"))
    args = parser.parse_args()
    if args.seal_sha256 is not None and (
        len(args.seal_sha256) != 64
        or any(char not in "0123456789abcdef" for char in args.seal_sha256)
    ):
        parser.error("--seal-sha256 must be 64 lowercase hexadecimal characters")
    arm = "multi_role_conclusions"
    candidate_space = {
        "language": "typed_runtime_policy_v2",
        "features": ["total_chunks", "max_chunks", "num_sequences", "resident_grid_capacity_ctas"],
        "predicate_operators": ["le", "gt"],
        "max_policy_depth": 2,
        "cpc_values": [1, 2, 3, 4, 5, 6, 8, 9, 12, 16, 17, 18, 20, 24, 32],
        "raw_cuda_edits": False,
        "canonicalization": "semantic_fields_only_full_sha256",
    }
    arm_spec = {
        "backend_instances": 4,
        "roles": ["capacity", "tail_skew", "resource", "critic_synthesizer"],
        "round0": "four_private_role_proposals",
        "round1": "verified_receipt_ranked_conclusions_shared_after_barrier",
        "selection": "B300_correctness_then_latency_top_k",
        "variation": "critic_synthesizer_must_compose_two_distinct_parent_subtrees",
    }
    protocol = {
        "purpose": "engineering_pilot_not_a_statistical_multiagent_superiority_claim",
        "arms": {arm: arm_spec},
        "trajectory_seeds": [91001],
        "rounds": 2,
        "proposal_opportunities_per_round": 4,
        "gpu_executor": "single_B300_FIFO_with_round_barrier",
        "branching": "receipt_ranked_top_k_then_lineage_checked_compound_mutation",
        "memory": "arm_scoped_append_only_hash_chain_verified_conclusions_only",
        "candidate_space": candidate_space,
        "budget": {
            "contract": {
                "trajectories": 1,
                "proposal_tokens_per_arm": 200000,
                "proposal_slots_per_arm": 8,
                "screen_slots_per_arm": 8,
                "qualification_slots_per_arm": 2,
                "gpu_seconds_per_arm": 60.0,
                "gpu_balance_tolerance_fraction": 0.05,
            },
            "per_trajectory_caps": {
                "proposal_tokens": 200000,
                "proposal_slots": 8,
                "screen_slots": 8,
                "qualification_slots": 2,
                "gpu_seconds": 60.0,
            },
            "proposal_charge": "all_calls_failures_invalid_and_duplicates",
        },
        "heldout": {
            "dataset": "stage9_best_swarm_pilot_remote_only_randomized_v2",
            "opaque_profile_ids": [f"s9pilot-heldout-{index:02d}" for index in range(1, 9)],
            "total_chunk_envelope": [288, 336, 400, 464],
            "sequence_count_choices": [1, 2, 4],
            "seal_sha256": args.seal_sha256,
            "custody": "B300_host_only_never_agent_host",
            "winner_freeze_before_unseal": True,
        },
        "proposal_backend": {
            "runtime": "codex-cli-0.144.6-ephemeral-repo-external-read-only-failclosed-on-tool-event",
            "model": "gpt-5.6-sol",
            "reasoning_effort": "high",
            "common_prompt_sha256": sha256((ROOT / "STAGE9_AGENT_PROTOCOL.md").read_bytes()).hexdigest(),
            "output_schema_sha256": sha256((ROOT / "stage9_agent_output_schema.json").read_bytes()).hexdigest(),
            "development_summary_sha256": sha256((ROOT / "evidence/stage9_development_summary.json").read_bytes()).hexdigest(),
            "runner_sha256": sha256((ROOT / "run_stage9_agents.py").read_bytes()).hexdigest(),
            "gpu_evaluator_sha256": sha256((ROOT / "stage3_remote/benchmarks/bench_flash_kda_stage9_ablation.py").read_bytes()).hexdigest(),
            "arm_protocol_digests": {arm: digest(arm_spec)},
            "tool_policy": "any observed tool event rejects_the_complete_request",
            "isolation": "fresh_private_tmp_outside_repo_minimal_environment_no_heldout_secret",
        },
    }
    report = {
        "schema_version": 2,
        "status": "amended_ready_for_execution" if args.seal_sha256 else "amended_waiting_for_new_heldout_seal",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pilot": True,
        "supersedes_protocol_digest": args.supersedes_protocol_digest,
        "amendment_reason": (
            "round1 receipt adapter corrected and synthesizer serialized after three typed-verified scouts"
            if args.supersedes_protocol_digest else None
        ),
        "claim_boundary": "Pipeline feasibility only; no multi-agent superiority claim.",
        "protocol": protocol,
        "protocol_digest": digest(protocol),
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
