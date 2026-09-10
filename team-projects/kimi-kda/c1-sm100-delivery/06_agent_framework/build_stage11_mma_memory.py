#!/usr/bin/env python3
"""Convert archived C1 measurements into typed Stage-11 migration memory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict
from enum import Enum
from pathlib import Path

from kda_ir.mma_migration import (
    ConclusionState,
    FailureClass,
    InstructionSpec,
    LaunchTopology,
    LogicalPhysicalTile,
    MathSiteContract,
    MemorySpace,
    MmaFamily,
    MigrationCandidate,
    MigrationFailureCard,
    MigrationTier,
    ResourceContract,
    Tcgen05Protocol,
    canonical_candidate_id,
    verify_failure_card,
    verify_mma_migration,
)
from kda_ir.mma_knowledge import (
    ChallengeStatus,
    CounterfactualChallenge,
    KnowledgeDisposition,
    KnowledgeKind,
    KnowledgeMaturity,
    MigrationKnowledgeClaim,
    verify_knowledge_claim,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_default(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"cannot serialize {type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--capabilities", type=Path, required=True)
    parser.add_argument("--cross-path-evidence", type=Path)
    parser.add_argument("--v16-screen-evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    profile = json.loads(args.profile.read_text())
    expected = {
        profile["sass_evidence_ref"]: profile["sass_evidence_sha256"],
        profile["semantic_evidence_ref"]: profile["semantic_evidence_sha256"],
        profile["tcgen_evidence_ref"]: profile["tcgen_evidence_sha256"],
        profile["official"]["evidence_ref"]: profile["ncu_evidence_sha256"],
    }
    for relative, digest in expected.items():
        actual = sha256(args.evidence_root / relative)
        if actual != digest:
            raise ValueError(f"evidence digest mismatch: {relative}: {actual} != {digest}")

    site = MathSiteContract(
        site_id="k2_phase6_state_update",
        logical_mnk=(128, 128, 16),
        input_dtype="bf16",
        accumulator_dtype="f32",
        output_dtype="bf16",
        operand_major=("k", "k"),
        rounding_dag_sha256=profile["semantic_evidence_sha256"],
        source_ref="FlashKDA/csrc/smxx/fwd_kernel2.cuh:659",
    )
    direct = MigrationCandidate(
        schema_version="typed_mma_migration_v2",
        name="tcgen05-phase6-direct-swap",
        tier=MigrationTier.INSTRUCTION_ONLY,
        baseline_sha256=profile["sass_evidence_sha256"],
        capability_table_sha256=sha256(args.capabilities),
        target_arch=profile["target_arch"],
        chunk=16,
        math_sites=(site,),
        instructions=(InstructionSpec(site.site_id, MmaFamily.SM100_TCGEN05, "dense.bf16.f32.cta1.m128n128k16", (128, 128, 16), MemorySpace.TMEM),),
        tiles=(LogicalPhysicalTile(site.site_id, site.logical_mnk, site.logical_mnk, False, False),),
        protocols=(Tcgen05Protocol(site.site_id, 1, "mma_warp", "mma_warp", 0, 128, True, True, True, True, True),),
        resources=ResourceContract(192, profile["tcgen_l1_v128_registers"], profile["tcgen_l1_v128_static_smem_bytes"], 128, profile["tcgen_l1_v128_active_blocks_per_sm"]),
        preserves_token_order=True,
        preserves_rounding_dag=True,
        mutation_paths=("instructions.k2_phase6_state_update", "protocols.k2_phase6_state_update", "resources"),
        launch_topology=LaunchTopology(12, 1, "sequence_head", (("load", 1), ("compute", 4), ("store", 1))),
        comparison_baseline_id="sha256:" + profile["sass_evidence_sha256"],
    )
    diagnostics = verify_mma_migration(direct)
    if diagnostics:
        raise ValueError("; ".join(f"{d.code}: {d.message}" for d in diagnostics))
    direct_id = canonical_candidate_id(direct)
    card = MigrationFailureCard(
        technique_id="tcgen05-phase6-direct-swap",
        candidate_id=direct_id,
        state=ConclusionState.SLOWER,
        failure_class=FailureClass.NO_PRACTICAL_GAIN,
        causal_status="observed_and_sass_confirmed",
        applicability_key=f"{profile['profile_id']}:phase6:v128:grid12:inner64:l0",
        exact_conditions=(
            "NVIDIA B300 SXM6 AC / sm_103a",
            "Kimi-K3 TP8 local_heads=12, D=128, CHUNK=16",
            "Phase-6 logical m128n128k16, BF16 inputs, FP32 accumulate",
            "L0 preferred on-chip layout, grid=12, inner=64",
        ),
        observed_speedup=profile["tcgen_l0_v128_speedup"],
        evidence_refs=(profile["tcgen_evidence_ref"], profile["semantic_evidence_ref"]),
        evidence_sha256=(profile["tcgen_evidence_sha256"], profile["semantic_evidence_sha256"]),
        surviving_fact="After amortizing fixed protocol cost, direct Phase-6 tcgen05 remained slower than mma.sync (0.919742x).",
        do_not_generalize_beyond="This does not reject cross-phase TMEM residency, another GPU/toolchain, decode, or a full dataflow rewrite.",
        reopen_condition=(
            "A typed design keeps compatible intermediates TMEM-resident across Phases 1/3/4/6.",
            "A changed target/toolchain/resource receipt invalidates this applicability key.",
        ),
    )
    card_diagnostics = verify_failure_card(card)
    if card_diagnostics:
        raise ValueError("; ".join(f"{d.code}: {d.message}" for d in card_diagnostics))

    v16_screen_summary = None
    v16_status = ChallengeStatus.PLANNED
    v16_refs: tuple[str, ...] = ()
    if args.v16_screen_evidence is not None:
        with args.v16_screen_evidence.open(newline="") as source:
            rows = list(csv.DictReader(line for line in source if not line.startswith("#")))
        selected = {
            (row["level"], int(row["inner"])): row
            for row in rows
            if int(row["V"]) == 16 and int(row["grid"]) == 96
        }
        required = {("L0", 1), ("L1", 1), ("L0", 64), ("L1", 64)}
        if set(selected) != required:
            raise ValueError("V16 screen needs exactly L0/L1 x inner1/64 at V16 grid96")
        v16_status = ChallengeStatus.INCONCLUSIVE
        v16_refs = (str(args.v16_screen_evidence),)
        v16_screen_summary = {
            "evidence_ref": str(args.v16_screen_evidence),
            "sha256": sha256(args.v16_screen_evidence),
            "job_id": 24111,
            "target": "NVIDIA B300 SXM6 AC / sm_103a",
            "correctness": "output_and_state_probe_passed",
            "l0_inner1_speedup": float(selected[("L0", 1)]["speedup_median"]),
            "l0_inner64_speedup": float(selected[("L0", 64)]["speedup_median"]),
            "l1_inner1_speedup": float(selected[("L1", 1)]["speedup_median"]),
            "l1_inner64_speedup": float(selected[("L1", 64)]["speedup_median"]),
            "interpretation": (
                "Thin-N tcgen05 has a positive amortized preferred-layout core, "
                "but per-iteration scalar materialization erases it; this is a "
                "mechanism screen, not completion of the full ValueSlice cell."
            ),
        }

    challenges = [
        CounterfactualChallenge(
            "cross-phase-carrier", "lifecycle_amortization",
            ("math_sites", "tmem_carriers", "protocols"),
            ("launch_topology.grid_ctas=12", "CHUNK16", "rounding_dag"),
            ChallengeStatus.PLANNED,
        ),
        CounterfactualChallenge(
            "value-slice-thin-n", "parallelism_and_tile",
            ("launch_topology.value_slices", "launch_topology.grid_ctas", "tiles"),
            ("per-site lifecycle", "CHUNK16", "rounding_dag"),
            v16_status,
            v16_refs,
        ),
        CounterfactualChallenge(
            "saturated-hoisted-control", "mechanism_control",
            ("launch_topology.grid_ctas", "protocols.alloc_scope"),
            ("math_site", "m128n128k16", "rounding_dag"),
            ChallengeStatus.PLANNED,
        ),
    ]
    cross_path_summary = None
    if args.cross_path_evidence is not None:
        rows = json.loads(args.cross_path_evidence.read_text())
        if not rows or any(row.get("correctness_peer") != "passed" for row in rows):
            raise ValueError("cross-path evidence must contain only correctness-passing rows")
        speedups = [float(row["speedup_vs_flash_kda_peer_raw"]) for row in rows]
        cross_digest = sha256(args.cross_path_evidence)
        challenges.insert(
            0,
            CounterfactualChallenge(
                "cake-full-stack-sm100", "cross_phase_full_stack",
                ("math_sites", "tmem_carriers", "launch_topology", "roles", "pipelines"),
                ("public_math_contract", "CHUNK16", "B300", "Kimi-K3-H12-profiles"),
                ChallengeStatus.OBSERVED_REFUTE,
                (str(args.cross_path_evidence),),
            ),
        )
        cross_path_summary = {
            "evidence_ref": str(args.cross_path_evidence),
            "sha256": cross_digest,
            "job_id": rows[0].get("hardware", {}).get("device_name", "B300"),
            "profiles": len(rows),
            "all_correct": True,
            "raw_speedup_min": min(speedups),
            "raw_speedup_max": max(speedups),
            "raw_speedup_geomean": math.exp(sum(math.log(x) for x in speedups) / len(speedups)),
            "interpretation": "Refutes extrapolating the direct-swap loss into a blanket SM100-migration negative; does not isolate tcgen05 causality.",
        }

    claim = MigrationKnowledgeClaim(
        claim_id="direct-m128-negative-v1",
        proposition="Do not repeat the same isolated Phase-6 m128n128 tcgen05 swap.",
        kind=KnowledgeKind.PERFORMANCE_DECISION,
        maturity=KnowledgeMaturity.SINGLE_PATH_OBSERVATION,
        disposition=KnowledgeDisposition.SCOPED_PRIOR,
        applicability_key=card.applicability_key,
        supporting_evidence=(card.candidate_id, *card.evidence_refs),
        competing_mechanisms=(
            "per-site TMEM allocation/completion overhead",
            "grid-12 latency hiding",
            "TMEM readback and BF16 materialization",
            "SMEM descriptor feed and single-thread issue latency",
        ),
        challenges=tuple(challenges),
        invalidation_keys=card.reopen_condition,
    )
    claim_diagnostics = verify_knowledge_claim(claim)
    if claim_diagnostics:
        raise ValueError("; ".join(f"{d.code}: {d.message}" for d in claim_diagnostics))

    output = {
        "schema_version": "stage11_mma_memory_v1",
        "profile_id": profile["profile_id"],
        "decision": "stop_direct_swap_keep_cross_phase_reopenable",
        "candidate": {**asdict(direct), "candidate_id": direct_id},
        "failure_cards": [asdict(card)],
        "knowledge_claims": [asdict(claim)],
        "cross_path_summary": cross_path_summary,
        "v16_screen_summary": v16_screen_summary,
        "reopenable_hypothesis": {
            "tier": MigrationTier.PIPELINE_REMAP.value,
            "name": "tcgen05-cross-phase-tmem-residency",
            "state": ConclusionState.INCONCLUSIVE.value,
            "required_sites": ["k2_phase1", "k2_phase3", "k2_phase4", "k2_phase6_state_update"],
            "entry_gate": "preserve CHUNK16, token order, and archived BF16/FP16 rounding DAG",
            "qualification_gate": "public output+final-state correctness and paired lower confidence bound above the optimized mma.sync incumbent",
        },
        "evidence_digests_verified": expected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True, default=json_default) + "\n")


if __name__ == "__main__":
    main()
