"""Auditable Stage-11 fixtures for compound SM100 MMA migration candidates."""

from __future__ import annotations

import hashlib

from .mma_migration import (
    InstructionSpec,
    LaunchTopology,
    LogicalPhysicalTile,
    LogicalPhysicalTransform,
    MathSiteContract,
    MemorySpace,
    MmaFamily,
    MigrationCandidate,
    MigrationTier,
    ResourceContract,
    Tcgen05Protocol,
    TmemCarrier,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def p34_shared_lifecycle_candidate() -> MigrationCandidate:
    """Return the CHUNK16 Phase-3/4 TMEM-lifecycle experiment.

    Phase 3 and Phase 4 deliberately reuse accumulator columns ``[0, 16)``.
    Their event intervals do not overlap, so this is one serialized allocation
    plan rather than two simultaneously-live allocations.  The rounded BF16 U
    value crossing the phase boundary has its own explicit carrier in columns
    ``[16, 32)``; this keeps the source rounding boundary visible in the IR.
    """

    phase3 = MathSiteContract(
        site_id="k2_phase3_inv_u_transposed",
        logical_mnk=(16, 128, 16),
        input_dtype="bf16",
        accumulator_dtype="f32",
        output_dtype="bf16",
        operand_major=("k", "k"),
        rounding_dag_sha256=_digest("flashkda-phase3-inv-u-bf16-rounding-boundary"),
        source_ref="FlashKDA/csrc/smxx/fwd_kernel2.cuh:620",
    )
    phase4 = MathSiteContract(
        site_id="k2_phase4_mqk_u_transposed",
        logical_mnk=(16, 128, 16),
        input_dtype="bf16",
        accumulator_dtype="f32",
        output_dtype="bf16",
        operand_major=("k", "k"),
        rounding_dag_sha256=_digest("flashkda-phase4-mqk-u-bf16-rounding-boundary"),
        source_ref="FlashKDA/csrc/smxx/fwd_kernel2.cuh:645",
    )
    sites = (phase3, phase4)
    form = "dense.bf16.f32.cta1.m128n16k16"
    instructions = tuple(
        InstructionSpec(site.site_id, MmaFamily.SM100_TCGEN05, form, (128, 16, 16), MemorySpace.TMEM)
        for site in sites
    )
    tiles = tuple(
        LogicalPhysicalTile(
            site.site_id,
            site.logical_mnk,
            (128, 16, 16),
            False,
            False,
            LogicalPhysicalTransform.TRANSPOSE_OUTPUT_SWAP_OPERANDS,
        )
        for site in sites
    )
    protocols = (
        Tcgen05Protocol(phase3.site_id, 1, "umma_issue_warp", "umma_issue_warp", 0, 16, True, True, True, True, True, 0, 1),
        Tcgen05Protocol(phase4.site_id, 1, "umma_issue_warp", "umma_issue_warp", 0, 16, True, True, True, True, True, 2, 3),
    )
    rounded_u = TmemCarrier(
        carrier_id="phase3_rounded_u_bf16",
        dtype="bf16",
        logical_layout="transposed_m128n16_chunk16",
        column_begin=16,
        column_end=32,
        producer_site=phase3.site_id,
        consumer_sites=(phase4.site_id,),
        live_from=1,
        live_until=2,
        preserves_rounding_boundary=True,
    )
    # Real archived receipts, rather than hashes of descriptive labels.
    baseline_digest = "f8c8471d660821234c0a16c1cc62e5c4472c03efab00792318d1b3741efa4eed"
    capability_digest = "c123aa09d551f635629a6219617991410b371ef6f2802ee4b63e2ac11bcbc529"
    return MigrationCandidate(
        schema_version="typed_mma_migration_v2",
        name="phase3-phase4-shared-tmem-lifecycle",
        tier=MigrationTier.PIPELINE_REMAP,
        baseline_sha256=baseline_digest,
        capability_table_sha256=capability_digest,
        target_arch="sm_103a",
        chunk=16,
        math_sites=sites,
        instructions=instructions,
        tiles=tiles,
        protocols=protocols,
        resources=ResourceContract(192, 40, 98304, 32, 1),
        preserves_token_order=True,
        preserves_rounding_dag=True,
        mutation_paths=(
            "instructions.k2_phase3_inv_u_transposed",
            "instructions.k2_phase4_mqk_u_transposed",
            "protocols",
            "tmem_carriers.phase3_rounded_u_bf16",
            "launch_topology.role_map",
        ),
        launch_topology=LaunchTopology(
            grid_ctas=12,
            value_slices=1,
            ownership_axis="sequence_head",
            role_map=(("load", 1), ("umma_issue", 1), ("compute", 3), ("store", 1)),
        ),
        tmem_carriers=(rounded_u,),
        comparison_baseline_id="sha256:" + baseline_digest,
    )
