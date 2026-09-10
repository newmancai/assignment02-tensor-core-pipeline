"""Evidence-first MMA migration assessment for the C1 main line.

The module deliberately separates three operations:

1. establish what instruction path and workload were measured;
2. diagnose the physical limiter and answer the migration question;
3. emit bounded optimization proposals for the existing agent loop.

It does not let a language-model proposal override a hardware receipt.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum


class MigrationAction(str, Enum):
    KEEP = "keep"
    MEASURE = "measure"
    STOP = "stop"


@dataclass(frozen=True)
class KernelMeasurement:
    label: str
    recurrence_ctas: int
    duration_us: float
    sm_throughput_percent: float
    dram_throughput_percent: float
    tensor_elapsed_percent: float
    evidence_ref: str

    def __post_init__(self) -> None:
        if self.recurrence_ctas <= 0 or self.duration_us <= 0:
            raise ValueError("CTA count and duration must be positive")
        for value in (
            self.sm_throughput_percent,
            self.dram_throughput_percent,
            self.tensor_elapsed_percent,
        ):
            if not 0 <= value <= 100:
                raise ValueError("throughput percentages must lie in [0, 100]")


@dataclass(frozen=True)
class MmaMigrationProfile:
    profile_id: str
    target_arch: str
    sm_count: int
    local_heads: int
    hmmas_in_recurrence_sass: int
    tcgen_in_recurrence_sass: int
    official: KernelMeasurement
    value_rows: int
    value_rows_independent: bool
    dominant_phase_m: int
    current_mma_atom_m: int
    tcgen_min_m: int
    phase6_m: int
    tcgen_l0_v128_speedup: float
    tcgen_l1_v128_speedup: float
    tcgen_l0_v16_speedup: float
    tcgen_l1_v16_speedup: float
    tcgen_l0_v128_mma_active_blocks_per_sm: int
    tcgen_l0_v128_active_blocks_per_sm: int
    tcgen_l1_v128_mma_active_blocks_per_sm: int
    tcgen_l1_v128_active_blocks_per_sm: int
    tcgen_l1_v128_mma_static_smem_bytes: int
    tcgen_l1_v128_static_smem_bytes: int
    tcgen_l1_v128_mma_registers: int
    tcgen_l1_v128_registers: int
    sass_evidence_ref: str
    sass_evidence_sha256: str
    semantic_evidence_ref: str
    semantic_evidence_sha256: str
    tcgen_evidence_ref: str
    tcgen_evidence_sha256: str
    ncu_evidence_sha256: str

    def __post_init__(self) -> None:
        positive_geometry = (
            self.sm_count,
            self.local_heads,
            self.value_rows,
            self.dominant_phase_m,
            self.current_mma_atom_m,
            self.tcgen_min_m,
            self.phase6_m,
            self.tcgen_l0_v128_mma_active_blocks_per_sm,
            self.tcgen_l0_v128_active_blocks_per_sm,
            self.tcgen_l1_v128_mma_active_blocks_per_sm,
            self.tcgen_l1_v128_active_blocks_per_sm,
        )
        if min(positive_geometry) <= 0:
            raise ValueError("physical geometry and residency values must be positive")
        if min(
            self.tcgen_l1_v128_mma_static_smem_bytes,
            self.tcgen_l1_v128_static_smem_bytes,
            self.tcgen_l1_v128_mma_registers,
            self.tcgen_l1_v128_registers,
        ) < 0:
            raise ValueError("resource values cannot be negative")
        if self.hmmas_in_recurrence_sass < 0 or self.tcgen_in_recurrence_sass < 0:
            raise ValueError("opcode counts cannot be negative")
        evidence_digests = (
            self.sass_evidence_sha256,
            self.semantic_evidence_sha256,
            self.tcgen_evidence_sha256,
            self.ncu_evidence_sha256,
        )
        if any(
            len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
            for value in evidence_digests
        ):
            raise ValueError("evidence SHA-256 digests must be lowercase hexadecimal")
        if min(
            self.tcgen_l0_v128_speedup,
            self.tcgen_l1_v128_speedup,
            self.tcgen_l0_v16_speedup,
            self.tcgen_l1_v16_speedup,
        ) <= 0:
            raise ValueError("probe speedups must be positive")


@dataclass(frozen=True)
class PhysicalFinding:
    code: str
    conclusion: str
    evidence: str
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True)
class MmaOptimizationProposal:
    rank: int
    proposal_id: str
    action: MigrationAction
    physical_target: str
    change: str
    gate: str
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True)
class MmaMigrationAssessment:
    profile_id: str
    answer: str
    findings: tuple[PhysicalFinding, ...]
    proposals: tuple[MmaOptimizationProposal, ...]
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def assess_mma_migration(profile: MmaMigrationProfile) -> MmaMigrationAssessment:
    """Answer C1 from measured physical variables, then rank MMA work.

    Thresholds are intentionally conservative and interpretable.  A low-utilization
    diagnosis requires both SM and DRAM throughput below 15%, not occupancy alone;
    grid underfill additionally requires fewer recurrence CTAs than physical SMs.
    A direct ISA swap is stopped only when both the optimistic V128 core probe and
    the integration-envelope probe are slower than the current MMA path.
    """

    if profile.target_arch not in {"sm_100a", "sm_103a"}:
        raise ValueError("this assessment is scoped to data-center Blackwell")

    instruction_path_confirmed = (
        profile.hmmas_in_recurrence_sass > 0
        and profile.tcgen_in_recurrence_sass == 0
    )
    underfilled = (
        profile.official.recurrence_ctas < profile.sm_count
        and profile.official.sm_throughput_percent < 15.0
        and profile.official.dram_throughput_percent < 15.0
    )
    independent_parallelism_exists = (
        profile.value_rows_independent and profile.value_rows > 1
    )
    direct_tcgen_failed = (
        profile.tcgen_l0_v128_speedup < 1.0
        and profile.tcgen_l1_v128_speedup < 1.0
    )
    tcgen_core_signal = (
        profile.tcgen_l0_v16_speedup > 1.0
        and profile.tcgen_l1_v16_speedup < 1.0
    )
    dominant_shape_favors_current_mma = (
        profile.dominant_phase_m == profile.current_mma_atom_m
        and profile.dominant_phase_m < profile.tcgen_min_m
        and profile.phase6_m >= profile.tcgen_min_m
    )
    tcgen_reduces_residency = (
        profile.tcgen_l0_v128_active_blocks_per_sm
        < profile.tcgen_l0_v128_mma_active_blocks_per_sm
        and profile.tcgen_l1_v128_active_blocks_per_sm
        < profile.tcgen_l1_v128_mma_active_blocks_per_sm
    )

    if not instruction_path_confirmed:
        raise ValueError("SASS receipt does not confirm an HMMA-only recurrence path")

    official_coverage = profile.official.recurrence_ctas / profile.sm_count

    findings = (
        PhysicalFinding(
            "MMA001",
            "The measured recurrence uses HMMA rather than tcgen05.",
            (
                f"SASS contains {profile.hmmas_in_recurrence_sass} recurrence HMMA "
                f"instructions and {profile.tcgen_in_recurrence_sass} TCGEN/UTCMMA."
            ),
            (profile.sass_evidence_ref,),
        ),
        PhysicalFinding(
            "MMA002",
            "The first-order limiter is grid/recurrence latency, not peak Tensor Core or HBM throughput.",
            (
                f"Official recurrence launches {profile.official.recurrence_ctas}/{profile.sm_count} "
                f"CTAs/SMs ({official_coverage:.1%} coverage) at "
                f"{profile.official.sm_throughput_percent:.2f}% SM and "
                f"{profile.official.dram_throughput_percent:.2f}% DRAM throughput."
            ),
            (profile.official.evidence_ref,),
        ),
        PhysicalFinding(
            "MMA003",
            "The Value dimension exposes legal parallel work without breaking chunk recurrence.",
            (
                f"The state has {profile.value_rows} independent Value rows; partitioning rows "
                "does not change the reduction order within an output element."
            ),
            (profile.semantic_evidence_ref,),
        ),
        PhysicalFinding(
            "MMA004",
            "A Phase-6-only tcgen05 replacement does not beat the present V128 dataflow.",
            (
                f"V128 MMA/tcgen speedup is {profile.tcgen_l0_v128_speedup:.3f}x in L0 "
                f"and {profile.tcgen_l1_v128_speedup:.3f}x in L1."
            ),
            (profile.tcgen_evidence_ref,),
        ),
        PhysicalFinding(
            "MMA005",
            "The remaining tcgen05 opportunity is dataflow integration, not a bare instruction swap.",
            (
                f"V16 core-only L0 reaches {profile.tcgen_l0_v16_speedup:.3f}x, but the "
                f"integration envelope falls to {profile.tcgen_l1_v16_speedup:.3f}x."
            ),
            (profile.tcgen_evidence_ref,),
        ),
        PhysicalFinding(
            "MMA006",
            "The dominant M=16 phases match the current atom; only the M=128 state update is a natural tcgen05 site.",
            (
                f"Dominant phase M={profile.dominant_phase_m}, current MMA atom M="
                f"{profile.current_mma_atom_m}, tcgen05 minimum M={profile.tcgen_min_m}, "
                f"and Phase-6 M={profile.phase6_m}."
            ),
            (profile.semantic_evidence_ref,),
        ),
        PhysicalFinding(
            "MMA007",
            "The measured tcgen05 path sharply reduces active-block residency.",
            (
                "For V128, active blocks/SM change from "
                f"{profile.tcgen_l0_v128_mma_active_blocks_per_sm} to "
                f"{profile.tcgen_l0_v128_active_blocks_per_sm} in L0 and from "
                f"{profile.tcgen_l1_v128_mma_active_blocks_per_sm} to "
                f"{profile.tcgen_l1_v128_active_blocks_per_sm} in L1; L1 static SMEM is "
                f"{profile.tcgen_l1_v128_mma_static_smem_bytes} versus "
                f"{profile.tcgen_l1_v128_static_smem_bytes} bytes and registers are "
                f"{profile.tcgen_l1_v128_mma_registers} versus "
                f"{profile.tcgen_l1_v128_registers}."
            ),
            (profile.tcgen_evidence_ref,),
        ),
    )

    if not (
        underfilled
        and independent_parallelism_exists
        and direct_tcgen_failed
        and dominant_shape_favors_current_mma
        and tcgen_reduces_residency
    ):
        answer = (
            "Evidence is insufficient for a migration decision; preserve the current "
            "MMA path and collect the missing grid, throughput, or integrated tcgen05 receipt."
        )
    else:
        answer = (
            "Do not mechanically migrate the whole FlashKDA recurrence from mma.sync to "
            "tcgen05. Keep the verified m16n8k16 path, first optimize its exposed parallelism "
            "and issue pipeline, and reopen tcgen05 only for a cross-phase TMEM-resident dataflow."
        )

    proposals = (
        MmaOptimizationProposal(
            1,
            "mma-existing-path-parallelism",
            MigrationAction.KEEP,
            "grid underfill",
            "Retain mma.sync and expose independent Value rows as additional CTAs.",
            "Output and final state remain bitwise equal; select by workload/device guard.",
            (profile.official.evidence_ref, profile.semantic_evidence_ref),
        ),
        MmaOptimizationProposal(
            2,
            "mma-existing-path-issue-overlap",
            MigrationAction.MEASURE,
            "TMA/MMA load-use latency inside each CTA",
            "Search typed Phase-6 and Phase-1 prefetch distances without changing arithmetic order.",
            "Require scope-identical paired timing, correctness, and a positive confidence lower bound.",
            (profile.official.evidence_ref,),
        ),
        MmaOptimizationProposal(
            3,
            "tcgen05-cross-phase-residency",
            MigrationAction.MEASURE if tcgen_core_signal else MigrationAction.STOP,
            "TMEM allocation, reformat, readback, and synchronization overhead",
            "Keep operands or accumulators TMEM-resident across multiple compatible phases.",
            "Reopen only if an integrated K2 slice beats the optimized mma.sync incumbent and reports its compiled residency.",
            (profile.tcgen_evidence_ref,),
        ),
        MmaOptimizationProposal(
            4,
            "tcgen05-phase6-direct-swap",
            MigrationAction.STOP if direct_tcgen_failed else MigrationAction.MEASURE,
            "instruction generation alone",
            "Replace only the Phase-6 MMA while preserving the current surrounding dataflow.",
            "Stopped when the optimistic V128 core probe is already below 1.0x.",
            (profile.tcgen_evidence_ref,),
        ),
    )

    return MmaMigrationAssessment(
        profile_id=profile.profile_id,
        answer=answer,
        findings=findings,
        proposals=proposals,
        claim_boundary=(
            "Decision applies to the measured single-B300/SM103a Kimi-K3 H12 prefill "
            "recurrence and the archived Phase-6 probes; decode, other GPUs, and an "
            "unimplemented cross-phase TMEM design remain out of scope."
        ),
    )
