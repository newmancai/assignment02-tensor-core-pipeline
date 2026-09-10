"""Typed contracts for architecture-specific MMA migration experiments.

This module complements the schedule IR.  An SM80 ``mma.sync`` to SM100
``tcgen05`` migration changes accumulator storage, issue/completion semantics,
and often physical tile extent; it is therefore not a PipelineKind toggle.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum


class MigrationTier(str, Enum):
    BASELINE_REPLAY = "baseline_replay"
    INSTRUCTION_ONLY = "instruction_only"
    PADDING_OR_PACKING = "padding_or_packing"
    PIPELINE_REMAP = "warp_role_or_pipeline_remap"
    CHUNK_RETILE = "chunk_retile_32_64"


class MmaFamily(str, Enum):
    SM80_MMA_SYNC = "sm80_mma_sync"
    SM100_TCGEN05 = "sm100_tcgen05"


class MemorySpace(str, Enum):
    REGISTER = "register"
    SHARED = "shared"
    TMEM = "tmem"


class LogicalPhysicalTransform(str, Enum):
    IDENTITY = "identity"
    TRANSPOSE_OUTPUT_SWAP_OPERANDS = "transpose_output_swap_operands"


class ConclusionState(str, Enum):
    SCHEMA_REJECTED = "schema_rejected"
    STATIC_REJECTED = "static_rejected"
    COMPILE_FAILED = "compile_failed"
    CORRECTNESS_FAILED = "correctness_failed"
    RESOURCE_FAILED = "resource_failed"
    SLOWER = "slower"
    QUALIFIED = "qualified"
    INCONCLUSIVE = "inconclusive"


class FailureClass(str, Enum):
    ISA_UNREPRESENTABLE = "isa_unrepresentable"
    COMPILE = "compile"
    NUMERICAL = "numerical"
    RESOURCE_CLIFF = "resource_cliff"
    LATENCY_REGRESSION = "latency_regression"
    NO_PRACTICAL_GAIN = "no_practical_gain"


@dataclass(frozen=True)
class MathSiteContract:
    site_id: str
    logical_mnk: tuple[int, int, int]
    input_dtype: str
    accumulator_dtype: str
    output_dtype: str
    operand_major: tuple[str, str]
    rounding_dag_sha256: str
    source_ref: str


@dataclass(frozen=True)
class InstructionSpec:
    site_id: str
    family: MmaFamily
    form_id: str
    instruction_mnk: tuple[int, int, int]
    accumulator_space: MemorySpace
    cta_group: int = 1


@dataclass(frozen=True)
class LogicalPhysicalTile:
    site_id: str
    logical_mnk: tuple[int, int, int]
    physical_mnk: tuple[int, int, int]
    zero_fills_padding: bool
    suppresses_padded_stores: bool
    transform: LogicalPhysicalTransform = LogicalPhysicalTransform.IDENTITY


@dataclass(frozen=True)
class Tcgen05Protocol:
    site_id: str
    issuer_threads: int
    alloc_owner: str
    dealloc_owner: str
    tmem_column_begin: int
    tmem_column_end: int
    alloc_dominates_issue: bool
    issue_has_commit: bool
    commit_has_wait: bool
    wait_dominates_readback: bool
    readback_dominates_dealloc: bool
    live_from: int = 0
    live_until: int = 0


@dataclass(frozen=True)
class ResourceContract:
    threads_per_cta: int
    registers_per_thread: int
    shared_bytes: int
    tmem_columns: int
    predicted_resident_ctas_per_sm: int


@dataclass(frozen=True)
class LaunchTopology:
    grid_ctas: int
    value_slices: int
    ownership_axis: str
    role_map: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class TmemCarrier:
    carrier_id: str
    dtype: str
    logical_layout: str
    column_begin: int
    column_end: int
    producer_site: str
    consumer_sites: tuple[str, ...]
    live_from: int
    live_until: int
    preserves_rounding_boundary: bool


@dataclass(frozen=True)
class MigrationCandidate:
    schema_version: str
    name: str
    tier: MigrationTier
    baseline_sha256: str
    capability_table_sha256: str
    target_arch: str
    chunk: int
    math_sites: tuple[MathSiteContract, ...]
    instructions: tuple[InstructionSpec, ...]
    tiles: tuple[LogicalPhysicalTile, ...]
    protocols: tuple[Tcgen05Protocol, ...]
    resources: ResourceContract
    preserves_token_order: bool
    preserves_rounding_dag: bool
    parent_ids: tuple[str, ...] = ()
    mutation_paths: tuple[str, ...] = ()
    launch_topology: LaunchTopology | None = None
    tmem_carriers: tuple[TmemCarrier, ...] = ()
    comparison_baseline_id: str = ""


@dataclass(frozen=True)
class MigrationDiagnostic:
    code: str
    message: str


@dataclass(frozen=True)
class MigrationFailureCard:
    technique_id: str
    candidate_id: str
    state: ConclusionState
    failure_class: FailureClass
    causal_status: str
    applicability_key: str
    exact_conditions: tuple[str, ...]
    observed_speedup: float
    evidence_refs: tuple[str, ...]
    evidence_sha256: tuple[str, ...]
    surviving_fact: str
    do_not_generalize_beyond: str
    reopen_condition: tuple[str, ...]


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _is_strong_id(value: str) -> bool:
    """Return whether *value* is a namespaced, full SHA-256 identifier."""

    prefix, separator, digest = value.partition(":")
    return separator == ":" and prefix in {"sha256", "mma256"} and _is_sha256(digest)


def _json_default(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"cannot serialize {type(value).__name__}")


def canonical_candidate_id(candidate: MigrationCandidate) -> str:
    payload = asdict(candidate)
    payload.pop("parent_ids", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default)
    return "mma256:" + hashlib.sha256(encoded.encode()).hexdigest()


def verify_mma_migration(candidate: MigrationCandidate) -> tuple[MigrationDiagnostic, ...]:
    """Fail closed on math, ISA, padding, TMEM lifecycle, and resources."""

    out: list[MigrationDiagnostic] = []
    if candidate.schema_version not in {"typed_mma_migration_v1", "typed_mma_migration_v2"}:
        out.append(MigrationDiagnostic("TCG001", "unsupported migration schema"))
    if candidate.target_arch not in {"sm_100a", "sm_103a"}:
        out.append(MigrationDiagnostic("TCG002", "tcgen05 target must be sm_100a or sm_103a"))
    for label, digest in (
        ("baseline", candidate.baseline_sha256),
        ("capability table", candidate.capability_table_sha256),
    ):
        if not _is_sha256(digest):
            out.append(MigrationDiagnostic("TCG003", f"{label} needs a full SHA-256 digest"))

    sites = {site.site_id: site for site in candidate.math_sites}
    if len(sites) != len(candidate.math_sites):
        out.append(MigrationDiagnostic("TCG004", "duplicate math site id"))
    specs = {spec.site_id: spec for spec in candidate.instructions}
    tiles = {tile.site_id: tile for tile in candidate.tiles}
    protocols = {protocol.site_id: protocol for protocol in candidate.protocols}
    if set(specs) != set(sites) or set(tiles) != set(sites):
        out.append(MigrationDiagnostic("TCG005", "every math site needs one instruction and tile"))

    for site in candidate.math_sites:
        if not _is_sha256(site.rounding_dag_sha256):
            out.append(MigrationDiagnostic("TCG006", f"{site.site_id} lacks a rounding-DAG digest"))
        tile = tiles.get(site.site_id)
        spec = specs.get(site.site_id)
        if tile and tile.logical_mnk != site.logical_mnk:
            out.append(MigrationDiagnostic("TCG101", f"{site.site_id} changes logical contraction"))
        if spec and spec.family is MmaFamily.SM100_TCGEN05:
            m, n, k = spec.instruction_mnk
            legal_m = {64, 128} if spec.cta_group == 1 else {128, 256}
            n_step = 8 if spec.cta_group == 1 else 16
            if m not in legal_m or not (n_step <= n <= 256 and n % n_step == 0) or k != 16:
                out.append(MigrationDiagnostic("TCG201", f"illegal tcgen05 tile for {site.site_id}"))
            if spec.accumulator_space is not MemorySpace.TMEM:
                out.append(MigrationDiagnostic("TCG202", f"tcgen05 accumulator for {site.site_id} must be TMEM"))
            protocol = protocols.get(site.site_id)
            if protocol is None:
                out.append(MigrationDiagnostic("TCG203", f"tcgen05 site {site.site_id} lacks a completion protocol"))
            elif not (
                protocol.issuer_threads == 1
                and protocol.alloc_dominates_issue
                and protocol.issue_has_commit
                and protocol.commit_has_wait
                and protocol.wait_dominates_readback
                and protocol.readback_dominates_dealloc
            ):
                out.append(MigrationDiagnostic("TCG204", f"incomplete tcgen05 lifecycle at {site.site_id}"))

        if tile:
            if tile.transform is LogicalPhysicalTransform.TRANSPOSE_OUTPUT_SWAP_OPERANDS:
                logical_m, logical_n, logical_k = tile.logical_mnk
                if tile.physical_mnk != (logical_n, logical_m, logical_k):
                    out.append(MigrationDiagnostic("TCG102", f"invalid transpose/swap equivalence at {site.site_id}"))
            elif tile.physical_mnk != tile.logical_mnk:
                if not tile.zero_fills_padding or not tile.suppresses_padded_stores:
                    out.append(MigrationDiagnostic("TCG205", f"padding proof incomplete at {site.site_id}"))

    tcgen_sites = {site_id for site_id, spec in specs.items() if spec.family is MmaFamily.SM100_TCGEN05}
    if set(protocols) != tcgen_sites:
        out.append(MigrationDiagnostic("TCG206", "protocol set must equal tcgen05 site set"))
    ranges = sorted(
        (p.tmem_column_begin, p.tmem_column_end, p.live_from, p.live_until, p.site_id)
        for p in candidate.protocols
    )
    for begin, end, live_from, live_until, site_id in ranges:
        if begin < 0 or end <= begin:
            out.append(MigrationDiagnostic("TCG207", f"invalid TMEM range at {site_id}"))
        if live_from > live_until:
            out.append(MigrationDiagnostic("TCG209", f"invalid TMEM lifetime at {site_id}"))
    for index, left in enumerate(ranges):
        for right in ranges[index + 1 :]:
            columns_overlap = left[0] < right[1] and right[0] < left[1]
            lifetimes_overlap = left[2] <= right[3] and right[2] <= left[3]
            if columns_overlap and lifetimes_overlap:
                out.append(MigrationDiagnostic("TCG208", f"overlapping live TMEM ranges: {left[4]}, {right[4]}"))

    carrier_ranges = []
    for carrier in candidate.tmem_carriers:
        if carrier.producer_site not in sites or any(x not in sites for x in carrier.consumer_sites):
            out.append(MigrationDiagnostic("TCG210", f"carrier {carrier.carrier_id} references an unknown site"))
        if carrier.column_begin < 0 or carrier.column_end <= carrier.column_begin or carrier.live_from > carrier.live_until:
            out.append(MigrationDiagnostic("TCG211", f"carrier {carrier.carrier_id} has an invalid range/lifetime"))
        if not carrier.preserves_rounding_boundary:
            out.append(MigrationDiagnostic("TCG212", f"carrier {carrier.carrier_id} crosses a frozen rounding boundary"))
        carrier_ranges.append((carrier.column_begin, carrier.column_end, carrier.live_from, carrier.live_until, carrier.carrier_id))
    for index, left in enumerate(carrier_ranges):
        for right in carrier_ranges[index + 1 :]:
            if left[0] < right[1] and right[0] < left[1] and left[2] <= right[3] and right[2] <= left[3]:
                out.append(MigrationDiagnostic("TCG213", f"overlapping live carriers: {left[4]}, {right[4]}"))

    if candidate.tier is MigrationTier.INSTRUCTION_ONLY:
        if candidate.chunk != 16 or not candidate.preserves_token_order or not candidate.preserves_rounding_dag:
            out.append(MigrationDiagnostic("TCG301", "instruction-only must preserve CHUNK16, token order, and rounding DAG"))
    if candidate.tier is MigrationTier.CHUNK_RETILE and candidate.chunk not in {32, 64}:
        out.append(MigrationDiagnostic("TCG302", "chunk-retile tier requires CHUNK32 or CHUNK64"))
    if candidate.resources.predicted_resident_ctas_per_sm < 1:
        out.append(MigrationDiagnostic("TCG401", "candidate predicts no resident CTA"))
    required_columns = max(
        [end for _, end, _, _, _ in ranges]
        + [carrier.column_end for carrier in candidate.tmem_carriers],
        default=0,
    )
    if candidate.resources.tmem_columns < required_columns:
        out.append(MigrationDiagnostic("TCG402", "TMEM resource contract does not cover allocated ranges"))
    if min(
        candidate.resources.threads_per_cta,
        candidate.resources.registers_per_thread,
        candidate.resources.shared_bytes,
        candidate.resources.tmem_columns,
    ) < 0:
        out.append(MigrationDiagnostic("TCG403", "resource quantities cannot be negative"))
    if candidate.schema_version == "typed_mma_migration_v2":
        if candidate.launch_topology is None or candidate.launch_topology.grid_ctas <= 0 or candidate.launch_topology.value_slices <= 0:
            out.append(MigrationDiagnostic("TCG601", "v2 candidates require an explicit launch topology"))
        if not _is_strong_id(candidate.comparison_baseline_id):
            out.append(MigrationDiagnostic("TCG602", "v2 candidates require a namespaced full-SHA comparison baseline id"))
    return tuple(out)


def verify_failure_card(card: MigrationFailureCard) -> tuple[MigrationDiagnostic, ...]:
    out: list[MigrationDiagnostic] = []
    if not card.candidate_id.startswith("mma256:") or len(card.candidate_id) != 71:
        out.append(MigrationDiagnostic("TCG501", "failure card needs a full canonical candidate id"))
    if card.state not in {ConclusionState.SLOWER, ConclusionState.CORRECTNESS_FAILED, ConclusionState.RESOURCE_FAILED, ConclusionState.COMPILE_FAILED}:
        out.append(MigrationDiagnostic("TCG502", "inconclusive/qualified results are not reusable failure cards"))
    if card.observed_speedup <= 0:
        out.append(MigrationDiagnostic("TCG503", "speedup must be positive"))
    if not card.exact_conditions or not card.reopen_condition:
        out.append(MigrationDiagnostic("TCG504", "failure scope and reopen conditions are mandatory"))
    if len(card.evidence_refs) != len(card.evidence_sha256) or any(not _is_sha256(x) for x in card.evidence_sha256):
        out.append(MigrationDiagnostic("TCG505", "every evidence reference needs a full SHA-256"))
    return tuple(out)
