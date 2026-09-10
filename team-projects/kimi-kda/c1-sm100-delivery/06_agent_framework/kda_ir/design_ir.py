"""Typed design space for cross-route FlashKDA optimization.

The original :mod:`kda_ir.model` is deliberately a small executable schedule
grammar.  This module sits one level above it: it represents the architectural
choices that differ between the HMMA and tcgen05 lanes and a small dataflow IR
for proving when layout materializations can be removed.

The types are intentionally finite.  They cover the mechanisms exercised by
the C1 evidence rather than pretending that arbitrary CUDA source has already
been lowered into an IR.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from hashlib import sha256
import json
from typing import Mapping, Union

from .verify import Diagnostic


class MmaRoute(str, Enum):
    HMMA = "hmma"
    TCGEN05 = "tcgen05"


class SchedulerKind(str, Enum):
    STATIC = "static"
    STATIC_PERSISTENT = "static_persistent"
    CLC_DYNAMIC_PERSISTENT = "clc_dynamic_persistent"


class PipelineKind(str, Enum):
    SYNC = "sync"
    TMA_UMMA = "tma_umma"
    UMMA_ASYNC = "umma_async"
    UMMA_UMMA = "umma_umma"


class Carrier(str, Enum):
    REGISTER = "register"
    SHARED = "shared"
    TMEM = "tmem"


class PhysicalLayout(str, Enum):
    ROW_MAJOR = "row_major"
    COL_MAJOR = "col_major"
    HMMA_A_FRAGMENT = "hmma_a_fragment"
    HMMA_B_FRAGMENT = "hmma_b_fragment"
    TCGEN05_32B_SWIZZLED_MK = "tcgen05_32b_swizzled_mk"
    TCGEN05_32B_SWIZZLED_NK = "tcgen05_32b_swizzled_nk"
    TMEM_ACCUMULATOR = "tmem_accumulator"
    TMEM_OPERAND_A = "tmem_operand_a"


class TmemLifetime(str, Enum):
    NONE = "none"
    PER_MMA = "per_mma"
    CROSS_PHASE = "cross_phase"


class MaterializationPlacement(str, Enum):
    NONE = "none"
    ONCE_PER_CALL = "once_per_call"
    ONCE_PER_CTA = "once_per_cta"
    PER_RECURRENCE_STEP = "per_recurrence_step"


@dataclass(frozen=True)
class KernelDesign:
    """Finite architectural axes shared by the two optimization lanes."""

    design_id: str
    route: MmaRoute
    value_slice: int
    compute_warps: int
    accumulator_carrier: Carrier
    operand_b_layout: PhysicalLayout
    materialization: MaterializationPlacement
    scheduler: SchedulerKind
    pipeline: PipelineKind
    tmem_lifetime: TmemLifetime
    producer_emits_consumer_layout: bool
    preserves_bf16_rounding: bool = True
    chunk: int = 16

    def canonical_id(self) -> str:
        payload = {
            "route": self.route.value,
            "value_slice": self.value_slice,
            "compute_warps": self.compute_warps,
            "accumulator_carrier": self.accumulator_carrier.value,
            "operand_b_layout": self.operand_b_layout.value,
            "materialization": self.materialization.value,
            "scheduler": self.scheduler.value,
            "pipeline": self.pipeline.value,
            "tmem_lifetime": self.tmem_lifetime.value,
            "producer_emits_consumer_layout": self.producer_emits_consumer_layout,
            "preserves_bf16_rounding": self.preserves_bf16_rounding,
            "chunk": self.chunk,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return "design-ir:" + sha256(encoded.encode()).hexdigest()


def verify_kernel_design(design: KernelDesign) -> tuple[Diagnostic, ...]:
    """Reject physically impossible or causally ambiguous combinations."""

    diagnostics: list[Diagnostic] = []
    if design.chunk != 16:
        diagnostics.append(Diagnostic("KIR1201", "current KDA contract requires CHUNK=16"))
    if (
        design.value_slice not in (16, 32, 64, 128)
        or 128 % design.value_slice
    ):
        diagnostics.append(
            Diagnostic("KIR1202", "value slice must be one of 16, 32, 64, 128")
        )
    column_blocks = design.value_slice // 16
    if design.compute_warps not in (1, 2, 4):
        diagnostics.append(Diagnostic("KIR1203", "compute warps must be 1, 2, or 4"))
    elif column_blocks <= 0 or column_blocks % design.compute_warps:
        diagnostics.append(
            Diagnostic("KIR1204", "16-column blocks must divide evenly across compute warps")
        )
    elif column_blocks // design.compute_warps > 2:
        diagnostics.append(
            Diagnostic("KIR1205", "this lowering supports at most two column blocks per warp")
        )

    if not design.preserves_bf16_rounding:
        diagnostics.append(
            Diagnostic("KIR1206", "the official BF16 rounding boundary must be preserved")
        )

    if design.producer_emits_consumer_layout and design.materialization is not MaterializationPlacement.NONE:
        diagnostics.append(
            Diagnostic("KIR1207", "producer-ready layout cannot also carry a materialization")
        )

    if design.route is MmaRoute.HMMA:
        if design.accumulator_carrier is not Carrier.REGISTER:
            diagnostics.append(Diagnostic("KIR1208", "HMMA accumulator must remain register-resident"))
        if design.operand_b_layout is not PhysicalLayout.HMMA_B_FRAGMENT:
            diagnostics.append(Diagnostic("KIR1209", "HMMA requires its register B-fragment layout"))
        if design.pipeline is not PipelineKind.SYNC:
            diagnostics.append(Diagnostic("KIR1210", "HMMA lane does not use a UMMA pipeline"))
        if design.tmem_lifetime is not TmemLifetime.NONE:
            diagnostics.append(Diagnostic("KIR1211", "HMMA cannot claim a TMEM lifetime"))
    elif design.route is MmaRoute.TCGEN05:
        if design.accumulator_carrier is not Carrier.TMEM:
            diagnostics.append(Diagnostic("KIR1212", "tcgen05 accumulation requires TMEM"))
        if design.operand_b_layout is not PhysicalLayout.TCGEN05_32B_SWIZZLED_NK:
            diagnostics.append(
                Diagnostic("KIR1213", "tcgen05 BF16 K16 requires the typed swizzled B descriptor")
            )
        if design.pipeline is PipelineKind.SYNC:
            diagnostics.append(Diagnostic("KIR1214", "tcgen05 requires an asynchronous UMMA pipeline"))
        if design.tmem_lifetime is TmemLifetime.NONE:
            diagnostics.append(Diagnostic("KIR1215", "tcgen05 requires an explicit TMEM lifecycle"))

    if (
        design.scheduler is SchedulerKind.CLC_DYNAMIC_PERSISTENT
        and design.pipeline not in (PipelineKind.TMA_UMMA, PipelineKind.UMMA_ASYNC)
    ):
        diagnostics.append(
            Diagnostic("KIR1216", "CLC work fetch requires a producer-capable asynchronous pipeline")
        )
    if design.tmem_lifetime is TmemLifetime.CROSS_PHASE and not design.producer_emits_consumer_layout:
        diagnostics.append(
            Diagnostic("KIR1217", "cross-phase TMEM residency requires a producer-ready layout contract")
        )
    return tuple(diagnostics)


class ScalarType(str, Enum):
    BF16 = "bf16"
    FP32 = "fp32"


class TransformKind(str, Enum):
    COPY = "copy"
    RELAYOUT = "relayout"
    ROUND = "round"


@dataclass(frozen=True)
class ValueType:
    shape: tuple[int, ...]
    dtype: ScalarType
    layout: PhysicalLayout
    carrier: Carrier

    @property
    def bytes(self) -> int:
        elements = 1
        for extent in self.shape:
            elements *= extent
        return elements * (2 if self.dtype is ScalarType.BF16 else 4)


@dataclass(frozen=True)
class Value:
    name: str
    type: ValueType
    external: bool = False
    output: bool = False


@dataclass(frozen=True)
class TransformOp:
    op_id: str
    source: str
    result: str
    kind: TransformKind
    placement: MaterializationPlacement
    semantic_boundary: bool = False


@dataclass(frozen=True)
class MatmulOp:
    op_id: str
    lhs: str
    rhs: str
    result: str
    route: MmaRoute
    m: int
    n: int
    k: int


DataflowOp = Union[TransformOp, MatmulOp]


@dataclass(frozen=True)
class DataflowProgram:
    program_id: str
    values: tuple[Value, ...]
    operations: tuple[DataflowOp, ...]
    recurrence_steps: int = 1


def verify_dataflow(program: DataflowProgram) -> tuple[Diagnostic, ...]:
    """Check SSA identity and the operand contracts of the two MMA routes."""

    diagnostics: list[Diagnostic] = []
    values = {value.name: value for value in program.values}
    if len(values) != len(program.values):
        diagnostics.append(Diagnostic("KIR1220", "value names must be unique"))
    op_ids = [operation.op_id for operation in program.operations]
    if len(set(op_ids)) != len(op_ids):
        diagnostics.append(Diagnostic("KIR1221", "operation IDs must be unique"))
    if program.recurrence_steps <= 0:
        diagnostics.append(Diagnostic("KIR1222", "recurrence steps must be positive"))

    defined = {value.name for value in program.values if value.external}
    for operation in program.operations:
        inputs = (
            (operation.source,)
            if isinstance(operation, TransformOp)
            else (operation.lhs, operation.rhs)
        )
        result = operation.result
        for name in (*inputs, result):
            if name not in values:
                diagnostics.append(Diagnostic("KIR1223", f"operation references unknown value {name}"))
        for name in inputs:
            if name in values and name not in defined:
                diagnostics.append(Diagnostic("KIR1224", f"value {name} is used before definition"))
        if result in defined:
            diagnostics.append(Diagnostic("KIR1225", f"value {result} has multiple definitions"))
        defined.add(result)
        if any(name not in values for name in (*inputs, result)):
            continue

        if isinstance(operation, TransformOp):
            source = values[operation.source].type
            target = values[operation.result].type
            if source.shape != target.shape:
                diagnostics.append(Diagnostic("KIR1226", "materialization cannot change logical shape"))
            if operation.kind is TransformKind.COPY and (
                source.dtype != target.dtype or source.layout != target.layout
            ):
                diagnostics.append(Diagnostic("KIR1227", "copy must preserve dtype and layout"))
            if operation.kind is TransformKind.RELAYOUT and source.dtype != target.dtype:
                diagnostics.append(Diagnostic("KIR1228", "relayout cannot change dtype"))
            if operation.kind is TransformKind.ROUND:
                if not (
                    source.dtype is ScalarType.FP32
                    and target.dtype is ScalarType.BF16
                    and operation.semantic_boundary
                ):
                    diagnostics.append(
                        Diagnostic("KIR1229", "FP32-to-BF16 round must be an explicit semantic boundary")
                    )
            elif operation.semantic_boundary:
                diagnostics.append(
                    Diagnostic("KIR1230", "only an explicit round may be a semantic boundary")
                )
            continue

        lhs = values[operation.lhs].type
        rhs = values[operation.rhs].type
        result_type = values[operation.result].type
        if operation.k != 16 or operation.m % 16 or operation.n % 8:
            diagnostics.append(Diagnostic("KIR1231", "current BF16 MMA tiles require M16, N8, K16 multiples"))
        if operation.route is MmaRoute.HMMA:
            if lhs.layout is not PhysicalLayout.HMMA_A_FRAGMENT or lhs.carrier is not Carrier.REGISTER:
                diagnostics.append(Diagnostic("KIR1238", "HMMA lhs must be a register A fragment"))
            if rhs.layout is not PhysicalLayout.HMMA_B_FRAGMENT or rhs.carrier is not Carrier.REGISTER:
                diagnostics.append(Diagnostic("KIR1232", "HMMA rhs must be a register B fragment"))
            if result_type.carrier is not Carrier.REGISTER:
                diagnostics.append(Diagnostic("KIR1233", "HMMA result must be register-resident"))
        else:
            legal_shared_a = (
                lhs.layout is PhysicalLayout.TCGEN05_32B_SWIZZLED_MK
                and lhs.carrier is Carrier.SHARED
            )
            legal_tmem_a = (
                lhs.layout is PhysicalLayout.TMEM_OPERAND_A
                and lhs.carrier is Carrier.TMEM
            )
            if not (legal_shared_a or legal_tmem_a):
                diagnostics.append(
                    Diagnostic(
                        "KIR1239",
                        "tcgen05 lhs must be a typed shared-MK or A-TMEM operand; "
                        "an accumulator fragment is not an A-layout identity",
                    )
                )
            if rhs.layout is not PhysicalLayout.TCGEN05_32B_SWIZZLED_NK or rhs.carrier is not Carrier.SHARED:
                diagnostics.append(Diagnostic("KIR1234", "tcgen05 rhs must be shared 32B-swizzled NK"))
            if result_type.layout is not PhysicalLayout.TMEM_ACCUMULATOR or result_type.carrier is not Carrier.TMEM:
                diagnostics.append(Diagnostic("KIR1235", "tcgen05 result must use the TMEM accumulator layout"))
        if lhs.dtype is not ScalarType.BF16 or rhs.dtype is not ScalarType.BF16:
            diagnostics.append(Diagnostic("KIR1236", "the current MMA contract consumes BF16 operands"))
        if lhs.shape != (operation.m, operation.k):
            diagnostics.append(Diagnostic("KIR1240", "MMA lhs shape must equal (M,K)"))
        if rhs.shape != (operation.k, operation.n):
            diagnostics.append(Diagnostic("KIR1241", "MMA rhs shape must equal (K,N)"))
        if result_type.shape != (operation.m, operation.n):
            diagnostics.append(Diagnostic("KIR1242", "MMA result shape must equal (M,N)"))
        if result_type.dtype is not ScalarType.FP32:
            diagnostics.append(Diagnostic("KIR1243", "the current MMA contract accumulates FP32"))
    for value in program.values:
        if value.output and value.name not in defined:
            diagnostics.append(Diagnostic("KIR1237", f"output {value.name} is never defined"))
    return tuple(diagnostics)


@dataclass(frozen=True)
class CanonicalizationAction:
    rule: str
    removed_ops: tuple[str, ...]
    detail: str


@dataclass(frozen=True)
class CanonicalizationResult:
    program: DataflowProgram
    actions: tuple[CanonicalizationAction, ...]


def _replace_inputs(operation: DataflowOp, aliases: Mapping[str, str]) -> DataflowOp:
    def resolve(name: str) -> str:
        seen = set()
        while name in aliases and name not in seen:
            seen.add(name)
            name = aliases[name]
        return name

    if isinstance(operation, TransformOp):
        return replace(operation, source=resolve(operation.source))
    return replace(operation, lhs=resolve(operation.lhs), rhs=resolve(operation.rhs))


def canonicalize_dataflow(program: DataflowProgram) -> CanonicalizationResult:
    """Reach a small layout normal form without crossing rounding boundaries.

    Rules are deterministic and conservative: remove physical identities,
    common identical materializations, cancel exact relayout round trips, and
    fuse a single-use copy/relayout chain into one direct materialization.
    """

    diagnostics = verify_dataflow(program)
    if diagnostics:
        raise ValueError("\n".join(str(item) for item in diagnostics))

    value_map = {value.name: value for value in program.values}
    operations: list[DataflowOp] = list(program.operations)
    actions: list[CanonicalizationAction] = []
    changed = True
    while changed:
        changed = False
        uses: dict[str, int] = {}
        for operation in operations:
            names = (
                (operation.source,)
                if isinstance(operation, TransformOp)
                else (operation.lhs, operation.rhs)
            )
            for name in names:
                uses[name] = uses.get(name, 0) + 1

        aliases: dict[str, str] = {}
        kept: list[DataflowOp] = []
        cse: dict[tuple[object, ...], str] = {}
        index = 0
        while index < len(operations):
            operation = _replace_inputs(operations[index], aliases)
            if not isinstance(operation, TransformOp):
                kept.append(operation)
                index += 1
                continue
            source_type = value_map[operation.source].type
            result_type = value_map[operation.result].type

            if operation.kind is not TransformKind.ROUND and source_type == result_type:
                aliases[operation.result] = operation.source
                actions.append(
                    CanonicalizationAction(
                        "identity_materialization",
                        (operation.op_id,),
                        f"alias {operation.result} to {operation.source}",
                    )
                )
                changed = True
                index += 1
                continue

            key = (
                operation.source,
                result_type,
                operation.kind,
                operation.placement,
                operation.semantic_boundary,
            )
            if not operation.semantic_boundary and key in cse:
                aliases[operation.result] = cse[key]
                actions.append(
                    CanonicalizationAction(
                        "common_materialization",
                        (operation.op_id,),
                        f"reuse {cse[key]} for {operation.result}",
                    )
                )
                changed = True
                index += 1
                continue
            cse[key] = operation.result

            if index + 1 < len(operations):
                next_op = _replace_inputs(operations[index + 1], aliases)
                if (
                    isinstance(next_op, TransformOp)
                    and next_op.source == operation.result
                    and uses.get(operation.result, 0) == 1
                    and not operation.semantic_boundary
                    and not next_op.semantic_boundary
                    and operation.kind in (TransformKind.COPY, TransformKind.RELAYOUT)
                    and next_op.kind in (TransformKind.COPY, TransformKind.RELAYOUT)
                    and operation.placement is next_op.placement
                ):
                    final_type = value_map[next_op.result].type
                    if source_type == final_type:
                        aliases[next_op.result] = operation.source
                        actions.append(
                            CanonicalizationAction(
                                "cancel_layout_round_trip",
                                (operation.op_id, next_op.op_id),
                                f"alias {next_op.result} to {operation.source}",
                            )
                        )
                    else:
                        kept.append(
                            TransformOp(
                                op_id=f"canonical:{operation.op_id}+{next_op.op_id}",
                                source=operation.source,
                                result=next_op.result,
                                kind=(
                                    TransformKind.COPY
                                    if source_type.layout == final_type.layout
                                    else TransformKind.RELAYOUT
                                ),
                                placement=operation.placement,
                            )
                        )
                        actions.append(
                            CanonicalizationAction(
                                "fuse_materialization_chain",
                                (operation.op_id, next_op.op_id),
                                f"materialize {operation.source} directly as {next_op.result}",
                            )
                        )
                    changed = True
                    index += 2
                    continue
            kept.append(operation)
            index += 1

        operations = [_replace_inputs(operation, aliases) for operation in kept]

    referenced = {
        name
        for operation in operations
        for name in (
            (operation.source, operation.result)
            if isinstance(operation, TransformOp)
            else (operation.lhs, operation.rhs, operation.result)
        )
    }
    values = tuple(
        value
        for value in program.values
        if value.external or value.output or value.name in referenced
    )
    normalized = replace(program, values=values, operations=tuple(operations))
    diagnostics = verify_dataflow(normalized)
    if diagnostics:
        raise AssertionError("canonicalization produced invalid IR: " + "; ".join(map(str, diagnostics)))
    return CanonicalizationResult(normalized, tuple(actions))


@dataclass(frozen=True)
class MaterializationMetrics:
    operation_count: int
    hot_relayout_count: int
    bytes_per_cta: int


def materialization_metrics(program: DataflowProgram) -> MaterializationMetrics:
    values = {value.name: value for value in program.values}
    count = 0
    hot = 0
    total_bytes = 0
    for operation in program.operations:
        if not isinstance(operation, TransformOp):
            continue
        count += 1
        repetitions = (
            program.recurrence_steps
            if operation.placement is MaterializationPlacement.PER_RECURRENCE_STEP
            else 1
        )
        total_bytes += values[operation.result].type.bytes * repetitions
        if (
            operation.kind is TransformKind.RELAYOUT
            and operation.placement is MaterializationPlacement.PER_RECURRENCE_STEP
        ):
            hot += 1
    return MaterializationMetrics(count, hot, total_bytes)


def paired_warp_hmma_design(*, value_slice: int, compact: bool) -> KernelDesign:
    """Construct the measured HMMA baseline or two-column-block-per-warp child."""

    column_blocks = value_slice // 16
    compute_warps = max(1, column_blocks // 2) if compact else min(column_blocks, 4)
    return KernelDesign(
        design_id=f"hmma-v{value_slice}-{'paired' if compact else 'baseline'}",
        route=MmaRoute.HMMA,
        value_slice=value_slice,
        compute_warps=compute_warps,
        accumulator_carrier=Carrier.REGISTER,
        operand_b_layout=PhysicalLayout.HMMA_B_FRAGMENT,
        materialization=MaterializationPlacement.NONE,
        scheduler=SchedulerKind.STATIC,
        pipeline=PipelineKind.SYNC,
        tmem_lifetime=TmemLifetime.NONE,
        producer_emits_consumer_layout=True,
    )


def tcgen_phase6_bridge_program(*, recurrence_steps: int = 1) -> DataflowProgram:
    """Model the current register -> logical SMEM -> preferred SMEM bridge."""

    bf16 = ScalarType.BF16
    values = (
        Value(
            "a",
            ValueType(
                (128, 16),
                bf16,
                PhysicalLayout.TCGEN05_32B_SWIZZLED_MK,
                Carrier.SHARED,
            ),
            external=True,
        ),
        Value("u_reg", ValueType((16, 16), bf16, PhysicalLayout.HMMA_B_FRAGMENT, Carrier.REGISTER), external=True),
        Value("u_logical", ValueType((16, 16), bf16, PhysicalLayout.ROW_MAJOR, Carrier.SHARED)),
        Value("u_preferred", ValueType((16, 16), bf16, PhysicalLayout.TCGEN05_32B_SWIZZLED_NK, Carrier.SHARED)),
        Value("delta", ValueType((128, 16), ScalarType.FP32, PhysicalLayout.TMEM_ACCUMULATOR, Carrier.TMEM), output=True),
    )
    operations: tuple[DataflowOp, ...] = (
        TransformOp(
            "spill_u_logical",
            "u_reg",
            "u_logical",
            TransformKind.RELAYOUT,
            MaterializationPlacement.PER_RECURRENCE_STEP,
        ),
        TransformOp(
            "reorder_u_preferred",
            "u_logical",
            "u_preferred",
            TransformKind.RELAYOUT,
            MaterializationPlacement.PER_RECURRENCE_STEP,
        ),
        MatmulOp("phase6_tcgen05", "a", "u_preferred", "delta", MmaRoute.TCGEN05, 128, 16, 16),
    )
    return DataflowProgram("tcgen-phase6-bridge", values, operations, recurrence_steps)
