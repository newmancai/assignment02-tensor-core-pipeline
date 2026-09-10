"""Evidence-aware portfolio composition over :mod:`kda_ir.design_ir`.

Unlike the Stage-9 schedule coordinator, this layer does not merge only
identical candidates.  Strategy roles contribute non-conflicting typed axis
updates; the coordinator materializes compound designs and retains one correct
measured elite per architectural niche.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from itertools import combinations
from typing import Iterable

from .design_ir import KernelDesign, verify_kernel_design
from .verify import Diagnostic


class PortfolioRole(str, Enum):
    LAYOUT = "layout"
    RESIDENCY = "residency"
    SCHEDULER = "scheduler"
    PARALLELISM = "parallelism"
    NUMERICS = "numerics"


@dataclass(frozen=True)
class AxisUpdate:
    field: str
    value: object


@dataclass(frozen=True)
class DesignPatch:
    patch_id: str
    role: PortfolioRole
    updates: tuple[AxisUpdate, ...]
    claim: str
    evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class ComposedDesign:
    design: KernelDesign
    patch_ids: tuple[str, ...]
    roles: tuple[PortfolioRole, ...]
    claims: tuple[str, ...]
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True)
class RejectedComposition:
    patch_ids: tuple[str, ...]
    diagnostics: tuple[Diagnostic, ...]


@dataclass(frozen=True)
class PortfolioResult:
    candidates: tuple[ComposedDesign, ...]
    rejected: tuple[RejectedComposition, ...]


_EDITABLE_FIELDS = frozenset(
    {
        "value_slice",
        "compute_warps",
        "accumulator_carrier",
        "operand_b_layout",
        "materialization",
        "scheduler",
        "pipeline",
        "tmem_lifetime",
        "producer_emits_consumer_layout",
        "preserves_bf16_rounding",
    }
)


def _compose(base: KernelDesign, patches: tuple[DesignPatch, ...]) -> ComposedDesign | RejectedComposition:
    updates: dict[str, object] = {}
    diagnostics: list[Diagnostic] = []
    for patch in patches:
        local_fields = [update.field for update in patch.updates]
        if len(set(local_fields)) != len(local_fields):
            diagnostics.append(Diagnostic("KIR1250", f"patch {patch.patch_id} updates one axis twice"))
        for update in patch.updates:
            if update.field not in _EDITABLE_FIELDS:
                diagnostics.append(Diagnostic("KIR1251", f"axis {update.field} is not editable"))
            elif update.field in updates and updates[update.field] != update.value:
                diagnostics.append(Diagnostic("KIR1252", f"conflicting updates for {update.field}"))
            else:
                updates[update.field] = update.value
    if diagnostics:
        return RejectedComposition(tuple(patch.patch_id for patch in patches), tuple(diagnostics))
    design = replace(
        base,
        design_id="compound:" + "+".join(patch.patch_id for patch in patches),
        **updates,
    )
    diagnostics.extend(verify_kernel_design(design))
    if diagnostics:
        return RejectedComposition(tuple(patch.patch_id for patch in patches), tuple(diagnostics))
    return ComposedDesign(
        design,
        tuple(patch.patch_id for patch in patches),
        tuple(patch.role for patch in patches),
        tuple(patch.claim for patch in patches),
        tuple(sorted({ref for patch in patches for ref in patch.evidence_refs})),
    )


def coordinate_portfolio(
    base: KernelDesign,
    patches: Iterable[DesignPatch],
    *,
    max_order: int = 3,
) -> PortfolioResult:
    """Enumerate compatible single and compound interventions deterministically."""

    items = tuple(sorted(patches, key=lambda item: item.patch_id))
    if len({item.patch_id for item in items}) != len(items):
        raise ValueError("patch IDs must be unique")
    if max_order <= 0:
        raise ValueError("max_order must be positive")

    candidates: dict[str, ComposedDesign] = {}
    rejected: list[RejectedComposition] = []
    for order in range(1, min(max_order, len(items)) + 1):
        for group in combinations(items, order):
            # A compound candidate represents collaboration only when it brings
            # together distinct strategy roles.
            if order > 1 and len({patch.role for patch in group}) < 2:
                continue
            result = _compose(base, group)
            if isinstance(result, RejectedComposition):
                rejected.append(result)
            else:
                candidates.setdefault(result.design.canonical_id(), result)
    ordered = tuple(candidates[key] for key in sorted(candidates))
    return PortfolioResult(ordered, tuple(rejected))


def design_niche(design: KernelDesign) -> tuple[object, ...]:
    """A quality-diversity cell: route, residency, scheduling, and warp density."""

    blocks_per_warp = (design.value_slice // 16) // design.compute_warps
    return (
        design.route.value,
        design.accumulator_carrier.value,
        design.scheduler.value,
        design.tmem_lifetime.value,
        blocks_per_warp,
        design.producer_emits_consumer_layout,
    )


@dataclass(frozen=True)
class MeasuredDesign:
    candidate: ComposedDesign
    speedup: float
    correctness_passed: bool
    scope_identical: bool


def portfolio_elites(rows: Iterable[MeasuredDesign]) -> dict[tuple[object, ...], MeasuredDesign]:
    """Keep the fastest valid candidate per mechanism niche, not one global winner."""

    elites: dict[tuple[object, ...], MeasuredDesign] = {}
    for row in rows:
        if (
            not row.correctness_passed
            or not row.scope_identical
            or verify_kernel_design(row.candidate.design)
        ):
            continue
        niche = design_niche(row.candidate.design)
        incumbent = elites.get(niche)
        if incumbent is None or (
            row.speedup,
            row.candidate.design.canonical_id(),
        ) > (
            incumbent.speedup,
            incumbent.candidate.design.canonical_id(),
        ):
            elites[niche] = row
    return elites
