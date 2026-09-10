"""Scope-explicit closure certificates for finite typed design envelopes.

An optimization lane is closed only with respect to a declared set of typed
designs and a declared benchmark scope.  This avoids turning a local negative
result into a claim about all CUDA programs or all Blackwell workloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
from typing import Iterable

from .design_ir import KernelDesign, verify_kernel_design
from .verify import Diagnostic


class TerminalDisposition(str, Enum):
    QUALIFIED = "qualified"
    MEASURED_BELOW_GATE = "measured_below_gate"


@dataclass(frozen=True)
class DesignEnvelope:
    envelope_id: str
    scope: str
    designs: tuple[KernelDesign, ...]


@dataclass(frozen=True)
class DesignObservation:
    canonical_design_id: str
    disposition: TerminalDisposition
    correctness_passed: bool
    evidence_ref: str
    speedup: float


@dataclass(frozen=True)
class ClosureCertificate:
    envelope_id: str
    scope: str
    envelope_digest: str
    closed: bool
    covered_design_ids: tuple[str, ...]
    missing_design_ids: tuple[str, ...]
    diagnostics: tuple[Diagnostic, ...]


def _envelope_digest(envelope: DesignEnvelope, design_ids: tuple[str, ...]) -> str:
    payload = {
        "envelope_id": envelope.envelope_id,
        "scope": envelope.scope,
        "design_ids": design_ids,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "envelope:" + sha256(encoded.encode()).hexdigest()


def certify_design_envelope(
    envelope: DesignEnvelope,
    observations: Iterable[DesignObservation],
) -> ClosureCertificate:
    """Prove coverage of a finite envelope, without claiming a global optimum."""

    diagnostics: list[Diagnostic] = []
    design_ids: list[str] = []
    for design in envelope.designs:
        invalid = verify_kernel_design(design)
        if invalid:
            diagnostics.append(
                Diagnostic(
                    "KIR1260",
                    f"envelope contains invalid design {design.design_id}: "
                    + ",".join(item.code for item in invalid),
                )
            )
        design_ids.append(design.canonical_id())
    if len(set(design_ids)) != len(design_ids):
        diagnostics.append(Diagnostic("KIR1261", "envelope contains duplicate semantic designs"))

    rows = tuple(observations)
    row_ids = [row.canonical_design_id for row in rows]
    if len(set(row_ids)) != len(row_ids):
        diagnostics.append(Diagnostic("KIR1262", "a design has multiple terminal observations"))
    expected = set(design_ids)
    for row in rows:
        if row.canonical_design_id not in expected:
            diagnostics.append(Diagnostic("KIR1263", "observation lies outside the envelope"))
        if not row.evidence_ref:
            diagnostics.append(Diagnostic("KIR1264", "terminal observation requires an evidence reference"))
        if not row.correctness_passed:
            diagnostics.append(Diagnostic("KIR1265", "failed correctness is not a terminal performance result"))
        if not (row.speedup > 0.0):
            diagnostics.append(Diagnostic("KIR1266", "terminal speedup must be positive"))

    covered = tuple(sorted(expected.intersection(row_ids)))
    missing = tuple(sorted(expected.difference(row_ids)))
    return ClosureCertificate(
        envelope.envelope_id,
        envelope.scope,
        _envelope_digest(envelope, tuple(sorted(design_ids))),
        not diagnostics and not missing,
        covered,
        missing,
        tuple(diagnostics),
    )
