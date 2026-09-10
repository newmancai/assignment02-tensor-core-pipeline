"""Verifier-guided schedule evolution and failure-knowledge collection."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from .lower import lowering_plan
from .model import Schedule, Target
from .verify import Diagnostic, verify


@dataclass(frozen=True)
class Proposal:
    schedule: Schedule
    rationale: str
    parent: str | None = None


@dataclass(frozen=True)
class ScheduleMetrics:
    launches: int
    value_partitions: int
    shared_bytes: int
    resident_ctas_per_sm: int


@dataclass(frozen=True)
class AcceptedProposal:
    proposal: Proposal
    metrics: ScheduleMetrics


@dataclass(frozen=True)
class RejectedProposal:
    proposal: Proposal
    diagnostics: tuple[Diagnostic, ...]


@dataclass
class FailureLedger:
    """Aggregate recurring failures without automatically changing the language."""

    counts: Counter[str] = field(default_factory=Counter)
    examples: dict[str, str] = field(default_factory=dict)

    def observe(self, rejected: RejectedProposal) -> None:
        for diagnostic in rejected.diagnostics:
            self.counts[diagnostic.code] += 1
            self.examples.setdefault(diagnostic.code, diagnostic.message)

    def promotion_candidates(self, minimum_occurrences: int = 2) -> tuple[dict[str, object], ...]:
        """Return evidence for human/agent review; promotion is never implicit."""

        return tuple(
            {
                "diagnostic": code,
                "occurrences": count,
                "example": self.examples[code],
            }
            for code, count in sorted(self.counts.items())
            if count >= minimum_occurrences
        )


@dataclass(frozen=True)
class EvolutionResult:
    accepted: tuple[AcceptedProposal, ...]
    rejected: tuple[RejectedProposal, ...]
    pareto_frontier: tuple[AcceptedProposal, ...]


def _metrics(schedule: Schedule, target: Target) -> ScheduleMetrics:
    plan = lowering_plan(schedule, target)
    return ScheduleMetrics(
        launches=len(plan["launches"]),
        value_partitions=128 // schedule.tile.value,
        shared_bytes=schedule.resources.shared_bytes,
        resident_ctas_per_sm=int(plan["estimated_resident_ctas_per_sm"]),
    )


def _dominates(left: ScheduleMetrics, right: ScheduleMetrics) -> bool:
    no_worse = (
        left.launches <= right.launches
        and left.value_partitions <= right.value_partitions
        and left.shared_bytes <= right.shared_bytes
        and left.resident_ctas_per_sm >= right.resident_ctas_per_sm
    )
    strictly_better = left != right
    return no_worse and strictly_better


def evolve(
    proposals: Iterable[Proposal],
    target: Target,
    ledger: FailureLedger | None = None,
) -> EvolutionResult:
    """Verify proposals and retain a cost-model-free resource Pareto frontier."""

    accepted: list[AcceptedProposal] = []
    rejected: list[RejectedProposal] = []
    for proposal in proposals:
        diagnostics = verify(proposal.schedule, target)
        if diagnostics:
            item = RejectedProposal(proposal, diagnostics)
            rejected.append(item)
            if ledger is not None:
                ledger.observe(item)
            continue
        accepted.append(AcceptedProposal(proposal, _metrics(proposal.schedule, target)))

    frontier = tuple(
        candidate
        for candidate in accepted
        if not any(
            _dominates(other.metrics, candidate.metrics)
            for other in accepted
            if other is not candidate
        )
    )
    return EvolutionResult(tuple(accepted), tuple(rejected), frontier)

