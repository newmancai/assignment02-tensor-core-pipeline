"""Evidence-mediated coordination for the Stage-9 schedule-policy backend.

The frozen ``Schedule`` model is one executable specialization, not a universal
proposal schema.  Open research proposals and scoped cross-route experience are
represented by :mod:`kda_ir.research_memory`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from .evolve import AcceptedProposal, FailureLedger, Proposal, RejectedProposal, evolve
from .model import Schedule, Target


class AgentRole(str, Enum):
    PROFILE_ANALYST = "profile_analyst"
    SCHEDULE_EXPLORER = "schedule_explorer"
    SAFETY_CRITIC = "safety_critic"
    MEASUREMENT_JUDGE = "measurement_judge"
    META_REVIEWER = "meta_reviewer"


@dataclass(frozen=True)
class AgentContribution:
    proposal_id: str
    role: AgentRole
    proposal: Proposal
    claim: str
    evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class CoordinatedCandidate:
    schedule: Schedule
    accepted: AcceptedProposal
    proposal_ids: tuple[str, ...]
    roles: tuple[AgentRole, ...]
    claims: tuple[str, ...]
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True)
class CoordinationResult:
    candidates: tuple[CoordinatedCandidate, ...]
    rejected: tuple[RejectedProposal, ...]


class ActivationStatus(str, Enum):
    REJECT = "reject"
    MEASURE = "measure"
    ACTIVATE = "activate"


@dataclass(frozen=True)
class ActivationEvidence:
    verifier_clean: bool
    output_and_state_correct: bool = False
    scope_identical: bool = False
    bootstrap_95_lower: float | None = None
    fallback_defined: bool = False
    receipt_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class ActivationDecision:
    status: ActivationStatus
    reasons: tuple[str, ...]


def coordinate_contributions(
    contributions: Iterable[AgentContribution],
    target: Target,
    ledger: FailureLedger | None = None,
) -> CoordinationResult:
    """Verify first, then merge identical schedules inside this frozen grammar.

    Agent agreement is retained as provenance but never substitutes for the
    verifier.  Equality is structural because Schedule is a frozen dataclass.
    Callers must not use this function to deduplicate source-distinct kernels,
    architecture-specific proposal families, or ideas awaiting an IR extension.
    """

    items = tuple(contributions)
    if len({item.proposal_id for item in items}) != len(items):
        raise ValueError("proposal IDs must be unique within a coordination round")
    evolved = evolve((item.proposal for item in items), target, ledger)
    accepted_by_schedule = {item.proposal.schedule: item for item in evolved.accepted}
    grouped: dict[Schedule, list[AgentContribution]] = {}
    for contribution in items:
        if contribution.proposal.schedule in accepted_by_schedule:
            grouped.setdefault(contribution.proposal.schedule, []).append(contribution)

    candidates = []
    for schedule, group in grouped.items():
        refs = tuple(sorted({ref for item in group for ref in item.evidence_refs}))
        candidates.append(
            CoordinatedCandidate(
                schedule=schedule,
                accepted=accepted_by_schedule[schedule],
                proposal_ids=tuple(item.proposal_id for item in group),
                roles=tuple(item.role for item in group),
                claims=tuple(item.claim for item in group),
                evidence_refs=refs,
            )
        )
    return CoordinationResult(tuple(candidates), evolved.rejected)


def activation_decision(evidence: ActivationEvidence) -> ActivationDecision:
    """Apply objective activation gates after multi-agent proposal synthesis."""

    if not evidence.verifier_clean:
        return ActivationDecision(ActivationStatus.REJECT, ("typed verifier failed",))
    missing = []
    if not evidence.output_and_state_correct:
        missing.append("output/final-state correctness")
    if not evidence.scope_identical:
        missing.append("scope-identical public measurement")
    if evidence.bootstrap_95_lower is None:
        missing.append("bootstrap confidence interval")
    elif evidence.bootstrap_95_lower <= 1.0:
        missing.append("positive 95% speedup lower bound")
    if not evidence.fallback_defined:
        missing.append("fallback")
    if missing:
        return ActivationDecision(ActivationStatus.MEASURE, tuple(missing))
    return ActivationDecision(
        ActivationStatus.ACTIVATE,
        ("all proof-carrying activation gates passed",),
    )
