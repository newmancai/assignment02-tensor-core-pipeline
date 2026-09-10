"""Anti-local-optimum knowledge layer for MMA migration research.

Measurements are evidence, not rules.  A claim records which independent
counterfactual paths have challenged it and may only be promoted according to
that coverage.  Performance claims can never become static verifier rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .mma_migration import MigrationDiagnostic


class KnowledgeKind(str, Enum):
    ARCHITECTURAL_FACT = "architectural_fact"
    MECHANISM_HYPOTHESIS = "mechanism_hypothesis"
    PERFORMANCE_DECISION = "performance_decision"


class KnowledgeMaturity(str, Enum):
    ATTEMPT = "attempt"
    SINGLE_PATH_OBSERVATION = "single_path_observation"
    MULTI_PATH_CORROBORATED = "multi_path_corroborated"
    CAUSAL_MECHANISM = "causal_mechanism"
    TRANSFER_VALIDATED = "transfer_validated"


class KnowledgeDisposition(str, Enum):
    ATTEMPT_JOURNAL = "attempt_journal"
    SCOPED_PRIOR = "scoped_prior"
    REUSABLE_MECHANISM = "reusable_mechanism"
    VERIFIER_RULE = "verifier_rule"


class ChallengeStatus(str, Enum):
    PLANNED = "planned"
    STATIC_REJECTED = "static_rejected"
    COMPILE_FAILED = "compile_failed"
    OBSERVED_SUPPORT = "observed_support"
    OBSERVED_REFUTE = "observed_refute"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class CounterfactualChallenge:
    challenge_id: str
    family: str
    changed_subtrees: tuple[str, ...]
    held_fixed: tuple[str, ...]
    status: ChallengeStatus
    evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class MigrationKnowledgeClaim:
    claim_id: str
    proposition: str
    kind: KnowledgeKind
    maturity: KnowledgeMaturity
    disposition: KnowledgeDisposition
    applicability_key: str
    supporting_evidence: tuple[str, ...]
    competing_mechanisms: tuple[str, ...]
    challenges: tuple[CounterfactualChallenge, ...]
    invalidation_keys: tuple[str, ...]


def verify_knowledge_claim(claim: MigrationKnowledgeClaim) -> tuple[MigrationDiagnostic, ...]:
    out: list[MigrationDiagnostic] = []
    if not claim.applicability_key or not claim.invalidation_keys:
        out.append(MigrationDiagnostic("TCG701", "knowledge needs scope and invalidation keys"))
    resolved = {
        ChallengeStatus.STATIC_REJECTED,
        ChallengeStatus.OBSERVED_SUPPORT,
        ChallengeStatus.OBSERVED_REFUTE,
    }
    resolved_families = {item.family for item in claim.challenges if item.status in resolved}
    if claim.maturity in {
        KnowledgeMaturity.MULTI_PATH_CORROBORATED,
        KnowledgeMaturity.CAUSAL_MECHANISM,
        KnowledgeMaturity.TRANSFER_VALIDATED,
    } and len(resolved_families) < 2:
        out.append(MigrationDiagnostic("TCG702", "multi-path maturity needs two resolved independent challenge families"))
    if claim.maturity is KnowledgeMaturity.CAUSAL_MECHANISM and len(claim.competing_mechanisms) != 1:
        out.append(MigrationDiagnostic("TCG703", "causal maturity requires one surviving mechanism"))
    if claim.kind is KnowledgeKind.PERFORMANCE_DECISION and claim.disposition is KnowledgeDisposition.VERIFIER_RULE:
        out.append(MigrationDiagnostic("TCG704", "performance decisions cannot become verifier rules"))
    if claim.disposition is KnowledgeDisposition.REUSABLE_MECHANISM and claim.maturity not in {
        KnowledgeMaturity.CAUSAL_MECHANISM,
        KnowledgeMaturity.TRANSFER_VALIDATED,
    }:
        out.append(MigrationDiagnostic("TCG705", "reusable mechanisms require causal or transfer maturity"))
    if claim.disposition is KnowledgeDisposition.SCOPED_PRIOR and claim.maturity is KnowledgeMaturity.ATTEMPT:
        out.append(MigrationDiagnostic("TCG706", "an unobserved attempt cannot become a scoped prior"))
    if any(not item.changed_subtrees or not item.held_fixed for item in claim.challenges):
        out.append(MigrationDiagnostic("TCG707", "every challenge must state changed and held-fixed subtrees"))
    return tuple(out)
