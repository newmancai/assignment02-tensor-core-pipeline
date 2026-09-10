"""Deterministic closed loop around agent proposals and hardware evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Callable, Iterable, Protocol

from .model import Schedule, Target
from .multi_agent import (
    ActivationEvidence,
    ActivationStatus,
    AgentContribution,
    CoordinatedCandidate,
    activation_decision,
    coordinate_contributions,
)


def canonical_candidate_id(schedule: Schedule) -> str:
    """Content-address a typed schedule independently of proposer wording."""

    payload = json.dumps(asdict(schedule), sort_keys=True, separators=(",", ":"))
    return f"kir:{sha256(payload.encode()).hexdigest()[:16]}"


@dataclass(frozen=True)
class ScreenResult:
    candidate_id: str
    latency_us: float
    output_and_state_correct: bool
    scope_identical: bool
    evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class QualificationResult:
    candidate_id: str
    speedup: float
    bootstrap_95: tuple[float, float]
    output_and_state_correct: bool
    scope_identical: bool
    fallback_defined: bool
    evidence_refs: tuple[str, ...] = ()


class MeasurementOracle(Protocol):
    def screen(self, candidate: CoordinatedCandidate) -> ScreenResult: ...

    def qualify(self, candidate: CoordinatedCandidate) -> QualificationResult: ...


@dataclass(frozen=True)
class Conclusion:
    round_index: int
    profile_id: str
    candidate_id: str
    state: str
    conclusion: str
    evidence_refs: tuple[str, ...] = ()


@dataclass
class ConclusionsMemory:
    """Store compact outcomes, never hidden reasoning or raw measurement arrays."""

    entries: list[Conclusion]

    @classmethod
    def empty(cls) -> "ConclusionsMemory":
        return cls([])

    def append(self, conclusion: Conclusion) -> None:
        self.entries.append(conclusion)

    def to_dict(self) -> dict[str, object]:
        return {"schema_version": 1, "entries": [asdict(item) for item in self.entries]}

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")


@dataclass(frozen=True)
class RoundResult:
    round_index: int
    candidate_count: int
    rejected_count: int
    screened: tuple[ScreenResult, ...]
    qualified: tuple[QualificationResult, ...]
    activated_candidate_id: str | None
    stopped_reason: str


@dataclass(frozen=True)
class LoopResult:
    rounds: tuple[RoundResult, ...]
    incumbent_candidate_id: str | None
    stopped_reason: str


class AutonomousRuntimeProfileAgent:
    """Run propose/verify/screen/qualify/activate rounds under fixed budgets.

    Language-model workers may implement ``proposal_source``. Verification,
    deduplication, hardware measurement, activation, memory, and stopping remain
    deterministic so the loop can be audited and replayed.
    """

    def __init__(
        self,
        target: Target,
        oracle: MeasurementOracle,
        *,
        profile_id: str = "unspecified-profile",
        top_k: int = 1,
        max_rounds: int = 4,
        plateau_rounds: int = 2,
        max_screened_candidates: int = 32,
        memory: ConclusionsMemory | None = None,
    ) -> None:
        if min(top_k, max_rounds, plateau_rounds, max_screened_candidates) <= 0:
            raise ValueError("loop budgets must be positive")
        if not profile_id:
            raise ValueError("profile_id must be non-empty")
        self.target = target
        self.oracle = oracle
        self.profile_id = profile_id
        self.top_k = top_k
        self.max_rounds = max_rounds
        self.plateau_rounds = plateau_rounds
        self.max_screened_candidates = max_screened_candidates
        self.memory = memory or ConclusionsMemory.empty()

    def run(
        self,
        proposal_source: Callable[
            [int, tuple[Conclusion, ...]], Iterable[AgentContribution]
        ],
    ) -> LoopResult:
        rounds = []
        incumbent_id: str | None = None
        incumbent_lower = 1.0
        plateau = 0
        screened_budget = 0
        seen: set[str] = set()

        for round_index in range(1, self.max_rounds + 1):
            contributions = tuple(
                proposal_source(round_index, tuple(self.memory.entries))
            )
            coordinated = coordinate_contributions(contributions, self.target)
            candidates = []
            for candidate in coordinated.candidates:
                candidate_id = canonical_candidate_id(candidate.schedule)
                if candidate_id not in seen:
                    candidates.append((candidate_id, candidate))
                    seen.add(candidate_id)
            for rejected in coordinated.rejected:
                rejected_id = canonical_candidate_id(rejected.proposal.schedule)
                codes = ",".join(item.code for item in rejected.diagnostics)
                self.memory.append(
                    Conclusion(
                        round_index,
                        self.profile_id,
                        rejected_id,
                        "verifier_rejected",
                        f"typed verifier rejected candidate: {codes}",
                    )
                )

            remaining = self.max_screened_candidates - screened_budget
            if remaining <= 0:
                stopped = "screen_budget_exhausted"
                rounds.append(
                    RoundResult(
                        round_index,
                        len(candidates),
                        len(coordinated.rejected),
                        (),
                        (),
                        None,
                        stopped,
                    )
                )
                return LoopResult(tuple(rounds), incumbent_id, stopped)
            candidates = candidates[:remaining]

            screens = []
            candidate_by_id = {}
            for candidate_id, candidate in candidates:
                result = self.oracle.screen(candidate)
                if result.candidate_id != candidate_id:
                    raise ValueError("screen receipt candidate ID mismatch")
                screens.append(result)
                candidate_by_id[candidate_id] = candidate
                screened_budget += 1
                state = (
                    "screened"
                    if result.output_and_state_correct and result.scope_identical
                    else "screen_rejected"
                )
                self.memory.append(
                    Conclusion(
                        round_index,
                        self.profile_id,
                        candidate_id,
                        state,
                        f"screen latency {result.latency_us:.6f} us",
                        result.evidence_refs,
                    )
                )

            eligible = sorted(
                (
                    item
                    for item in screens
                    if item.output_and_state_correct and item.scope_identical
                ),
                key=lambda item: item.latency_us,
            )[: self.top_k]
            qualifications = []
            activated_id = None
            for screen in eligible:
                result = self.oracle.qualify(candidate_by_id[screen.candidate_id])
                if result.candidate_id != screen.candidate_id:
                    raise ValueError("qualification receipt candidate ID mismatch")
                qualifications.append(result)
                decision = activation_decision(
                    ActivationEvidence(
                        verifier_clean=True,
                        output_and_state_correct=result.output_and_state_correct,
                        scope_identical=result.scope_identical,
                        bootstrap_95_lower=result.bootstrap_95[0],
                        fallback_defined=result.fallback_defined,
                        receipt_refs=result.evidence_refs,
                    )
                )
                self.memory.append(
                    Conclusion(
                        round_index,
                        self.profile_id,
                        result.candidate_id,
                        decision.status.value,
                        (
                            f"paired speedup {result.speedup:.6f}x; "
                            f"95% CI [{result.bootstrap_95[0]:.6f}, "
                            f"{result.bootstrap_95[1]:.6f}]"
                        ),
                        result.evidence_refs,
                    )
                )
                if (
                    decision.status == ActivationStatus.ACTIVATE
                    and result.bootstrap_95[0] > incumbent_lower
                ):
                    incumbent_id = result.candidate_id
                    incumbent_lower = result.bootstrap_95[0]
                    activated_id = result.candidate_id

            if activated_id is None:
                plateau += 1
            else:
                plateau = 0
            stopped = "continue"
            if not candidates:
                stopped = "no_new_candidates"
            elif plateau >= self.plateau_rounds:
                stopped = "evidence_plateau"
            rounds.append(
                RoundResult(
                    round_index,
                    len(candidates),
                    len(coordinated.rejected),
                    tuple(screens),
                    tuple(qualifications),
                    activated_id,
                    stopped,
                )
            )
            if stopped != "continue":
                return LoopResult(tuple(rounds), incumbent_id, stopped)

        return LoopResult(tuple(rounds), incumbent_id, "max_rounds")
