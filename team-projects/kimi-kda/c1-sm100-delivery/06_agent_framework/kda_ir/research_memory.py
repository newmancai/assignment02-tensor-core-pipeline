"""Open research objective and two-level memory for SM100 migration work.

The Stage 9 policy tree is one executable case-study backend.  It is not the
proposal language of the general agent framework.  This module keeps research
intent and scoped experience outside that narrow schedule schema so ideas that
need a new carrier or IR extension are retained without being called invalid.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path


class ResearchLane(str, Enum):
    HMMA = "hmma"
    TCGEN05 = "tcgen05"
    MATCHED_COUNTERFACTUAL = "matched_counterfactual"
    CROSS_CUTTING = "cross_cutting"


class RepresentationStatus(str, Enum):
    EXECUTABLE_EXISTING = "executable_existing"
    PATCH_REQUIRED = "patch_required"
    NEEDS_IR_EXTENSION = "needs_ir_extension"
    PROTOTYPE_FIRST = "prototype_first"


class EvidenceStatus(str, Enum):
    HYPOTHESIS = "hypothesis"
    COMPILED = "compiled"
    CORRECTNESS_CHECKED = "correctness_checked"
    SCREENED = "screened"
    QUALIFIED = "qualified"
    DEPLOYMENT_EVIDENCE = "deployment_evidence"
    DIAGNOSTIC_ONLY = "diagnostic_only"
    SUPERSEDED = "superseded"


class MechanismStatus(str, Enum):
    OBSERVATION = "observation"
    LOCALIZATION = "localization"
    CONTROLLED_INTERVENTION = "controlled_intervention"


class PermittedUse(str, Enum):
    RANKING_PRIOR = "ranking_prior"
    REQUIRES_RETEST = "requires_retest"
    QUALIFICATION_CANDIDATE = "qualification_candidate"
    DEPLOYMENT = "deployment"


@dataclass(frozen=True)
class ResearchObjective:
    objective_id: str
    primary_question: str
    challenge_ladder: tuple[str, ...]
    non_substitution_rules: tuple[str, ...]


SM100_MIGRATION_OBJECTIVE = ResearchObjective(
    objective_id="flashkda-sm80-hmma-to-sm100-v1",
    primary_question=(
        "FlashKDA 官方 kernel 当前使用 SM80 MMA，分析迁移到 SM100 是否值得"
    ),
    challenge_ladder=(
        "H0: official HMMA",
        "H1: strongest HMMA after an explicit, comparable search budget",
        "T0: structurally matched tcgen05 plus necessary protocol",
        "T1: strongest tcgen05 after an explicit, comparable search budget",
        "T2: strongest guarded SM100 deployment path",
    ),
    non_substitution_rules=(
        "Do not replace the migration question with cpc tuning, concurrency, or framework construction.",
        "Do not report CAKE full-stack gain as the isolated tcgen05 opcode effect.",
        "Do not report official HMMA as the strongest available HMMA after stronger-HMMA evidence exists.",
        "Do not require HMMA and tcgen05 proposals to share architecture-specific fields.",
    ),
)


@dataclass(frozen=True)
class ProgramIRRecord:
    """Lower-layer implementation identity, including explicitly unknown fields."""

    design_id: str
    objective_id: str
    lane: ResearchLane
    semantic_contract_id: str
    representation_status: RepresentationStatus
    requested_backend: str
    requested_route: str | None
    actual_route: str | None
    changed_subtrees: tuple[str, ...]
    source_refs: tuple[str, ...] = ()
    build_refs: tuple[str, ...] = ()
    unknown_fields: tuple[str, ...] = ()

    @property
    def execution_ready(self) -> bool:
        return self.representation_status == RepresentationStatus.EXECUTABLE_EXISTING

    @property
    def observed_identity_matches(self) -> bool:
        if self.actual_route is None:
            return False
        return self.requested_route is None or self.requested_route == self.actual_route

    def canonical_id(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return f"program-ir:{sha256(payload.encode()).hexdigest()}"


@dataclass(frozen=True)
class ExperienceIRRecord:
    """Upper-layer claim with transfer conditions and allowed downstream use."""

    observation_id: str
    proposition: str
    compared_design_ids: tuple[str, ...]
    profile_id: str
    contract_id: str
    evidence_status: EvidenceStatus
    mechanism_status: MechanismStatus
    matched_preconditions: tuple[str, ...]
    mismatched_preconditions: tuple[str, ...]
    unknown_preconditions: tuple[str, ...]
    competing_mechanisms: tuple[str, ...]
    invalidation_keys: tuple[str, ...]
    reopen_conditions: tuple[str, ...]
    permitted_use: PermittedUse
    evidence_refs: tuple[str, ...] = ()


@dataclass
class DualIRMemory:
    objective: ResearchObjective
    programs: list[ProgramIRRecord]
    experiences: list[ExperienceIRRecord]

    @classmethod
    def empty(
        cls, objective: ResearchObjective = SM100_MIGRATION_OBJECTIVE
    ) -> "DualIRMemory":
        return cls(objective, [], [])

    def add_program(self, program: ProgramIRRecord) -> None:
        if program.objective_id != self.objective.objective_id:
            raise ValueError("program objective differs from the active mainline")
        self.programs.append(program)

    def add_experience(self, experience: ExperienceIRRecord) -> None:
        known = {program.design_id for program in self.programs}
        missing = set(experience.compared_design_ids).difference(known)
        if missing:
            raise ValueError(f"experience references unknown designs: {sorted(missing)}")
        self.experiences.append(experience)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "objective": asdict(self.objective),
            "program_ir": [asdict(item) for item in self.programs],
            "experience_ir": [asdict(item) for item in self.experiences],
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")


def deterministic_proposal_disposition(program: ProgramIRRecord) -> str:
    """Classify readiness without confusing representation gaps with rejection."""

    if program.representation_status == RepresentationStatus.NEEDS_IR_EXTENSION:
        return "retain_needs_ir_extension"
    if program.representation_status == RepresentationStatus.PATCH_REQUIRED:
        return "retain_patch_required"
    if program.representation_status == RepresentationStatus.PROTOTYPE_FIRST:
        return "retain_prototype_first"
    if not program.observed_identity_matches:
        return "diagnostic_identity_unconfirmed"
    return "ready_for_correctness_screen"
