"""Validation for the preregistered Stage 11 tcgen05 causal ladder.

The ladder separates two orthogonal ways of challenging the archived direct
swap: amortising its lifecycle and changing its thin-N work geometry.  This
module validates the experiment declaration only; it contains no GPU result or
kernel lowering authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FactorState(str, Enum):
    BASELINE = "baseline"
    MUTATED = "mutated"


class KnowledgeMaturity(str, Enum):
    ATTEMPT = "attempt"
    SINGLE_PATH_OBSERVATION = "single_path_observation"
    MULTI_PATH_CORROBORATED = "multi_path_corroborated"
    CAUSAL_MECHANISM = "causal_mechanism"
    TRANSFER_VALIDATED = "transfer_validated"


@dataclass(frozen=True)
class LadderDiagnostic:
    code: str
    message: str


@dataclass(frozen=True)
class PathDefinition:
    path_id: str
    factor: str
    changed_subtrees: tuple[str, ...]
    held_fixed: tuple[str, ...]
    independent_agent_role: str
    falsifier: str


@dataclass(frozen=True)
class ScreenPlan:
    ordered_gates: tuple[str, ...]
    profiles: tuple[str, ...]
    comparators: tuple[str, ...]
    continue_rule: str


@dataclass(frozen=True)
class LadderCandidate:
    candidate_key: str
    lifecycle: FactorState
    thin_n: FactorState
    changed_subtrees: tuple[str, ...]
    held_fixed: tuple[str, ...]
    strongest_baseline: str
    attribution_sources: tuple[str, ...]
    screen: ScreenPlan
    stop_rules: tuple[str, ...]
    available_after: tuple[str, ...] = ()


@dataclass(frozen=True)
class CausalLadderPreregistration:
    schema_version: str
    experiment_id: str
    paths: tuple[PathDefinition, ...]
    candidates: tuple[LadderCandidate, ...]
    required_held_fixed: tuple[str, ...]
    strongest_baseline: str
    forbidden_attribution_sources: tuple[str, ...]
    interaction_contrast: str
    cross_agent_protocol: tuple[str, ...]
    knowledge_policy: tuple[str, ...]
    contains_gpu_results: bool = False


def validate_causal_ladder(
    prereg: CausalLadderPreregistration,
) -> tuple[LadderDiagnostic, ...]:
    """Fail closed when the four-cell ladder cannot identify A, B and A×B."""

    out: list[LadderDiagnostic] = []
    if prereg.schema_version != "stage11_tcgen05_causal_ladder_v1":
        out.append(LadderDiagnostic("S11L001", "unsupported causal-ladder schema"))
    if prereg.contains_gpu_results:
        out.append(LadderDiagnostic("S11L002", "preregistration cannot contain GPU results"))
    baseline_prefix, separator, baseline_digest = prereg.strongest_baseline.partition(":")
    if not (
        separator == ":"
        and baseline_prefix in {"sha256", "mma256"}
        and len(baseline_digest) == 64
        and all(char in "0123456789abcdef" for char in baseline_digest)
    ):
        out.append(LadderDiagnostic("S11L008", "strongest baseline needs a namespaced full SHA-256 identity"))

    path_map = {path.factor: path for path in prereg.paths}
    if len(prereg.paths) != 2 or set(path_map) != {"lifecycle", "thin_n"}:
        out.append(LadderDiagnostic("S11L003", "exactly lifecycle and thin_n paths are required"))
    if len({path.path_id for path in prereg.paths}) != len(prereg.paths):
        out.append(LadderDiagnostic("S11L004", "path IDs must be unique"))
    if len({path.independent_agent_role for path in prereg.paths}) != len(prereg.paths):
        out.append(LadderDiagnostic("S11L005", "orthogonal paths require independent agent roles"))
    if any(
        not path.changed_subtrees or not path.held_fixed or not path.falsifier
        for path in prereg.paths
    ):
        out.append(LadderDiagnostic("S11L006", "every path needs mutations, invariants, and a falsifier"))
    if len(path_map) == 2 and set(path_map["lifecycle"].changed_subtrees) & set(
        path_map["thin_n"].changed_subtrees
    ):
        out.append(LadderDiagnostic("S11L007", "A and B mutation subtrees must be disjoint"))

    cells = {(candidate.lifecycle, candidate.thin_n): candidate for candidate in prereg.candidates}
    expected_cells = {
        (FactorState.BASELINE, FactorState.BASELINE),
        (FactorState.MUTATED, FactorState.BASELINE),
        (FactorState.BASELINE, FactorState.MUTATED),
        (FactorState.MUTATED, FactorState.MUTATED),
    }
    if len(prereg.candidates) != 4 or set(cells) != expected_cells:
        out.append(LadderDiagnostic("S11L010", "the complete 2x2 factorial is required"))
    if len({candidate.candidate_key for candidate in prereg.candidates}) != len(prereg.candidates):
        out.append(LadderDiagnostic("S11L011", "candidate keys must be unique"))

    required_fixed = set(prereg.required_held_fixed)
    forbidden = set(prereg.forbidden_attribution_sources)
    for candidate in prereg.candidates:
        if not candidate.changed_subtrees:
            out.append(LadderDiagnostic("S11L012", f"{candidate.candidate_key} has no changed subtrees"))
        if not required_fixed.issubset(candidate.held_fixed):
            out.append(LadderDiagnostic("S11L013", f"{candidate.candidate_key} omits a global invariant"))
        if candidate.strongest_baseline != prereg.strongest_baseline:
            out.append(LadderDiagnostic("S11L014", f"{candidate.candidate_key} changes the strongest baseline"))
        if forbidden & set(candidate.attribution_sources):
            out.append(LadderDiagnostic("S11L015", f"{candidate.candidate_key} uses forbidden causal evidence"))
        if not candidate.attribution_sources:
            out.append(LadderDiagnostic("S11L016", f"{candidate.candidate_key} lacks candidate-matched evidence"))
        if not candidate.screen.ordered_gates or candidate.screen.ordered_gates[0] != "static_verifier":
            out.append(LadderDiagnostic("S11L017", f"{candidate.candidate_key} must screen statically first"))
        if not candidate.screen.profiles or not candidate.screen.comparators:
            out.append(LadderDiagnostic("S11L018", f"{candidate.candidate_key} has an incomplete screen"))
        if not candidate.screen.continue_rule or not candidate.stop_rules:
            out.append(LadderDiagnostic("S11L019", f"{candidate.candidate_key} lacks continue/stop rules"))

    if len(path_map) == 2 and expected_cells.issubset(cells):
        control = cells[(FactorState.BASELINE, FactorState.BASELINE)]
        a_only = cells[(FactorState.MUTATED, FactorState.BASELINE)]
        b_only = cells[(FactorState.BASELINE, FactorState.MUTATED)]
        interaction = cells[(FactorState.MUTATED, FactorState.MUTATED)]
        a_paths = set(path_map["lifecycle"].changed_subtrees)
        b_paths = set(path_map["thin_n"].changed_subtrees)
        if not a_paths.issubset(a_only.changed_subtrees) or set(a_only.changed_subtrees) & b_paths:
            out.append(LadderDiagnostic("S11L020", "A-only cell does not isolate lifecycle"))
        if not b_paths.issubset(b_only.changed_subtrees) or set(b_only.changed_subtrees) & a_paths:
            out.append(LadderDiagnostic("S11L021", "B-only cell does not isolate thin-N"))
        if not (a_paths | b_paths).issubset(interaction.changed_subtrees):
            out.append(LadderDiagnostic("S11L022", "interaction cell does not contain both paths"))
        required_parents = {a_only.candidate_key, b_only.candidate_key}
        if not required_parents.issubset(interaction.available_after):
            out.append(LadderDiagnostic("S11L023", "interaction must wait for both independent path freezes"))
        if control.candidate_key not in a_only.screen.comparators or control.candidate_key not in b_only.screen.comparators:
            out.append(LadderDiagnostic("S11L024", "main-effect screens must include the direct-swap control"))

    required_agent_rules = {
        "A_and_B_use_independent_agent_contexts",
        "no_raw_trajectory_exchange_before_path_freeze",
        "interaction_authored_only_after_A_and_B_freeze",
    }
    if not required_agent_rules.issubset(prereg.cross_agent_protocol):
        out.append(LadderDiagnostic("S11L030", "cross-agent independence protocol is incomplete"))
    required_knowledge_rules = {
        "single_path_decision_grade_negative_is_reopenable_scoped_prior",
        "unresolved_or_noise_band_result_is_attempt_journal_only",
        "A_or_B_negative_does_not_cancel_AB",
        "reusable_negative_requires_two_resolved_independent_paths_and_AB",
        "target_toolchain_profile_or_held_fixed_change_invalidates_reuse",
    }
    if not required_knowledge_rules.issubset(prereg.knowledge_policy):
        out.append(LadderDiagnostic("S11L031", "anti-local-optimum knowledge policy is incomplete"))
    if "(AB-A)-(B-control)" not in prereg.interaction_contrast:
        out.append(LadderDiagnostic("S11L032", "interaction contrast is not preregistered"))
    return tuple(out)
