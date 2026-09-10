"""Budget, sealing, and statistical contracts for Stage 9 ablations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import math
import random
import statistics


class AblationArm(str, Enum):
    SINGLE_GENERALIST = "single_generalist"
    SINGLE_ROLEPLAY = "single_roleplay"
    MULTI_HOMOGENEOUS = "multi_homogeneous"
    MULTI_ROLE_NO_SHARE = "multi_role_no_share"
    MULTI_ROLE_CONCLUSIONS = "multi_role_conclusions"


@dataclass(frozen=True)
class EqualBudgetContract:
    """Common per-arm opportunity caps, aggregated across trajectories."""

    trajectories: int
    proposal_tokens_per_arm: int
    proposal_slots_per_arm: int
    screen_slots_per_arm: int
    qualification_slots_per_arm: int
    gpu_seconds_per_arm: float
    gpu_balance_tolerance_fraction: float = 0.05

    def __post_init__(self) -> None:
        if min(
            self.trajectories,
            self.proposal_tokens_per_arm,
            self.proposal_slots_per_arm,
            self.screen_slots_per_arm,
            self.qualification_slots_per_arm,
        ) <= 0:
            raise ValueError("ablation budgets must be positive")
        if self.gpu_seconds_per_arm <= 0:
            raise ValueError("GPU budget must be positive")
        if not 0 <= self.gpu_balance_tolerance_fraction < 1:
            raise ValueError("GPU balance tolerance must be in [0, 1)")


@dataclass(frozen=True)
class SharedExperimentEnvelope:
    """Configuration that must be byte-identical across all arms."""

    model_digest: str
    model_config_digest: str
    profile_split_digest: str
    common_prompt_digest: str
    candidate_space_digest: str
    tool_manifest_digest: str

    def __post_init__(self) -> None:
        if not all(self.__dict__.values()):
            raise ValueError("shared-envelope digests must be non-empty")


@dataclass(frozen=True)
class TrajectoryReceipt:
    """One paired stochastic search episode and its frozen held-out outcome."""

    trajectory_id: str
    seed: int
    winner_candidate_id: str
    heldout_log_speedups: tuple[float, ...]
    heldout_seal_sha256: str
    winner_frozen_at_utc: str
    heldout_unsealed_at_utc: str

    @property
    def heldout_metric(self) -> float:
        return statistics.fmean(self.heldout_log_speedups)

    def __post_init__(self) -> None:
        if not self.trajectory_id or not self.winner_candidate_id:
            raise ValueError("trajectory and winner IDs must be non-empty")
        if not self.heldout_log_speedups or not all(
            math.isfinite(value) for value in self.heldout_log_speedups
        ):
            raise ValueError("held-out log speedups must be finite and non-empty")


@dataclass(frozen=True)
class ArmReceipt:
    arm: AblationArm
    shared_envelope: SharedExperimentEnvelope
    arm_protocol_digest: str
    trajectories: tuple[TrajectoryReceipt, ...]
    proposal_token_cap: int
    proposal_tokens_actual: int
    proposal_slot_cap: int
    proposal_slots_charged: int
    unique_candidates: int
    screen_slot_cap: int
    screen_slots_charged: int
    screen_attempts_actual: int
    qualification_slot_cap: int
    qualification_slots_charged: int
    qualification_attempts_actual: int
    gpu_second_cap: float
    gpu_seconds_actual: float
    compound_candidates: int = 0

    def __post_init__(self) -> None:
        if not self.arm_protocol_digest:
            raise ValueError("arm protocol digest must be non-empty")
        integer_counts = (
            self.proposal_token_cap,
            self.proposal_tokens_actual,
            self.proposal_slot_cap,
            self.proposal_slots_charged,
            self.unique_candidates,
            self.screen_slot_cap,
            self.screen_slots_charged,
            self.screen_attempts_actual,
            self.qualification_slot_cap,
            self.qualification_slots_charged,
            self.qualification_attempts_actual,
            self.compound_candidates,
        )
        if min(integer_counts) < 0 or self.gpu_second_cap <= 0 or self.gpu_seconds_actual < 0:
            raise ValueError("receipt budgets and counts must be non-negative")


@dataclass(frozen=True)
class BootstrapContrast:
    treatment: AblationArm
    control: AblationArm
    estimate: float
    simultaneous_lower: float
    confidence: float
    paired_trajectory_ids: tuple[str, ...]


@dataclass(frozen=True)
class AblationAudit:
    claim_ready: bool
    checks: tuple[tuple[str, bool], ...]
    reasons: tuple[str, ...]
    contrasts: tuple[BootstrapContrast, ...] = ()


def _parse_utc(value: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError("timestamps must include a timezone")
    return parsed


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def paired_bootstrap_contrast(
    treatment: ArmReceipt,
    control: ArmReceipt,
    *,
    family_size: int = 3,
    confidence: float = 0.95,
    iterations: int = 20_000,
    seed: int = 9,
) -> BootstrapContrast:
    """Bootstrap paired trajectory means with Bonferroni simultaneous coverage."""

    if family_size <= 0 or not 0 < confidence < 1 or iterations <= 0:
        raise ValueError("invalid bootstrap configuration")
    left = {row.trajectory_id: row for row in treatment.trajectories}
    right = {row.trajectory_id: row for row in control.trajectories}
    if set(left) != set(right):
        raise ValueError("paired arms must contain identical trajectory IDs")
    ids = tuple(sorted(left))
    if not ids:
        raise ValueError("at least one paired trajectory is required")
    differences = [left[item].heldout_metric - right[item].heldout_metric for item in ids]
    rng = random.Random(seed)
    draws = sorted(
        statistics.fmean(rng.choice(differences) for _ in differences)
        for _ in range(iterations)
    )
    alpha_each = (1.0 - confidence) / family_size
    lower_index = min(iterations - 1, max(0, math.floor(alpha_each * iterations)))
    return BootstrapContrast(
        treatment=treatment.arm,
        control=control.arm,
        estimate=statistics.fmean(differences),
        simultaneous_lower=draws[lower_index],
        confidence=confidence,
        paired_trajectory_ids=ids,
    )


def audit_equal_budget_ablation(
    receipts: tuple[ArmReceipt, ...],
    contract: EqualBudgetContract,
    *,
    bootstrap_iterations: int = 20_000,
    bootstrap_seed: int = 9,
) -> AblationAudit:
    """Audit opportunity fairness, sealing, and the preregistered paired tests."""

    required = set(AblationArm)
    arms = {receipt.arm for receipt in receipts}
    by_arm = {receipt.arm: receipt for receipt in receipts}
    all_trajectories = tuple(row for receipt in receipts for row in receipt.trajectories)
    trajectory_keys = [
        tuple((row.trajectory_id, row.seed) for row in receipt.trajectories)
        for receipt in receipts
    ]
    shape_counts = {len(row.heldout_log_speedups) for row in all_trajectories}
    seal_hashes = {row.heldout_seal_sha256 for row in all_trajectories}

    checks = {
        "all_arms_present_once": len(receipts) == len(required) and arms == required,
        "shared_experiment_envelope": len({r.shared_envelope for r in receipts}) == 1,
        "arm_protocols_are_separate": (
            len({r.arm_protocol_digest for r in receipts}) == len(receipts)
            and all(r.arm_protocol_digest for r in receipts)
        ),
        "paired_trajectory_ids_and_seeds": (
            bool(trajectory_keys)
            and all(len(keys) == len(set(keys)) for keys in trajectory_keys)
            and len(set(trajectory_keys)) == 1
        ),
        "trajectory_budget_equal": all(
            len(receipt.trajectories) == contract.trajectories for receipt in receipts
        ),
        "common_nonempty_heldout_shape_count": len(shape_counts) == 1 and shape_counts != {0},
        "configured_caps_match_contract": all(
            receipt.proposal_token_cap == contract.proposal_tokens_per_arm
            and receipt.proposal_slot_cap == contract.proposal_slots_per_arm
            and receipt.screen_slot_cap == contract.screen_slots_per_arm
            and receipt.qualification_slot_cap == contract.qualification_slots_per_arm
            and receipt.gpu_second_cap == contract.gpu_seconds_per_arm
            for receipt in receipts
        ),
        "token_cap_respected": all(
            receipt.proposal_tokens_actual <= receipt.proposal_token_cap
            for receipt in receipts
        ),
        "proposal_slot_cap_respected": all(
            receipt.proposal_slots_charged <= receipt.proposal_slot_cap
            for receipt in receipts
        ),
        "screen_charged_vs_actual_valid": all(
            receipt.screen_attempts_actual <= receipt.screen_slots_charged
            <= receipt.screen_slot_cap
            for receipt in receipts
        ),
        "qualification_charged_vs_actual_valid": all(
            receipt.qualification_attempts_actual <= receipt.qualification_slots_charged
            <= receipt.qualification_slot_cap
            for receipt in receipts
        ),
        "qualification_attempts_do_not_exceed_screens": all(
            receipt.qualification_attempts_actual <= receipt.screen_attempts_actual
            for receipt in receipts
        ),
        "unique_candidates_do_not_exceed_proposals": all(
            receipt.unique_candidates <= receipt.proposal_slots_charged
            for receipt in receipts
        ),
        "screen_attempts_do_not_exceed_unique_candidates": all(
            receipt.screen_attempts_actual <= receipt.unique_candidates
            for receipt in receipts
        ),
        "gpu_cap_respected": all(
            receipt.gpu_seconds_actual
            <= receipt.gpu_second_cap * (1.0 + contract.gpu_balance_tolerance_fraction)
            for receipt in receipts
        ),
        "single_shared_heldout_seal": (
            len(seal_hashes) == 1 and bool(seal_hashes) and all(_valid_sha256(x) for x in seal_hashes)
        ),
        "winner_frozen_before_heldout_unseal": all(
            _parse_utc(row.winner_frozen_at_utc) < _parse_utc(row.heldout_unsealed_at_utc)
            for row in all_trajectories
        ),
    }

    contrasts: tuple[BootstrapContrast, ...] = ()
    if checks["all_arms_present_once"] and checks["paired_trajectory_ids_and_seeds"]:
        treatment = by_arm[AblationArm.MULTI_ROLE_CONCLUSIONS]
        controls = (
            by_arm[AblationArm.SINGLE_ROLEPLAY],
            by_arm[AblationArm.MULTI_HOMOGENEOUS],
            by_arm[AblationArm.MULTI_ROLE_NO_SHARE],
        )
        contrasts = tuple(
            paired_bootstrap_contrast(
                treatment,
                control,
                family_size=len(controls),
                iterations=bootstrap_iterations,
                seed=bootstrap_seed,
            )
            for control in controls
        )
    checks["simultaneous_paired_lower_bounds_positive"] = (
        len(contrasts) == 3 and all(item.simultaneous_lower > 0 for item in contrasts)
    )
    checks["cross_lineage_compound_candidate_present"] = (
        AblationArm.MULTI_ROLE_CONCLUSIONS in by_arm
        and by_arm[AblationArm.MULTI_ROLE_CONCLUSIONS].compound_candidates > 0
    )
    reasons = tuple(name for name, passed in checks.items() if not passed)
    return AblationAudit(not reasons, tuple(checks.items()), reasons, contrasts)


def multi_agent_advantage(
    receipts: tuple[ArmReceipt, ...], contract: EqualBudgetContract
) -> tuple[bool, str]:
    """Require joint gains beyond roleplay, best-of-N, and no-sharing controls."""

    audit = audit_equal_budget_ablation(receipts, contract)
    if not audit.claim_ready:
        return False, "ablation_not_claim_ready: " + ",".join(audit.reasons)
    return True, "multi_role_conclusions_beats_all_controls_with_simultaneous_bounds"
