"""Budget and receipt contracts for falsifiable multi-agent ablations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AblationArm(str, Enum):
    SINGLE_GENERALIST = "single_generalist"
    SINGLE_ROLEPLAY = "single_roleplay"
    MULTI_HOMOGENEOUS = "multi_homogeneous"
    MULTI_ROLE_CONCLUSIONS = "multi_role_conclusions"


@dataclass(frozen=True)
class EqualBudgetContract:
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
class ArmReceipt:
    arm: AblationArm
    model_digest: str
    profile_split_digest: str
    prompt_protocol_digest: str
    candidate_space_digest: str
    trajectories: int
    proposal_tokens: int
    proposal_slots: int
    unique_candidates: int
    screen_slots: int
    qualification_slots: int
    gpu_seconds: float
    heldout_metric: float | None
    heldout_sealed_until_freeze: bool
    compound_candidates: int = 0


@dataclass(frozen=True)
class AblationAudit:
    claim_ready: bool
    checks: tuple[tuple[str, bool], ...]
    reasons: tuple[str, ...]


def audit_equal_budget_ablation(
    receipts: tuple[ArmReceipt, ...], contract: EqualBudgetContract
) -> AblationAudit:
    """Audit fairness and evidence needed before claiming multi-agent benefit."""

    required = set(AblationArm)
    arms = {receipt.arm for receipt in receipts}
    shared_fields = (
        "model_digest",
        "profile_split_digest",
        "prompt_protocol_digest",
        "candidate_space_digest",
    )
    checks = {
        "all_arms_present_once": len(receipts) == len(required) and arms == required,
        "same_model_profile_prompt_and_space": all(
            len({getattr(receipt, field) for receipt in receipts}) == 1
            for field in shared_fields
        ),
        "trajectory_budget_equal": all(
            receipt.trajectories == contract.trajectories for receipt in receipts
        ),
        "token_budget_not_exceeded": all(
            receipt.proposal_tokens <= contract.proposal_tokens_per_arm
            for receipt in receipts
        ),
        "proposal_slots_charged_equally": all(
            receipt.proposal_slots == contract.proposal_slots_per_arm
            for receipt in receipts
        ),
        "screen_slots_charged_equally": all(
            receipt.screen_slots == contract.screen_slots_per_arm for receipt in receipts
        ),
        "qualification_slots_charged_equally": all(
            receipt.qualification_slots == contract.qualification_slots_per_arm
            for receipt in receipts
        ),
        "unique_candidates_do_not_exceed_slots": all(
            receipt.unique_candidates <= receipt.proposal_slots for receipt in receipts
        ),
        "gpu_seconds_balanced": all(
            abs(receipt.gpu_seconds - contract.gpu_seconds_per_arm)
            <= contract.gpu_seconds_per_arm * contract.gpu_balance_tolerance_fraction
            for receipt in receipts
        ),
        "heldout_was_sealed": all(
            receipt.heldout_sealed_until_freeze for receipt in receipts
        ),
        "heldout_metric_present": all(
            receipt.heldout_metric is not None for receipt in receipts
        ),
    }
    reasons = tuple(name for name, passed in checks.items() if not passed)
    return AblationAudit(not reasons, tuple(checks.items()), reasons)


def multi_agent_advantage(
    receipts: tuple[ArmReceipt, ...], contract: EqualBudgetContract
) -> tuple[bool, str]:
    """Apply the minimum operational test for collaboration beyond roleplay/N."""

    audit = audit_equal_budget_ablation(receipts, contract)
    if not audit.claim_ready:
        return False, "ablation_not_claim_ready: " + ",".join(audit.reasons)
    by_arm = {receipt.arm: receipt for receipt in receipts}
    multi = by_arm[AblationArm.MULTI_ROLE_CONCLUSIONS]
    roleplay = by_arm[AblationArm.SINGLE_ROLEPLAY]
    homogeneous = by_arm[AblationArm.MULTI_HOMOGENEOUS]
    if multi.compound_candidates <= 0:
        return False, "no_cross_lineage_compound_candidate"
    if not (
        multi.heldout_metric > roleplay.heldout_metric
        and multi.heldout_metric > homogeneous.heldout_metric
    ):
        return False, "multi_role_does_not_beat_roleplay_and_homogeneous_controls"
    return True, "collaboration_advantage_candidate; statistical CI still required"
