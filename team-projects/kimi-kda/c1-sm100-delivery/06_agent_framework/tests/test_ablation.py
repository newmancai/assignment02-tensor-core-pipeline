from __future__ import annotations

from dataclasses import replace
import unittest

from kda_ir import (
    AblationArm,
    ArmReceipt,
    EqualBudgetContract,
    SharedExperimentEnvelope,
    TrajectoryReceipt,
    audit_equal_budget_ablation,
    multi_agent_advantage,
)


SEAL = "a" * 64


class AblationContractTest(unittest.TestCase):
    def setUp(self):
        self.contract = EqualBudgetContract(3, 150000, 24, 24, 4, 60.0)
        self.envelope = SharedExperimentEnvelope(
            "model", "model-config", "new-heldout-split", "common-prompt", "space", "tools"
        )
        metrics = {
            AblationArm.SINGLE_GENERALIST: (0.008, 0.009, 0.010),
            AblationArm.SINGLE_ROLEPLAY: (0.010, 0.011, 0.012),
            AblationArm.MULTI_HOMOGENEOUS: (0.011, 0.012, 0.013),
            AblationArm.MULTI_ROLE_NO_SHARE: (0.012, 0.013, 0.014),
            AblationArm.MULTI_ROLE_CONCLUSIONS: (0.020, 0.021, 0.022),
        }
        self.receipts = tuple(
            ArmReceipt(
                arm=arm,
                shared_envelope=self.envelope,
                arm_protocol_digest=f"protocol-{arm.value}",
                trajectories=tuple(
                    TrajectoryReceipt(
                        trajectory_id=f"trajectory-{index}",
                        seed=100 + index,
                        winner_candidate_id=f"{arm.value}-winner-{index}",
                        heldout_log_speedups=(metric, metric + 0.001),
                        heldout_seal_sha256=SEAL,
                        winner_frozen_at_utc="2026-09-09T01:00:00+00:00",
                        heldout_unsealed_at_utc="2026-09-09T02:00:00+00:00",
                    )
                    for index, metric in enumerate(values)
                ),
                proposal_token_cap=150000,
                proposal_tokens_actual=120000,
                proposal_slot_cap=24,
                proposal_slots_charged=24,
                unique_candidates=20,
                screen_slot_cap=24,
                screen_slots_charged=20,
                screen_attempts_actual=20,
                qualification_slot_cap=4,
                qualification_slots_charged=4,
                qualification_attempts_actual=4,
                gpu_second_cap=60.0,
                gpu_seconds_actual=55.0,
                compound_candidates=(
                    1 if arm is AblationArm.MULTI_ROLE_CONCLUSIONS else 0
                ),
            )
            for arm, values in metrics.items()
        )

    def test_ready_ablation_requires_all_controls_and_simultaneous_bounds(self):
        audit = audit_equal_budget_ablation(
            self.receipts, self.contract, bootstrap_iterations=1000
        )
        self.assertTrue(audit.claim_ready)
        self.assertEqual(len(audit.contrasts), 3)
        self.assertTrue(all(item.simultaneous_lower > 0 for item in audit.contrasts))
        advantage, reason = multi_agent_advantage(self.receipts, self.contract)
        self.assertTrue(advantage)
        self.assertIn("simultaneous", reason)

    def test_caps_are_fair_without_requiring_equal_actual_usage(self):
        altered = (replace(self.receipts[0], proposal_tokens_actual=1000),) + self.receipts[1:]
        audit = audit_equal_budget_ablation(altered, self.contract, bootstrap_iterations=100)
        self.assertTrue(audit.claim_ready)

    def test_token_cap_overrun_blocks_claim(self):
        altered = (replace(self.receipts[0], proposal_tokens_actual=150001),) + self.receipts[1:]
        audit = audit_equal_budget_ablation(altered, self.contract, bootstrap_iterations=100)
        self.assertFalse(audit.claim_ready)
        self.assertIn("token_cap_respected", audit.reasons)

    def test_duplicate_and_invalid_proposals_need_not_be_screened(self):
        altered = (
            replace(
                self.receipts[0],
                unique_candidates=8,
                screen_slots_charged=8,
                screen_attempts_actual=8,
            ),
        ) + self.receipts[1:]
        audit = audit_equal_budget_ablation(altered, self.contract, bootstrap_iterations=100)
        self.assertTrue(audit.claim_ready)

    def test_winner_must_freeze_before_heldout_unseal(self):
        trajectory = replace(
            self.receipts[0].trajectories[0],
            winner_frozen_at_utc="2026-09-09T03:00:00+00:00",
        )
        altered = (
            replace(
                self.receipts[0],
                trajectories=(trajectory,) + self.receipts[0].trajectories[1:],
            ),
        ) + self.receipts[1:]
        audit = audit_equal_budget_ablation(altered, self.contract, bootstrap_iterations=100)
        self.assertFalse(audit.claim_ready)
        self.assertIn("winner_frozen_before_heldout_unseal", audit.reasons)

    def test_no_share_control_is_required(self):
        audit = audit_equal_budget_ablation(self.receipts[:-2] + self.receipts[-1:], self.contract)
        self.assertFalse(audit.claim_ready)
        self.assertIn("all_arms_present_once", audit.reasons)

    def test_missing_compound_candidate_blocks_claim(self):
        altered = self.receipts[:-1] + (replace(self.receipts[-1], compound_candidates=0),)
        advantage, reason = multi_agent_advantage(altered, self.contract)
        self.assertFalse(advantage)
        self.assertIn("cross_lineage_compound_candidate_present", reason)

    def test_point_advantage_without_positive_paired_bound_blocks_claim(self):
        weak_rows = tuple(
            replace(row, heldout_log_speedups=(-0.020, -0.019))
            if index == 0
            else row
            for index, row in enumerate(self.receipts[-1].trajectories)
        )
        altered = self.receipts[:-1] + (replace(self.receipts[-1], trajectories=weak_rows),)
        audit = audit_equal_budget_ablation(altered, self.contract, bootstrap_iterations=1000)
        self.assertFalse(audit.claim_ready)
        self.assertIn("simultaneous_paired_lower_bounds_positive", audit.reasons)


if __name__ == "__main__":
    unittest.main()
