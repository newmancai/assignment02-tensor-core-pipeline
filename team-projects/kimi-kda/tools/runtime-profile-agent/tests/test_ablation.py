from __future__ import annotations

from dataclasses import replace
import unittest

from kda_ir import (
    AblationArm,
    ArmReceipt,
    EqualBudgetContract,
    audit_equal_budget_ablation,
    multi_agent_advantage,
)


class AblationContractTest(unittest.TestCase):
    def setUp(self):
        self.contract = EqualBudgetContract(3, 150000, 24, 24, 4, 60.0)
        metrics = {
            AblationArm.SINGLE_GENERALIST: 0.010,
            AblationArm.SINGLE_ROLEPLAY: 0.012,
            AblationArm.MULTI_HOMOGENEOUS: 0.013,
            AblationArm.MULTI_ROLE_CONCLUSIONS: 0.018,
        }
        self.receipts = tuple(
            ArmReceipt(
                arm,
                "model",
                "split",
                "prompt",
                "space",
                3,
                149000,
                24,
                20,
                24,
                4,
                59.0,
                metric,
                True,
                1 if arm is AblationArm.MULTI_ROLE_CONCLUSIONS else 0,
            )
            for arm, metric in metrics.items()
        )

    def test_ready_ablation_requires_all_matched_controls(self):
        audit = audit_equal_budget_ablation(self.receipts, self.contract)
        self.assertTrue(audit.claim_ready)
        advantage, reason = multi_agent_advantage(self.receipts, self.contract)
        self.assertTrue(advantage)
        self.assertIn("statistical CI", reason)

    def test_extra_multi_agent_budget_blocks_claim(self):
        altered = self.receipts[:-1] + (
            replace(self.receipts[-1], proposal_slots=25, screen_slots=25),
        )
        audit = audit_equal_budget_ablation(altered, self.contract)
        self.assertFalse(audit.claim_ready)
        self.assertIn("proposal_slots_charged_equally", audit.reasons)

    def test_best_of_n_without_compound_candidate_is_not_emergence(self):
        altered = self.receipts[:-1] + (
            replace(self.receipts[-1], compound_candidates=0),
        )
        advantage, reason = multi_agent_advantage(altered, self.contract)
        self.assertFalse(advantage)
        self.assertEqual(reason, "no_cross_lineage_compound_candidate")


if __name__ == "__main__":
    unittest.main()
