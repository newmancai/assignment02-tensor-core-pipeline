from __future__ import annotations

import unittest

from kda_ir import (
    B300,
    ActivationEvidence,
    ActivationStatus,
    AgentContribution,
    AgentRole,
    Proposal,
    RuntimeWorkloadProfile,
    activation_decision,
    break_token_order,
    cake_bt16_prepare_chain,
    coordinate_contributions,
    prepare_grid_geometry,
    recommend_one_resident_wave_cpc,
    set_prepare_chunks_per_cta,
)


class RuntimeProfileTest(unittest.TestCase):
    def test_profile_exposes_prepare_and_chain_geometry(self):
        profile = RuntimeWorkloadProfile(
            num_heads=12,
            seq_lens=(1300, 547, 2048, 963, 271, 3063),
            packed=True,
            sm_count=148,
        )
        self.assertEqual(profile.total_chunks, 515)
        self.assertEqual(profile.max_chunks, 192)
        self.assertAlmostEqual(profile.effective_sequence_parallelism, 515 / 192)
        baseline = prepare_grid_geometry(profile, 4)
        candidate = prepare_grid_geometry(profile, 9)
        self.assertEqual((baseline.rectangular_ctas, baseline.waves), (1548, 11))
        self.assertEqual((candidate.rectangular_ctas, candidate.waves), (696, 5))
        self.assertFalse(candidate.wave_quantized)

    def test_profile_rejects_invalid_geometry(self):
        with self.assertRaises(ValueError):
            RuntimeWorkloadProfile(12, (), True, 148)
        profile = RuntimeWorkloadProfile(12, (512,), False, 148)
        with self.assertRaises(ValueError):
            prepare_grid_geometry(profile, 0)

    def test_resident_wave_policy_scales_with_total_work(self):
        expected = {256: 5, 512: 9, 1024: 17}
        for total_chunks, cpc in expected.items():
            profile = RuntimeWorkloadProfile(
                num_heads=12,
                seq_lens=(total_chunks * 16,),
                packed=False,
                sm_count=148,
            )
            policy = recommend_one_resident_wave_cpc(profile, 5)
            self.assertEqual(policy.chunks_per_cta, cpc)
            self.assertLessEqual(policy.rectangular_ctas, 148 * 5)
            self.assertLessEqual(policy.occupancy_aware_waves, 1.0)

    def test_resident_wave_policy_rejects_impossible_or_invalid_capacity(self):
        profile = RuntimeWorkloadProfile(1000, (4096,), False, 1)
        with self.assertRaises(ValueError):
            recommend_one_resident_wave_cpc(profile, 0)
        with self.assertRaises(ValueError):
            recommend_one_resident_wave_cpc(profile, 1)


class MultiAgentCoordinationTest(unittest.TestCase):
    def test_independent_proposals_merge_only_after_verification(self):
        candidate = set_prepare_chunks_per_cta(cake_bt16_prepare_chain(), 9)
        contributions = (
            AgentContribution(
                "profile-1",
                AgentRole.PROFILE_ANALYST,
                Proposal(candidate, "reduce long-profile prepare waves"),
                "B300 long H12 profiles have excess prepare CTAs",
                ("evidence/b300_stage5_h12_cpc_heldout.json",),
            ),
            AgentContribution(
                "explorer-1",
                AgentRole.SCHEDULE_EXPLORER,
                Proposal(candidate, "refined typed work assignment"),
                "cpc=9 is verifier-clean",
                ("evidence/b300_stage4_h12_cpc_refine.json",),
            ),
            AgentContribution(
                "unsafe-1",
                AgentRole.SAFETY_CRITIC,
                Proposal(break_token_order(candidate), "adversarial recurrence rewrite"),
                "test that consensus cannot bypass legality",
            ),
        )
        result = coordinate_contributions(contributions, B300)
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(result.candidates[0].proposal_ids, ("profile-1", "explorer-1"))
        self.assertEqual(len(result.rejected), 1)
        self.assertEqual(result.rejected[0].diagnostics[0].code, "KIR404")

    def test_activation_is_evidence_gated_not_vote_gated(self):
        incomplete = activation_decision(
            ActivationEvidence(
                verifier_clean=True,
                output_and_state_correct=True,
                scope_identical=True,
                bootstrap_95_lower=0.999,
                fallback_defined=True,
            )
        )
        self.assertEqual(incomplete.status, ActivationStatus.MEASURE)
        complete = activation_decision(
            ActivationEvidence(
                verifier_clean=True,
                output_and_state_correct=True,
                scope_identical=True,
                bootstrap_95_lower=1.034,
                fallback_defined=True,
            )
        )
        self.assertEqual(complete.status, ActivationStatus.ACTIVATE)
        invalid = activation_decision(ActivationEvidence(verifier_clean=False))
        self.assertEqual(invalid.status, ActivationStatus.REJECT)


if __name__ == "__main__":
    unittest.main()
