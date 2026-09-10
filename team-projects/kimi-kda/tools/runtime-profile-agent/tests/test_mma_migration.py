from __future__ import annotations

import unittest

from kda_ir import (
    KernelMeasurement,
    MigrationAction,
    MmaMigrationProfile,
    assess_mma_migration,
)


def measured_profile(**overrides):
    values = dict(
        profile_id="b300-h12-t8192",
        target_arch="sm_103a",
        sm_count=148,
        local_heads=12,
        hmmas_in_recurrence_sass=3640,
        tcgen_in_recurrence_sass=0,
        official=KernelMeasurement("official_v128", 12, 1270.0, 2.64, 1.24, 2.48, "ncu.csv"),
        value_rows=128,
        value_rows_independent=True,
        dominant_phase_m=16,
        current_mma_atom_m=16,
        tcgen_min_m=64,
        phase6_m=128,
        tcgen_l0_v128_speedup=0.919742,
        tcgen_l1_v128_speedup=0.256278,
        tcgen_l0_v16_speedup=1.500761,
        tcgen_l1_v16_speedup=0.777985,
        tcgen_l0_v128_mma_active_blocks_per_sm=12,
        tcgen_l0_v128_active_blocks_per_sm=1,
        tcgen_l1_v128_mma_active_blocks_per_sm=5,
        tcgen_l1_v128_active_blocks_per_sm=1,
        tcgen_l1_v128_mma_static_smem_bytes=41472,
        tcgen_l1_v128_static_smem_bytes=45580,
        tcgen_l1_v128_mma_registers=38,
        tcgen_l1_v128_registers=39,
        sass_evidence_ref="sass.csv",
        sass_evidence_sha256="a" * 64,
        semantic_evidence_ref="kernel.cu",
        semantic_evidence_sha256="b" * 64,
        tcgen_evidence_ref="tcgen.csv",
        tcgen_evidence_sha256="c" * 64,
        ncu_evidence_sha256="d" * 64,
    )
    values.update(overrides)
    return MmaMigrationProfile(**values)


class MmaMigrationAssessmentTest(unittest.TestCase):
    def test_measured_c1_profile_answers_before_proposing(self):
        result = assess_mma_migration(measured_profile())
        self.assertIn("Do not mechanically migrate", result.answer)
        self.assertEqual(
            [item.code for item in result.findings],
            ["MMA001", "MMA002", "MMA003", "MMA004", "MMA005", "MMA006", "MMA007"],
        )
        self.assertEqual(result.proposals[0].proposal_id, "mma-existing-path-parallelism")
        self.assertEqual(result.proposals[0].action, MigrationAction.KEEP)
        self.assertEqual(result.proposals[-1].action, MigrationAction.STOP)

    def test_core_only_tcgen_signal_keeps_cross_phase_candidate_measurable(self):
        result = assess_mma_migration(measured_profile())
        cross_phase = next(item for item in result.proposals if item.proposal_id == "tcgen05-cross-phase-residency")
        self.assertEqual(cross_phase.action, MigrationAction.MEASURE)

    def test_unconfirmed_instruction_path_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "HMMA-only"):
            assess_mma_migration(measured_profile(tcgen_in_recurrence_sass=1))

    def test_missing_parallelism_evidence_does_not_overclaim(self):
        result = assess_mma_migration(measured_profile(value_rows_independent=False))
        self.assertIn("insufficient", result.answer)

    def test_shape_mismatch_does_not_overclaim(self):
        result = assess_mma_migration(measured_profile(dominant_phase_m=64))
        self.assertIn("insufficient", result.answer)

    def test_missing_residency_penalty_does_not_overclaim(self):
        result = assess_mma_migration(
            measured_profile(
                tcgen_l0_v128_active_blocks_per_sm=12,
                tcgen_l1_v128_active_blocks_per_sm=5,
            )
        )
        self.assertIn("insufficient", result.answer)


if __name__ == "__main__":
    unittest.main()
