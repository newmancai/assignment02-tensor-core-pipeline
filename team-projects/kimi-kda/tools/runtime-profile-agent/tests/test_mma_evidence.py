from __future__ import annotations

import unittest
from pathlib import Path

from kda_ir import build_mma_profile_from_evidence


ROOT = Path(__file__).resolve().parents[3]


class MmaEvidenceTest(unittest.TestCase):
    def test_archived_receipts_build_expected_physical_profile(self):
        profile = build_mma_profile_from_evidence(
            sass_path=ROOT / "experiments/data/sass_opcode_summary.csv",
            ncu_summary_path=ROOT
            / "experiments/final_campaign/data/raw/05_targeted_ncu_summary_17965.csv",
            tcgen_path=ROOT
            / "experiments/final_campaign/data/raw/03_tcgen05_probe_17937.csv",
            semantic_path=ROOT / "experiments/final_campaign/analysis_tcgen05.md",
        )

        self.assertEqual(profile.hmmas_in_recurrence_sass, 3640)
        self.assertEqual(profile.tcgen_in_recurrence_sass, 0)
        self.assertEqual(profile.official.duration_us, 1270.0)
        self.assertEqual(profile.official.recurrence_ctas, 12)
        self.assertAlmostEqual(profile.tcgen_l0_v128_speedup, 0.919742)
        self.assertAlmostEqual(profile.tcgen_l1_v128_speedup, 0.256278)
        self.assertAlmostEqual(profile.tcgen_l0_v16_speedup, 1.500761)
        self.assertAlmostEqual(profile.tcgen_l1_v16_speedup, 0.777985)
        self.assertEqual(
            (
                profile.tcgen_l0_v128_mma_active_blocks_per_sm,
                profile.tcgen_l0_v128_active_blocks_per_sm,
                profile.tcgen_l1_v128_mma_active_blocks_per_sm,
                profile.tcgen_l1_v128_active_blocks_per_sm,
            ),
            (12, 1, 5, 1),
        )
        self.assertEqual(
            (
                profile.tcgen_l1_v128_mma_static_smem_bytes,
                profile.tcgen_l1_v128_static_smem_bytes,
                profile.tcgen_l1_v128_mma_registers,
                profile.tcgen_l1_v128_registers,
            ),
            (41472, 45580, 38, 39),
        )
        self.assertEqual(len(profile.sass_evidence_sha256), 64)
        self.assertEqual(len(profile.ncu_evidence_sha256), 64)
        self.assertEqual(len(profile.tcgen_evidence_sha256), 64)
        self.assertEqual(len(profile.semantic_evidence_sha256), 64)


if __name__ == "__main__":
    unittest.main()
