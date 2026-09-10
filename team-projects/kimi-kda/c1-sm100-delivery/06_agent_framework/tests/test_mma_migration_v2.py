from dataclasses import replace
import unittest

from kda_ir.mma_migration import (
    LogicalPhysicalTransform,
    canonical_candidate_id,
    verify_mma_migration,
)
from kda_ir.mma_migration_fixtures import p34_shared_lifecycle_candidate


class MmaMigrationV2Test(unittest.TestCase):
    def test_p34_fixture_freezes_grid_baseline_chunk_and_rounding_boundary(self):
        candidate = p34_shared_lifecycle_candidate()

        self.assertEqual(verify_mma_migration(candidate), ())
        self.assertEqual(len(candidate.math_sites), 2)
        self.assertEqual(candidate.math_sites[0].logical_mnk, (16, 128, 16))
        self.assertEqual(candidate.tiles[0].physical_mnk, (128, 16, 16))
        self.assertIs(
            candidate.tiles[0].transform,
            LogicalPhysicalTransform.TRANSPOSE_OUTPUT_SWAP_OPERANDS,
        )
        self.assertEqual(candidate.launch_topology.grid_ctas, 12)
        self.assertEqual(candidate.chunk, 16)
        self.assertTrue(candidate.preserves_rounding_dag)
        self.assertEqual(
            candidate.comparison_baseline_id,
            "sha256:f8c8471d660821234c0a16c1cc62e5c4472c03efab00792318d1b3741efa4eed",
        )
        self.assertEqual(
            candidate.capability_table_sha256,
            "c123aa09d551f635629a6219617991410b371ef6f2802ee4b63e2ac11bcbc529",
        )
        self.assertEqual(candidate.tmem_carriers[0].dtype, "bf16")
        self.assertTrue(candidate.tmem_carriers[0].preserves_rounding_boundary)
        self.assertRegex(canonical_candidate_id(candidate), r"^mma256:[0-9a-f]{64}$")

    def test_same_columns_are_legal_when_event_lifetimes_do_not_overlap(self):
        candidate = p34_shared_lifecycle_candidate()
        left, right = candidate.protocols

        self.assertEqual(
            (left.tmem_column_begin, left.tmem_column_end),
            (right.tmem_column_begin, right.tmem_column_end),
        )
        self.assertLess(left.live_until, right.live_from)
        self.assertNotIn("TCG208", {item.code for item in verify_mma_migration(candidate)})

    def test_same_columns_are_rejected_when_event_lifetimes_really_overlap(self):
        candidate = p34_shared_lifecycle_candidate()
        left, right = candidate.protocols
        overlapping = replace(right, live_from=left.live_until)
        broken = replace(candidate, protocols=(left, overlapping))

        self.assertIn("TCG208", {item.code for item in verify_mma_migration(broken)})

    def test_v2_rejects_weak_comparison_baseline_id(self):
        broken = replace(p34_shared_lifecycle_candidate(), comparison_baseline_id="baseline-grid12")

        self.assertIn("TCG602", {item.code for item in verify_mma_migration(broken)})

    def test_carrier_cannot_erase_the_frozen_rounding_boundary(self):
        candidate = p34_shared_lifecycle_candidate()
        carrier = replace(candidate.tmem_carriers[0], preserves_rounding_boundary=False)
        broken = replace(candidate, tmem_carriers=(carrier,))

        self.assertIn("TCG212", {item.code for item in verify_mma_migration(broken)})

    def test_transpose_swap_transform_must_prove_the_physical_shape(self):
        candidate = p34_shared_lifecycle_candidate()
        broken_tile = replace(candidate.tiles[0], physical_mnk=(64, 16, 16))
        broken = replace(candidate, tiles=(broken_tile, candidate.tiles[1]))

        self.assertIn("TCG102", {item.code for item in verify_mma_migration(broken)})


if __name__ == "__main__":
    unittest.main()
