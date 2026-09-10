from dataclasses import replace
import unittest

from kda_ir.mma_migration import (
    ConclusionState,
    FailureClass,
    InstructionSpec,
    LogicalPhysicalTile,
    MathSiteContract,
    MemorySpace,
    MmaFamily,
    MigrationCandidate,
    MigrationFailureCard,
    MigrationTier,
    ResourceContract,
    Tcgen05Protocol,
    canonical_candidate_id,
    verify_failure_card,
    verify_mma_migration,
)


def direct_swap() -> MigrationCandidate:
    site = MathSiteContract("k2_phase6", (128, 128, 16), "bf16", "f32", "bf16", ("k", "k"), "a" * 64, "fwd_kernel2.cuh:659")
    return MigrationCandidate(
        "typed_mma_migration_v1", "phase6-direct", MigrationTier.INSTRUCTION_ONLY,
        "b" * 64, "c" * 64, "sm_103a", 16, (site,),
        (InstructionSpec(site.site_id, MmaFamily.SM100_TCGEN05, "f16bf16.ss.m128n128k16", (128, 128, 16), MemorySpace.TMEM),),
        (LogicalPhysicalTile(site.site_id, site.logical_mnk, site.logical_mnk, False, False),),
        (Tcgen05Protocol(site.site_id, 1, "warp0", "warp0", 0, 128, True, True, True, True, True),),
        ResourceContract(192, 39, 45580, 128, 1), True, True,
        mutation_paths=("instructions.k2_phase6", "protocols.k2_phase6"),
    )


class MmaMigrationIrTest(unittest.TestCase):
    def test_valid_direct_swap_has_stable_full_id(self):
        candidate = direct_swap()
        self.assertEqual(verify_mma_migration(candidate), ())
        self.assertEqual(canonical_candidate_id(candidate), canonical_candidate_id(candidate))
        self.assertEqual(len(canonical_candidate_id(candidate)), 71)

    def test_instruction_only_cannot_hide_algorithm_change(self):
        broken = replace(direct_swap(), chunk=32, preserves_rounding_dag=False)
        self.assertIn("TCG301", {item.code for item in verify_mma_migration(broken)})

    def test_tcgen_requires_complete_tmem_lifecycle(self):
        protocol = replace(direct_swap().protocols[0], commit_has_wait=False)
        broken = replace(direct_swap(), protocols=(protocol,))
        self.assertIn("TCG204", {item.code for item in verify_mma_migration(broken)})

    def test_padding_requires_zero_fill_and_store_suppression(self):
        tile = replace(direct_swap().tiles[0], physical_mnk=(128, 256, 16))
        broken = replace(direct_swap(), tiles=(tile,))
        self.assertIn("TCG205", {item.code for item in verify_mma_migration(broken)})

    def test_scoped_negative_is_reusable_but_inconclusive_is_not(self):
        candidate_id = canonical_candidate_id(direct_swap())
        card = MigrationFailureCard(
            "tcgen05-phase6-direct-swap", candidate_id, ConclusionState.SLOWER,
            FailureClass.NO_PRACTICAL_GAIN, "observed", "b300-sm103a:v128:grid12:inner64:l0",
            ("B300 SM103a", "Phase-6 m128n128k16", "inner=64", "L0"), 0.919742,
            ("03_tcgen05_probe_17937.csv",), ("d" * 64,),
            "Protocol-amortized tcgen05 remained slower than mma.sync.",
            "Other phases, GPUs, layouts, or cross-phase TMEM residency.",
            ("cross-phase TMEM residency removes reformat/readback", "new target or toolchain"),
        )
        self.assertEqual(verify_failure_card(card), ())
        self.assertIn(
            "TCG502",
            {item.code for item in verify_failure_card(replace(card, state=ConclusionState.INCONCLUSIVE))},
        )


if __name__ == "__main__":
    unittest.main()
