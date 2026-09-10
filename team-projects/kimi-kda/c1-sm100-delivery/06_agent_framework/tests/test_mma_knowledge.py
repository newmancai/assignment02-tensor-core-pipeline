from dataclasses import replace
import unittest

from kda_ir.mma_knowledge import (
    ChallengeStatus,
    CounterfactualChallenge,
    KnowledgeDisposition,
    KnowledgeKind,
    KnowledgeMaturity,
    MigrationKnowledgeClaim,
    verify_knowledge_claim,
)


def direct_swap_claim() -> MigrationKnowledgeClaim:
    challenges = (
        CounterfactualChallenge("lifetime", "lifecycle_amortization", ("tmem_carriers", "protocols"), ("launch_topology", "rounding_dag"), ChallengeStatus.PLANNED),
        CounterfactualChallenge("thin_n", "parallelism_and_tile", ("launch_topology", "tiles"), ("per_site_lifecycle", "rounding_dag"), ChallengeStatus.PLANNED),
        CounterfactualChallenge("saturated", "mechanism_control", ("launch_topology.grid_ctas", "protocols.alloc_scope"), ("math_site", "tile", "rounding_dag"), ChallengeStatus.PLANNED),
    )
    return MigrationKnowledgeClaim(
        "direct-m128-negative", "The archived Phase-6 direct swap is not worth repeating.",
        KnowledgeKind.PERFORMANCE_DECISION, KnowledgeMaturity.SINGLE_PATH_OBSERVATION,
        KnowledgeDisposition.SCOPED_PRIOR, "b300:h12:phase6:m128n128:grid12:inner64:l0",
        ("03_tcgen05_probe_17937.csv",),
        ("allocation overhead", "grid underfill", "TMEM readback", "descriptor feed latency"),
        challenges, ("cross_phase_carrier", "grid96_thin_n", "target_or_toolchain_change"),
    )


class MmaKnowledgeTest(unittest.TestCase):
    def test_single_path_negative_is_only_a_scoped_prior(self):
        self.assertEqual(verify_knowledge_claim(direct_swap_claim()), ())

    def test_planned_paths_cannot_fake_multi_path_maturity(self):
        claim = replace(direct_swap_claim(), maturity=KnowledgeMaturity.MULTI_PATH_CORROBORATED)
        self.assertIn("TCG702", {x.code for x in verify_knowledge_claim(claim)})

    def test_performance_result_never_becomes_verifier_rule(self):
        claim = replace(direct_swap_claim(), disposition=KnowledgeDisposition.VERIFIER_RULE)
        self.assertIn("TCG704", {x.code for x in verify_knowledge_claim(claim)})


if __name__ == "__main__":
    unittest.main()
