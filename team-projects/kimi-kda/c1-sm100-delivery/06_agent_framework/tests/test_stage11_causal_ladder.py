from dataclasses import replace
import unittest

from kda_ir.stage11_causal_ladder import (
    FactorState,
    KnowledgeMaturity,
    validate_causal_ladder,
)
from stage11_preregister_causal_ladder import build_preregistration


class Stage11CausalLadderTest(unittest.TestCase):
    def test_frozen_ladder_is_valid_and_complete(self):
        prereg = build_preregistration()
        self.assertEqual(validate_causal_ladder(prereg), ())
        thin_n = next(path for path in prereg.paths if path.factor == "thin_n")
        self.assertEqual(
            thin_n.changed_subtrees,
            (
                "tiles.phase6.physical_mnk_m128n16k16",
                "launch.value_columns_v16_eight_slices",
                "launch.grid_h12_x_8_equals_96",
            ),
        )
        self.assertEqual(
            {(row.lifecycle, row.thin_n) for row in prereg.candidates},
            {
                (FactorState.BASELINE, FactorState.BASELINE),
                (FactorState.MUTATED, FactorState.BASELINE),
                (FactorState.BASELINE, FactorState.MUTATED),
                (FactorState.MUTATED, FactorState.MUTATED),
            },
        )
        self.assertEqual(
            [item.value for item in KnowledgeMaturity],
            [
                "attempt",
                "single_path_observation",
                "multi_path_corroborated",
                "causal_mechanism",
                "transfer_validated",
            ],
        )

    def test_single_path_cannot_cancel_interaction(self):
        prereg = build_preregistration()
        broken = replace(
            prereg,
            knowledge_policy=tuple(
                item for item in prereg.knowledge_policy
                if item != "A_or_B_negative_does_not_cancel_AB"
            ),
        )
        self.assertIn("S11L031", {item.code for item in validate_causal_ladder(broken)})

    def test_baseline_must_resolve_to_a_frozen_receipt(self):
        prereg = build_preregistration()
        broken = replace(prereg, strongest_baseline="optimized_sm80")

        self.assertIn("S11L008", {item.code for item in validate_causal_ladder(broken)})

    def test_cake_full_stack_cannot_be_used_for_attribution(self):
        prereg = build_preregistration()
        candidate = prereg.candidates[1]
        broken_candidate = replace(
            candidate,
            attribution_sources=candidate.attribution_sources + ("cake_full_stack_result",),
        )
        broken = replace(
            prereg,
            candidates=(prereg.candidates[0], broken_candidate, *prereg.candidates[2:]),
        )
        self.assertIn("S11L015", {item.code for item in validate_causal_ladder(broken)})

    def test_interaction_waits_for_both_independent_paths(self):
        prereg = build_preregistration()
        interaction = replace(prereg.candidates[3], available_after=("cell_10_lifecycle",))
        broken = replace(prereg, candidates=(*prereg.candidates[:3], interaction))
        self.assertIn("S11L023", {item.code for item in validate_causal_ladder(broken)})


if __name__ == "__main__":
    unittest.main()
