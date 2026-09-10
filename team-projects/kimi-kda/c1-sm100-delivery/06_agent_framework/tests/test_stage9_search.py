from __future__ import annotations

import unittest

from kda_ir.stage9_policy import (
    PolicyFeature,
    PolicyLeaf,
    PolicyNode,
    PolicyPredicate,
    PredicateOperator,
    RuntimePolicyCandidate,
    canonical_policy_id,
)
from kda_ir.stage9_search import (
    MeasuredPolicy,
    SearchLane,
    deterministic_mutations,
    elite_archive,
    lane_violations,
    policy_niche,
    threshold_catalog,
)


def branch(feature, threshold, then_cpc, else_cpc):
    return RuntimePolicyCandidate(
        PolicyNode(
            PolicyPredicate(feature, PredicateOperator.LE, threshold),
            PolicyLeaf(then_cpc),
            PolicyLeaf(else_cpc),
        )
    )


class Stage9SearchTest(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"total_chunks": 256, "max_chunks": 67, "num_sequences": 4,
             "resident_grid_capacity_ctas": 740},
            {"total_chunks": 512, "max_chunks": 512, "num_sequences": 1,
             "resident_grid_capacity_ctas": 740},
            {"total_chunks": 1024, "max_chunks": 262, "num_sequences": 4,
             "resident_grid_capacity_ctas": 740},
        ]

    def test_role_names_are_enforced_by_policy_semantics(self):
        wave = branch(PolicyFeature.TOTAL_CHUNKS, 512, 5, 9)
        tail = branch(PolicyFeature.MAX_CHUNKS, 262, 5, 9)
        self.assertEqual(lane_violations(SearchLane.WAVE_CAPACITY, wave), ())
        self.assertTrue(lane_violations(SearchLane.TAIL_SKEW, wave))
        self.assertEqual(lane_violations(SearchLane.TAIL_SKEW, tail), ())

    def test_mixed_lane_requires_two_physical_feature_families(self):
        mixed = RuntimePolicyCandidate(
            PolicyNode(
                PolicyPredicate(PolicyFeature.TOTAL_CHUNKS, PredicateOperator.LE, 512),
                PolicyLeaf(5),
                PolicyNode(
                    PolicyPredicate(PolicyFeature.NUM_SEQUENCES, PredicateOperator.GT, 2),
                    PolicyLeaf(9),
                    PolicyLeaf(17),
                ),
            )
        )
        self.assertEqual(lane_violations(SearchLane.MIXED_INTERACTION, mixed), ())

    def test_thresholds_come_from_observed_values_and_midpoints(self):
        catalog = threshold_catalog(self.rows)
        self.assertEqual(catalog[PolicyFeature.TOTAL_CHUNKS], (256, 384, 512, 768, 1024))

    def test_deterministic_mutations_are_unique_and_lane_clean(self):
        parent = RuntimePolicyCandidate(
            PolicyNode(
                PolicyPredicate(PolicyFeature.TOTAL_CHUNKS, PredicateOperator.LE, 384),
                PolicyLeaf(5),
                PolicyNode(
                    PolicyPredicate(PolicyFeature.TOTAL_CHUNKS, PredicateOperator.LE, 768),
                    PolicyLeaf(9),
                    PolicyLeaf(17),
                ),
            )
        )
        mutations = deterministic_mutations(parent, self.rows, lane=SearchLane.WAVE_CAPACITY)
        ids = [canonical_policy_id(item.candidate) for item in mutations]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertGreaterEqual(len(ids), 3)
        self.assertTrue(all(not lane_violations(SearchLane.WAVE_CAPACITY, item.candidate) for item in mutations))
        for lane in (SearchLane.TAIL_SKEW, SearchLane.MIXED_INTERACTION):
            projected = deterministic_mutations(parent, self.rows, lane=lane)
            self.assertTrue(projected)
            self.assertTrue(all(not lane_violations(lane, item.candidate) for item in projected))

    def test_archive_keeps_best_correct_candidate_per_niche(self):
        slow = branch(PolicyFeature.TOTAL_CHUNKS, 384, 5, 9)
        fast = branch(PolicyFeature.TOTAL_CHUNKS, 512, 5, 9)
        tail = branch(PolicyFeature.MAX_CHUNKS, 262, 5, 9)
        archive = elite_archive((
            MeasuredPolicy(slow, 0.01, True),
            MeasuredPolicy(fast, 0.02, True),
            MeasuredPolicy(tail, 0.015, True),
        ))
        self.assertEqual(len(archive), 2)
        self.assertEqual(archive[policy_niche(fast)].candidate, fast)


if __name__ == "__main__":
    unittest.main()
