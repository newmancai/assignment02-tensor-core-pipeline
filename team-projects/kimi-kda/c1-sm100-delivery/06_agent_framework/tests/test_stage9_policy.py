from __future__ import annotations

import unittest

from kda_ir import (
    LineageFragment,
    PolicyEvaluationContext,
    PolicyFeature,
    PolicyLeaf,
    PolicyNode,
    PolicyPredicate,
    PredicateOperator,
    RuntimePolicyCandidate,
    RuntimeWorkloadProfile,
    canonical_policy_id,
    evaluate_policy,
    policy_subtree_id,
    verify_compound_lineage,
    verify_policy,
)


def node(feature, threshold, then_branch, else_branch, operator=PredicateOperator.LE):
    return PolicyNode(
        PolicyPredicate(feature, operator, threshold),
        then_branch,
        else_branch,
    )


class Stage9PolicyTest(unittest.TestCase):
    def test_evaluates_public_profile_features(self):
        profile = RuntimeWorkloadProfile(
            num_heads=12,
            seq_lens=(1300, 547, 2048, 963),
            packed=True,
            sm_count=148,
        )
        context = PolicyEvaluationContext.from_profile(
            profile, resident_ctas_per_sm=5
        )
        candidate = RuntimePolicyCandidate(
            node(
                PolicyFeature.RESIDENT_GRID_CAPACITY_CTAS,
                700,
                PolicyLeaf(4),
                node(
                    PolicyFeature.MAX_CHUNKS,
                    127,
                    PolicyLeaf(9),
                    PolicyLeaf(12),
                ),
            )
        )
        self.assertEqual(context.resident_grid_capacity_ctas, 740)
        self.assertEqual(evaluate_policy(candidate, context), 12)

    def test_validator_enforces_depth_and_frozen_leaf_set(self):
        too_deep = RuntimePolicyCandidate(
            node(
                PolicyFeature.TOTAL_CHUNKS,
                128,
                node(
                    PolicyFeature.MAX_CHUNKS,
                    64,
                    node(
                        PolicyFeature.NUM_SEQUENCES,
                        1,
                        PolicyLeaf(7),
                        PolicyLeaf(8),
                    ),
                    PolicyLeaf(9),
                ),
                PolicyLeaf(12),
            )
        )
        codes = {item.code for item in verify_policy(too_deep)}
        self.assertIn("KIR902", codes)
        self.assertIn("KIR903", codes)

    def test_canonical_id_ignores_agent_provenance(self):
        root = node(
            PolicyFeature.TOTAL_CHUNKS, 512, PolicyLeaf(9), PolicyLeaf(17)
        )
        first = RuntimePolicyCandidate(root, origin_proposal_ids=("scout-a",))
        second = RuntimePolicyCandidate(root, origin_proposal_ids=("scout-b",))
        self.assertEqual(canonical_policy_id(first), canonical_policy_id(second))
        self.assertEqual(len(canonical_policy_id(first).removeprefix("kir-policy:")), 64)
        self.assertEqual(len(policy_subtree_id(root).removeprefix("kir-policy-subtree:")), 64)

    def test_compound_lineage_requires_distinct_inherited_fragments(self):
        left_leaf = PolicyLeaf(4)
        right_leaf = PolicyLeaf(12)
        left = RuntimePolicyCandidate(
            node(PolicyFeature.TOTAL_CHUNKS, 256, left_leaf, PolicyLeaf(9))
        )
        right = RuntimePolicyCandidate(
            node(PolicyFeature.MAX_CHUNKS, 64, right_leaf, PolicyLeaf(16))
        )
        left_id = canonical_policy_id(left)
        right_id = canonical_policy_id(right)
        child = RuntimePolicyCandidate(
            node(PolicyFeature.NUM_SEQUENCES, 4, left_leaf, right_leaf),
            parent_ids=(left_id, right_id),
            origin_proposal_ids=("profile-1", "resource-1"),
            lineage_fragments=(
                LineageFragment(left_id, policy_subtree_id(left_leaf)),
                LineageFragment(right_id, policy_subtree_id(right_leaf)),
            ),
        )
        self.assertEqual(
            verify_compound_lineage(child, {left_id: left, right_id: right}), ()
        )

        unproved = RuntimePolicyCandidate(
            child.root,
            parent_ids=child.parent_ids,
            lineage_fragments=(
                LineageFragment(left_id, policy_subtree_id(left_leaf)),
            ),
        )
        codes = {
            item.code
            for item in verify_compound_lineage(
                unproved, {left_id: left, right_id: right}
            )
        }
        self.assertIn("KIR914", codes)
        self.assertIn("KIR917", codes)


if __name__ == "__main__":
    unittest.main()
