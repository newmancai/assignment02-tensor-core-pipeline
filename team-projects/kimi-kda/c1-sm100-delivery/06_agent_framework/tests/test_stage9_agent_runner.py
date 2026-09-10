from __future__ import annotations

import json
import unittest

from kda_ir import (
    canonical_policy_id, evaluate_policy, PolicyEvaluationContext,
    PolicyLeaf, RuntimePolicyCandidate,
)
from run_stage9_agents import _parse_flat_policy, _tool_events


def leaf(cpc: int) -> dict:
    return {
        "kind": "leaf",
        "feature": "none",
        "cmp": "none",
        "threshold": 0,
        "cpc": cpc,
        "then_index": -1,
        "else_index": -1,
    }


class Stage9AgentRunnerTest(unittest.TestCase):
    def test_flat_tree_parses_to_typed_policy(self):
        raw = {
            "root_index": 0,
            "nodes": [
                {
                    "kind": "branch",
                    "feature": "total_chunks",
                    "cmp": "le",
                    "threshold": 512,
                    "cpc": 0,
                    "then_index": 1,
                    "else_index": 2,
                },
                leaf(9),
                leaf(17),
            ],
            "parent_policy_ids": [],
        }
        candidate = _parse_flat_policy(raw, origin_id="p0", parents={})
        self.assertTrue(canonical_policy_id(candidate).startswith("kir-policy:"))
        self.assertEqual(
            evaluate_policy(candidate, PolicyEvaluationContext(512, 512, 1, 740)),
            9,
        )
        self.assertEqual(
            evaluate_policy(candidate, PolicyEvaluationContext(1024, 1024, 1, 740)),
            17,
        )

    def test_cycle_and_unreachable_nodes_are_rejected(self):
        cyclic = {
            "root_index": 0,
            "nodes": [
                {
                    "kind": "branch",
                    "feature": "total_chunks",
                    "cmp": "le",
                    "threshold": 512,
                    "cpc": 0,
                    "then_index": 0,
                    "else_index": 1,
                },
                leaf(9),
            ],
            "parent_policy_ids": [],
        }
        with self.assertRaisesRegex(ValueError, "cycle"):
            _parse_flat_policy(cyclic, origin_id="p0", parents={})
        unreachable = {"root_index": 0, "nodes": [leaf(9), leaf(17)], "parent_policy_ids": []}
        with self.assertRaisesRegex(ValueError, "unreachable"):
            _parse_flat_policy(unreachable, origin_id="p0", parents={})

    def test_synthesizer_cannot_bypass_compound_with_empty_parents(self):
        raw = {"root_index": 0, "nodes": [leaf(9)], "parent_policy_ids": []}
        with self.assertRaisesRegex(ValueError, "two distinct parent"):
            _parse_flat_policy(
                raw, origin_id="synth", parents={}, require_compound=True
            )

    def test_local_parser_rejects_schema_bypassed_comparator(self):
        raw = {
            "root_index": 0,
            "nodes": [
                {
                    "kind": "branch", "feature": "total_chunks", "cmp": "none",
                    "threshold": 1, "cpc": 0, "then_index": 1, "else_index": 2,
                },
                leaf(9), leaf(17),
            ],
            "parent_policy_ids": [],
        }
        with self.assertRaisesRegex(ValueError, "malformed branch"):
            _parse_flat_policy(raw, origin_id="p0", parents={})

    def test_single_parent_mutation_is_not_misclassified_as_compound(self):
        parent = RuntimePolicyCandidate(PolicyLeaf(9))
        parent_id = canonical_policy_id(parent)
        raw = {"root_index": 0, "nodes": [leaf(9)], "parent_policy_ids": [parent_id]}
        child = _parse_flat_policy(raw, origin_id="mutation", parents={parent_id: parent})
        self.assertEqual(child.parent_ids, (parent_id,))

    def test_any_tool_event_fails_closed(self):
        safe = json.dumps(
            {"type": "item.completed", "item": {"type": "agent_message", "text": "{}"}}
        )
        command = json.dumps(
            {"type": "item.completed", "item": {"type": "command_execution", "command": "ls"}}
        )
        self.assertEqual(_tool_events(safe), [])
        self.assertEqual(len(_tool_events(safe + "\n" + command)), 1)


if __name__ == "__main__":
    unittest.main()
