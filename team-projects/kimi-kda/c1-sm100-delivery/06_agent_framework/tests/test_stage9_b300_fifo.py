import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = ROOT / "evidence/stage9_multiagent_ablation_amendment_v2.json"
MODULE_PATH = ROOT / "stage3_remote/benchmarks/bench_flash_kda_stage9_ablation.py"
SPEC = importlib.util.spec_from_file_location("stage9_b300", MODULE_PATH)
stage9 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = stage9
SPEC.loader.exec_module(stage9)


def leaf(value):
    return {"kind": "leaf", "chunks_per_cta": value}


class Stage9PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.amendment, cls.config = stage9.load_amendment(AMENDMENT)

    def test_development_split_digest_is_frozen(self):
        cases = (
            SimpleNamespace(name="h12_profile_fixed_8192", num_heads=12, seq_lens=(8192,), packed=False),
            SimpleNamespace(name="h12_profile_balanced_2", num_heads=12, seq_lens=(4000, 4192), packed=True),
            SimpleNamespace(name="h12_profile_balanced_4", num_heads=12, seq_lens=(1952, 2016, 2080, 2144), packed=True),
            SimpleNamespace(name="h12_profile_skew", num_heads=12, seq_lens=(64, 64, 64, 8000), packed=True),
            SimpleNamespace(name="h12_w256_fixed_4096", num_heads=12, seq_lens=(4096,), packed=False),
            SimpleNamespace(name="h12_w256_balanced_4", num_heads=12, seq_lens=(976, 1008, 1040, 1072), packed=True),
            SimpleNamespace(name="h12_w1024_fixed_16384", num_heads=12, seq_lens=(16384,), packed=False),
            SimpleNamespace(name="h12_w1024_balanced_4", num_heads=12, seq_lens=(4000, 4064, 4128, 4192), packed=True),
        )
        self.assertEqual(
            stage9.canonical_digest(stage9.development_profile_descriptor(cases)),
            "3e19faba80632a7b4b71febd5a036f26274ad3e5407e091603759f336b6d662e",
        )

    def test_policy_is_canonical_and_resolves_piecewise(self):
        policy = {
            "kind": "node",
            "predicate": {"feature": "total_chunks", "operator": "le", "threshold": 512},
            "then": leaf(5),
            "else": leaf(17),
        }
        features = {
            "total_chunks": 320,
            "max_chunks": 320,
            "num_sequences": 1,
            "resident_grid_capacity_ctas": 740,
        }
        self.assertEqual(stage9.resolve_policy(policy, features, self.config), 5)
        self.assertEqual(stage9.canonical_policy_id(policy, self.config), stage9.canonical_policy_id(json.loads(json.dumps(policy)), self.config))
        self.assertTrue(stage9.canonical_policy_id(policy, self.config).startswith("kir-policy:"))

    def test_policy_rejects_raw_or_overdeep_nodes(self):
        with self.assertRaises(ValueError):
            stage9.validate_policy({"op": "run_shell", "value": 9}, self.config)
        branch = {"kind": "node", "predicate": {"feature": "total_chunks", "operator": "gt", "threshold": 1}, "then": leaf(9), "else": leaf(4)}
        overdeep = {"kind": "node", "predicate": {"feature": "max_chunks", "operator": "le", "threshold": 10}, "then": branch, "else": branch}
        overdeep = {"kind": "node", "predicate": {"feature": "num_sequences", "operator": "le", "threshold": 4}, "then": overdeep, "else": leaf(9)}
        with self.assertRaises(ValueError):
            stage9.validate_policy(overdeep, self.config)

    def test_rotating_fifo_changes_first_arm_and_preserves_all_slots(self):
        slots = []
        for arm in self.config.arms:
            for trajectory in range(self.config.trajectories):
                for local_slot in range(self.config.proposal_slots_per_trajectory):
                    proposal_slot = trajectory * self.config.proposal_slots_per_trajectory + local_slot
                    policy = leaf(9)
                    slots.append(stage9.ProposalSlot(arm, trajectory, proposal_slot, policy, stage9.canonical_policy_id(policy, self.config), "valid", "", (), (), (), ()))
        queue = stage9.rotating_fifo(slots, self.config)
        expected = len(self.config.arms) * self.config.proposal_slots_per_arm
        self.assertEqual(len(queue), expected)
        self.assertEqual(len({(item.arm, item.trajectory_id, item.proposal_slot) for item in queue}), expected)
        self.assertGreater(len({queue[index * len(self.config.arms)].arm for index in range(5)}), 1)

    def test_two_round_barrier_and_durable_deduplication(self):
        policy = leaf(9)
        policy_id = stage9.canonical_policy_id(policy, self.config)
        slots = []
        for arm in self.config.arms:
            for trajectory in range(self.config.trajectories):
                for local_slot in range(self.config.proposal_slots_per_trajectory):
                    proposal_slot = trajectory * self.config.proposal_slots_per_trajectory + local_slot
                    slots.append(stage9.ProposalSlot(arm, trajectory, proposal_slot, policy, policy_id, "valid", "", (), (), (), ()))

        round0_plan = stage9.screen_round_plan(slots, self.config, 0)
        self.assertEqual(sum(measured for _slot, measured in round0_plan), 1)
        round0_receipts = [
            {
                "arm": slot.arm,
                "trajectory_id": slot.trajectory_id,
                "proposal_slot": slot.proposal_slot,
                "round_index": 0,
                "policy_id": slot.policy_id,
                "measured": measured,
            }
            for slot, measured in round0_plan
        ]
        with self.assertRaisesRegex(ValueError, "both screen rounds"):
            stage9.require_complete_screen_barrier(round0_receipts, self.config)

        round1_plan = stage9.screen_round_plan(slots, self.config, 1, round0_receipts)
        self.assertEqual(sum(measured for _slot, measured in round1_plan), 0)
        round1_receipts = [
            {
                "arm": slot.arm,
                "trajectory_id": slot.trajectory_id,
                "proposal_slot": slot.proposal_slot,
                "round_index": 1,
                "policy_id": slot.policy_id,
                "measured": measured,
            }
            for slot, measured in round1_plan
        ]
        stage9.require_complete_screen_barrier(
            [*round0_receipts, *round1_receipts], self.config
        )

    def test_manifest_requires_exact_budget_and_policy_digest(self):
        slots = []
        for arm in self.config.arms:
            for trajectory in range(self.config.trajectories):
                for local_slot in range(self.config.proposal_slots_per_trajectory):
                    proposal_slot = trajectory * self.config.proposal_slots_per_trajectory + local_slot
                    policy = leaf(min(self.config.cpc_values))
                    slots.append({"arm": arm, "trajectory": trajectory, "proposal_slot": proposal_slot, "status": "valid", "error": "", "policy": policy, "policy_id": stage9.canonical_policy_id(policy, self.config), "parent_policy_ids": []})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            metadata = {
                "schema_version": 2,
                "status": "proposal_frozen",
                "arms": list(self.config.arms),
            }
            path.write_text(json.dumps({**metadata, "proposals": slots, "proposal_manifest_sha256": stage9.canonical_digest(slots)}))
            _manifest, parsed = stage9.load_proposal_manifest(path, self.config)
            self.assertEqual(len(parsed), len(self.config.arms) * self.config.proposal_slots_per_arm)
            slots[0]["policy_id"] = "kir-policy:bad"
            path.write_text(json.dumps({**metadata, "proposals": slots, "proposal_manifest_sha256": stage9.canonical_digest(slots)}))
            with self.assertRaises(ValueError):
                stage9.load_proposal_manifest(path, self.config)

    def test_heldout_commitment_detects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            secret = Path(directory) / "secret.json"
            commitment = Path(directory) / "commitment.json"
            stage9.create_heldout_secret(secret, commitment, self.config)
            loaded = stage9.verify_heldout_commitment(secret, commitment, self.config)
            self.assertEqual(len(loaded["profiles"]), 8)
            self.assertEqual(tuple(row["opaque_id"] for row in loaded["profiles"]), self.config.opaque_profile_ids)
            self.assertEqual({sum((length + 15) // 16 for length in row["seq_lens"]) for row in loaded["profiles"]}, {288, 336, 400, 464})
            payload = json.loads(secret.read_text())
            payload["profiles"][0]["seq_lens"] = [5136]
            secret.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                stage9.verify_heldout_commitment(secret, commitment, self.config)


if __name__ == "__main__":
    unittest.main()
