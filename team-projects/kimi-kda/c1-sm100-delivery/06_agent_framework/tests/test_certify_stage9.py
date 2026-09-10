from __future__ import annotations

from hashlib import sha256
import json
from math import exp
from pathlib import Path
import tempfile
import unittest

from certify_stage9 import CertificationError, build_certificate, canonical_digest, file_digest
from kda_ir import AblationArm


ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


class SyntheticStage9Evidence:
    def __init__(self, directory: Path) -> None:
        self.root = directory
        names = (
            "amendment", "manifest", "commitment", "screen_jsonl", "qualification_jsonl",
            "screen_round0_summary", "screen_round1_summary", "qualification_summary",
            "freeze", "heldout_jsonl", "reveal", "heldout_summary",
            "screen_round0_accounting", "screen_round1_accounting",
            "qualification_accounting", "heldout_accounting",
        )
        for name in names:
            suffix = ".jsonl" if name.endswith("jsonl") else ".json"
            setattr(self, name, directory / f"{name}{suffix}")
        self._build()

    @property
    def kwargs(self) -> dict:
        output = {f"{name}_path": getattr(self, name) for name in (
            "amendment", "manifest", "commitment", "screen_jsonl", "qualification_jsonl",
            "freeze", "heldout_jsonl", "reveal", "heldout_summary", "heldout_accounting",
        )}
        output["screen_summary_paths"] = (self.screen_round0_summary, self.screen_round1_summary)
        output["qualification_summary_path"] = self.qualification_summary
        output["development_accounting_paths"] = (
            self.screen_round0_accounting, self.screen_round1_accounting,
            self.qualification_accounting,
        )
        return output

    def _build(self) -> None:
        amendment = json.loads((ROOT / "evidence/stage9_multiagent_ablation_amendment_v2.json").read_text())
        seal = "b" * 64
        amendment["protocol"]["heldout"]["seal_sha256"] = seal
        amendment["protocol_digest"] = canonical_digest(amendment["protocol"])
        amendment["status"] = "amended_ready_for_execution"
        write_json(self.amendment, amendment)
        protocol = amendment["protocol"]
        arms = tuple(protocol["arms"])
        seeds = list(range(91001, 91009))
        arm_protocol_digests = {arm: canonical_digest(protocol["arms"][arm]) for arm in arms}
        arm_prompt_digests = {arm: sha256(f"prompt:{arm}".encode()).hexdigest() for arm in arms}
        common_prompt = "c" * 64
        slots = []
        ids = {}
        agent_receipts = []
        for arm in arms:
            for trajectory in range(8):
                agent_receipts.append({
                    "arm": arm, "trajectory": trajectory,
                    "model": "test-model", "reasoning_effort": "medium",
                    "prompt_sha256": sha256(f"{arm}:{trajectory}".encode()).hexdigest(),
                    "token_usage": {"total_tokens": 800},
                })
                for slot in range(8):
                    policy = {
                        "op": "set_prepare_chunks_per_cta", "value": 9,
                        "_synthetic_nonce": f"{arm}:{trajectory}:{slot}",
                    }
                    policy_id = "kir-policy:" + canonical_digest(policy)
                    ids[(arm, trajectory, slot)] = policy_id
                    slots.append({
                        "arm": arm, "trajectory": trajectory,
                        "proposal_slot": trajectory * 8 + slot,
                        "seed": seeds[trajectory],
                        "status": "valid",
                        "policy_id": policy_id, "policy": policy,
                        "token_count": 100, "proposer_ids": [f"{arm}-agent"],
                        "proposer_roles": ["generalist"], "parent_policy_ids": [],
                        "lineage_delta_refs": [],
                    })
        treatment = AblationArm.MULTI_ROLE_CONCLUSIONS.value
        parents = [ids[(treatment, 0, 0)], ids[(treatment, 0, 1)]]
        child = next(row for row in slots if row["arm"] == treatment and row["trajectory"] == 0 and row["proposal_slot"] == 2)
        child["parent_policy_ids"] = parents
        child["lineage_delta_refs"] = ["left-subtree", "right-subtree"]
        manifest = {
            "schema_version": 2,
            "status": "proposal_frozen",
            "language": protocol["candidate_space"]["language"],
            "amendment_sha256": file_digest(self.amendment),
            "amendment_protocol_digest": amendment["protocol_digest"],
            "candidate_space_digest": canonical_digest(protocol["candidate_space"]),
            "model_digest": "d" * 64,
            "model_config_digest": "e" * 64,
            "model": "test-model",
            "reasoning_effort": "medium",
            "profile_split_digest": "f" * 64,
            "common_prompt_digest": common_prompt,
            "tool_manifest_digest": "1" * 64,
            "arm_prompt_digests": arm_prompt_digests,
            "arm_protocol_digests": arm_protocol_digests,
            "prompt_protocol_digest": canonical_digest({"common": common_prompt, "arms": arm_prompt_digests}),
            "trajectory_seeds": seeds,
            "agent_receipts": agent_receipts,
            "proposals": slots,
        }
        manifest["proposal_manifest_sha256"] = canonical_digest(slots)
        write_json(self.manifest, manifest)
        write_json(self.commitment, {"schema_version": 1, "heldout_digest": seal, "profile_count": 8})

        screen_rows = []
        qualification_rows = []
        winners = []
        heldout_rows = []
        opaque = protocol["heldout"]["opaque_profile_ids"]
        metric = {
            AblationArm.SINGLE_GENERALIST.value: 0.01,
            AblationArm.SINGLE_ROLEPLAY.value: 0.02,
            AblationArm.MULTI_HOMOGENEOUS.value: 0.03,
            AblationArm.MULTI_ROLE_NO_SHARE.value: 0.04,
            treatment: 0.08,
        }
        for row in slots:
            screen_rows.append({
                "arm": row["arm"], "trajectory_id": row["trajectory"],
                "proposal_slot": row["proposal_slot"], "policy_id": row["policy_id"],
                "screen_slot_charged": True, "measured": True,
                "evaluation_wall_seconds": 1.0, "result": {"status": "ok"},
            })
        for arm in arms:
            for trajectory in range(8):
                winner_slot = 2 if arm == treatment and trajectory == 0 else 0
                policy_id = ids[(arm, trajectory, winner_slot)]
                winner_policy = next(row["policy"] for row in slots if row["policy_id"] == policy_id)
                qualification_rows.append({
                    "arm": arm, "trajectory_id": trajectory, "policy_id": policy_id,
                    "evaluation_wall_seconds": 1.0, "result": {"status": "ok"},
                })
                winner_parents = parents if arm == treatment and trajectory == 0 else []
                winners.append({
                    "arm": arm, "trajectory_id": trajectory, "policy_id": policy_id,
                    "policy": winner_policy,
                    "parent_policy_ids": winner_parents, "lineage_delta_refs": [],
                })
                result_rows = [
                    {
                        "profile_id": profile_id,
                        "speedup": exp(metric[arm] + trajectory * 0.0001),
                        "bootstrap_95": [
                            exp(metric[arm] + trajectory * 0.0001 - 0.001),
                            exp(metric[arm] + trajectory * 0.0001 + 0.001),
                        ],
                        "correctness": {"passed": True},
                    }
                    for profile_id in opaque
                ]
                heldout_rows.append({
                    "phase": "heldout",
                    "arm": arm, "trajectory_id": trajectory, "policy_id": policy_id,
                    "winner_set_digest": "pending", "evaluation_wall_seconds": 1.0,
                    "result": {"status": "ok", "rows": result_rows},
                })
                if arm == treatment and trajectory == 0:
                    for index, parent in enumerate(parents):
                        heldout_rows.append({
                            "phase": "heldout_lineage_control", "arm": arm,
                            "trajectory_id": trajectory, "policy_id": parent,
                            "winner_policy_id": policy_id,
                            "winner_set_digest": "pending", "evaluation_wall_seconds": 1.0,
                            "result": {"status": "ok", "rows": [
                                {
                                    "profile_id": profile_id, "speedup": exp(0.05),
                                    "bootstrap_95": [exp(0.049), exp(0.051)],
                                    "correctness": {"passed": True},
                                }
                                for profile_id in opaque
                            ]},
                        })
        winner_digest = canonical_digest(winners)
        for row in heldout_rows:
            row["winner_set_digest"] = winner_digest
        write_jsonl(self.screen_jsonl, screen_rows)
        write_jsonl(self.qualification_jsonl, qualification_rows)
        freeze = {
            "schema_version": 1, "status": "policy_frozen_before_heldout_reveal",
            "manifest_sha256": file_digest(self.manifest),
            "amendment_sha256": file_digest(self.amendment),
            "amendment_protocol_digest": amendment["protocol_digest"],
            "heldout_commitment_sha256": file_digest(self.commitment),
            "heldout_digest": seal, "development_profile_split_digest": "f" * 64,
            "primary_winners": winners, "winner_set_digest": winner_digest,
        }
        write_json(self.freeze, freeze)
        write_jsonl(self.heldout_jsonl, heldout_rows)
        write_json(self.reveal, {
            "status": "revealed_after_policy_freeze", "heldout_digest": seal,
            "winner_set_digest": winner_digest,
            "commitment_sha256": file_digest(self.commitment),
            "profiles": [{"opaque_id": profile_id} for profile_id in opaque],
        })
        for round_index, path in enumerate((self.screen_round0_summary, self.screen_round1_summary)):
            write_json(path, {
                "phase": "screen-round", "round_index": round_index,
                "protocol_digest": amendment["protocol_digest"],
                "manifest_sha256": file_digest(self.manifest),
            })
        write_json(self.qualification_summary, {
            "single_writer_fifo": True,
            "hardware": {
                "device_name": "NVIDIA B300 SXM6 AC",
                "compute_capability": [10, 3], "multiprocessor_count": 148,
            },
            "screen_slots": len(screen_rows),
            "qualification_slots": len(qualification_rows),
            "artifacts": {str(path): file_digest(path) for path in (
                self.manifest, self.screen_jsonl, self.qualification_jsonl, self.freeze
            )},
        })
        compound = [{
            "arm": treatment, "trajectory_id": 0,
            "child_policy_id": ids[(treatment, 0, 2)], "parent_policy_ids": parents,
            "child_minus_parent_log_speedup": [
                {"parent_policy_id": parent, "simultaneous_lower": 0.028}
                for parent in parents
            ],
        }]
        write_json(self.heldout_summary, {
            "single_writer_fifo": True, "compound_lineage_tests": compound,
            "hardware": {
                "device_name": "NVIDIA B300 SXM6 AC",
                "compute_capability": [10, 3], "multiprocessor_count": 148,
            },
            "artifacts": {str(path): file_digest(path) for path in (
                self.commitment, self.freeze, self.heldout_jsonl, self.reveal
            )},
        })
        for index, path in enumerate((
            self.screen_round0_accounting,
            self.screen_round1_accounting,
            self.qualification_accounting,
        )):
            write_json(path, {
                "exit_code": 0, "gpu_count": 1, "allocated_gpu_seconds": 200.0,
                "started_at_utc": f"2026-09-09T00:0{index * 3}:00+00:00",
                "ended_at_utc": f"2026-09-09T00:0{index * 3 + 2}:00+00:00",
            })
        write_json(self.heldout_accounting, {
            "exit_code": 0, "gpu_count": 1, "allocated_gpu_seconds": 200.0,
            "started_at_utc": "2026-09-09T00:09:00+00:00",
            "ended_at_utc": "2026-09-09T00:12:20+00:00",
        })


class CertifyStage9Test(unittest.TestCase):
    def test_complete_synthetic_evidence_certifies(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = SyntheticStage9Evidence(Path(directory))
            certificate = build_certificate(**evidence.kwargs, bootstrap_iterations=500)
            self.assertTrue(certificate["claim_ready"])
            self.assertEqual(len(certificate["arm_receipts"]), 5)
            self.assertEqual(len(certificate["audit"]["contrasts"]), 3)

    def test_tampered_heldout_receipt_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = SyntheticStage9Evidence(Path(directory))
            with evidence.heldout_jsonl.open("a") as output:
                output.write(json.dumps({"arm": "forged"}) + "\n")
            with self.assertRaisesRegex(CertificationError, "does not bind"):
                build_certificate(**evidence.kwargs, bootstrap_iterations=100)

    def test_heldout_allocation_before_freeze_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = SyntheticStage9Evidence(Path(directory))
            accounting = json.loads(evidence.heldout_accounting.read_text())
            accounting["started_at_utc"] = "2026-09-09T00:05:00+00:00"
            write_json(evidence.heldout_accounting, accounting)
            with self.assertRaisesRegex(CertificationError, "before winner freeze"):
                build_certificate(**evidence.kwargs, bootstrap_iterations=100)

    def test_compound_below_preregistered_margin_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = SyntheticStage9Evidence(Path(directory))
            summary = json.loads(evidence.heldout_summary.read_text())
            summary["compound_lineage_tests"][0]["child_minus_parent_log_speedup"][0]["simultaneous_lower"] = 0.001
            write_json(evidence.heldout_summary, summary)
            with self.assertRaisesRegex(CertificationError, "lower bound|parent margin"):
                build_certificate(**evidence.kwargs, bootstrap_iterations=100)


if __name__ == "__main__":
    unittest.main()
