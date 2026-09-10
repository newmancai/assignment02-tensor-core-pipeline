#!/usr/bin/env python3
"""Fail-closed integrity certificate for the one-trajectory Stage 9 swarm pilot."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from math import exp
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence"


def canonical(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    paths = {
        "protocol": EVIDENCE / "stage9_best_swarm_pilot_protocol.json",
        "manifest": EVIDENCE / "stage9_best_swarm_pilot_runs_sol/proposal_manifest_v5.json",
        "screen": EVIDENCE / "b300_stage9_pilot_v5_screen.jsonl",
        "qualification": EVIDENCE / "b300_stage9_pilot_v5_qualification.jsonl",
        "heldout": EVIDENCE / "b300_stage9_pilot_v5_heldout.jsonl",
        "freeze": EVIDENCE / "stage9_pilot_v5_winner_freeze.json",
        "reveal": EVIDENCE / "stage9_pilot_v5_heldout_reveal.json",
        "commitment": EVIDENCE / "stage9_best_swarm_pilot_commitment_v5.json",
    }
    protocol = json.loads(paths["protocol"].read_text())
    manifest = json.loads(paths["manifest"].read_text())
    screen, qualification, heldout = (read_jsonl(paths[name]) for name in ("screen", "qualification", "heldout"))
    freeze, reveal, commitment = (json.loads(paths[name].read_text()) for name in ("freeze", "reveal", "commitment"))
    require(protocol.get("pilot") is True and protocol.get("status") == "amended_ready_for_execution", "pilot protocol is not sealed")
    require(canonical(protocol["protocol"]) == protocol["protocol_digest"], "protocol digest mismatch")
    require(manifest.get("status") == "proposal_frozen" and len(manifest.get("proposals", ())) == 8, "manifest is incomplete")
    require(canonical(manifest["proposals"]) == manifest["proposal_manifest_sha256"], "manifest semantic digest mismatch")
    require(manifest["protocol_digest"] == protocol["protocol_digest"], "manifest/protocol mismatch")
    receipts = manifest["agent_receipts"]
    require(len(receipts) == 8 and all(row["status"] == "ok" and row["tool_event_count"] == 0 for row in receipts), "agent receipt failed or used a tool")
    total_tokens = sum(row["token_usage"]["total_tokens"] for row in receipts)
    require(total_tokens <= protocol["protocol"]["budget"]["contract"]["proposal_tokens_per_arm"], "agent token cap exceeded")
    require({row["proposal_slot"] for row in screen} == set(range(8)), "screen coordinates incomplete")
    require(all(row["result"]["status"] == "ok" for row in screen), "screen failure")
    require(len(qualification) == 2 and all(row["result"]["status"] == "ok" for row in qualification), "qualification failure")
    require(freeze["status"] == "policy_frozen_before_heldout_reveal", "winner not frozen")
    require(reveal["status"] == "revealed_after_policy_freeze", "heldout reveal state invalid")
    require(reveal["heldout_digest"] == commitment["heldout_digest"] == protocol["protocol"]["heldout"]["seal_sha256"], "heldout commitment mismatch")
    primary = [row for row in heldout if row["phase"] == "heldout"]
    controls = [row for row in heldout if row["phase"] == "heldout_lineage_control"]
    require(len(primary) == 1 and len(controls) == 2, "heldout query set incomplete")
    require(all(row["result"]["status"] == "ok" for row in heldout), "heldout evaluation failed")
    require(len(primary[0]["result"]["rows"]) == 8 and all(row["correctness"]["passed"] for row in primary[0]["result"]["rows"]), "heldout correctness failed")
    winner = freeze["primary_winners"][0]
    require(primary[0]["policy_id"] == winner["policy_id"] and len(set(winner["parent_policy_ids"])) == 2, "winner is not the frozen compound")
    child_speedup = primary[0]["result"]["geomean_speedup"]
    parent_speedups = {row["policy_id"]: row["result"]["geomean_speedup"] for row in controls}
    all_profile_lowers_above_one = all(row["bootstrap_95"][0] > 1.0 for row in primary[0]["result"]["rows"])
    report = {
        "schema_version": 1,
        "status": "pilot_completed_not_activation_certified",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": "One engineering trajectory demonstrates the closed loop; it does not establish multi-agent superiority or production activation.",
        "agent": {"calls": len(receipts), "total_tokens": total_tokens, "tool_events": 0},
        "search": {"proposal_slots": 8, "unique_policies": len({row["policy_id"] for row in manifest["proposals"]}), "screen_receipts": 8, "qualification_receipts": 2},
        "winner": {"policy_id": winner["policy_id"], "parent_policy_ids": winner["parent_policy_ids"], "development_geomean_speedup": winner["qualification"]["geomean_speedup"]},
        "heldout": {"profiles": 8, "correct": True, "geomean_speedup": child_speedup, "parent_geomean_speedups": parent_speedups, "child_minus_best_parent_fraction": child_speedup / max(parent_speedups.values()) - 1.0, "all_profile_95_lower_bounds_above_one": all_profile_lowers_above_one},
        "activation": {"authorized": False, "reason": "single pilot trajectory and heterogeneous per-profile confidence bounds"},
        "artifact_sha256": {name: file_hash(path) for name, path in paths.items()},
    }
    output = EVIDENCE / "stage9_best_swarm_pilot_certificate.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
