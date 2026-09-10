#!/usr/bin/env python3
"""Fail-closed certificate builder for the Stage 9 multi-agent ablation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from hashlib import sha256
import json
from math import log
from pathlib import Path
from typing import Any, Iterable

from kda_ir import (
    AblationArm,
    ArmReceipt,
    EqualBudgetContract,
    SharedExperimentEnvelope,
    TrajectoryReceipt,
    audit_equal_budget_ablation,
)


class CertificationError(ValueError):
    pass


def canonical_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode()).hexdigest()


def file_digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CertificationError(f"cannot read JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise CertificationError(f"{path} must contain one JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as error:
        raise CertificationError(f"cannot read JSONL artifact {path}: {error}") from error
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise CertificationError(f"{path} must contain non-empty JSON objects")
    return rows


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CertificationError(message)


def require_artifact(summary: dict[str, Any], path: Path) -> None:
    artifacts = summary.get("artifacts")
    require(isinstance(artifacts, dict), "summary is missing artifact hashes")
    require(file_digest(path) in artifacts.values(), f"summary does not bind {path}")


def parse_time(value: object, name: str) -> datetime:
    require(isinstance(value, str), f"{name} timestamp missing")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise CertificationError(f"invalid {name} timestamp") from error
    require(parsed.tzinfo is not None, f"{name} timestamp has no timezone")
    return parsed


def _event_seconds(rows: Iterable[dict[str, Any]], arm: str) -> float:
    return sum(float(row.get("evaluation_wall_seconds", 0.0)) for row in rows if row.get("arm") == arm)


def _allocate_gpu_seconds(
    arms: tuple[str, ...],
    screen_rows: list[dict[str, Any]],
    qualification_rows: list[dict[str, Any]],
    heldout_rows: list[dict[str, Any]],
    development_accountings: tuple[dict[str, Any], ...],
    heldout_accounting: dict[str, Any],
) -> dict[str, float]:
    development = screen_rows + qualification_rows
    dev_weights = {arm: _event_seconds(development, arm) for arm in arms}
    held_weights = {arm: _event_seconds(heldout_rows, arm) for arm in arms}
    require(sum(dev_weights.values()) > 0 and sum(held_weights.values()) > 0, "missing per-event GPU wall time")
    require(all(row.get("exit_code") == 0 for row in development_accountings) and heldout_accounting.get("exit_code") == 0, "GPU evaluator process failed")
    require(all(row.get("gpu_count") == 1 for row in development_accountings) and heldout_accounting.get("gpu_count") == 1, "Stage9 requires one visible GPU")
    dev_total = sum(float(row["allocated_gpu_seconds"]) for row in development_accountings)
    held_total = float(heldout_accounting["allocated_gpu_seconds"])
    return {
        arm: dev_total * dev_weights[arm] / sum(dev_weights.values())
        + held_total * held_weights[arm] / sum(held_weights.values())
        for arm in arms
    }


def _sha_field(value: object, name: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value),
        f"{name} must be a lowercase SHA-256",
    )
    return value


def build_certificate(
    *,
    amendment_path: Path,
    manifest_path: Path,
    commitment_path: Path,
    screen_jsonl_path: Path,
    qualification_jsonl_path: Path,
    screen_summary_paths: tuple[Path, ...],
    qualification_summary_path: Path,
    freeze_path: Path,
    heldout_jsonl_path: Path,
    reveal_path: Path,
    heldout_summary_path: Path,
    development_accounting_paths: tuple[Path, ...],
    heldout_accounting_path: Path,
    bootstrap_iterations: int | None = None,
) -> dict[str, Any]:
    amendment = read_json(amendment_path)
    manifest = read_json(manifest_path)
    commitment = read_json(commitment_path)
    screen_summaries = tuple(read_json(path) for path in screen_summary_paths)
    qualification_summary = read_json(qualification_summary_path)
    freeze = read_json(freeze_path)
    reveal = read_json(reveal_path)
    heldout_summary = read_json(heldout_summary_path)
    development_accountings = tuple(read_json(path) for path in development_accounting_paths)
    heldout_accounting = read_json(heldout_accounting_path)
    screen_rows = read_jsonl(screen_jsonl_path)
    qualification_rows = read_jsonl(qualification_jsonl_path)
    heldout_rows = read_jsonl(heldout_jsonl_path)

    require(amendment.get("schema_version") == 2, "Stage9 certificate requires amendment schema v2")
    protocol = amendment.get("protocol")
    require(isinstance(protocol, dict) and canonical_digest(protocol) == amendment.get("protocol_digest"), "amendment protocol digest mismatch")
    require(amendment.get("amendment_timing") == "before_first_stage9_agent_call_and_before_first_stage9_GPU_run", "amendment timing is not pre-execution")
    arm_specs = protocol["arms"]
    arms = tuple(arm_specs)
    require(set(arms) == {arm.value for arm in AblationArm}, "amendment must contain all five causal arms")
    contract = EqualBudgetContract(**protocol["budget"]["contract"])
    candidate_space_digest = canonical_digest(protocol["candidate_space"])
    heldout_seal = _sha_field(protocol["heldout"].get("seal_sha256"), "amendment heldout seal")
    require(heldout_seal == commitment.get("heldout_digest"), "commitment differs from amendment heldout seal")

    require(manifest.get("schema_version") == 2, "proposal manifest schema mismatch")
    require(manifest.get("status") == "proposal_frozen", "proposal manifest is not frozen")
    require(manifest.get("amendment_sha256") == file_digest(amendment_path), "proposal manifest amendment file hash mismatch")
    require(manifest.get("amendment_protocol_digest") == amendment["protocol_digest"], "proposal manifest amendment mismatch")
    require(manifest.get("candidate_space_digest") == candidate_space_digest, "proposal candidate-space digest mismatch")
    common_prompt_digest = _sha_field(manifest.get("common_prompt_digest"), "common prompt digest")
    model_digest = _sha_field(manifest.get("model_digest"), "model digest")
    model_config_digest = _sha_field(manifest.get("model_config_digest"), "model config digest")
    profile_split_digest = _sha_field(manifest.get("profile_split_digest"), "profile split digest")
    tool_manifest_digest = _sha_field(manifest.get("tool_manifest_digest"), "tool manifest digest")
    arm_prompt_digests = manifest.get("arm_prompt_digests")
    require(isinstance(arm_prompt_digests, dict) and set(arm_prompt_digests) == set(arms), "arm prompt digest map mismatch")
    for arm, digest in arm_prompt_digests.items():
        _sha_field(digest, f"{arm} prompt digest")
    expected_prompt_protocol = canonical_digest({"common": common_prompt_digest, "arms": arm_prompt_digests})
    require(manifest.get("prompt_protocol_digest") == expected_prompt_protocol, "prompt protocol digest mismatch")
    arm_protocol_digests = {arm: canonical_digest(arm_specs[arm]) for arm in arms}
    require(manifest.get("arm_protocol_digests") == arm_protocol_digests, "arm protocol digests differ from amendment")

    agent_receipts = manifest.get("agent_receipts")
    require(isinstance(agent_receipts, list) and agent_receipts, "proposal manifest lacks agent execution receipts")
    for receipt in agent_receipts:
        arm = receipt.get("arm")
        require(arm in arms, "agent receipt has an unknown arm")
        require(receipt.get("model") == manifest.get("model"), "agent receipt model mismatch")
        require(receipt.get("reasoning_effort") == manifest.get("reasoning_effort"), "agent receipt reasoning effort mismatch")
        _sha_field(receipt.get("prompt_sha256"), "agent prompt digest")
        require(type(receipt.get("trajectory")) is int and 0 <= receipt["trajectory"] < contract.trajectories, "agent receipt trajectory is invalid")
        token_count = receipt.get("token_usage", {}).get("total_tokens")
        require(type(token_count) is int and token_count >= 0, "agent receipt token count is invalid")
    require(
        {(row["arm"], row["trajectory"]) for row in agent_receipts}
        == {(arm, trajectory) for arm in arms for trajectory in range(contract.trajectories)},
        "agent receipts do not cover every arm/trajectory",
    )

    slots = manifest.get("proposals")
    require(isinstance(slots, list), "proposal manifest slots missing")
    require(manifest.get("proposal_manifest_sha256") == canonical_digest(slots), "proposal manifest semantic digest mismatch")
    expected_coordinates = {
        (arm, trajectory, trajectory * protocol["budget"]["per_trajectory_caps"]["proposal_slots"] + slot)
        for arm in arms
        for trajectory in range(contract.trajectories)
        for slot in range(protocol["budget"]["per_trajectory_caps"]["proposal_slots"])
    }
    coordinates = {(row.get("arm"), row.get("trajectory"), row.get("proposal_slot")) for row in slots}
    require(len(slots) == len(expected_coordinates) and coordinates == expected_coordinates, "proposal manifest is not complete and unique")
    for row in slots:
        require(row.get("seed") == manifest["trajectory_seeds"][row["trajectory"]], "proposal seed differs from trajectory block")
        if row.get("status") == "valid":
            require(row.get("policy_id") == "kir-policy:" + canonical_digest(row.get("policy")), "proposal policy ID does not match canonical policy")
        else:
            require(row.get("policy") is None and not row.get("policy_id"), "invalid proposal carries executable policy")
    for arm in arms:
        require(
            sum(int(row["token_usage"]["total_tokens"]) for row in agent_receipts if row["arm"] == arm)
            <= contract.proposal_tokens_per_arm,
            f"{arm} exceeds proposal token cap",
        )

    require(len(screen_summaries) == 2, "exactly two screen-round summaries are required")
    require({row.get("round_index") for row in screen_summaries} == {0, 1}, "screen-round summaries are incomplete")
    require(all(row.get("protocol_digest") == amendment["protocol_digest"] for row in screen_summaries), "screen-round protocol digest mismatch")
    require(all(row.get("manifest_sha256") == file_digest(manifest_path) for row in screen_summaries), "screen-round manifest hash mismatch")
    require(qualification_summary.get("single_writer_fifo") is True, "qualification summary does not attest single-writer FIFO")
    for path in (manifest_path, screen_jsonl_path, qualification_jsonl_path, freeze_path):
        require_artifact(qualification_summary, path)
    require(heldout_summary.get("single_writer_fifo") is True, "heldout summary does not attest single-writer FIFO")
    for path in (commitment_path, freeze_path, heldout_jsonl_path, reveal_path):
        require_artifact(heldout_summary, path)
    screen_hardware = qualification_summary.get("hardware")
    heldout_hardware = heldout_summary.get("hardware")
    require(isinstance(screen_hardware, dict) and screen_hardware == heldout_hardware, "GPU summaries have different hardware")
    require(
        screen_hardware.get("device_name") == "NVIDIA B300 SXM6 AC"
        and screen_hardware.get("compute_capability") == [10, 3]
        and screen_hardware.get("multiprocessor_count") == 148,
        "Stage9 evidence is not from the frozen B300 target",
    )
    require(qualification_summary.get("screen_slots") == len(screen_rows), "screen summary count mismatch")
    require(qualification_summary.get("qualification_slots") == len(qualification_rows), "qualification summary count mismatch")
    require(freeze.get("manifest_sha256") == file_digest(manifest_path), "winner freeze manifest hash mismatch")
    require(freeze.get("amendment_sha256") == file_digest(amendment_path), "winner freeze amendment file hash mismatch")
    require(freeze.get("amendment_protocol_digest") == amendment["protocol_digest"], "winner freeze amendment protocol mismatch")
    require(freeze.get("heldout_commitment_sha256") == file_digest(commitment_path), "winner freeze commitment hash mismatch")
    require(freeze.get("heldout_digest") == heldout_seal, "winner freeze heldout seal mismatch")
    require(freeze.get("development_profile_split_digest") == profile_split_digest, "winner freeze development split mismatch")
    winners = freeze.get("primary_winners")
    require(isinstance(winners, list) and freeze.get("winner_set_digest") == canonical_digest(winners), "winner set digest mismatch")
    winner_by_coordinate = {(row.get("arm"), row.get("trajectory_id")): row for row in winners}
    expected_winners = {(arm, trajectory) for arm in arms for trajectory in range(contract.trajectories)}
    require(len(winners) == len(expected_winners) and set(winner_by_coordinate) == expected_winners, "winner freeze is incomplete")
    slot_by_id = {row["policy_id"]: row for row in slots}
    require(
        all(
            winner.get("policy_id") in slot_by_id
            and winner.get("policy") == slot_by_id[winner["policy_id"]].get("policy")
            for winner in winners
        ),
        "winner freeze policy is absent from or differs from the proposal manifest",
    )

    require(reveal.get("status") == "revealed_after_policy_freeze", "heldout reveal status mismatch")
    require(reveal.get("heldout_digest") == heldout_seal, "heldout reveal seal mismatch")
    require(reveal.get("winner_set_digest") == freeze["winner_set_digest"], "heldout reveal winner-set mismatch")
    require(reveal.get("commitment_sha256") == file_digest(commitment_path), "heldout reveal commitment hash mismatch")
    require(
        max(parse_time(row.get("ended_at_utc"), "development end") for row in development_accountings)
        < parse_time(heldout_accounting.get("started_at_utc"), "heldout start"),
        "heldout allocation began before winner freeze allocation completed",
    )

    screen_coordinates = {
        (row.get("arm"), row.get("trajectory_id"), row.get("proposal_slot"))
        for row in screen_rows
    }
    require(
        len(screen_rows) == len(expected_coordinates)
        and screen_coordinates == expected_coordinates
        and all(row.get("screen_slot_charged") is True for row in screen_rows),
        "screen ledger does not charge every proposal opportunity exactly once",
    )
    qualified = {
        (row.get("arm"), row.get("trajectory_id"), row.get("policy_id"))
        for row in qualification_rows
        if row.get("result", {}).get("status") == "ok"
    }
    require(
        all((arm, trajectory, winner_by_coordinate[(arm, trajectory)].get("policy_id")) in qualified for arm, trajectory in expected_winners),
        "a frozen winner lacks successful development qualification",
    )

    primary_heldout_rows = [row for row in heldout_rows if row.get("phase") == "heldout"]
    lineage_control_rows = [row for row in heldout_rows if row.get("phase") == "heldout_lineage_control"]
    heldout_by_coordinate = {(row.get("arm"), row.get("trajectory_id")): row for row in primary_heldout_rows}
    require(len(primary_heldout_rows) == len(expected_winners) and set(heldout_by_coordinate) == expected_winners, "heldout must contain exactly 40 primary results")
    require(len(primary_heldout_rows) + len(lineage_control_rows) == len(heldout_rows), "heldout ledger contains an unknown phase")
    opaque_ids = set(protocol["heldout"]["opaque_profile_ids"])
    require(len(opaque_ids) > 0, "amendment has no opaque heldout IDs")
    require(commitment.get("profile_count") == len(opaque_ids), "heldout commitment profile count mismatch")
    require({row.get("opaque_id") for row in reveal.get("profiles", ())} == opaque_ids, "heldout reveal profiles differ from amendment")
    for coordinate, row in heldout_by_coordinate.items():
        winner = winner_by_coordinate[coordinate]
        require(row.get("policy_id") == winner.get("policy_id"), "heldout policy differs from frozen winner")
        require(row.get("winner_set_digest") == freeze["winner_set_digest"], "heldout row winner-set mismatch")
        result = row.get("result", {})
        require(result.get("status") == "ok", "heldout winner failed evaluation")
        profile_rows = result.get("rows")
        require(isinstance(profile_rows, list) and {item.get("profile_id") for item in profile_rows} == opaque_ids, "heldout profile IDs differ from amendment")
        require(all(item.get("correctness", {}).get("passed") is True and float(item.get("speedup", 0)) > 0 for item in profile_rows), "heldout correctness or speedup invalid")

    gpu_seconds = _allocate_gpu_seconds(
        arms, screen_rows, qualification_rows, heldout_rows, development_accountings, heldout_accounting
    )
    seeds = manifest.get("trajectory_seeds")
    require(isinstance(seeds, list) and len(seeds) == contract.trajectories and len(set(seeds)) == len(seeds), "trajectory seeds missing or duplicated")
    envelope = SharedExperimentEnvelope(
        model_digest=model_digest,
        model_config_digest=model_config_digest,
        profile_split_digest=profile_split_digest,
        common_prompt_digest=common_prompt_digest,
        candidate_space_digest=candidate_space_digest,
        tool_manifest_digest=tool_manifest_digest,
    )
    freeze_time = max(
        parse_time(row.get("ended_at_utc"), "development end")
        for row in development_accountings
    ).isoformat()
    arm_receipts = []
    for arm_name in arms:
        arm = AblationArm(arm_name)
        arm_slots = [row for row in slots if row["arm"] == arm_name]
        arm_screen = [row for row in screen_rows if row.get("arm") == arm_name]
        arm_qualification = [row for row in qualification_rows if row.get("arm") == arm_name]
        trajectories = []
        for trajectory_id, seed in enumerate(seeds):
            heldout = heldout_by_coordinate[(arm_name, trajectory_id)]
            trajectories.append(
                TrajectoryReceipt(
                    trajectory_id=str(trajectory_id),
                    seed=int(seed),
                    winner_candidate_id=heldout["policy_id"],
                    heldout_log_speedups=tuple(log(float(item["speedup"])) for item in heldout["result"]["rows"]),
                    heldout_seal_sha256=heldout_seal,
                    winner_frozen_at_utc=freeze_time,
                    heldout_unsealed_at_utc=heldout_accounting["started_at_utc"],
                )
            )
        arm_receipts.append(
            ArmReceipt(
                arm=arm,
                shared_envelope=envelope,
                arm_protocol_digest=arm_protocol_digests[arm_name],
                trajectories=tuple(trajectories),
                proposal_token_cap=contract.proposal_tokens_per_arm,
                proposal_tokens_actual=sum(
                    int(row["token_usage"]["total_tokens"])
                    for row in agent_receipts if row["arm"] == arm_name
                ),
                proposal_slot_cap=contract.proposal_slots_per_arm,
                proposal_slots_charged=len(arm_slots),
                unique_candidates=len({row["policy_id"] for row in arm_slots if row.get("status") == "valid"}),
                screen_slot_cap=contract.screen_slots_per_arm,
                screen_slots_charged=len(arm_screen),
                screen_attempts_actual=sum(row.get("measured") is True for row in arm_screen),
                qualification_slot_cap=contract.qualification_slots_per_arm,
                qualification_slots_charged=len(arm_qualification),
                qualification_attempts_actual=len(arm_qualification),
                gpu_second_cap=contract.gpu_seconds_per_arm,
                gpu_seconds_actual=gpu_seconds[arm_name],
                compound_candidates=sum(len(set(row.get("parent_policy_ids", ()))) >= 2 for row in arm_slots),
            )
        )

    compound_margin = float(protocol["statistics"]["compound_log_speedup_margin"])
    compound_tests = heldout_summary.get("compound_lineage_tests")
    require(isinstance(compound_tests, list) and compound_tests, "heldout summary lacks compound child-v-parent tests")
    valid_compound = 0
    slot_ids = {row.get("policy_id") for row in slots}
    for item in compound_tests:
        require(item.get("arm") == AblationArm.MULTI_ROLE_CONCLUSIONS.value, "compound test belongs to wrong arm")
        coordinate = (item["arm"], item["trajectory_id"])
        winner = winner_by_coordinate.get(coordinate)
        require(winner is not None and item.get("child_policy_id") == winner.get("policy_id"), "compound child is not the frozen winner")
        parents = item.get("parent_policy_ids")
        require(isinstance(parents, list) and len(set(parents)) >= 2 and set(parents).issubset(slot_ids), "compound parents are unresolved")
        require(set(parents) == set(winner.get("parent_policy_ids", ())), "compound parents differ from frozen lineage")
        contrasts = item.get("child_minus_parent_log_speedup")
        require(isinstance(contrasts, list) and {row.get("parent_policy_id") for row in contrasts} == set(parents), "compound parent contrasts incomplete")
        child_receipt = heldout_by_coordinate[coordinate]
        child_profiles = {row["profile_id"]: row for row in child_receipt["result"]["rows"]}
        controls = {
            row.get("policy_id"): row
            for row in lineage_control_rows
            if row.get("arm") == item["arm"]
            and row.get("trajectory_id") == item["trajectory_id"]
            and row.get("winner_policy_id") == item["child_policy_id"]
        }
        require(set(controls) == set(parents), "compound lineage-control receipts incomplete")
        for contrast in contrasts:
            parent_profiles = {
                row["profile_id"]: row
                for row in controls[contrast["parent_policy_id"]]["result"]["rows"]
            }
            require(set(parent_profiles) == set(child_profiles), "compound child/parent profile mismatch")
            recomputed = min(
                log(float(child_profiles[profile_id]["bootstrap_95"][0]))
                - log(float(parent_profiles[profile_id]["bootstrap_95"][1]))
                for profile_id in child_profiles
            )
            require(abs(recomputed - float(contrast["simultaneous_lower"])) <= 1e-12, "compound lower bound differs from raw heldout receipts")
        require(all(float(row.get("simultaneous_lower", float("-inf"))) > compound_margin for row in contrasts), "compound child does not clear preregistered parent margin")
        valid_compound += 1
    require(valid_compound > 0, "no valid compound winner")

    iterations = bootstrap_iterations or int(protocol["statistics"]["bootstrap_iterations"])
    audit = audit_equal_budget_ablation(tuple(arm_receipts), contract, bootstrap_iterations=iterations, bootstrap_seed=int(protocol["statistics"]["bootstrap_seed"]))
    require(audit.claim_ready, "ablation audit rejected claim: " + ",".join(audit.reasons))
    return {
        "schema_version": 1,
        "status": "stage9_claim_ready",
        "claim_ready": True,
        "amendment_protocol_digest": amendment["protocol_digest"],
        "heldout_seal_sha256": heldout_seal,
        "budget_contract": asdict(contract),
        "arm_receipts": [asdict(item) for item in arm_receipts],
        "audit": asdict(audit),
        "compound_lineage_tests": compound_tests,
        "artifacts": {
            str(path): file_digest(path)
            for path in (
                amendment_path, manifest_path, commitment_path, screen_jsonl_path,
                qualification_jsonl_path, *screen_summary_paths,
                qualification_summary_path, freeze_path,
                heldout_jsonl_path, reveal_path, heldout_summary_path,
                *development_accounting_paths, heldout_accounting_path,
            )
        },
        "claim_boundary": "One sealed Stage9 campaign on one B300; no cross-device or deployment claim.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("evidence")
    parser.add_argument("--amendment", type=Path, default=root / "stage9_multiagent_ablation_amendment_v2.json")
    parser.add_argument("--manifest", type=Path, default=root / "stage9_policy_manifest.json")
    parser.add_argument("--commitment", type=Path, default=root / "stage9_heldout_commitment.json")
    parser.add_argument("--screen-jsonl", type=Path, default=root / "b300_stage9_screen.jsonl")
    parser.add_argument("--qualification-jsonl", type=Path, default=root / "b300_stage9_qualification.jsonl")
    parser.add_argument("--screen-round0-summary", type=Path, default=root / "b300_stage9_screen_round0_summary.json")
    parser.add_argument("--screen-round1-summary", type=Path, default=root / "b300_stage9_screen_round1_summary.json")
    parser.add_argument("--qualification-summary", type=Path, default=root / "b300_stage9_qualification_summary.json")
    parser.add_argument("--freeze", type=Path, default=root / "stage9_winner_freeze.json")
    parser.add_argument("--heldout-jsonl", type=Path, default=root / "b300_stage9_heldout.jsonl")
    parser.add_argument("--reveal", type=Path, default=root / "stage9_heldout_reveal.json")
    parser.add_argument("--heldout-summary", type=Path, default=root / "b300_stage9_heldout_summary.json")
    parser.add_argument("--screen-round0-accounting", type=Path, default=root / "b300_stage9_screen_round0_accounting.json")
    parser.add_argument("--screen-round1-accounting", type=Path, default=root / "b300_stage9_screen_round1_accounting.json")
    parser.add_argument("--qualification-accounting", type=Path, default=root / "b300_stage9_qualification_accounting.json")
    parser.add_argument("--heldout-accounting", type=Path, default=root / "b300_stage9_heldout_accounting.json")
    parser.add_argument("--output", type=Path, default=root / "b300_stage9_certificate.json")
    args = parser.parse_args()
    try:
        certificate = build_certificate(
            amendment_path=args.amendment, manifest_path=args.manifest,
            commitment_path=args.commitment, screen_jsonl_path=args.screen_jsonl,
            qualification_jsonl_path=args.qualification_jsonl,
            screen_summary_paths=(args.screen_round0_summary, args.screen_round1_summary),
            qualification_summary_path=args.qualification_summary,
            freeze_path=args.freeze,
            heldout_jsonl_path=args.heldout_jsonl, reveal_path=args.reveal,
            heldout_summary_path=args.heldout_summary,
            development_accounting_paths=(
                args.screen_round0_accounting,
                args.screen_round1_accounting,
                args.qualification_accounting,
            ),
            heldout_accounting_path=args.heldout_accounting,
        )
    except (CertificationError, KeyError, TypeError, ValueError) as error:
        certificate = {"schema_version": 1, "status": "stage9_claim_rejected", "claim_ready": False, "reasons": [f"{type(error).__name__}: {error}"]}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n")
        raise SystemExit(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
