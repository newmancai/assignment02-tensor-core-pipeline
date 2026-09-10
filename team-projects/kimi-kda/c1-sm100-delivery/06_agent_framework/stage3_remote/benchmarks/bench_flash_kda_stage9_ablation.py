#!/usr/bin/env python3
"""Single-B300 FIFO evaluator for the Stage 9 equal-budget ablation.

The proposal manifest is data, never executable code.  Two ``screen-round``
calls append idempotent receipts; ``qualification`` refuses to run before that
barrier and freezes one winner per trajectory before ``heldout`` is allowed to
open the remote-only profile file.
The held-out file is committed with a salted canonical SHA-256 digest before
proposal generation and revealed only after the winner-freeze digest exists.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import exp, log
import os
from pathlib import Path
import secrets
from statistics import fmean
import time
from typing import Any, Iterable


EXPECTED_ROUTE = "bt16_prepare_chain_m64"
EXPECTED_TARGET = "sm_103a"
EXPECTED_PHYSICAL_VARIANTS = (
    "sm_103a:flashkda_bf16_bt16_prepare_0d8e6c8011",
    "sm_103a:flashkda_bf16_bt16_chain_m64_c68ffebac9",
)
BASELINE_CPC = 9
RESIDENT_GRID_CAPACITY_CTAS = 740


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def canonical_digest(value: object) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


@dataclass(frozen=True)
class ProtocolConfig:
    arms: tuple[str, ...]
    trajectories: int
    proposal_slots_per_arm: int
    proposal_slots_per_trajectory: int
    qualification_slots_per_trajectory: int
    rounds: int
    proposal_opportunities_per_round: int
    token_cap_per_arm: int
    token_cap_per_trajectory: int
    cpc_values: frozenset[int]
    features: frozenset[str]
    comparisons: frozenset[str]
    max_policy_depth: int
    language: str
    protocol_digest: str
    candidate_space_digest: str
    opaque_profile_ids: tuple[str, ...]


def load_amendment(path: Path, *, require_external_seal: bool = False) -> tuple[dict[str, Any], ProtocolConfig]:
    amendment = _read_json(path)
    if amendment.get("schema_version") != 2:
        raise ValueError("Stage9 requires the schema-v2 amendment")
    protocol = amendment.get("protocol")
    if not isinstance(protocol, dict) or canonical_digest(protocol) != amendment.get("protocol_digest"):
        raise ValueError("Stage9 amendment protocol digest mismatch")
    budget = protocol["budget"]
    contract = budget["contract"]
    per_trajectory = budget["per_trajectory_caps"]
    candidate_space = protocol["candidate_space"]
    heldout = protocol["heldout"]
    arms = tuple(protocol["arms"])
    trajectories = int(contract["trajectories"])
    proposal_slots = int(contract["proposal_slots_per_arm"])
    proposal_slots_per_trajectory = int(per_trajectory["proposal_slots"])
    qualification_slots_per_trajectory = int(per_trajectory["qualification_slots"])
    if proposal_slots != trajectories * proposal_slots_per_trajectory:
        raise ValueError("amendment proposal budgets are internally inconsistent")
    if int(contract["qualification_slots_per_arm"]) != trajectories * qualification_slots_per_trajectory:
        raise ValueError("amendment qualification budgets are internally inconsistent")
    if int(contract["screen_slots_per_arm"]) != proposal_slots:
        raise ValueError("amendment screen/proposal opportunities differ")
    if require_external_seal:
        if amendment.get("status") != "amended_ready_for_execution":
            raise ValueError("Stage9 amendment is not execution-ready")
        seal = heldout.get("seal_sha256")
        if not isinstance(seal, str) or len(seal) != 64:
            raise ValueError("amendment has no valid external held-out seal")
    return amendment, ProtocolConfig(
        arms=arms,
        trajectories=trajectories,
        proposal_slots_per_arm=proposal_slots,
        proposal_slots_per_trajectory=proposal_slots_per_trajectory,
        qualification_slots_per_trajectory=qualification_slots_per_trajectory,
        rounds=int(protocol["rounds"]),
        proposal_opportunities_per_round=int(protocol["proposal_opportunities_per_round"]),
        token_cap_per_arm=int(contract["proposal_tokens_per_arm"]),
        token_cap_per_trajectory=int(per_trajectory["proposal_tokens"]),
        cpc_values=frozenset(int(value) for value in candidate_space["cpc_values"]),
        features=frozenset(candidate_space["features"]),
        comparisons=frozenset(candidate_space["predicate_operators"]),
        max_policy_depth=int(candidate_space["max_policy_depth"]),
        language=str(candidate_space["language"]),
        protocol_digest=str(amendment["protocol_digest"]),
        candidate_space_digest=canonical_digest(candidate_space),
        opaque_profile_ids=tuple(heldout["opaque_profile_ids"]),
    )


def canonical_policy_id(policy: dict[str, Any], config: ProtocolConfig) -> str:
    validate_policy(policy, config)
    return f"kir-policy:{canonical_digest(policy)}"


def validate_policy(node: object, config: ProtocolConfig, *, depth: int = 0) -> None:
    if not isinstance(node, dict):
        raise ValueError("policy node must be an object")
    if set(node) == {"kind", "chunks_per_cta"}:
        if node["kind"] != "leaf":
            raise ValueError("unsupported policy leaf kind")
        if type(node["chunks_per_cta"]) is not int or node["chunks_per_cta"] not in config.cpc_values:
            raise ValueError("policy leaf cpc is outside the frozen candidate space")
        return
    if set(node) != {"kind", "predicate", "then", "else"} or node.get("kind") != "node":
        raise ValueError("policy node must contain exactly kind/predicate/then/else")
    if depth >= config.max_policy_depth:
        raise ValueError("policy conditional depth exceeds the amendment")
    predicate = node["predicate"]
    if not isinstance(predicate, dict) or set(predicate) != {"feature", "operator", "threshold"}:
        raise ValueError("predicate must contain exactly feature/operator/threshold")
    if predicate["feature"] not in config.features or predicate["operator"] not in config.comparisons:
        raise ValueError("predicate is outside the frozen candidate space")
    if type(predicate["threshold"]) is not int or predicate["threshold"] < 0:
        raise ValueError("predicate threshold must be a non-negative integer")
    validate_policy(node["then"], config, depth=depth + 1)
    validate_policy(node["else"], config, depth=depth + 1)
    if canonical_digest(node["then"]) == canonical_digest(node["else"]):
        raise ValueError("policy predicate branches must be semantically distinct")


def resolve_policy(node: dict[str, Any], features: dict[str, int], config: ProtocolConfig) -> int:
    validate_policy(node, config)
    missing = config.features.difference(features)
    if missing:
        raise ValueError(f"profile is missing policy features: {sorted(missing)}")
    current = node
    while current["kind"] != "leaf":
        predicate = current["predicate"]
        observed = features[predicate["feature"]]
        passed = observed <= predicate["threshold"] if predicate["operator"] == "le" else observed > predicate["threshold"]
        current = current["then"] if passed else current["else"]
    return int(current["chunks_per_cta"])


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class ProposalSlot:
    arm: str
    trajectory_id: int
    proposal_slot: int
    policy: dict[str, Any] | None
    policy_id: str
    status: str
    error: str
    proposer_ids: tuple[str, ...]
    proposer_roles: tuple[str, ...]
    parent_policy_ids: tuple[str, ...]
    lineage_delta_refs: tuple[str, ...]

    def global_slot(self, config: ProtocolConfig) -> int:
        return self.proposal_slot


def load_proposal_manifest(path: Path, config: ProtocolConfig, *, require_complete: bool = True) -> tuple[dict[str, Any], tuple[ProposalSlot, ...]]:
    manifest = _read_json(path)
    accepted_statuses = {"proposal_frozen", "round_0_proposal_frozen", "round_1_proposal_frozen"}
    if manifest.get("schema_version") != 2 or manifest.get("status") not in accepted_statuses:
        raise ValueError("unexpected Stage9 proposal manifest schema/status")
    raw_slots = manifest.get("proposals")
    if not isinstance(raw_slots, list):
        raise ValueError("proposal manifest slots must be a list")
    supplied_manifest_digest = manifest.get("proposal_manifest_sha256")
    if supplied_manifest_digest is not None and supplied_manifest_digest != canonical_digest(raw_slots):
        raise ValueError("proposal manifest semantic digest mismatch")
    if set(manifest.get("arms", ())) != set(config.arms):
        raise ValueError("proposal manifest arms differ from the amendment")
    slots: list[ProposalSlot] = []
    seen_coordinates: set[tuple[str, int, int]] = set()
    for raw in raw_slots:
        if not isinstance(raw, dict):
            raise ValueError("proposal slot must be an object")
        arm = raw.get("arm")
        trajectory = raw.get("trajectory")
        proposal_slot = raw.get("proposal_slot")
        policy = raw.get("policy")
        if arm not in config.arms or type(trajectory) is not int or type(proposal_slot) is not int:
            raise ValueError("invalid arm/trajectory/proposal coordinate")
        first_slot = trajectory * config.proposal_slots_per_trajectory
        if not 0 <= trajectory < config.trajectories or not first_slot <= proposal_slot < first_slot + config.proposal_slots_per_trajectory:
            raise ValueError("proposal coordinate exceeds the frozen budget")
        coordinate = (arm, trajectory, proposal_slot)
        if coordinate in seen_coordinates:
            raise ValueError(f"duplicate proposal coordinate {coordinate}")
        seen_coordinates.add(coordinate)
        status = str(raw.get("status", ""))
        supplied_id = str(raw.get("policy_id", ""))
        if status == "valid":
            validate_policy(policy, config)
            computed_id = canonical_policy_id(policy, config)
            if supplied_id != computed_id:
                raise ValueError("supplied policy_id does not match canonical policy")
        else:
            computed_id = ""
            if policy is not None or supplied_id:
                raise ValueError("non-valid proposal must not carry executable policy semantics")
        slots.append(
            ProposalSlot(
                arm=arm,
                trajectory_id=trajectory,
                proposal_slot=proposal_slot,
                policy=policy,
                policy_id=computed_id,
                status=status,
                error=str(raw.get("error", "")),
                proposer_ids=tuple(raw.get("proposer_ids", ())),
                proposer_roles=tuple(raw.get("proposer_roles", ())),
                parent_policy_ids=tuple(raw.get("parent_policy_ids", ())),
                lineage_delta_refs=tuple(raw.get("lineage_delta_refs", ())),
            )
        )
    if require_complete:
        expected = {
            (arm, trajectory, slot)
            for arm in config.arms
            for trajectory in range(config.trajectories)
            for slot in range(trajectory * config.proposal_slots_per_trajectory, (trajectory + 1) * config.proposal_slots_per_trajectory)
        }
        if seen_coordinates != expected:
            raise ValueError("manifest must contain exactly 24 proposal slots per arm")
    return manifest, tuple(slots)


def rotating_fifo(slots: Iterable[ProposalSlot], config: ProtocolConfig) -> tuple[ProposalSlot, ...]:
    if config.rounds * config.proposal_opportunities_per_round != config.proposal_slots_per_trajectory:
        raise ValueError("amendment round opportunities do not match per-trajectory slots")
    by_coordinate = {(slot.arm, slot.global_slot(config)): slot for slot in slots}
    output = []
    for trajectory in range(config.trajectories):
        for proposal_round in range(config.rounds):
            for opportunity in range(config.proposal_opportunities_per_round):
                global_slot = trajectory * config.proposal_slots_per_trajectory + proposal_round * config.proposal_opportunities_per_round + opportunity
                rotation = (trajectory + proposal_round + opportunity) % len(config.arms)
                for offset in range(len(config.arms)):
                    arm = config.arms[(rotation + offset) % len(config.arms)]
                    key = (arm, global_slot)
                    if key in by_coordinate:
                        output.append(by_coordinate[key])
    if len(output) != len(by_coordinate):
        raise ValueError("FIFO input has duplicate or out-of-budget coordinates")
    return tuple(output)


def screen_round_plan(
    slots: Iterable[ProposalSlot],
    config: ProtocolConfig,
    round_index: int,
    existing_receipts: Iterable[dict[str, Any]] = (),
) -> tuple[tuple[ProposalSlot, bool], ...]:
    if not 0 <= round_index < config.rounds:
        raise ValueError("round_index is outside the amendment")
    existing = tuple(existing_receipts)
    coordinates = {(row["arm"], row["trajectory_id"], row["proposal_slot"]) for row in existing}
    measured_ids = {row["policy_id"] for row in existing if row.get("policy_id") and row.get("measured")}
    start = round_index * config.proposal_opportunities_per_round
    stop = start + config.proposal_opportunities_per_round
    plan = []
    for slot in rotating_fifo(slots, config):
        local_slot = slot.proposal_slot % config.proposal_slots_per_trajectory
        if not start <= local_slot < stop or (slot.arm, slot.trajectory_id, slot.proposal_slot) in coordinates:
            continue
        measured = slot.status == "valid" and slot.policy_id not in measured_ids
        plan.append((slot, measured))
        if measured:
            measured_ids.add(slot.policy_id)
    return tuple(plan)


def require_complete_screen_barrier(receipts: Iterable[dict[str, Any]], config: ProtocolConfig) -> None:
    rows = tuple(receipts)
    expected = {
        (arm, trajectory, slot)
        for arm in config.arms
        for trajectory in range(config.trajectories)
        for slot in range(
            trajectory * config.proposal_slots_per_trajectory,
            (trajectory + 1) * config.proposal_slots_per_trajectory,
        )
    }
    actual = {(row["arm"], row["trajectory_id"], row["proposal_slot"]) for row in rows}
    if actual != expected or {row.get("round_index") for row in rows} != set(range(config.rounds)):
        raise ValueError("qualification blocked until both screen rounds are complete")


def _heldout_profiles(config: ProtocolConfig) -> list[dict[str, Any]]:
    # Generate genuinely new shapes inside a preregistered envelope.  A fresh
    # seal therefore changes the hidden workload, not merely its nonce.
    totals = [288, 336, 400, 464] * 2
    profiles = []
    for index, total_chunks in enumerate(totals):
        sequence_count = (1, 2, 4)[secrets.randbelow(3)]
        remaining = total_chunks - sequence_count * 4
        chunks = [4] * sequence_count
        for position in range(sequence_count - 1):
            take = secrets.randbelow(remaining + 1)
            chunks[position] += take
            remaining -= take
        chunks[-1] += remaining
        # Secretly permute the skew allocation and use non-multiple lengths
        # while preserving the committed total chunk count.
        for position in range(sequence_count - 1, 0, -1):
            other = secrets.randbelow(position + 1)
            chunks[position], chunks[other] = chunks[other], chunks[position]
        lengths = [chunk * 16 - secrets.randbelow(16) for chunk in chunks]
        profiles.append({
            "label": f"randomized_{index}", "num_heads": 12,
            "seq_lens": lengths, "packed": sequence_count > 1,
            "seed": 20000 + secrets.randbelow(1_000_000_000),
        })
    if len(config.opaque_profile_ids) != len(profiles):
        raise ValueError("amendment opaque ID count does not match held-out generator")
    for profile, opaque_id in zip(profiles, config.opaque_profile_ids):
        profile["opaque_id"] = opaque_id
    return profiles


def create_heldout_secret(secret_path: Path, commitment_path: Path, config: ProtocolConfig) -> dict[str, Any]:
    if secret_path.exists() or commitment_path.exists():
        raise FileExistsError("refusing to overwrite held-out secret or commitment")
    nonce = secrets.token_hex(32)
    profiles = []
    for profile in _heldout_profiles(config):
        item = dict(profile)
        profiles.append(item)
    secret = {"schema_version": 1, "seal_nonce": nonce, "profiles": profiles}
    _write_json(secret_path, secret)
    os.chmod(secret_path, 0o600)
    commitment = {
        "schema_version": 1,
        "algorithm": "sha256-canonical-json-v1",
        "heldout_digest": canonical_digest(secret),
        "profile_count": len(profiles),
        "status": "sealed_before_policy_freeze",
    }
    _write_json(commitment_path, commitment)
    return commitment


def verify_heldout_commitment(secret_path: Path, commitment_path: Path, config: ProtocolConfig) -> dict[str, Any]:
    secret = _read_json(secret_path)
    commitment = _read_json(commitment_path)
    if commitment.get("heldout_digest") != canonical_digest(secret):
        raise ValueError("held-out secret does not match the preregistered commitment")
    profiles = secret.get("profiles")
    if not isinstance(profiles, list) or len(profiles) != commitment.get("profile_count"):
        raise ValueError("held-out profile count does not match commitment")
    if tuple(item.get("opaque_id") for item in profiles) != config.opaque_profile_ids:
        raise ValueError("held-out opaque IDs differ from the amendment")
    for item in profiles:
        if not isinstance(item, dict):
            raise ValueError("invalid held-out profile")
        lengths = item.get("seq_lens")
        if item.get("num_heads") != 12 or not isinstance(lengths, list) or not lengths:
            raise ValueError("held-out profile violates the H12 contract")
        total_chunks = sum((int(length) + 15) // 16 for length in lengths)
        if total_chunks not in (288, 336, 400, 464):
            raise ValueError("held-out profile lies outside the frozen randomized work envelope")
    return secret


def _profile_features(case: Any) -> dict[str, int]:
    chunks = tuple((int(length) + 15) // 16 for length in case.seq_lens)
    return {
        "total_chunks": sum(chunks),
        "max_chunks": max(chunks),
        "num_sequences": len(chunks),
        "resident_grid_capacity_ctas": RESIDENT_GRID_CAPACITY_CTAS,
    }


def development_profile_descriptor(cases: Iterable[Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "split": "stage6_plus_stage8_development",
        "profiles": [
            {
                "profile_id": case.name,
                "num_heads": int(case.num_heads),
                "seq_lens": [int(value) for value in case.seq_lens],
                "packed": bool(case.packed),
                "features": _profile_features(case),
            }
            for case in cases
        ],
    }


def _gpu_helpers() -> dict[str, Any]:
    import numpy as np
    import torch
    from flashinfer.testing import bench_gpu_time
    from bench_flash_kda_h12_physical_search import _bt16_prepare_grid
    from bench_flash_kda_h12_profile_matrix import PROFILE_CASES as STAGE6_CASES
    from bench_flash_kda_h12_workscale_matrix import PROFILE_CASES as STAGE8_CASES
    from bench_recurrent_kda_prefill import Case, _hardware_metadata, _make_case, _require_cupti, _timing_iteration_budget

    return locals()


def _assert_route(prepared: Any) -> None:
    metadata = prepared.metadata
    if metadata.get("variant") != EXPECTED_ROUTE or metadata.get("target") != EXPECTED_TARGET:
        raise RuntimeError(f"Stage9 route/target drift: {metadata}")
    if tuple(metadata.get("physical_variants", ())) != EXPECTED_PHYSICAL_VARIANTS:
        raise RuntimeError(f"Stage9 physical variant drift: {metadata}")


def _run_with_cpc(prepared: Any, cpc: int, bt16_context: Any) -> None:
    with bt16_context(cpc, True):
        prepared.candidate_run()


def _measure(prepared: Any, cpc: int, *, dry: int, repeat: int, helpers: dict[str, Any]) -> list[float]:
    bench_gpu_time = helpers["bench_gpu_time"]
    bt16_context = helpers["_bt16_prepare_grid"]
    return [
        float(value)
        for value in bench_gpu_time(
            lambda: _run_with_cpc(prepared, cpc, bt16_context),
            enable_cupti=True,
            cold_l2_cache=True,
            use_cuda_graph=False,
            dry_run_iters=dry,
            repeat_iters=repeat,
        )
    ]


def _correctness(prepared: Any, cpc: int, helpers: dict[str, Any]) -> dict[str, Any]:
    torch = helpers["torch"]
    bt16_context = helpers["_bt16_prepare_grid"]
    prepared.reset_state_pools()
    _run_with_cpc(prepared, BASELINE_CPC, bt16_context)
    torch.cuda.synchronize()
    baseline_output = prepared.candidate_output.clone()
    baseline_state = prepared.candidate_state_pool[0].clone()
    prepared.reset_state_pools()
    _run_with_cpc(prepared, cpc, bt16_context)
    torch.cuda.synchronize()
    output_delta = float((prepared.candidate_output.float() - baseline_output.float()).abs().max())
    state_delta = float((prepared.candidate_state_pool[0].float() - baseline_state.float()).abs().max())
    try:
        torch.testing.assert_close(prepared.candidate_output, baseline_output, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(prepared.candidate_state_pool[0], baseline_state, atol=1e-2, rtol=1e-2)
    except AssertionError:
        passed = False
    else:
        passed = True
    return {"passed": passed, "output_max_abs": output_delta, "final_state_max_abs": state_delta}


def _make_prepared(case: Any, *, rotations: int, helpers: dict[str, Any]) -> Any:
    with helpers["_bt16_prepare_grid"](BASELINE_CPC, True):
        prepared = helpers["_make_case"](
            case,
            state_rotations=rotations,
            candidate_route="dispatcher",
            candidate_backend="cake",
        )
    _assert_route(prepared)
    return prepared


def _bootstrap_speedup(baseline: list[float], candidate: list[float], seed: int) -> tuple[float, float, float]:
    np = _gpu_helpers()["np"]
    a = np.asarray(baseline)
    b = np.asarray(candidate)
    rng = np.random.default_rng(seed)
    draws = 10_000
    a_m = np.median(a[rng.integers(0, a.size, size=(draws, a.size))], axis=1)
    b_m = np.median(b[rng.integers(0, b.size, size=(draws, b.size))], axis=1)
    ratios = a_m / b_m
    return float(np.median(a) / np.median(b)), float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))


def _evaluate_screen(slot: ProposalSlot, cases: tuple[Any, ...], args: argparse.Namespace, helpers: dict[str, Any], config: ProtocolConfig) -> dict[str, Any]:
    np = helpers["np"]
    rows = []
    for case in cases:
        features = _profile_features(case)
        cpc = resolve_policy(slot.policy, features, config)
        prepared = _make_prepared(case, rotations=args.state_rotations, helpers=helpers)
        correctness = _correctness(prepared, cpc, helpers)
        baseline_samples = []
        samples = []
        if correctness["passed"]:
            prepared.reset_state_pools()
            baseline_samples = _measure(
                prepared, BASELINE_CPC, dry=args.screen_dry_run_iters,
                repeat=args.screen_repeat_iters, helpers=helpers,
            )
            prepared.reset_state_pools()
            samples = _measure(
                prepared, cpc, dry=args.screen_dry_run_iters,
                repeat=args.screen_repeat_iters, helpers=helpers,
            )
        rows.append(
            {
                "profile_id": case.name,
                "features": features,
                "resolved_cpc": cpc,
                "concrete_schedule_id": f"kir:{canonical_digest({'policy_id': slot.policy_id, 'profile': features, 'cpc': cpc})[:16]}",
                "correctness": correctness,
                "candidate_median_ms": float(np.median(samples)) if samples else None,
                "baseline_median_ms": float(np.median(baseline_samples)) if baseline_samples else None,
                "screen_log_speedup": (
                    log(float(np.median(baseline_samples)) / float(np.median(samples)))
                    if baseline_samples and samples else None
                ),
                "candidate_samples_ms": samples,
                "route": prepared.metadata["variant"],
                "target": prepared.metadata["target"],
                "physical_variants": prepared.metadata["physical_variants"],
            }
        )
        del prepared
        helpers["torch"].cuda.empty_cache()
    valid = all(
        row["correctness"]["passed"]
        and row["candidate_median_ms"] is not None
        and row["baseline_median_ms"] is not None
        for row in rows
    )
    # A raw latency score cannot be compared across the mandatory inter-round
    # job boundary because clocks and thermal state can drift.  Rank by the
    # within-event paired baseline ratio instead.
    score = fmean(row["screen_log_speedup"] for row in rows) if valid else None
    return {"status": "ok" if valid else "correctness_failed", "score": score, "rows": rows}


def _paired_policy(policy: dict[str, Any], cases: tuple[Any, ...], *, order_seed: int, args: argparse.Namespace, helpers: dict[str, Any], config: ProtocolConfig, heldout: bool = False) -> dict[str, Any]:
    np = helpers["np"]
    rows = []
    for profile_index, case in enumerate(cases):
        features = _profile_features(case)
        cpc = resolve_policy(policy, features, config)
        prepared = _make_prepared(case, rotations=args.state_rotations, helpers=helpers)
        correctness = _correctness(prepared, cpc, helpers)
        order = ("baseline", "candidate", "candidate", "baseline") if (order_seed + profile_index) % 2 == 0 else ("candidate", "baseline", "baseline", "candidate")
        samples = {"baseline": [], "candidate": []}
        if correctness["passed"]:
            for name in order:
                selected_cpc = BASELINE_CPC if name == "baseline" else cpc
                prepared.reset_state_pools()
                samples[name].extend(_measure(prepared, selected_cpc, dry=args.qualification_dry_run_iters, repeat=args.qualification_repeat_iters, helpers=helpers))
        speedup, lower, upper = _bootstrap_speedup(samples["baseline"], samples["candidate"], int(case.seed)) if samples["baseline"] and samples["candidate"] else (0.0, 0.0, 0.0)
        rows.append(
            {
                "profile_id": getattr(case, "opaque_id", case.name),
                "features": features,
                "resolved_cpc": cpc,
                "correctness": correctness,
                "baseline_samples_ms": samples["baseline"],
                "candidate_samples_ms": samples["candidate"],
                "speedup": speedup,
                "bootstrap_95": [lower, upper],
                "pair_order": "/".join(order),
                "route": prepared.metadata["variant"],
                "target": prepared.metadata["target"],
                "physical_variants": prepared.metadata["physical_variants"],
                "split": "heldout" if heldout else "development",
            }
        )
        del prepared
        helpers["torch"].cuda.empty_cache()
    valid = all(row["correctness"]["passed"] for row in rows)
    metric = fmean(log(max(row["speedup"], 1e-300)) for row in rows) if valid else None
    return {"status": "ok" if valid else "correctness_failed", "uniform_log_speedup": metric, "geomean_speedup": exp(metric) if valid else None, "rows": rows}


def _screen_rank(result: dict[str, Any]) -> float:
    score = result.get("score")
    return float(score) if score is not None else float("-inf")


def _baseline_slot(arm: str, trajectory: int, config: ProtocolConfig) -> ProposalSlot:
    policy = {"kind": "leaf", "chunks_per_cta": BASELINE_CPC}
    return ProposalSlot(
        arm=arm,
        trajectory_id=trajectory,
        proposal_slot=trajectory * config.proposal_slots_per_trajectory,
        policy=policy,
        policy_id=canonical_policy_id(policy, config),
        status="synthetic_baseline_no_winner",
        error="no verifier-clean screened candidate",
        proposer_ids=(),
        proposer_roles=(),
        parent_policy_ids=(),
        lineage_delta_refs=(),
    )


def _select_qualification_slots(slots: tuple[ProposalSlot, ...], screen_by_policy: dict[str, dict[str, Any]], config: ProtocolConfig) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for arm in config.arms:
        arm_slots = [slot for slot in slots if slot.arm == arm]
        selected: list[dict[str, Any]] = []
        for trajectory in range(config.trajectories):
            candidates = [
                slot
                for slot in arm_slots
                if slot.trajectory_id == trajectory
                and slot.status == "valid"
                and slot.policy_id in screen_by_policy
                and screen_by_policy[slot.policy_id].get("status") == "ok"
            ]
            winner = max(candidates, key=lambda item: (_screen_rank(screen_by_policy[item.policy_id]), -item.proposal_slot)) if candidates else _baseline_slot(arm, trajectory, config)
            selected.append({"kind": "trajectory_primary", "trajectory_id": trajectory, "slot": winner})
            remaining = [slot for slot in candidates if slot.policy_id != winner.policy_id]
            if arm == "multi_role_conclusions":
                compound = [slot for slot in remaining if len(set(slot.parent_policy_ids)) >= 2]
                diagnostic_pool = compound or remaining
            else:
                diagnostic_pool = remaining
            diagnostic = max(diagnostic_pool, key=lambda item: (_screen_rank(screen_by_policy[item.policy_id]), -item.proposal_slot)) if diagnostic_pool else winner
            selected.append({"kind": "trajectory_diagnostic", "trajectory_id": trajectory, "slot": diagnostic})
        output[arm] = selected
    return output


def _development_context(manifest: dict[str, Any], helpers: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any], str]:
    cases = tuple(helpers["STAGE6_CASES"]) + tuple(helpers["STAGE8_CASES"])
    if len(cases) != 8 or len({case.name for case in cases}) != 8:
        raise RuntimeError("Stage9 development split must contain Stage6+Stage8 eight profiles")
    development = development_profile_descriptor(cases)
    digest = canonical_digest(development)
    return cases, development, digest


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_screen_round(args: argparse.Namespace) -> None:
    helpers = _gpu_helpers()
    helpers["_require_cupti"]()
    amendment, config = load_amendment(args.amendment, require_external_seal=True)
    if not 0 <= args.round_index < config.rounds:
        raise ValueError("round_index is outside the amendment")
    commitment = _read_json(args.heldout_commitment)
    if amendment["protocol"]["heldout"]["seal_sha256"] != commitment.get("heldout_digest"):
        raise ValueError("amendment held-out seal does not match the supplied commitment")
    manifest, slots = load_proposal_manifest(args.manifest, config, require_complete=False)
    cases, _development, development_digest = _development_context(manifest, helpers)
    manifest_sha = _file_sha256(args.manifest)
    existing = _read_jsonl(args.screen_jsonl)
    expected_protocol = config.protocol_digest
    if any(row.get("protocol_digest") != expected_protocol for row in existing):
        raise ValueError("existing screen receipt belongs to another protocol")
    existing_coordinates = {(row["arm"], row["trajectory_id"], row["proposal_slot"]) for row in existing}
    expected_round_coordinates = {
        (arm, trajectory, trajectory * config.proposal_slots_per_trajectory + local_slot)
        for arm in config.arms
        for trajectory in range(config.trajectories)
        for local_slot in range(
            args.round_index * config.proposal_opportunities_per_round,
            (args.round_index + 1) * config.proposal_opportunities_per_round,
        )
    }
    available_coordinates = {(slot.arm, slot.trajectory_id, slot.proposal_slot) for slot in slots}
    if not expected_round_coordinates.issubset(available_coordinates | existing_coordinates):
        raise ValueError("screen round manifest is missing reserved proposal opportunities")
    screen_by_policy: dict[str, dict[str, Any]] = {}
    first_coordinate: dict[str, dict[str, Any]] = {}
    for row in existing:
        if row.get("policy_id") and row["result"].get("status") != "proposal_not_verifier_clean":
            screen_by_policy.setdefault(row["policy_id"], row["result"])
            first_coordinate.setdefault(row["policy_id"], {"arm": row["arm"], "trajectory_id": row["trajectory_id"], "proposal_slot": row["proposal_slot"]})
    appended = []
    for queue_index, (slot, planned_measurement) in enumerate(screen_round_plan(slots, config, args.round_index, existing)):
        valid = slot.status == "valid"
        cache_key = canonical_digest({"policy_id": slot.policy_id, "profile_split_digest": development_digest, "protocol_digest": config.protocol_digest}) if valid else None
        measured = valid and slot.policy_id not in screen_by_policy
        if measured != planned_measurement:
            raise RuntimeError("screen cache plan diverged from durable receipts")
        if measured:
            event_started = time.perf_counter()
            try:
                screen_by_policy[slot.policy_id] = _evaluate_screen(slot, cases, args, helpers, config)
            except Exception as error:
                screen_by_policy[slot.policy_id] = {"status": "error", "score": None, "error_type": type(error).__name__, "error": str(error), "rows": []}
            screen_by_policy[slot.policy_id]["evaluation_wall_seconds"] = time.perf_counter() - event_started
            first_coordinate[slot.policy_id] = {"arm": slot.arm, "trajectory_id": slot.trajectory_id, "proposal_slot": slot.proposal_slot}
        elif not valid:
            result = {"status": "proposal_not_verifier_clean", "score": None, "error": slot.error, "rows": []}
        receipt = {
            "schema_version": 2,
            "phase": "screen",
            "round_index": args.round_index,
            "queue_index_within_round": queue_index,
            "arm": slot.arm,
            "trajectory_id": slot.trajectory_id,
            "proposal_slot": slot.proposal_slot,
            "policy_id": slot.policy_id,
            "policy": slot.policy,
            "screen_slot_charged": True,
            "measured": measured,
            "cache_key": cache_key,
            "reused_from": None if measured or not valid else first_coordinate[slot.policy_id],
            "evaluation_wall_seconds": screen_by_policy[slot.policy_id]["evaluation_wall_seconds"] if measured else 0.0,
            "manifest_sha256": manifest_sha,
            "protocol_digest": config.protocol_digest,
            "result": screen_by_policy[slot.policy_id] if valid else result,
        }
        appended.append(receipt)
    _write_jsonl(args.screen_jsonl, [*existing, *appended])
    _write_json(
        args.summary_json,
        {
            "schema_version": 2,
            "phase": "screen-round",
            "round_index": args.round_index,
            "new_receipts": len(appended),
            "total_receipts": len(existing) + len(appended),
            "protocol_digest": config.protocol_digest,
            "manifest_sha256": manifest_sha,
            "screen_jsonl_sha256": _file_sha256(args.screen_jsonl),
        },
    )


def run_qualification(args: argparse.Namespace) -> None:
    helpers = _gpu_helpers()
    helpers["_require_cupti"]()
    amendment, config = load_amendment(args.amendment, require_external_seal=True)
    commitment = _read_json(args.heldout_commitment)
    if amendment["protocol"]["heldout"]["seal_sha256"] != commitment.get("heldout_digest"):
        raise ValueError("amendment held-out seal does not match commitment")
    manifest, slots = load_proposal_manifest(args.manifest, config)
    cases, development, development_digest = _development_context(manifest, helpers)
    screen_receipts = _read_jsonl(args.screen_jsonl)
    manifest_sha = _file_sha256(args.manifest)
    if any(row.get("protocol_digest") != config.protocol_digest for row in screen_receipts):
        raise ValueError("screen receipts are not bound to this amendment")
    for row in screen_receipts:
        expected_cache_key = canonical_digest({"policy_id": row["policy_id"], "profile_split_digest": development_digest, "protocol_digest": config.protocol_digest}) if row.get("policy_id") else None
        if row.get("cache_key") != expected_cache_key:
            raise ValueError("screen receipt cache key mismatch")
    require_complete_screen_barrier(screen_receipts, config)
    final_slots = {(slot.arm, slot.trajectory_id, slot.proposal_slot): slot for slot in slots}
    for row in screen_receipts:
        slot = final_slots[(row["arm"], row["trajectory_id"], row["proposal_slot"])]
        if row.get("policy_id") != slot.policy_id or row.get("policy") != slot.policy:
            raise ValueError("incremental screen receipt differs from the final proposal manifest")
    screen_by_policy = {row["policy_id"]: row["result"] for row in screen_receipts if row.get("policy_id")}
    selections = _select_qualification_slots(slots, screen_by_policy, config)
    qualification_receipts = []
    qualification_results: dict[tuple[str, int], dict[str, Any]] = {}
    queue_index = 0
    for trajectory in range(config.trajectories):
      for within_trajectory in range(config.qualification_slots_per_trajectory):
        selection_index = trajectory * config.qualification_slots_per_trajectory + within_trajectory
        rotation = (trajectory + within_trajectory) % len(config.arms)
        for arm_offset in range(len(config.arms)):
            arm = config.arms[(rotation + arm_offset) % len(config.arms)]
            selection = selections[arm][selection_index]
            slot = selection["slot"]
            event_started = time.perf_counter()
            try:
                result = _paired_policy(slot.policy, cases, order_seed=selection_index + config.arms.index(arm), args=args, helpers=helpers, config=config)
            except Exception as error:
                result = {"status": "error", "uniform_log_speedup": None, "geomean_speedup": None, "error_type": type(error).__name__, "error": str(error), "rows": []}
            event_wall_seconds = time.perf_counter() - event_started
            result["evaluation_wall_seconds"] = event_wall_seconds
            qualification_results[(arm, selection_index)] = result
            qualification_receipts.append(
                {
                    "schema_version": 1,
                    "phase": "qualification",
                    "queue_index": queue_index,
                    "arm": arm,
                    "selection_index": selection_index,
                    "selection_kind": selection["kind"],
                    "trajectory_id": selection["trajectory_id"],
                    "policy_id": slot.policy_id,
                    "policy": slot.policy,
                    "proposal_coordinate": {"trajectory_id": slot.trajectory_id, "proposal_slot": slot.proposal_slot},
                    "evaluation_wall_seconds": event_wall_seconds,
                    "result": result,
                }
            )
            queue_index += 1
    freeze_rows = []
    slots_by_id = {slot.policy_id: slot for slot in slots if slot.status == "valid"}
    for arm in config.arms:
        for trajectory in range(config.trajectories):
            first = trajectory * config.qualification_slots_per_trajectory
            selection_indices = range(first, first + config.qualification_slots_per_trajectory)
            eligible = [
                index for index in selection_indices
                if qualification_results[(arm, index)].get("status") == "ok"
                and qualification_results[(arm, index)].get("uniform_log_speedup") is not None
            ]
            if not eligible:
                raise RuntimeError("trajectory has no successful qualification candidate")
            selection_index = max(
                eligible,
                key=lambda index: (
                    qualification_results[(arm, index)]["uniform_log_speedup"],
                    selections[arm][index]["slot"].policy_id,
                ),
            )
            selection = selections[arm][selection_index]
            slot = selection["slot"]
            lineage_controls = []
            if arm == "multi_role_conclusions" and len(set(slot.parent_policy_ids)) >= 2:
                for parent_id in slot.parent_policy_ids[:2]:
                    parent = slots_by_id.get(parent_id)
                    if parent is not None:
                        lineage_controls.append({"policy_id": parent.policy_id, "policy": parent.policy})
            freeze_rows.append(
                {
                    "arm": arm,
                    "trajectory_id": trajectory,
                    "policy_id": slot.policy_id,
                    "policy": slot.policy,
                    "parent_policy_ids": list(slot.parent_policy_ids),
                    "lineage_delta_refs": list(slot.lineage_delta_refs),
                    "lineage_controls": lineage_controls,
                    "qualification": qualification_results[(arm, selection_index)],
                }
            )
    freeze = {
        "schema_version": 1,
        "status": "policy_frozen_before_heldout_reveal",
        "manifest_sha256": _file_sha256(args.manifest),
        "amendment_sha256": _file_sha256(args.amendment),
        "amendment_protocol_digest": config.protocol_digest,
        "heldout_commitment_sha256": _file_sha256(args.heldout_commitment),
        "heldout_digest": _read_json(args.heldout_commitment).get("heldout_digest"),
        "manifest_metadata_digest": canonical_digest({key: value for key, value in manifest.items() if key != "proposals"}),
        "development_profile_ids": [case.name for case in cases],
        "development_profile_split_digest": manifest["profile_split_digest"],
        "executable_development_profile_digest": canonical_digest(development),
        "primary_winners": freeze_rows,
    }
    freeze["winner_set_digest"] = canonical_digest(freeze_rows)
    _write_jsonl(args.screen_jsonl, screen_receipts)
    _write_jsonl(args.qualification_jsonl, qualification_receipts)
    _write_json(args.winner_freeze_json, freeze)
    summary = {
        "schema_version": 1,
        "phase": "qualification",
        "hardware": helpers["_hardware_metadata"](helpers["torch"].device("cuda")),
        "single_writer_fifo": True,
        "screen_slots": len(screen_receipts),
        "screen_round_barrier_passed": True,
        "qualification_slots": len(qualification_receipts),
        "amendment_protocol_digest": config.protocol_digest,
        "artifacts": {
            str(args.manifest): _file_sha256(args.manifest),
            str(args.screen_jsonl): _file_sha256(args.screen_jsonl),
            str(args.qualification_jsonl): _file_sha256(args.qualification_jsonl),
            str(args.winner_freeze_json): _file_sha256(args.winner_freeze_json),
        },
    }
    _write_json(args.summary_json, summary)


def _secret_cases(secret: dict[str, Any], Case: Any) -> tuple[Any, ...]:
    output = []
    for raw in secret["profiles"]:
        case = Case(raw["opaque_id"], int(raw["num_heads"]), tuple(int(value) for value in raw["seq_lens"]), bool(raw["packed"]), int(raw["seed"]))
        output.append(case)
    return tuple(output)


def run_heldout(args: argparse.Namespace) -> None:
    helpers = _gpu_helpers()
    helpers["_require_cupti"]()
    amendment, config = load_amendment(args.amendment, require_external_seal=True)
    secret = verify_heldout_commitment(args.heldout_secret, args.heldout_commitment, config)
    if amendment["protocol"]["heldout"]["seal_sha256"] != canonical_digest(secret):
        raise ValueError("amendment is not bound to this held-out secret")
    freeze = _read_json(args.winner_freeze_json)
    if freeze.get("status") != "policy_frozen_before_heldout_reveal" or len(freeze.get("primary_winners", ())) != len(config.arms) * config.trajectories:
        raise ValueError("heldout requires a complete pre-existing winner freeze")
    if freeze.get("amendment_protocol_digest") != config.protocol_digest:
        raise ValueError("winner freeze belongs to a different amendment")
    if freeze.get("winner_set_digest") != canonical_digest(freeze["primary_winners"]):
        raise ValueError("winner freeze digest mismatch")
    if freeze.get("heldout_commitment_sha256") != _file_sha256(args.heldout_commitment):
        raise ValueError("held-out commitment differs from the one bound into winner freeze")
    if freeze.get("heldout_digest") != canonical_digest(secret):
        raise ValueError("winner freeze was not bound to this held-out split")
    cases = _secret_cases(secret, helpers["Case"])
    by_coordinate = {(row["arm"], row["trajectory_id"]): row for row in freeze["primary_winners"]}
    receipts = []
    queue_index = 0
    for trajectory in range(config.trajectories):
        rotation = trajectory % len(config.arms)
        for offset in range(len(config.arms)):
            arm = config.arms[(rotation + offset) % len(config.arms)]
            winner = by_coordinate[(arm, trajectory)]
            event_started = time.perf_counter()
            try:
                result = _paired_policy(winner["policy"], cases, order_seed=trajectory + config.arms.index(arm), args=args, helpers=helpers, config=config, heldout=True)
            except Exception as error:
                result = {"status": "error", "uniform_log_speedup": None, "geomean_speedup": None, "error_type": type(error).__name__, "error": str(error), "rows": []}
            event_wall_seconds = time.perf_counter() - event_started
            result["evaluation_wall_seconds"] = event_wall_seconds
            receipts.append(
                {
                    "schema_version": 1,
                    "phase": "heldout",
                    "queue_index": queue_index,
                    "arm": arm,
                    "trajectory_id": trajectory,
                    "policy_id": winner["policy_id"],
                    "winner_set_digest": freeze["winner_set_digest"],
                    "evaluation_wall_seconds": event_wall_seconds,
                    "result": result,
                }
            )
            queue_index += 1
            if arm == "multi_role_conclusions":
                for control_index, control in enumerate(winner.get("lineage_controls", ())):
                    event_started = time.perf_counter()
                    try:
                        control_result = _paired_policy(control["policy"], cases, order_seed=trajectory + control_index + 1, args=args, helpers=helpers, config=config, heldout=True)
                    except Exception as error:
                        control_result = {"status": "error", "uniform_log_speedup": None, "geomean_speedup": None, "error_type": type(error).__name__, "error": str(error), "rows": []}
                    control_wall = time.perf_counter() - event_started
                    control_result["evaluation_wall_seconds"] = control_wall
                    receipts.append(
                        {
                            "schema_version": 1,
                            "phase": "heldout_lineage_control",
                            "queue_index": queue_index,
                            "arm": arm,
                            "trajectory_id": trajectory,
                            "query_role": f"frozen_lineage_control_{control_index + 1}",
                            "policy_id": control["policy_id"],
                            "winner_policy_id": winner["policy_id"],
                            "winner_set_digest": freeze["winner_set_digest"],
                            "evaluation_wall_seconds": control_wall,
                            "result": control_result,
                        }
                    )
                    queue_index += 1
    _write_jsonl(args.heldout_jsonl, receipts)
    reveal = {
        "schema_version": 1,
        "status": "revealed_after_policy_freeze",
        "commitment_sha256": _file_sha256(args.heldout_commitment),
        "heldout_digest": canonical_digest(secret),
        "winner_set_digest": freeze["winner_set_digest"],
        "seal_nonce": secret["seal_nonce"],
        "profiles": secret["profiles"],
    }
    _write_json(args.heldout_reveal_json, reveal)
    arm_metrics = {}
    for arm in config.arms:
        values = [row["result"]["uniform_log_speedup"] for row in receipts if row["arm"] == arm and row["phase"] == "heldout"]
        valid = all(value is not None for value in values)
        aggregate = fmean(values) if valid else None
        arm_metrics[arm] = {"trajectory_log_speedups": values, "uniform_log_speedup": aggregate, "geomean_speedup": exp(aggregate) if aggregate is not None else None}
    compound_tests = []
    for child in (row for row in receipts if row["phase"] == "heldout" and row["arm"] == "multi_role_conclusions"):
        winner = by_coordinate[(child["arm"], child["trajectory_id"])]
        parents = list(winner.get("parent_policy_ids", ()))
        controls = [
            row
            for row in receipts
            if row["phase"] == "heldout_lineage_control"
            and row["trajectory_id"] == child["trajectory_id"]
        ]
        if len(set(parents)) < 2 or len(controls) != len(parents):
            continue
        child_profiles = {row["profile_id"]: row for row in child["result"]["rows"]}
        contrasts = []
        for control in controls:
            parent_profiles = {row["profile_id"]: row for row in control["result"]["rows"]}
            per_profile_lowers = [
                log(float(child_profiles[profile_id]["bootstrap_95"][0]))
                - log(float(parent_profiles[profile_id]["bootstrap_95"][1]))
                for profile_id in child_profiles
            ]
            contrasts.append(
                {
                    "parent_policy_id": control["policy_id"],
                    "simultaneous_lower": min(per_profile_lowers),
                    "per_profile_conservative_lowers": per_profile_lowers,
                    "method": "min_profile_log_child_lower_minus_log_parent_upper",
                }
            )
        compound_tests.append(
            {
                "arm": child["arm"],
                "trajectory_id": child["trajectory_id"],
                "child_policy_id": child["policy_id"],
                "parent_policy_ids": parents,
                "child_minus_parent_log_speedup": contrasts,
            }
        )
    summary = {
        "schema_version": 1,
        "phase": "heldout",
        "hardware": helpers["_hardware_metadata"](helpers["torch"].device("cuda")),
        "single_writer_fifo": True,
        "heldout_was_sealed_until_freeze": True,
        "arm_metrics": arm_metrics,
        "compound_lineage_tests": compound_tests,
        "artifacts": {
            str(args.heldout_commitment): _file_sha256(args.heldout_commitment),
            str(args.winner_freeze_json): _file_sha256(args.winner_freeze_json),
            str(args.heldout_jsonl): _file_sha256(args.heldout_jsonl),
            str(args.heldout_reveal_json): _file_sha256(args.heldout_reveal_json),
        },
    }
    _write_json(args.summary_json, summary)
    if any(
        row["phase"] == "heldout" and row["result"].get("status") != "ok"
        for row in receipts
    ):
        raise RuntimeError("one or more held-out evaluations failed")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("create-heldout-seal", "screen-round", "qualification", "heldout"), required=True)
    parser.add_argument("--round-index", type=int)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--amendment", type=Path)
    parser.add_argument("--heldout-secret", type=Path)
    parser.add_argument("--heldout-commitment", type=Path)
    parser.add_argument("--heldout-reveal-json", type=Path)
    parser.add_argument("--screen-jsonl", type=Path)
    parser.add_argument("--qualification-jsonl", type=Path)
    parser.add_argument("--heldout-jsonl", type=Path)
    parser.add_argument("--winner-freeze-json", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--state-rotations", type=int, default=256)
    parser.add_argument("--screen-dry-run-iters", type=int, default=5)
    parser.add_argument("--screen-repeat-iters", type=int, default=25)
    parser.add_argument("--qualification-dry-run-iters", type=int, default=20)
    parser.add_argument("--qualification-repeat-iters", type=int, default=100)
    return parser


def _require_paths(args: argparse.Namespace, names: tuple[str, ...]) -> None:
    missing = [name for name in names if getattr(args, name) is None]
    if missing:
        raise SystemExit("missing required arguments: " + ", ".join("--" + name.replace("_", "-") for name in missing))


def main() -> None:
    args = _parser().parse_args()
    if args.mode == "create-heldout-seal":
        _require_paths(args, ("amendment", "heldout_secret", "heldout_commitment"))
        _amendment, config = load_amendment(args.amendment)
        _write_json(Path(str(args.heldout_commitment) + ".receipt"), create_heldout_secret(args.heldout_secret, args.heldout_commitment, config))
    elif args.mode == "screen-round":
        _require_paths(args, ("amendment", "manifest", "heldout_commitment", "screen_jsonl", "summary_json"))
        if args.round_index is None:
            raise SystemExit("--round-index is required for screen-round")
        run_screen_round(args)
    elif args.mode == "qualification":
        _require_paths(args, ("amendment", "manifest", "heldout_commitment", "screen_jsonl", "qualification_jsonl", "winner_freeze_json", "summary_json"))
        run_qualification(args)
    else:
        _require_paths(args, ("amendment", "heldout_secret", "heldout_commitment", "heldout_reveal_json", "heldout_jsonl", "winner_freeze_json", "summary_json"))
        run_heldout(args)


if __name__ == "__main__":
    main()
