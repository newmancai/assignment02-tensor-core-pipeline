#!/usr/bin/env python3
"""Run one sealed Stage 9 cpc-policy round in isolated, auditable Codex lanes.

Historical case-study runner only.  Its fixed tree grammar must not be used as
the general HMMA/tcgen05 exploration boundary; see ``SM100_OPEN_AGENT_PROTOCOL``
and ``kda_ir.research_memory`` for the open proposal and scoped-memory layers.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent
ATREX_ROOT = ROOT / "third_party" / "atrex-kernel-agent"
from kda_ir import (  # noqa: E402
    LineageFragment, PolicyFeature, PolicyLeaf, PolicyNode, PolicyPredicate,
    PredicateOperator, RuntimePolicyCandidate, canonical_policy_id,
    policy_subtree_id, verify_compound_lineage, verify_policy,
)

KNOWN_ARMS = (
    "single_generalist", "single_roleplay", "multi_homogeneous",
    "multi_role_no_share", "multi_role_conclusions",
)
SEEDS = (91001, 91002, 91003, 91004, 91005, 91006, 91007, 91008)
ROLES = ("capacity", "tail_skew", "resource", "critic_synthesizer")
SLOTS_PER_ROUND = 4
TOKEN_CAP_TRAJECTORY = 62_500
TOKEN_CAP_ARM = 500_000


def _atrex():
    if str(ATREX_ROOT) not in sys.path:
        sys.path.insert(0, str(ATREX_ROOT))
    from orchestrator.agent_runtime.adapter import CodexAdapter
    from orchestrator.durable_state import durable_write_json
    return CodexAdapter, durable_write_json


def _sha(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _semantic(expression) -> dict:
    if isinstance(expression, PolicyLeaf):
        return {"kind": "leaf", "chunks_per_cta": expression.chunks_per_cta}
    return {
        "kind": "node",
        "predicate": {"feature": expression.predicate.feature.value,
                      "operator": expression.predicate.operator.value,
                      "threshold": expression.predicate.threshold},
        "then": _semantic(expression.then_branch),
        "else": _semantic(expression.else_branch),
    }


def _from_semantic(raw: dict):
    if raw.get("kind") == "leaf":
        return PolicyLeaf(int(raw["chunks_per_cta"]))
    if raw.get("kind") != "node":
        raise ValueError("unknown semantic policy node")
    pred = raw["predicate"]
    return PolicyNode(
        PolicyPredicate(PolicyFeature(pred["feature"]), PredicateOperator(pred["operator"]), int(pred["threshold"])),
        _from_semantic(raw["then"]), _from_semantic(raw["else"]),
    )


def _subtrees(expression) -> tuple:
    if isinstance(expression, PolicyLeaf):
        return (expression,)
    return (expression, *_subtrees(expression.then_branch), *_subtrees(expression.else_branch))


def _parse_flat_policy(raw: dict, *, origin_id: str,
                       parents: dict[str, RuntimePolicyCandidate],
                       require_compound: bool = False):
    nodes, visiting, visited = raw["nodes"], set(), set()

    def build(index: int):
        if not 0 <= index < len(nodes):
            raise ValueError("child index outside node array")
        if index in visiting:
            raise ValueError("policy node cycle")
        visiting.add(index); visited.add(index)
        node = nodes[index]
        if node["kind"] == "leaf":
            if not (node["feature"] == node["cmp"] == "none" and node["threshold"] == 0
                    and node["then_index"] == node["else_index"] == -1 and node["cpc"] > 0):
                raise ValueError("malformed leaf node")
            result = PolicyLeaf(int(node["cpc"]))
        elif node["kind"] == "branch":
            if not (node["feature"] != "none" and node["cmp"] in {"le", "gt"}
                    and node["threshold"] > 0 and node["cpc"] == 0
                    and node["then_index"] >= 0 and node["else_index"] >= 0):
                raise ValueError("malformed branch node")
            result = PolicyNode(
                PolicyPredicate(PolicyFeature(node["feature"]), PredicateOperator(node["cmp"]), int(node["threshold"])),
                build(int(node["then_index"])), build(int(node["else_index"])),
            )
        else:
            raise ValueError("unknown node kind")
        visiting.remove(index)
        return result

    root = build(int(raw["root_index"]))
    if visited != set(range(len(nodes))):
        raise ValueError("unreachable policy nodes are forbidden")
    parent_ids = tuple(dict.fromkeys(raw.get("parent_policy_ids", ())))
    if require_compound and len(parent_ids) < 2:
        raise ValueError("synthesizer requires two distinct parent IDs")
    candidate = RuntimePolicyCandidate(root, parent_ids=parent_ids, origin_proposal_ids=(origin_id,))
    diagnostics = verify_policy(candidate)
    if diagnostics:
        raise ValueError(";".join(item.code for item in diagnostics))
    if parent_ids:
        child_ids = {policy_subtree_id(item) for item in _subtrees(root)}
        choices = {}
        for parent_id in parent_ids:
            if parent_id not in parents:
                raise ValueError("unknown parent policy ID")
            choices[parent_id] = sorted(child_ids & {policy_subtree_id(item) for item in _subtrees(parents[parent_id].root)})
            if not choices[parent_id]:
                raise ValueError("child inherits no exact parent subtree")
        used, fragments = set(), []
        for parent_id in sorted(parent_ids, key=lambda item: (len(choices[item]), item)):
            selected = next((item for item in choices[parent_id] if item not in used), None)
            if selected is None:
                raise ValueError("parents do not contribute distinct subtrees")
            used.add(selected); fragments.append(LineageFragment(parent_id, selected))
        candidate = RuntimePolicyCandidate(root, parent_ids=parent_ids,
                                           origin_proposal_ids=(origin_id,),
                                           lineage_fragments=tuple(fragments))
        if len(parent_ids) >= 2:
            diagnostics = verify_compound_lineage(candidate, parents)
            if diagnostics:
                raise ValueError(";".join(item.code for item in diagnostics))
    return candidate


def _tool_events(stdout: str) -> list[dict]:
    violations = []
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        item = event.get("item")
        if isinstance(item, dict) and item.get("type") not in {None, "agent_message", "reasoning"}:
            violations.append(event)
    return violations


def _lanes(arm: str) -> tuple[str, ...]:
    if arm in {"single_generalist", "single_roleplay"}:
        return ("single",)
    if arm == "multi_homogeneous":
        return tuple(f"generalist_{i}" for i in range(4))
    return ROLES


def _instruction(arm: str, lane: str, count: int, memory: list[dict], compound: bool) -> str:
    descriptions = {
        "single_generalist": "Act as one generalist and produce a diverse portfolio.",
        "single_roleplay": "Sequentially role-play profile analyst, explorer, critic, and reviewer in one logical context.",
        "multi_homogeneous": "Act as an independent generalist lane; you cannot see other lanes.",
        "multi_role_no_share": f"Act only as specialized role {lane}; you cannot see other lanes.",
        "multi_role_conclusions": f"Act only as specialized role {lane}; shared conclusions were released after the round barrier.",
    }
    method = descriptions[arm]
    if memory and lane != "critic_synthesizer":
        method += " Propose a role-specific counterfactual mutation; do not repeat the incumbent policy semantics."
    if compound:
        method += " Produce one compound policy with at least two supplied parent_policy_ids and a distinct exact subtree from each."
    return f"{method}\nReturn exactly {count} proposal(s).\nVerified arm-scoped memory:\n" + json.dumps(memory, sort_keys=True, separators=(",", ":"))


def _safe_env(codex_home: Path) -> dict[str, str]:
    names = ("PATH", "SSL_CERT_FILE", "SSL_CERT_DIR", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY")
    env = {name: os.environ[name] for name in names if name in os.environ}
    env.update({"HOME": str(codex_home), "CODEX_HOME": str(codex_home), "TMPDIR": str(codex_home / "tmp")})
    return env


def _copy_identity(destination: Path) -> None:
    source = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))
    for name in ("auth.json", "installation_id", "models_cache.json"):
        if (source / name).is_file():
            shutil.copy2(source / name, destination / name)


def _run_request(*, output_dir: Path, arm: str, trajectory: int, round_index: int,
                 seed: int, lane: str, count: int, common_text: str,
                 development: dict, memory: list[dict], model: str,
                 reasoning_effort: str, timeout_s: int, require_compound: bool) -> dict:
    request_id = f"{arm}.t{trajectory}.r{round_index}.{lane}"
    archive = output_dir / "requests" / arm / f"trajectory_{trajectory}" / f"round_{round_index}" / lane
    archive.mkdir(parents=True, exist_ok=True, mode=0o700)
    result_path, event_path, receipt_path = archive / "proposal_output.json", archive / "events.jsonl", archive / "receipt.json"
    prompt = "\n\n".join((
        common_text,
        f"Experiment identity: arm={arm} trajectory={trajectory} round={round_index} lane={lane} seed={seed}.",
        _instruction(arm, lane, count, memory, require_compound),
        "Development evidence:\n" + json.dumps(development, sort_keys=True, separators=(",", ":")),
        "Do not call tools, inspect files, or discuss hidden/held-out data. Return only schema-valid JSON.",
    ))
    schema = (ROOT / "stage9_agent_output_schema.json").read_bytes()
    request_digest = _sha({"prompt_sha256": sha256(prompt.encode()).hexdigest(), "model": model,
                           "reasoning_effort": reasoning_effort, "schema_sha256": sha256(schema).hexdigest()})
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("request_digest") != request_digest:
            raise RuntimeError(f"stale request cache for {request_id}")
        return receipt
    Adapter, durable_write_json = _atrex()
    base = {"request_id": request_id, "request_digest": request_digest, "arm": arm,
            "trajectory": trajectory, "round": round_index, "seed": seed, "lane": lane,
            "requested_proposals": count, "model": model, "reasoning_effort": reasoning_effort,
            "prompt_sha256": sha256(prompt.encode()).hexdigest(), "status": "started",
            "started_at_utc": datetime.now(timezone.utc).isoformat()}
    durable_write_json(receipt_path, base, indent=2, ensure_ascii=False)
    stdout = stderr = ""; exit_status = -1; started = time.time()
    with tempfile.TemporaryDirectory(prefix="stage9-agent-") as temp:
        workspace, home = Path(temp) / "workspace", Path(temp) / "codex-home"
        workspace.mkdir(mode=0o700); home.mkdir(mode=0o700); (home / "tmp").mkdir(mode=0o700)
        _copy_identity(home)
        schema_path, temp_result = workspace / "schema.json", workspace / "result.json"
        schema_path.write_bytes(schema)
        command = ["codex", "exec", "--json", "--color", "never", "--ephemeral",
                   "--ignore-user-config", "--skip-git-repo-check", "--sandbox", "read-only",
                   "-m", model, "-c", f'model_reasoning_effort="{reasoning_effort}"',
                   "--output-schema", str(schema_path), "-o", str(temp_result), "-C", str(workspace), prompt]
        try:
            completed = subprocess.run(command, cwd=workspace, env=_safe_env(home), capture_output=True,
                                       text=True, timeout=timeout_s, check=False)
            stdout, stderr, exit_status = completed.stdout, completed.stderr, completed.returncode
            if temp_result.exists():
                shutil.copy2(temp_result, result_path)
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
    event_path.write_text(stdout)
    _events, usage = Adapter().normalize_stream(stdout)
    violations = _tool_events(stdout)
    status, error = "ok", ""
    if exit_status == -1: status, error = "timeout", stderr[-1000:]
    elif exit_status != 0: status, error = "agent_failed", stderr[-1000:]
    elif violations: status, error = "tool_access_rejected", "tool/file-access event observed"
    elif usage.total_tokens is None: status, error = "usage_unavailable", "exact token charge required"
    elif not result_path.exists(): status, error = "missing_output", "structured output absent"
    receipt = {**base, "status": status, "error": error, "exit_status": exit_status,
               "elapsed_s": time.time() - started, "token_usage": asdict(usage),
               "tool_event_count": len(violations),
               "output_path": str(result_path.relative_to(output_dir)),
               "events_path": str(event_path.relative_to(output_dir)),
               "finished_at_utc": datetime.now(timezone.utc).isoformat()}
    durable_write_json(receipt_path, receipt, indent=2, ensure_ascii=False)
    return receipt


def _raw(output_dir: Path, receipt: dict) -> list[dict]:
    if receipt["status"] != "ok":
        return []
    return list(json.loads((output_dir / receipt["output_path"]).read_text()).get("proposals", ()))[:receipt["requested_proposals"]]


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _feedback(receipts: list[dict], previous: list[dict], *, arm: str, trajectory: int, lane: str) -> list[dict]:
    proposals = {row["policy_id"]: row for row in previous if row.get("arm") == arm
                 and row.get("trajectory") == trajectory and row.get("status") == "valid"}
    rows = [row for row in receipts if row.get("arm") == arm
            and row.get("trajectory_id", row.get("trajectory")) == trajectory
            and row.get("policy_id") in proposals]
    rows.sort(key=lambda row: (
        row.get("result", {}).get("status") != "ok",
        -float(row.get("result", {}).get("score") or float("-inf")),
        row.get("policy_id", ""),
    ))
    output = []
    for rank, row in enumerate(rows):
        proposal = proposals[row["policy_id"]]
        if arm in {"multi_homogeneous", "multi_role_no_share"} and proposal.get("lane") != lane:
            continue
        result = row.get("result", {})
        profile_conclusions = [
            {
                "profile_id": item.get("profile_id"),
                "resolved_cpc": item.get("resolved_cpc"),
                "candidate_median_ms": item.get("candidate_median_ms"),
                "correct": item.get("correctness", {}).get("passed"),
            }
            for item in result.get("rows", ())
        ]
        output.append({"rank": rank, "policy_id": row["policy_id"], "policy": proposal["policy"],
                       "origin_lane": proposal.get("lane"), "screen_status": result.get("status"),
                       "score": result.get("score"), "profile_conclusions": profile_conclusions})
    return output


def _parents(memory: list[dict]) -> dict[str, RuntimePolicyCandidate]:
    result = {}
    for row in memory:
        candidate = RuntimePolicyCandidate(_from_semantic(row["policy"]))
        if canonical_policy_id(candidate) != row["policy_id"]:
            raise ValueError("feedback policy ID mismatch")
        result[row["policy_id"]] = candidate
    return result


def _parse_receipt_rows(output_dir: Path, current: list[dict], specs: list[tuple]) -> list[dict]:
    rows = []
    for receipt, (lane, _count, memory, compound) in zip(current, specs):
        parents = _parents(memory)
        for index, raw in enumerate(_raw(output_dir, receipt)):
            origin = f"{receipt['request_id']}.p{index}"
            try:
                candidate = _parse_flat_policy(
                    raw, origin_id=origin, parents=parents, require_compound=compound
                )
                status, error = "valid", ""
                policy_id, policy = canonical_policy_id(candidate), _semantic(candidate.root)
            except Exception as exc:
                status, error, policy_id, policy = "invalid", f"{type(exc).__name__}:{exc}", "", None
            rows.append({
                "origin_proposal_id": origin, "lane": lane, "status": status,
                "error": error, "policy_id": policy_id, "policy": policy,
                "hypothesis": raw.get("hypothesis", ""),
                "parent_policy_ids": raw.get("parent_policy_ids", []),
                "evidence_refs": raw.get("evidence_refs", []),
                "lineage_delta_refs": raw.get("lineage_delta_refs", []),
            })
    return rows


def _memory(output_dir: Path, arm: str, trajectory: int, round_index: int, conclusions: list[dict]) -> str:
    path = output_dir / "memory" / arm / f"trajectory_{trajectory}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _jsonl(path) if path.exists() else []
    if any(row["round"] == round_index for row in existing):
        return existing[-1]["entry_hash"]
    body = {"round": round_index, "previous_hash": existing[-1]["entry_hash"] if existing else "0" * 64,
            "conclusions": conclusions}
    body["entry_hash"] = _sha(body)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush(); os.fsync(handle.fileno())
    return body["entry_hash"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-index", type=int, choices=(0, 1), required=True)
    parser.add_argument("--screen-receipts", type=Path)
    parser.add_argument("--round0-manifest", type=Path)
    parser.add_argument("--synth-attempt-tag", default="")
    parser.add_argument("--output-dir", type=Path, default=Path("evidence/stage9_agent_runs_v2"))
    parser.add_argument("--model", default="gpt-5.3-codex-spark")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--amendment", type=Path, default=Path("evidence/stage9_multiagent_ablation_amendment_v2.json"))
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    common_path, schema_path = ROOT / "STAGE9_AGENT_PROTOCOL.md", ROOT / "stage9_agent_output_schema.json"
    dev_path = ROOT / "evidence/stage9_development_summary.json"
    common, development = common_path.read_text(), json.loads(dev_path.read_text())
    amendment_path = args.amendment if args.amendment.is_absolute() else ROOT / args.amendment
    amendment_bytes = amendment_path.read_bytes(); amendment = json.loads(amendment_bytes)
    if amendment.get("status") != "amended_ready_for_execution":
        raise RuntimeError("Stage 9 amendment is not sealed and ready")
    protocol, backend = amendment["protocol"], amendment["protocol"]["proposal_backend"]
    for key, path in {"common_prompt_sha256": common_path, "output_schema_sha256": schema_path,
                      "development_summary_sha256": dev_path, "runner_sha256": Path(__file__).resolve()}.items():
        if sha256(path.read_bytes()).hexdigest() != backend[key]:
            raise RuntimeError(f"frozen input drift: {key}")
    contract = protocol["budget"]["contract"]
    arms = tuple(protocol["arms"])
    seeds = tuple(protocol.get("trajectory_seeds", SEEDS[: int(contract["trajectories"])]))
    per_trajectory = protocol["budget"]["per_trajectory_caps"]
    slots_per_trajectory = int(per_trajectory["proposal_slots"])
    token_cap_trajectory = int(per_trajectory["proposal_tokens"])
    if (not arms or not set(arms).issubset(KNOWN_ARMS)
            or contract["trajectories"] != len(seeds)
            or contract["proposal_slots_per_arm"] != len(seeds) * slots_per_trajectory
            or protocol["rounds"] != 2
            or protocol["proposal_opportunities_per_round"] != SLOTS_PER_ROUND
            or backend["model"] != args.model or backend["reasoning_effort"] != args.reasoning_effort):
        raise RuntimeError("runner does not match frozen amendment")
    previous = []; screens = []
    if args.round_index == 1:
        prior_path = args.round0_manifest or (args.output_dir / "round_0_manifest.json")
        if not prior_path.exists() or args.screen_receipts is None:
            raise RuntimeError("round 1 requires round-0 manifest and GPU receipts")
        previous = json.loads(prior_path.read_text())["proposals"]
        receipt_path = args.screen_receipts if args.screen_receipts.is_absolute() else ROOT / args.screen_receipts
        screens = _jsonl(receipt_path)
        expected = {(r["arm"], r["trajectory"], r["proposal_slot"]) for r in previous}
        observed = {(r.get("arm"), r.get("trajectory_id", r.get("trajectory")), r.get("proposal_slot")) for r in screens}
        if not expected.issubset(observed):
            raise RuntimeError("round-0 GPU barrier incomplete")

    receipts, proposals = [], []
    for arm in arms:
        prior_arm_receipts = [
            row for row in (json.loads(prior_path.read_text()).get("agent_receipts", []) if args.round_index else [])
            if row.get("arm") == arm
        ]
        arm_tokens = sum(int(row.get("token_usage", {}).get("total_tokens") or 0) for row in prior_arm_receipts)
        for trajectory, seed in enumerate(seeds):
            specs = []
            sequential_synth = arm == "multi_role_conclusions" and args.round_index == 1
            initial_lanes = tuple(lane for lane in _lanes(arm) if not (sequential_synth and lane == "critic_synthesizer"))
            for lane in initial_lanes:
                memory = _feedback(screens, previous, arm=arm, trajectory=trajectory, lane=lane) if args.round_index else []
                _memory(args.output_dir, arm, trajectory, args.round_index, memory)
                specs.append((lane, SLOTS_PER_ROUND if len(_lanes(arm)) == 1 else 1, memory, False))
            requests = [dict(output_dir=args.output_dir, arm=arm, trajectory=trajectory,
                             round_index=args.round_index, seed=seed, lane=lane, count=count,
                             common_text=common, development=development, memory=memory,
                             model=args.model, reasoning_effort=args.reasoning_effort,
                             timeout_s=args.timeout, require_compound=compound)
                        for lane, count, memory, compound in specs]
            with ThreadPoolExecutor(max_workers=min(args.workers, len(requests))) as pool:
                current = [future.result() for future in [pool.submit(_run_request, **request) for request in requests]]
            rows = _parse_receipt_rows(args.output_dir, current, specs)
            if sequential_synth:
                base_memory = _feedback(screens, previous, arm=arm, trajectory=trajectory, lane="critic_synthesizer")
                new_parents = [
                    {"rank": len(base_memory) + index, "policy_id": row["policy_id"],
                     "policy": row["policy"], "origin_lane": row["lane"],
                     "screen_status": "typed_verified_round1_parent", "score": None,
                     "profile_conclusions": []}
                    for index, row in enumerate(rows) if row["status"] == "valid"
                ]
                synth_memory = base_memory + new_parents
                synth_lane = "critic_synthesizer" + (f"_{args.synth_attempt_tag}" if args.synth_attempt_tag else "")
                synth_spec = [(synth_lane, 1, synth_memory, True)]
                synth_request = dict(
                    output_dir=args.output_dir, arm=arm, trajectory=trajectory,
                    round_index=args.round_index, seed=seed, lane=synth_lane, count=1,
                    common_text=common, development=development, memory=synth_memory,
                    model=args.model, reasoning_effort=args.reasoning_effort,
                    timeout_s=args.timeout, require_compound=True,
                )
                synth_receipt = _run_request(**synth_request)
                current.append(synth_receipt)
                rows.extend(_parse_receipt_rows(args.output_dir, [synth_receipt], synth_spec))
            receipts.extend(current)
            prior_trajectory_receipts = [row for row in prior_arm_receipts if row.get("trajectory") == trajectory]
            charged_receipts = prior_trajectory_receipts + current
            token_total = sum(int(row.get("token_usage", {}).get("total_tokens") or 0) for row in charged_receipts)
            if any(row.get("token_usage", {}).get("total_tokens") is None for row in charged_receipts):
                token_total = token_cap_trajectory + 1
            if token_total > token_cap_trajectory:
                for row in current:
                    if row["status"] == "ok":
                        row["status"] = "trajectory_token_cap_exceeded"
            while len(rows) < SLOTS_PER_ROUND:
                rows.append({"origin_proposal_id": f"{arm}.t{trajectory}.r{args.round_index}.missing{len(rows)}",
                             "lane": "missing", "status": "missing_charged", "error": "reserved slot absent",
                             "policy_id": "", "policy": None, "hypothesis": "", "parent_policy_ids": [],
                             "evidence_refs": [], "lineage_delta_refs": []})
            for local_slot, row in enumerate(rows[:SLOTS_PER_ROUND]):
                row.update({"proposal_slot": trajectory * slots_per_trajectory + args.round_index * SLOTS_PER_ROUND + local_slot,
                            "round": args.round_index, "trajectory": trajectory, "seed": seed, "arm": arm})
                proposals.append(row)
            arm_tokens += sum(int(row.get("token_usage", {}).get("total_tokens") or 0) for row in current)
        if arm_tokens > int(contract["proposal_tokens_per_arm"]):
            raise RuntimeError(f"{arm} token cap exceeded: {arm_tokens}")

    manifest = {"schema_version": 2, "language": protocol["candidate_space"]["language"],
                "status": f"round_{args.round_index}_proposal_frozen", "round": args.round_index,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "amendment_sha256": sha256(amendment_bytes).hexdigest(),
                "amendment_protocol_digest": amendment["protocol_digest"],
                "protocol_digest": amendment["protocol_digest"],
                "candidate_space_digest": _sha(protocol["candidate_space"]),
                "arms": list(arms), "trajectory_seeds": list(seeds),
                "proposal_slots_per_arm": int(contract["proposal_slots_per_arm"]),
                "model": args.model, "reasoning_effort": args.reasoning_effort,
                "common_prompt_digest": sha256(common.encode()).hexdigest(),
                "common_prompt_sha256": sha256(common.encode()).hexdigest(),
                "arm_prompt_digests": backend["arm_protocol_digests"],
                "arm_protocol_digests": backend["arm_protocol_digests"],
                "model_digest": _sha({"model": args.model}),
                "model_config_digest": _sha({"model": args.model, "reasoning_effort": args.reasoning_effort}),
                "tool_manifest_digest": _sha({"tools": [], "policy": backend["tool_policy"]}),
                "prompt_protocol_digest": _sha({"common": sha256(common.encode()).hexdigest(), "arms": backend["arm_protocol_digests"]}),
                "development_summary_sha256": sha256(dev_path.read_bytes()).hexdigest(),
                "profile_split_digest": development.get("profile_split_digest", sha256(dev_path.read_bytes()).hexdigest()),
                "agent_receipts": receipts, "receipts": receipts, "proposals": proposals}
    manifest["proposal_manifest_sha256"] = _sha(proposals)
    _Adapter, durable = _atrex()
    round_path = args.output_dir / f"round_{args.round_index}_manifest.json"
    durable(round_path, manifest, indent=2, ensure_ascii=False)
    if args.round_index == 1:
        prior = json.loads(prior_path.read_text())
        merged = dict(manifest); merged.update({"status": "proposal_frozen", "round": None,
                                                "agent_receipts": prior["agent_receipts"] + receipts,
                                                "receipts": prior["receipts"] + receipts,
                                                "proposals": prior["proposals"] + proposals})
        merged["proposal_manifest_sha256"] = _sha(merged["proposals"])
        durable(args.output_dir / "proposal_manifest.json", merged, indent=2, ensure_ascii=False)
    print(round_path)


if __name__ == "__main__":
    main()
