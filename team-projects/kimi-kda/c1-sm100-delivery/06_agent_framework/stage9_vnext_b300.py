#!/usr/bin/env python3
"""Screen diversity-preserving Stage 9 mutations against their v5 parent."""

from __future__ import annotations

import argparse
import importlib.util
import json
from math import exp, log
from pathlib import Path
from statistics import fmean
import sys
import time


ROOT = Path(__file__).resolve().parent
BENCHMARK = ROOT / "stage3_remote" / "benchmarks" / "bench_flash_kda_stage9_ablation.py"

from kda_ir.stage9_policy import (  # noqa: E402
    PolicyEvaluationContext,
    PolicyLeaf,
    PolicyNode,
    RuntimePolicyCandidate,
    canonical_policy_id,
    evaluate_policy,
)
from kda_ir.stage9_search import SearchLane, deterministic_mutations, policy_niche  # noqa: E402


def _load_benchmark():
    spec = importlib.util.spec_from_file_location("stage9_vnext_base", BENCHMARK)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Stage 9 B300 benchmark")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _from_semantic(raw: dict):
    if raw["kind"] == "leaf":
        return PolicyLeaf(int(raw["chunks_per_cta"]))
    predicate = raw["predicate"]
    from kda_ir.stage9_policy import PolicyFeature, PolicyPredicate, PredicateOperator
    return PolicyNode(
        PolicyPredicate(
            PolicyFeature(predicate["feature"]),
            PredicateOperator(predicate["operator"]),
            int(predicate["threshold"]),
        ),
        _from_semantic(raw["then"]),
        _from_semantic(raw["else"]),
    )


def _semantic(expression):
    if isinstance(expression, PolicyLeaf):
        return {"kind": "leaf", "chunks_per_cta": expression.chunks_per_cta}
    return {
        "kind": "node",
        "predicate": {
            "feature": expression.predicate.feature.value,
            "operator": expression.predicate.operator.value,
            "threshold": expression.predicate.threshold,
        },
        "then": _semantic(expression.then_branch),
        "else": _semantic(expression.else_branch),
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _profile_context(case, resident_ctas_per_sm: int = 5) -> PolicyEvaluationContext:
    chunks = tuple((int(length) + 15) // 16 for length in case.seq_lens)
    return PolicyEvaluationContext(
        total_chunks=sum(chunks),
        max_chunks=max(chunks),
        num_sequences=len(chunks),
        resident_grid_capacity_ctas=148 * resident_ctas_per_sm,
    )


def _evaluate_pair(parent, child, cases, *, args, helpers, repeat: int) -> dict:
    rows = []
    for profile_index, case in enumerate(cases):
        context = _profile_context(case)
        parent_cpc = evaluate_policy(parent, context)
        child_cpc = evaluate_policy(child, context)
        prepared = helpers["_make_prepared"](
            case, rotations=args.state_rotations, helpers=helpers
        )
        correctness = helpers["_correctness"](prepared, child_cpc, helpers)
        order = (
            ("parent", "child", "child", "parent")
            if profile_index % 2 == 0
            else ("child", "parent", "parent", "child")
        )
        samples = {"parent": [], "child": []}
        if correctness["passed"]:
            for name in order:
                prepared.reset_state_pools()
                cpc = parent_cpc if name == "parent" else child_cpc
                samples[name].extend(
                    helpers["_measure"](
                        prepared,
                        cpc,
                        dry=args.dry_run_iters,
                        repeat=repeat,
                        helpers=helpers,
                    )
                )
        speedup, lower, upper = (
            helpers["_bootstrap_speedup"](
                samples["parent"], samples["child"], int(case.seed)
            )
            if samples["parent"] and samples["child"]
            else (0.0, 0.0, 0.0)
        )
        rows.append({
            "profile_id": case.name,
            "parent_cpc": parent_cpc,
            "child_cpc": child_cpc,
            "correctness": correctness,
            "parent_samples_ms": samples["parent"],
            "child_samples_ms": samples["child"],
            "child_over_parent_speedup": speedup,
            "bootstrap_95": [lower, upper],
            "pair_order": "/".join(order),
            "route": prepared.metadata["variant"],
        })
        del prepared
        helpers["torch"].cuda.empty_cache()
    valid = all(row["correctness"]["passed"] for row in rows)
    score = fmean(log(max(row["child_over_parent_speedup"], 1e-300)) for row in rows) if valid else None
    return {
        "status": "ok" if valid else "correctness_failed",
        "child_minus_parent_uniform_log_speedup": score,
        "child_over_parent_geomean_speedup": exp(score) if score is not None else None,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "evidence/stage9_best_swarm_pilot_runs_sol/proposal_manifest_v5.json",
    )
    parser.add_argument(
        "--development",
        type=Path,
        default=ROOT / "evidence/stage9_development_summary.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-rotations", type=int, default=256)
    parser.add_argument("--dry-run-iters", type=int, default=5)
    parser.add_argument("--screen-repeat-iters", type=int, default=25)
    parser.add_argument("--qualification-repeat-iters", type=int, default=100)
    args = parser.parse_args()

    benchmark = _load_benchmark()
    helpers = benchmark._gpu_helpers()
    helpers.update({
        "_make_prepared": benchmark._make_prepared,
        "_correctness": benchmark._correctness,
        "_measure": benchmark._measure,
        "_bootstrap_speedup": benchmark._bootstrap_speedup,
    })
    helpers["_require_cupti"]()
    manifest = json.loads(args.manifest.read_text())
    development = json.loads(args.development.read_text())
    source = next(
        row for row in manifest["proposals"]
        if row["round"] == 0 and row["status"] == "valid"
    )
    parent = RuntimePolicyCandidate(_from_semantic(source["policy"]))
    parent_id = canonical_policy_id(parent)
    cases = tuple(helpers["STAGE6_CASES"]) + tuple(helpers["STAGE8_CASES"])
    if len(cases) != len(development["rows"]):
        raise RuntimeError("development evidence and executable cases differ")

    mutations = []
    for lane in (
        SearchLane.WAVE_CAPACITY,
        SearchLane.TAIL_SKEW,
        SearchLane.MIXED_INTERACTION,
    ):
        for mutation in deterministic_mutations(parent, development["rows"], lane=lane):
            mutations.append((lane, mutation))

    started = time.perf_counter()
    measured_by_behavior = {}
    screen_rows = []
    for lane, mutation in mutations:
        behavior = tuple(
            evaluate_policy(mutation.candidate, _profile_context(case))
            for case in cases
        )
        measured = behavior not in measured_by_behavior
        if measured:
            measured_by_behavior[behavior] = _evaluate_pair(
                parent,
                mutation.candidate,
                cases,
                args=args,
                helpers=helpers,
                repeat=args.screen_repeat_iters,
            )
        result = measured_by_behavior[behavior]
        screen_rows.append({
            "lane": lane.value,
            "operator": mutation.operator,
            "policy_id": canonical_policy_id(mutation.candidate),
            "policy": _semantic(mutation.candidate.root),
            "parent_policy_id": parent_id,
            "niche": policy_niche(mutation.candidate),
            "development_behavior_cpc": behavior,
            "measured": measured,
            "result": result,
        })

    winners = []
    for lane in (
        SearchLane.WAVE_CAPACITY,
        SearchLane.TAIL_SKEW,
        SearchLane.MIXED_INTERACTION,
    ):
        eligible = [
            row for row in screen_rows
            if row["lane"] == lane.value and row["result"]["status"] == "ok"
        ]
        winner = max(
            eligible,
            key=lambda row: (
                row["result"]["child_minus_parent_uniform_log_speedup"],
                row["policy_id"],
            ),
        )
        child = RuntimePolicyCandidate(_from_semantic(winner["policy"]), parent_ids=(parent_id,))
        qualification = _evaluate_pair(
            parent,
            child,
            cases,
            args=args,
            helpers=helpers,
            repeat=args.qualification_repeat_iters,
        )
        winners.append({**{key: value for key, value in winner.items() if key != "result"},
                        "screen_result": winner["result"],
                        "qualification_result": qualification})

    report = {
        "schema_version": 1,
        "purpose": "development_only_vnext_niche_screen_no_model_calls",
        "parent_policy_id": parent_id,
        "parent_policy": source["policy"],
        "candidate_count": len(screen_rows),
        "unique_development_behaviors": len(measured_by_behavior),
        "screen_repeat_iters": args.screen_repeat_iters,
        "qualification_repeat_iters": args.qualification_repeat_iters,
        "elapsed_seconds": time.perf_counter() - started,
        "hardware": helpers["_hardware_metadata"](helpers["torch"].device("cuda")),
        "screen": screen_rows,
        "niche_winners": winners,
    }
    _write_json(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()
