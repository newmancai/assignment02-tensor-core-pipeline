#!/usr/bin/env python3
"""Summarize a paired block comparison from a dual-IR B300 round."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * probability)
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--case", required=True)
    parser.add_argument("--incumbent", required=True)
    parser.add_argument("--challenger", required=True)
    parser.add_argument("--expected-challenger-route")
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--practical-gate", type=float, default=1.02)
    args = parser.parse_args()

    rows = {}
    for path in args.result_dir.glob(f"{args.case}_*.json"):
        if path.name.endswith("_execution_identity.json"):
            continue
        row = json.loads(path.read_text())
        rows[(row["variant"], int(row["block"]))] = row

    blocks = sorted(
        block
        for variant, block in rows
        if variant == args.incumbent and (args.challenger, block) in rows
    )
    if not blocks:
        raise SystemExit("no matched blocks")

    route_checks = []
    correctness_checks = []
    block_ratios = []
    for block in blocks:
        incumbent = rows[(args.incumbent, block)]
        challenger = rows[(args.challenger, block)]
        correctness_checks.append(
            all(
                item["passed"]
                for row in (incumbent, challenger)
                for item in row["correctness"].values()
            )
        )
        observed = challenger["decision"].get("observed_modules", [])
        route_checks.append(
            args.expected_challenger_route is None
            or any(
                item.get("variant") == args.expected_challenger_route
                and item.get("target") == "sm103a"
                for item in observed
            )
        )
        block_ratios.append(incumbent["median_ms"] / challenger["median_ms"])

    rng = random.Random(20260910)
    draws = []
    for _ in range(args.bootstrap):
        ratios = []
        for block in blocks:
            incumbent = rows[(args.incumbent, block)]["samples_ms"]
            challenger = rows[(args.challenger, block)]["samples_ms"]
            incumbent_median = statistics.median(
                rng.choices(incumbent, k=len(incumbent))
            )
            challenger_median = statistics.median(
                rng.choices(challenger, k=len(challenger))
            )
            ratios.append(incumbent_median / challenger_median)
        draws.append(statistics.median(ratios))

    payload = {
        "case": args.case,
        "incumbent": args.incumbent,
        "challenger": args.challenger,
        "blocks": blocks,
        "block_speedups": block_ratios,
        "median_speedup": statistics.median(block_ratios),
        "bootstrap_95": [percentile(draws, 0.025), percentile(draws, 0.975)],
        "correctness_passed": all(correctness_checks),
        "route_identity_passed": all(route_checks),
        "qualification_passed": (
            all(correctness_checks)
            and all(route_checks)
            and statistics.median(block_ratios) >= args.practical_gate
            and percentile(draws, 0.025) > 1.0
        ),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
