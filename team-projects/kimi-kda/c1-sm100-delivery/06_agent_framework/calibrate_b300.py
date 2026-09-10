#!/usr/bin/env python3
"""Build a proof-carrying B300 calibration and deployment-tactic report."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path


def _load(path: Path):
    return json.loads(path.read_text())


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round(fraction * (len(ordered) - 1))]


def _bootstrap_ratio(
    baseline: list[float], candidate: list[float], *, seed: int, draws: int = 4000
) -> tuple[float, float]:
    generator = random.Random(seed)
    ratios = []
    for _ in range(draws):
        base = statistics.median(generator.choices(baseline, k=len(baseline)))
        cand = statistics.median(generator.choices(candidate, k=len(candidate)))
        ratios.append(base / cand)
    return _quantile(ratios, 0.025), _quantile(ratios, 0.975)


def _geomean(values: list[float]) -> float:
    return math.exp(statistics.fmean(math.log(value) for value in values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.evidence_dir

    corpus = _load(root / "evolution_typed_corpus.json")
    typed = {item["variant"]: item for item in corpus["candidates"]}
    baseline_rows = {item["name"]: item for item in _load(root / "b300_legacy_cake.json")}
    auto_rows = {item["name"]: item for item in _load(root / "b300_legacy_auto.json")}
    candidate_report = _load(root / "b300_evolution6_st_stable.json")
    candidate_rows = {item["name"]: item for item in candidate_report["rows"]}
    if set(baseline_rows) != set(candidate_rows):
        raise ValueError("baseline and candidate case sets differ")

    cases = []
    speedups = []
    for index, name in enumerate(baseline_rows):
        baseline = baseline_rows[name]
        candidate = candidate_rows[name]
        variant = candidate["variant"]
        if variant not in typed:
            raise ValueError(f"candidate lacks a typed semantic delta: {variant}")
        speedup = float(baseline["median_ms"]) / float(candidate["median_ms"])
        ci_low, ci_high = _bootstrap_ratio(
            [float(value) for value in baseline["samples_ms"]],
            [float(value) for value in candidate["samples_ms"]],
            seed=20260908 + index,
        )
        auto = auto_rows[name]
        cases.append(
            {
                "name": name,
                "typed_delta": typed[variant],
                "correct": bool(candidate["correct"]),
                "baseline_median_us": float(baseline["median_ms"]) * 1000,
                "candidate_median_us": float(candidate["median_ms"]) * 1000,
                "speedup": speedup,
                "speedup_bootstrap_95pct": [ci_low, ci_high],
                "activation_tactic": (
                    "activate_after_public-adapter validation"
                    if candidate["correct"] and ci_low > 1.0
                    else "retain-baseline"
                ),
                "current_public_auto_backend": auto["resolved_backend"],
                "current_public_auto_variant": auto["variant"],
            }
        )
        speedups.append(speedup)

    accounts = []
    for path in sorted(root.glob("b300_*accounting.json")):
        account = _load(path)
        accounts.append(
            {
                "file": path.name,
                "experiment": account["experiment"],
                "exit_code": account["exit_code"],
                "wall_seconds": account["wall_seconds"],
                "allocated_gpu_seconds": account["allocated_gpu_seconds"],
                "estimated_allocation_energy_joules": account["telemetry"][
                    "estimated_allocation_energy_joules"
                ],
                "peak_memory_used_mib": account["telemetry"]["peak_memory_used_mib"],
            }
        )
    retained_successes = [item for item in accounts if item["exit_code"] == 0]

    benchmark_payloads = [
        _load(root / "b300_evolution6_results.json")["rows"],
        candidate_report["rows"],
        list(baseline_rows.values()),
        list(auto_rows.values()),
    ]
    timed_samples_ms = [
        float(sample)
        for rows in benchmark_payloads
        for row in rows
        for sample in row["samples_ms"]
    ]
    public_auto_is_inactive = all(
        row["resolved_backend"] == "cake" for row in auto_rows.values()
    )
    report = {
        "schema_version": 1,
        "analysis": "b300-typed-delta-calibration-and-deployment-tactic",
        "hardware": candidate_report["hardware"],
        "candidate_corpus": {
            key: corpus[key]
            for key in (
                "candidate_count",
                "semantic_fingerprint_count",
                "topology_counts",
                "verifier_diagnostics",
            )
        },
        "calibration": {
            "case_count": len(cases),
            "all_candidates_correct": all(item["correct"] for item in cases),
            "geomean_directional_speedup_vs_cake_public": _geomean(speedups),
            "minimum_speedup": min(speedups),
            "maximum_speedup": max(speedups),
            "activation_eligible_case_count": sum(
                item["activation_tactic"].startswith("activate") for item in cases
            ),
            "cases": cases,
            "comparison_scope": (
                "CUPTI cold-L2 GPU timing; CAKE through public recurrent_kda, "
                "candidate through prepared evolution launch"
            ),
            "interpretation_limit": (
                "directional kernel evidence, not a production API speedup claim, "
                "until the public adapter is activated and remeasured"
            ),
        },
        "deployment_gap": {
            "public_auto_still_resolves_to_cake_for_all_cases": public_auto_is_inactive,
            "required_tactic": (
                "make verified evolution schedules an explicit public backend/route, "
                "then rerun the identical rotating-state public benchmark"
            ),
        },
        "compute_accounting": {
            "retained_successful_run_count": len(retained_successes),
            "allocated_gpu_seconds": sum(
                item["allocated_gpu_seconds"] for item in retained_successes
            ),
            "allocated_gpu_minutes": sum(
                item["allocated_gpu_seconds"] for item in retained_successes
            )
            / 60,
            "estimated_allocation_energy_joules": sum(
                item["estimated_allocation_energy_joules"] or 0
                for item in retained_successes
            ),
            "cupti_timed_sample_count": len(timed_samples_ms),
            "sum_of_cupti_sample_duration_seconds": sum(timed_samples_ms) / 1000,
            "accounts": accounts,
            "coverage_note": (
                "lower bound: retained successful manifests only; early pre-kernel "
                "environment failures were overwritten before append-only attempts "
                "were added to the runner"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "all_correct": report["calibration"]["all_candidates_correct"],
                "geomean_speedup": report["calibration"][
                    "geomean_directional_speedup_vs_cake_public"
                ],
                "activation_eligible": report["calibration"][
                    "activation_eligible_case_count"
                ],
                "allocated_gpu_seconds": report["compute_accounting"][
                    "allocated_gpu_seconds"
                ],
                "public_auto_inactive": public_auto_is_inactive,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
