#!/usr/bin/env python3
"""Build and validate the Stage 6 matched-profile evidence certificate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean, pstdev


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _pearson(xs: list[float], ys: list[float]) -> float:
    x_mean, y_mean = fmean(xs), fmean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - x_mean) ** 2 for x in xs)
        * sum((y - y_mean) ** 2 for y in ys)
    )
    return numerator / denominator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--screen",
        type=Path,
        default=Path("evidence/b300_stage6_h12_profile_matrix.json"),
    )
    parser.add_argument(
        "--screen-accounting",
        type=Path,
        default=Path("evidence/b300_stage6_h12_profile_matrix_accounting.json"),
    )
    parser.add_argument(
        "--pair",
        type=Path,
        default=Path("evidence/b300_stage6_h12_profile_pair.json"),
    )
    parser.add_argument(
        "--pair-accounting",
        type=Path,
        default=Path("evidence/b300_stage6_h12_profile_pair_accounting.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evidence/b300_stage6_h12_profile_certificate.json"),
    )
    args = parser.parse_args()

    screen = json.loads(args.screen.read_text())
    screen_accounting = json.loads(args.screen_accounting.read_text())
    pair = json.loads(args.pair.read_text())
    pair_accounting = json.loads(args.pair_accounting.read_text())
    if screen_accounting["exit_code"] != 0 or pair_accounting["exit_code"] != 0:
        raise ValueError("Stage 6 contains a failed authoritative attempt")
    if not pair["all_correct"] or not pair["scope_identical"]:
        raise ValueError("paired correctness or scope identity failed")

    screen_by_name = {row["name"]: row for row in screen["rows"]}
    results = []
    effective_parallelism = []
    log_speedups = []
    absolute_savings = []
    for row in pair["rows"]:
        screened = screen_by_name[row["name"]]
        timings = {
            item["chunks_per_cta"]: item["median_ms"]
            for item in screened["candidates"]
            if "median_ms" in item
        }
        winner = min(timings, key=timings.get)
        all_screen_correct = all(
            item.get("correctness", {}).get("passed", False)
            for item in screened["candidates"]
        )
        if winner != 9 or not all_screen_correct or not screened["route_invariant"]:
            raise ValueError(f"screen gate failed for {row['name']}")
        if row["bootstrap_95"][0] <= 1.0 or row["decision"] != "supports_cpc9":
            raise ValueError(f"paired confidence gate failed for {row['name']}")
        parallelism = row["total_chunks"] / row["max_chunks"]
        saving_us = 1000 * (
            row["baseline_median_ms"] - row["candidate_median_ms"]
        )
        effective_parallelism.append(parallelism)
        log_speedups.append(math.log(row["speedup"]))
        absolute_savings.append(saving_us)
        results.append(
            {
                "name": row["name"],
                "seq_lens": row["seq_lens"],
                "total_chunks": row["total_chunks"],
                "max_chunks": row["max_chunks"],
                "effective_sequence_parallelism": parallelism,
                "screen_winner_cpc": winner,
                "baseline_us": 1000 * row["baseline_median_ms"],
                "candidate_us": 1000 * row["candidate_median_ms"],
                "absolute_savings_us": saving_us,
                "speedup": row["speedup"],
                "bootstrap_95": row["bootstrap_95"],
                "route": row["route"],
                "physical_variants": row["physical_variants"],
            }
        )

    geomean = math.exp(fmean(log_speedups))
    total_gpu_seconds = (
        screen_accounting["allocated_gpu_seconds"]
        + pair_accounting["allocated_gpu_seconds"]
    )
    total_energy = sum(
        item["telemetry"]["estimated_allocation_energy_joules"]
        for item in (screen_accounting, pair_accounting)
    )
    certificate = {
        "schema_version": 1,
        "stage": "stage6-kimi-k3-tp8-h12-matched-profile-mechanism",
        "status": "cpc9_wins_screen_and_paired_confirmation_on_all_profiles",
        "parent_certificate": "evidence/b300_stage5_h12_heldout_certificate.json",
        "question": (
            "At fixed total prepare work, does cpc=9 remain optimal while the "
            "recurrent critical path changes?"
        ),
        "matched_contract": {
            "num_heads": 12,
            "total_tokens": 8192,
            "total_chunks": 512,
            "profile_count": 4,
            "screened_cpc": screen["cpc_candidates"],
        },
        "protocol": pair["protocol"],
        "all_correct": True,
        "scope_identical": True,
        "screen_winner_count": len(results),
        "paired_support_count": len(results),
        "results": results,
        "summary": {
            "profile_geomean_speedup": geomean,
            "minimum_bootstrap_95_lower": min(
                row["bootstrap_95"][0] for row in results
            ),
            "absolute_savings_mean_us": fmean(absolute_savings),
            "absolute_savings_range_us": [
                min(absolute_savings),
                max(absolute_savings),
            ],
            "absolute_savings_cv": pstdev(absolute_savings) / fmean(absolute_savings),
            "effective_parallelism_log_speedup_pearson": _pearson(
                effective_parallelism, log_speedups
            ),
        },
        "interpretation": (
            "The nearly constant absolute saving and increasing relative speedup "
            "as the recurrent critical path shortens support a prepare/chain "
            "mixture hypothesis. Kernel-level prepare/chain attribution is still "
            "required for causal confirmation."
        ),
        "artifacts": {
            "screen_benchmark_sha256": _sha256(
                Path("stage3_remote/benchmarks/bench_flash_kda_h12_profile_matrix.py")
            ),
            "screen_result_sha256": _sha256(args.screen),
            "screen_accounting_sha256": _sha256(args.screen_accounting),
            "pair_benchmark_sha256": _sha256(
                Path("stage3_remote/benchmarks/bench_flash_kda_h12_profile_pair.py")
            ),
            "pair_result_sha256": _sha256(args.pair),
            "pair_accounting_sha256": _sha256(args.pair_accounting),
        },
        "accounting": {
            "slurm_job_ids": [
                screen_accounting["scheduler"]["slurm_job_id"],
                pair_accounting["scheduler"]["slurm_job_id"],
            ],
            "allocated_gpu_seconds": total_gpu_seconds,
            "estimated_allocation_energy_joules": total_energy,
        },
        "claim_boundary": (
            "This is a preregistered matched-profile mechanism study and paired "
            "confirmation on one B300. The same profiles selected cpc=9 and then "
            "confirmed it, so this is not a new untouched held-out deployment "
            "qualification, a prepare/chain causal attribution, or a multi-agent "
            "versus single-agent comparison."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
