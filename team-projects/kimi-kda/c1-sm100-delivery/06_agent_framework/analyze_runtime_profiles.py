#!/usr/bin/env python3
"""Join Stage 4/5 measurements with hardware-normalized runtime profiles."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean

from kda_ir.runtime_profile import RuntimeWorkloadProfile, runtime_profile_receipt


def _load_rows(path: Path) -> list[dict[str, object]]:
    report = json.loads(path.read_text())
    rows = []
    for row in report["rows"]:
        baseline = row.get("baseline", {})
        route = row.get("route") or baseline.get("variant")
        rows.append(
            {
                "name": row["name"],
                "seq_lens": row["seq_lens"],
                "layout": row["layout"],
                "route": route,
                "affected": row["affected"],
                "speedup": row["speedup"],
                "bootstrap_95": row["bootstrap_95"],
            }
        )
    return rows


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    x_mean, y_mean = fmean(xs), fmean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_scale = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_scale = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    return numerator / (x_scale * y_scale) if x_scale and y_scale else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--development",
        type=Path,
        default=Path("evidence/b300_stage4_h12_cpc_pair_v2.json"),
    )
    parser.add_argument(
        "--heldout",
        type=Path,
        default=Path("evidence/b300_stage5_h12_cpc_heldout.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evidence/b300_stage6_runtime_profile_analysis.json"),
    )
    args = parser.parse_args()

    enriched = []
    for split, path in (("development", args.development), ("heldout", args.heldout)):
        for row in _load_rows(path):
            profile = RuntimeWorkloadProfile(
                num_heads=12,
                seq_lens=tuple(row["seq_lens"]),
                packed=row["layout"] == "packed",
                sm_count=148,
            )
            receipt = runtime_profile_receipt(profile, (4, 9))
            baseline, candidate = receipt["prepare_grids"]
            enriched.append(
                {
                    **row,
                    "split": split,
                    "profile": receipt,
                    "cpc4_to_cpc9": {
                        "scheduled_cta_ratio": (
                            candidate["scheduled_ctas"] / baseline["scheduled_ctas"]
                        ),
                        "waves_removed": baseline["waves"] - candidate["waves"],
                        "tail_utilization_delta": (
                            candidate["tail_utilization"]
                            - baseline["tail_utilization"]
                        ),
                    },
                }
            )

    affected = [row for row in enriched if row["affected"]]
    log_speedups = [math.log(float(row["speedup"])) for row in affected]
    feature_values = {
        "total_chunks": [float(row["profile"]["total_chunks"]) for row in affected],
        "max_chunks": [float(row["profile"]["max_chunks"]) for row in affected],
        "effective_sequence_parallelism": [
            float(row["profile"]["effective_sequence_parallelism"])
            for row in affected
        ],
        "chunk_cv": [float(row["profile"]["chunk_cv"]) for row in affected],
        "baseline_prepare_waves": [
            float(row["profile"]["prepare_grids"][0]["waves"]) for row in affected
        ],
        "waves_removed": [
            float(row["cpc4_to_cpc9"]["waves_removed"]) for row in affected
        ],
    }
    correlations = {
        name: _pearson(values, log_speedups) for name, values in feature_values.items()
    }
    report = {
        "schema_version": 1,
        "analysis": "b300-stage6-runtime-profile-mechanism-retrospective",
        "source_evidence": [str(args.development), str(args.heldout)],
        "claim_boundary": (
            "Descriptive post-hoc analysis of six affected B300 observations; "
            "correlations prioritize counterfactual experiments and are not a "
            "learned deployment policy or causal attribution."
        ),
        "observed_rows": enriched,
        "affected_observation_count": len(affected),
        "all_affected_lower_bounds_above_one": all(
            float(row["bootstrap_95"][0]) > 1.0 for row in affected
        ),
        "feature_log_speedup_pearson": correlations,
        "mechanism_hypotheses": [
            {
                "priority": 1,
                "hypothesis": "prepare CTA launch and scheduling overhead",
                "falsifier": (
                    "prepare-only timing does not fall when cpc=9 removes prepare waves"
                ),
            },
            {
                "priority": 2,
                "hypothesis": "prepare/chain critical-path mixture",
                "falsifier": (
                    "matched-total-chunk profiles with different maximum sequence "
                    "lengths show identical end-to-end gains and phase attribution"
                ),
            },
            {
                "priority": 3,
                "hypothesis": "tail and wave quantization interaction",
                "falsifier": (
                    "matched rectangular grids show no latency discontinuity across "
                    "an SM-wave boundary"
                ),
            },
        ],
        "next_experiment": {
            "principle": "match total chunks while varying recurrent critical path",
            "cpc_candidates": [3, 4, 6, 9, 11, 12],
            "profiles": [
                {"name": "fixed-8192", "seq_lens": [8192]},
                {"name": "uniform-2048x4", "seq_lens": [2048, 2048, 2048, 2048]},
                {"name": "balanced-4096x2", "seq_lens": [4096, 4096]},
                {"name": "skew-64-64-64-8000", "seq_lens": [64, 64, 64, 8000]},
            ],
            "required_counters": ["public_total", "prepare_only", "chain_only"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
