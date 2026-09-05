"""Recompute the tcgen05 review summary using Python's standard library.

Run without arguments next to tcgen_sweep_19844_1.csv and _2.csv, or pass
explicit sweep CSV paths. JSON is written to stdout; input files are unchanged.
Slope fits describe the repeated-operand L0 probe, not the real KDA recurrence.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean, median


def load_rows(path: Path) -> dict:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(line for line in handle if not line.startswith("#")))
    parsed = {}
    for row in rows:
        key = (row["level"], int(row["V"]), int(row["grid"]), int(row["inner"]))
        if key in parsed:
            raise ValueError(f"duplicate measurement {key} in {path}")
        parsed[key] = {
            "mma_us": float(row["mma_median_kernel_us"]),
            "tcgen_us": float(row["tcgen_median_kernel_us"]),
            "warmup": int(row["warmup"]),
            "iterations": int(row["iters"]),
            "repeats": int(row["repeats"]),
        }
        if min(parsed[key]["mma_us"], parsed[key]["tcgen_us"]) <= 0:
            raise ValueError(f"nonpositive latency at {key} in {path}")
    return parsed


def fit(points: list[tuple[int, float]]) -> dict:
    xmean = fmean(x for x, _ in points)
    ymean = fmean(y for _, y in points)
    slope = sum((x - xmean) * (y - ymean) for x, y in points) / sum(
        (x - xmean) ** 2 for x, _ in points
    )
    intercept = ymean - slope * xmean
    residuals = [y - intercept - slope * x for x, y in points]
    return {
        "slope_us_per_inner": slope,
        "intercept_us": intercept,
        "max_absolute_residual_us": max(abs(value) for value in residuals),
        "adjacent_secants_us_per_inner": [
            {"inner_interval": [x0, x1], "slope": (y1 - y0) / (x1 - x0)}
            for (x0, y0), (x1, y1) in zip(points, points[1:])
        ],
    }


def summarize(path: Path) -> dict:
    rows = load_rows(path)
    result = {"file": str(path), "row_count": len(rows), "grids": {}}
    for grid in sorted({key[2] for key in rows if key[:2] == ("L0", 128)}):
        inners = sorted(key[3] for key in rows if key[:3] == ("L0", 128, grid))
        measurements = []
        for inner in inners:
            row = rows[("L0", 128, grid, inner)]
            measurements.append(
                {"inner": inner, **row, "speedup": row["mma_us"] / row["tcgen_us"]}
            )
        winning = next((i for i, row in enumerate(measurements) if row["speedup"] > 1), None)
        bracket = None
        if winning is not None:
            bracket = {
                "last_preceding_sample_not_faster": inners[winning - 1] if winning else None,
                "first_sample_faster": inners[winning],
                "interpretation": "sample bracket only; not an interpolated exact crossing",
            }
        fits = {
            name: fit([(inner, rows[("L0", 128, grid, inner)][f"{name}_us"]) for inner in (128, 256, 512)])
            for name in ("mma", "tcgen")
        }
        fits["mma_over_tcgen_slope"] = (
            fits["mma"]["slope_us_per_inner"] / fits["tcgen"]["slope_us_per_inner"]
        )
        l1 = rows[("L1", 128, grid, 512)]
        result["grids"][grid] = {
            "L0_measurements": measurements,
            "observed_crossover_bracket": bracket,
            "L0_linear_fit_inners_128_256_512": fits,
            "L1_inner512_speedup": l1["mma_us"] / l1["tcgen_us"],
        }
    return result


def summarize_boundaries(path: Path) -> dict:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.startswith('{')]
    if not any(row['kind'] == 'complete' for row in records):
        raise ValueError('follow-up log did not reach completion')
    alignment = {row['mode']: row for row in records if row['kind'] == 'alignment_child'}
    if alignment['aligned']['returncode'] != 0 or alignment['view']['returncode'] != -6:
        raise ValueError('alignment controls do not match the review finding')
    perf = [row for row in records if row['kind'] == 'perf']
    if len(perf) != 6:
        raise ValueError('expected three paired V128/V16 repeats')
    metrics = ('hot_ms', 'pre_call_eviction_ms', 'graph_hot_ms', 'graph_pre_call_eviction_ms')
    summary = {}
    for metric in metrics:
        values = {vs: median(row[metric] for row in perf if row['value_slice'] == vs) for vs in (128, 16)}
        summary[metric] = {'v128_ms': values[128], 'v16_ms': values[16],
                           'reduction_pct': 100 * (1 - values[16] / values[128])}
    return {'file': str(path), 'performance': summary,
            'numerical_records': [row for row in records if row['kind'] in
                                  ('empty_sequence_identity', 'segmentation', 'state_alias', 'auto_wrapper')],
            'alignment_returncodes': {mode: row['returncode'] for mode, row in alignment.items()},
            'scope': 'pre-call cache perturbation only; not sustained contention or guaranteed cold K2'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path)
    args = parser.parse_args()
    paths = args.files or sorted(Path(__file__).parent.glob("tcgen_sweep_19844_*.csv"))
    if not paths:
        parser.error("no input sweep CSVs found")
    result = {
        "scope": "V128 repeated-operand L0 probe; not full K2 or KDA speedup",
        "slope_interpretation": "descriptive high-inner fit, not an asymptotic guarantee",
        "rounds": [summarize(path) for path in paths],
    }
    boundaries = Path(__file__).parent / 'followup_19845.log'
    if boundaries.exists():
        result['boundaries'] = summarize_boundaries(boundaries)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
