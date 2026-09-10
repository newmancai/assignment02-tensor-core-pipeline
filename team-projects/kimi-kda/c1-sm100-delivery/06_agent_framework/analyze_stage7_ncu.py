#!/usr/bin/env python3
"""Reduce raw Nsight Compute CSVs into a compact Stage 7 evidence record."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path


METRICS = {
    "block_size": "launch__block_size",
    "grid_size": "launch__grid_size",
    "sm_count": "launch__sm_count",
    "waves_per_sm": "launch__waves_per_multiprocessor",
    "registers_per_thread": "launch__registers_per_thread",
    "allocated_registers_per_thread": "launch__registers_per_thread_allocated",
    "shared_memory_bytes": "launch__shared_mem_per_block",
    "active_warps_percent": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm_throughput_percent": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_throughput_percent": "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "l2_throughput_percent": "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu_duration_ns": "gpu__time_duration.sum",
    "instructions": "inst_executed",
    "smsp_instructions": "smsp__inst_executed.sum",
    "replayer_passes": "profiler__replayer_passes",
}


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _number(value: str) -> int | float:
    cleaned = value.replace(",", "").strip()
    number = float(cleaned)
    return int(number) if number.is_integer() else number


def _read_raw(path: Path) -> dict[str, int | float | str]:
    lines = path.read_text().splitlines()
    header_index = next(i for i, line in enumerate(lines) if line.startswith('"ID"'))
    reader = csv.reader(lines[header_index:])
    header = next(reader)
    units = next(reader)
    values = next(reader)
    row = dict(zip(header, values))
    unit_map = dict(zip(header, units))
    result: dict[str, int | float | str] = {
        "kernel_name": row["Kernel Name"],
        "device": row["Device"],
    }
    for label, metric in METRICS.items():
        result[label] = _number(row[metric])
        if unit_map.get(metric):
            result[f"{label}_unit"] = unit_map[metric]
    return result


def _relative(candidate: float, baseline: float) -> float:
    return candidate / baseline - 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpc4", type=Path, default=Path("evidence/stage7_ncu/prepare-cpc4.csv"))
    parser.add_argument("--cpc9", type=Path, default=Path("evidence/stage7_ncu/prepare-cpc9.csv"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evidence/b300_stage7_h12_prepare_ncu_summary.json"),
    )
    args = parser.parse_args()
    baseline = _read_raw(args.cpc4)
    candidate = _read_raw(args.cpc9)
    relative = {
        key: _relative(float(candidate[key]), float(baseline[key]))
        for key in (
            "grid_size",
            "waves_per_sm",
            "gpu_duration_ns",
            "instructions",
            "smsp_instructions",
            "active_warps_percent",
            "sm_throughput_percent",
            "dram_throughput_percent",
            "l2_throughput_percent",
        )
    }
    report = {
        "schema_version": 1,
        "analysis": "stage7-h12-fixed8192-prepare-ncu-summary",
        "baseline_cpc4": baseline,
        "candidate_cpc9": candidate,
        "candidate_relative_to_baseline": relative,
        "interpretation": {
            "cta_grid_reduction_percent": -100.0 * relative["grid_size"],
            "occupancy_aware_wave_transition": "2.08_to_0.92_crosses_one_resident_wave",
            "resource_footprint_unchanged": all(
                baseline[key] == candidate[key]
                for key in (
                    "block_size",
                    "registers_per_thread",
                    "allocated_registers_per_thread",
                    "shared_memory_bytes",
                )
            ),
            "hypothesis": (
                "The prepare saving is primarily a CTA decomposition and scheduling-tail effect: "
                "cpc=9 reduces the grid below one occupancy-aware resident wave while leaving "
                "the per-CTA resource footprint unchanged. The small instruction decrease is not "
                "proportional to the duration decrease."
            ),
        },
        "claim_boundary": (
            "Representative Nsight Compute counters for the fixed-8192 prepare kernel explain "
            "the mechanism but are not timing gates. Four-profile CUPTI activity medians provide "
            "the authoritative phase timing and transfer evidence."
        ),
        "artifacts": {str(path): _digest(path) for path in (args.cpc4, args.cpc9)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["interpretation"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
