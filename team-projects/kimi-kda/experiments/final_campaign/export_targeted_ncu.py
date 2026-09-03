"""Export a compact long table and one-row-per-kernel NCU comparison."""

from __future__ import annotations

import argparse
import csv
import io
import math
from pathlib import Path
import re
import subprocess


KEEP_NAMES = {
    "Duration",
    "Compute (SM) Throughput",
    "DRAM Throughput",
    "Achieved Occupancy",
    "No Eligible",
    "Eligible Warps Per Scheduler",
    "Registers Per Thread",
    "Dynamic Shared Memory Per Block",
}

ALIASES = {
    "duration": {"Duration", "gpu__time_duration.sum"},
    "compute_sm": {
        "Compute (SM) Throughput",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    },
    "dram": {
        "DRAM Throughput",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    },
    "occupancy": {
        "Achieved Occupancy",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
    },
    "eligible": {
        "Eligible Warps Per Scheduler",
        "smsp__warps_eligible.avg.per_cycle_active",
    },
    "no_eligible": {"No Eligible"},
    "registers": {"Registers Per Thread"},
    "dynamic_smem": {"Dynamic Shared Memory Per Block"},
}


def import_details(report: Path) -> list[dict[str, str]]:
    completed = subprocess.run(
        ["ncu", "--import", str(report), "--page", "details", "--csv"],
        check=True,
        capture_output=True,
        text=True,
    )
    start = completed.stdout.find('"ID","Process ID"')
    if start < 0:
        raise RuntimeError(f"NCU details CSV header not found in {report}")
    rows = list(csv.DictReader(io.StringIO(completed.stdout[start:])))
    if not rows:
        raise RuntimeError(f"no metric rows found in {report}")
    return rows


def parse_dim(text: str) -> tuple[int, int, int]:
    values = [int(value) for value in re.findall(r"\d+", text)]
    if len(values) != 3:
        raise ValueError(f"cannot parse launch dimension: {text!r}")
    return values[0], values[1], values[2]


def clean_value(text: str) -> str:
    return text.replace(",", "").strip()


def choose_metric(rows: list[dict[str, str]], alias: str) -> tuple[str, str]:
    exact_names = ALIASES[alias]
    for row in rows:
        if row["Metric Name"] in exact_names:
            return clean_value(row["Metric Value"]), row["Metric Unit"]
    return "", ""


def choose_tensor_metric(rows: list[dict[str, str]]) -> tuple[str, str, str]:
    # Use elapsed-cycle normalization for cross-variant comparisons.  The
    # ``...sustained_active`` form only describes SMs/subpartitions while they
    # are active, so it can fall when ValueSlice spreads thinner CTAs across
    # more SMs even though whole-device tensor activity increases.
    preferred = (
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
        "smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
        "smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    )
    for metric in preferred:
        for row in rows:
            if row["Metric Name"] == metric:
                return (
                    clean_value(row["Metric Value"]),
                    row["Metric Unit"],
                    row["Metric Name"],
                )
    return "", "", ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--long-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument(
        "reports", nargs="+", help="label=/absolute/path/to/report.ncu-rep"
    )
    args = parser.parse_args()

    long_rows: list[dict[str, str]] = []
    summaries: list[dict[str, str | int]] = []
    for spec in args.reports:
        label, raw_path = spec.split("=", 1)
        report = Path(raw_path)
        rows = import_details(report)
        grid = rows[0]["Grid Size"]
        block = rows[0]["Block Size"]
        grid_xyz = parse_dim(grid)
        block_xyz = parse_dim(block)

        selected = []
        for row in rows:
            name = row["Metric Name"]
            lowered = name.lower()
            if (
                name in KEEP_NAMES
                or any(name in names for names in ALIASES.values())
                or ("tensor" in lowered and ("active" in lowered or "throughput" in lowered))
            ):
                selected.append(row)
                long_rows.append(
                    {
                        "label": label,
                        "report": report.name,
                        "kernel": row["Kernel Name"],
                        "grid_size": grid,
                        "block_size": block,
                        "section": row["Section Name"],
                        "metric": name,
                        "value": clean_value(row["Metric Value"]),
                        "unit": row["Metric Unit"],
                    }
                )
        if not selected:
            raise RuntimeError(f"no requested metrics found in {report}")

        summary: dict[str, str | int] = {
            "label": label,
            "report": report.name,
            "kernel": "_flash_kda_fwd_recurrence",
            "grid_size": grid,
            "cta_count": math.prod(grid_xyz),
            "block_size": block,
            "threads_per_cta": math.prod(block_xyz),
        }
        for alias in ALIASES:
            value, unit = choose_metric(rows, alias)
            summary[f"{alias}_value"] = value
            summary[f"{alias}_unit"] = unit
        tensor_value, tensor_unit, tensor_name = choose_tensor_metric(rows)
        summary["tensor_value"] = tensor_value
        summary["tensor_unit"] = tensor_unit
        summary["tensor_metric"] = tensor_name
        summaries.append(summary)

    expected_ctas = {"official_v128": 12, "valueslice_v16": 96}
    for summary in summaries:
        label = str(summary["label"])
        if label in expected_ctas and summary["cta_count"] != expected_ctas[label]:
            raise RuntimeError(
                f"unexpected grid for {label}: got {summary['cta_count']} CTAs, "
                f"expected {expected_ctas[label]}"
            )

    args.long_output.parent.mkdir(parents=True, exist_ok=True)
    long_fields = [
        "label", "report", "kernel", "grid_size", "block_size",
        "section", "metric", "value", "unit",
    ]
    with args.long_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=long_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(long_rows)

    summary_fields = list(summaries[0].keys())
    with args.summary_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)

    print(f"wrote {len(long_rows)} rows to {args.long_output}")
    print(f"wrote {len(summaries)} rows to {args.summary_output}")
    for summary in summaries:
        print(
            f"SUMMARY label={summary['label']} grid={summary['grid_size']} "
            f"duration={summary['duration_value']} {summary['duration_unit']} "
            f"sm={summary['compute_sm_value']} {summary['compute_sm_unit']} "
            f"dram={summary['dram_value']} {summary['dram_unit']} "
            f"tensor={summary['tensor_value']} {summary['tensor_unit']}"
        )


if __name__ == "__main__":
    main()
