"""Export a compact, auditable metric table from Nsight Compute reports."""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path
import subprocess


METRIC_NAMES = {
    "Duration",
    "Compute (SM) Throughput",
    "DRAM Throughput",
    "L2 Hit Rate",
    "Achieved Occupancy",
    "No Eligible",
    "Eligible Warps Per Scheduler",
    "Registers Per Thread",
    "Dynamic Shared Memory Per Block",
}


def read_details(report: Path) -> list[dict[str, str]]:
    completed = subprocess.run(
        ["ncu", "--import", str(report), "--page", "details", "--csv"],
        check=True, capture_output=True, text=True,
    )
    start = completed.stdout.find('"ID","Process ID"')
    if start < 0:
        raise RuntimeError(f"CSV header not found in {report}")
    return list(csv.DictReader(io.StringIO(completed.stdout[start:])))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("reports", nargs="+", help="label=/path/to/report.ncu-rep")
    args = parser.parse_args()

    output_rows: list[dict[str, str]] = []
    for spec in args.reports:
        label, raw_path = spec.split("=", 1)
        rows = read_details(Path(raw_path))
        if not rows:
            raise RuntimeError(f"no metric rows in {raw_path}")
        base = {
            "label": label,
            "report": Path(raw_path).name,
            "grid_size": rows[0]["Grid Size"],
            "block_size": rows[0]["Block Size"],
        }
        for row in rows:
            name = row["Metric Name"]
            if name not in METRIC_NAMES:
                continue
            output_rows.append(
                {
                    **base,
                    "section": row["Section Name"],
                    "metric": name,
                    "value": row["Metric Value"].replace(",", ""),
                    "unit": row["Metric Unit"],
                }
            )

    fieldnames = ["label", "report", "grid_size", "block_size", "section", "metric", "value", "unit"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
