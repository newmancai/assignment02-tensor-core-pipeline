"""Export compact, auditable Nsight Systems data from the archived report.

The SQLite database is produced by ``nsys export``.  The companion stats file
contains the ``nvtx_gpu_proj_trace`` table.  This script intentionally uses only
the Python standard library so it can be rerun on a clean machine.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sqlite3


RANGE_NAMES = ("official_v128", "auto_valueslice")


def kernel_class(name: str) -> str:
    if "elementwise_kernel" in name:
        return "copy"
    if "_flash_kda_fwd_prepare" in name:
        return "prepare"
    if "_flash_kda_fwd_recurrence" in name:
        return "recurrence"
    return "other"


def export_timeline(database: Path, output: Path) -> None:
    query = """
    WITH ranges AS (
      SELECT start, end,
             COALESCE(text, (SELECT value FROM StringIds WHERE id=textId)) AS range_name
      FROM NVTX_EVENTS
      WHERE COALESCE(text, (SELECT value FROM StringIds WHERE id=textId))
            IN ('official_v128', 'auto_valueslice')
    )
    SELECT r.range_name, r.start, r.end, k.start, k.end, s.value,
           k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, k.blockZ,
           k.registersPerThread, k.dynamicSharedMemory
    FROM ranges r
    JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON k.start >= r.start AND k.end <= r.end
    JOIN StringIds s ON s.id = k.shortName
    ORDER BY r.start, k.start
    """
    with sqlite3.connect(database) as connection:
        rows = connection.execute(query).fetchall()

    iteration = {name: -1 for name in RANGE_NAMES}
    normalized = []
    for row in rows:
        range_name, range_start, _, start, end, name, *launch = row
        klass = kernel_class(name)
        if klass == "copy":
            iteration[range_name] += 1
        normalized.append(
            [
                range_name,
                iteration[range_name],
                klass,
                (start - range_start) / 1_000,
                (end - start) / 1_000,
                name,
                *launch,
            ]
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "range_name",
                "iteration",
                "kernel_class",
                "range_relative_start_us",
                "duration_us",
                "short_name",
                "grid_x",
                "grid_y",
                "grid_z",
                "block_x",
                "block_y",
                "block_z",
                "registers_per_thread",
                "dynamic_smem_bytes",
            ]
        )
        writer.writerows(normalized)


def export_ranges(stats: Path, output: Path) -> None:
    lines = stats.read_text(encoding="utf-8", errors="replace").splitlines()
    header_index = next(
        index for index, line in enumerate(lines) if line.startswith("Name,Projected Start")
    )
    parsed = csv.DictReader(lines[header_index:])
    selected = []
    for row in parsed:
        raw_name = row.get("Name", "").lstrip(":")
        if raw_name not in RANGE_NAMES:
            if selected:
                break
            continue
        selected.append(
            [
                raw_name,
                int(row["Projected Duration (ns)"]) / 1_000,
                int(row["Orig Duration (ns)"]) / 1_000,
                int(row["NumGPUOps"]),
                5,
            ]
        )

    if len(selected) != 2:
        raise RuntimeError(f"expected two NVTX ranges, found {len(selected)}")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "range_name",
                "projected_duration_us",
                "original_duration_us",
                "gpu_operations",
                "iterations",
            ]
        )
        writer.writerows(selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("stats", type=Path)
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--ranges", type=Path, required=True)
    args = parser.parse_args()
    export_timeline(args.database, args.timeline)
    export_ranges(args.stats, args.ranges)
    print(f"wrote {args.timeline} and {args.ranges}")


if __name__ == "__main__":
    main()
