#!/usr/bin/env python3
"""Compare logical and producer-ready global-B tcgen05 probe receipts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load(path: Path) -> dict[int, dict[str, str]]:
    with path.open() as stream:
        rows = [line for line in stream if not line.startswith("#")]
    return {int(row["inner"]): row for row in csv.DictReader(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logical", type=Path, required=True)
    parser.add_argument("--physical", type=Path, required=True)
    parser.add_argument("--logical-log", type=Path, required=True)
    parser.add_argument("--physical-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logical = load(args.logical)
    physical = load(args.physical)
    logical_log = args.logical_log.read_text(errors="replace")
    physical_log = args.physical_log.read_text(errors="replace")
    validation_passed = all(
        "SUMMARY,PASS" in text and "FAIL" not in text
        for text in (logical_log, physical_log)
    )
    comparisons = []
    for inner in sorted(set(logical) & set(physical)):
        logical_us = float(logical[inner]["tcgen_preferred_median_kernel_us"])
        physical_us = float(physical[inner]["tcgen_preferred_median_kernel_us"])
        hmma_us = float(physical[inner]["mma_median_kernel_us"])
        comparisons.append(
            {
                "inner": inner,
                "logical_preferred_us": logical_us,
                "producer_ready_physical_us": physical_us,
                "physical_over_logical_speedup": logical_us / physical_us,
                "physical_tcgen_over_hmma": hmma_us / physical_us,
            }
        )
    payload = {
        "schema_version": 1,
        "round": 9,
        "candidate": "tcgen05 V16 producer-ready physical B^T global contract",
        "scope": "Phase-6 state+gate mechanism probe; not public-full FlashKDA",
        "validation_passed": validation_passed,
        "comparison": comparisons,
        "interpretation_boundary": (
            "The physical input removes the strided logical-KV read from the probe. "
            "It estimates a producer-ready contract but does not prove that Phase 4 "
            "can emit the layout for free in the integrated kernel."
        ),
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if not validation_passed:
        raise SystemExit("correctness validation failed")


if __name__ == "__main__":
    main()
