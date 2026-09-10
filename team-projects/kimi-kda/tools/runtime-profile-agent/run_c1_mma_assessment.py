#!/usr/bin/env python3
"""Generate the evidence-first C1 MMA migration decision receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kda_ir import KernelMeasurement, MmaMigrationProfile, assess_mma_migration


def _measurement(data: dict[str, object]) -> KernelMeasurement:
    return KernelMeasurement(**data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = json.loads(args.profile.read_text())
    data["official"] = _measurement(data["official"])
    assessment = assess_mma_migration(MmaMigrationProfile(**data))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(assessment.to_dict(), indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
