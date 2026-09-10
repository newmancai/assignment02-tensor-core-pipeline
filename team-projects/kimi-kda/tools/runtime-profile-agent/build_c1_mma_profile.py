#!/usr/bin/env python3
"""Build the C1 physical profile directly from archived SASS/NCU/probe files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kda_ir import build_mma_profile_from_evidence, profile_to_dict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sass", type=Path, required=True)
    parser.add_argument("--ncu", type=Path, required=True)
    parser.add_argument("--tcgen", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    profile = build_mma_profile_from_evidence(
        sass_path=args.sass,
        ncu_summary_path=args.ncu,
        tcgen_path=args.tcgen,
        semantic_path=args.semantic,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile_to_dict(profile), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
