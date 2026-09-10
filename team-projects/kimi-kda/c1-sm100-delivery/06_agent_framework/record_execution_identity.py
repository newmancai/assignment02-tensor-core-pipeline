#!/usr/bin/env python3
"""Write an allowlisted Python/import/native/header execution receipt."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from kda_ir.execution_identity import (
        collect_execution_identity,
        write_execution_identity,
    )
except ModuleNotFoundError:
    # A remote benchmark directory may carry this CLI and execution_identity.py
    # without installing the whole research package.
    from execution_identity import collect_execution_identity, write_execution_identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", action="append", default=[])
    parser.add_argument("--dependency-file", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = collect_execution_identity(args.module, args.dependency_file)
    write_execution_identity(receipt, args.output)


if __name__ == "__main__":
    main()
