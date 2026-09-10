#!/usr/bin/env python3
"""Collect representative NCU prepare-kernel reports for cpc=4 and cpc=9."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for cpc in (4, 9):
        csv_path = args.output_dir / f"prepare-cpc{cpc}.csv"
        command = [
            "/usr/local/bin/ncu",
            "--target-processes",
            "all",
            "--kernel-name-base",
            "demangled",
            "--kernel-name",
            "regex:.*prepare.*",
            "--launch-skip",
            "1",
            "--launch-count",
            "1",
            "--section",
            "LaunchStats",
            "--section",
            "Occupancy",
            "--section",
            "SpeedOfLight",
            "--section",
            "InstructionStats",
            "--csv",
            "--page",
            "raw",
            "--log-file",
            str(csv_path),
            "--force-overwrite",
            sys.executable,
            str(args.benchmark),
            "--cpc",
            str(cpc),
        ]
        completed = subprocess.run(command, check=False, text=True, capture_output=True)
        runs.append(
            {
                "cpc": cpc,
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "csv": str(csv_path),
            }
        )
        if completed.returncode:
            raise SystemExit(completed.returncode)
    report = {
        "schema_version": 1,
        "analysis": "stage7-h12-fixed8192-prepare-ncu",
        "claim_boundary": (
            "Representative prepare-kernel counters for one Stage 6 profile; "
            "the four-profile CUPTI activity study is the transfer evidence."
        ),
        "runs": runs,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
