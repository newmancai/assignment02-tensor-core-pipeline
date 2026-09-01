"""Summarize Tensor Core opcodes from a cuobjdump SASS listing."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from pathlib import Path
import re


OPCODE = re.compile(r"\b(?:HMMA|TCGEN[0-9A-Z_]*|UTCMMA)(?:\.[0-9A-Z]+)*\b")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    args = parser.parse_args()

    counts = {"all": Counter(), "recurrence": Counter()}
    samples: list[str] = []
    current_function = ""
    with args.input.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("Function :"):
                current_function = stripped.split(":", 1)[1].strip()
                continue
            matches = OPCODE.findall(line)
            if not matches:
                continue
            counts["all"].update(matches)
            if "_flash_kda_fwd_recurrence" in current_function:
                counts["recurrence"].update(matches)
                if len(samples) < 40:
                    samples.append(f"{current_function}\n{stripped}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["scope", "opcode", "count"])
        for scope in ("all", "recurrence"):
            for opcode, count in counts[scope].most_common():
                writer.writerow([scope, opcode, count])
        for family in ("HMMA", "TCGEN", "UTCMMA"):
            total = sum(
                count for opcode, count in counts["recurrence"].items()
                if opcode.startswith(family)
            )
            writer.writerow(["recurrence_family_total", family, total])

    args.samples.write_text("\n\n".join(samples) + "\n", encoding="utf-8")
    print(f"wrote {args.output} and {args.samples}")


if __name__ == "__main__":
    main()
