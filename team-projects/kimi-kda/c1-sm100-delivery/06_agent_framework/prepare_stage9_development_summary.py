#!/usr/bin/env python3
"""Build the only performance evidence visible to Stage 9 proposal agents."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path


SOURCES = (
    Path("evidence/b300_stage6_h12_profile_matrix.json"),
    Path("evidence/b300_stage8_h12_workscale_matrix.json"),
)
ALLOWED_CPC = {1, 2, 3, 4, 5, 6, 8, 9, 12, 16, 17, 18, 20, 24, 32}


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evidence/stage9_development_summary.json"),
    )
    args = parser.parse_args()

    rows = []
    source_hashes = {}
    for path in SOURCES:
        raw = path.read_bytes()
        source_hashes[str(path)] = sha256(raw).hexdigest()
        report = json.loads(raw)
        for row in report["rows"]:
            measurements = []
            for candidate in row["candidates"]:
                cpc = int(candidate["chunks_per_cta"])
                if cpc not in ALLOWED_CPC or "median_ms" not in candidate:
                    continue
                if not candidate.get("correctness", {}).get("passed", False):
                    continue
                measurements.append(
                    {
                        "cpc": cpc,
                        "latency_us": round(float(candidate["median_ms"]) * 1000.0, 6),
                    }
                )
            rows.append(
                {
                    "profile_id": row["name"],
                    "total_chunks": int(row["total_chunks"]),
                    "max_chunks": int(row["max_chunks"]),
                    "num_sequences": len(row["seq_lens"]),
                    "resident_grid_capacity_ctas": 740,
                    "measurements": measurements,
                }
            )

    public_payload = {
        "schema_version": 1,
        "target": "NVIDIA B300 SXM6 AC / sm_103a",
        "route": "bt16_prepare_chain_m64",
        "allowed_cpc": sorted(ALLOWED_CPC),
        "objective": "uniform_profile_mean_log_speedup; lower latency is better",
        "rows": rows,
    }
    report = {
        **public_payload,
        "public_payload_sha256": _digest(public_payload),
        "source_sha256": source_hashes,
        "visibility": "proposal_agents_may_read; heldout profiles and prior conclusions excluded",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
