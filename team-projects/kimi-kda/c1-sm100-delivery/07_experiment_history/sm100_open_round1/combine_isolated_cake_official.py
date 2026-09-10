#!/usr/bin/env python3
"""Combine independently loaded official and CAKE CUPTI measurements."""

import argparse
import json
import math
from pathlib import Path


ALIASES = {
    "h96_fixed8192": "h96_fixed_8192",
    "h96_uniform": "h96_uniform_1024x8",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--official", type=Path, required=True)
    p.add_argument("--cake-h12", type=Path, required=True)
    p.add_argument("--cake-legacy", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    official_payload = json.loads(a.official.read_text())
    official = {x["name"]: x for x in official_payload["cases"]}
    cake_rows = json.loads(a.cake_h12.read_text()) + json.loads(a.cake_legacy.read_text())
    cake = {ALIASES.get(x["name"], x["name"]): x for x in cake_rows}
    rows = []
    for name, base in official.items():
        candidate = cake[name]
        speedup = base["official_median_ms"] / candidate["median_ms"]
        rows.append({
            "name": name,
            "num_heads": base["num_heads"],
            "seq_lens": base["seq_lens"],
            "layout": base["layout"],
            "seed": base["seed"],
            "official_median_ms": base["official_median_ms"],
            "official_blocks": len(base["block_medians_ms"]["official_hmma"]),
            "cake_median_ms": candidate["median_ms"],
            "cake_blocks": len(candidate.get("block_medians_ms", [candidate["median_ms"]])),
            "speedup_official_over_cake": speedup,
            "cake_resolved_backend": candidate["resolved_backend"],
            "cake_variant": candidate["variant"],
            "cake_physical_variants": candidate["physical_variants"],
        })
    groups = {}
    for heads in (12, 96):
        values = [x["speedup_official_over_cake"] for x in rows if x["num_heads"] == heads]
        groups[f"h{heads}_geomean_speedup"] = math.prod(values) ** (1 / len(values))
        groups[f"h{heads}_range"] = [min(values), max(values)]
    payload = {
        "schema_version": "cake_vs_official_isolated_process_v1",
        "comparison_scope": "same seeds/shapes and CUPTI cold-L2 public state semantics; extensions never coexist in one process",
        "reason": "same-process extension loading changed absolute timing and CAKE route selection",
        "correctness_evidence": [
            "evidence/b300_stage11_cake_vs_official_h12.json",
            "experiments/sm100_open_round1/evidence/b300_stage12_cake_vs_official_legacy.json",
            "evidence/b300_stage3_public_paired.json"
        ],
        "rows": rows,
        "aggregate": groups,
        "limitation": "CAKE has one isolated timing block per case; use this as corrected magnitude evidence, not a paired confidence interval"
    }
    a.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(groups, indent=2))


if __name__ == "__main__":
    main()
