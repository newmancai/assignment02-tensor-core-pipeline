#!/usr/bin/env python3
"""Emit a typed semantic inventory for frozen Blackwell KDA candidates."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from kda_ir.semantic_delta import corpus_from_source_dir, deduplicate_semantics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    corpus = corpus_from_source_dir(args.source_dir)
    unique = deduplicate_semantics(corpus)
    topology_counts = Counter(delta.topology.value for delta in corpus)
    report = {
        "schema_version": 1,
        "analysis": "typed-semantic-delta-corpus",
        "candidate_count": len(corpus),
        "semantic_fingerprint_count": len(unique),
        "duplicate_semantics": len(corpus) - len(unique),
        "topology_counts": dict(sorted(topology_counts.items())),
        "verifier_diagnostics": [],
        "candidates": [delta.as_dict() for delta in corpus],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in report if key != "candidates"}, sort_keys=True))


if __name__ == "__main__":
    main()
