#!/usr/bin/env python3
"""Build a proof-linked Stage 3 public activation certificate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPECTED_ACTIVATIONS = {
    "h96_fixed8192",
    "h96_uniform",
    "h64_fixed8192",
    "h64_mixed",
    "h64_uniform",
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "evidence" / "b300_stage3_activation_certificate.json",
    )
    args = parser.parse_args()

    source_path = ROOT / "evidence" / "source_to_ir_a1418fd1ae.json"
    corpus_path = ROOT / "evidence" / "evolution_typed_corpus.json"
    paired_path = ROOT / "evidence" / "b300_stage3_public_paired.json"
    source = _load(source_path)
    corpus = _load(corpus_path)
    paired = _load(paired_path)

    if corpus["verifier_diagnostics"]:
        raise RuntimeError("typed evolution corpus is not verifier-clean")
    if corpus["candidate_count"] != 29 or corpus["semantic_fingerprint_count"] != 29:
        raise RuntimeError("expected 29 unique typed semantic candidates")
    if not paired["all_correct"] or not paired["scope_identical"]:
        raise RuntimeError("public paired evidence is incomplete")

    candidates = {candidate["variant"]: candidate for candidate in corpus["candidates"]}
    rows = []
    for row in paired["rows"]:
        lower, upper = row["bootstrap_95"]
        activated = row["decision"] == "activate"
        expected_decision = (
            "activate"
            if row["evolution"]["resolved_backend"] == "evolution" and lower > 1.0
            else "retain_cake"
        )
        if row["decision"] != expected_decision:
            raise RuntimeError(f"decision rule mismatch for {row['name']}")
        variant = row["evolution"]["variant"] if activated else None
        typed_delta = None if variant is None else candidates.get(variant)
        if activated and typed_delta is None:
            raise RuntimeError(f"activated variant {variant} has no typed certificate")
        rows.append(
            {
                "name": row["name"],
                "shape": {
                    "num_heads": row["num_heads"],
                    "seq_lens": row["seq_lens"],
                    "layout": row["layout"],
                },
                "decision": row["decision"],
                "resolved_backend": row["evolution"]["resolved_backend"],
                "variant": variant,
                "typed_semantic_delta": typed_delta,
                "cake_median_us": row["cake"]["median_ms"] * 1000.0,
                "deployment_median_us": row["evolution"]["median_ms"] * 1000.0,
                "speedup": row["speedup"],
                "bootstrap_95": [lower, upper],
                "correct": row["correct"],
            }
        )

    actual_activations = {row["name"] for row in rows if row["decision"] == "activate"}
    if actual_activations != EXPECTED_ACTIVATIONS:
        raise RuntimeError(
            f"activation set mismatch: expected {EXPECTED_ACTIVATIONS}, "
            f"got {actual_activations}"
        )
    retained = {row["name"] for row in rows if row["decision"] == "retain_cake"}
    if retained != {"h96_mixed"}:
        raise RuntimeError(f"unexpected retained shapes: {retained}")

    selected_speedups = [
        row["speedup"] if row["decision"] == "activate" else 1.0 for row in rows
    ]
    attempts = []
    for path in sorted((ROOT / "evidence" / "attempts").glob("*stage3*.json")):
        attempt = _load(path)
        attempts.append(
            {
                "file": str(path.relative_to(ROOT)),
                "sha256": _sha256(path),
                "job_id": attempt["scheduler"]["slurm_job_id"],
                "experiment": attempt["experiment"],
                "exit_code": attempt["exit_code"],
                "allocated_gpu_seconds": attempt["allocated_gpu_seconds"],
                "estimated_allocation_energy_joules": attempt["telemetry"][
                    "estimated_allocation_energy_joules"
                ],
            }
        )

    implementation_paths = (
        ROOT / "stage3_remote" / "flashinfer" / "kda.py",
        ROOT / "stage3_remote" / "flashinfer" / "kda_evolution.py",
        ROOT
        / "stage3_remote"
        / "benchmarks"
        / "bench_flash_kda_stage3_public_pair.py",
    )
    certificate = {
        "schema_version": 1,
        "stage": 3,
        "status": "dispatchable",
        "public_backend": "evolution",
        "auto_backend_changed": False,
        "fallback_backend": "cake",
        "proof_chain": {
            "source_to_ir": {
                "file": str(source_path.relative_to(ROOT)),
                "sha256": _sha256(source_path),
                "kernel_name": source["kernel_name"],
                "qk_first_edge_verified": not source["qk_first_edge_diagnostics"],
            },
            "typed_corpus": {
                "file": str(corpus_path.relative_to(ROOT)),
                "sha256": _sha256(corpus_path),
                "candidate_count": corpus["candidate_count"],
                "semantic_fingerprint_count": corpus[
                    "semantic_fingerprint_count"
                ],
                "verifier_diagnostics": corpus["verifier_diagnostics"],
            },
            "public_paired_benchmark": {
                "file": str(paired_path.relative_to(ROOT)),
                "sha256": _sha256(paired_path),
                "scope_identical": paired["scope_identical"],
                "all_correct": paired["all_correct"],
                "protocol": "paired CAKE/evolution, cold-L2 CUPTI, 20+100 x 2 blocks",
            },
        },
        "summary": {
            "shape_count": len(rows),
            "activation_count": len(actual_activations),
            "retain_cake_count": len(retained),
            "deployment_geomean_speedup": math.prod(selected_speedups)
            ** (1.0 / len(selected_speedups)),
            "minimum_activated_speedup_lower_bound": min(
                row["bootstrap_95"][0]
                for row in rows
                if row["decision"] == "activate"
            ),
            "stage3_attempt_gpu_seconds": sum(
                attempt["allocated_gpu_seconds"] for attempt in attempts
            ),
            "stage3_attempt_energy_joules": sum(
                attempt["estimated_allocation_energy_joules"] or 0.0
                for attempt in attempts
            ),
        },
        "activation_rows": rows,
        "implementation": [
            {
                "file": str(path.relative_to(ROOT)),
                "sha256": _sha256(path),
            }
            for path in implementation_paths
        ],
        "attempt_ledger": attempts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n")
    print(json.dumps(certificate["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
