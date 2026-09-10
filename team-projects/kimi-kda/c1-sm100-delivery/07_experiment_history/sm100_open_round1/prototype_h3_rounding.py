#!/usr/bin/env python3
"""Numerical falsifier for the H3 precomputed P/W KDA rewrite.

This is deliberately a CPU model, not a performance benchmark.  It compares
the current chunk-local ordering

    U = INV @ (beta * (V - K @ S))

with a state-independent preparation

    P = INV @ (beta * V)
    W = INV @ (beta * K)
    U = P - W @ S

under explicit BF16 storage boundaries.  It also propagates U through the
output and final-state consumers so a locally small U error cannot hide a
larger externally visible error.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def bf16(x: np.ndarray) -> np.ndarray:
    """Round float32 values to BF16 (round-to-nearest-even), returned as f32."""

    value = np.asarray(x, dtype=np.float32)
    bits = value.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def metrics(actual: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    diff = actual.astype(np.float64) - reference.astype(np.float64)
    ref64 = reference.astype(np.float64)
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "relative_l2": float(np.linalg.norm(diff.ravel()) / max(np.linalg.norm(ref64.ravel()), 1e-30)),
        "normalized_linf": float(np.max(np.abs(diff)) / max(np.max(np.abs(ref64)), 1e-30)),
    }


def consumers(
    u: np.ndarray,
    q: np.ndarray,
    mqk: np.ndarray,
    state: np.ndarray,
    k_restored: np.ndarray,
    gate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    q_state = bf16(q @ state)
    local = bf16(mqk @ u)
    out = bf16(q_state + local)
    delta = k_restored.T @ u
    next_state = bf16(gate[:, None] * state + delta)
    return out, next_state


def one_case(seed: int, scale: float, c: int, d: int, vdim: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    k = bf16(rng.normal(0.0, scale, (c, d)))
    value = bf16(rng.normal(0.0, scale, (c, vdim)))
    state = bf16(rng.normal(0.0, scale, (d, vdim)))
    q = bf16(rng.normal(0.0, scale, (c, d)))
    mqk = bf16(np.tril(rng.normal(0.0, scale, (c, c))))
    k_restored = bf16(rng.normal(0.0, scale, (c, d)))
    beta = bf16(rng.uniform(0.05, 0.95, c))
    gate = rng.uniform(0.7, 1.0, d).astype(np.float32)

    # Keep INV representative of a stable unit-lower-triangular local solve.
    inv = np.eye(c, dtype=np.float32)
    inv += np.tril(rng.normal(0.0, min(scale, 0.2), (c, c)), k=-1).astype(np.float32)
    inv = bf16(inv)

    # Approximate the source-visible BF16 boundaries at K2 phases 1 and 3.
    ks = bf16(k @ state)
    residual = bf16(bf16(value - ks) * beta[:, None])
    u_official = bf16(inv @ residual)

    # H3 stores state-independent P and W as BF16 workspace values.  The W@S
    # contraction accumulates in FP32 and is rounded only with the final U.
    p = bf16(inv @ bf16(beta[:, None] * value))
    w = bf16(inv @ bf16(beta[:, None] * k))
    u_h3 = bf16(p - (w @ state))

    out_official, state_official = consumers(u_official, q, mqk, state, k_restored, gate)
    out_h3, state_h3 = consumers(u_h3, q, mqk, state, k_restored, gate)

    # Common high-precision reference for an algorithm-equivalence screen.
    inv64, beta64 = inv.astype(np.float64), beta.astype(np.float64)
    k64, value64, state64 = k.astype(np.float64), value.astype(np.float64), state.astype(np.float64)
    u_ref = inv64 @ (beta64[:, None] * (value64 - k64 @ state64))
    p_ref = inv64 @ (beta64[:, None] * value64)
    w_ref = inv64 @ (beta64[:, None] * k64)
    identity_error = float(np.max(np.abs(u_ref - (p_ref - w_ref @ state64))))
    out_ref = q.astype(np.float64) @ state64 + mqk.astype(np.float64) @ u_ref
    state_ref = gate.astype(np.float64)[:, None] * state64 + k_restored.astype(np.float64).T @ u_ref

    targets = {
        "u": (u_official, u_h3, u_ref),
        "output": (out_official, out_h3, out_ref),
        "final_state": (state_official, state_h3, state_ref),
    }
    target_results = {}
    for name, (official, h3, reference) in targets.items():
        target_results[name] = {
            "official_vs_fp64": metrics(official, reference),
            "h3_vs_fp64": metrics(h3, reference),
            "h3_vs_official": metrics(h3, official),
            "bf16_equal": bool(np.array_equal(h3, official)),
        }
    return {
        "seed": seed,
        "scale": scale,
        "identity_max_abs": identity_error,
        "targets": target_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument("--value-dim", type=int, default=128)
    args = parser.parse_args()

    scales = (0.02, 0.1, 0.5, 1.0)
    cases = [
        one_case(seed, scale, args.chunk, args.d, args.value_dim)
        for scale in scales
        for seed in range(args.seeds)
    ]

    target_names = ("u", "output", "final_state")
    summary = {}
    for target in target_names:
        official_rel = max(c["targets"][target]["official_vs_fp64"]["relative_l2"] for c in cases)
        h3_rel = max(c["targets"][target]["h3_vs_fp64"]["relative_l2"] for c in cases)
        official_linf = max(c["targets"][target]["official_vs_fp64"]["normalized_linf"] for c in cases)
        h3_linf = max(c["targets"][target]["h3_vs_fp64"]["normalized_linf"] for c in cases)
        summary[target] = {
            "all_bf16_equal": all(c["targets"][target]["bf16_equal"] for c in cases),
            "worst_official_relative_l2": official_rel,
            "worst_h3_relative_l2": h3_rel,
            "worst_official_normalized_linf": official_linf,
            "worst_h3_normalized_linf": h3_linf,
            "relative_l2_amplification": h3_rel / max(official_rel, 1e-30),
            "normalized_linf_amplification": h3_linf / max(official_linf, 1e-30),
        }

    # Fixed before inspecting results: this is only an entry gate for a full
    # multi-chunk oracle, not a production correctness threshold.
    numerical_gate = all(
        row["worst_h3_relative_l2"] <= max(2.0 * row["worst_official_relative_l2"], 0.01)
        and row["worst_h3_normalized_linf"] <= max(4.0 * row["worst_official_normalized_linf"], 0.02)
        for row in summary.values()
    )
    payload = {
        "schema_version": "sm100_open_h3_rounding_probe_v1",
        "scope": "single_chunk_cpu_numerical_falsifier_not_gpu_performance",
        "shape": {"chunk": args.chunk, "d": args.d, "value_dim": args.value_dim},
        "case_count": len(cases),
        "scales": scales,
        "numerical_contract": {
            "compatibility": "algorithm_equivalent_not_official_bitwise",
            "relative_l2_gate": "h3 <= max(2 * official, 0.01) for U/output/final_state",
            "normalized_linf_gate": "h3 <= max(4 * official, 0.02) for U/output/final_state",
        },
        "max_real_identity_error": max(c["identity_max_abs"] for c in cases),
        "summary": summary,
        "entry_gate_passed": numerical_gate,
        "cases": cases,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
