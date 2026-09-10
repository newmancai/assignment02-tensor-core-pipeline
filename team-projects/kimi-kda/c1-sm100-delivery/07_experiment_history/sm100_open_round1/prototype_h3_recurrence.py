#!/usr/bin/env python3
"""Multi-chunk numerical screen for H3 P/W precomputation.

Inputs begin at FlashKDA's chunk-workspace boundary.  This isolates recurrence,
rounding, tails, and state propagation; it does not validate K1 construction or
claim input-level equivalence to the official/FLA reference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from prototype_h3_rounding import bf16, metrics


def make_chunk(
    rng: np.random.Generator,
    actual: int,
    c: int,
    d: int,
    vdim: int,
    scale: float,
    decay: str,
    cancellation_state: np.ndarray | None = None,
) -> dict[str, np.ndarray | int]:
    k = bf16(rng.normal(0.0, scale, (c, d)))
    value = bf16(rng.normal(0.0, scale, (c, vdim)))
    if cancellation_state is not None:
        value = bf16(k @ cancellation_state + rng.normal(0.0, scale * 1e-2, (c, vdim)))
    q = bf16(rng.normal(0.0, scale, (c, d)))
    mqk = bf16(np.tril(rng.normal(0.0, scale, (c, c))))
    kr = bf16(rng.normal(0.0, scale, (c, d)))
    beta = bf16(rng.uniform(0.05, 0.95, c))
    inv = np.eye(c, dtype=np.float32)
    inv += np.tril(rng.normal(0.0, min(scale, 0.15), (c, c)), k=-1).astype(np.float32)
    inv = bf16(inv)
    if decay == "strong":
        gate = rng.uniform(0.05, 0.5, d).astype(np.float32)
    elif decay == "weak":
        gate = rng.uniform(0.98, 1.0, d).astype(np.float32)
    else:
        gate = rng.uniform(0.3, 1.0, d).astype(np.float32)

    if actual < c:
        k[actual:] = 0
        value[actual:] = 0
        q[actual:] = 0
        kr[actual:] = 0
        beta[actual:] = 0
        mqk[actual:, :] = 0
        mqk[:, actual:] = 0
        inv[actual:, :] = 0
        inv[:, actual:] = 0
        inv[np.arange(actual, c), np.arange(actual, c)] = 1
    return {
        "actual": actual,
        "k": k,
        "value": value,
        "q": q,
        "mqk": mqk,
        "kr": kr,
        "beta": beta,
        "inv": inv,
        "gate": gate,
    }


def step(chunk: dict[str, np.ndarray | int], state: np.ndarray, route: str) -> tuple[np.ndarray, np.ndarray]:
    actual = int(chunk["actual"])
    k = chunk["k"]
    value = chunk["value"]
    q = chunk["q"]
    mqk = chunk["mqk"]
    kr = chunk["kr"]
    beta = chunk["beta"]
    inv = chunk["inv"]
    gate = chunk["gate"]
    assert isinstance(k, np.ndarray) and isinstance(value, np.ndarray)
    assert isinstance(q, np.ndarray) and isinstance(mqk, np.ndarray)
    assert isinstance(kr, np.ndarray) and isinstance(beta, np.ndarray)
    assert isinstance(inv, np.ndarray) and isinstance(gate, np.ndarray)

    if route == "official":
        ks = bf16(k @ state)
        residual = bf16(bf16(value - ks) * beta[:, None])
        u = bf16(inv @ residual)
        out = bf16(bf16(q @ state) + bf16(mqk @ u))
        next_state = bf16(gate[:, None] * state + kr.T @ u)
    elif route == "h3":
        p = bf16(inv @ bf16(beta[:, None] * value))
        w = bf16(inv @ bf16(beta[:, None] * k))
        u = bf16(p - w @ state)
        out = bf16(bf16(q @ state) + bf16(mqk @ u))
        next_state = bf16(gate[:, None] * state + kr.T @ u)
    elif route == "fp64":
        state = state.astype(np.float64)
        k64, value64 = k.astype(np.float64), value.astype(np.float64)
        beta64, inv64 = beta.astype(np.float64), inv.astype(np.float64)
        u = inv64 @ (beta64[:, None] * (value64 - k64 @ state))
        out = q.astype(np.float64) @ state + mqk.astype(np.float64) @ u
        next_state = gate.astype(np.float64)[:, None] * state + kr.astype(np.float64).T @ u
    else:
        raise ValueError(route)
    return out[:actual].copy(), next_state


def run_chunks(chunks: list[dict[str, np.ndarray | int]], initial: np.ndarray, route: str) -> tuple[list[np.ndarray], np.ndarray]:
    state = initial.copy()
    outputs = []
    for chunk in chunks:
        out, state = step(chunk, state, route)
        outputs.append(out)
    return outputs, state


def scenario(seed: int, length: int, decay: str, cancellation: bool, c: int, d: int, vdim: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    initial = bf16(rng.normal(0.0, 0.15, (d, vdim)))
    chunks = []
    remaining = length
    while remaining > 0:
        actual = min(c, remaining)
        chunks.append(make_chunk(
            rng,
            actual,
            c,
            d,
            vdim,
            0.08,
            decay,
            initial if cancellation and not chunks else None,
        ))
        remaining -= actual

    official_out, official_state = run_chunks(chunks, initial, "official")
    h3_out, h3_state = run_chunks(chunks, initial, "h3")
    ref_out, ref_state = run_chunks(chunks, initial.astype(np.float64), "fp64")
    official_joined = np.concatenate(official_out, axis=0)
    h3_joined = np.concatenate(h3_out, axis=0)
    ref_joined = np.concatenate(ref_out, axis=0)

    # Same chunk identities and boundary, but state is handed through two calls.
    split = max(1, len(chunks) // 2)
    first_out, first_state = run_chunks(chunks[:split], initial, "h3")
    second_out, split_state = run_chunks(chunks[split:], first_state, "h3")
    split_joined = np.concatenate(first_out + second_out, axis=0)
    handoff_equal = bool(np.array_equal(split_joined, h3_joined) and np.array_equal(split_state, h3_state))

    return {
        "seed": seed,
        "length": length,
        "chunks": len(chunks),
        "decay": decay,
        "cancellation": cancellation,
        "handoff_bf16_equal": handoff_equal,
        "output": {
            "official_vs_fp64": metrics(official_joined, ref_joined),
            "h3_vs_fp64": metrics(h3_joined, ref_joined),
            "h3_vs_official": metrics(h3_joined, official_joined),
            "bf16_equal": bool(np.array_equal(h3_joined, official_joined)),
        },
        "final_state": {
            "official_vs_fp64": metrics(official_state, ref_state),
            "h3_vs_fp64": metrics(h3_state, ref_state),
            "h3_vs_official": metrics(h3_state, official_state),
            "bf16_equal": bool(np.array_equal(h3_state, official_state)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument("--value-dim", type=int, default=128)
    args = parser.parse_args()

    specs = [(17, "mixed", False), (33, "mixed", False), (65, "mixed", False)]
    specs += [(32, "strong", False), (64, "mixed", False), (256, "weak", False), (1024, "weak", False)]
    specs += [(64, "weak", True)]
    cases = [
        scenario(seed, length, decay, cancellation, args.chunk, args.d, args.value_dim)
        for length, decay, cancellation in specs
        for seed in range(4)
    ]

    failures = []
    for case in cases:
        if not case["handoff_bf16_equal"]:
            failures.append({"seed": case["seed"], "length": case["length"], "reason": "handoff"})
        for target in ("output", "final_state"):
            official = case[target]["official_vs_fp64"]
            h3 = case[target]["h3_vs_fp64"]
            if h3["relative_l2"] > max(2.0 * official["relative_l2"], 0.02):
                failures.append({"seed": case["seed"], "length": case["length"], "target": target, "reason": "relative_l2"})
            if h3["normalized_linf"] > max(4.0 * official["normalized_linf"], 0.05):
                failures.append({"seed": case["seed"], "length": case["length"], "target": target, "reason": "normalized_linf"})

    summary = {}
    for target in ("output", "final_state"):
        summary[target] = {
            "all_bf16_equal": all(case[target]["bf16_equal"] for case in cases),
            "worst_official_relative_l2": max(case[target]["official_vs_fp64"]["relative_l2"] for case in cases),
            "worst_h3_relative_l2": max(case[target]["h3_vs_fp64"]["relative_l2"] for case in cases),
            "worst_official_normalized_linf": max(case[target]["official_vs_fp64"]["normalized_linf"] for case in cases),
            "worst_h3_normalized_linf": max(case[target]["h3_vs_fp64"]["normalized_linf"] for case in cases),
        }
    payload = {
        "schema_version": "sm100_open_h3_recurrence_probe_v1",
        "scope": "chunk_workspace_boundary_cpu_numerical_falsifier_not_input_level_or_gpu_performance",
        "shape": {"chunk": args.chunk, "d": args.d, "value_dim": args.value_dim},
        "case_count": len(cases),
        "numerical_contract": {
            "compatibility": "algorithm_equivalent_not_official_bitwise",
            "relative_l2_gate": "h3 <= max(2 * official, 0.02) per scenario",
            "normalized_linf_gate": "h3 <= max(4 * official, 0.05) per scenario",
            "state_handoff": "bitwise equal for identical precomputed chunk sequence",
        },
        "summary": summary,
        "entry_gate_passed": not failures,
        "failures": failures,
        "cases": cases,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
