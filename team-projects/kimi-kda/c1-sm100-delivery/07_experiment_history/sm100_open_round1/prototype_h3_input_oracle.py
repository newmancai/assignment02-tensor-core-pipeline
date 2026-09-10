#!/usr/bin/env python3
"""Input-level CPU correctness oracle for the H3 P/W rewrite.

The same q/k/v/g/beta inputs feed three routes:

* ``official`` mirrors the CHUNK=16 ordering and storage dtypes in
  ``FlashKDA/tests/torch_ref.py``;
* ``h3`` changes only U construction to precomputed P/W;
* ``naive`` is the FP64 token recurrence from ``fla_kda_ref/naive.py``.

This is a numerical falsifier, not a bit-exact replacement for the CUDA
reference: NumPy cannot reproduce ``tanh.approx``, ``ex2.approx.ftz`` or the
kernel's warp reduction instruction by instruction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


LOG2E = np.float32(1.4426950408889634)
LOWER_BOUND = np.float32(-5.0)
CHUNK = 16


def bf16(x: np.ndarray) -> np.ndarray:
    """Round float32 to BF16/RNE, retaining a float32 carrier."""

    value = np.ascontiguousarray(x, dtype=np.float32)
    bits = value.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return np.ascontiguousarray((rounded & np.uint32(0xFFFF0000)).view(np.float32))


def fp16(x: np.ndarray) -> np.ndarray:
    # The full L product can overflow in its upper triangle before that region
    # is masked, matching the reference's compute-then-tril ordering.
    with np.errstate(over="ignore"):
        return np.asarray(x, dtype=np.float16).astype(np.float32)


def fp16acc_mm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Small matrix multiply with an FP16 rounding boundary after each FMA."""

    acc = np.zeros((a.shape[0], b.shape[1]), dtype=np.float16)
    for kk in range(a.shape[1]):
        product = (a[:, kk, None].astype(np.float16) * b[kk, None, :].astype(np.float16)).astype(np.float16)
        acc = (acc + product).astype(np.float16)
    return acc.astype(np.float32)


def normalize_input(x: np.ndarray) -> np.ndarray:
    """Shared BF16 normalization boundary for all routes."""

    x32 = np.asarray(x, dtype=np.float32)
    inv = np.float32(1.0) / np.sqrt(np.sum(x32 * x32, axis=-1, keepdims=True) + np.float32(1e-6))
    return bf16(x32 * inv)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x32 = np.asarray(x, dtype=np.float32)
    return np.float32(0.5) * np.tanh(np.float32(0.5) * x32) + np.float32(0.5)


def activate_gate(raw_g: np.ndarray, a_log: np.ndarray, dt_bias: np.ndarray) -> np.ndarray:
    """Mirror torch_ref.py:161-164, producing per-token log2 decays."""

    shifted = raw_g.astype(np.float32) + dt_bias[None, :]
    a_exp = np.exp2(a_log.astype(np.float32) * LOG2E)
    return LOWER_BOUND * LOG2E * sigmoid(a_exp[None, :] * shifted)


def metrics(actual: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    diff = actual.astype(np.float64) - reference.astype(np.float64)
    ref = reference.astype(np.float64)
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "relative_l2": float(np.linalg.norm(diff.ravel()) / max(np.linalg.norm(ref.ravel()), 1e-30)),
        "normalized_linf": float(np.max(np.abs(diff)) / max(np.max(np.abs(ref)), 1e-30)),
    }


def prepare_inputs(seed: int, lengths: list[int], d: int, strong: bool, nonzero_state: bool) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    total = sum(lengths)
    q = normalize_input(rng.normal(0.0, 1.0, (total, d)))
    k = normalize_input(rng.normal(0.0, 1.0, (total, d)))
    v = bf16(rng.normal(0.0, 0.25, (total, d)))
    raw_g = bf16(rng.normal(-2.0 if strong else 0.0, 0.7, (total, d)))
    beta_logits = bf16(rng.normal(0.0, 1.0, total))
    a_log = rng.uniform(-0.25, 0.25, d).astype(np.float32)
    dt_bias = rng.uniform(-1.0 if strong else -0.2, 0.1, d).astype(np.float32)
    g_log2 = activate_gate(raw_g, a_log, dt_bias)
    beta = bf16(sigmoid(beta_logits))
    initial = (
        bf16(rng.normal(0.0, 0.08, (len(lengths), d, d)))
        if nonzero_state
        else np.zeros((len(lengths), d, d), dtype=np.float32)
    )
    offsets = np.cumsum([0] + lengths).tolist()
    return {
        "q": q,
        "k": k,
        "v": v,
        "raw_g": raw_g,
        "beta_logits": beta_logits,
        "g_log2": g_log2,
        "beta": beta,
        "a_log": a_log,
        "dt_bias": dt_bias,
        "initial": initial,
        "cu_seqlens": offsets,
        "scale": np.float32(d ** -0.5),
    }


def local_inverse(kd: np.ndarray, kinv: np.ndarray, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mirror torch_ref.py:211-229, including the FP16 Neumann solve."""

    c = kd.shape[0]
    lower = fp16((kd @ kinv.T).astype(np.float32))
    lower = fp16(np.tril(lower, k=-1) * fp16(beta[:, None]))
    mq_identity = np.eye(c, dtype=np.float32)
    inv = fp16(mq_identity - lower)
    l2 = fp16acc_mm(lower, lower)
    inv = fp16(inv + fp16acc_mm(inv, l2))
    l4 = fp16acc_mm(l2, l2)
    inv = fp16(inv + fp16acc_mm(inv, l4))
    l8 = fp16acc_mm(l4, l4)
    inv = fp16(inv + fp16acc_mm(inv, l8))
    return bf16(inv), lower


def chunk_step(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    g_log2: np.ndarray,
    beta: np.ndarray,
    state: np.ndarray,
    scale: np.float32,
    route: str,
) -> tuple[np.ndarray, np.ndarray]:
    actual, d = q.shape
    q_pad = np.zeros((CHUNK, d), dtype=np.float32)
    k_pad = np.zeros_like(q_pad)
    v_pad = np.zeros_like(q_pad)
    g_pad = np.zeros_like(q_pad)
    b_pad = np.zeros(CHUNK, dtype=np.float32)
    q_pad[:actual], k_pad[:actual], v_pad[:actual] = q, k, v
    g_pad[:actual], b_pad[:actual] = g_log2, beta

    gcum = np.cumsum(g_pad, axis=0, dtype=np.float32)
    epos = bf16(np.exp2(gcum).astype(np.float32))
    eneg = bf16(np.exp2(-gcum).astype(np.float32))
    kd = bf16(k_pad * epos)
    qd = bf16(bf16(q_pad * epos) * bf16(np.asarray(scale)))
    kinv = bf16(k_pad * eneg)
    gtotal_bf16 = bf16(np.exp2(gcum[-1]).astype(np.float32))
    kr = bf16(kinv * gtotal_bf16[None, :])
    inv, _ = local_inverse(kd, kinv, b_pad)
    mqk = bf16(qd @ kinv.T)
    mqk = bf16(np.tril(mqk))

    if route == "official":
        ks = bf16(kd @ state)
        residual = bf16(bf16(v_pad - ks) * b_pad[:, None])
        u = bf16(inv @ residual)
    elif route == "h3":
        p = bf16(inv @ bf16(b_pad[:, None] * v_pad))
        w = bf16(inv @ bf16(b_pad[:, None] * kd))
        u = bf16(p - w @ state)
    else:
        raise ValueError(route)

    out = bf16(bf16(qd @ state) + bf16(mqk @ u))
    gate_total = np.exp2(gcum[-1]).astype(np.float32)
    state = bf16(gate_total[:, None] * state + kr.T @ u)
    return out[:actual].copy(), state


def run_chunked(data: dict[str, object], route: str) -> tuple[np.ndarray, np.ndarray]:
    q, k, v = data["q"], data["k"], data["v"]
    g_log2, beta = data["g_log2"], data["beta"]
    initial, offsets = data["initial"], data["cu_seqlens"]
    assert isinstance(q, np.ndarray) and isinstance(k, np.ndarray) and isinstance(v, np.ndarray)
    assert isinstance(g_log2, np.ndarray) and isinstance(beta, np.ndarray)
    assert isinstance(initial, np.ndarray) and isinstance(offsets, list)
    outputs = np.zeros_like(v)
    final = np.empty_like(initial)
    for seq in range(len(offsets) - 1):
        state = initial[seq].copy()
        bos, eos = offsets[seq], offsets[seq + 1]
        for start in range(bos, eos, CHUNK):
            end = min(start + CHUNK, eos)
            out, state = chunk_step(
                q[start:end], k[start:end], v[start:end], g_log2[start:end], beta[start:end],
                state, data["scale"], route,
            )
            outputs[start:end] = out
        final[seq] = state
    return outputs, final


def run_naive(data: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    """FP64 form of fla_kda_ref/naive.py:59-63 on quantized inputs."""

    q = data["q"].astype(np.float64) * float(data["scale"])
    k, v = data["k"].astype(np.float64), data["v"].astype(np.float64)
    g_log2, beta = data["g_log2"].astype(np.float64), data["beta"].astype(np.float64)
    initial, offsets = data["initial"].astype(np.float64), data["cu_seqlens"]
    outputs = np.zeros_like(v)
    final = np.empty_like(initial)
    for seq in range(len(offsets) - 1):
        state = initial[seq].copy()
        for token in range(offsets[seq], offsets[seq + 1]):
            state *= np.exp2(g_log2[token])[:, None]
            residual = v[token] - k[token] @ state
            state += beta[token] * k[token, :, None] * residual[None, :]
            outputs[token] = q[token] @ state
        final[seq] = state
    return outputs, final


def one_case(seed: int, lengths: list[int], d: int, strong: bool, nonzero: bool) -> dict[str, object]:
    data = prepare_inputs(seed, lengths, d, strong, nonzero)
    official_out, official_state = run_chunked(data, "official")
    h3_out, h3_state = run_chunked(data, "h3")
    naive_out, naive_state = run_naive(data)
    result: dict[str, object] = {
        "seed": seed,
        "lengths": lengths,
        "packed": len(lengths) > 1,
        "strong_decay": strong,
        "nonzero_initial_state": nonzero,
    }
    for name, official, h3, naive in (
        ("output", official_out, h3_out, naive_out),
        ("final_state", official_state, h3_state, naive_state),
    ):
        result[name] = {
            "official_vs_naive": metrics(official, naive),
            "h3_vs_naive": metrics(h3, naive),
            "h3_vs_official": metrics(h3, official),
            "bf16_equal": bool(np.array_equal(h3, official)),
            "finite": bool(np.isfinite(official).all() and np.isfinite(h3).all() and np.isfinite(naive).all()),
        }
    return result


def continuous_case(seed: int, d: int) -> dict[str, object]:
    """Compare T65 with calls T32 then T33 using the returned state."""

    full = prepare_inputs(seed, [65], d, strong=False, nonzero_state=True)
    result: dict[str, object] = {"seed": seed, "split": [32, 33]}
    for route in ("official", "h3"):
        one_out, one_state = run_chunked(full, route)
        pieces = []
        state = full["initial"].copy()
        for start, end in ((0, 32), (32, 65)):
            part = dict(full)
            for key in ("q", "k", "v", "raw_g", "beta_logits", "g_log2", "beta"):
                part[key] = full[key][start:end]
            part["initial"] = state
            part["cu_seqlens"] = [0, end - start]
            out, state = run_chunked(part, route)
            pieces.append(out)
        joined = np.concatenate(pieces, axis=0)
        result[route] = {
            "output_equal": bool(np.array_equal(joined, one_out)),
            "final_state_equal": bool(np.array_equal(state, one_state)),
            "output_max_abs": float(np.max(np.abs(joined - one_out))),
            "final_state_max_abs": float(np.max(np.abs(state - one_state))),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    specs = [
        ([16], False, False),
        ([17], False, True),
        ([37], True, True),
        ([65], False, True),
        ([17, 33, 65], False, True),
        ([4, 8, 12], True, True),
    ]
    cases = [
        one_case(seed, lengths, args.d, strong, nonzero)
        for lengths, strong, nonzero in specs
        for seed in range(args.seeds)
    ]
    continuous = [continuous_case(seed, args.d) for seed in range(args.seeds)]

    failures = []
    for case in cases:
        for target in ("output", "final_state"):
            row = case[target]
            official = row["official_vs_naive"]
            h3 = row["h3_vs_naive"]
            if not row["finite"]:
                failures.append({"case": case, "target": target, "reason": "nonfinite"})
            if h3["relative_l2"] > max(2.0 * official["relative_l2"], 0.03):
                failures.append({"seed": case["seed"], "lengths": case["lengths"], "target": target, "reason": "relative_l2"})
            if h3["normalized_linf"] > max(4.0 * official["normalized_linf"], 0.08):
                failures.append({"seed": case["seed"], "lengths": case["lengths"], "target": target, "reason": "normalized_linf"})
    for row in continuous:
        for route in ("official", "h3"):
            if not row[route]["output_equal"] or not row[route]["final_state_equal"]:
                failures.append({"seed": row["seed"], "route": route, "reason": "continuous_handoff"})

    summary = {}
    for target in ("output", "final_state"):
        summary[target] = {
            "all_h3_official_bf16_equal": all(case[target]["bf16_equal"] for case in cases),
            "worst_official_relative_l2": max(case[target]["official_vs_naive"]["relative_l2"] for case in cases),
            "worst_h3_relative_l2": max(case[target]["h3_vs_naive"]["relative_l2"] for case in cases),
            "worst_official_normalized_linf": max(case[target]["official_vs_naive"]["normalized_linf"] for case in cases),
            "worst_h3_normalized_linf": max(case[target]["h3_vs_naive"]["normalized_linf"] for case in cases),
        }
    payload = {
        "schema_version": "sm100_open_h3_input_oracle_v1",
        "scope": "cpu_input_level_numerical_falsifier_not_cuda_bitwise_or_gpu_performance",
        "shape": {"chunk": CHUNK, "d": args.d, "value_dim": args.d},
        "case_count": len(cases),
        "numerical_contract": {
            "compatibility": "algorithm_equivalent_not_official_bitwise",
            "common_reference": "FP64 FLA-style per-token recurrence on shared quantized inputs",
            "relative_l2_gate": "h3 <= max(2 * official, 0.03) per case and target",
            "normalized_linf_gate": "h3 <= max(4 * official, 0.08) per case and target",
            "continuous_call_gate": "bitwise equality to one call when split on a C16 boundary",
        },
        "known_approximation": "NumPy uses exact tanh/exp2 and ordinary normalization, not CUDA approx instructions/reduction order",
        "summary": summary,
        "continuous_calls": continuous,
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
