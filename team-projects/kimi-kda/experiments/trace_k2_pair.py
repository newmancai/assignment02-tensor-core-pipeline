"""Capture a compact Nsight Systems trace for official V128 vs auto ValueSlice."""

from __future__ import annotations

import argparse
import json
import math
import os

import flash_kda
import torch


DIM = 128


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    shape = (1, args.tokens, args.heads, DIM)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    g = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    beta = torch.randn(shape[:-1], device="cuda", dtype=torch.bfloat16)
    a_log = torch.rand(args.heads, device="cuda", dtype=torch.float32)
    dt_bias = torch.rand((args.heads, DIM), device="cuda", dtype=torch.float32)
    initial_state = torch.randn(
        (1, args.heads, DIM, DIM), device="cuda", dtype=torch.bfloat16
    )
    final_state = torch.empty_like(initial_state)
    out = torch.empty_like(v)

    def run_once() -> None:
        flash_kda.fwd(
            q, k, v, g, beta, 1.0 / math.sqrt(DIM), out,
            a_log, dt_bias, -5.0, initial_state, final_state,
        )

    # Warm both paths before capture so allocation/JIT noise stays outside the trace.
    os.environ["FLASH_KDA_K2_VALUE_SLICE"] = "128"
    run_once()
    os.environ.pop("FLASH_KDA_K2_VALUE_SLICE", None)
    run_once()
    torch.cuda.synchronize()

    decision = flash_kda.explain_k2_dispatch(
        q, initial_state=initial_state, final_state=final_state
    )
    print("auto_dispatch=" + json.dumps(decision, sort_keys=True))

    torch.cuda.profiler.start()
    os.environ["FLASH_KDA_K2_VALUE_SLICE"] = "128"
    with torch.cuda.nvtx.range("official_v128"):
        for _ in range(args.iterations):
            run_once()
        torch.cuda.synchronize()

    os.environ.pop("FLASH_KDA_K2_VALUE_SLICE", None)
    with torch.cuda.nvtx.range("auto_valueslice"):
        for _ in range(args.iterations):
            run_once()
        torch.cuda.synchronize()
    torch.cuda.profiler.stop()

    print("PASS")


if __name__ == "__main__":
    main()
