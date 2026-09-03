"""Benchmark K3-relevant fixed and packed FlashKDA shapes on one GPU.

The V128 path is the compatibility baseline from the patched extension.  It is
deliberately not labelled "official": parity with the untouched upstream build
is checked separately.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

import flash_kda
import flash_kda_C


DIM = 128
LOWER_BOUND = -5.0
MODES = ("compat_v128", "v16", "v32", "v64", "auto")
CASES = (
    ("fixed_1x8192", [8192], False),
    ("packed_1x8192", [8192], True),
    ("packed_ragged6", [1300, 547, 2048, 963, 271, 3063], True),
    ("packed_8x1024", [1024] * 8, True),
    ("packed_32x256", [256] * 32, True),
)


def make_inputs(lengths: list[int], heads: int, seed: int, packed: bool):
    torch.manual_seed(seed)
    total = sum(lengths)
    shape = (1, total, heads, DIM)
    q = F.normalize(torch.randn(shape, device="cuda", dtype=torch.float32), p=2, dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn(shape, device="cuda", dtype=torch.float32), p=2, dim=-1).to(torch.bfloat16)
    v = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    g = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    beta = torch.randn(shape[:-1], device="cuda", dtype=torch.bfloat16)
    a_log = torch.rand(heads, device="cuda", dtype=torch.float32)
    dt_bias = torch.rand((heads, DIM), device="cuda", dtype=torch.float32)
    nseq = len(lengths) if packed else 1
    initial_state = torch.randn((nseq, heads, DIM, DIM), device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(v)
    final_state = torch.empty_like(initial_state)
    cu_seqlens = None
    if packed:
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        cu_seqlens = torch.tensor(offsets, device="cuda", dtype=torch.int64)
    return (q, k, v, g, beta, a_log, dt_bias, initial_state, out, final_state, cu_seqlens)


def select_mode(mode: str) -> None:
    os.environ.pop("FLASH_KDA_K2_DISPATCH", None)
    if mode == "auto":
        os.environ.pop("FLASH_KDA_K2_VALUE_SLICE", None)
        return
    value = {"compat_v128": 128, "v16": 16, "v32": 32, "v64": 64}[mode]
    os.environ["FLASH_KDA_K2_VALUE_SLICE"] = str(value)


def make_runner(inputs):
    q, k, v, g, beta, a_log, dt_bias, initial_state, out, final_state, cu_seqlens = inputs

    def run():
        flash_kda.fwd(
            q,
            k,
            v,
            g,
            beta,
            1.0 / math.sqrt(DIM),
            out,
            a_log,
            dt_bias,
            LOWER_BOUND,
            initial_state,
            final_state,
            cu_seqlens,
        )

    return run


def measure(run, warmup: int, iterations: int) -> dict[str, float]:
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for i in range(iterations):
        starts[i].record()
        run()
        ends[i].record()
    torch.cuda.synchronize()
    samples = [float(start.elapsed_time(end)) for start, end in zip(starts, ends)]
    ordered = sorted(samples)
    p90_index = min(len(ordered) - 1, math.ceil(0.90 * len(ordered)) - 1)
    return {
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "p90_ms": ordered[p90_index],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "stdev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


@torch.inference_mode()
def check_bitwise(inputs) -> dict[str, bool]:
    runner = make_runner(inputs)
    select_mode("compat_v128")
    runner()
    torch.cuda.synchronize()
    baseline_out = inputs[8].clone()
    baseline_state = inputs[9].clone()
    results = {}
    for mode in ("v16", "v32", "v64"):
        select_mode(mode)
        runner()
        torch.cuda.synchronize()
        results[f"{mode}_out_equal"] = torch.equal(inputs[8], baseline_out)
        results[f"{mode}_state_equal"] = torch.equal(inputs[9], baseline_state)
    return results


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    print(f"extension={flash_kda_C.__file__}")
    print(f"device={flash_kda_C.get_device_characteristics()}")
    rows = []
    for case_index, (case_name, lengths, packed) in enumerate(CASES):
        inputs = make_inputs(lengths, args.heads, 20260903 + case_index, packed)
        equality = check_bitwise(inputs)
        if not all(equality.values()):
            raise AssertionError((case_name, equality))
        q, _, _, _, _, _, _, initial_state, _, final_state, cu_seqlens = inputs
        select_mode("auto")
        decision = flash_kda.explain_k2_dispatch(q, initial_state, final_state, cu_seqlens)
        print(f"case={case_name} lengths={lengths} auto_decision={decision} correctness={equality}")

        runner = make_runner(inputs)
        for repeat in range(args.repeats):
            order = list(MODES)
            random.Random(20260903 + case_index * 100 + repeat).shuffle(order)
            for order_index, mode in enumerate(order):
                select_mode(mode)
                stats = measure(runner, args.warmup, args.iterations)
                row = {
                    "case": case_name,
                    "packed": packed,
                    "nseq": len(lengths) if packed else 1,
                    "total_tokens": sum(lengths),
                    "heads": args.heads,
                    "mode": mode,
                    "repeat": repeat,
                    "order_index": order_index,
                    **stats,
                }
                rows.append(row)
                print(
                    f"result case={case_name} repeat={repeat} mode={mode} "
                    f"median_ms={stats['median_ms']:.6f} p90_ms={stats['p90_ms']:.6f}"
                )
        del inputs
        torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"csv={args.output}")


if __name__ == "__main__":
    main()
