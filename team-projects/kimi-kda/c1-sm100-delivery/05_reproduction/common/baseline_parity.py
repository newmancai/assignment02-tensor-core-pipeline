"""Check untouched upstream FlashKDA against the patched V128 compatibility path."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F


DIM = 128


def make_inputs(lengths: list[int], heads: int, state_dtype: torch.dtype, seed: int, packed: bool):
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
    initial_state = torch.randn((nseq, heads, DIM, DIM), device="cuda", dtype=state_dtype)
    out = torch.empty_like(v)
    final_state = torch.empty_like(initial_state)
    cu_seqlens = None
    if packed:
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        cu_seqlens = torch.tensor(offsets, device="cuda", dtype=torch.int64)
    return (q, k, v, g, beta, a_log, dt_bias, initial_state, out, final_state, cu_seqlens)


def make_runner(flash_kda, inputs):
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
            -5.0,
            initial_state,
            final_state,
            cu_seqlens,
        )

    return run


def event_samples(run, warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for index in range(iterations):
        starts[index].record()
        run()
        ends[index].record()
    torch.cuda.synchronize()
    return [float(start.elapsed_time(end)) for start, end in zip(starts, ends)]


@torch.inference_mode()
def run_runtime(args) -> None:
    import flash_kda
    import flash_kda_C

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"label={args.label}")
    print(f"package={flash_kda.__file__}")
    print(f"extension={flash_kda_C.__file__}")
    cases = (
        ("fixed_t513_bf16", [513], 3, torch.bfloat16, False, False),
        ("fixed_t513_fp32", [513], 3, torch.float32, False, False),
        ("ragged_small_bf16", [31, 47, 19], 3, torch.bfloat16, True, False),
        ("fixed_h12_t8192", [8192], 12, torch.bfloat16, False, True),
        ("packed_h12_t8192", [8192], 12, torch.bfloat16, True, True),
    )
    timing_rows = []
    for index, (name, lengths, heads, dtype, packed, timed) in enumerate(cases):
        inputs = make_inputs(lengths, heads, dtype, 2026090300 + index, packed)
        run = make_runner(flash_kda, inputs)
        run()
        torch.cuda.synchronize()
        torch.save(
            {"out": inputs[8].cpu(), "final_state": inputs[9].cpu()},
            args.output / f"{args.label}_{name}.pt",
        )
        if timed:
            all_samples = []
            for _ in range(args.repeats):
                all_samples.extend(event_samples(run, args.warmup, args.iterations))
            ordered = sorted(all_samples)
            timing_rows.append(
                {
                    "label": args.label,
                    "case": name,
                    "mean_ms": statistics.fmean(all_samples),
                    "median_ms": statistics.median(all_samples),
                    "p90_ms": ordered[math.ceil(0.9 * len(ordered)) - 1],
                    "min_ms": ordered[0],
                    "max_ms": ordered[-1],
                    "samples": len(all_samples),
                }
            )
    (args.output / f"{args.label}_timing.json").write_text(json.dumps(timing_rows, indent=2) + "\n")


def compare(args) -> None:
    rows = []
    official_files = sorted(args.output.glob("official_*.pt"))
    if not official_files:
        raise RuntimeError(f"no official tensors found in {args.output}")
    for official_path in official_files:
        suffix = official_path.name.removeprefix("official_")
        patched_path = args.output / f"patched_v128_{suffix}"
        official = torch.load(official_path, map_location="cpu")
        patched = torch.load(patched_path, map_location="cpu")
        for tensor_name in ("out", "final_state"):
            lhs = official[tensor_name]
            rhs = patched[tensor_name]
            delta = (lhs.float() - rhs.float()).abs()
            row = {
                "case": suffix.removesuffix(".pt"),
                "tensor": tensor_name,
                "equal": torch.equal(lhs, rhs),
                "max_abs": float(delta.max()),
                "rmse": float(torch.sqrt(torch.mean(delta.square()))),
            }
            rows.append(row)
            print(row)
    (args.output / "baseline_parity.json").write_text(json.dumps(rows, indent=2) + "\n")
    if not all(row["equal"] for row in rows):
        raise AssertionError("patched V128 is not bitwise equal to untouched upstream")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--label", choices=("official", "patched_v128"), required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--warmup", type=int, default=20)
    run_parser.add_argument("--iterations", type=int, default=200)
    run_parser.add_argument("--repeats", type=int, default=3)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        run_runtime(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
