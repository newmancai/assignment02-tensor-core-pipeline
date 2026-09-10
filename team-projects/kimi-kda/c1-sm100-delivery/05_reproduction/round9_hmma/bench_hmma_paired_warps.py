#!/usr/bin/env python3
"""B300 public-full comparison of baseline and paired-warps HMMA K2."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys


ROOT = Path("<REMOTE_HOME>")
CASES = {
    "balanced3_v32": (12, (2731, 2731, 2730), 32),
    "balanced6_v64": (12, (1366, 1366, 1365, 1365, 1365, 1365), 64),
}
VARIANTS = ("official", "baseline", "paired")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(variant: str, paired_build_lib: Path):
    if variant == "official":
        directory, name = ROOT / "FlashKDA-c1-official", "flash_kda_C"
    elif variant == "baseline":
        directory, name = (
            ROOT / "kda-zero-state-20260905/clean_build/lib",
            "flash_kda_phase1_C",
        )
    else:
        directory, name = paired_build_lib, "flash_kda_paired_warps_C"
    directory = directory.resolve(strict=True)
    expected = sorted(directory.glob(f"{name}*.so"))
    if len(expected) != 1:
        raise RuntimeError(f"expected one {name} binary in {directory}, got {expected}")
    sys.path.insert(0, str(directory))
    try:
        module = importlib.import_module(name)
    finally:
        sys.path.pop(0)
    actual = Path(module.__file__).resolve(strict=True)
    if actual != expected[0].resolve():
        raise RuntimeError(f"loaded {actual}, expected {expected[0]}")
    return module, actual


def worker(args) -> None:
    import torch
    from flashinfer.testing import bench_gpu_time

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        raise RuntimeError("Round 9 requires B300/SM103")
    heads, lengths, value_slice = CASES[args.case]
    total = sum(lengths)
    sequences = len(lengths)
    seed = 42900 + list(CASES).index(args.case)
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def rand(shape, dtype=torch.bfloat16):
        return torch.randn(shape, generator=generator, device="cuda").to(dtype)

    shape = (1, total, heads, 128)
    q, k, v, g = [rand(shape) for _ in range(4)]
    beta = rand((1, total, heads))
    a_log = torch.rand(heads, generator=generator, device="cuda")
    dt_bias = torch.rand((heads, 128), generator=generator, device="cuda")
    initial = (rand((sequences, heads, 128, 128), torch.float32) * 0.25).to(torch.bfloat16)
    states = initial.unsqueeze(0).expand(args.state_rotations, *initial.shape).clone()
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    cu_seqlens = torch.tensor(offsets, dtype=torch.int64, device="cuda")
    output = torch.empty_like(q)
    final = torch.empty_like(initial)
    module, binary = load_module(args.variant, args.paired_build_lib)
    workspace = torch.empty(
        module.get_workspace_size(total, heads, sequences),
        dtype=torch.uint8,
        device="cuda",
    )
    extra = {} if args.variant == "official" else {"k2_value_slice": value_slice}

    def invoke(state):
        module.fwd(
            q,
            k,
            v,
            g,
            beta,
            1 / math.sqrt(128),
            output,
            workspace,
            a_log,
            dt_bias,
            -5.0,
            initial_state=state,
            final_state=final,
            cu_seqlens=cu_seqlens,
            **extra,
        )
        state.copy_(final)

    probe_state = initial.clone()
    invoke(probe_state)
    torch.cuda.synchronize()
    actual = {"output": output.cpu(), "state": probe_state.cpu()}
    reference_path = args.output / f"{args.case}_official.pt"
    if args.variant == "official" and args.block == 0:
        torch.save(actual, reference_path)
    reference = torch.load(reference_path, weights_only=True)
    correctness = {}
    for key, value in actual.items():
        peer = reference[key]
        delta = value.float() - peer.float()
        correctness[key] = {
            "finite": bool(torch.isfinite(value).all()),
            "bitwise": torch.equal(value, peer),
            "max_abs": float(delta.abs().max()),
        }
    passed = all(item["finite"] and item["bitwise"] for item in correctness.values())
    compute_warps = (
        None
        if args.variant == "official"
        else min(value_slice // 16, 4)
        if args.variant == "baseline"
        else max(1, value_slice // 32)
    )
    result = {
        "schema_version": 1,
        "case": args.case,
        "lengths": lengths,
        "variant": args.variant,
        "block": args.block,
        "scope": "preallocated public-full forward plus same-stream state copy-back",
        "identity": {
            "module": module.__name__,
            "binary": str(binary),
            "binary_sha256": sha256(binary),
            "value_slice": 128 if args.variant == "official" else value_slice,
            "compute_warps": compute_warps,
            "device": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
        },
        "correctness": correctness,
        "status": "correctness_failed_not_timed" if not passed else "measured",
        "samples_ms": [],
        "median_ms": None,
    }
    path = args.output / f"{args.case}_{args.variant}_{args.block}.json"
    if not passed:
        path.write_text(json.dumps(result, indent=2) + "\n")
        return

    cursor = 0

    def run():
        nonlocal cursor
        if cursor >= len(states):
            raise RuntimeError("state rotation exhausted")
        state = states[cursor]
        cursor += 1
        invoke(state)

    samples = [
        float(value)
        for value in bench_gpu_time(
            run,
            enable_cupti=True,
            cold_l2_cache=True,
            use_cuda_graph=False,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.repeat_iters,
        )
    ]
    result.update(
        {"samples_ms": samples, "median_ms": statistics.median(samples), "calls": cursor}
    )
    path.write_text(json.dumps(result, indent=2) + "\n")


def controller(args) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for case_name in args.case or CASES:
        for block in range(args.blocks):
            order = VARIANTS[block:] + VARIANTS[:block]
            if block == 0:
                order = ("official", "baseline", "paired")
            for variant in order:
                command = [
                    sys.executable,
                    __file__,
                    "--worker",
                    "--case",
                    case_name,
                    "--variant",
                    variant,
                    "--block",
                    str(block),
                    "--dry-run-iters",
                    str(args.dry_run_iters),
                    "--repeat-iters",
                    str(args.repeat_iters),
                    "--state-rotations",
                    str(args.state_rotations),
                    "--paired-build-lib",
                    str(args.paired_build_lib),
                    "--output",
                    str(args.output),
                ]
                stem = f"{case_name}_{variant}_{block}"
                with (args.output / f"{stem}.log").open("w") as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                row = json.loads((args.output / f"{stem}.json").read_text())
                rows.append(row)
                print(stem, row["status"], row["median_ms"], flush=True)

    summary = []
    for case_name in args.case or CASES:
        medians = {
            variant: statistics.median(
                row["median_ms"]
                for row in rows
                if row["case"] == case_name and row["variant"] == variant
            )
            for variant in VARIANTS
        }
        summary.append(
            {
                "case": case_name,
                "median_ms": medians,
                "paired_over_baseline": medians["baseline"] / medians["paired"],
                "paired_over_official": medians["official"] / medians["paired"],
            }
        )
    payload = {
        "schema_version": 1,
        "round": 9,
        "candidate": "HMMA two 16-column blocks per compute warp",
        "scope": "preallocated public-full forward plus same-stream state copy-back",
        "correctness_contract": "output and final state finite and bitwise equal to official",
        "measurement": {
            "timing": "CUPTI cold-L2",
            "blocks": args.blocks,
            "dry_run_iters": args.dry_run_iters,
            "repeat_iters": args.repeat_iters,
            "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        },
        "summary": summary,
        "results": rows,
    }
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--case", action="append", choices=tuple(CASES))
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--dry-run-iters", type=int, default=12)
    parser.add_argument("--repeat-iters", type=int, default=40)
    parser.add_argument("--state-rotations", type=int, default=64)
    parser.add_argument("--paired-build-lib", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parsed = parser.parse_args()
    if parsed.worker:
        if len(parsed.case or ()) != 1 or parsed.variant is None:
            parser.error("worker requires exactly one --case and --variant")
        parsed.case = parsed.case[0]
        worker(parsed)
    else:
        controller(parsed)
