#!/usr/bin/env python3
"""Isolated-process B300 screen of HMMA Phase-1 lookahead L2 versus L3."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys


ROOT = Path("<REMOTE_HOME>")
CASE = {
    "name": "h12_fixed8192_nonzero_bf16_state",
    "heads": 12,
    "tokens": 8192,
    "seed": 42808,
}
VARIANTS = ("official", "hmma_l2", "hmma_l3")
MODULES = {
    "official": (ROOT / "FlashKDA-c1-official", "flash_kda_C", 128),
    "hmma_l2": (
        ROOT / "kda-zero-state-20260905/clean_build/lib",
        "flash_kda_phase1_C",
        16,
    ),
}


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(variant: str, l3_build_lib: Path):
    if variant == "hmma_l3":
        directory, name, value_slice = l3_build_lib, "flash_kda_phase1_l3_C", 16
    else:
        directory, name, value_slice = MODULES[variant]
    directory = directory.resolve(strict=True)
    expected = sorted(directory.glob(f"{name}*.so"))
    if len(expected) != 1:
        raise RuntimeError(f"expected one {name}*.so in {directory}, got {expected}")
    sys.path.insert(0, str(directory))
    try:
        module = importlib.import_module(name)
    finally:
        sys.path.pop(0)
    actual = Path(module.__file__).resolve(strict=True)
    if actual != expected[0].resolve():
        raise RuntimeError(f"loaded {actual}, expected {expected[0]}")
    return module, actual, value_slice


def worker(args) -> None:
    import torch
    from flashinfer.testing import bench_gpu_time

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        raise RuntimeError("Round 8 requires a B300/SM103 device")
    module, binary, value_slice = load_module(args.variant, args.l3_build_lib)
    heads, tokens, seed = CASE["heads"], CASE["tokens"], CASE["seed"]
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def rand(shape, dtype=torch.bfloat16):
        return torch.randn(shape, generator=generator, device="cuda").to(dtype)

    shape = (1, tokens, heads, 128)
    q, k, v, g = [rand(shape) for _ in range(4)]
    beta = rand((1, tokens, heads))
    a_log = torch.rand(heads, generator=generator, device="cuda")
    dt_bias = torch.rand((heads, 128), generator=generator, device="cuda")
    initial = (rand((1, heads, 128, 128), torch.float32) * 0.25).to(torch.bfloat16)
    states = initial.unsqueeze(0).expand(args.state_rotations, *initial.shape).clone()
    output = torch.empty_like(q)
    final = torch.empty_like(initial)
    workspace = torch.empty(
        module.get_workspace_size(tokens, heads, 1), dtype=torch.uint8, device="cuda"
    )

    def invoke(state):
        kwargs = {
            "initial_state": state,
            "final_state": final,
            "cu_seqlens": None,
        }
        if args.variant != "official":
            kwargs["k2_value_slice"] = value_slice
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
            **kwargs,
        )
        state.copy_(final)

    probe_state = initial.clone()
    invoke(probe_state)
    torch.cuda.synchronize()
    actual = {"output": output.cpu(), "state": probe_state.cpu()}
    reference_path = args.output / "official_peer.pt"
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
    identity = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "variant": args.variant,
        "requested_module": module.__name__,
        "actual_module_file": str(binary),
        "binary_sha256": sha256(binary),
        "k2_value_slice": value_slice,
        "phase1_depth": {"official": 1, "hmma_l2": 2, "hmma_l3": 3}[args.variant],
        "phase6_state_prefetch": 1 if args.variant == "official" else 4,
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
    }
    result = {
        "schema_version": 1,
        "case": CASE,
        "variant": args.variant,
        "block": args.block,
        "scope": "preallocated public full forward plus same-stream state copy-back",
        "identity": identity,
        "correctness": correctness,
        "status": "correctness_failed_not_timed" if not passed else "measured",
        "samples_ms": [],
        "median_ms": None,
    }
    result_path = args.output / f"{args.variant}_{args.block}.json"
    if not passed:
        result_path.write_text(json.dumps(result, indent=2))
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
        {
            "samples_ms": samples,
            "median_ms": statistics.median(samples),
            "timed_calls": cursor,
        }
    )
    result_path.write_text(json.dumps(result, indent=2))


def controller(args) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for block in range(args.blocks):
        order = VARIANTS[block:] + VARIANTS[:block]
        if block == 0:
            order = ("official",) + tuple(item for item in order if item != "official")
        for variant in order:
            command = [
                sys.executable,
                __file__,
                "--worker",
                "--variant",
                variant,
                "--block",
                str(block),
                "--blocks",
                str(args.blocks),
                "--dry-run-iters",
                str(args.dry_run_iters),
                "--repeat-iters",
                str(args.repeat_iters),
                "--state-rotations",
                str(args.state_rotations),
                "--l3-build-lib",
                str(args.l3_build_lib),
                "--output",
                str(args.output),
            ]
            stem = f"{variant}_{block}"
            with (args.output / f"{stem}.log").open("w") as log:
                subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, check=True
                )
            row = json.loads((args.output / f"{stem}.json").read_text())
            rows.append(row)
            print(stem, row["status"], row["median_ms"], flush=True)

    medians = {}
    for variant in VARIANTS:
        values = [row["median_ms"] for row in rows if row["variant"] == variant]
        medians[variant] = (
            statistics.median(values)
            if len(values) == args.blocks and all(value is not None for value in values)
            else None
        )
    l2_over_l3 = (
        medians["hmma_l2"] / medians["hmma_l3"]
        if medians["hmma_l2"] is not None and medians["hmma_l3"] is not None
        else None
    )
    summary = {
        "schema_version": 1,
        "round": 8,
        "candidate": "HMMA V16 Phase-1 lookahead depth 3",
        "case": CASE,
        "scope": "preallocated public full forward plus same-stream state copy-back",
        "measurement": {
            "one_implementation_per_worker": True,
            "timing": "CUPTI cold-L2",
            "cuda_graph": False,
            "blocks": args.blocks,
            "dry_run_iters": args.dry_run_iters,
            "repeat_iters": args.repeat_iters,
            "balanced_cyclic_order": True,
            "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        },
        "correctness_contract": "output and updated state must be finite and bitwise equal to the same official peer",
        "median_ms": medians,
        "l2_over_l3_speedup": l2_over_l3,
        "practical_positive_threshold": 1.03,
        "screen_decision": (
            "advance_l3_to_qualification"
            if l2_over_l3 is not None and l2_over_l3 >= 1.03
            else "stop_l3_below_practical_gate"
            if l2_over_l3 is not None
            else "invalid_missing_measurement"
        ),
        "results": rows,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--dry-run-iters", type=int, default=20)
    parser.add_argument("--repeat-iters", type=int, default=100)
    parser.add_argument("--state-rotations", type=int, default=128)
    parser.add_argument("--l3-build-lib", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parsed = parser.parse_args()
    if parsed.worker:
        if parsed.variant is None:
            parser.error("--worker requires --variant")
        worker(parsed)
    else:
        controller(parsed)
