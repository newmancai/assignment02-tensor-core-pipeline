#!/usr/bin/env python3
"""CUPTI comparison with one extension loaded per worker process.

The controller alternates official/H3 worker processes inside one Slurm
allocation.  This avoids ELF/CUDA symbol interposition between the two
extensions while retaining paired block order and identical generated inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path


CASES = (
    ("h12_packed_512x32", 12, (512,) * 32, "packed", 12000),
    ("h12_packed_128x8", 12, (128,) * 8, "packed", 12001),
    ("h12_fixed_512", 12, (512,), "fixed", 12002),
    ("h12_fixed_8192", 12, (8192,), "fixed", 12003),
    ("h12_packed_mixed", 12, (1300, 547, 2048, 963, 271, 3063), "packed", 12004),
    ("h12_packed_1024x8", 12, (1024,) * 8, "packed", 12005),
    ("h96_fixed_8192", 96, (8192,), "fixed", 10000),
    ("h96_mixed", 96, (1300, 547, 2048, 963, 271, 3063), "packed", 10001),
    ("h96_uniform_1024x8", 96, (1024,) * 8, "packed", 10002),
)
VARIANTS = ("official_hmma", "h3_matched_hmma")
ALL_VARIANTS = VARIANTS + ("cake",)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def worker(args: argparse.Namespace) -> None:
    import numpy as np
    import torch
    from flashinfer.testing import bench_gpu_time

    name, heads, seq_lens, layout, seed = CASES[args.case_index]
    module = None
    files = []
    if args.variant != "cake":
        root = Path(args.source_dir).resolve(strict=True)
        files = sorted(root.glob(f"{args.module_name}*.so"))
        if len(files) != 1:
            raise RuntimeError(f"expected one {args.module_name}*.so, got {files}")
        sys.path.insert(0, str(root))
        module = importlib.import_module(args.module_name)
        if Path(module.__file__).resolve() != files[0].resolve():
            raise RuntimeError(f"extension resolved outside requested root: {module.__file__}")
    if torch.cuda.get_device_capability() != (10, 3):
        raise RuntimeError("B300/SM103 required")

    total = sum(seq_lens)
    sequences = len(seq_lens) if layout == "packed" else 1
    offsets = [0]
    for length in seq_lens:
        offsets.append(offsets[-1] + length)
    cu = torch.tensor(offsets, dtype=torch.int64, device="cuda") if layout == "packed" else None
    gen = torch.Generator(device="cuda").manual_seed(seed)
    shape = (1, total, heads, 128)
    rand = lambda shape, dtype=torch.bfloat16: torch.randn(shape, generator=gen, device="cuda").to(dtype)
    q, k, v, g = [rand(shape) for _ in range(4)]
    beta = rand((1, total, heads))
    a_log = torch.rand(heads, generator=gen, device="cuda")
    dt_bias = torch.rand((heads, 128), generator=gen, device="cuda")
    initial = (rand((sequences, heads, 128, 128), torch.float32) * 0.25).to(torch.bfloat16)
    states = initial.unsqueeze(0).expand(args.state_rotations, *initial.shape).clone()
    out = torch.empty_like(q)
    final = torch.empty_like(initial)
    if args.variant == "cake":
        from flashinfer.kda import recurrent_kda
        workspace = None
    else:
        size_fn = getattr(module, "get_workspace_size_h3", None) or module.get_workspace_size
        run_fn = getattr(module, "fwd_h3", None) or module.fwd
        workspace = torch.empty(int(size_fn(total, heads, sequences)), dtype=torch.uint8, device="cuda")
    cursor = 0

    def run() -> None:
        nonlocal cursor
        if cursor >= args.state_rotations:
            raise RuntimeError(f"state rotations exhausted: {cursor}")
        state = states[cursor]
        cursor += 1
        if args.variant == "cake":
            recurrent_kda(q=q, k=k, v=v, g=g, beta=beta, A_log=a_log,
                dt_bias=dt_bias, scale=1 / math.sqrt(128), initial_state=state,
                output=out, output_final_state=False, use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True, lower_bound=-5.0, cu_seqlens=cu,
                beta_is_logit=True, backend="cake")
        else:
            run_fn(q, k, v, g, beta, 1 / math.sqrt(128), out, workspace,
                   a_log, dt_bias, -5.0, initial_state=state,
                   final_state=final, cu_seqlens=cu)
            state.copy_(final)

    samples = [float(x) for x in bench_gpu_time(
        run, enable_cupti=True, cold_l2_cache=True, use_cuda_graph=False,
        dry_run_iters=args.dry_run_iters, repeat_iters=args.repeat_iters)]
    payload = {
        "case": name,
        "variant": args.variant,
        "median_ms": float(np.median(samples)),
        "samples_ms": samples,
        "calls_consumed": cursor,
        "module_path": str(files[0]) if files else str(Path(importlib.import_module("flashinfer.kda").__file__).resolve()),
        "binary_sha256": sha256(files[0]) if files else None,
        "workspace_bytes": workspace.numel() if workspace is not None else None,
        "device": torch.cuda.get_device_name(),
    }
    print("ISOLATED_RESULT=" + json.dumps(payload, sort_keys=True))


def controller(args: argparse.Namespace) -> None:
    import numpy as np

    roots = {
        "official_hmma": (args.official_source_dir, "flash_kda_C"),
        "h3_matched_hmma": (args.h3_source_dir, "flash_kda_h3_C"),
        "cake": (Path.cwd(), "unused"),
    }
    variants = ("official_hmma", "cake") if args.comparison == "cake" else VARIANTS
    cases = []
    for case_index, (name, heads, seq_lens, layout, seed) in enumerate(CASES):
        blocks = {variant: [] for variant in variants}
        raw = {variant: [] for variant in variants}
        identities = {}
        for block in range(args.blocks):
            order = variants[block % 2:] + variants[:block % 2]
            for variant in order:
                root, module = roots[variant]
                cmd = [sys.executable, str(Path(__file__).resolve()), "--worker",
                       "--variant", variant, "--source-dir", str(root),
                       "--module-name", module, "--case-index", str(case_index),
                       "--state-rotations", str(args.state_rotations),
                       "--dry-run-iters", str(args.dry_run_iters),
                       "--repeat-iters", str(args.repeat_iters)]
                output = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
                line = next(x for x in reversed(output.splitlines()) if x.startswith("ISOLATED_RESULT="))
                result = json.loads(line.split("=", 1)[1])
                blocks[variant].append(result["median_ms"])
                raw[variant].extend(result["samples_ms"])
                identities[variant] = {k: result[k] for k in
                    ("module_path", "binary_sha256", "workspace_bytes")}
        official = np.asarray(blocks["official_hmma"])
        candidate_id = variants[1]
        candidate = np.asarray(blocks[candidate_id])
        speedup = float(np.median(official) / np.median(candidate))
        rng = np.random.default_rng(seed)
        draws = []
        for _ in range(10000):
            ix = rng.integers(0, len(official), len(official))
            draws.append(float(np.median(official[ix]) / np.median(candidate[ix])))
        cases.append({
            "name": name, "num_heads": heads, "seq_lens": list(seq_lens),
            "layout": layout, "seed": seed,
            "official_median_ms": float(np.median(official)),
            "candidate": candidate_id,
            "candidate_median_ms": float(np.median(candidate)),
            "speedup_official_over_candidate": speedup,
            "bootstrap_ci95": [float(x) for x in np.quantile(draws, (0.025, 0.975))],
            "block_order": [list(variants[i % 2:] + variants[:i % 2]) for i in range(args.blocks)],
            "block_medians_ms": blocks, "samples_ms": raw,
            "identities": identities,
        })
        print(name, f"official={np.median(official):.6f} ms",
              f"{candidate_id}={np.median(candidate):.6f} ms", f"speedup={speedup:.4f}x", flush=True)
    h12 = [x["speedup_official_over_candidate"] for x in cases if x["num_heads"] == 12]
    h96 = [x["speedup_official_over_candidate"] for x in cases if x["num_heads"] == 96]
    payload = {
        "schema_version": "isolated_process_cupti_v2",
        "comparison": args.comparison,
        "experiment_id": os.getenv("SLURM_JOB_ID", "interactive"),
        "reason_for_isolation": "avoid cross-extension ELF/CUDA symbol interposition",
        "timing_policy": {"backend": "cupti", "cold_l2": True,
            "one_extension_per_worker_process": True, "balanced_block_order": True,
            "dry_run_iters": args.dry_run_iters, "repeat_iters": args.repeat_iters,
            "blocks": args.blocks, "public_state_copy_back": True},
        "correctness_evidence": "evidence/h3_gpu_correctness.json" if args.comparison == "h3" else "evidence/b300_stage3_public_paired.json and stage12 peer checks",
        "cases": cases,
        "aggregate": {
            "h12_geomean_speedup": float(math.prod(h12) ** (1 / len(h12))),
            "h96_geomean_speedup": float(math.prod(h96) ** (1 / len(h96))),
            "decision": ("stop_h3" if math.prod(h12) ** (1 / len(h12)) <= 1.05 or min(h12) < 0.98 else "advance") if args.comparison == "h3" else "measured_gain",
        },
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["aggregate"], indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true")
    p.add_argument("--variant", choices=ALL_VARIANTS)
    p.add_argument("--source-dir", type=Path)
    p.add_argument("--module-name")
    p.add_argument("--case-index", type=int)
    p.add_argument("--official-source-dir", type=Path)
    p.add_argument("--h3-source-dir", type=Path)
    p.add_argument("--comparison", choices=("h3", "cake"), default="h3")
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--state-rotations", type=int, default=128)
    p.add_argument("--dry-run-iters", type=int, default=20)
    p.add_argument("--repeat-iters", type=int, default=100)
    p.add_argument("--json", type=Path)
    args = p.parse_args()
    if args.worker:
        worker(args)
    else:
        if args.official_source_dir is None or args.json is None or (args.comparison == "h3" and args.h3_source_dir is None):
            p.error("controller requires source directories and --json")
        controller(args)


if __name__ == "__main__":
    main()
