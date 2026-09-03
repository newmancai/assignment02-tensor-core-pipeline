"""Launch exactly one FlashKDA K2 recurrence inside an NCU capture range.

The surrounding sbatch script selects either the untouched official extension or
the integrated ValueSlice extension through PYTHONPATH.  Keeping the tensor
construction and warmup outside the capture range makes the two reports directly
comparable and avoids profiling allocator/prepare noise.
"""

from __future__ import annotations

import argparse
import json
import math
import os

import torch


DIM = 128


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()

    import flash_kda  # Imported after the sbatch script selects PYTHONPATH.

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
        )

    for _ in range(args.warmup):
        run_once()
    torch.cuda.synchronize()

    module_path = getattr(flash_kda, "__file__", "unknown")
    metadata = {
        "label": args.label,
        "shape": [1, args.tokens, args.heads, DIM],
        "warmup": args.warmup,
        "seed": args.seed,
        "flash_kda_module": module_path,
        "value_slice_env": os.environ.get("FLASH_KDA_K2_VALUE_SLICE"),
        "device": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
    }
    print("PROFILE_METADATA=" + json.dumps(metadata, sort_keys=True), flush=True)

    # NCU is configured with --capture-range cudaProfilerApi and a kernel-name
    # filter, so only the single K2 recurrence below is collected.
    torch.cuda.profiler.start()
    run_once()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    print("PROFILE_PASS", flush=True)


if __name__ == "__main__":
    main()
