#!/usr/bin/env python3
"""GPU correctness gate for an isolated ``flash_kda_h3_C`` extension.

This runner compares the H3 CUDA implementation with the official checkout's
``tests/torch_ref.py`` on identical inputs.  H3 changes BF16 rounding points,
so acceptance is algorithm-equivalent, not bitwise compatibility.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import torch
import torch.nn.functional as F


D = 128
LOWER_BOUND = -5.0
ATOL = 0.015625
RTOL = 0.03
RELATIVE_L2_LIMIT = 0.015
NORMALIZED_LINF_LIMIT = 0.05


@dataclass(frozen=True)
class Case:
    name: str
    lengths: tuple[int, ...]
    initial_state: bool
    state_dtype: torch.dtype

    @property
    def packed(self) -> bool:
        return len(self.lengths) > 1


CASES = (
    Case("fixed_c16_zero", (16,), False, torch.bfloat16),
    Case("fixed_tail17_nonzero", (17,), True, torch.bfloat16),
    Case("fixed_multichunk37_nonzero_fp32", (37,), True, torch.float32),
    Case("packed_4_17_33_nonzero", (4, 17, 33), True, torch.bfloat16),
)


def load_torch_ref(official_source_dir: Path):
    path = official_source_dir / "tests" / "torch_ref.py"
    if not path.is_file():
        raise FileNotFoundError(f"official torch reference not found: {path}")
    # torch_ref imports CUDA helper modules relative to its tests directory.
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("flashkda_official_torch_ref_h3_gate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.torch_ref, path.resolve()


def load_h3_extension(h3_source_dir: Path) -> ModuleType:
    sys.path.insert(0, str(h3_source_dir.resolve()))
    module = importlib.import_module("flash_kda_h3_C")
    module_path = Path(module.__file__).resolve()
    try:
        module_path.relative_to(h3_source_dir.resolve())
    except ValueError as exc:
        raise RuntimeError(
            f"flash_kda_h3_C resolved outside the requested checkout: {module_path}"
        ) from exc
    return module


def make_inputs(case: Case, seed: int, heads: int):
    torch.manual_seed(seed)
    if case.packed:
        batch, tokens = 1, sum(case.lengths)
        cu_seqlens = torch.tensor(
            [0, *torch.tensor(case.lengths).cumsum(0).tolist()],
            dtype=torch.int64,
            device="cuda",
        )
        states = len(case.lengths)
    else:
        batch, tokens = 1, case.lengths[0]
        cu_seqlens = None
        states = batch
    shape = (batch, tokens, heads, D)
    q = F.normalize(torch.randn(shape, device="cuda", dtype=torch.float32), dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn(shape, device="cuda", dtype=torch.float32), dim=-1).to(torch.bfloat16)
    v = (0.25 * torch.randn(shape, device="cuda", dtype=torch.float32)).to(torch.bfloat16)
    g = torch.randn(shape, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    beta = torch.randn((batch, tokens, heads), device="cuda", dtype=torch.float32).to(torch.bfloat16)
    a_log = torch.rand(heads, device="cuda", dtype=torch.float32) - 0.25
    dt_bias = torch.rand((heads, D), device="cuda", dtype=torch.float32) - 0.5
    initial = None
    if case.initial_state:
        initial = (0.08 * torch.randn((states, heads, D, D), device="cuda", dtype=torch.float32))
        initial = initial.to(torch.bfloat16).to(case.state_dtype)
    return q, k, v, g, beta, a_log, dt_bias, initial, cu_seqlens


def call_h3(
    module: ModuleType,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
):
    batch, tokens, heads, _ = q.shape
    sequences = int(cu_seqlens.numel() - 1) if cu_seqlens is not None else batch
    size_fn = getattr(module, "get_workspace_size_h3", None) or getattr(module, "get_workspace_size", None)
    run_fn = getattr(module, "fwd_h3", None) or getattr(module, "fwd", None)
    if size_fn is None or run_fn is None:
        raise AttributeError(
            "flash_kda_h3_C must export get_workspace_size_h3/get_workspace_size "
            "and fwd_h3/fwd"
        )
    workspace = torch.empty(size_fn(batch * tokens, heads, sequences), dtype=torch.uint8, device="cuda")
    out = torch.empty_like(q)
    final_dtype = initial.dtype if initial is not None else torch.bfloat16
    final = torch.empty((sequences, heads, D, D), dtype=final_dtype, device="cuda")
    run_fn(
        q, k, v, g, beta, float(D**-0.5), out, workspace, a_log, dt_bias, LOWER_BOUND,
        initial_state=initial, final_state=final, cu_seqlens=cu_seqlens,
    )
    return out, final, workspace.numel()


def call_reference(
    torch_ref,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
):
    sequences = int(cu_seqlens.numel() - 1) if cu_seqlens is not None else q.shape[0]
    out = torch.empty_like(q)
    final_dtype = initial.dtype if initial is not None else torch.bfloat16
    final = torch.empty((sequences, q.shape[2], D, D), dtype=final_dtype, device="cuda")
    torch_ref(
        q, k, v, g, beta, float(D**-0.5), out, a_log, dt_bias, LOWER_BOUND,
        initial_state=initial, final_state=final, cu_seqlens=cu_seqlens,
    )
    return out, final


def compare(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float | bool]:
    actual64, reference64 = actual.double(), reference.double()
    diff = actual64 - reference64
    ref_l2 = torch.linalg.vector_norm(reference64)
    ref_linf = reference64.abs().max()
    relative_l2 = torch.linalg.vector_norm(diff) / torch.clamp(ref_l2, min=1e-30)
    normalized_linf = diff.abs().max() / torch.clamp(ref_linf, min=1e-30)
    close = torch.isclose(actual, reference, atol=ATOL, rtol=RTOL)
    finite = torch.isfinite(actual).all() & torch.isfinite(reference).all()
    passed = finite & close.all() & (relative_l2 <= RELATIVE_L2_LIMIT) & (normalized_linf <= NORMALIZED_LINF_LIMIT)
    return {
        "passed": bool(passed.item()),
        "finite": bool(finite.item()),
        "allclose_fraction": float(close.float().mean().item()),
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "relative_l2": float(relative_l2.item()),
        "normalized_linf": float(normalized_linf.item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-source-dir", type=Path, required=True)
    parser.add_argument("--h3-source-dir", type=Path, required=True)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required")
    h3 = load_h3_extension(args.h3_source_dir)
    torch_ref, reference_path = load_torch_ref(args.official_source_dir)

    rows = []
    for case_index, case in enumerate(CASES):
        for seed in range(args.seeds):
            tensors = make_inputs(case, 20260910 + 100 * case_index + seed, args.heads)
            q, k, v, g, beta, a_log, dt_bias, initial, cu_seqlens = tensors
            h3_out, h3_final, workspace_bytes = call_h3(
                h3, q, k, v, g, beta, a_log, dt_bias,
                initial.clone() if initial is not None else None, cu_seqlens,
            )
            ref_out, ref_final = call_reference(
                torch_ref, q, k, v, g, beta, a_log, dt_bias,
                initial.clone() if initial is not None else None, cu_seqlens,
            )
            torch.cuda.synchronize()
            rows.append({
                "case": case.name,
                "seed": seed,
                "lengths": case.lengths,
                "packed": case.packed,
                "nonzero_initial_state": case.initial_state,
                "state_dtype": str(case.state_dtype),
                "workspace_bytes": workspace_bytes,
                "output": compare(h3_out, ref_out),
                "final_state": compare(h3_final, ref_final),
            })

    failures = [
        {"case": row["case"], "seed": row["seed"], "target": target}
        for row in rows
        for target in ("output", "final_state")
        if not row[target]["passed"]
    ]
    payload = {
        "schema_version": "sm100_open_h3_gpu_correctness_v1",
        "scope": "gpu_correctness_not_performance",
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "h3_module": str(Path(h3.__file__).resolve()),
        "official_torch_ref": str(reference_path),
        "contract": {
            "compatibility": "algorithm_equivalent_not_official_bitwise",
            "atol": ATOL,
            "rtol": RTOL,
            "relative_l2_limit": RELATIVE_L2_LIMIT,
            "normalized_linf_limit": NORMALIZED_LINF_LIMIT,
            "required": "finite and all elements isclose and both norm limits, for output and final_state",
        },
        "case_count": len(rows),
        "passed": not failures,
        "failures": failures,
        "cases": rows,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    print(encoded)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(encoded + "\n")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
