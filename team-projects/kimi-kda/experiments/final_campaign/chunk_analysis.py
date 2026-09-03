"""Quantify the FlashKDA CHUNK=16/32/64 design trade-offs.

The analytical part intentionally uses only the Python standard library.  It
models the exact ``ex2.approx.ftz.f32`` boundary used by FlashKDA, the upstream
workspace allocation formula, and the cost of naively extending the current
Neumann-series inverse.  Therefore it remains runnable on a login node without
CUDA, PyTorch, or FLA.

With ``--try-fla``, a small optional CUDA probe compares FLA chunk sizes 32 and
64 with a float64 recurrent reference and records CUDA-event latency.  Missing
or incompatible optional dependencies are recorded in the same CSV instead of
invalidating the offline analysis.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import struct
from pathlib import Path
from typing import Any


CHUNKS = (16, 32, 64)
LOG2E = math.log2(math.e)
F32_MIN_NORMAL = float.fromhex("0x1.000000p-126")
F32_MAX = float.fromhex("0x1.fffffep+127")
F32_MAX_LOG2 = math.log2(F32_MAX)

CSV_FIELDS = (
    "analysis_version",
    "chunk",
    "head_dim",
    "sequence_length",
    "heads",
    "lower_bound",
    "worst_cumsum_natural",
    "decay_log2_min",
    "restore_log2_max",
    "theoretical_decay_min",
    "theoretical_restore_max",
    "exp2_decay_min_ftz_f32",
    "exp2_restore_max_f32",
    "decay_zero_count_per_channel",
    "restore_inf_count_per_channel",
    "decay_zero_fraction",
    "restore_inf_fraction",
    "decay_zero_elements_per_tile",
    "restore_inf_elements_per_tile",
    "first_decay_zero_token",
    "first_restore_inf_token",
    "exp2_range_status",
    "workspace_formula",
    "workspace_bytes_per_tile",
    "tiles_per_head",
    "workspace_bytes_per_head",
    "workspace_mib_per_head",
    "workspace_bytes_all_heads",
    "workspace_mib_all_heads",
    "neumann_model",
    "neumann_power_levels",
    "neumann_dense_matmuls_per_chunk",
    "neumann_flops_per_chunk",
    "neumann_flops_per_sequence",
    "sm80_m16n8k16_per_chunk",
    "sm80_m16n8k16_per_sequence",
    "neumann_flops_per_chunk_vs_c16",
    "neumann_flops_per_sequence_vs_c16",
    "fla_status",
    "fla_error",
    "fla_version",
    "fla_module",
    "fla_device",
    "fla_sequence_length",
    "fla_heads",
    "fla_head_dim",
    "fla_warmup",
    "fla_iterations",
    "fla_output_max_abs",
    "fla_output_rmse",
    "fla_output_rel_rmse",
    "fla_state_max_abs",
    "fla_state_rmse",
    "fla_state_rel_rmse",
    "fla_output_all_finite",
    "fla_state_all_finite",
    "fla_mean_ms",
    "fla_median_ms",
    "fla_p90_ms",
    "fla_min_ms",
    "fla_max_ms",
)


def float32(value: float) -> float:
    """Round a finite Python float to IEEE-754 binary32."""

    return struct.unpack("=f", struct.pack("=f", value))[0]


def exp2_ftz_f32(log2_value: float) -> float:
    """Model ex2 in f32 with subnormal results flushed to zero."""

    if log2_value < -126.0:
        return 0.0
    if log2_value > F32_MAX_LOG2:
        return math.inf
    value = float32(2.0**log2_value)
    if abs(value) < F32_MIN_NORMAL:
        return 0.0
    return value


def safe_exp(value: float) -> float:
    try:
        return math.exp(value)
    except OverflowError:
        return math.inf


def first_index(values: list[float], predicate) -> int | str:
    for index, value in enumerate(values, start=1):
        if predicate(value):
            return index
    return ""


def analytical_row(
    chunk: int,
    head_dim: int,
    sequence_length: int,
    heads: int,
    lower_bound: float,
) -> dict[str, Any]:
    if chunk <= 0 or chunk & (chunk - 1):
        raise ValueError(f"chunk must be a positive power of two, got {chunk}")
    if head_dim <= 0 or sequence_length <= 0 or heads <= 0:
        raise ValueError("head_dim, sequence_length, and heads must be positive")
    if lower_bound >= 0:
        raise ValueError("lower_bound must be negative")

    # FlashKDA accumulates gate_scale = lower_bound * log2(e), then calls ex2.
    per_token_log2 = lower_bound * LOG2E
    decay_values = [exp2_ftz_f32(per_token_log2 * token) for token in range(1, chunk + 1)]
    restore_values = [exp2_ftz_f32(-per_token_log2 * token) for token in range(1, chunk + 1)]
    decay_zero_count = sum(value == 0.0 for value in decay_values)
    restore_inf_count = sum(math.isinf(value) for value in restore_values)

    tiles_per_head = math.ceil(sequence_length / chunk)
    # Three [C,D] bf16 tensors, one [D] fp32 tensor, and two [C,C] bf16 tensors.
    workspace_bytes_per_tile = 6 * chunk * head_dim + 4 * head_dim + 4 * chunk * chunk
    workspace_bytes_per_head = tiles_per_head * workspace_bytes_per_tile
    workspace_bytes_all_heads = heads * workspace_bytes_per_head

    # Current C16 code starts with I-L, then for L^(2,4,...,C/2) performs one
    # square and one INV update.  This is a deliberately naive dense extension.
    neumann_power_levels = int(math.log2(chunk)) - 1
    dense_matmuls = 2 * neumann_power_levels
    neumann_flops_per_chunk = dense_matmuls * 2 * chunk**3
    neumann_flops_per_sequence = tiles_per_head * neumann_flops_per_chunk

    # A full CxC-by-CxC product decomposed into SM80 m16n8k16 atoms.
    if chunk % 16 == 0:
        atoms_per_matmul = (chunk // 16) * (chunk // 8) * (chunk // 16)
        mma_per_chunk: int | str = dense_matmuls * atoms_per_matmul
        mma_per_sequence: int | str = tiles_per_head * int(mma_per_chunk)
    else:
        mma_per_chunk = ""
        mma_per_sequence = ""

    natural_cumsum = lower_bound * chunk
    decay_min_log2 = natural_cumsum * LOG2E
    restore_max_log2 = -decay_min_log2
    if not decay_zero_count and not restore_inf_count:
        range_status = "representable"
    elif decay_zero_count and restore_inf_count:
        range_status = "ftz_and_overflow"
    elif decay_zero_count:
        range_status = "ftz"
    else:
        range_status = "overflow"
    return {
        "analysis_version": "chunk-v1",
        "chunk": chunk,
        "head_dim": head_dim,
        "sequence_length": sequence_length,
        "heads": heads,
        "lower_bound": lower_bound,
        "worst_cumsum_natural": natural_cumsum,
        "decay_log2_min": decay_min_log2,
        "restore_log2_max": restore_max_log2,
        "theoretical_decay_min": safe_exp(natural_cumsum),
        "theoretical_restore_max": safe_exp(-natural_cumsum),
        "exp2_decay_min_ftz_f32": decay_values[-1],
        "exp2_restore_max_f32": restore_values[-1],
        "decay_zero_count_per_channel": decay_zero_count,
        "restore_inf_count_per_channel": restore_inf_count,
        "decay_zero_fraction": decay_zero_count / chunk,
        "restore_inf_fraction": restore_inf_count / chunk,
        "decay_zero_elements_per_tile": decay_zero_count * head_dim,
        "restore_inf_elements_per_tile": restore_inf_count * head_dim,
        "first_decay_zero_token": first_index(decay_values, lambda value: value == 0.0),
        "first_restore_inf_token": first_index(restore_values, math.isinf),
        "exp2_range_status": range_status,
        "workspace_formula": "6*C*D + 4*D + 4*C^2",
        "workspace_bytes_per_tile": workspace_bytes_per_tile,
        "tiles_per_head": tiles_per_head,
        "workspace_bytes_per_head": workspace_bytes_per_head,
        "workspace_mib_per_head": workspace_bytes_per_head / 2**20,
        "workspace_bytes_all_heads": workspace_bytes_all_heads,
        "workspace_mib_all_heads": workspace_bytes_all_heads / 2**20,
        "neumann_model": "naive dense extension of current C16 power series",
        "neumann_power_levels": neumann_power_levels,
        "neumann_dense_matmuls_per_chunk": dense_matmuls,
        "neumann_flops_per_chunk": neumann_flops_per_chunk,
        "neumann_flops_per_sequence": neumann_flops_per_sequence,
        "sm80_m16n8k16_per_chunk": mma_per_chunk,
        "sm80_m16n8k16_per_sequence": mma_per_sequence,
        "fla_status": "not_applicable_chunk16" if chunk == 16 else "not_requested",
        "fla_error": "",
    }


def add_relative_costs(rows: list[dict[str, Any]]) -> None:
    baseline = next(row for row in rows if row["chunk"] == 16)
    baseline_chunk = int(baseline["neumann_flops_per_chunk"])
    baseline_sequence = int(baseline["neumann_flops_per_sequence"])
    for row in rows:
        row["neumann_flops_per_chunk_vs_c16"] = int(row["neumann_flops_per_chunk"]) / baseline_chunk
        row["neumann_flops_per_sequence_vs_c16"] = int(row["neumann_flops_per_sequence"]) / baseline_sequence


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def error_metrics(actual, expected) -> dict[str, float | bool]:
    import torch

    actual64 = actual.detach().to(torch.float64)
    expected64 = expected.detach().to(torch.float64)
    difference = actual64 - expected64
    rmse = difference.square().mean().sqrt()
    reference_rmse = expected64.square().mean().sqrt()
    return {
        "max_abs": float(difference.abs().max()),
        "rmse": float(rmse),
        "rel_rmse": float(rmse / (reference_rmse + 1e-12)),
        "all_finite": bool(torch.isfinite(actual64).all()),
    }


def recurrent_reference(q, k, v, g, beta, initial_state, scale: float):
    """Small float64, token-by-token KDA reference using the exact input values."""

    import torch

    q64, k64, v64 = (tensor.to(torch.float64) for tensor in (q, k, v))
    g64, beta64 = g.to(torch.float64), beta.to(torch.float64)
    state = initial_state.to(torch.float64).clone()
    output = torch.empty_like(v64)
    for token in range(q.shape[1]):
        state = state * torch.exp(g64[:, token]).unsqueeze(-1)
        prediction = torch.einsum("bhk,bhkv->bhv", k64[:, token], state)
        update = (v64[:, token] - prediction) * beta64[:, token].unsqueeze(-1)
        state = state + torch.einsum("bhk,bhv->bhkv", k64[:, token], update)
        output[:, token] = torch.einsum("bhk,bhkv->bhv", q64[:, token] * scale, state)
    return output, state


def cuda_event_stats(run, warmup: int, iterations: int) -> dict[str, float]:
    import torch

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
    samples = [float(start.elapsed_time(end)) for start, end in zip(starts, ends)]
    ordered = sorted(samples)
    p90_index = min(len(ordered) - 1, math.ceil(0.90 * len(ordered)) - 1)
    return {
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "p90_ms": ordered[p90_index],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def optional_fla_probe(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    target_rows = {int(row["chunk"]): row for row in rows if int(row["chunk"]) in (32, 64)}
    for row in target_rows.values():
        row["fla_status"] = "optional_probe_not_completed"
        row["fla_sequence_length"] = args.fla_sequence_length
        row["fla_heads"] = args.fla_heads
        row["fla_head_dim"] = args.fla_head_dim
        row["fla_warmup"] = args.fla_warmup
        row["fla_iterations"] = args.fla_iterations
    write_csv(args.output, rows)

    try:
        # Prevent FLA from auto-dispatching back into FlashKDA: this probe is
        # specifically intended to exercise the Triton C32/C64 implementation.
        os.environ["FLA_FLASH_KDA"] = "0"
        import importlib.metadata

        import torch
        import torch.nn.functional as functional
        from fla.ops.kda import chunk_kda

        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is false")
        if args.fla_sequence_length <= 0 or args.fla_sequence_length % 64:
            raise ValueError("--fla-sequence-length must be a positive multiple of 64")
        if args.fla_head_dim <= 0 or args.fla_heads <= 0:
            raise ValueError("--fla-head-dim and --fla-heads must be positive")

        try:
            fla_version = importlib.metadata.version("flash-linear-attention")
        except importlib.metadata.PackageNotFoundError:
            fla_version = "unknown"
        import fla

        device = torch.device("cuda")
        torch.manual_seed(args.seed)
        shape = (1, args.fla_sequence_length, args.fla_heads, args.fla_head_dim)
        q = functional.normalize(torch.randn(shape, device=device, dtype=torch.float32), p=2, dim=-1).to(torch.bfloat16)
        k = functional.normalize(torch.randn(shape, device=device, dtype=torch.float32), p=2, dim=-1).to(torch.bfloat16)
        v = torch.randn(shape, device=device, dtype=torch.bfloat16)
        # Pre-activated, benign log gates isolate chunk algorithm correctness.
        g = (-0.01 - 0.09 * torch.rand(shape, device=device, dtype=torch.float32)).to(torch.bfloat16)
        beta = torch.sigmoid(
            torch.randn(shape[:-1], device=device, dtype=torch.float32)
        ).to(torch.bfloat16)
        initial_state = 0.01 * torch.randn(
            (1, args.fla_heads, args.fla_head_dim, args.fla_head_dim),
            device=device,
            dtype=torch.float32,
        )
        scale = 1.0 / math.sqrt(args.fla_head_dim)

        with torch.inference_mode():
            expected_output, expected_state = recurrent_reference(
                q, k, v, g, beta, initial_state, scale
            )

        common = {
            "q": q,
            "k": k,
            "v": v,
            "g": g,
            "beta": beta,
            "scale": scale,
            "initial_state": initial_state,
            "output_final_state": True,
            "use_qk_l2norm_in_kernel": False,
            "use_gate_in_kernel": False,
            "use_beta_sigmoid_in_kernel": False,
            "safe_gate": False,
        }
        device_label = f"{torch.cuda.get_device_name()} sm{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}"

        for chunk in (32, 64):
            row = target_rows[chunk]

            def run():
                return chunk_kda(chunk_size=chunk, **common)

            try:
                with torch.inference_mode():
                    actual_output, actual_state = run()
                    torch.cuda.synchronize()
                    output_error = error_metrics(actual_output, expected_output)
                    state_error = error_metrics(actual_state, expected_state)
                    latency = cuda_event_stats(run, args.fla_warmup, args.fla_iterations)

                row.update(
                    {
                        "fla_status": "ok",
                        "fla_error": "",
                        "fla_version": fla_version,
                        "fla_module": getattr(fla, "__file__", "unknown"),
                        "fla_device": device_label,
                        "fla_output_max_abs": output_error["max_abs"],
                        "fla_output_rmse": output_error["rmse"],
                        "fla_output_rel_rmse": output_error["rel_rmse"],
                        "fla_state_max_abs": state_error["max_abs"],
                        "fla_state_rmse": state_error["rmse"],
                        "fla_state_rel_rmse": state_error["rel_rmse"],
                        "fla_output_all_finite": output_error["all_finite"],
                        "fla_state_all_finite": state_error["all_finite"],
                        "fla_mean_ms": latency["mean_ms"],
                        "fla_median_ms": latency["median_ms"],
                        "fla_p90_ms": latency["p90_ms"],
                        "fla_min_ms": latency["min_ms"],
                        "fla_max_ms": latency["max_ms"],
                    }
                )
                print(
                    f"fla chunk={chunk} status=ok median_ms={latency['median_ms']:.6f} "
                    f"output_rel_rmse={output_error['rel_rmse']:.6e} "
                    f"state_rel_rmse={state_error['rel_rmse']:.6e}"
                )
                write_csv(args.output, rows)
            except Exception as error:  # Optional probe must not erase offline results.
                row["fla_status"] = "error"
                row["fla_error"] = f"{type(error).__name__}: {error}".replace("\n", " ")[:1000]
                print(f"fla chunk={chunk} status=error reason={row['fla_error']}")
                write_csv(args.output, rows)
    except Exception as error:
        reason = f"{type(error).__name__}: {error}".replace("\n", " ")[:1000]
        for row in target_rows.values():
            row["fla_status"] = "unavailable"
            row["fla_error"] = reason
        print(f"fla status=unavailable reason={reason}")
        write_csv(args.output, rows)


def validate(rows: list[dict[str, Any]]) -> None:
    by_chunk = {int(row["chunk"]): row for row in rows}
    if tuple(sorted(by_chunk)) != CHUNKS:
        raise AssertionError(f"expected rows for {CHUNKS}, got {tuple(sorted(by_chunk))}")
    expected_matmuls = {16: 6, 32: 8, 64: 10}
    for chunk in CHUNKS:
        head_dim = int(by_chunk[chunk]["head_dim"])
        expected_workspace = 6 * chunk * head_dim + 4 * head_dim + 4 * chunk * chunk
        if by_chunk[chunk]["workspace_bytes_per_tile"] != expected_workspace:
            raise AssertionError(f"unexpected workspace result for C{chunk}")
        if by_chunk[chunk]["neumann_dense_matmuls_per_chunk"] != expected_matmuls[chunk]:
            raise AssertionError(f"unexpected Neumann result for C{chunk}")

    # The assignment's canonical lower_bound=-5 case has a known boundary:
    # C16 is safe, while C32/C64 cross both FTZ and overflow at token 18.
    if all(math.isclose(float(row["lower_bound"]), -5.0) for row in rows):
        if by_chunk[16]["decay_zero_count_per_channel"] != 0:
            raise AssertionError("C16 unexpectedly crossed the f32 FTZ boundary")
        if by_chunk[16]["restore_inf_count_per_channel"] != 0:
            raise AssertionError("C16 unexpectedly crossed the f32 overflow boundary")
        for chunk in (32, 64):
            if by_chunk[chunk]["first_decay_zero_token"] != 18:
                raise AssertionError(f"C{chunk} should cross the f32 FTZ boundary at token 18")
            if by_chunk[chunk]["first_restore_inf_token"] != 18:
                raise AssertionError(f"C{chunk} should cross the f32 overflow boundary at token 18")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--lower-bound", type=float, default=-5.0)
    parser.add_argument("--try-fla", action="store_true")
    parser.add_argument("--fla-sequence-length", type=int, default=128)
    parser.add_argument("--fla-heads", type=int, default=1)
    parser.add_argument("--fla-head-dim", type=int, default=128)
    parser.add_argument("--fla-warmup", type=int, default=5)
    parser.add_argument("--fla-iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()
    if args.fla_warmup < 0 or args.fla_iterations <= 0:
        parser.error("--fla-warmup must be non-negative and --fla-iterations must be positive")
    return args


def main() -> None:
    args = parse_args()
    rows = [
        analytical_row(
            chunk=chunk,
            head_dim=args.head_dim,
            sequence_length=args.sequence_length,
            heads=args.heads,
            lower_bound=args.lower_bound,
        )
        for chunk in CHUNKS
    ]
    add_relative_costs(rows)
    validate(rows)
    write_csv(args.output, rows)
    for row in rows:
        print(
            f"offline chunk={row['chunk']} range={row['exp2_range_status']} "
            f"zero={row['decay_zero_count_per_channel']} inf={row['restore_inf_count_per_channel']} "
            f"workspace_per_head_mib={row['workspace_mib_per_head']:.6f} "
            f"neumann_sequence_vs_c16={row['neumann_flops_per_sequence_vs_c16']:.6f}"
        )
    print(f"offline_csv={args.output}")

    if args.try_fla:
        optional_fla_probe(rows, args)
    else:
        print("fla status=not_requested; pass --try-fla on a CUDA node to enable the optional probe")


if __name__ == "__main__":
    main()
