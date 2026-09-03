"""Validate patched FlashKDA against the pinned FLA KDA references.

This campaign intentionally separates two questions:

1. ValueSlice invariance: V16/V32/V64 must be bitwise equal to V128.
2. Numerical accuracy: FlashKDA and FLA ``chunk_kda`` are compared with the
   FP32-state recurrence in ``fla.ops.kda.naive`` using error metrics.

FlashKDA fuses q/k normalization, the lower-bound gate, and beta sigmoid.  The
naive reference does not, so those transformations are reproduced explicitly
in FP32.  FlashKDA also exposes state in V-first [N, H, V, K] layout while the
naive reference uses [B, H, K, V]; every naive state crossing is transposed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# This must be set before importing fla.ops.kda.chunk.  Otherwise its backend
# dispatcher may route the supposed Triton reference back into FlashKDA.
os.environ["FLA_FLASH_KDA"] = "0"

import torch

import flash_kda_C
from fla.ops.kda.chunk import chunk_kda
from fla.ops.kda.naive import naive_recurrent_kda


DIM = 128
LOWER_BOUND = -5.0
SCALE = 1.0 / math.sqrt(DIM)
VALUE_SLICES = (16, 32, 64, 128)
STATE_MODES = ("in_out", "in_only", "out_only", "none")
STATE_DTYPES = (torch.bfloat16, torch.float32)

# SHA-256 values of the assignment-provided fla_kda_ref snapshot at a3edffc.
EXPECTED_REFERENCE_HASHES = {
    "naive": "60a32285d4b67068ff633b48bbe8ab31028066d24f00d27e12199a88fc73f016",
    "chunk": "a15aa6ac257af5c8f7a1d6afc21a417391568e90a8898d4e228ddeb48cb0e9b8",
}


@dataclass(frozen=True)
class ReferenceCase:
    name: str
    lengths: tuple[int, ...]
    heads: int
    seed: int
    gate_regime: str
    state_mode: str = "in_out"
    state_dtypes: tuple[torch.dtype, ...] = STATE_DTYPES
    use_naive: bool = True
    use_chunk: bool = True

    @property
    def packed(self) -> bool:
        return len(self.lengths) > 1

    @property
    def smoke(self) -> bool:
        return sum(self.lengths) <= 512


@dataclass
class Inputs:
    lengths: tuple[int, ...]
    packed: bool
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    g: torch.Tensor
    beta: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    state_bf16_vk: torch.Tensor
    cu_seqlens: torch.Tensor | None

    @property
    def total_tokens(self) -> int:
        return sum(self.lengths)

    @property
    def nseq(self) -> int:
        return len(self.lengths) if self.packed else 1

    @property
    def heads(self) -> int:
        return self.q.shape[2]


def dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float32:
        return "fp32"
    return str(dtype).removeprefix("torch.")


def has_state_in(state_mode: str) -> bool:
    return state_mode in ("in_out", "in_only")


def has_state_out(state_mode: str) -> bool:
    return state_mode in ("in_out", "out_only")


def offsets_from_lengths(lengths: Iterable[int]) -> list[int]:
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + int(length))
    return offsets


def make_inputs(case: ReferenceCase) -> Inputs:
    torch.manual_seed(case.seed)
    total = sum(case.lengths)
    shape = (1, total, case.heads, DIM)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    beta = torch.randn(shape[:-1], device="cuda", dtype=torch.bfloat16)

    if case.gate_regime == "random":
        g = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        a_log = torch.empty(case.heads, device="cuda", dtype=torch.float32).uniform_(-1.0, 1.0)
        dt_bias = torch.randn((case.heads, DIM), device="cuda", dtype=torch.float32) * 0.5
    elif case.gate_regime == "long_memory":
        # With A_log=0 and zero bias, raw g=-8 produces a decay very close to 0.
        g = torch.full(shape, -8.0, device="cuda", dtype=torch.bfloat16)
        a_log = torch.zeros(case.heads, device="cuda", dtype=torch.float32)
        dt_bias = torch.zeros((case.heads, DIM), device="cuda", dtype=torch.float32)
    elif case.gate_regime == "fast_decay":
        g = torch.full(shape, 8.0, device="cuda", dtype=torch.bfloat16)
        a_log = torch.zeros(case.heads, device="cuda", dtype=torch.float32)
        dt_bias = torch.zeros((case.heads, DIM), device="cuda", dtype=torch.float32)
    else:
        raise ValueError(f"unknown gate regime: {case.gate_regime}")

    nseq = len(case.lengths) if case.packed else 1
    # Generate the common initial state directly in BF16.  The FP32 public-state
    # path receives the same representable values widened to FP32, isolating
    # recurrence/storage effects from a different initial quantization.
    state_bf16_vk = torch.randn(
        (nseq, case.heads, DIM, DIM), device="cuda", dtype=torch.bfloat16
    )
    cu_seqlens = None
    if case.packed:
        cu_seqlens = torch.tensor(
            offsets_from_lengths(case.lengths), device="cuda", dtype=torch.int64
        )
    return Inputs(
        lengths=case.lengths,
        packed=case.packed,
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        a_log=a_log,
        dt_bias=dt_bias,
        state_bf16_vk=state_bf16_vk,
        cu_seqlens=cu_seqlens,
    )


def state_for_dtype(inputs: Inputs, dtype: torch.dtype) -> torch.Tensor:
    return inputs.state_bf16_vk.clone().to(dtype)


@torch.inference_mode()
def run_flash(
    inputs: Inputs,
    value_slice: int,
    state_dtype: torch.dtype,
    state_mode: str,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if value_slice not in VALUE_SLICES:
        raise ValueError(f"unsupported value_slice={value_slice}")
    if state_mode not in STATE_MODES:
        raise ValueError(f"unsupported state_mode={state_mode}")

    initial_state = state_for_dtype(inputs, state_dtype) if has_state_in(state_mode) else None
    final_state = None
    if has_state_out(state_mode):
        final_state = torch.empty(
            (inputs.nseq, inputs.heads, DIM, DIM),
            device="cuda",
            dtype=state_dtype,
        )
    out = torch.empty_like(inputs.v)
    workspace = torch.empty(
        flash_kda_C.get_workspace_size(inputs.total_tokens, inputs.heads, inputs.nseq),
        device="cuda",
        dtype=torch.uint8,
    )
    flash_kda_C.fwd(
        inputs.q,
        inputs.k,
        inputs.v,
        inputs.g,
        inputs.beta,
        SCALE,
        out,
        workspace,
        inputs.a_log,
        inputs.dt_bias,
        LOWER_BOUND,
        initial_state,
        final_state,
        inputs.cu_seqlens,
        value_slice,
    )
    torch.cuda.synchronize()
    return out, final_state


def normalize_for_flash(x: torch.Tensor) -> torch.Tensor:
    """FP32 form of FlashKDA's fused q/k normalization."""
    x_fp32 = x.float()
    inv_norm = torch.rsqrt((x_fp32 * x_fp32).sum(dim=-1, keepdim=True) + 1.0e-6)
    return x_fp32 * inv_norm


def preprocess_for_naive(
    inputs: Inputs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reproduce FlashKDA's fused transformations as FP32 reference inputs."""
    q = normalize_for_flash(inputs.q)
    k = normalize_for_flash(inputs.k)
    v = inputs.v.float()
    gate_input = inputs.g.float() + inputs.dt_bias.view(1, 1, inputs.heads, DIM)
    decay = LOWER_BOUND * torch.sigmoid(
        inputs.a_log.exp().view(1, 1, inputs.heads, 1) * gate_input
    )
    beta = torch.sigmoid(inputs.beta.float())
    return q, k, v, decay, beta


@torch.inference_mode()
def run_naive(
    inputs: Inputs,
    state_mode: str,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run naive_recurrent_kda, splitting packed input sequence by sequence."""
    q, k, v, decay, beta = preprocess_for_naive(inputs)
    offsets = offsets_from_lengths(inputs.lengths)
    output_parts: list[torch.Tensor] = []
    final_parts: list[torch.Tensor] = []

    for sequence_index, (bos, eos) in enumerate(zip(offsets[:-1], offsets[1:])):
        initial_state_kv = None
        if has_state_in(state_mode):
            # FlashKDA state is [N,H,V,K]; naive.py expects [B,H,K,V].
            initial_state_kv = (
                inputs.state_bf16_vk[sequence_index : sequence_index + 1]
                .float()
                .transpose(-2, -1)
                .contiguous()
            )
        out_part, final_state_kv = naive_recurrent_kda(
            q=q[:, bos:eos],
            k=k[:, bos:eos],
            v=v[:, bos:eos],
            g=decay[:, bos:eos],
            beta=beta[:, bos:eos],
            scale=SCALE,
            initial_state=initial_state_kv,
            output_final_state=has_state_out(state_mode),
        )
        output_parts.append(out_part)
        if final_state_kv is not None:
            # Convert [1,H,K,V] back to FlashKDA's [1,H,V,K].
            final_parts.append(final_state_kv.transpose(-2, -1).contiguous())

    output = torch.cat(output_parts, dim=1)
    final_state = torch.cat(final_parts, dim=0) if final_parts else None
    torch.cuda.synchronize()
    return output, final_state


@torch.inference_mode()
def run_chunk(
    inputs: Inputs,
    state_mode: str,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run FLA's Triton chunk.py path with its FlashKDA backend disabled."""
    if os.environ.get("FLA_FLASH_KDA") != "0":
        raise RuntimeError("FLA_FLASH_KDA must be 0 before importing/calling chunk_kda")
    initial_state = None
    if has_state_in(state_mode):
        # The pinned FLA chunk reference accepts only FP32 public state and uses
        # V-first layout when state_v_first=True.
        initial_state = inputs.state_bf16_vk.float().contiguous()
    output, final_state = chunk_kda(
        q=inputs.q,
        k=inputs.k,
        v=inputs.v,
        g=inputs.g,
        beta=inputs.beta,
        scale=SCALE,
        initial_state=initial_state,
        output_final_state=has_state_out(state_mode),
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True,
        lower_bound=LOWER_BOUND,
        state_v_first=True,
        cu_seqlens=inputs.cu_seqlens,
        A_log=inputs.a_log,
        dt_bias=inputs.dt_bias,
        chunk_size=64,
    )
    torch.cuda.synchronize()
    return output, final_state


def tensor_metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float | int | bool]:
    actual_fp64 = actual.detach().to(torch.float64)
    reference_fp64 = reference.detach().to(torch.float64)
    diff = actual_fp64 - reference_fp64
    abs_diff = diff.abs()
    rmse = diff.square().mean().sqrt()
    reference_rms = reference_fp64.square().mean().sqrt()
    return {
        "numel": actual.numel(),
        "finite": bool(torch.isfinite(actual_fp64).all() and torch.isfinite(reference_fp64).all()),
        "max_abs": float(abs_diff.max().item()),
        "mean_abs": float(abs_diff.mean().item()),
        "rmse": float(rmse.item()),
        "reference_rms": float(reference_rms.item()),
        "rel_rmse": float((rmse / (reference_rms + 1.0e-12)).item()),
    }


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reference_metadata() -> dict[str, str]:
    import fla.ops.kda.chunk as chunk_module
    import fla.ops.kda.naive as naive_module

    naive_path = str(Path(naive_module.__file__).resolve())
    chunk_path = str(Path(chunk_module.__file__).resolve())
    try:
        fla_version = importlib.metadata.version("flash-linear-attention")
    except importlib.metadata.PackageNotFoundError:
        fla_version = "unknown"
    return {
        "fla_version": fla_version,
        "naive_path": naive_path,
        "chunk_path": chunk_path,
        "naive_sha256": file_sha256(naive_path),
        "chunk_sha256": file_sha256(chunk_path),
    }


def base_row(
    case: ReferenceCase,
    state_dtype: torch.dtype,
    candidate: str,
    reference: str,
    tensor_name: str,
    metadata: dict[str, str],
) -> dict[str, object]:
    return {
        "case": case.name,
        "layout": "ragged" if case.packed else "fixed",
        "lengths": "+".join(str(length) for length in case.lengths),
        "total_tokens": sum(case.lengths),
        "heads": case.heads,
        "seed": case.seed,
        "gate_regime": case.gate_regime,
        "state_mode": case.state_mode,
        "state_dtype": dtype_name(state_dtype),
        "candidate": candidate,
        "reference": reference,
        "tensor": tensor_name,
        "fla_version": metadata["fla_version"],
        "naive_sha256": metadata["naive_sha256"],
        "chunk_sha256": metadata["chunk_sha256"],
        "fla_flash_kda": os.environ.get("FLA_FLASH_KDA", ""),
    }


def append_comparison(
    rows: list[dict[str, object]],
    case: ReferenceCase,
    state_dtype: torch.dtype,
    candidate_name: str,
    candidate: tuple[torch.Tensor, torch.Tensor | None],
    reference_name: str,
    reference: tuple[torch.Tensor, torch.Tensor | None],
    metadata: dict[str, str],
    comparison_kind: str,
    hard_limit: float | None = None,
) -> None:
    pairs = [("output", candidate[0], reference[0])]
    if candidate[1] is not None or reference[1] is not None:
        if candidate[1] is None or reference[1] is None:
            raise AssertionError(
                f"state presence mismatch for {case.name}: {candidate_name} vs {reference_name}"
            )
        pairs.append(("final_state", candidate[1], reference[1]))

    for tensor_name, actual_tensor, reference_tensor in pairs:
        if actual_tensor.shape != reference_tensor.shape:
            raise AssertionError(
                f"shape mismatch for {case.name}/{tensor_name}: "
                f"{candidate_name}={actual_tensor.shape}, {reference_name}={reference_tensor.shape}"
            )
        metrics = tensor_metrics(actual_tensor, reference_tensor)
        bitwise_equal = torch.equal(actual_tensor, reference_tensor)
        within_limit = (
            bool(metrics["finite"])
            and (hard_limit is None or float(metrics["rel_rmse"]) <= hard_limit)
        )
        row = base_row(
            case, state_dtype, candidate_name, reference_name, tensor_name, metadata
        )
        row.update(
            {
                "comparison_kind": comparison_kind,
                "bitwise_equal": bitwise_equal,
                "hard_rel_rmse_limit": "" if hard_limit is None else hard_limit,
                "within_limit": within_limit,
                **metrics,
            }
        )
        rows.append(row)
        print(
            f"metric case={case.name} candidate={candidate_name} reference={reference_name} "
            f"tensor={tensor_name} max_abs={metrics['max_abs']:.6e} "
            f"rmse={metrics['rmse']:.6e} rel_rmse={metrics['rel_rmse']:.6e} "
            f"bitwise={bitwise_equal}"
        )


def run_bitwise_matrix(rows: list[dict[str, object]], metadata: dict[str, str]) -> None:
    layouts = (
        ReferenceCase("bitwise_fixed", (33,), 3, 20260930, "random"),
        ReferenceCase("bitwise_ragged", (15, 16, 17), 3, 20260931, "random"),
    )
    for layout_case in layouts:
        inputs = make_inputs(layout_case)
        for state_dtype in STATE_DTYPES:
            for state_mode in STATE_MODES:
                case = ReferenceCase(
                    name=layout_case.name,
                    lengths=layout_case.lengths,
                    heads=layout_case.heads,
                    seed=layout_case.seed,
                    gate_regime=layout_case.gate_regime,
                    state_mode=state_mode,
                    state_dtypes=(state_dtype,),
                )
                baseline = run_flash(inputs, 128, state_dtype, state_mode)
                for value_slice in (16, 32, 64):
                    candidate = run_flash(inputs, value_slice, state_dtype, state_mode)
                    append_comparison(
                        rows,
                        case,
                        state_dtype,
                        f"flash_v{value_slice}",
                        candidate,
                        "flash_v128",
                        baseline,
                        metadata,
                        comparison_kind="bitwise_valueslice",
                        hard_limit=0.0,
                    )
                    relevant_rows = rows[-(2 if has_state_out(state_mode) else 1) :]
                    if not all(bool(row["bitwise_equal"]) for row in relevant_rows):
                        raise AssertionError(
                            f"ValueSlice mismatch: {case.name}/{dtype_name(state_dtype)}/"
                            f"{state_mode}/V{value_slice}"
                        )
        del inputs
        torch.cuda.empty_cache()


def reference_cases() -> tuple[ReferenceCase, ...]:
    return (
        ReferenceCase("fixed_short_stateful", (257,), 1, 20261001, "random"),
        ReferenceCase("fixed_short_out_only", (33,), 1, 20261002, "random", "out_only"),
        ReferenceCase("ragged_short_stateful", (31, 47, 19), 1, 20261003, "random"),
        # H=1 is sufficient for the independent per-head recurrence and keeps
        # the pure-PyTorch long-sequence reference within the 15-minute job.
        ReferenceCase("long_8192_random", (8192,), 1, 20261004, "random"),
        ReferenceCase("long_8192_memory", (8192,), 1, 20261005, "long_memory"),
        # K3 TP8 representative shapes use the Triton chunk reference; running
        # the token-wise naive loop at H=12 would add cost without new head-wise
        # recurrence coverage.
        ReferenceCase(
            "k3_fixed_8192", (8192,), 12, 20261006, "random",
            state_dtypes=(torch.bfloat16,), use_naive=False,
        ),
        ReferenceCase(
            "k3_ragged6", (1300, 547, 2048, 963, 271, 3063), 12, 20261007, "random",
            state_dtypes=(torch.bfloat16,), use_naive=False,
        ),
        ReferenceCase(
            "k3_packed_8x1024", (1024,) * 8, 12, 20261008, "random",
            state_dtypes=(torch.bfloat16,), use_naive=False,
        ),
    )


def run_reference_matrix(
    rows: list[dict[str, object]],
    metadata: dict[str, str],
    smoke_rel_rmse_limit: float,
) -> None:
    for case in reference_cases():
        print(
            f"case_start name={case.name} lengths={case.lengths} heads={case.heads} "
            f"gate={case.gate_regime} state_mode={case.state_mode}"
        )
        inputs = make_inputs(case)
        naive_result = run_naive(inputs, case.state_mode) if case.use_naive else None
        chunk_result = run_chunk(inputs, case.state_mode) if case.use_chunk else None

        if naive_result is not None and chunk_result is not None:
            append_comparison(
                rows,
                case,
                torch.float32,
                "fla_chunk_fp32_state",
                chunk_result,
                "naive_fp32_state",
                naive_result,
                metadata,
                comparison_kind="independent_references",
                hard_limit=smoke_rel_rmse_limit if case.smoke else None,
            )

        for state_dtype in case.state_dtypes:
            flash_v128 = run_flash(inputs, 128, state_dtype, case.state_mode)
            flash_v16 = run_flash(inputs, 16, state_dtype, case.state_mode)
            append_comparison(
                rows,
                case,
                state_dtype,
                "flash_v16",
                flash_v16,
                "flash_v128",
                flash_v128,
                metadata,
                comparison_kind="bitwise_valueslice",
                hard_limit=0.0,
            )
            recent = rows[-(2 if has_state_out(case.state_mode) else 1) :]
            if not all(bool(row["bitwise_equal"]) for row in recent):
                raise AssertionError(
                    f"ValueSlice mismatch in reference case {case.name}/{dtype_name(state_dtype)}"
                )

            for flash_name, flash_result in (
                (f"flash_v128_{dtype_name(state_dtype)}_public_state", flash_v128),
                (f"flash_v16_{dtype_name(state_dtype)}_public_state", flash_v16),
            ):
                if naive_result is not None:
                    append_comparison(
                        rows,
                        case,
                        state_dtype,
                        flash_name,
                        flash_result,
                        "naive_fp32_state",
                        naive_result,
                        metadata,
                        comparison_kind="accuracy_vs_naive",
                        hard_limit=smoke_rel_rmse_limit if case.smoke else None,
                    )
                if chunk_result is not None:
                    append_comparison(
                        rows,
                        case,
                        state_dtype,
                        flash_name,
                        flash_result,
                        "fla_chunk_fp32_state",
                        chunk_result,
                        metadata,
                        comparison_kind="accuracy_vs_chunk",
                        hard_limit=smoke_rel_rmse_limit if case.smoke else None,
                    )
        del inputs, naive_result, chunk_result
        torch.cuda.empty_cache()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError("no correctness rows were generated")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--smoke-rel-rmse-limit",
        type=float,
        default=0.02,
        help="Hard relative-RMSE limit for <=512-token independent-reference cases.",
    )
    parser.add_argument(
        "--require-reference-hashes",
        action="store_true",
        help="Fail unless installed FLA naive.py/chunk.py match the a3edffc snapshot hashes.",
    )
    args = parser.parse_args()

    metadata = reference_metadata()
    print(f"extension={flash_kda_C.__file__}")
    print(f"device={flash_kda_C.get_device_characteristics()}")
    print(f"fla_version={metadata['fla_version']}")
    print(f"naive_path={metadata['naive_path']}")
    print(f"naive_sha256={metadata['naive_sha256']}")
    print(f"chunk_path={metadata['chunk_path']}")
    print(f"chunk_sha256={metadata['chunk_sha256']}")
    print(f"FLA_FLASH_KDA={os.environ['FLA_FLASH_KDA']}")

    hash_matches = {
        key: metadata[f"{key}_sha256"] == expected
        for key, expected in EXPECTED_REFERENCE_HASHES.items()
    }
    print(f"reference_hash_matches={hash_matches}")
    if args.require_reference_hashes and not all(hash_matches.values()):
        raise RuntimeError(
            "installed FLA references do not match the assignment snapshot: "
            f"{hash_matches}"
        )

    rows: list[dict[str, object]] = []
    run_bitwise_matrix(rows, metadata)
    run_reference_matrix(rows, metadata, args.smoke_rel_rmse_limit)
    write_csv(args.output, rows)
    print(f"csv={args.output} rows={len(rows)}")

    failed = [
        row
        for row in rows
        if not bool(row["finite"])
        or (
            row["hard_rel_rmse_limit"] != ""
            and not bool(row["within_limit"])
        )
        or (
            row["comparison_kind"] == "bitwise_valueslice"
            and not bool(row["bitwise_equal"])
        )
    ]
    if failed:
        preview = [
            (row["case"], row["candidate"], row["reference"], row["tensor"], row["rel_rmse"])
            for row in failed[:10]
        ]
        raise AssertionError(f"{len(failed)} strict correctness rows failed: {preview}")
    print("correctness_status=PASS")


if __name__ == "__main__":
    main()
