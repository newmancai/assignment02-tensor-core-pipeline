# Copyright (c) 2026 by FlashInfer team.
"""Search legal physical parameters inside the selected Kimi-K3 H12 routes."""

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import torch

from bench_recurrent_kda_prefill import (
    H12_CASES,
    _hardware_metadata,
    _make_case,
    _measure,
    _require_cupti,
    _timing_iteration_budget,
)
from flashinfer import kda_prefill


DIRECT_CASES = {
    "h12_packed_128x8",
}
BT16_CASES = {
    "h12_fixed_512",
    "h12_fixed_8192",
    "h12_packed_mixed",
}
BT16_CHUNKS_PER_CTA = (1, 2, 3, 4, 6, 8, 12, 16)


@contextmanager
def _direct_state_pack(kind: str):
    original = kda_prefill._FLASH_KDA_H12_DIRECT_N32_EARLY_STATE_PACK_MAX_SEQUENCE_LENGTH
    if kind == "short":
        replacement = 1 << 30
    elif kind == "long":
        replacement = 0
    else:
        raise ValueError(f"unknown direct state-pack kind {kind!r}")
    kda_prefill._FLASH_KDA_H12_DIRECT_N32_EARLY_STATE_PACK_MAX_SEQUENCE_LENGTH = (
        replacement
    )
    try:
        yield
    finally:
        kda_prefill._FLASH_KDA_H12_DIRECT_N32_EARLY_STATE_PACK_MAX_SEQUENCE_LENGTH = (
            original
        )


@contextmanager
def _bt16_prepare_grid(chunks_per_cta: int, wave_quantized: bool):
    original_cpc = kda_prefill._bt16_chunks_per_prepare_cta
    original_quantize = kda_prefill._wave_quantized_bt16_prepare_ctas
    kda_prefill._bt16_chunks_per_prepare_cta = (
        lambda *, num_heads, total_chunks: chunks_per_cta
    )
    if not wave_quantized:
        kda_prefill._wave_quantized_bt16_prepare_ctas = (
            lambda *, rectangular_ctas, num_heads, sm_count: rectangular_ctas
        )
    try:
        yield
    finally:
        kda_prefill._bt16_chunks_per_prepare_cta = original_cpc
        kda_prefill._wave_quantized_bt16_prepare_ctas = original_quantize


def _correct(candidate, baseline) -> dict[str, float | bool]:
    candidate.reset_state_pools()
    baseline.reset_state_pools()
    candidate.candidate_run()
    baseline.candidate_run()
    torch.cuda.synchronize()
    output_delta = float(
        (candidate.candidate_output.float() - baseline.candidate_output.float())
        .abs()
        .max()
    )
    state_delta = float(
        (
            candidate.candidate_state_pool[0].float()
            - baseline.candidate_state_pool[0].float()
        )
        .abs()
        .max()
    )
    try:
        torch.testing.assert_close(
            candidate.candidate_output,
            baseline.candidate_output,
            atol=1e-2,
            rtol=1e-2,
        )
        torch.testing.assert_close(
            candidate.candidate_state_pool[0],
            baseline.candidate_state_pool[0],
            atol=1e-2,
            rtol=1e-2,
        )
    except AssertionError:
        passed = False
    else:
        passed = True
    return {
        "passed": passed,
        "output_max_abs": output_delta,
        "final_state_max_abs": state_delta,
    }


def _candidate_specs(case_name: str, bt16_cpc: tuple[int, ...]):
    if case_name in DIRECT_CASES:
        # The short specialization has a hard <=128-token carrier contract.
        # Only test whether the more general long carrier can beat it at 128.
        yield {"family": "direct_state_pack", "kind": "long"}
    elif case_name in BT16_CASES:
        for chunks_per_cta in bt16_cpc:
            for wave_quantized in (True, False):
                yield {
                    "family": "bt16_prepare_grid",
                    "chunks_per_cta": chunks_per_cta,
                    "wave_quantized": wave_quantized,
                }


@contextmanager
def _candidate_context(spec: dict[str, object]):
    if spec["family"] == "direct_state_pack":
        with _direct_state_pack(str(spec["kind"])):
            yield
    elif spec["family"] == "bt16_prepare_grid":
        with _bt16_prepare_grid(
            int(spec["chunks_per_cta"]), bool(spec["wave_quantized"])
        ):
            yield
    else:
        raise ValueError(f"unknown candidate family {spec['family']!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run-iters", type=int, default=20)
    parser.add_argument("--repeat-iters", type=int, default=100)
    parser.add_argument("--state-rotations", type=int, default=256)
    parser.add_argument(
        "--case",
        action="append",
        choices=tuple(case.name for case in H12_CASES),
        help="run only this H12 case; repeat to select multiple cases",
    )
    parser.add_argument(
        "--bt16-cpc",
        type=int,
        nargs="+",
        default=BT16_CHUNKS_PER_CTA,
        help="BT16 chunks-per-prepare-CTA values to search",
    )
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    _require_cupti()
    dry_run_iters, repeat_iters = _timing_iteration_budget(
        state_rotation_capacity=args.state_rotations,
        requested_dry_run_iters=args.dry_run_iters,
        requested_repeat_iters=args.repeat_iters,
    )
    rows = []
    for case in H12_CASES:
        if args.case and case.name not in args.case:
            continue
        baseline = _make_case(
            case,
            state_rotations=args.state_rotations,
            candidate_route="dispatcher",
            candidate_backend="cake",
        )
        baseline.reset_state_pools()
        baseline_ms, baseline_samples = _measure(
            baseline.candidate_run,
            dry_run_iters=dry_run_iters,
            repeat_iters=repeat_iters,
        )
        candidates = []
        for spec in _candidate_specs(case.name, tuple(args.bt16_cpc)):
            candidate = None
            try:
                with _candidate_context(spec):
                    candidate = _make_case(
                        case,
                        state_rotations=args.state_rotations,
                        candidate_route="dispatcher",
                        candidate_backend="cake",
                    )
                    correctness = _correct(candidate, baseline)
                    result = {**spec, "metadata": candidate.metadata, "correctness": correctness}
                    if correctness["passed"]:
                        candidate.reset_state_pools()
                        median_ms, samples = _measure(
                            candidate.candidate_run,
                            dry_run_iters=dry_run_iters,
                            repeat_iters=repeat_iters,
                        )
                        result.update(
                            {
                                "median_ms": median_ms,
                                "samples_ms": samples,
                                "speedup_vs_dispatcher": baseline_ms / median_ms,
                            }
                        )
            except Exception as error:
                result = {
                    **spec,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            candidates.append(result)
            label = json.dumps(spec, sort_keys=True, separators=(",", ":"))
            if "median_ms" in result:
                print(
                    f"{case.name:<22} {label:<86} "
                    f"{result['median_ms'] * 1000:9.3f} us "
                    f"{result['speedup_vs_dispatcher']:.4f}x"
                )
            else:
                print(f"{case.name:<22} {label:<86} rejected")
            if candidate is not None:
                del candidate
        rows.append(
            {
                "name": case.name,
                "num_heads": case.num_heads,
                "seq_lens": list(case.seq_lens),
                "layout": "packed" if case.packed else "fixed",
                "dispatcher": {
                    "metadata": baseline.metadata,
                    "median_ms": baseline_ms,
                    "samples_ms": baseline_samples,
                },
                "candidates": candidates,
            }
        )
        del baseline
        torch.cuda.empty_cache()

    report = {
        "schema_version": 1,
        "analysis": "kimi-k3-tp8-h12-intra-route-physical-search",
        "hardware": _hardware_metadata(torch.device("cuda")),
        "protocol": {
            "timing_backend": "cupti",
            "cold_l2": True,
            "public_api_scope": True,
            "same_initial_state_per_timed_call": True,
            "dry_run_iters": dry_run_iters,
            "repeat_iters": repeat_iters,
        },
        "rows": rows,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
