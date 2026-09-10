# Copyright (c) 2026 by FlashInfer team.
"""Preregistered held-out H12 qualification of the BT16 prepare-grid policy."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from bench_flash_kda_h12_cpc_pair import (
    _bootstrap_speedup,
    _h12_long_cpc,
    _measure,
    _run,
)
from bench_recurrent_kda_prefill import (
    Case,
    _hardware_metadata,
    _make_case,
    _require_cupti,
    _timing_iteration_budget,
)


# Fixed before the first held-out GPU run. These cases are disjoint from the
# six Kimi-K3 H12 development-preset shapes.
HELDOUT_CASES = (
    Case("h12_holdout_fixed_1024", 12, (1024,), False, 13000),
    Case("h12_holdout_fixed_2048_boundary", 12, (2048,), False, 13001),
    Case("h12_holdout_fixed_4096", 12, (4096,), False, 13002),
    Case("h12_holdout_fixed_16384", 12, (16384,), False, 13003),
    Case("h12_holdout_packed_skew", 12, (64, 64, 64, 8000), True, 13004),
    Case("h12_holdout_packed_irregular", 12, (333, 777, 1555, 3999), True, 13005),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-cpc", type=int, default=4)
    parser.add_argument("--candidate-cpc", type=int, default=9)
    parser.add_argument("--dry-run-iters", type=int, default=20)
    parser.add_argument("--repeat-iters", type=int, default=100)
    parser.add_argument("--state-rotations", type=int, default=256)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    _require_cupti()
    dry_run_iters, repeat_iters = _timing_iteration_budget(
        state_rotation_capacity=args.state_rotations,
        requested_dry_run_iters=args.dry_run_iters,
        requested_repeat_iters=args.repeat_iters,
    )
    rows = []
    for case in HELDOUT_CASES:
        with _h12_long_cpc(args.baseline_cpc):
            prepared = _make_case(
                case,
                state_rotations=args.state_rotations,
                candidate_route="dispatcher",
                candidate_backend="cake",
            )
        prepared.reset_state_pools()
        _run(prepared, args.baseline_cpc)
        torch.cuda.synchronize()
        baseline_output = prepared.candidate_output.clone()
        baseline_state = prepared.candidate_state_pool[0].clone()
        prepared.reset_state_pools()
        _run(prepared, args.candidate_cpc)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            prepared.candidate_output, baseline_output, atol=1e-2, rtol=1e-2
        )
        torch.testing.assert_close(
            prepared.candidate_state_pool[0], baseline_state, atol=1e-2, rtol=1e-2
        )

        samples = {"baseline": [], "candidate": []}
        block_medians = {"baseline": [], "candidate": []}
        for name, chunks_per_cta in (
            ("baseline", args.baseline_cpc),
            ("candidate", args.candidate_cpc),
            ("candidate", args.candidate_cpc),
            ("baseline", args.baseline_cpc),
        ):
            prepared.reset_state_pools()
            block = _measure(
                prepared,
                chunks_per_cta,
                dry_run_iters=dry_run_iters,
                repeat_iters=repeat_iters,
            )
            samples[name].extend(block)
            block_medians[name].append(float(np.median(block)))

        speedup, lower, upper = _bootstrap_speedup(
            samples["baseline"], samples["candidate"], seed=case.seed
        )
        total_chunks = sum((length + 15) // 16 for length in case.seq_lens)
        affected = (
            prepared.metadata["variant"] == "bt16_prepare_chain_m64"
            and total_chunks > 128
        )
        decision = "supports_policy" if affected and lower > 1.0 else (
            "rejects_policy" if affected and upper < 1.0 else "inconclusive_or_unaffected"
        )
        rows.append(
            {
                "name": case.name,
                "num_heads": case.num_heads,
                "seq_lens": list(case.seq_lens),
                "layout": "packed" if case.packed else "fixed",
                "total_chunks": total_chunks,
                "route": prepared.metadata["variant"],
                "physical_variants": prepared.metadata["physical_variants"],
                "correct": True,
                "affected": affected,
                "baseline_median_ms": float(np.median(samples["baseline"])),
                "candidate_median_ms": float(np.median(samples["candidate"])),
                "baseline_block_medians_ms": block_medians["baseline"],
                "candidate_block_medians_ms": block_medians["candidate"],
                "baseline_samples_ms": samples["baseline"],
                "candidate_samples_ms": samples["candidate"],
                "speedup": speedup,
                "bootstrap_95": [lower, upper],
                "decision": decision,
            }
        )
        print(
            f"{case.name:<36} chunks={total_chunks:<5} "
            f"{prepared.metadata['variant']:<26} {speedup:.4f}x "
            f"[{lower:.4f}, {upper:.4f}] {decision}"
        )
        del prepared, baseline_output, baseline_state
        torch.cuda.empty_cache()

    report = {
        "schema_version": 1,
        "analysis": "preregistered-kimi-k3-tp8-h12-cpc-heldout",
        "preregistered_before_first_gpu_run": True,
        "hardware": _hardware_metadata(torch.device("cuda")),
        "all_correct": all(row["correct"] for row in rows),
        "scope_identical": True,
        "baseline_cpc": args.baseline_cpc,
        "candidate_cpc": args.candidate_cpc,
        "pair_order": "baseline/candidate/candidate/baseline",
        "rows": rows,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
