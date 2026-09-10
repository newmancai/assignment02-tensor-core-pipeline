# Copyright (c) 2026 by FlashInfer team.
"""Prospective interpolation test of the H12 resident-grid-capacity rule."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from bench_flash_kda_h12_cpc_pair import _bootstrap_speedup
from bench_flash_kda_h12_physical_search import _bt16_prepare_grid
from bench_recurrent_kda_prefill import (
    Case,
    _hardware_metadata,
    _make_case,
    _require_cupti,
    _timing_iteration_budget,
)
from flashinfer.testing import bench_gpu_time


# Frozen before the first Stage 10 GPU execution. These work levels were absent
# from Stages 4-8 and interpolate between the already observed 256/512/1024
# anchors. Candidate cpc values come only from the Stage 8 formula.
HELDOUT_CASES = (
    Case("h12_w384_fixed_6144", 12, (6144,), False, 16000),
    Case("h12_w384_balanced_4", 12, (1488, 1520, 1552, 1584), True, 16001),
    Case("h12_w768_fixed_12288", 12, (12288,), False, 16002),
    Case("h12_w768_balanced_4", 12, (2976, 3040, 3104, 3168), True, 16003),
)
BASELINE_CPC = 9
CANDIDATE_BY_TOTAL_CHUNKS = {384: 7, 768: 13}
SM_COUNT = 148
RESIDENT_CTAS_PER_SM = 5
RESIDENT_GRID_CAPACITY_CTAS = SM_COUNT * RESIDENT_CTAS_PER_SM
NUM_HEADS = 12
EXPECTED_TARGET = "sm_103a"
EXPECTED_PHYSICAL_VARIANTS = [
    "sm_103a:flashkda_bf16_bt16_prepare_0d8e6c8011",
    "sm_103a:flashkda_bf16_bt16_chain_m64_c68ffebac9",
]


def _run(prepared, cpc: int) -> None:
    with _bt16_prepare_grid(cpc, True):
        prepared.candidate_run()


def _measure(prepared, cpc: int, *, dry_run_iters: int, repeat_iters: int) -> list[float]:
    return [
        float(value)
        for value in bench_gpu_time(
            lambda: _run(prepared, cpc),
            enable_cupti=True,
            cold_l2_cache=True,
            use_cuda_graph=False,
            dry_run_iters=dry_run_iters,
            repeat_iters=repeat_iters,
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run-iters", type=int, default=20)
    parser.add_argument("--repeat-iters", type=int, default=100)
    parser.add_argument("--state-rotations", type=int, default=256)
    parser.add_argument("--epoch-id", type=int, required=True)
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
        total_chunks = sum((length + 15) // 16 for length in case.seq_lens)
        candidate_cpc = CANDIDATE_BY_TOTAL_CHUNKS[total_chunks]
        chunk_groups_per_capacity_fill = RESIDENT_GRID_CAPACITY_CTAS // NUM_HEADS
        expected_candidate_cpc = (
            total_chunks + chunk_groups_per_capacity_fill - 1
        ) // chunk_groups_per_capacity_fill
        if candidate_cpc != expected_candidate_cpc:
            raise RuntimeError(
                f"frozen prediction mismatch for {case.name}: "
                f"{candidate_cpc} != {expected_candidate_cpc}"
            )
        baseline_grid_ctas = (
            (total_chunks + BASELINE_CPC - 1) // BASELINE_CPC
        ) * NUM_HEADS
        candidate_grid_ctas = (
            (total_chunks + candidate_cpc - 1) // candidate_cpc
        ) * NUM_HEADS
        if candidate_grid_ctas > RESIDENT_GRID_CAPACITY_CTAS:
            raise RuntimeError("candidate grid does not fit resident CTA capacity")
        previous_grid_ctas = (
            (total_chunks + candidate_cpc - 2) // (candidate_cpc - 1)
        ) * NUM_HEADS
        if previous_grid_ctas <= RESIDENT_GRID_CAPACITY_CTAS:
            raise RuntimeError("candidate cpc is not the first capacity-fitting integer")
        with _bt16_prepare_grid(BASELINE_CPC, True):
            prepared = _make_case(
                case,
                state_rotations=args.state_rotations,
                candidate_route="dispatcher",
                candidate_backend="cake",
            )
        if prepared.metadata["variant"] != "bt16_prepare_chain_m64":
            raise RuntimeError(
                f"held-out route changed for {case.name}: {prepared.metadata}"
            )
        if prepared.metadata["target"] != EXPECTED_TARGET:
            raise RuntimeError(
                f"held-out target changed for {case.name}: {prepared.metadata}"
            )
        if prepared.metadata["physical_variants"] != EXPECTED_PHYSICAL_VARIANTS:
            raise RuntimeError(
                f"held-out physical variants changed for {case.name}: "
                f"{prepared.metadata}"
            )
        prepared.reset_state_pools()
        _run(prepared, BASELINE_CPC)
        torch.cuda.synchronize()
        baseline_output = prepared.candidate_output.clone()
        baseline_state = prepared.candidate_state_pool[0].clone()
        prepared.reset_state_pools()
        _run(prepared, candidate_cpc)
        torch.cuda.synchronize()
        output_max_abs = float(
            (prepared.candidate_output.float() - baseline_output.float()).abs().max()
        )
        final_state_max_abs = float(
            (
                prepared.candidate_state_pool[0].float()
                - baseline_state.float()
            ).abs().max()
        )
        torch.testing.assert_close(prepared.candidate_output, baseline_output, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(prepared.candidate_state_pool[0], baseline_state, atol=1e-2, rtol=1e-2)

        samples = {"baseline": [], "candidate": []}
        block_medians = {"baseline": [], "candidate": []}
        pair_order = (
            (("baseline", BASELINE_CPC), ("candidate", candidate_cpc),
             ("candidate", candidate_cpc), ("baseline", BASELINE_CPC))
            if args.epoch_id % 2 == 0
            else (("candidate", candidate_cpc), ("baseline", BASELINE_CPC),
                  ("baseline", BASELINE_CPC), ("candidate", candidate_cpc))
        )
        for name, cpc in pair_order:
            prepared.reset_state_pools()
            block = _measure(
                prepared,
                cpc,
                dry_run_iters=dry_run_iters,
                repeat_iters=repeat_iters,
            )
            samples[name].extend(block)
            block_medians[name].append(float(np.median(block)))

        speedup, lower, upper = _bootstrap_speedup(
            samples["baseline"], samples["candidate"], seed=case.seed
        )
        baseline_median = float(np.median(samples["baseline"]))
        candidate_median = float(np.median(samples["candidate"]))
        rows.append(
            {
                "name": case.name,
                "seq_lens": list(case.seq_lens),
                "layout": "packed" if case.packed else "fixed",
                "total_chunks": total_chunks,
                "max_chunks": max((length + 15) // 16 for length in case.seq_lens),
                "baseline_cpc": BASELINE_CPC,
                "predicted_candidate_cpc": candidate_cpc,
                "baseline_grid_ctas": baseline_grid_ctas,
                "candidate_grid_ctas": candidate_grid_ctas,
                "candidate_minus_one_grid_ctas": previous_grid_ctas,
                "resident_grid_capacity_ctas": RESIDENT_GRID_CAPACITY_CTAS,
                "route": prepared.metadata["variant"],
                "target": prepared.metadata["target"],
                "physical_variants": prepared.metadata["physical_variants"],
                "correct": True,
                "output_max_abs": output_max_abs,
                "final_state_max_abs": final_state_max_abs,
                "baseline_median_ms": baseline_median,
                "candidate_median_ms": candidate_median,
                "absolute_saving_us": (baseline_median - candidate_median) * 1000.0,
                "baseline_block_medians_ms": block_medians["baseline"],
                "candidate_block_medians_ms": block_medians["candidate"],
                "baseline_samples_ms": samples["baseline"],
                "candidate_samples_ms": samples["candidate"],
                "speedup": speedup,
                "bootstrap_95": [lower, upper],
                "within_epoch_diagnostic": (
                    "positive" if speedup > 1.0 else "non_positive"
                ),
            }
        )
        print(
            f"{case.name:<28} predicted_cpc={candidate_cpc:<2} "
            f"{speedup:.4f}x [{lower:.4f}, {upper:.4f}] "
            f"{rows[-1]['within_epoch_diagnostic']}"
        )
        del prepared, baseline_output, baseline_state
        torch.cuda.empty_cache()

    report = {
        "schema_version": 1,
        "analysis": "stage10-h12-resident-grid-capacity-prospective-interpolation",
        "epoch_id": args.epoch_id,
        "preregistered_before_first_gpu_run": True,
        "prediction_source": "evidence/b300_stage8_certificate.json",
        "selection_free": True,
        "hypothesis": (
            "The smallest integer cpc whose H-way prepare grid fits the measured "
            "resident CTA capacity predicts cpc=7 at total_chunks=384 and cpc=13 "
            "at total_chunks=768 before either work level is measured."
        ),
        "claim_boundary": (
            "Untouched interpolation qualification for two new work levels on one B300 "
            "and one unchanged H12 prepare resource footprint; not other heads/devices."
        ),
        "hardware": _hardware_metadata(torch.device("cuda")),
        "all_correct": all(row["correct"] for row in rows),
        "scope_identical": True,
        "baseline_cpc": BASELINE_CPC,
        "candidate_by_total_chunks": CANDIDATE_BY_TOTAL_CHUNKS,
        "capacity_rule": {
            "formula": "ceil(total_chunks / floor((sm_count * resident_ctas_per_sm) / num_heads))",
            "sm_count": SM_COUNT,
            "resident_ctas_per_sm": RESIDENT_CTAS_PER_SM,
            "resident_grid_capacity_ctas": RESIDENT_GRID_CAPACITY_CTAS,
            "num_heads": NUM_HEADS,
            "chunk_groups_per_capacity_fill": RESIDENT_GRID_CAPACITY_CTAS // NUM_HEADS,
            "interpretation": "first cpc whose rectangular prepare grid is at most the measured resident CTA capacity",
        },
        "pair_order": "/".join(name for name, _cpc in pair_order),
        "protocol": {
            "timing_backend": "cupti",
            "cold_l2": True,
            "same_prepared_object_per_pair": True,
            "same_initial_state_per_timed_call": True,
            "state_rotations": args.state_rotations,
            "blocks_per_arm": 2,
            "dry_run_iters_per_block": dry_run_iters,
            "measured_iterations_per_block": repeat_iters,
            "bootstrap_draws": 10000,
            "primary_resampling_unit": "independent process epoch",
            "familywise_gate": "Bonferroni one-sided 98.75% lower bound above one for all four profiles",
            "deployment_gate": "all familywise lower bounds at least 1.005",
        },
        "rows": rows,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
