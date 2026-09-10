# Copyright (c) 2026 by FlashInfer team.
"""Post-qualification phase attribution for the Stage 10 capacity rule."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from bench_flash_kda_h12_phase_activity import _bootstrap_delta, _columns, _measure_activity
from bench_flash_kda_h12_physical_search import _bt16_prepare_grid
from bench_flash_kda_h12_wave_policy_heldout import (
    BASELINE_CPC,
    CANDIDATE_BY_TOTAL_CHUNKS,
    EXPECTED_PHYSICAL_VARIANTS,
    EXPECTED_TARGET,
    HELDOUT_CASES,
)
from bench_recurrent_kda_prefill import _hardware_metadata, _make_case, _require_cupti
from flashinfer.testing.utils import get_l2_cache_size


CHAIN_EQUIVALENCE_MARGIN_FRACTION = 0.01


def _run(prepared, cpc: int) -> None:
    with _bt16_prepare_grid(cpc, True):
        prepared.candidate_run()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run-iters", type=int, default=20)
    parser.add_argument("--repeat-iters", type=int, default=100)
    parser.add_argument("--state-rotations", type=int, default=256)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    _require_cupti()

    rows = []
    for case in HELDOUT_CASES:
        total_chunks = sum((length + 15) // 16 for length in case.seq_lens)
        candidate_cpc = CANDIDATE_BY_TOTAL_CHUNKS[total_chunks]
        with _bt16_prepare_grid(BASELINE_CPC, True):
            prepared = _make_case(
                case,
                state_rotations=args.state_rotations,
                candidate_route="dispatcher",
                candidate_backend="cake",
            )
        if (
            prepared.metadata["variant"] != "bt16_prepare_chain_m64"
            or prepared.metadata["target"] != EXPECTED_TARGET
            or prepared.metadata["physical_variants"] != EXPECTED_PHYSICAL_VARIANTS
        ):
            raise RuntimeError(f"Stage10 phase route drift: {prepared.metadata}")

        prepared.reset_state_pools()
        _run(prepared, BASELINE_CPC)
        torch.cuda.synchronize()
        baseline_output = prepared.candidate_output.clone()
        baseline_state = prepared.candidate_state_pool[0].clone()
        prepared.reset_state_pools()
        _run(prepared, candidate_cpc)
        torch.cuda.synchronize()
        torch.testing.assert_close(prepared.candidate_output, baseline_output, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(prepared.candidate_state_pool[0], baseline_state, atol=1e-2, rtol=1e-2)

        samples = {"baseline": [], "candidate": []}
        kernel_names: set[str] = set()
        for name, cpc in (
            ("baseline", BASELINE_CPC),
            ("candidate", candidate_cpc),
            ("candidate", candidate_cpc),
            ("baseline", BASELINE_CPC),
        ):
            prepared.reset_state_pools()
            block, names = _measure_activity(
                prepared,
                cpc,
                dry_run_iters=args.dry_run_iters,
                repeat_iters=args.repeat_iters,
            )
            samples[name].extend(block)
            kernel_names.update(names)

        columns = {name: _columns(values) for name, values in samples.items()}
        phases = {}
        for phase_index, phase in enumerate(
            ("prepare_ms", "inter_phase_gap_ms", "chain_ms", "full_span_ms")
        ):
            point, lower, upper = _bootstrap_delta(
                columns["baseline"][phase],
                columns["candidate"][phase],
                seed=case.seed + phase_index,
            )
            phases[phase] = {
                "baseline_median_us": float(np.median(columns["baseline"][phase]) * 1000.0),
                "candidate_median_us": float(np.median(columns["candidate"][phase]) * 1000.0),
                "baseline_minus_candidate_us": point,
                "bootstrap_95_us": [lower, upper],
            }
        full_delta = phases["full_span_ms"]["baseline_minus_candidate_us"]
        prepare_delta = phases["prepare_ms"]["baseline_minus_candidate_us"]
        chain_baseline = phases["chain_ms"]["baseline_median_us"]
        chain_margin_us = chain_baseline * CHAIN_EQUIVALENCE_MARGIN_FRACTION
        chain_lower, chain_upper = phases["chain_ms"]["bootstrap_95_us"]
        rows.append(
            {
                "name": case.name,
                "seq_lens": list(case.seq_lens),
                "total_chunks": total_chunks,
                "baseline_cpc": BASELINE_CPC,
                "candidate_cpc": candidate_cpc,
                "route": prepared.metadata["variant"],
                "target": prepared.metadata["target"],
                "physical_variants": prepared.metadata["physical_variants"],
                "kernel_names": sorted(kernel_names),
                "correct": True,
                "samples": samples,
                "phases": phases,
                "prepare_fraction_of_full_delta": prepare_delta / full_delta if full_delta else None,
                "gates": {
                    "prepare_saving_positive_95": phases["prepare_ms"]["bootstrap_95_us"][0] > 0,
                    "full_span_saving_positive_95": phases["full_span_ms"]["bootstrap_95_us"][0] > 0,
                    "chain_equivalent_within_one_percent_95": (
                        chain_lower >= -chain_margin_us and chain_upper <= chain_margin_us
                    ),
                    "prepare_explains_at_least_90_percent": (
                        full_delta > 0 and prepare_delta / full_delta >= 0.90
                    ),
                },
            }
        )
        print(
            f"{case.name:<28} cpc={candidate_cpc:<2} "
            f"full={full_delta:7.3f} us prepare={prepare_delta:7.3f} us"
        )
        del prepared, baseline_output, baseline_state
        torch.cuda.empty_cache()

    report = {
        "schema_version": 1,
        "analysis": "stage10-h12-capacity-policy-post-qualification-phase-attribution",
        "qualification_source": "evidence/b300_stage10_certificate.json",
        "claim_boundary": (
            "Post-qualification mechanism attribution on the four frozen Stage10 profiles; "
            "it cannot change the prospective qualification decision."
        ),
        "hardware": _hardware_metadata(torch.device("cuda")),
        "baseline_cpc": BASELINE_CPC,
        "candidate_by_total_chunks": CANDIDATE_BY_TOTAL_CHUNKS,
        "pair_order": "baseline/candidate/candidate/baseline",
        "protocol": {
            "timing_backend": "cupti-activity",
            "cold_l2": True,
            "l2_flush_bytes": int(get_l2_cache_size(torch.device("cuda"))) * 2,
            "same_prepared_object_per_pair": True,
            "state_rotations": args.state_rotations,
            "blocks_per_arm": 2,
            "dry_run_iters_per_block": args.dry_run_iters,
            "measured_iterations_per_block": args.repeat_iters,
            "bootstrap_draws": 10000,
            "chain_equivalence_margin_fraction_frozen_before_phase_run": CHAIN_EQUIVALENCE_MARGIN_FRACTION,
        },
        "all_correct": all(row["correct"] for row in rows),
        "scope_identical": True,
        "all_mechanism_gates": all(
            all(row["gates"].values()) for row in rows
        ),
        "rows": rows,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
