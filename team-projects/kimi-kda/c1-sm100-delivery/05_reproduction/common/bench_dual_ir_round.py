#!/usr/bin/env python3
"""Two-lane B300 screen for the dual-IR agent loop.

Each implementation runs in a fresh worker.  The controller creates an
official peer result first, then rotates the candidate order across blocks.
The measured scope is the public state-updating call, including copy-back.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys


ROOT = Path("<REMOTE_HOME>")
MIXED = (1300, 547, 2048, 963, 271, 3063)
CASES = {
    "h12_packed_mixed": (12, MIXED),
    "h12_packed_balanced6": (12, (1366, 1366, 1365, 1365, 1365, 1365)),
    "h12_packed_balanced7": (12, (1171, 1171, 1170, 1170, 1170, 1170, 1170)),
    "h12_packed_balanced8": (12, (1024,) * 8),
    "h12_packed_balanced3": (12, (2731, 2731, 2730)),
    "h12_packed_balanced3_qual": (12, (2731, 2731, 2730)),
    "h12_packed_balanced4": (12, (2048,) * 4),
    "h12_packed_skew3": (12, (4096, 2048, 2048)),
    "h64_packed_mixed": (64, MIXED),
    "h96_packed_mixed": (96, MIXED),
    "h96_packed_mixed_tail4": (96, MIXED),
    "h96_packed_mixed_minimax172": (96, MIXED),
    "h96_packed_mixed_max169": (96, MIXED),
    "h96_packed_mixed_grid147": (96, MIXED),
    "h96_packed_mixed_reverse": (96, MIXED),
    "h96_fixed8192": (96, (8192,)),
}
CASE_VARIANTS = {
    "h12_packed_mixed": ("official", "hmma_v128", "hmma_v64", "cake"),
    "h12_packed_balanced6": ("official", "hmma_v128", "hmma_v64", "cake"),
    "h12_packed_balanced7": ("official", "hmma_v128", "hmma_v64", "cake"),
    "h12_packed_balanced8": ("official", "hmma_v128", "hmma_v64", "cake"),
    "h12_packed_balanced3": ("official", "hmma_v128", "hmma_v32", "cake"),
    "h12_packed_balanced3_qual": (
        "official", "hmma_v128", "hmma_v64", "hmma_v32", "cake"
    ),
    "h12_packed_balanced4": ("official", "hmma_v64", "hmma_v32", "cake"),
    "h12_packed_skew3": ("official", "hmma_v64", "hmma_v32", "cake"),
    "h64_packed_mixed": (
        "official",
        "cake",
        "evolution_incumbent",
        "evolution_virtual152",
    ),
    "h96_packed_mixed": (
        "official",
        "cake",
        "evolution_incumbent",
        "evolution_virtual152",
    ),
    "h96_packed_mixed_tail4": (
        "official",
        "cake",
        "evolution_incumbent",
        "evolution_tail4",
    ),
    "h96_packed_mixed_minimax172": (
        "official",
        "cake",
        "evolution_incumbent",
        "evolution_minimax172",
    ),
    "h96_packed_mixed_max169": (
        "official",
        "cake",
        "evolution_incumbent",
        "evolution_max169",
    ),
    "h96_packed_mixed_grid147": (
        "official",
        "cake",
        "evolution_incumbent",
        "evolution_grid147",
    ),
    "h96_packed_mixed_reverse": (
        "official",
        "cake",
        "evolution_incumbent",
        "evolution_reverse_tasks",
    ),
    "h96_fixed8192": ("official", "hmma_v128", "hmma_v64", "cake"),
}


def _write_execution_receipt(args, module_names, dependency_files):
    try:
        try:
            from kda_ir.execution_identity import (
                collect_execution_identity,
                write_execution_identity,
            )
        except ModuleNotFoundError:
            from execution_identity import (
                collect_execution_identity,
                write_execution_identity,
            )
        receipt = collect_execution_identity(module_names, dependency_files)
        path = args.output / (
            f"{args.case}_{args.variant}_{args.block}_execution_identity.json"
        )
        write_execution_identity(receipt, path)
        return {
            "status": "recorded",
            "path": str(path),
            "canonical_sha256": receipt.canonical_sha256(),
            "python_executable": receipt.python_executable,
            "module_files": {
                item.requested_name: item.file for item in receipt.modules
            },
            "native_extension_count": len(receipt.loaded_native_extensions),
        }
    except Exception as error:
        return {
            "status": "recording_failed",
            "error_type": type(error).__name__,
            "error": str(error),
        }


def _validate_virtual_schedule(evo, sequence_lengths, num_heads, device):
    schedule, counts, stride = evo._build_persistent_scalar_schedule(
        tuple(sequence_lengths), num_heads, 152, device
    )
    expected = sum((length + 31) // 32 for length in sequence_lengths) * num_heads
    observed = int(counts.sum().item())
    if observed != expected or int(counts.max().item()) != stride:
        raise RuntimeError(
            f"invalid virtual-152 schedule: work={observed}/{expected}, "
            f"max={int(counts.max().item())}, stride={stride}"
        )
    expected_stride = {64: 114, 96: 166}[num_heads]
    if stride != expected_stride:
        raise RuntimeError(f"unexpected virtual-152 stride {stride} != {expected_stride}")
    return schedule, counts, stride, expected


def worker(args):
    import torch
    from flashinfer.testing import bench_gpu_time

    heads, lengths = CASES[args.case]
    total, sequences = sum(lengths), len(lengths)
    seed = 42000 + list(CASES).index(args.case)
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def rand(shape, dtype=torch.bfloat16):
        return torch.randn(shape, generator=generator, device="cuda").to(dtype)

    shape = (1, total, heads, 128)
    q, k, v, g = [rand(shape) for _ in range(4)]
    beta = rand((1, total, heads))
    a_log = torch.rand(heads, generator=generator, device="cuda")
    dt_bias = torch.rand((heads, 128), generator=generator, device="cuda")
    initial = (rand((sequences, heads, 128, 128), torch.float32) * 0.25).to(
        torch.bfloat16
    )
    states = initial.unsqueeze(0).expand(args.state_rotations, *initial.shape).clone()
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    packed = args.case != "h96_fixed8192"
    cu_seqlens = (
        torch.tensor(offsets, dtype=torch.int64, device="cuda") if packed else None
    )
    output = torch.empty_like(q)
    final = torch.empty_like(initial)
    decision = {
        "device_sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
        "seed": seed,
    }

    if args.variant in {
        "cake",
        "evolution_incumbent",
        "evolution_virtual152",
        "evolution_tail4",
        "evolution_minimax172",
        "evolution_max169",
        "evolution_grid147",
        "evolution_reverse_tasks",
    }:
        from flashinfer.kda import recurrent_kda
        import flashinfer.kda_evolution as evo
        import flashinfer.kda_prefill as prefill

        backend = "cake" if args.variant == "cake" else "evolution"
        original_route = evo._route
        if backend == "evolution" and heads == 96 and tuple(lengths) == MIXED:
            # This is an explicit search candidate outside the deployment
            # activation set.  Open only this exact shape so the public adapter
            # still supplies its measured same-stream state copy-back.
            shape_key = (False, heads, tuple(lengths))
            evo._PUBLIC_EVOLUTION_ACTIVATION_SHAPES = frozenset(
                (*evo._PUBLIC_EVOLUTION_ACTIVATION_SHAPES, shape_key)
            )
            decision["search_only_activation"] = list(shape_key[:2]) + [list(shape_key[2])]
        if args.variant == "evolution_virtual152":
            if heads not in (64, 96) or tuple(lengths) != MIXED or not packed:
                raise ValueError("virtual-152 is restricted to the mixed H64/H96 route")

            def virtual_route(sequence_lengths, num_heads, fixed_layout, device):
                if tuple(sequence_lengths) == MIXED and num_heads == heads and not fixed_layout:
                    schedule, counts, stride, work = _validate_virtual_schedule(
                        evo, sequence_lengths, num_heads, device
                    )
                    decision.update(
                        {
                            "schedule_ctas": 152,
                            "schedule_stride": stride,
                            "schedule_work_items": work,
                            "schedule_validation": "count-conserving",
                        }
                    )
                    return evo._Route(
                        variant=f"m128_h{num_heads}_p1_s{stride}",
                        grid_x=152,
                        tile_schedule=schedule,
                        tile_schedule_counts=counts,
                    )
                return original_route(sequence_lengths, num_heads, fixed_layout, device)

            evo._route = virtual_route
        elif args.variant == "evolution_tail4":
            if heads != 96 or tuple(lengths) != MIXED or not packed:
                raise ValueError("tail4 is restricted to the mixed H96 route")

            def tail4_route(sequence_lengths, num_heads, fixed_layout, device):
                if tuple(sequence_lengths) == MIXED and num_heads == 96 and not fixed_layout:
                    recipes = (
                        (21, (41, 64, 64)), (17, (9, 18, 18, 31, 96)),
                        (17, (31, 41, 96)), (16, (18, 18, 41, 96)),
                        (12, (9, 31, 31, 96)), (11, (9, 64, 96)),
                        (9, (9, 18, 41, 96)), (8, (9, 9, 9, 41, 96)),
                        (7, (9, 31, 64, 64)), (5, (31, 31, 41, 64)),
                        (4, (18, 18, 31, 41, 64)), (3, (9, 9, 18, 64, 64)),
                        (3, (64, 96)), (2, (18, 18, 31, 96)),
                        (2, (9, 31, 31, 31, 64)), (2, (18, 31, 41, 41, 41)),
                        (1, (9, 9, 18, 31, 96)), (1, (9, 31, 41, 41, 41)),
                        (1, (9, 18, 64, 64)), (1, (9, 9, 41, 41, 64)),
                        (1, (31, 31, 31, 64)), (1, (9, 18, 31, 41, 64)),
                        (1, (18, 41, 41, 64)), (1, (9, 41, 41, 64)),
                        (1, (31, 64, 64)),
                    )
                    task_queues = {}
                    for sequence, length in enumerate(sequence_lengths):
                        chunks = (length + 31) // 32
                        task_queues[chunks] = [
                            sequence * num_heads + head for head in range(num_heads)
                        ]
                    rows = []
                    counts = []
                    task_count = 0
                    for multiplicity, pattern in recipes:
                        for _ in range(multiplicity):
                            row = []
                            for chunks in pattern:
                                if not task_queues[chunks]:
                                    raise RuntimeError(f"tail4 exhausted chunk class {chunks}")
                                task = task_queues[chunks].pop(0)
                                task_count += 1
                                row.extend(
                                    task | (local_chunk << 10) | (chunks << 18)
                                    for local_chunk in range(chunks)
                                )
                            if len(row) > 173:
                                raise RuntimeError("tail4 row exceeds s173 carrier")
                            counts.append(len(row))
                            rows.extend(row + [0] * (173 - len(row)))
                    if len(counts) != 148 or task_count != len(sequence_lengths) * num_heads:
                        raise RuntimeError(
                            f"tail4 coverage mismatch: rows={len(counts)}, tasks={task_count}"
                        )
                    if any(task_queues.values()) or sum(counts) != 24864 or max(counts) != 173:
                        raise RuntimeError("tail4 failed count conservation")
                    schedule = torch.tensor(rows, dtype=torch.int32, device=device)
                    count_tensor = torch.tensor(counts, dtype=torch.int32, device=device)
                    decision.update(
                        {
                            "schedule_ctas": 148,
                            "schedule_stride": 173,
                            "schedule_work_items": sum(counts),
                            "schedule_task_count": task_count,
                            "schedule_max_task_count_at_stride": 4,
                            "schedule_order": "tail4_recipe_v1",
                            "schedule_validation": "task-and-chunk-conserving",
                        }
                    )
                    return evo._Route(
                        variant="m128_h96_p1_s173",
                        grid_x=148,
                        tile_schedule=schedule,
                        tile_schedule_counts=count_tensor,
                    )
                return original_route(sequence_lengths, num_heads, fixed_layout, device)

            evo._route = tail4_route
        elif args.variant == "evolution_minimax172":
            if heads != 96 or tuple(lengths) != MIXED or not packed:
                raise ValueError("minimax172 is restricted to the mixed H96 route")

            def minimax172_route(sequence_lengths, num_heads, fixed_layout, device):
                if tuple(sequence_lengths) == MIXED and num_heads == 96 and not fixed_layout:
                    recipes = (
                        (31, (9, 18, 41, 96)), (25, (41, 64, 64)),
                        (24, (9, 18, 18, 31, 96)), (22, (31, 41, 96)),
                        (17, (9, 31, 64, 64)), (11, (9, 31, 31, 96)),
                        (6, (18, 18, 31, 41, 64)), (4, (9, 64, 96)),
                        (3, (9, 9, 9, 41, 96)), (2, (18, 31, 41, 41, 41)),
                        (1, (18, 41, 41, 64)), (1, (18, 18, 31, 96)),
                        (1, (31, 31, 41, 64)),
                    )
                    task_queues = {}
                    for sequence, length in enumerate(sequence_lengths):
                        chunks = (length + 31) // 32
                        task_queues[chunks] = [
                            sequence * num_heads + head for head in range(num_heads)
                        ]
                    rows = []
                    counts = []
                    task_count = 0
                    for multiplicity, pattern in recipes:
                        for _ in range(multiplicity):
                            row = []
                            for chunks in pattern:
                                if not task_queues[chunks]:
                                    raise RuntimeError(
                                        f"minimax172 exhausted chunk class {chunks}"
                                    )
                                task = task_queues[chunks].pop(0)
                                task_count += 1
                                row.extend(
                                    task | (local_chunk << 10) | (chunks << 18)
                                    for local_chunk in range(chunks)
                                )
                            if len(row) > 172:
                                raise RuntimeError("minimax172 row exceeds load 172")
                            counts.append(len(row))
                            rows.extend(row + [0] * (173 - len(row)))
                    if len(counts) != 148 or task_count != len(sequence_lengths) * num_heads:
                        raise RuntimeError(
                            f"minimax172 coverage mismatch: rows={len(counts)}, "
                            f"tasks={task_count}"
                        )
                    if any(task_queues.values()) or sum(counts) != 24864 or max(counts) != 172:
                        raise RuntimeError("minimax172 failed count conservation")
                    schedule = torch.tensor(rows, dtype=torch.int32, device=device)
                    count_tensor = torch.tensor(counts, dtype=torch.int32, device=device)
                    decision.update(
                        {
                            "schedule_ctas": 148,
                            "schedule_stride": 173,
                            "schedule_max_load": 172,
                            "schedule_work_items": sum(counts),
                            "schedule_task_count": task_count,
                            "schedule_order": "minimax172_recipe_v1",
                            "schedule_validation": "task-and-chunk-conserving",
                        }
                    )
                    return evo._Route(
                        variant="m128_h96_p1_s173",
                        grid_x=148,
                        tile_schedule=schedule,
                        tile_schedule_counts=count_tensor,
                    )
                return original_route(sequence_lengths, num_heads, fixed_layout, device)

            evo._route = minimax172_route
        elif args.variant == "evolution_max169":
            if heads != 96 or tuple(lengths) != MIXED or not packed:
                raise ValueError("max169 is restricted to the mixed H96 route")

            def max169_route(sequence_lengths, num_heads, fixed_layout, device):
                if tuple(sequence_lengths) == MIXED and num_heads == 96 and not fixed_layout:
                    recipes = (
                        (44, (31, 41, 96)), (23, (41, 64, 64)),
                        (19, (9, 64, 96)), (14, (9, 31, 31, 96)),
                        (10, (9, 18, 18, 18, 41, 64)),
                        (9, (9, 9, 18, 18, 18, 96)), (9, (9, 31, 64, 64)),
                        (6, (9, 18, 41, 96)), (3, (18, 18, 18, 18, 96)),
                        (2, (9, 9, 9, 18, 18, 41, 64)),
                        (2, (18, 18, 18, 31, 41, 41)),
                        (2, (9, 9, 9, 18, 31, 31, 31, 31)),
                        (1, (31, 31, 41, 64)), (1, (9, 9, 9, 9, 18, 18, 96)),
                        (1, (9, 18, 18, 18, 31, 31, 41)),
                        (1, (9, 9, 18, 18, 31, 41, 41)),
                        (1, (9, 18, 18, 41, 41, 41)),
                    )
                    task_queues = {}
                    for sequence, length in enumerate(sequence_lengths):
                        chunks = (length + 31) // 32
                        task_queues[chunks] = [
                            sequence * num_heads + head for head in range(num_heads)
                        ]
                    rows = []
                    counts = []
                    task_count = 0
                    for multiplicity, pattern in recipes:
                        for _ in range(multiplicity):
                            row = []
                            for chunks in pattern:
                                if not task_queues[chunks]:
                                    raise RuntimeError(f"max169 exhausted chunk class {chunks}")
                                task = task_queues[chunks].pop(0)
                                task_count += 1
                                row.extend(
                                    task | (local_chunk << 10) | (chunks << 18)
                                    for local_chunk in range(chunks)
                                )
                            if len(row) > 169:
                                raise RuntimeError("max169 row exceeds load 169")
                            counts.append(len(row))
                            rows.extend(row + [0] * (173 - len(row)))
                    if len(counts) != 148 or task_count != len(sequence_lengths) * num_heads:
                        raise RuntimeError(
                            f"max169 coverage mismatch: rows={len(counts)}, tasks={task_count}"
                        )
                    if any(task_queues.values()) or sum(counts) != 24864 or max(counts) != 169:
                        raise RuntimeError("max169 failed count conservation")
                    schedule = torch.tensor(rows, dtype=torch.int32, device=device)
                    count_tensor = torch.tensor(counts, dtype=torch.int32, device=device)
                    decision.update(
                        {
                            "schedule_ctas": 148,
                            "schedule_stride": 173,
                            "schedule_max_load": 169,
                            "schedule_work_items": sum(counts),
                            "schedule_task_count": task_count,
                            "schedule_order": "max169_recipe_v1",
                            "schedule_validation": "task-and-chunk-conserving",
                            "minimax_proof": "168 impossible and 169 constructively reachable",
                        }
                    )
                    return evo._Route(
                        variant="m128_h96_p1_s173",
                        grid_x=148,
                        tile_schedule=schedule,
                        tile_schedule_counts=count_tensor,
                    )
                return original_route(sequence_lengths, num_heads, fixed_layout, device)

            evo._route = max169_route
        elif args.variant == "evolution_grid147":
            if heads != 96 or tuple(lengths) != MIXED or not packed:
                raise ValueError("grid-147 is restricted to the mixed H96 route")

            def grid147_route(sequence_lengths, num_heads, fixed_layout, device):
                if tuple(sequence_lengths) == MIXED and num_heads == 96 and not fixed_layout:
                    schedule, counts, stride = evo._build_persistent_scalar_schedule(
                        tuple(sequence_lengths), num_heads, 147, device
                    )
                    expected = sum(
                        (length + 31) // 32 for length in sequence_lengths
                    ) * num_heads
                    observed = int(counts.sum().item())
                    if observed != expected or int(counts.max().item()) != stride:
                        raise RuntimeError(
                            f"invalid grid-147 schedule: work={observed}/{expected}, "
                            f"max={int(counts.max().item())}, stride={stride}"
                        )
                    if stride != 173:
                        raise RuntimeError(f"grid-147 requires stride 173, got {stride}")
                    decision.update(
                        {
                            "schedule_ctas": 147,
                            "schedule_stride": stride,
                            "schedule_work_items": expected,
                            "schedule_validation": "count-conserving",
                        }
                    )
                    return evo._Route(
                        variant="m128_h96_p1_s173",
                        grid_x=147,
                        tile_schedule=schedule,
                        tile_schedule_counts=counts,
                    )
                return original_route(sequence_lengths, num_heads, fixed_layout, device)

            evo._route = grid147_route
        elif args.variant == "evolution_reverse_tasks":
            if heads != 96 or tuple(lengths) != MIXED or not packed:
                raise ValueError("reverse-tasks is restricted to the mixed H96 route")

            def reverse_tasks_route(sequence_lengths, num_heads, fixed_layout, device):
                if tuple(sequence_lengths) == MIXED and num_heads == 96 and not fixed_layout:
                    schedule, counts, stride = evo._build_persistent_scalar_schedule(
                        tuple(sequence_lengths), num_heads, 148, device
                    )
                    if stride != 173:
                        raise RuntimeError(f"reverse-tasks requires stride 173, got {stride}")
                    rows = schedule.view(148, stride).cpu().tolist()
                    count_values = counts.cpu().tolist()
                    reordered = []
                    for row, count in zip(rows, count_values, strict=True):
                        used = row[:count]
                        groups = []
                        for encoded in used:
                            task = encoded & 1023
                            if not groups or (groups[-1][0] & 1023) != task:
                                groups.append([encoded])
                            else:
                                groups[-1].append(encoded)
                        flat = [encoded for group in reversed(groups) for encoded in group]
                        reordered.extend(flat + [0] * (stride - count))
                    schedule = torch.tensor(reordered, dtype=torch.int32, device=device)
                    expected = sum(
                        (length + 31) // 32 for length in sequence_lengths
                    ) * num_heads
                    if int(counts.sum().item()) != expected:
                        raise RuntimeError("reverse-tasks changed the work count")
                    decision.update(
                        {
                            "schedule_ctas": 148,
                            "schedule_stride": stride,
                            "schedule_work_items": expected,
                            "schedule_order": "reverse_task_groups_preserve_chunk_order",
                            "schedule_validation": "count-conserving",
                        }
                    )
                    return evo._Route(
                        variant="m128_h96_p1_s173",
                        grid_x=148,
                        tile_schedule=schedule,
                        tile_schedule_counts=counts,
                    )
                return original_route(sequence_lengths, num_heads, fixed_layout, device)

            evo._route = reverse_tasks_route

        observed_modules = []
        original_evo_load = evo.load_flash_kda_evolution_module
        original_cake_load = prefill._get_flash_kda_prefill_module

        def record_evo(variant, target):
            loaded = original_evo_load(variant, target)
            observed_modules.append(
                {
                    "kind": "evolution",
                    "variant": variant,
                    "target": target,
                    "loaded_type": type(loaded).__name__,
                    "loaded_file": getattr(loaded, "__file__", None),
                }
            )
            return loaded

        def record_cake(variant, target):
            loaded = original_cake_load(variant, target)
            observed_modules.append(
                {
                    "kind": "cake",
                    "variant": variant,
                    "target": target,
                    "loaded_type": type(loaded).__name__,
                    "loaded_file": getattr(loaded, "__file__", None),
                }
            )
            return loaded

        evo.load_flash_kda_evolution_module = record_evo
        prefill._get_flash_kda_prefill_module = record_cake

        def invoke(state):
            recurrent_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                A_log=a_log,
                dt_bias=dt_bias,
                scale=1 / math.sqrt(128),
                initial_state=state,
                output=output,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                lower_bound=-5.0,
                cu_seqlens=cu_seqlens,
                beta_is_logit=True,
                backend=backend,
            )

        module_path = importlib.import_module("flashinfer.kda").__file__
        identity_modules = (
            "torch",
            "flashinfer.kda",
            "flashinfer.kda_prefill",
            "flashinfer.kda_evolution",
        )
        identity_dependencies = tuple(
            path
            for name in identity_modules[1:]
            if (path := getattr(importlib.import_module(name), "__file__", None))
        )
    else:
        if args.variant == "official":
            sys.path.insert(0, str(ROOT / "FlashKDA-c1-official"))
            module = importlib.import_module("flash_kda_C")
            extra = {}
            decision.update({"value_slice": 128, "reason": "official"})
        else:
            sys.path.insert(0, str(ROOT / "kda-zero-state-20260905/clean_build/lib"))
            module = importlib.import_module("flash_kda_phase1_C")
            sys.modules["flash_kda_C"] = module
            sys.path.insert(0, str(ROOT / "kda-zero-state-20260905/clean_source"))
            wrapper = importlib.import_module("flash_kda")
            forced = int(args.variant.removeprefix("hmma_v"))
            os.environ["FLASH_KDA_K2_VALUE_SLICE"] = str(forced)
            decision.update(wrapper.explain_k2_dispatch(q, initial, final, cu_seqlens))
            extra = {"k2_value_slice": forced}
        module_path = module.__file__
        identity_modules = ["torch", module.__name__]
        if args.variant != "official":
            identity_modules.append("flash_kda")
        identity_dependencies = tuple(
            path
            for name in identity_modules[1:]
            if (path := getattr(importlib.import_module(name), "__file__", None))
        )
        workspace = torch.empty(
            module.get_workspace_size(total, heads, sequences),
            dtype=torch.uint8,
            device="cuda",
        )

        def invoke(state):
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
                initial_state=state,
                final_state=final,
                cu_seqlens=cu_seqlens,
                **extra,
            )
            state.copy_(final)

    probe_state = initial.clone()
    invoke(probe_state)
    torch.cuda.synchronize()
    if "observed_modules" in locals():
        decision["observed_modules"] = observed_modules
        decision["route_identity_passed"] = (
            args.variant == "cake"
            or any(item["kind"] == "evolution" for item in observed_modules)
        )
        evo.load_flash_kda_evolution_module = original_evo_load
        prefill._get_flash_kda_prefill_module = original_cake_load

    decision["execution_identity"] = _write_execution_receipt(
        args, identity_modules, identity_dependencies
    )

    actual = {"output": output.cpu(), "state": probe_state.cpu()}
    reference_path = args.output / f"{args.case}_official.pt"
    if args.variant == "official" and args.block == 0:
        torch.save(actual, reference_path)
    reference = torch.load(reference_path, weights_only=True)
    checks = {}
    for key, value in actual.items():
        peer = reference[key]
        delta = value.float() - peer.float()
        rel = float(
            torch.linalg.vector_norm(delta)
            / torch.linalg.vector_norm(peer.float()).clamp_min(1e-30)
        )
        nlinf = float(delta.abs().max() / peer.float().abs().max().clamp_min(1e-30))
        finite = bool(torch.isfinite(value).all())
        historical = finite and bool(
            torch.isclose(value, peer, atol=0.01, rtol=0.01).all()
        )
        checks[key] = {
            "finite": finite,
            "bitwise": torch.equal(value, peer),
            "relative_l2": rel,
            "normalized_linf": nlinf,
            "max_abs": float(delta.abs().max()),
            "historical_atol_rtol_1e2_passed": historical,
            "passed": historical,
        }
    result = {
        "case": args.case,
        "variant": args.variant,
        "block": args.block,
        "decision": decision,
        "module_path": module_path,
        "correctness": checks,
        "samples_ms": [],
        "median_ms": None,
    }
    identity_passed = decision.get("route_identity_passed", True)
    if not identity_passed or not all(item["passed"] for item in checks.values()):
        result["status"] = (
            "route_identity_failed_not_timed"
            if not identity_passed
            else "correctness_failed_not_timed"
        )
        (args.output / f"{args.case}_{args.variant}_{args.block}.json").write_text(
            json.dumps(result, indent=2)
        )
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
            "status": "measured",
            "samples_ms": samples,
            "median_ms": statistics.median(samples),
            "calls": cursor,
            "device": torch.cuda.get_device_name(),
            "capability": torch.cuda.get_device_capability(),
        }
    )
    (args.output / f"{args.case}_{args.variant}_{args.block}.json").write_text(
        json.dumps(result, indent=2)
    )


def controller(args):
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    selected_cases = args.case or list(CASES)
    for case_name in selected_cases:
        variants = CASE_VARIANTS[case_name]
        for block in range(args.blocks):
            order = variants[block % len(variants) :] + variants[: block % len(variants)]
            if block == 0:
                order = ("official",) + tuple(v for v in order if v != "official")
            for variant in order:
                command = [
                    sys.executable,
                    __file__,
                    "--worker",
                    "--case",
                    case_name,
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
                    "--output",
                    str(args.output),
                ]
                stem = f"{case_name}_{variant}_{block}"
                with (args.output / f"{stem}.log").open("w") as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                row = json.loads((args.output / f"{stem}.json").read_text())
                rows.append(row)
                print(stem, row["median_ms"], row["decision"], row["status"], flush=True)

    summaries = []
    for case_name in selected_cases:
        medians = {}
        for variant in CASE_VARIANTS[case_name]:
            values = [
                row["median_ms"]
                for row in rows
                if row["case"] == case_name and row["variant"] == variant
            ]
            medians[variant] = (
                statistics.median(values)
                if values and all(value is not None for value in values)
                else None
            )
        if case_name == "h12_packed_balanced3":
            incumbent, challenger = "hmma_v128", "hmma_v32"
        elif case_name in {
            "h12_packed_balanced3_qual",
            "h12_packed_balanced4",
            "h12_packed_skew3",
        }:
            incumbent, challenger = "hmma_v64", "hmma_v32"
        elif case_name.startswith(("h12", "h96_fixed")):
            incumbent, challenger = "hmma_v128", "hmma_v64"
        elif case_name == "h96_packed_mixed_grid147":
            incumbent, challenger = "evolution_incumbent", "evolution_grid147"
        elif case_name == "h96_packed_mixed_tail4":
            incumbent, challenger = "evolution_incumbent", "evolution_tail4"
        elif case_name == "h96_packed_mixed_minimax172":
            incumbent, challenger = "evolution_incumbent", "evolution_minimax172"
        elif case_name == "h96_packed_mixed_max169":
            incumbent, challenger = "evolution_incumbent", "evolution_max169"
        elif case_name == "h96_packed_mixed_reverse":
            incumbent, challenger = "evolution_incumbent", "evolution_reverse_tasks"
        else:
            incumbent, challenger = "evolution_incumbent", "evolution_virtual152"
        speedup = (
            medians[incumbent] / medians[challenger]
            if medians[incumbent] is not None and medians[challenger] is not None
            else None
        )
        summaries.append(
            {
                "case": case_name,
                "median_ms": medians,
                "incumbent": incumbent,
                "challenger": challenger,
                "challenger_speedup": speedup,
            }
        )
    payload = {
        "schema_version": 1,
        "scope": "public-full state-updating call; one implementation per worker",
        "correctness_contract": "official-peer torch.isclose atol=rtol=0.01 on output and final state",
        "measurement": {
            "timing": "CUPTI cold-L2",
            "cuda_graph": False,
            "balanced_cyclic_order": True,
            "blocks": args.blocks,
            "dry_run_iters": args.dry_run_iters,
            "repeat_iters": args.repeat_iters,
            "job_id": os.getenv("SLURM_JOB_ID"),
        },
        "summary": summaries,
        "results": rows,
    }
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--case", action="append", choices=tuple(CASES))
    parser.add_argument("--variant", choices=tuple(sorted({v for vs in CASE_VARIANTS.values() for v in vs})))
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--dry-run-iters", type=int, default=8)
    parser.add_argument("--repeat-iters", type=int, default=20)
    parser.add_argument("--state-rotations", type=int, default=48)
    parser.add_argument("--output", type=Path, required=True)
    parsed = parser.parse_args()
    if parsed.worker:
        if len(parsed.case or ()) != 1 or parsed.variant is None:
            parser.error("worker requires one --case and --variant")
        parsed.case = parsed.case[0]
        worker(parsed)
    else:
        controller(parsed)
