from __future__ import annotations

import argparse
import json
from pathlib import Path

from kda_ir import (
    B300,
    cutlass_task_scheduling_plan,
    qk_first_edge_schedule,
    recover_qk_first_edge,
    verify,
)
from kda_ir.flashinfer_catalog import load_flashinfer_catalog
from kda_ir.frozen_source import parse_frozen_source, check_source_variant_contract


parser = argparse.ArgumentParser()
parser.add_argument("source")
parser.add_argument("metadata")
parser.add_argument("--variant-id")
parser.add_argument("--output", type=Path)
args = parser.parse_args()

source = parse_frozen_source(args.source)
catalog = load_flashinfer_catalog(args.metadata)
if args.variant_id:
    matches = [item for item in catalog.variants if item.variant_id == args.variant_id]
else:
    basename = Path(args.source).name
    matches = [item for item in catalog.variants if Path(item.body_relpath).name == basename]
    if not matches:
        matches = [item for item in catalog.variants if item.body_sha256 == source.source_sha256]
if not matches:
    raise SystemExit("no metadata variant references this frozen body")
diagnostics = {
    variant.variant_id: [
        str(item) for item in check_source_variant_contract(source, variant)
    ]
    for variant in matches
}
qk_edge, qk_diagnostics = recover_qk_first_edge(source)
result = {
    "schema_version": 1,
    "analysis": "source_to_ir_qk_first_edge",
    "variant_ids": [variant.variant_id for variant in matches],
    "kernel_name": source.kernel_name,
    "resources": {
        "threads": source.launch_threads,
        "smem_bytes": source.smem_bytes,
        "tmem_columns": source.tmem_columns,
    },
    "launch": {
        "chunk_tokens": source.chunk_tokens,
        "value_rows": source.value_rows,
    },
    "roles": [
        {"name": role.name, "warps": [role.first_warp, role.last_warp]}
        for role in source.roles
    ],
    "pipelines": sorted({barrier.pipeline for barrier in source.barriers}),
    "barrier_groups": len(source.barriers),
    "barriers": sum(item.stages for item in source.barriers),
    "metadata_diagnostics": diagnostics,
    "qk_first_edge_diagnostics": [str(item) for item in qk_diagnostics],
}
if qk_edge is not None:
    source_schedule = qk_first_edge_schedule(source, qk_edge)
    schedule_diagnostics = verify(source_schedule, B300)
    task_plan = (
        cutlass_task_scheduling_plan(source_schedule, B300)
        if not schedule_diagnostics
        else None
    )
    result["qk_first_edge"] = {
        "source_sha256": qk_edge.source_sha256,
        "stages": qk_edge.stages,
        "producer": qk_edge.ready_producer,
        "consumers": list(qk_edge.ready_consumers),
        "ready": {
            "mode": qk_edge.ready_arrival.value,
            "init_count": qk_edge.ready_init_count,
        },
        "completion_arrivers": list(qk_edge.free_arrivers),
        "reuse_waiter": qk_edge.reuse_waiter,
        "free": {
            "barrier": qk_edge.free_barrier,
            "mode": qk_edge.free_arrival.value,
            "init_count": qk_edge.free_init_count,
        },
        "completion_dominance": [
            {
                "consumer": proof.consumer,
                "barrier_path": list(proof.barrier_path),
                "source_lines": list(proof.source_lines),
            }
            for proof in qk_edge.completion_proofs
        ],
        "schedule_diagnostics": [str(item) for item in schedule_diagnostics],
        "required_backend_primitives": (
            task_plan["split_phase_lowering_analysis"][0][
                "required_backend_primitives"
            ]
            if task_plan is not None
            else []
        ),
        "role_ranges": task_plan["role_ranges"] if task_plan is not None else {},
    }
rendered = json.dumps(result, indent=2)
if args.output is not None:
    args.output.write_text(rendered + "\n")
print(rendered)
