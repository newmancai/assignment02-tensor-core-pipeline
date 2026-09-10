"""Instantiate verified construction plans with the optional CUTLASS TS runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .verify import Diagnostic


_GROUP_KINDS = {
    "create_async_async_pipeline_cfg": ("async", "async"),
    "create_tma_async_pipeline_cfg": ("hardware", "async"),
    "create_tma_umma_pipeline_cfg": ("hardware", "hardware"),
    "create_umma_async_pipeline_cfg": ("hardware", "async"),
    "create_async_umma_pipeline_cfg": ("async", "hardware"),
    "create_umma_umma_pipeline_cfg": ("hardware", "hardware"),
}


@dataclass(frozen=True)
class CutlassTsCodegenCapabilities:
    """Compiler/runtime features required by executable custom TS work.

    These are separate from schedule legality. A plan can be structurally
    valid while the installed DSL cannot preserve its typed objects across
    staged warp dispatch.
    """

    preserves_frozen_types_in_staged_control_flow: bool
    statically_elides_unconditional_pipeline_gates: bool


def custom_work_codegen_diagnostics(
    capabilities: CutlassTsCodegenCapabilities,
) -> tuple[Diagnostic, ...]:
    """Reject a known-broken CUTLASS TS custom-work codegen environment."""

    diagnostics = []
    if not capabilities.preserves_frozen_types_in_staged_control_flow:
        diagnostics.append(
            Diagnostic(
                "KIR710",
                "staged control flow erases frozen pipeline types before custom work",
            )
        )
    if not capabilities.statically_elides_unconditional_pipeline_gates:
        diagnostics.append(
            Diagnostic(
                "KIR711",
                "unconditional pipeline gates lower to runtime branches carrying meta state",
            )
        )
    return tuple(diagnostics)


def pipeline_runtime_specs(plan: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    task_threads = {
        task["name"]: int(task["num_warps"]) * 32 for task in plan["tasks"]
    }
    specs = []
    for item in plan["pipelines"]:
        factory = item["factory"]
        producer_kind, consumer_kind = _GROUP_KINDS[factory]
        producer_threads = sum(
            task_threads.get(name, 0) for name in item["producer_group"]
        )
        consumer_threads = sum(
            task_threads.get(name, 0) for name in item["consumer_group"]
        )
        specs.append(
            {
                **item,
                "producer_group_size": item.get(
                    "producer_group_size",
                    1 if producer_kind == "hardware" else producer_threads,
                ),
                "consumer_group_size": item.get(
                    "consumer_group_size",
                    1 if consumer_kind == "hardware" else consumer_threads,
                ),
            }
        )
    return tuple(specs)


def instantiate_pipeline_configs(plan: dict[str, Any]) -> dict[str, object]:
    """Build real CooperativeGroup and PipelineConfig objects.

    Import is deliberately lazy: the agent-facing IR and verifier remain usable
    without accepting the NVIDIA EULA runtime dependency.
    """

    from cutlass import pipeline  # type: ignore[import-not-found]
    from cutlass.experimental.task_scheduling.enums import (  # type: ignore[import-not-found]
        SignalingThreads,
    )
    from cutlass.experimental.task_scheduling.resources import (  # type: ignore[import-not-found]
        PipelineConfig,
    )

    signaling = {
        "All": SignalingThreads.All,
        "CtaLeader": SignalingThreads.CtaLeader,
        "TaskWarpLeader": SignalingThreads.TaskWarpLeader,
        "CtaLeader|TaskWarpLeader": SignalingThreads.CtaLeader
        | SignalingThreads.TaskWarpLeader,
    }
    configs = {}
    for spec in pipeline_runtime_specs(plan):
        producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, spec["producer_group_size"]
        )
        consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, spec["consumer_group_size"]
        )
        kwargs = {
            "num_stages": spec["num_stages"],
            "producer_group": producer_group,
            "consumer_group": consumer_group,
            "cta_layout_vmnk": tuple(spec["cta_layout_vmnk"]),
            "producer_signaling_threads": signaling[
                spec["producer_signaling_threads"]
            ],
            "consumer_signaling_threads": signaling[
                spec["consumer_signaling_threads"]
            ],
        }
        if spec["factory"] in {
            "create_tma_async_pipeline_cfg",
            "create_tma_umma_pipeline_cfg",
        }:
            kwargs["num_bytes"] = spec["num_bytes"]
        factory = getattr(PipelineConfig, spec["factory"])
        configs[spec["resource"]] = factory(**kwargs)
    return configs
