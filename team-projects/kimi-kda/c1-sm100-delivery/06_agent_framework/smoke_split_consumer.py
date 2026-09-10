"""Compile/run the native cooperative-arrival qk_full topology on CUTLASS TS.

This is intentionally an EULA-bound runtime smoke test, not part of the
CPU-only agent IR.  It proves the split-consumer topology before any KDA math
is moved into CuTe DSL.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass import pipeline
from cutlass.experimental.task_scheduling import domain_loop, schedule
from cutlass.experimental.task_scheduling.memory import SmemAllocation, SmemAllocator
from cutlass.experimental.task_scheduling.resources import (
    MemoryResource,
    PipelineConfig,
    StageInfo,
)
from cutlass.experimental.task_scheduling.task import Task
from cutlass.experimental.task_scheduling.task_manager import TaskManager
from cuda.bindings import driver as cuda_drv


STAGES = 5
PRODUCER_WARPS = 4
COMPUTE_WARPS = 4
RECYCLER_WARPS = 1
TOTAL_WARPS = PRODUCER_WARPS + COMPUTE_WARPS + RECYCLER_WARPS
PRODUCER_THREADS = PRODUCER_WARPS * 32
COMPUTE_THREADS = COMPUTE_WARPS * 32
RECYCLER_THREADS = RECYCLER_WARPS * 32
MARKER_VALUES = TOTAL_WARPS * 32


@dataclass(kw_only=True)
class NoWorkResource(MemoryResource):
    """Concrete captured-schedule resource with inherited no-op hooks."""

    pass


@dataclass(kw_only=True)
class StageValueResource(MemoryResource):
    _alloc: cutlass.Constexpr[Optional[SmemAllocation]] = None

    def get_smem_requirements(self) -> list[SmemAllocation]:
        if self._alloc is None:
            self._alloc = SmemAllocation(
                "qk_stage_values", STAGES * PRODUCER_THREADS * 4, alignment=16
            )
        return [self._alloc]

    @cute.jit
    def producer_work(self, stage_info: StageInfo) -> None:
        payload = cutlass.Array(
            stage_info.context.smem_base.data_ptr() + self._alloc.offset,
            dtype=cutlass.Int32,
            shape=(STAGES * PRODUCER_THREADS,),
            addrspace=3,
        )
        thread = cute.arch.thread_idx()[0]
        offset = stage_info.stage_idx * PRODUCER_THREADS + thread
        payload[offset] = cutlass.Int32(stage_info.loop_offset) + 1


@dataclass(kw_only=True)
class ComputeProbeResource(MemoryResource):
    smem_offset: cutlass.Constexpr[int] = 0
    _alloc: cutlass.Constexpr[Optional[SmemAllocation]] = None

    def get_smem_requirements(self) -> list[SmemAllocation]:
        if self._alloc is None:
            self._alloc = SmemAllocation(
                "compute_probe_values", STAGES * COMPUTE_THREADS * 4, alignment=16
            )
        return [self._alloc]

    @cute.jit
    def producer_work(self, stage_info: StageInfo) -> None:
        payload = cutlass.Array(
            stage_info.context.smem_base.data_ptr() + self.smem_offset,
            dtype=cutlass.Int32,
            shape=(STAGES * PRODUCER_THREADS,),
            addrspace=3,
        )
        thread = cute.arch.thread_idx()[0] - PRODUCER_THREADS
        # The probe itself is not pipelined, so its StageInfo has no physical
        # stage index.  This smoke executes exactly one pass over all five qk
        # stages; the logical loop offset is therefore the qk physical stage.
        source_offset = stage_info.loop_offset * PRODUCER_THREADS
        probe = cutlass.Array(
            stage_info.context.smem_base.data_ptr() + self._alloc.offset,
            dtype=cutlass.Int32,
            shape=(STAGES * COMPUTE_THREADS,),
            addrspace=3,
        )
        probe_offset = stage_info.loop_offset * COMPUTE_THREADS + thread
        probe[probe_offset] = payload[source_offset]


def build_cooperative_manager(
    *, verbose: bool = False
) -> tuple[TaskManager, ComputeProbeResource]:
    cfg = PipelineConfig.create_async_async_pipeline_cfg(
        num_stages=STAGES,
        producer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread, PRODUCER_WARPS * 32
        ),
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread, RECYCLER_WARPS * 32
        ),
        cta_layout_vmnk=(1, 1, 1, 1),
    )
    source = NoWorkResource(name="source", is_barrier=True)
    qk_full = StageValueResource(
        name="qk_full_cooperative",
        pipeline_config=cfg,
    )
    sink = NoWorkResource(name="sink", is_barrier=True)

    smem_allocator = SmemAllocator()
    smem_allocator.add_resource(qk_full)
    compute_probe = ComputeProbeResource(
        smem_offset=0,
        name="compute_probe",
    )
    smem_allocator.add_resource(compute_probe)
    smem_allocator.compute_layout()
    assert qk_full._alloc.offset == 0

    @schedule
    def producer_schedule(src: MemoryResource, qk: MemoryResource) -> None:
        with domain_loop(0, STAGES, 1):
            src.consumer_work()
            qk.acquire()
            qk.producer_work()
            qk.commit()

    @schedule
    def compute_schedule(qk: MemoryResource, probe: MemoryResource) -> None:
        with domain_loop(0, STAGES, 1):
            qk.wait()
            probe.producer_work()

    @schedule
    def recycler_schedule(qk: MemoryResource, dst: MemoryResource) -> None:
        with domain_loop(0, STAGES, 1):
            qk.wait()
            qk.consumer_work()
            qk.release()
            dst.producer_work()

    tasks = [
        Task(
            src_resources=[source],
            dst_resources=[qk_full],
            warp_idx=0,
            num_warps=PRODUCER_WARPS,
            schedule=producer_schedule(source, qk_full),
            name="PrepProducer",
        ),
        Task(
            src_resources=[qk_full],
            dst_resources=[compute_probe],
            warp_idx=PRODUCER_WARPS,
            num_warps=COMPUTE_WARPS,
            schedule=compute_schedule(qk_full, compute_probe),
            name="ComputeWaiter",
        ),
        Task(
            src_resources=[qk_full],
            dst_resources=[sink],
            warp_idx=PRODUCER_WARPS + COMPUTE_WARPS,
            num_warps=RECYCLER_WARPS,
            schedule=recycler_schedule(qk_full, sink),
            name="MmaRecycler",
        ),
    ]
    manager = TaskManager(
        tasks=tasks,
        resource_dependency_graph={
            qk_full: [source],
            compute_probe: [qk_full],
            sink: [qk_full],
        },
        smem_allocator=smem_allocator,
        verbose=verbose,
        exhaustive_deadlock_race_check=True,
    )
    return manager, compute_probe


class CooperativeSplitConsumerSmoke:
    @cute.jit
    def __call__(self, marker: cute.Tensor, stream: cuda_drv.CUstream) -> None:
        self.kernel(marker).launch(
            grid=(1, 1, 1),
            block=(TOTAL_WARPS * 32, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(self, marker: cute.Tensor) -> None:
        manager, compute_probe = build_cooperative_manager(verbose=False)
        manager.setup_resources_and_tasks()
        cute.arch.mbarrier_init_fence()
        pipeline.agent_sync(pipeline.Agent.ThreadBlock)
        manager.run()
        pipeline.agent_sync(pipeline.Agent.ThreadBlock)

        thread = cute.arch.thread_idx()[0]
        probe_thread = thread % COMPUTE_THREADS
        probe = cutlass.Array(
            manager.smem_allocator.smem_base.data_ptr()
            + compute_probe._alloc.offset,
            dtype=cutlass.Int32,
            shape=(STAGES * COMPUTE_THREADS,),
            addrspace=3,
        )
        marker[thread] = (
            probe[probe_thread]
            + probe[COMPUTE_THREADS + probe_thread]
            + probe[2 * COMPUTE_THREADS + probe_thread]
            + probe[3 * COMPUTE_THREADS + probe_thread]
            + probe[4 * COMPUTE_THREADS + probe_thread]
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--debug-dynamic", action="store_true")
    args = parser.parse_args()
    manager, _ = build_cooperative_manager(verbose=True)
    cfg = next(
        resource.pipeline_config
        for resource in manager.resources
        if resource.name == "qk_full_cooperative"
    )
    print(
        "validated",
        f"advance_on_wait={cfg.advance_on_wait}",
        f"producer_group={cfg.producer_group.size}",
        f"consumer_group={cfg.consumer_group.size}",
    )
    if args.validate_only:
        return

    if args.debug_dynamic:
        from cutlass.base_dsl.dsl import set_dynamic_debug

        set_dynamic_debug(True, max_depth=4)

    import torch

    marker = torch.zeros(MARKER_VALUES, dtype=torch.int32, device="cuda")
    marker_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (MARKER_VALUES,), stride_order=(0,), assumed_align=4
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    compiled = cute.compile(
        CooperativeSplitConsumerSmoke(),
        marker_fake,
        stream_fake,
        options="--enable-tvm-ffi --opt-level 3",
    )
    compiled(marker)
    torch.cuda.synchronize()
    observed = sorted(set(int(value) for value in marker.cpu().tolist()))
    print("executed", f"marker_values={observed}", f"count={marker.numel()}")
    if observed != [15]:
        raise RuntimeError(f"split-consumer visibility failed: {observed}")


if __name__ == "__main__":
    main()
