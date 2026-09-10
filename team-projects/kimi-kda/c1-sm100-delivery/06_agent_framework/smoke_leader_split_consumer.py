"""Compile/run the exact elect-one split-consumer protocol on CUTLASS DSL.

Unlike ``smoke_split_consumer.py``, this test initializes both phase barriers
with count one and elects exactly one producer/recycler lane to arrive.  Four
compute warps remain wait-only consumers.  The implementation uses public
CUTLASS pipeline primitives directly while the Task Scheduling integration is
kept behind ``LeaderArrivalSplitConsumerPlan``.
"""

from __future__ import annotations

import argparse

import torch
import cutlass
import cutlass.cute as cute
from cutlass import pipeline
from cuda.bindings import driver as cuda_drv


STAGES = 5
ITERATIONS = STAGES * 2
PRODUCER_WARPS = 4
COMPUTE_WARPS = 4
RECYCLER_WARPS = 1
TOTAL_WARPS = PRODUCER_WARPS + COMPUTE_WARPS + RECYCLER_WARPS
PRODUCER_THREADS = PRODUCER_WARPS * 32
COMPUTE_THREADS = COMPUTE_WARPS * 32
MARKER_VALUES = TOTAL_WARPS * 32


class LeaderArrivalSplitConsumerSmoke:
    @cute.jit
    def __call__(self, marker: cute.Tensor, stream: cuda_drv.CUstream) -> None:
        self.kernel(marker).launch(
            grid=(1, 1, 1),
            block=(MARKER_VALUES, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(self, marker: cute.Tensor) -> None:
        barrier_storage = cutlass.Array(
            cutlass.Int64,
            STAGES * 2,
            space=cutlass.AddressSpace.smem,
            alignment=8,
        )
        barrier_ptr = cute.make_ptr(
            cutlass.Int64,
            barrier_storage.data_ptr(),
            mem_space=cutlass.AddressSpace.smem,
        )
        qk_stage_values = cutlass.Array(
            cutlass.Int32,
            STAGES * PRODUCER_THREADS,
            space=cutlass.AddressSpace.smem,
            alignment=16,
        )
        compute_probe = cutlass.Array(
            cutlass.Int32,
            ITERATIONS * COMPUTE_THREADS,
            space=cutlass.AddressSpace.smem,
            alignment=16,
        )
        qk_full = pipeline.PipelineAsync.create(
            num_stages=STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            barrier_storage=barrier_ptr,
            defer_sync=True,
        )
        producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, STAGES
        )
        compute_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, STAGES
        )
        recycler_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, STAGES
        )
        producer_done = pipeline.NamedBarrier(
            barrier_id=1, num_threads=PRODUCER_THREADS
        )
        consumers_done = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=(COMPUTE_WARPS + RECYCLER_WARPS) * 32,
        )

        cute.arch.mbarrier_init_fence()
        pipeline.agent_sync(pipeline.Agent.ThreadBlock)

        thread = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp < PRODUCER_WARPS:
            for step in range(ITERATIONS):
                stage = step % STAGES
                qk_full.producer_acquire(producer_state)
                qk_stage_values[stage * PRODUCER_THREADS + thread] = step + 1
                producer_done.arrive_and_wait()
                if warp == 0:
                    with cute.arch.elect_one():
                        qk_full.producer_commit(producer_state)
                producer_state.advance()
        elif warp < PRODUCER_WARPS + COMPUTE_WARPS:
            compute_thread = thread - PRODUCER_THREADS
            for step in range(ITERATIONS):
                stage = step % STAGES
                qk_full.consumer_wait(compute_state)
                compute_probe[step * COMPUTE_THREADS + compute_thread] = (
                    qk_stage_values[stage * PRODUCER_THREADS + compute_thread]
                )
                consumers_done.arrive_and_wait()
                compute_state.advance()
        else:
            for _ in range(ITERATIONS):
                qk_full.consumer_wait(recycler_state)
                consumers_done.arrive_and_wait()
                with cute.arch.elect_one():
                    qk_full.consumer_release(recycler_state)
                recycler_state.advance()

        pipeline.agent_sync(pipeline.Agent.ThreadBlock)
        probe_thread = thread % COMPUTE_THREADS
        observed_sum = cutlass.Int32(0)
        for step in range(ITERATIONS):
            observed_sum += compute_probe[step * COMPUTE_THREADS + probe_thread]
        marker[thread] = observed_sum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug-dynamic", action="store_true")
    args = parser.parse_args()
    if args.debug_dynamic:
        from cutlass.base_dsl.dsl import set_dynamic_debug

        set_dynamic_debug(True, max_depth=4)

    marker = torch.zeros(MARKER_VALUES, dtype=torch.int32, device="cuda")
    marker_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (MARKER_VALUES,), stride_order=(0,), assumed_align=4
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    compiled = cute.compile(
        LeaderArrivalSplitConsumerSmoke(),
        marker_fake,
        stream_fake,
        options="--enable-tvm-ffi --opt-level 3",
    )
    compiled(marker)
    torch.cuda.synchronize()
    observed = sorted(set(int(value) for value in marker.cpu().tolist()))
    print(
        "executed exact-elect-one",
        f"marker_values={observed}",
        f"count={marker.numel()}",
    )
    expected = ITERATIONS * (ITERATIONS + 1) // 2
    if observed != [expected]:
        raise RuntimeError(f"leader split-consumer visibility failed: {observed}")


if __name__ == "__main__":
    main()
