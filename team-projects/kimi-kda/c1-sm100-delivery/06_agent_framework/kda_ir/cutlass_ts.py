"""Lower KDA IR into an inspectable CUTLASS Task Scheduling construction plan."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass

from .model import ArrivalMode, MemorySpace, PipelineKind, Schedule, Target
from .verify import require_valid


_PIPELINE_FACTORIES = {
    PipelineKind.ASYNC_ASYNC: "create_async_async_pipeline_cfg",
    PipelineKind.TMA_ASYNC: "create_tma_async_pipeline_cfg",
    PipelineKind.TMA_UMMA: "create_tma_umma_pipeline_cfg",
    PipelineKind.UMMA_ASYNC: "create_umma_async_pipeline_cfg",
    PipelineKind.ASYNC_UMMA: "create_async_umma_pipeline_cfg",
    PipelineKind.UMMA_UMMA: "create_umma_umma_pipeline_cfg",
}


@dataclass(frozen=True)
class LeaderArrivalSplitConsumerPlan:
    """Typed ABI contract for the backend primitive missing from native TS."""

    resource: str
    stages: int
    producer: str
    waiters: tuple[str, ...]
    reuse_waiter: str
    ready_init_count: int
    free_init_count: int
    ready_arrival: ArrivalMode
    free_arrival: ArrivalMode
    completion_join: str | None
    completion_participants: tuple[str, ...]
    completion_waiter: str | None
    completion_arrival_count: int
    completion_arrival_mode: ArrivalMode | None
    waiters_advance_without_release: bool = True


def leader_arrival_split_consumer_plans(
    schedule: Schedule,
) -> tuple[LeaderArrivalSplitConsumerPlan, ...]:
    """Return exact primitive contracts for fanouts native TS cannot preserve."""

    role_threads = {role.name: role.warps * 32 for role in schedule.roles}
    completion_joins = {join.name: join for join in schedule.completion_joins}
    plans = []
    for fanout in schedule.fanouts:
        join = completion_joins.get(fanout.completion_join)
        native_free = sum(
            role_threads.get(name, 0)
            for name in (join.participants if join else (fanout.reuse_waiter,))
        )
        native_exact = (
            fanout.ready_arrival is ArrivalMode.COOPERATIVE
            and fanout.free_arrival is ArrivalMode.COOPERATIVE
            and fanout.ready_init_count == role_threads.get(fanout.producer, 0)
            and fanout.free_init_count == native_free
        )
        if not native_exact:
            plans.append(
                LeaderArrivalSplitConsumerPlan(
                    resource=fanout.name,
                    stages=fanout.stages,
                    producer=fanout.producer,
                    waiters=fanout.consumers,
                    reuse_waiter=fanout.reuse_waiter,
                    ready_init_count=fanout.ready_init_count,
                    free_init_count=fanout.free_init_count,
                    ready_arrival=fanout.ready_arrival,
                    free_arrival=fanout.free_arrival,
                    completion_join=join.name if join else None,
                    completion_participants=join.participants if join else (),
                    completion_waiter=join.waiter if join else None,
                    completion_arrival_count=join.arrival_count if join else 0,
                    completion_arrival_mode=join.arrival_mode if join else None,
                )
            )
    return tuple(plans)


def split_phase_lowering_analysis(schedule: Schedule) -> tuple[dict[str, object], ...]:
    """Classify native arrival support separately from completion dominance.

    Task Scheduling supports wait-only and release-only tasks via
    ``advance_on_wait``.  Its ordinary AsyncAsync wrapper still commits and
    releases with whole cooperative groups, so an elect-one frozen protocol
    needs a leader-arrival backend primitive for exact lowering.
    """

    role_threads = {role.name: role.warps * 32 for role in schedule.roles}
    completion_joins = {join.name: join for join in schedule.completion_joins}
    out = []
    for fanout in schedule.fanouts:
        native_ready = role_threads.get(fanout.producer, 0)
        join = completion_joins.get(fanout.completion_join)
        native_free = sum(
            role_threads.get(name, 0)
            for name in (join.participants if join else (fanout.reuse_waiter,))
        )
        native_arrivals_exact = (
            fanout.ready_arrival is ArrivalMode.COOPERATIVE
            and fanout.free_arrival is ArrivalMode.COOPERATIVE
            and fanout.ready_init_count == native_ready
            and fanout.free_init_count == native_free
        )
        needs_completion_join = fanout.completion_join is not None
        required_primitives = []
        if not native_arrivals_exact:
            required_primitives.append("LeaderArrivalSplitConsumer")
        if needs_completion_join:
            required_primitives.append("CompletionJoin")
        out.append(
            {
                "resource": fanout.name,
                "split_consumer_support": "advance_on_wait",
                "native_arrival_counts_exact": native_arrivals_exact,
                "exact_native_async_async": not required_primitives,
                "frozen_arrival_counts": {
                    "ready": fanout.ready_init_count,
                    "free": fanout.free_init_count,
                },
                "arrival_modes": {
                    "ready": fanout.ready_arrival.value,
                    "free": fanout.free_arrival.value,
                },
                "native_cooperative_arrival_counts": {
                    "ready": native_ready,
                    "free": native_free,
                },
                "completion_join": fanout.completion_join,
                "required_backend_primitives": required_primitives,
                "searchable_native_rewrite": not native_arrivals_exact,
            }
        )
    return tuple(out)


def cutlass_task_scheduling_plan(
    schedule: Schedule, target: Target
) -> dict[str, object]:
    """Describe exact upstream objects to build without importing CUTLASS locally."""

    require_valid(schedule, target)
    role_ranges: dict[str, dict[str, int]] = {}
    next_warp = 0
    for role in schedule.roles:
        warp_idx = role.first_warp if role.first_warp is not None else next_warp
        role_ranges[role.name] = {"warp_idx": warp_idx, "num_warps": role.warps}
        next_warp = max(next_warp, warp_idx + role.warps)

    ops_by_role = defaultdict(list)
    reads_by_role = defaultdict(set)
    writes_by_role = defaultdict(set)
    dependency_graph: dict[str, set[str]] = defaultdict(set)
    for op in schedule.pipeline.operations:
        steps = []
        steps.extend({"action": "acquire", "resource": name} for name in op.acquires)
        steps.extend({"action": "wait", "resource": name} for name in op.waits)
        steps.extend(
            {"action": "fanout_wait", "resource": name}
            for name in op.fanout_waits
        )
        steps.extend(
            {"action": "completion_wait", "resource": name}
            for name in op.completion_waits
        )
        steps.append({"action": "work", "operation": op.kind, "name": op.name})
        steps.extend({"action": "commit", "resource": name} for name in op.commits)
        steps.extend({"action": "release", "resource": name} for name in op.releases)
        steps.extend(
            {"action": "publish", "resource": name} for name in op.publishes
        )
        steps.extend(
            {"action": "completion_arrive", "resource": name}
            for name in op.completion_arrivals
        )
        steps.extend(
            {"action": "authorize_reuse", "resource": name}
            for name in op.recycles
        )
        ops_by_role[op.role].extend(steps)
        reads_by_role[op.role].update(op.reads)
        writes_by_role[op.role].update(op.writes)
        for dst in op.writes:
            dependency_graph[dst].update(src for src in op.reads if src != dst)

    tasks = []
    for role in schedule.roles:
        assignment = role_ranges[role.name]
        tasks.append(
            {
                "name": role.name,
                "warp_idx": assignment["warp_idx"],
                "num_warps": assignment["num_warps"],
                "src_resources": sorted(reads_by_role[role.name]),
                "dst_resources": sorted(writes_by_role[role.name]),
                "schedule": ops_by_role[role.name],
            }
        )

    return {
        "api": "cutlass.experimental.task_scheduling",
        "target": target.arch,
        "role_ranges": role_ranges,
        "resources": [
            {
                "name": buffer.name,
                "kind": (
                    "SmemAllocation"
                    if buffer.space is MemorySpace.SHARED
                    else "TmemAllocation"
                    if buffer.space is MemorySpace.TMEM
                    else "MemoryResource"
                ),
                "space": buffer.space.value,
                "size_bytes": buffer.size_bytes,
                "alignment": buffer.alignment,
                "lifetime": [buffer.live_from, buffer.live_until],
            }
            for buffer in schedule.buffers
        ],
        "pipelines": [
            {
                "resource": barrier.name,
                "factory_owner": "PipelineConfig",
                "factory": _PIPELINE_FACTORIES[barrier.kind],
                "num_stages": barrier.stages,
                "num_bytes": barrier.transaction_bytes,
                "producer_group": list(barrier.producers),
                "consumer_group": list(barrier.consumers),
                "producer_signaling_threads": barrier.producer_signaling.value,
                "consumer_signaling_threads": barrier.consumer_signaling.value,
                "cta_layout_vmnk": list(barrier.cluster_shape),
            }
            for barrier in schedule.barriers
        ],
        "resource_dependency_graph": {
            name: sorted(upstreams) for name, upstreams in dependency_graph.items()
        },
        "tasks": tasks,
        "allocator": {
            "smem_data_bytes": sum(
                buffer.size_bytes
                for buffer in schedule.buffers
                if buffer.space is MemorySpace.SHARED
            ),
            "barrier_smem_bytes": sum(
                barrier.stages * 2 * 8 for barrier in schedule.barriers
            )
            + sum(fanout.stages * 2 * 8 for fanout in schedule.fanouts),
            "tmem_columns": schedule.resources.tmem_columns,
        },
        "unlowered_edges": [
            {
                "resource": fanout.name,
                "required_primitive": "SplitPhaseFanout",
                "producer": fanout.producer,
                "consumers": list(fanout.consumers),
                "reuse_waiter": fanout.reuse_waiter,
                "num_stages": fanout.stages,
            }
            for fanout in schedule.fanouts
        ],
        "split_phase_lowering_analysis": list(
            split_phase_lowering_analysis(schedule)
        ),
        "backend_primitive_contracts": [
            {
                **asdict(contract),
                "ready_arrival": contract.ready_arrival.value,
                "free_arrival": contract.free_arrival.value,
                "waiters": list(contract.waiters),
                "completion_participants": list(contract.completion_participants),
                "completion_arrival_mode": (
                    contract.completion_arrival_mode.value
                    if contract.completion_arrival_mode is not None
                    else None
                ),
            }
            for contract in leader_arrival_split_consumer_plans(schedule)
        ],
        "completion_join_contracts": [
            {
                **asdict(join),
                "participants": list(join.participants),
                "arrival_mode": join.arrival_mode.value,
            }
            for join in schedule.completion_joins
        ],
    }
