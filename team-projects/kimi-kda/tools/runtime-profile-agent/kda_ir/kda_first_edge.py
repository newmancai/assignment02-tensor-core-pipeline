"""Exact first-edge contract for porting frozen FlashKDA into CUTLASS TS.

This module deliberately separates the directly representable TMA hand-off from
the following split-phase fan-out.  Treating both as one ordinary CAKE memory
resource would silently change the frozen physical schedule.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .frozen_source import FrozenSourceContract
from .model import (
    ArrivalMode,
    Buffer,
    CompletionJoin,
    MemorySpace,
    Operation,
    Pipeline,
    RecurrenceKind,
    RecurrencePlan,
    ResourceUse,
    RoleKind,
    Schedule,
    SplitPhaseFanout,
    TileShape,
    WarpRole,
)
from .verify import Diagnostic


@dataclass(frozen=True)
class CompletionDominanceProof:
    consumer: str
    barrier_path: tuple[str, ...]
    source_lines: tuple[int, ...]


@dataclass(frozen=True)
class KdaQkFirstEdge:
    source_sha256: str
    stages: int
    chunk_tokens: int
    q_stage_bytes: int
    k_stage_bytes: int
    stage_stride_bytes: int
    tma_transaction_bytes: int
    prep_warps: int
    logical_prep_instances: int
    warps_per_instance: int
    ready_producer: str
    ready_consumers: tuple[str, ...]
    ready_arrival: ArrivalMode
    ready_init_count: int
    free_arrivers: tuple[str, ...]
    free_arrival: ArrivalMode
    free_init_count: int
    reuse_waiter: str
    completion_proofs: tuple[CompletionDominanceProof, ...]
    free_barrier: str

    @property
    def prep_consumer_threads(self) -> int:
        return self.warps_per_instance * 32


def _roles_using(
    source: FrozenSourceContract, barrier: str, action: str
) -> tuple[str, ...]:
    role_order = {role.name: index for index, role in enumerate(source.roles)}
    return tuple(
        sorted(
            {
                use.role
                for use in source.barrier_uses
                if use.barrier == barrier and use.action == action
            },
            key=role_order.__getitem__,
        )
    )


def _arrival_mode(
    source: FrozenSourceContract,
    barrier: str,
    roles: tuple[str, ...],
    init_count: int,
) -> ArrivalMode | None:
    role_map = {role.name: role for role in source.roles}
    warps = sum(role_map[name].warps for name in roles if name in role_map)
    arrival_uses = [
        use
        for use in source.barrier_uses
        if use.barrier == barrier and use.role in roles and use.action == "arrive"
    ]
    elect_one = bool(arrival_uses) and all(
        use.elect_one
        for use in arrival_uses
    )
    if init_count == 1 and elect_one:
        return ArrivalMode.ELECT_ONE
    if init_count == warps and elect_one:
        return ArrivalMode.ELECT_ONE_PER_WARP
    if init_count == warps * 32:
        return ArrivalMode.COOPERATIVE
    return None


def _completion_proof(
    source: FrozenSourceContract,
    consumer: str,
    ready_barrier: str,
    free_barrier: str,
    reuse_waiter: str,
) -> CompletionDominanceProof | None:
    uses = source.barrier_uses
    edges: dict[int, set[int]] = {index: set() for index in range(len(uses))}
    by_role: dict[str, list[int]] = {}
    for index, use in enumerate(uses):
        by_role.setdefault(use.role, []).append(index)
    for indices in by_role.values():
        ordered = sorted(indices, key=lambda index: uses[index].order)
        for left, right in zip(ordered, ordered[1:]):
            edges[left].add(right)
    barriers = {use.barrier for use in uses}
    for barrier in barriers:
        arrivals = [
            index
            for index, use in enumerate(uses)
            if use.barrier == barrier and use.action == "arrive"
        ]
        waits = [
            index
            for index, use in enumerate(uses)
            if use.barrier == barrier and use.action == "wait"
        ]
        for arrival in arrivals:
            edges[arrival].update(waits)

    starts = [
        index
        for index, use in enumerate(uses)
        if use.role == consumer
        and use.barrier == ready_barrier
        and use.action == "wait"
    ]
    targets = {
        index
        for index, use in enumerate(uses)
        if use.role == reuse_waiter
        and use.barrier == free_barrier
        and use.action == "wait"
    }
    queue = deque((index, (index,)) for index in starts)
    visited = set(starts)
    while queue:
        node, path = queue.popleft()
        if node in targets:
            barrier_path = []
            for index in path:
                barrier = uses[index].barrier
                if not barrier_path or barrier_path[-1] != barrier:
                    barrier_path.append(barrier)
            return CompletionDominanceProof(
                consumer=consumer,
                barrier_path=tuple(barrier_path),
                source_lines=tuple(uses[index].source_line for index in path),
            )
        for successor in edges[node]:
            if successor not in visited:
                visited.add(successor)
                queue.append((successor, (*path, successor)))
    return None


def recover_qk_first_edge(
    source: FrozenSourceContract,
) -> tuple[KdaQkFirstEdge | None, tuple[Diagnostic, ...]]:
    """Recover only facts evidenced by the frozen generated source."""

    diagnostics: list[Diagnostic] = []
    barriers = {item.name: item for item in source.barriers}
    qk_raw = barriers.get("qk_raw_full")
    qk_ready = barriers.get("qk_full")
    smem_free = barriers.get("smem_free")
    if qk_raw is None:
        diagnostics.append(Diagnostic("KIR701", "missing qk_raw_full barrier"))
    if qk_ready is None:
        diagnostics.append(Diagnostic("KIR702", "missing qk_full barrier"))
    if smem_free is None:
        diagnostics.append(Diagnostic("KIR703", "missing split smem_free barrier"))

    q_bytes = source.define("SMEM_SMEM_Q_RAW_PREFETCH_STAGE_BYTES")
    k_bytes = source.define("SMEM_SMEM_KD_STAGE_BYTES")
    stride = source.define("SMEM_SMEM_QD_STRIDE")
    prep = next((role for role in source.roles if role.name == "prep"), None)
    compute = next((role for role in source.roles if role.name == "compute"), None)
    mma = next((role for role in source.roles if role.name == "mma"), None)
    if q_bytes is None or k_bytes is None or stride is None:
        diagnostics.append(Diagnostic("KIR704", "missing Q/K stage allocation defines"))
    if prep is None or prep.warps % 4:
        diagnostics.append(
            Diagnostic("KIR705", "prep role must factor into four-warp stage instances")
        )
    if compute is None or mma is None:
        diagnostics.append(
            Diagnostic("KIR706", "qk_full fan-out requires compute and mma roles")
        )
    if qk_raw and qk_ready and smem_free:
        if not (qk_raw.stages == qk_ready.stages == smem_free.stages):
            diagnostics.append(
                Diagnostic("KIR707", "Q/K ready and free stage counts differ")
            )
        if qk_raw.init_count != 1 or qk_ready.init_count != 1:
            diagnostics.append(
                Diagnostic("KIR708", "Q/K producer arrival contract is not single-threaded")
            )
        ready_producers = _roles_using(source, qk_ready.name, "arrive")
        ready_consumers = _roles_using(source, qk_ready.name, "wait")
        free_arrivers = _roles_using(source, smem_free.name, "arrive")
        free_waiters = _roles_using(source, smem_free.name, "wait")
        if (
            len(ready_producers) != 1
            or not ready_consumers
            or not free_arrivers
            or len(free_waiters) != 1
        ):
            diagnostics.append(
                Diagnostic(
                    "KIR712",
                    "Q/K fan-out source uses do not identify one publisher, consumers, completion arrivals, and one reuse waiter",
                )
            )
        else:
            ready_mode = _arrival_mode(
                source, qk_ready.name, ready_producers, qk_ready.init_count
            )
            free_mode = _arrival_mode(
                source, smem_free.name, free_arrivers, smem_free.init_count
            )
            if ready_mode is None or free_mode is None:
                diagnostics.append(
                    Diagnostic(
                        "KIR713",
                        "Q/K fan-out arrival counts do not match a typed arrival mode",
                    )
                )
            proofs = tuple(
                proof
                for role in ready_consumers
                if (
                    proof := _completion_proof(
                        source,
                        role,
                        qk_ready.name,
                        smem_free.name,
                        free_waiters[0],
                    )
                )
                is not None
            )
            if len(proofs) != len(ready_consumers):
                diagnostics.append(
                    Diagnostic(
                        "KIR714",
                        "not every qk_full consumer is transitively ordered before smem_free reuse",
                    )
                )
    if diagnostics:
        return None, tuple(diagnostics)

    assert qk_raw and qk_ready and smem_free
    assert q_bytes is not None and k_bytes is not None and stride is not None
    assert prep is not None and compute is not None and mma is not None
    assert source.chunk_tokens is not None
    ready_producers = _roles_using(source, qk_ready.name, "arrive")
    ready_consumers = _roles_using(source, qk_ready.name, "wait")
    free_arrivers = _roles_using(source, smem_free.name, "arrive")
    free_waiters = _roles_using(source, smem_free.name, "wait")
    ready_mode = _arrival_mode(
        source, qk_ready.name, ready_producers, qk_ready.init_count
    )
    free_mode = _arrival_mode(
        source, smem_free.name, free_arrivers, smem_free.init_count
    )
    assert ready_mode is not None and free_mode is not None
    proofs = tuple(
        proof
        for role in ready_consumers
        if (
            proof := _completion_proof(
                source, role, qk_ready.name, smem_free.name, free_waiters[0]
            )
        )
        is not None
    )
    logical_instances = prep.warps // 4
    if logical_instances != qk_raw.stages:
        return None, (
            Diagnostic(
                "KIR709", "prep instance count must equal qk_raw_full stage count"
            ),
        )
    return (
        KdaQkFirstEdge(
            source_sha256=source.source_sha256,
            stages=qk_raw.stages,
            chunk_tokens=source.chunk_tokens,
            q_stage_bytes=q_bytes,
            k_stage_bytes=k_bytes,
            stage_stride_bytes=stride,
            tma_transaction_bytes=q_bytes + k_bytes,
            prep_warps=prep.warps,
            logical_prep_instances=logical_instances,
            warps_per_instance=4,
            ready_producer=ready_producers[0],
            ready_consumers=ready_consumers,
            ready_arrival=ready_mode,
            ready_init_count=qk_ready.init_count,
            free_arrivers=free_arrivers,
            free_arrival=free_mode,
            free_init_count=smem_free.init_count,
            reuse_waiter=free_waiters[0],
            completion_proofs=proofs,
            free_barrier=smem_free.name,
        ),
        (),
    )


def _role_kind(name: str) -> RoleKind:
    if "mma" in name:
        return RoleKind.MMA
    if name in {"prep", "beta_prefetch"}:
        return RoleKind.LOAD
    if name == "epilogue":
        return RoleKind.STORE
    return RoleKind.RECURRENCE

def qk_first_edge_schedule(
    source: FrozenSourceContract, edge: KdaQkFirstEdge
) -> Schedule:
    """Materialize the recovered source facts as a verifiable schedule slice."""

    if source.source_sha256 != edge.source_sha256:
        raise ValueError("Q/K edge provenance does not match the frozen source")
    roles = tuple(
        WarpRole(
            role.name,
            _role_kind(role.name),
            role.warps,
            first_warp=role.first_warp,
        )
        for role in source.roles
    )
    completion_name = f"{edge.free_barrier}_completion"
    completion_join = CompletionJoin(
        completion_name,
        participants=edge.free_arrivers,
        waiter=edge.reuse_waiter,
        arrival_count=edge.free_init_count,
        arrival_mode=edge.free_arrival,
        covered_roles=edge.ready_consumers,
    )
    fanout = SplitPhaseFanout(
        "qk_full",
        producer=edge.ready_producer,
        consumers=edge.ready_consumers,
        reuse_waiter=edge.reuse_waiter,
        stages=edge.stages,
        ready_init_count=edge.ready_init_count,
        free_init_count=edge.free_init_count,
        ready_arrival=edge.ready_arrival,
        free_arrival=edge.free_arrival,
        completion_join=completion_name,
    )
    operations = [
        Operation(
            "publish_qk_full",
            0,
            edge.ready_producer,
            "publish",
            writes=("qk_storage",),
            publishes=(fanout.name,),
        )
    ]
    operations.extend(
        Operation(
            f"consume_qk_{role}",
            1,
            role,
            "consume",
            reads=("qk_storage",),
            fanout_waits=(fanout.name,),
            completion_arrivals=(completion_name,)
            if role in edge.free_arrivers
            else (),
        )
        for role in edge.ready_consumers
    )
    operations.append(
        Operation(
            "authorize_qk_reuse",
            2,
            edge.reuse_waiter,
            "reuse",
            completion_waits=(completion_name,),
            recycles=(fanout.name,),
        )
    )
    return Schedule(
        name=f"source-derived-qk-{source.source_sha256[:12]}",
        tile=TileShape(
            tokens=edge.chunk_tokens,
            heads=1,
            key=128,
            value=source.value_rows or 128,
        ),
        roles=roles,
        buffers=(
            Buffer(
                "qk_storage",
                MemorySpace.SHARED,
                edge.stage_stride_bytes * edge.stages,
                128,
                0,
                1,
            ),
            Buffer("state", MemorySpace.GLOBAL, 4, 4, 0, 2, initialized=True),
        ),
        barriers=(),
        pipeline=Pipeline(edge.stages, tuple(operations)),
        resources=ResourceUse(
            threads_per_cta=source.launch_threads,
            shared_bytes=source.smem_bytes,
            tmem_columns=source.tmem_columns,
            registers_per_thread=0,
        ),
        recurrence=RecurrencePlan(
            RecurrenceKind.DIRECT, "state", edge.chunk_tokens
        ),
        fanouts=(fanout,),
        completion_joins=(completion_join,),
    )


def qk_raw_runtime_plan(edge: KdaQkFirstEdge) -> dict[str, object]:
    """Lower the directly representable qk_raw TMA -> prep-instance edge.

    qk_full is intentionally not emitted as PipelineConfig here. Split-consumer
    state is native to TS, but exact elect-one ready/free arrivals require the
    narrower LeaderArrivalSplitConsumer backend primitive.
    """

    return {
        "api": "cutlass.experimental.task_scheduling",
        "tasks": [
            {"name": "prep_instance", "warp_idx": 12, "num_warps": 4},
        ],
        "pipelines": [
            {
                "resource": "qk_raw_full",
                "factory": "create_tma_async_pipeline_cfg",
                "num_stages": edge.stages,
                "num_bytes": edge.tma_transaction_bytes,
                "producer_group": ["tma_engine"],
                "consumer_group": ["prep_instance"],
                "producer_group_size": 1,
                "consumer_group_size": edge.prep_consumer_threads,
                "producer_signaling_threads": "All",
                "consumer_signaling_threads": "All",
                "cta_layout_vmnk": [1, 1, 1, 1],
            }
        ],
        "unlowered_edges": [
            {
                "resource": "qk_full",
                "required_primitive": "LeaderArrivalSplitConsumer",
                "source_sha256": edge.source_sha256,
                "producer": edge.ready_producer,
                "consumers": list(edge.ready_consumers),
                "completion_arrivers": list(edge.free_arrivers),
                "reuse_waiter": edge.reuse_waiter,
                "free_barrier": edge.free_barrier,
                "split_consumer_support": "advance_on_wait",
                "frozen_arrival_counts": {
                    "ready": edge.ready_init_count,
                    "free": edge.free_init_count,
                },
                "arrival_modes": {
                    "ready": edge.ready_arrival.value,
                    "free": edge.free_arrival.value,
                },
                "completion_dominance": [
                    {
                        "consumer": proof.consumer,
                        "barrier_path": list(proof.barrier_path),
                        "source_lines": list(proof.source_lines),
                    }
                    for proof in edge.completion_proofs
                ],
                "searchable_native_rewrite": True,
            }
        ],
    }
