"""Typed physical-schedule IR for the KDA kernel-evolution prototype."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RoleKind(str, Enum):
    LOAD = "load"
    MMA = "mma"
    RECURRENCE = "recurrence"
    STORE = "store"


class MemorySpace(str, Enum):
    GLOBAL = "global"
    SHARED = "shared"
    REGISTER = "register"
    TMEM = "tmem"


class RecurrenceKind(str, Enum):
    DIRECT = "direct"
    PREPARE_CHAIN = "prepare_chain"
    PIECE_PERSISTENT = "piece_persistent"


class PipelineKind(str, Enum):
    ASYNC_ASYNC = "AsyncAsync"
    TMA_ASYNC = "TmaAsync"
    TMA_UMMA = "TmaUmma"
    UMMA_ASYNC = "UmmaAsync"
    ASYNC_UMMA = "AsyncUmma"
    UMMA_UMMA = "UmmaUmma"


class SignalingThreads(str, Enum):
    ALL = "All"
    CTA_LEADER = "CtaLeader"
    TASK_WARP_LEADER = "TaskWarpLeader"
    CTA_AND_TASK_WARP_LEADER = "CtaLeader|TaskWarpLeader"


class ArrivalMode(str, Enum):
    """Which threads contribute to a split-phase barrier arrival."""

    ELECT_ONE = "elect_one"
    ELECT_ONE_PER_WARP = "elect_one_per_warp"
    COOPERATIVE = "cooperative"


@dataclass(frozen=True)
class Target:
    arch: str
    sm_count: int
    max_threads_per_cta: int
    max_warps_per_cta: int
    max_warps_per_sm: int
    max_shared_bytes_per_cta: int
    shared_bytes_per_sm: int
    max_tmem_columns: int
    max_registers_per_thread: int
    max_ctas_per_sm: int = 8


@dataclass(frozen=True)
class TileShape:
    tokens: int
    heads: int
    key: int
    value: int


@dataclass(frozen=True)
class WarpRole:
    name: str
    kind: RoleKind
    warps: int
    first_warp: int | None = None

    @property
    def last_warp(self) -> int | None:
        return (
            self.first_warp + self.warps - 1
            if self.first_warp is not None
            else None
        )


@dataclass(frozen=True)
class Buffer:
    name: str
    space: MemorySpace
    size_bytes: int
    alignment: int
    live_from: int
    live_until: int
    initialized: bool = False


@dataclass(frozen=True)
class Barrier:
    name: str
    producers: tuple[str, ...]
    consumers: tuple[str, ...]
    kind: PipelineKind
    stages: int
    transaction_bytes: int = 0
    producer_signaling: SignalingThreads = SignalingThreads.ALL
    consumer_signaling: SignalingThreads = SignalingThreads.ALL
    cluster_shape: tuple[int, int, int, int] = (1, 1, 1, 1)


@dataclass(frozen=True)
class Operation:
    name: str
    order: int
    role: str
    kind: str
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    acquires: tuple[str, ...] = ()
    commits: tuple[str, ...] = ()
    waits: tuple[str, ...] = ()
    releases: tuple[str, ...] = ()
    publishes: tuple[str, ...] = ()
    fanout_waits: tuple[str, ...] = ()
    completion_arrivals: tuple[str, ...] = ()
    completion_waits: tuple[str, ...] = ()
    recycles: tuple[str, ...] = ()


@dataclass(frozen=True)
class CompletionJoin:
    """Join consumer completion before a pipeline stage can be recycled."""

    name: str
    participants: tuple[str, ...]
    waiter: str
    arrival_count: int
    arrival_mode: ArrivalMode = ArrivalMode.COOPERATIVE
    covered_roles: tuple[str, ...] = ()

    @property
    def coverage(self) -> tuple[str, ...]:
        return self.covered_roles or self.participants


@dataclass(frozen=True)
class SplitPhaseFanout:
    """One ready publication, many consumers, then guarded storage reuse."""

    name: str
    producer: str
    consumers: tuple[str, ...]
    reuse_waiter: str
    stages: int
    ready_init_count: int = 1
    free_init_count: int = 1
    ready_arrival: ArrivalMode = ArrivalMode.ELECT_ONE
    free_arrival: ArrivalMode = ArrivalMode.ELECT_ONE
    completion_join: str | None = None


@dataclass(frozen=True)
class Pipeline:
    stages: int
    operations: tuple[Operation, ...]


@dataclass(frozen=True)
class ResourceUse:
    threads_per_cta: int
    shared_bytes: int
    tmem_columns: int
    registers_per_thread: int


@dataclass(frozen=True)
class RecurrencePlan:
    kind: RecurrenceKind
    state_buffer: str
    piece_tokens: int
    token_quantum: int = 16
    preserves_token_order: bool = True
    prepare_chunks_per_cta: int = 1


@dataclass(frozen=True)
class Schedule:
    name: str
    tile: TileShape
    roles: tuple[WarpRole, ...]
    buffers: tuple[Buffer, ...]
    barriers: tuple[Barrier, ...]
    pipeline: Pipeline
    resources: ResourceUse
    recurrence: RecurrencePlan
    fanouts: tuple[SplitPhaseFanout, ...] = ()
    completion_joins: tuple[CompletionJoin, ...] = ()
