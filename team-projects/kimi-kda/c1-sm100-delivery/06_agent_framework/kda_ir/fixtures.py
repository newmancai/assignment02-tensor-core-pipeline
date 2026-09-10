"""Known schedules used as expressiveness and verifier fixtures."""

from __future__ import annotations

from dataclasses import replace

from .model import (
    Barrier,
    Buffer,
    MemorySpace,
    Operation,
    Pipeline,
    PipelineKind,
    RecurrenceKind,
    RecurrencePlan,
    ResourceUse,
    RoleKind,
    Schedule,
    Target,
    TileShape,
    WarpRole,
)


B300 = Target(
    arch="sm103a",
    sm_count=148,
    max_threads_per_cta=1024,
    max_warps_per_cta=32,
    max_warps_per_sm=64,
    max_shared_bytes_per_cta=232_448,
    shared_bytes_per_sm=232_448,
    max_tmem_columns=512,
    max_registers_per_thread=255,
)


def _direct(name: str, value_tile: int, shared_bytes: int, registers: int) -> Schedule:
    return Schedule(
        name=name,
        tile=TileShape(tokens=16, heads=1, key=128, value=value_tile),
        roles=(
            WarpRole("loader", RoleKind.LOAD, 1),
            WarpRole("mma", RoleKind.MMA, 3),
            WarpRole("state", RoleKind.RECURRENCE, 3),
            WarpRole("storer", RoleKind.STORE, 1),
        ),
        buffers=(
            Buffer("inputs", MemorySpace.GLOBAL, 4096, 16, 0, 3, initialized=True),
            Buffer("tiles", MemorySpace.SHARED, shared_bytes // 2, 128, 0, 1),
            Buffer("delta", MemorySpace.SHARED, shared_bytes // 2, 128, 1, 2),
            Buffer("state", MemorySpace.GLOBAL, 32_768, 16, 0, 3, initialized=True),
            Buffer("output", MemorySpace.GLOBAL, 4096, 16, 3, 3),
        ),
        barriers=(
            Barrier(
                "tiles_ready",
                ("loader",),
                ("mma",),
                PipelineKind.TMA_ASYNC,
                2,
                transaction_bytes=4096,
            ),
            Barrier(
                "delta_ready", ("mma",), ("state",), PipelineKind.ASYNC_ASYNC, 2
            ),
        ),
        pipeline=Pipeline(
            stages=2,
            operations=(
                Operation(
                    "tma_load",
                    0,
                    "loader",
                    "tma_load",
                    ("inputs",),
                    ("tiles",),
                    acquires=("tiles_ready",),
                    commits=("tiles_ready",),
                ),
                Operation(
                    "mma_delta",
                    1,
                    "mma",
                    "mma",
                    ("tiles",),
                    ("delta",),
                    acquires=("delta_ready",),
                    commits=("delta_ready",),
                    waits=("tiles_ready",),
                    releases=("tiles_ready",),
                ),
                Operation(
                    "state_update",
                    2,
                    "state",
                    "recurrence",
                    ("delta", "state"),
                    ("state",),
                    waits=("delta_ready",),
                    releases=("delta_ready",),
                ),
                Operation("store", 3, "storer", "store", ("state",), ("output",)),
            ),
        ),
        resources=ResourceUse(256, shared_bytes + 64, 0, registers),
        recurrence=RecurrencePlan(RecurrenceKind.DIRECT, "state", 16),
    )


def official_flashkda() -> Schedule:
    return _direct("official-sm80-mma-physical-schedule", 128, 196_608, 224)


def value_sliced_v16() -> Schedule:
    return _direct("value-sliced-v16", 16, 98_304, 128)


def cake_bt16_prepare_chain() -> Schedule:
    schedule = _direct("cake-class-bt16-prepare-chain", 64, 131_072, 160)
    operations = (
        schedule.pipeline.operations[0],
        replace(schedule.pipeline.operations[1], kind="recurrence_prepare"),
        replace(schedule.pipeline.operations[2], kind="recurrence_chain"),
        schedule.pipeline.operations[3],
    )
    barriers = (
        replace(schedule.barriers[0], kind=PipelineKind.TMA_UMMA, stages=3),
        replace(schedule.barriers[1], kind=PipelineKind.UMMA_ASYNC, stages=3),
    )
    return replace(
        schedule,
        barriers=barriers,
        pipeline=replace(schedule.pipeline, stages=3, operations=operations),
        resources=replace(schedule.resources, shared_bytes=131_072 + 96),
        recurrence=RecurrencePlan(
            RecurrenceKind.PREPARE_CHAIN,
            "state",
            16,
            prepare_chunks_per_cta=4,
        ),
    )
