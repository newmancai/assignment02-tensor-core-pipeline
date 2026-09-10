"""Structure-preserving mutations exposed to an optimization agent."""

from __future__ import annotations

from dataclasses import replace

from .model import MemorySpace, RecurrenceKind, Schedule


def retile_value(schedule: Schedule, value: int) -> Schedule:
    if value <= 0 or 128 % value:
        raise ValueError("value tile must be a positive divisor of 128")
    ratio = value / schedule.tile.value
    buffers = tuple(
        replace(buffer, size_bytes=max(128, int(buffer.size_bytes * ratio)))
        if buffer.space is MemorySpace.SHARED
        else buffer
        for buffer in schedule.buffers
    )
    data_shared_bytes = sum(
        buffer.size_bytes for buffer in buffers if buffer.space is MemorySpace.SHARED
    )
    barrier_shared_bytes = sum(barrier.stages * 2 * 8 for barrier in schedule.barriers)
    barrier_shared_bytes += sum(fanout.stages * 2 * 8 for fanout in schedule.fanouts)
    return replace(
        schedule,
        name=f"{schedule.name}-v{value}",
        tile=replace(schedule.tile, value=value),
        buffers=buffers,
        resources=replace(
            schedule.resources,
            shared_bytes=data_shared_bytes + barrier_shared_bytes,
        ),
    )


def set_pipeline_stages(schedule: Schedule, stages: int) -> Schedule:
    if stages <= 0:
        raise ValueError("pipeline stages must be positive")
    barriers = tuple(replace(barrier, stages=stages) for barrier in schedule.barriers)
    fanouts = tuple(replace(fanout, stages=stages) for fanout in schedule.fanouts)
    data_shared_bytes = sum(
        buffer.size_bytes
        for buffer in schedule.buffers
        if buffer.space is MemorySpace.SHARED
    )
    barrier_shared_bytes = sum(barrier.stages * 2 * 8 for barrier in barriers)
    barrier_shared_bytes += sum(fanout.stages * 2 * 8 for fanout in fanouts)
    return replace(
        schedule,
        barriers=barriers,
        fanouts=fanouts,
        pipeline=replace(schedule.pipeline, stages=stages),
        resources=replace(
            schedule.resources,
            shared_bytes=data_shared_bytes + barrier_shared_bytes,
        ),
    )


def break_token_order(schedule: Schedule) -> Schedule:
    """Deliberately invalid mutation used to exercise a learned verifier rule."""

    return replace(
        schedule,
        recurrence=replace(schedule.recurrence, preserves_token_order=False),
    )


def set_prepare_chunks_per_cta(schedule: Schedule, chunks: int) -> Schedule:
    """Retile only the host work assignment of a prepare-chain schedule."""

    if schedule.recurrence.kind is not RecurrenceKind.PREPARE_CHAIN:
        raise ValueError("prepare work assignment requires a prepare-chain schedule")
    if chunks <= 0:
        raise ValueError("prepare chunks per CTA must be positive")
    return replace(
        schedule,
        name=f"{schedule.name}-prepare-cpc{chunks}",
        recurrence=replace(schedule.recurrence, prepare_chunks_per_cta=chunks),
    )
