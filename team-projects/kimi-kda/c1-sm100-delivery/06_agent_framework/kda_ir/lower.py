"""Lower verified IR to an inspectable backend-neutral launch plan."""

from __future__ import annotations

from dataclasses import asdict

from .model import RecurrenceKind, Schedule, Target
from .verify import require_valid


def lowering_plan(schedule: Schedule, target: Target) -> dict[str, object]:
    require_valid(schedule, target)
    role_warps = sum(role.warps for role in schedule.roles)
    resource_limits = [target.max_ctas_per_sm, target.max_warps_per_sm // role_warps]
    if schedule.resources.shared_bytes:
        resource_limits.append(
            target.shared_bytes_per_sm // schedule.resources.shared_bytes
        )
    if schedule.resources.tmem_columns:
        resource_limits.append(target.max_tmem_columns // schedule.resources.tmem_columns)
    launches = (
        ["prepare", "chain"]
        if schedule.recurrence.kind is RecurrenceKind.PREPARE_CHAIN
        else [schedule.recurrence.kind.value]
    )
    return {
        "schedule": schedule.name,
        "target": target.arch,
        "tile": asdict(schedule.tile),
        "warp_roles": {role.name: role.warps for role in schedule.roles},
        "warp_ranges": {
            role.name: (
                [role.first_warp, role.first_warp + role.warps - 1]
                if role.first_warp is not None
                else None
            )
            for role in schedule.roles
        },
        "launches": launches,
        "prepare_grid": (
            {"chunks_per_cta": schedule.recurrence.prepare_chunks_per_cta}
            if schedule.recurrence.kind is RecurrenceKind.PREPARE_CHAIN
            else None
        ),
        "estimated_resident_ctas_per_sm": max(1, min(resource_limits)),
        "operations": [
            {
                "order": op.order,
                "role": op.role,
                "kind": op.kind,
                "waits": list(op.waits),
                "acquires": list(op.acquires),
                "commits": list(op.commits),
                "releases": list(op.releases),
                "publishes": list(op.publishes),
                "fanout_waits": list(op.fanout_waits),
                "completion_arrivals": list(op.completion_arrivals),
                "completion_waits": list(op.completion_waits),
                "recycles": list(op.recycles),
            }
            for op in schedule.pipeline.operations
        ],
        "split_phase_fanouts": [asdict(fanout) for fanout in schedule.fanouts],
        "completion_joins": [
            {
                **asdict(join),
                "arrival_mode": join.arrival_mode.value,
            }
            for join in schedule.completion_joins
        ],
    }
