"""Static verifier: reject unsafe schedules before CUDA generation."""

from __future__ import annotations

from dataclasses import dataclass

from .model import (
    ArrivalMode,
    MemorySpace,
    PipelineKind,
    RecurrenceKind,
    RoleKind,
    Schedule,
    SignalingThreads,
    Target,
)


@dataclass(frozen=True)
class Diagnostic:
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


class VerificationError(ValueError):
    def __init__(self, diagnostics: list[Diagnostic]):
        self.diagnostics = tuple(diagnostics)
        super().__init__("\n".join(map(str, diagnostics)))


def _duplicates(names: list[str]) -> set[str]:
    return {name for name in names if names.count(name) > 1}


def verify(schedule: Schedule, target: Target) -> tuple[Diagnostic, ...]:
    """Return all discovered violations rather than failing at the first one."""

    out: list[Diagnostic] = []
    roles = {role.name: role for role in schedule.roles}
    buffers = {buffer.name: buffer for buffer in schedule.buffers}
    barriers = {barrier.name: barrier for barrier in schedule.barriers}
    fanouts = {fanout.name: fanout for fanout in schedule.fanouts}
    completion_joins = {join.name: join for join in schedule.completion_joins}

    for kind, names in (
        ("role", [x.name for x in schedule.roles]),
        ("buffer", [x.name for x in schedule.buffers]),
        ("barrier", [x.name for x in schedule.barriers]),
        ("fanout", [x.name for x in schedule.fanouts]),
        ("completion join", [x.name for x in schedule.completion_joins]),
        ("operation", [x.name for x in schedule.pipeline.operations]),
    ):
        for name in sorted(_duplicates(names)):
            out.append(Diagnostic("KIR001", f"duplicate {kind} name {name!r}"))

    role_warps = sum(role.warps for role in schedule.roles)
    if any(role.warps <= 0 for role in schedule.roles):
        out.append(Diagnostic("KIR101", "every warp role must own at least one warp"))
    positioned_roles = [role for role in schedule.roles if role.first_warp is not None]
    if positioned_roles and len(positioned_roles) != len(schedule.roles):
        out.append(Diagnostic("KIR105", "warp roles must be either all explicit or all implicit"))
    if not positioned_roles and schedule.resources.threads_per_cta != role_warps * 32:
        out.append(
            Diagnostic(
                "KIR102",
                "threads_per_cta must equal 32 times the sum of role warps",
            )
        )
    if len(positioned_roles) == len(schedule.roles):
        occupied_warps: set[int] = set()
        launch_warps = schedule.resources.threads_per_cta // 32
        for role in schedule.roles:
            assert role.first_warp is not None
            role_warp_ids = set(
                range(role.first_warp, role.first_warp + role.warps)
            )
            if role.first_warp < 0 or max(role_warp_ids, default=-1) >= launch_warps:
                out.append(
                    Diagnostic(
                        "KIR106",
                        f"warp role {role.name!r} lies outside the CTA launch range",
                    )
                )
            if occupied_warps & role_warp_ids:
                out.append(
                    Diagnostic(
                        "KIR107",
                        f"warp role {role.name!r} overlaps another explicit role",
                    )
                )
            occupied_warps.update(role_warp_ids)
    if role_warps > target.max_warps_per_cta:
        out.append(Diagnostic("KIR103", "warp allocation exceeds target capacity"))
    if schedule.resources.threads_per_cta > target.max_threads_per_cta:
        out.append(Diagnostic("KIR104", "thread allocation exceeds target capacity"))

    allocated_shared = sum(
        buffer.size_bytes
        for buffer in schedule.buffers
        if buffer.space is MemorySpace.SHARED
    )
    allocated_shared += sum(barrier.stages * 2 * 8 for barrier in schedule.barriers)
    allocated_shared += sum(fanout.stages * 2 * 8 for fanout in schedule.fanouts)
    if allocated_shared > schedule.resources.shared_bytes:
        out.append(
            Diagnostic(
                "KIR301",
                f"shared buffers require {allocated_shared} bytes but contract declares "
                f"{schedule.resources.shared_bytes}",
            )
        )
    if schedule.resources.shared_bytes > target.max_shared_bytes_per_cta:
        out.append(Diagnostic("KIR302", "shared-memory contract exceeds target capacity"))
    if schedule.resources.tmem_columns > target.max_tmem_columns:
        out.append(Diagnostic("KIR303", "TMEM contract exceeds target capacity"))
    if schedule.resources.registers_per_thread > target.max_registers_per_thread:
        out.append(Diagnostic("KIR304", "register contract exceeds target capacity"))

    for buffer in schedule.buffers:
        if buffer.size_bytes <= 0:
            out.append(Diagnostic("KIR201", f"buffer {buffer.name!r} has non-positive size"))
        if buffer.alignment <= 0 or buffer.alignment & (buffer.alignment - 1):
            out.append(
                Diagnostic("KIR202", f"buffer {buffer.name!r} alignment is not a power of two")
            )
        if buffer.live_from > buffer.live_until:
            out.append(Diagnostic("KIR203", f"buffer {buffer.name!r} has inverted lifetime"))

    writes: dict[str, list[int]] = {name: [] for name in buffers}
    barrier_acquires: dict[str, list[tuple[int, str]]] = {
        name: [] for name in barriers
    }
    barrier_commits: dict[str, list[tuple[int, str]]] = {
        name: [] for name in barriers
    }
    barrier_waits: dict[str, list[tuple[int, str]]] = {name: [] for name in barriers}
    barrier_releases: dict[str, list[tuple[int, str]]] = {
        name: [] for name in barriers
    }
    fanout_publishes: dict[str, list[tuple[int, str]]] = {
        name: [] for name in fanouts
    }
    fanout_waits: dict[str, list[tuple[int, str]]] = {name: [] for name in fanouts}
    fanout_recycles: dict[str, list[tuple[int, str]]] = {
        name: [] for name in fanouts
    }
    completion_arrivals: dict[str, list[tuple[int, str]]] = {
        name: [] for name in completion_joins
    }
    completion_waits: dict[str, list[tuple[int, str]]] = {
        name: [] for name in completion_joins
    }
    for op in schedule.pipeline.operations:
        if op.role not in roles:
            out.append(Diagnostic("KIR110", f"operation {op.name!r} uses unknown role {op.role!r}"))
        for access in op.reads + op.writes:
            if access not in buffers:
                out.append(
                    Diagnostic("KIR210", f"operation {op.name!r} references unknown buffer {access!r}")
                )
                continue
            buffer = buffers[access]
            if not buffer.live_from <= op.order <= buffer.live_until:
                out.append(
                    Diagnostic(
                        "KIR211",
                        f"operation {op.name!r} accesses {access!r} outside its lifetime",
                    )
                )
        for name in op.writes:
            if name in writes:
                writes[name].append(op.order)
        for name in op.acquires:
            if name not in barriers:
                out.append(Diagnostic("KIR120", f"operation {op.name!r} acquires unknown pipeline {name!r}"))
            else:
                barrier_acquires[name].append((op.order, op.role))
        for name in op.commits:
            if name not in barriers:
                out.append(Diagnostic("KIR121", f"operation {op.name!r} commits unknown pipeline {name!r}"))
            else:
                barrier_commits[name].append((op.order, op.role))
        for name in op.waits:
            if name not in barriers:
                out.append(Diagnostic("KIR122", f"operation {op.name!r} waits on unknown pipeline {name!r}"))
            else:
                barrier_waits[name].append((op.order, op.role))
        for name in op.releases:
            if name not in barriers:
                out.append(Diagnostic("KIR123", f"operation {op.name!r} releases unknown pipeline {name!r}"))
            else:
                barrier_releases[name].append((op.order, op.role))
        for name in op.publishes:
            if name not in fanouts:
                out.append(Diagnostic("KIR150", f"operation {op.name!r} publishes unknown fanout {name!r}"))
            else:
                fanout_publishes[name].append((op.order, op.role))
        for name in op.fanout_waits:
            if name not in fanouts:
                out.append(Diagnostic("KIR151", f"operation {op.name!r} waits on unknown fanout {name!r}"))
            else:
                fanout_waits[name].append((op.order, op.role))
        for name in op.completion_arrivals:
            if name not in completion_joins:
                out.append(
                    Diagnostic(
                        "KIR171",
                        f"operation {op.name!r} arrives at unknown completion join {name!r}",
                    )
                )
            else:
                completion_arrivals[name].append((op.order, op.role))
        for name in op.completion_waits:
            if name not in completion_joins:
                out.append(
                    Diagnostic(
                        "KIR171",
                        f"operation {op.name!r} waits on unknown completion join {name!r}",
                    )
                )
            else:
                completion_waits[name].append((op.order, op.role))
        for name in op.recycles:
            if name not in fanouts:
                out.append(Diagnostic("KIR152", f"operation {op.name!r} recycles unknown fanout {name!r}"))
            else:
                fanout_recycles[name].append((op.order, op.role))

    for op in schedule.pipeline.operations:
        for name in op.reads:
            if name not in buffers or buffers[name].initialized:
                continue
            if not any(order <= op.order for order in writes[name]):
                out.append(
                    Diagnostic(
                        "KIR212",
                        f"operation {op.name!r} reads {name!r} before any producer writes it",
                    )
                )

    for barrier in schedule.barriers:
        acquires = barrier_acquires[barrier.name]
        commits = barrier_commits[barrier.name]
        waits = barrier_waits[barrier.name]
        releases = barrier_releases[barrier.name]
        lifecycle = {
            "acquire": acquires,
            "commit": commits,
            "wait": waits,
            "release": releases,
        }
        for action, events in lifecycle.items():
            if len(events) != 1:
                out.append(
                    Diagnostic(
                        "KIR130",
                        f"pipeline {barrier.name!r} requires one {action} in the "
                        f"captured schedule, observed {len(events)}",
                    )
                )
        for _, role in acquires + commits:
            if role not in barrier.producers:
                out.append(Diagnostic("KIR131", f"role {role!r} cannot produce pipeline {barrier.name!r}"))
        for _, role in waits + releases:
            if role not in barrier.consumers:
                out.append(Diagnostic("KIR133", f"role {role!r} cannot consume pipeline {barrier.name!r}"))
        if all(len(events) == 1 for events in lifecycle.values()):
            acquire_order = acquires[0][0]
            commit_order = commits[0][0]
            wait_order = waits[0][0]
            release_order = releases[0][0]
            if not acquire_order <= commit_order < wait_order <= release_order:
                out.append(Diagnostic("KIR134", f"pipeline {barrier.name!r} violates acquire/commit/wait/release order"))

        if barrier.stages <= 0:
            out.append(Diagnostic("KIR135", f"pipeline {barrier.name!r} must have positive stages"))
        if barrier.stages != schedule.pipeline.stages:
            out.append(Diagnostic("KIR136", f"pipeline {barrier.name!r} stage count differs from the captured schedule"))
        tma_producer = barrier.kind in (PipelineKind.TMA_ASYNC, PipelineKind.TMA_UMMA)
        umma_producer = barrier.kind in (PipelineKind.UMMA_ASYNC, PipelineKind.UMMA_UMMA)
        umma_consumer = barrier.kind in (
            PipelineKind.TMA_UMMA,
            PipelineKind.ASYNC_UMMA,
            PipelineKind.UMMA_UMMA,
        )
        if tma_producer and barrier.transaction_bytes <= 0:
            out.append(Diagnostic("KIR137", f"TMA pipeline {barrier.name!r} needs a transaction-byte contract"))
        if not tma_producer and barrier.transaction_bytes:
            out.append(Diagnostic("KIR138", f"non-TMA pipeline {barrier.name!r} cannot declare transaction bytes"))
        if tma_producer and not any(
            roles.get(name) and roles[name].kind is RoleKind.LOAD
            for name in barrier.producers
        ):
            out.append(Diagnostic("KIR139", f"TMA pipeline {barrier.name!r} has no load producer"))
        if umma_producer and not any(
            roles.get(name) and roles[name].kind is RoleKind.MMA
            for name in barrier.producers
        ):
            out.append(Diagnostic("KIR140", f"UMMA pipeline {barrier.name!r} has no MMA producer"))
        if umma_consumer and not any(
            roles.get(name) and roles[name].kind is RoleKind.MMA
            for name in barrier.consumers
        ):
            out.append(Diagnostic("KIR141", f"UMMA pipeline {barrier.name!r} has no MMA consumer"))
        if (
            barrier.producer_signaling is SignalingThreads.TASK_WARP_LEADER
            and not tma_producer
        ):
            out.append(Diagnostic("KIR142", "TaskWarpLeader producer signaling is only valid for TMA pipelines"))
        if any(extent <= 0 for extent in barrier.cluster_shape):
            out.append(Diagnostic("KIR143", f"pipeline {barrier.name!r} has invalid cluster shape"))

    for fanout in schedule.fanouts:
        publishes = fanout_publishes[fanout.name]
        waits = fanout_waits[fanout.name]
        recycles = fanout_recycles[fanout.name]
        if fanout.producer not in roles:
            out.append(Diagnostic("KIR153", f"fanout {fanout.name!r} has unknown producer"))
        if not fanout.consumers or any(name not in roles for name in fanout.consumers):
            out.append(Diagnostic("KIR154", f"fanout {fanout.name!r} has unknown or empty consumers"))
        if fanout.reuse_waiter not in roles:
            out.append(Diagnostic("KIR155", f"fanout {fanout.name!r} has unknown reuse waiter"))
        if len(set(fanout.consumers)) != len(fanout.consumers):
            out.append(Diagnostic("KIR156", f"fanout {fanout.name!r} repeats a consumer"))
        if fanout.stages <= 0 or fanout.stages != schedule.pipeline.stages:
            out.append(Diagnostic("KIR157", f"fanout {fanout.name!r} stage count differs from the schedule"))
        if fanout.ready_init_count <= 0 or fanout.free_init_count <= 0:
            out.append(Diagnostic("KIR158", f"fanout {fanout.name!r} has invalid arrival counts"))
        producer_warps = (
            roles[fanout.producer].warps if fanout.producer in roles else 0
        )
        producer_threads = producer_warps * 32
        join = completion_joins.get(fanout.completion_join)
        free_arrival_roles = (
            join.participants if join is not None else (fanout.reuse_waiter,)
        )
        free_arrival_warps = sum(
            roles[name].warps for name in free_arrival_roles if name in roles
        )
        free_arrival_threads = free_arrival_warps * 32
        if (
            fanout.ready_arrival is ArrivalMode.ELECT_ONE
            and fanout.ready_init_count != 1
        ):
            out.append(
                Diagnostic(
                    "KIR163",
                    f"fanout {fanout.name!r} elect-one ready count must be one",
                )
            )
        if (
            fanout.free_arrival is ArrivalMode.ELECT_ONE
            and fanout.free_init_count != 1
        ):
            out.append(
                Diagnostic(
                    "KIR164",
                    f"fanout {fanout.name!r} elect-one free count must be one",
                )
            )
        if (
            fanout.ready_arrival is ArrivalMode.COOPERATIVE
            and fanout.ready_init_count != producer_threads
        ):
            out.append(
                Diagnostic(
                    "KIR165",
                    f"fanout {fanout.name!r} cooperative ready count must equal "
                    "producer threads",
                )
            )
        if (
            fanout.free_arrival is ArrivalMode.COOPERATIVE
            and fanout.free_init_count != free_arrival_threads
        ):
            out.append(
                Diagnostic(
                    "KIR166",
                    f"fanout {fanout.name!r} cooperative free count must equal "
                    "completion-arrival threads",
                )
            )
        if (
            fanout.ready_arrival is ArrivalMode.ELECT_ONE_PER_WARP
            and fanout.ready_init_count != producer_warps
        ):
            out.append(
                Diagnostic(
                    "KIR175",
                    f"fanout {fanout.name!r} per-warp ready count must equal producer warps",
                )
            )
        if (
            fanout.free_arrival is ArrivalMode.ELECT_ONE_PER_WARP
            and fanout.free_init_count != free_arrival_warps
        ):
            out.append(
                Diagnostic(
                    "KIR176",
                    f"fanout {fanout.name!r} per-warp free count must equal completion-arrival warps",
                )
            )
        if len(publishes) != 1 or publishes[0][1] != fanout.producer:
            out.append(Diagnostic("KIR159", f"fanout {fanout.name!r} requires one producer publication"))
        observed_waiters = sorted(role for _, role in waits)
        if observed_waiters != sorted(fanout.consumers):
            out.append(Diagnostic("KIR160", f"fanout {fanout.name!r} waiters differ from consumers"))
        if len(recycles) != 1 or recycles[0][1] != fanout.reuse_waiter:
            out.append(Diagnostic("KIR161", f"fanout {fanout.name!r} requires one reuse authorization"))
        if len(publishes) == 1 and waits and len(recycles) == 1:
            if not publishes[0][0] < min(order for order, _ in waits) <= recycles[0][0]:
                out.append(Diagnostic("KIR162", f"fanout {fanout.name!r} violates publish/wait/recycle order"))
        if len(fanout.consumers) > 1 and fanout.completion_join not in completion_joins:
            out.append(
                Diagnostic(
                    "KIR167",
                    f"fanout {fanout.name!r} requires a completion join before recycling",
                )
            )
        elif fanout.completion_join in completion_joins:
            join = completion_joins[fanout.completion_join]
            if sorted(join.coverage) != sorted(fanout.consumers):
                out.append(
                    Diagnostic(
                        "KIR168",
                        f"fanout {fanout.name!r} completion coverage differs from consumers",
                    )
                )
            if join.waiter != fanout.reuse_waiter:
                out.append(
                    Diagnostic(
                        "KIR169",
                        f"fanout {fanout.name!r} reuse waiter must wait on its completion join",
                    )
                )
            join_waits = completion_waits[join.name]
            if len(recycles) == 1 and len(join_waits) == 1:
                if join_waits[0][0] > recycles[0][0]:
                    out.append(
                        Diagnostic(
                            "KIR174",
                            f"fanout {fanout.name!r} recycles before completion join wait",
                        )
                    )

    for join in schedule.completion_joins:
        participants_valid = (
            bool(join.participants)
            and all(name in roles for name in join.participants)
            and len(set(join.participants)) == len(join.participants)
            and join.waiter in roles
            and all(name in roles for name in join.coverage)
            and len(set(join.coverage)) == len(join.coverage)
        )
        if not participants_valid:
            out.append(
                Diagnostic(
                    "KIR168",
                    f"completion join {join.name!r} has invalid participants",
                )
            )
        expected_arrivals = (
            sum(roles[name].warps * 32 for name in join.participants if name in roles)
            if join.arrival_mode is ArrivalMode.COOPERATIVE
            else sum(roles[name].warps for name in join.participants if name in roles)
            if join.arrival_mode is ArrivalMode.ELECT_ONE_PER_WARP
            else 1
        )
        if join.arrival_count != expected_arrivals:
            out.append(
                Diagnostic(
                    "KIR170",
                    f"completion join {join.name!r} arrival count must be {expected_arrivals}",
                )
            )
        observed_arrivals = sorted(
            role for _, role in completion_arrivals[join.name]
        )
        if observed_arrivals != sorted(join.participants):
            out.append(
                Diagnostic(
                    "KIR172",
                    f"completion join {join.name!r} arrivals differ from participants",
                )
            )
        observed_waits = completion_waits[join.name]
        if len(observed_waits) != 1 or observed_waits[0][1] != join.waiter:
            out.append(
                Diagnostic(
                    "KIR173",
                    f"completion join {join.name!r} requires one wait by {join.waiter!r}",
                )
            )
        if completion_arrivals[join.name] and len(observed_waits) == 1:
            if max(order for order, _ in completion_arrivals[join.name]) > observed_waits[0][0]:
                out.append(
                    Diagnostic(
                        "KIR174",
                        f"completion join {join.name!r} wait does not follow all arrivals",
                    )
                )

    recurrence = schedule.recurrence
    if recurrence.state_buffer not in buffers:
        out.append(Diagnostic("KIR401", "recurrence references an unknown state buffer"))
    if recurrence.piece_tokens <= 0 or recurrence.token_quantum <= 0:
        out.append(Diagnostic("KIR402", "recurrence piece and quantum must be positive"))
    elif recurrence.piece_tokens % recurrence.token_quantum:
        out.append(Diagnostic("KIR403", "recurrence piece violates the numerical token quantum"))
    if not recurrence.preserves_token_order:
        out.append(Diagnostic("KIR404", "recurrence schedule does not preserve token order"))
    if recurrence.prepare_chunks_per_cta <= 0:
        out.append(Diagnostic("KIR406", "prepare chunks per CTA must be positive"))
    if (
        recurrence.kind is not RecurrenceKind.PREPARE_CHAIN
        and recurrence.prepare_chunks_per_cta != 1
    ):
        out.append(
            Diagnostic(
                "KIR407",
                "prepare work assignment is only valid for prepare-chain recurrence",
            )
        )
    op_kinds = {op.kind for op in schedule.pipeline.operations}
    if recurrence.kind is RecurrenceKind.PREPARE_CHAIN and not {
        "recurrence_prepare",
        "recurrence_chain",
    }.issubset(op_kinds):
        out.append(Diagnostic("KIR405", "prepare-chain schedule is missing a physical phase"))

    return tuple(out)


def require_valid(schedule: Schedule, target: Target) -> Schedule:
    diagnostics = list(verify(schedule, target))
    if diagnostics:
        raise VerificationError(diagnostics)
    return schedule
