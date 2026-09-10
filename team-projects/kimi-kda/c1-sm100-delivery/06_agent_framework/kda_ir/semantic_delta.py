"""Typed schedule deltas for the frozen Blackwell KDA evolution corpus."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable

from .verify import Diagnostic


class DispatchTopology(str, Enum):
    SCALAR_TILE = "scalar_tile"
    VALUE_TILE = "value_tile"
    VALUE64_SPLIT = "value64_split"


@dataclass(frozen=True)
class WorkloadSignature:
    fixed_layout: bool
    num_heads: int
    sequence_lengths: tuple[int, ...]

    @property
    def uniform(self) -> bool:
        return len(set(self.sequence_lengths)) == 1

    @property
    def full_chunks(self) -> bool:
        return all(length % 32 == 0 for length in self.sequence_lengths)


@dataclass(frozen=True)
class BlackwellScheduleDelta:
    """A physical schedule choice, separated from the kernel's math body."""

    variant: str
    topology: DispatchTopology
    value_rows: int
    num_heads: int
    full_chunks: bool | None
    token_extent: int | None
    persistent_tasks: int
    grid_stride: int
    has_explicit_tile_schedule: bool
    source: str | None = None

    @property
    def fingerprint(self) -> tuple[object, ...]:
        return (
            self.topology.value,
            self.value_rows,
            self.num_heads,
            self.full_chunks,
            self.token_extent,
            self.persistent_tasks,
            self.grid_stride,
            self.has_explicit_tile_schedule,
        )

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["topology"] = self.topology.value
        return payload


_SCALAR = re.compile(r"^m128_h(?P<h>\d+)_p(?P<p>\d+)_s(?P<s>\d+)$")
_VALUE64 = re.compile(r"^m64_f(?P<f>[01])_t(?P<t>\d+)_h(?P<h>\d+)$")
_VALUE_TILE = re.compile(
    r"^vtile_f(?P<f>[01])_t(?P<t>\d+)_h(?P<h>\d+)_p(?P<p>\d+)_s(?P<s>\d+)$"
)
_SOURCE_PREFIX = "cake_flashkda_blackwell_evolution_"


def parse_blackwell_schedule_delta(
    variant: str, *, source: str | None = None
) -> BlackwellScheduleDelta:
    if match := _SCALAR.fullmatch(variant):
        return BlackwellScheduleDelta(
            variant=variant,
            topology=DispatchTopology.SCALAR_TILE,
            value_rows=128,
            num_heads=int(match["h"]),
            full_chunks=None,
            token_extent=None,
            persistent_tasks=int(match["p"]),
            grid_stride=int(match["s"]),
            has_explicit_tile_schedule=True,
            source=source,
        )
    if match := _VALUE64.fullmatch(variant):
        return BlackwellScheduleDelta(
            variant=variant,
            topology=DispatchTopology.VALUE64_SPLIT,
            value_rows=64,
            num_heads=int(match["h"]),
            full_chunks=bool(int(match["f"])),
            token_extent=int(match["t"]),
            persistent_tasks=0,
            grid_stride=2 * int(match["h"]),
            has_explicit_tile_schedule=False,
            source=source,
        )
    if match := _VALUE_TILE.fullmatch(variant):
        return BlackwellScheduleDelta(
            variant=variant,
            topology=DispatchTopology.VALUE_TILE,
            value_rows=128,
            num_heads=int(match["h"]),
            full_chunks=bool(int(match["f"])),
            token_extent=int(match["t"]),
            persistent_tasks=int(match["p"]),
            grid_stride=int(match["s"]),
            has_explicit_tile_schedule=False,
            source=source,
        )
    raise ValueError(f"unrecognized Blackwell KDA schedule variant: {variant}")


def verify_blackwell_schedule_delta(
    delta: BlackwellScheduleDelta,
) -> tuple[Diagnostic, ...]:
    diagnostics: list[Diagnostic] = []
    if delta.num_heads <= 0 or delta.grid_stride <= 0:
        diagnostics.append(Diagnostic("KIR801", "head count and grid stride must be positive"))
    if delta.topology is DispatchTopology.VALUE64_SPLIT:
        if delta.value_rows != 64 or delta.num_heads != 64:
            diagnostics.append(
                Diagnostic("KIR802", "value64 split requires value_rows=64 and H=64")
            )
    elif delta.value_rows != 128:
        diagnostics.append(Diagnostic("KIR803", "m128/value-tile schedules require value_rows=128"))
    if delta.topology is DispatchTopology.SCALAR_TILE:
        if not delta.has_explicit_tile_schedule or delta.token_extent is not None:
            diagnostics.append(
                Diagnostic("KIR804", "scalar-tile schedule requires an explicit task list")
            )
    elif delta.has_explicit_tile_schedule or delta.token_extent is None:
        diagnostics.append(
            Diagnostic("KIR805", "value-partitioned schedule must encode token extent")
        )
    if delta.full_chunks and delta.token_extent is not None and delta.token_extent % 32:
        diagnostics.append(
            Diagnostic("KIR806", "full-chunk schedule requires a 32-token-aligned extent")
        )
    return tuple(diagnostics)


def is_workload_compatible(
    delta: BlackwellScheduleDelta, workload: WorkloadSignature
) -> bool:
    if delta.num_heads != workload.num_heads:
        return False
    if delta.topology is DispatchTopology.VALUE64_SPLIT:
        return (
            workload.fixed_layout
            and len(workload.sequence_lengths) == 1
            and delta.token_extent == workload.sequence_lengths[0]
            and delta.full_chunks == workload.full_chunks
        )
    if delta.topology is DispatchTopology.VALUE_TILE:
        return (
            (workload.fixed_layout or workload.uniform)
            and delta.token_extent == workload.sequence_lengths[0]
            and delta.full_chunks == workload.full_chunks
        )
    return not workload.fixed_layout and not workload.uniform


def corpus_from_source_dir(source_dir: Path) -> tuple[BlackwellScheduleDelta, ...]:
    deltas = []
    for path in sorted(source_dir.glob(f"{_SOURCE_PREFIX}*.cu")):
        variant = path.stem.removeprefix(_SOURCE_PREFIX)
        delta = parse_blackwell_schedule_delta(variant, source=str(path))
        diagnostics = verify_blackwell_schedule_delta(delta)
        if diagnostics:
            details = "; ".join(f"{item.code}: {item.message}" for item in diagnostics)
            raise ValueError(f"invalid schedule delta {variant}: {details}")
        deltas.append(delta)
    return tuple(deltas)


def deduplicate_semantics(
    deltas: Iterable[BlackwellScheduleDelta],
) -> tuple[BlackwellScheduleDelta, ...]:
    unique: dict[tuple[object, ...], BlackwellScheduleDelta] = {}
    for delta in deltas:
        unique.setdefault(delta.fingerprint, delta)
    return tuple(unique.values())
