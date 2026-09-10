"""Hardware-normalized runtime-profile features for KDA schedule evolution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from statistics import fmean, pstdev


@dataclass(frozen=True)
class RuntimeWorkloadProfile:
    """The workload facts an agent may use without inspecting timing labels."""

    num_heads: int
    seq_lens: tuple[int, ...]
    packed: bool
    sm_count: int
    token_quantum: int = 16

    def __post_init__(self) -> None:
        if self.num_heads <= 0 or self.sm_count <= 0 or self.token_quantum <= 0:
            raise ValueError("heads, SM count, and token quantum must be positive")
        if not self.seq_lens or any(length <= 0 for length in self.seq_lens):
            raise ValueError("sequence lengths must be non-empty and positive")

    @property
    def chunks_per_sequence(self) -> tuple[int, ...]:
        return tuple(ceil(length / self.token_quantum) for length in self.seq_lens)

    @property
    def total_chunks(self) -> int:
        return sum(self.chunks_per_sequence)

    @property
    def max_chunks(self) -> int:
        return max(self.chunks_per_sequence)

    @property
    def effective_sequence_parallelism(self) -> float:
        """Total recurrent work divided by the longest sequence critical path."""

        return self.total_chunks / self.max_chunks

    @property
    def max_chunk_fraction(self) -> float:
        return self.max_chunks / self.total_chunks

    @property
    def chunk_cv(self) -> float:
        chunks = self.chunks_per_sequence
        return pstdev(chunks) / fmean(chunks) if len(chunks) > 1 else 0.0


@dataclass(frozen=True)
class PrepareGridGeometry:
    chunks_per_cta: int
    rectangular_ctas: int
    scheduled_ctas: int
    waves: int  # CTA scheduling rounds if only one CTA per SM is counted.
    tail_utilization: float
    wave_quantized: bool


@dataclass(frozen=True)
class ResidentWaveRecommendation:
    """CPC that first fits one full resident-grid capacity fill."""

    resident_ctas_per_sm: int
    resident_grid_capacity_ctas: int
    chunk_groups_per_wave: int
    chunks_per_cta: int
    rectangular_ctas: int
    occupancy_aware_waves: float


def recommend_one_resident_wave_cpc(
    profile: RuntimeWorkloadProfile, resident_ctas_per_sm: int
) -> ResidentWaveRecommendation:
    """Return the smallest cpc whose H-way grid fits resident CTA capacity.

    ``resident_ctas_per_sm`` is a physical-receipt value conditioned on the
    compiled kernel's block, register, shared-memory, and cluster footprint. It
    must be re-derived when any of those facts or the target changes.
    """

    if resident_ctas_per_sm <= 0:
        raise ValueError("resident CTAs per SM must be positive")
    capacity = profile.sm_count * resident_ctas_per_sm
    chunk_groups = capacity // profile.num_heads
    if chunk_groups <= 0:
        raise ValueError("resident grid capacity cannot hold one CTA per head")
    chunks_per_cta = ceil(profile.total_chunks / chunk_groups)
    rectangular = ceil(profile.total_chunks / chunks_per_cta) * profile.num_heads
    return ResidentWaveRecommendation(
        resident_ctas_per_sm=resident_ctas_per_sm,
        resident_grid_capacity_ctas=capacity,
        chunk_groups_per_wave=chunk_groups,
        chunks_per_cta=chunks_per_cta,
        rectangular_ctas=rectangular,
        occupancy_aware_waves=rectangular / capacity,
    )


def prepare_grid_geometry(
    profile: RuntimeWorkloadProfile,
    chunks_per_cta: int,
    *,
    quantize_min_waves: int = 8,
    quantize_min_retained_percent: int = 98,
) -> PrepareGridGeometry:
    """Mirror the current FlashKDA BT16 host-grid arithmetic.

    This is a geometry model, not a latency predictor.  It intentionally exposes
    the receipt that a profile analyst and a measurement judge can discuss.
    """

    if chunks_per_cta <= 0:
        raise ValueError("chunks per CTA must be positive")
    rectangular = ceil(profile.total_chunks / chunks_per_cta) * profile.num_heads
    scheduled = rectangular
    if rectangular >= quantize_min_waves * profile.sm_count:
        full_wave = (rectangular // profile.sm_count) * profile.sm_count
        retained = 100 * full_wave / rectangular
        if full_wave >= profile.num_heads and retained >= quantize_min_retained_percent:
            scheduled = full_wave
    waves = ceil(scheduled / profile.sm_count)
    return PrepareGridGeometry(
        chunks_per_cta=chunks_per_cta,
        rectangular_ctas=rectangular,
        scheduled_ctas=scheduled,
        waves=waves,
        tail_utilization=scheduled / (waves * profile.sm_count),
        wave_quantized=scheduled != rectangular,
    )


def runtime_profile_receipt(
    profile: RuntimeWorkloadProfile,
    candidates: tuple[int, ...] = (1, 3, 4, 6, 9, 12, 16),
    *,
    resident_ctas_per_sm: int | None = None,
) -> dict[str, object]:
    """Return a JSON-ready profile and counterfactual prepare-grid receipt."""

    if not candidates:
        raise ValueError("at least one chunks-per-CTA candidate is required")
    receipt = {
        "num_heads": profile.num_heads,
        "seq_lens": list(profile.seq_lens),
        "layout": "packed" if profile.packed else "fixed",
        "sm_count": profile.sm_count,
        "token_quantum": profile.token_quantum,
        "total_chunks": profile.total_chunks,
        "max_chunks": profile.max_chunks,
        "effective_sequence_parallelism": profile.effective_sequence_parallelism,
        "max_chunk_fraction": profile.max_chunk_fraction,
        "chunk_cv": profile.chunk_cv,
        "prepare_grids": [
            asdict(prepare_grid_geometry(profile, candidate))
            for candidate in candidates
        ],
    }
    if resident_ctas_per_sm is not None:
        receipt["one_resident_wave_policy"] = asdict(
            recommend_one_resident_wave_cpc(profile, resident_ctas_per_sm)
        )
    return receipt
