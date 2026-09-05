"""Conservative, calibrated dispatch for FlashKDA Kernel 2 value slicing.

The analytical features come from the K2 grid, recurrence length, duplicated
TMA traffic, and compiled resource usage.  Latency coefficients are B300
offline calibrations; unsupported devices and shapes deliberately use V128.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping


_DIMENSION = 128
_CHUNK = 16
_COMMON_TILE_BYTES = 13_888
_B300_L2_BYTES = 132_644_864
_MIN_RELATIVE_GAIN = 0.03
_MIN_ABSOLUTE_GAIN_MS = 0.005


@dataclass(frozen=True)
class Variant:
    value_slice: int
    threads: int
    registers_per_thread: int
    shared_memory_bytes: int
    intercept_ms: float
    per_sequence_head_ms: float
    extra_cta_layer_ms: float

    @property
    def slices(self) -> int:
        return _DIMENSION // self.value_slice


@dataclass(frozen=True)
class DispatchDecision:
    value_slice: int
    reason: str
    predicted_ms: Mapping[int, float]
    resident_blocks_per_sm: Mapping[int, int]
    reuse_footprint_bytes: int
    reuse_over_l2: float

    def as_dict(self) -> dict:
        return asdict(self)


# Fits exclude transition points and use independent-process medians.  Each
# tuple is (tokens, per-variant intercept/per-x/extra-layer coefficients).
_BF16_CALIBRATIONS = {
    2048: {
        16: (0.151090621, 0.000845711, 0.010853938),
        32: (0.155862086, 0.000797256, 0.019375225),
        64: (0.169650461, 0.000753838, 0.035217981),
        128: (0.205233538, 0.000790399, 0.0),
    },
    4096: {
        16: (0.279204257, 0.001661092, 0.017893640),
        32: (0.288008119, 0.001543180, 0.036878412),
        64: (0.315241212, 0.001443923, 0.071728845),
        128: (0.386640773, 0.001500288, 0.0),
    },
    8192: {
        16: (0.537473150, 0.003068970, 0.036876153),
        32: (0.553008273, 0.002937679, 0.076584545),
        64: (0.604635765, 0.002835139, 0.146840557),
        128: (0.748772236, 0.002931072, 0.0),
    },
}

_FP32_CALIBRATIONS = {
    4096: {
        16: (0.288545070, 0.001620028, 0.023031568),
        32: (0.296811066, 0.001531913, 0.037417786),
        64: (0.334322258, 0.001496270, 0.069331863),
        128: (0.378462821, 0.001509650, 0.0),
    },
}

_RESOURCES = {
    16: (96, 54, 49_808),
    32: (128, 58, 56_624),
    64: (192, 58, 70_256),
    128: (192, 73, 100_792),
}


def _fallback(reason: str, *, reuse_bytes: int = 0, l2_bytes: int = 0) -> DispatchDecision:
    ratio = reuse_bytes / l2_bytes if l2_bytes else math.inf
    return DispatchDecision(128, reason, {}, {}, reuse_bytes, ratio)


def _interpolate_coefficients(tokens: int, state_fp32: bool) -> Mapping[int, tuple[float, float, float]] | None:
    table = _FP32_CALIBRATIONS if state_fp32 else _BF16_CALIBRATIONS
    if tokens in table:
        return table[tokens]
    points = sorted(table)
    if tokens < points[0] or tokens > points[-1]:
        return None
    lower = max(point for point in points if point < tokens)
    upper = min(point for point in points if point > tokens)
    weight = (tokens - lower) / (upper - lower)
    return {
        value: tuple(
            left + weight * (right - left)
            for left, right in zip(table[lower][value], table[upper][value])
        )
        for value in _RESOURCES
    }


def _resident_blocks(resource: tuple[int, int, int], device: Mapping[str, int]) -> int:
    threads, registers, shared_memory = resource
    limits = (
        int(device["shared_memory_per_sm"]) // shared_memory,
        int(device["registers_per_sm"]) // (threads * registers),
        int(device["max_threads_per_sm"]) // threads,
        int(device["max_blocks_per_sm"]),
    )
    return max(0, min(limits))


def select_k2_value_slice(
    *,
    batch: int,
    tokens_per_sequence: int,
    heads: int,
    state_fp32: bool,
    is_varlen: bool,
    device: Mapping[str, int],
) -> DispatchDecision:
    """Score the calibrated candidates and return a guarded K2 selection."""

    sequence_heads = batch * heads
    tiles = math.ceil(tokens_per_sequence / _CHUNK)
    reuse_bytes = sequence_heads * tiles * _COMMON_TILE_BYTES
    l2_bytes = int(device.get("l2_bytes", 0))

    if is_varlen:
        return _fallback("varlen_not_calibrated", reuse_bytes=reuse_bytes, l2_bytes=l2_bytes)
    if (int(device.get("major", -1)), int(device.get("minor", -1))) != (10, 3):
        return _fallback("architecture_not_calibrated", reuse_bytes=reuse_bytes, l2_bytes=l2_bytes)
    if int(device.get("sm_count", 0)) != 148:
        return _fallback("sm_topology_not_calibrated", reuse_bytes=reuse_bytes, l2_bytes=l2_bytes)
    if not (0.95 * _B300_L2_BYTES <= l2_bytes <= 1.05 * _B300_L2_BYTES):
        return _fallback("l2_capacity_not_calibrated", reuse_bytes=reuse_bytes, l2_bytes=l2_bytes)
    if not (1 <= sequence_heads <= 96):
        return _fallback("sequence_head_domain_exceeded", reuse_bytes=reuse_bytes, l2_bytes=l2_bytes)

    coefficients = _interpolate_coefficients(tokens_per_sequence, state_fp32)
    if coefficients is None:
        return _fallback("recurrence_length_not_calibrated", reuse_bytes=reuse_bytes, l2_bytes=l2_bytes)

    sm_count = int(device["sm_count"])
    residents = {
        value: _resident_blocks(resource, device)
        for value, resource in _RESOURCES.items()
    }
    scores: dict[int, float] = {}
    for value, resource in _RESOURCES.items():
        if residents[value] == 0:
            continue
        slices = _DIMENSION // value
        grid = slices * sequence_heads
        ctas_on_busiest_sm = math.ceil(grid / sm_count)

        # The offline Phi calibration covers at most two CTA layers.  Rejecting
        # extrapolation is intentional; those candidates are not needed by the
        # validated high-value envelope.
        if ctas_on_busiest_sm > 2:
            continue
        intercept, per_x, layer_step = coefficients[value]
        scores[value] = (
            intercept
            + per_x * sequence_heads
            + layer_step * max(0, ctas_on_busiest_sm - 1)
        )

    if 128 not in scores:
        return _fallback("official_variant_not_feasible", reuse_bytes=reuse_bytes, l2_bytes=l2_bytes)

    candidate = min(scores, key=scores.get)
    official_ms = scores[128]
    gain_ms = official_ms - scores[candidate]
    relative_gain = gain_ms / official_ms
    if candidate == 128:
        reason = "official_predicted_fastest"
    elif gain_ms < _MIN_ABSOLUTE_GAIN_MS or relative_gain < _MIN_RELATIVE_GAIN:
        candidate = 128
        reason = "gain_below_guard_band"
    else:
        reason = "calibrated_score_with_guard_band"

    return DispatchDecision(
        candidate,
        reason,
        scores,
        residents,
        reuse_bytes,
        reuse_bytes / l2_bytes,
    )
