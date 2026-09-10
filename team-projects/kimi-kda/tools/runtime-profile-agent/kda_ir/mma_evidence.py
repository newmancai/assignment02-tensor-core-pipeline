"""Load a normalized MMA migration profile from archived raw evidence."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import asdict
from pathlib import Path

from .mma_migration import KernelMeasurement, MmaMigrationProfile


def _csv_rows(path: Path, *, comments: bool = False) -> list[dict[str, str]]:
    lines = path.read_text().splitlines()
    if comments:
        lines = [line for line in lines if line and not line.startswith("#")]
    return list(csv.DictReader(lines))


def _one(rows: list[dict[str, str]], **keys: str) -> dict[str, str]:
    matches = [row for row in rows if all(row.get(key) == value for key, value in keys.items())]
    if len(matches) != 1:
        raise ValueError(f"expected one row for {keys}, found {len(matches)}")
    return matches[0]


def _duration_us(value: str, unit: str) -> float:
    if unit == "us":
        return float(value)
    if unit == "ms":
        return float(value) * 1000.0
    raise ValueError(f"unsupported duration unit: {unit}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_mma_profile_from_evidence(
    *,
    sass_path: Path,
    ncu_summary_path: Path,
    tcgen_path: Path,
    semantic_path: Path,
    profile_id: str = "b300-sm103a-kimi-k3-h12-t8192-prefill",
    target_arch: str = "sm_103a",
    sm_count: int = 148,
    local_heads: int = 12,
    value_rows: int = 128,
) -> MmaMigrationProfile:
    """Parse the exact rows used by the C1 decision instead of copying metrics."""

    sass_rows = _csv_rows(sass_path)
    hmmas = int(_one(sass_rows, scope="recurrence_family_total", opcode="HMMA")["count"])
    tcgen = sum(
        int(_one(sass_rows, scope="recurrence_family_total", opcode=opcode)["count"])
        for opcode in ("TCGEN", "UTCMMA")
    )

    ncu = _one(_csv_rows(ncu_summary_path), label="official_v128")
    official = KernelMeasurement(
        label=ncu["label"],
        recurrence_ctas=int(ncu["cta_count"]),
        duration_us=_duration_us(ncu["duration_value"], ncu["duration_unit"]),
        sm_throughput_percent=float(ncu["compute_sm_value"]),
        dram_throughput_percent=float(ncu["dram_value"]),
        tensor_elapsed_percent=float(ncu["tensor_value"]),
        evidence_ref=str(ncu_summary_path),
    )

    probes = _csv_rows(tcgen_path, comments=True)
    l0_v128 = _one(probes, level="L0", V="128", grid="12", inner="64")
    l1_v128 = _one(probes, level="L1", V="128", grid="12", inner="64")
    l0_v16 = _one(probes, level="L0", V="16", grid="12", inner="64")
    l1_v16 = _one(probes, level="L1", V="16", grid="12", inner="64")

    return MmaMigrationProfile(
        profile_id=profile_id,
        target_arch=target_arch,
        sm_count=sm_count,
        local_heads=local_heads,
        hmmas_in_recurrence_sass=hmmas,
        tcgen_in_recurrence_sass=tcgen,
        official=official,
        value_rows=value_rows,
        value_rows_independent=True,
        dominant_phase_m=16,
        current_mma_atom_m=16,
        tcgen_min_m=64,
        phase6_m=128,
        tcgen_l0_v128_speedup=float(l0_v128["speedup_median"]),
        tcgen_l1_v128_speedup=float(l1_v128["speedup_median"]),
        tcgen_l0_v16_speedup=float(l0_v16["speedup_median"]),
        tcgen_l1_v16_speedup=float(l1_v16["speedup_median"]),
        tcgen_l0_v128_mma_active_blocks_per_sm=int(l0_v128["mma_active_blocks_per_sm"]),
        tcgen_l0_v128_active_blocks_per_sm=int(l0_v128["tcgen_active_blocks_per_sm"]),
        tcgen_l1_v128_mma_active_blocks_per_sm=int(l1_v128["mma_active_blocks_per_sm"]),
        tcgen_l1_v128_active_blocks_per_sm=int(l1_v128["tcgen_active_blocks_per_sm"]),
        tcgen_l1_v128_mma_static_smem_bytes=int(l1_v128["mma_static_smem_B"]),
        tcgen_l1_v128_static_smem_bytes=int(l1_v128["tcgen_static_smem_B"]),
        tcgen_l1_v128_mma_registers=int(l1_v128["mma_regs"]),
        tcgen_l1_v128_registers=int(l1_v128["tcgen_regs"]),
        sass_evidence_ref=str(sass_path),
        sass_evidence_sha256=_sha256(sass_path),
        semantic_evidence_ref=str(semantic_path),
        semantic_evidence_sha256=_sha256(semantic_path),
        tcgen_evidence_ref=str(tcgen_path),
        tcgen_evidence_sha256=_sha256(tcgen_path),
        ncu_evidence_sha256=_sha256(ncu_summary_path),
    )


def profile_to_dict(profile: MmaMigrationProfile) -> dict[str, object]:
    return asdict(profile)
