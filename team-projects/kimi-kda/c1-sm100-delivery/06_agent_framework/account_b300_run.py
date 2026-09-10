#!/usr/bin/env python3
"""Run one command while recording auditable single-node GPU resource usage."""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


def _nvidia_smi(query: str) -> list[list[str]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        [field.strip() for field in line.split(",")]
        for line in result.stdout.splitlines()
        if line.strip()
    ]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    if args.sample_interval <= 0:
        parser.error("--sample-interval must be positive")

    hardware_rows = _nvidia_smi("index,name,uuid,memory.total,power.limit")
    samples: list[dict[str, float]] = []
    stop = threading.Event()

    def sample() -> None:
        while not stop.is_set():
            stamp = time.monotonic()
            try:
                rows = _nvidia_smi("index,power.draw,utilization.gpu,memory.used")
                for row in rows:
                    samples.append(
                        {
                            "monotonic_seconds": stamp,
                            "gpu_index": float(row[0]),
                            "power_w": float(row[1]),
                            "utilization_percent": float(row[2]),
                            "memory_used_mib": float(row[3]),
                        }
                    )
            except (subprocess.CalledProcessError, ValueError):
                pass
            stop.wait(args.sample_interval)

    started_at = _utc_now()
    started = time.monotonic()
    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    completed = subprocess.run(command, check=False)
    stop.set()
    sampler.join(timeout=max(1.0, 2 * args.sample_interval))
    ended = time.monotonic()
    ended_at = _utc_now()

    wall_seconds = ended - started
    powers = [sample["power_w"] for sample in samples]
    utilizations = [sample["utilization_percent"] for sample in samples]
    # The fixed-rate integral intentionally includes compilation and idle gaps:
    # it is allocation energy, not kernel-only energy.
    energy_joules = sum(powers) * args.sample_interval
    report = {
        "schema_version": 1,
        "experiment": args.experiment,
        "command": command,
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "wall_seconds": wall_seconds,
        "exit_code": completed.returncode,
        "gpu_count": len(hardware_rows),
        "allocated_gpu_seconds": wall_seconds * len(hardware_rows),
        "hardware": [
            {
                "index": int(row[0]),
                "name": row[1],
                "uuid": row[2],
                "memory_total_mib": float(row[3]),
                "power_limit_w": float(row[4]),
            }
            for row in hardware_rows
        ],
        "telemetry": {
            "sample_interval_seconds": args.sample_interval,
            "sample_count": len(samples),
            "average_power_w": statistics.fmean(powers) if powers else None,
            "peak_power_w": max(powers) if powers else None,
            "estimated_allocation_energy_joules": energy_joules if powers else None,
            "average_gpu_utilization_percent": (
                statistics.fmean(utilizations) if utilizations else None
            ),
            "peak_memory_used_mib": (
                max(sample["memory_used_mib"] for sample in samples)
                if samples
                else None
            ),
        },
        "accounting_scope": (
            "single-node visible-GPU allocation; telemetry includes compilation, "
            "benchmark setup, synchronization, and idle gaps"
        ),
        "scheduler": {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
            "slurm_node_list": os.environ.get("SLURM_JOB_NODELIST"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(serialized)
    # Keep every attempt even when the stable "latest" path is reused. Failed
    # compiler/runtime experiments are part of the compute budget, not noise.
    safe_experiment = re.sub(r"[^A-Za-z0-9_.-]+", "-", args.experiment).strip("-")
    compact_started = re.sub(r"[^0-9]", "", started_at)[:14]
    job_id = os.environ.get("SLURM_JOB_ID", "no-slurm")
    attempt_dir = args.output.parent / "attempts"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    attempt_path = attempt_dir / f"{compact_started}-{job_id}-{safe_experiment}.json"
    attempt_path.write_text(serialized)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
