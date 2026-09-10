#!/usr/bin/env python3
"""Create and build an isolated HMMA Phase-1 lookahead-depth-3 extension."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


FROZEN_SOURCE_SHA256 = {
    "csrc/smxx/fwd_kernel2.cuh": "950684947df54a3432732468c138a571bca08b44ab96b91c8540d9f0fd97db31",
    "csrc/smxx/fwd_launch.cu": "ffe7a15ad1196d1b3d771a55b3f9bdc3900e9ddac5bd51b20f0f2cec27c9643c",
    "setup.py": "487b01f3bdc8f9f232fc547b8f6519215899193e4ee076c5eae734c6858fbda8",
    "csrc/flash_kda.cpp": "2c07c7ef52007bc7d8f09bfca3d2ef870ad3f111ddea54d1474f65ffa4311b2d",
}
EXPECTED_MODULE = "flash_kda_phase1_l3_C"
ENVIRONMENT_KEYS = (
    "CUDA_HOME",
    "MAX_JOBS",
    "NVCC_THREADS",
    "FLASH_KDA_CUDA_ARCHS",
    "FLASH_KDA_ENABLE_V16_PREFETCH4",
    "FLASH_KDA_ENABLE_V16_PHASE1_PREFETCH",
    "FLASH_KDA_EXTENSION_NAME",
    "FLASH_KDA_VERSION_SUFFIX",
    "CPATH",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes(root: Path) -> dict[str, str]:
    return {name: sha256(root / name) for name in FROZEN_SOURCE_SHA256}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-source", type=Path, required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = args.base_source.resolve(strict=True)
    patch = args.patch.resolve(strict=True)
    output = args.output.resolve()
    source = output / "source"
    build_lib = output / "build" / "lib"
    build_temp = output / "build" / "temp"
    receipt_path = output / "build_receipt.json"
    build_log = output / "build.log"
    output.mkdir(parents=True, exist_ok=False)

    observed = source_hashes(base)
    if observed != FROZEN_SOURCE_SHA256:
        receipt_path.write_text(
            json.dumps(
                {
                    "status": "frozen_source_identity_failed",
                    "expected": FROZEN_SOURCE_SHA256,
                    "observed": observed,
                },
                indent=2,
            )
        )
        raise RuntimeError("frozen HMMA source hashes do not match")

    shutil.copytree(base, source, symlinks=True)
    patch_run = subprocess.run(
        ["patch", "--batch", "--forward", "-p1", "-i", str(patch)],
        cwd=source,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if patch_run.returncode != 0:
        receipt_path.write_text(
            json.dumps(
                {
                    "status": "patch_failed",
                    "patch_sha256": sha256(patch),
                    "patch_output": patch_run.stdout,
                },
                indent=2,
            )
        )
        raise RuntimeError("L3 source patch failed")

    kernel_text = (source / "csrc/smxx/fwd_kernel2.cuh").read_text()
    launch_text = (source / "csrc/smxx/fwd_launch.cu").read_text()
    if "Phase1Prefetch == 3" not in kernel_text:
        raise RuntimeError("patched kernel does not admit Phase1Prefetch=3")
    if "kPhase1Prefetch = HasStateIn ? 3 : 4" not in launch_text:
        raise RuntimeError("patched launcher does not select L3 for state input")

    expected_environment = {
        "FLASH_KDA_CUDA_ARCHS": "103a",
        "FLASH_KDA_ENABLE_V16_PREFETCH4": "1",
        "FLASH_KDA_ENABLE_V16_PHASE1_PREFETCH": "1",
        "FLASH_KDA_EXTENSION_NAME": EXPECTED_MODULE,
    }
    mismatched = {
        key: {"expected": value, "actual": os.getenv(key)}
        for key, value in expected_environment.items()
        if os.getenv(key) != value
    }
    if mismatched:
        raise RuntimeError(f"build environment mismatch: {mismatched}")

    command = [
        sys.executable,
        "setup.py",
        "build_ext",
        "--build-lib",
        str(build_lib),
        "--build-temp",
        str(build_temp),
        "--force",
    ]
    started = datetime.now(timezone.utc).isoformat()
    with build_log.open("w") as log:
        completed = subprocess.run(
            command,
            cwd=source,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    binaries = sorted(build_lib.glob(f"{EXPECTED_MODULE}*.so"))
    log_text = build_log.read_text(errors="replace")
    receipt = {
        "schema_version": 1,
        "candidate": "HMMA-V16-Phase1Lookahead3-state-in",
        "status": "built" if completed.returncode == 0 and len(binaries) == 1 else "build_failed",
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "base_source": str(base),
        "base_source_sha256": observed,
        "patch": str(patch),
        "patch_sha256": sha256(patch),
        "patch_output": patch_run.stdout,
        "patched_source_sha256": source_hashes(source),
        "command": command,
        "returncode": completed.returncode,
        "environment": {key: os.getenv(key) for key in ENVIRONMENT_KEYS},
        "expected_module": EXPECTED_MODULE,
        "binaries": [
            {"path": str(binary), "sha256": sha256(binary)} for binary in binaries
        ],
        "ptxas_resource_lines": [
            line.strip()
            for line in log_text.splitlines()
            if "ptxas info" in line
            and ("Used " in line or "spill stores" in line or "stack frame" in line)
        ],
    }
    receipt_path.write_text(json.dumps(receipt, indent=2))
    if receipt["status"] != "built":
        raise RuntimeError(
            f"L3 build failed: returncode={completed.returncode}, binaries={binaries}"
        )
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
