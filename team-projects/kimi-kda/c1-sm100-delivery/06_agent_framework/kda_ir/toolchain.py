"""Feature-safe compile plans for frozen Blackwell schedule bodies."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .frozen_source import FrozenSourceContract
from .model import Target
from .verify import Diagnostic


class ToolchainPlanError(ValueError):
    pass


@dataclass(frozen=True)
class CompiledKernelResources:
    arch: str
    kernel_name: str
    registers_per_thread: int
    stack_bytes: int
    static_shared_bytes: int


_CUBIN_ARCH_RE = re.compile(r"arch = (sm_\d+a?)")
_CUBIN_RESOURCE_RE = re.compile(
    r"Function ([A-Za-z_][A-Za-z0-9_]*):\s*\n"
    r"\s*REG:(\d+) STACK:(\d+) SHARED:(\d+)"
)


def normalize_feature_arch(arch: str) -> str:
    compact = arch.lower().replace("_", "")
    if compact not in {"sm100a", "sm103a"}:
        raise ToolchainPlanError(
            f"expected an architecture-specific Blackwell target, got {arch!r}"
        )
    return f"sm_{compact[2:]}"


def nvcc_compile_plan(
    source: FrozenSourceContract,
    arch: str,
    input_path: str | Path,
    output_path: str | Path,
) -> tuple[str, ...]:
    """Return argv that preserves the `a` feature target required by tcgen05.

    `nvcc -arch=sm_100a` was observed to emit `.target sm_100` with CUDA 13.0.88
    for the frozen KDA body. The explicit compute/code pair is therefore part
    of the checked physical-schedule contract rather than a free-form flag.
    """

    target = normalize_feature_arch(arch)
    compute = target.replace("sm_", "compute_", 1)
    if source.tmem_columns and not target.endswith("a"):
        raise ToolchainPlanError("TMEM schedules require an architecture target")
    return (
        "-std=c++17",
        "--use_fast_math",
        "-diag-suppress=177",
        f"-gencode=arch={compute},code={target}",
        "-c",
        str(input_path),
        "-o",
        str(output_path),
    )


def parse_cuobjdump_resources(output: str) -> CompiledKernelResources:
    arch = _CUBIN_ARCH_RE.search(output)
    resource = _CUBIN_RESOURCE_RE.search(output)
    if not arch or not resource:
        raise ToolchainPlanError("unrecognized cuobjdump resource output")
    return CompiledKernelResources(
        arch=arch.group(1),
        kernel_name=resource.group(1),
        registers_per_thread=int(resource.group(2)),
        stack_bytes=int(resource.group(3)),
        static_shared_bytes=int(resource.group(4)),
    )


def check_compiled_resources(
    source: FrozenSourceContract,
    target: Target,
    compiled: CompiledKernelResources,
) -> tuple[Diagnostic, ...]:
    out = []
    if normalize_feature_arch(target.arch) != compiled.arch:
        out.append(Diagnostic("KIR701", "compiled object architecture differs from target"))
    if source.kernel_name != compiled.kernel_name:
        out.append(Diagnostic("KIR702", "compiled kernel name differs from source"))
    total_shared = source.smem_bytes + compiled.static_shared_bytes
    if total_shared > target.max_shared_bytes_per_cta:
        out.append(
            Diagnostic(
                "KIR703",
                f"dynamic plus static shared memory requires {total_shared} bytes",
            )
        )
    if compiled.registers_per_thread > target.max_registers_per_thread:
        out.append(Diagnostic("KIR704", "compiled register use exceeds target"))
    return tuple(out)
