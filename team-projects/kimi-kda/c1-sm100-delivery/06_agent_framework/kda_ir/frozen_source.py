"""Recover the explicit schedule contract retained in frozen CAKE CUDA source."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from .flashinfer_catalog import FrozenVariant
from .verify import Diagnostic


_DEFINE_RE = re.compile(r"^#define\s+([A-Z0-9_]+)\s+(\d+)\s*$", re.MULTILINE)
_KERNEL_RE = re.compile(
    r"__global__\s+__launch_bounds__\((\d+)\)\s+void\s+([A-Za-z_][A-Za-z0-9_]*)",
    re.MULTILINE,
)
_PIPELINE_RE = re.compile(r"// --- pipeline '([^']+)' ---")
_BARRIER_RE = re.compile(
    r"//\s+([A-Za-z_][A-Za-z0-9_]*):\s+(\d+) barriers, init_count=(\d+)"
)
_ROLE_RE = re.compile(
    r"// ---- Role: ([A-Za-z_][A-Za-z0-9_]*) ----\s*\n"
    r"\s*(?:}\s*else\s+)?if \(([^)]+)\)"
)
_BARRIER_USE_RE = re.compile(
    r"(mbarrier_wait|mbarrier_arrive|elect_commit)\(\s*"
    r"([A-Za-z_][A-Za-z0-9_]*)_addr"
)


@dataclass(frozen=True)
class FrozenRole:
    name: str
    first_warp: int
    last_warp: int

    @property
    def warps(self) -> int:
        return self.last_warp - self.first_warp + 1


@dataclass(frozen=True)
class FrozenBarrierGroup:
    pipeline: str
    name: str
    stages: int
    init_count: int


@dataclass(frozen=True)
class FrozenBarrierUse:
    role: str
    barrier: str
    action: str
    order: int
    source_line: int
    elect_one: bool = False


@dataclass(frozen=True)
class FrozenSourceContract:
    kernel_name: str
    launch_threads: int
    smem_bytes: int
    tmem_columns: int
    source_sha256: str
    chunk_tokens: int | None
    value_rows: int | None
    roles: tuple[FrozenRole, ...]
    barriers: tuple[FrozenBarrierGroup, ...]
    barrier_uses: tuple[FrozenBarrierUse, ...] = ()
    defines: tuple[tuple[str, int], ...] = ()

    def define(self, name: str) -> int | None:
        return dict(self.defines).get(name)


class FrozenSourceError(ValueError):
    pass


def _warp_range(condition: str) -> tuple[int, int]:
    exact = re.fullmatch(r"warp\s*==\s*(\d+)", condition.strip())
    if exact:
        warp = int(exact.group(1))
        return warp, warp
    bounded = re.fullmatch(
        r"warp\s*>=\s*(\d+)\s*&&\s*warp\s*<=\s*(\d+)", condition.strip()
    )
    if bounded:
        return int(bounded.group(1)), int(bounded.group(2))
    prefix = re.fullmatch(r"warp\s*<=\s*(\d+)", condition.strip())
    if prefix:
        return 0, int(prefix.group(1))
    raise FrozenSourceError(f"unsupported frozen role condition: {condition!r}")


def parse_frozen_source(path: str | Path) -> FrozenSourceContract:
    source_path = Path(path)
    raw = source_path.read_bytes()
    text = raw.decode("utf-8")
    defines = {name: int(value) for name, value in _DEFINE_RE.findall(text)}
    kernel = _KERNEL_RE.search(text)
    if not kernel:
        raise FrozenSourceError("missing __global__ __launch_bounds__ kernel")
    for required in ("SMEM_TOTAL", "TMEM_NCOLS"):
        if required not in defines:
            raise FrozenSourceError(f"missing required define {required}")

    role_matches = list(_ROLE_RE.finditer(text))
    roles = []
    barrier_uses = []
    for role_index, match in enumerate(role_matches):
        name, condition = match.groups()
        first, last = _warp_range(condition)
        roles.append(FrozenRole(name, first, last))
        body_end = (
            role_matches[role_index + 1].start()
            if role_index + 1 < len(role_matches)
            else len(text)
        )
        body = text[match.end() : body_end]
        for order, use in enumerate(_BARRIER_USE_RE.finditer(body)):
            absolute_start = match.end() + use.start()
            prefix = body[max(0, use.start() - 256) : use.start()]
            operation = use.group(1)
            barrier_uses.append(
                FrozenBarrierUse(
                    role=name,
                    barrier=use.group(2),
                    action="wait" if operation == "mbarrier_wait" else "arrive",
                    order=order,
                    source_line=text.count("\n", 0, absolute_start) + 1,
                    elect_one=operation == "elect_commit"
                    or bool(
                        re.search(
                            r"if\s*\(\s*elect_sync\(\)\s*\)\s*\{[^{}]*$",
                            prefix,
                            re.DOTALL,
                        )
                    ),
                )
            )

    barriers = []
    current_pipeline = ""
    for line in text.splitlines():
        pipeline = _PIPELINE_RE.search(line)
        if pipeline:
            current_pipeline = pipeline.group(1)
            continue
        barrier = _BARRIER_RE.search(line)
        if barrier:
            if not current_pipeline:
                raise FrozenSourceError("barrier group appears before a pipeline header")
            barriers.append(
                FrozenBarrierGroup(
                    current_pipeline,
                    barrier.group(1),
                    int(barrier.group(2)),
                    int(barrier.group(3)),
                )
            )

    chunk_tokens = None
    qd_bytes = defines.get("SMEM_SMEM_QD_STAGE_BYTES")
    if qd_bytes is not None and qd_bytes % (128 * 2) == 0:
        chunk_tokens = qd_bytes // (128 * 2)
    value_rows = None
    v_bytes = defines.get("SMEM_SMEM_V_STAGE_BYTES")
    if chunk_tokens and v_bytes is not None and v_bytes % (chunk_tokens * 2) == 0:
        value_rows = v_bytes // (chunk_tokens * 2)

    return FrozenSourceContract(
        kernel_name=kernel.group(2),
        launch_threads=int(kernel.group(1)),
        smem_bytes=defines["SMEM_TOTAL"],
        tmem_columns=defines["TMEM_NCOLS"],
        source_sha256=hashlib.sha256(raw).hexdigest(),
        chunk_tokens=chunk_tokens,
        value_rows=value_rows,
        roles=tuple(roles),
        barriers=tuple(barriers),
        barrier_uses=tuple(barrier_uses),
        defines=tuple(sorted(defines.items())),
    )


def check_source_variant_contract(
    source: FrozenSourceContract, variant: FrozenVariant
) -> tuple[Diagnostic, ...]:
    out = []
    if source.launch_threads != variant.threads:
        out.append(Diagnostic("KIR601", "source launch bounds differ from metadata"))
    if source.smem_bytes != variant.smem_bytes:
        out.append(Diagnostic("KIR602", "source shared memory differs from metadata"))
    if variant.body_sha256 and source.source_sha256 != variant.body_sha256:
        out.append(Diagnostic("KIR603", "source digest differs from frozen body digest"))
    if source.chunk_tokens != variant.launch_contract.get("chunk_tokens"):
        out.append(Diagnostic("KIR604", "source chunk size differs from launch contract"))
    if source.value_rows != variant.launch_contract.get("value_rows"):
        out.append(Diagnostic("KIR605", "source value rows differ from launch contract"))
    if variant.kernel_name and source.kernel_name != variant.kernel_name:
        out.append(Diagnostic("KIR606", "source kernel name differs from metadata"))
    return tuple(out)
