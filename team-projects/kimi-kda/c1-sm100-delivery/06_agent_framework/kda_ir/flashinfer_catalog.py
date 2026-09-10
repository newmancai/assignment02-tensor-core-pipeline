"""Read FlashInfer's frozen generated-KDA productization metadata."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .model import Schedule, Target
from .verify import Diagnostic


class CatalogError(ValueError):
    pass


@dataclass(frozen=True)
class FrozenVariant:
    variant_id: str
    arch: str
    abi_family: str
    state_mode: str
    threads: int
    smem_bytes: int
    use_pdl: bool
    kernel_name: str
    body_relpath: str
    body_sha256: str
    launch_contract: dict[str, Any]
    selector_keys: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class FrozenCatalog:
    schema_version: int
    selector_schema_version: int
    variants: tuple[FrozenVariant, ...]


def load_flashinfer_catalog(path: str | Path) -> FrozenCatalog:
    payload = json.loads(Path(path).read_text())
    required = {
        "schema_version",
        "physical_selector_schema_version",
        "variants",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise CatalogError(f"metadata is missing required fields: {missing}")
    variants = []
    for index, raw in enumerate(payload["variants"]):
        try:
            variants.append(
                FrozenVariant(
                    variant_id=raw["variant_id"],
                    arch=raw["arch"],
                    abi_family=raw["abi_family"],
                    state_mode=raw["state_mode"],
                    threads=int(raw["threads"]),
                    smem_bytes=int(raw["smem_bytes"]),
                    use_pdl=bool(raw["use_pdl"]),
                    kernel_name=raw.get("kernel_name", ""),
                    body_relpath=raw.get("body_relpath", ""),
                    body_sha256=raw.get("body_sha256", ""),
                    launch_contract=dict(raw["launch_contract"]),
                    selector_keys=tuple(
                        dict(item["selector_key"])
                        for item in raw.get("physical_selectors", ())
                    ),
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            raise CatalogError(f"invalid variant at index {index}: {error}") from error
    ids = [variant.variant_id for variant in variants]
    if len(ids) != len(set(ids)):
        raise CatalogError("variant_id values must be unique")
    return FrozenCatalog(
        schema_version=int(payload["schema_version"]),
        selector_schema_version=int(payload["physical_selector_schema_version"]),
        variants=tuple(variants),
    )


def summarize_catalog(catalog: FrozenCatalog, arch: str | None = None) -> dict[str, Any]:
    variants = tuple(
        variant
        for variant in catalog.variants
        if arch is None or _normalize_arch(variant.arch) == _normalize_arch(arch)
    )
    routes = Counter(
        selector.get("route", "<unselected>")
        for variant in variants
        for selector in variant.selector_keys or ({},)
    )
    return {
        "schema_version": catalog.schema_version,
        "selector_schema_version": catalog.selector_schema_version,
        "arch": arch,
        "variants": len(variants),
        "abi_families": dict(sorted(Counter(x.abi_family for x in variants).items())),
        "routes": dict(sorted(routes.items())),
        "pdl_variants": sum(variant.use_pdl for variant in variants),
    }


def _normalize_arch(arch: str) -> str:
    return arch.replace("_", "").lower()


def check_frozen_export_contract(
    schedule: Schedule, target: Target, variant: FrozenVariant
) -> tuple[Diagnostic, ...]:
    """Check facts exposed by the frozen artifact; it cannot reconstruct Cake IR."""

    out = []
    if _normalize_arch(target.arch) != _normalize_arch(variant.arch):
        out.append(Diagnostic("KIR501", "target architecture differs from frozen variant"))
    if schedule.resources.threads_per_cta != variant.threads:
        out.append(Diagnostic("KIR502", "thread contract differs from frozen variant"))
    if schedule.resources.shared_bytes != variant.smem_bytes:
        out.append(Diagnostic("KIR503", "shared-memory contract differs from frozen variant"))
    launch = variant.launch_contract
    if launch.get("chunk_tokens") != schedule.tile.tokens:
        out.append(Diagnostic("KIR504", "chunk-token contract differs from frozen variant"))
    if launch.get("value_rows") != schedule.tile.value:
        out.append(Diagnostic("KIR505", "value-row contract differs from frozen variant"))
    return tuple(out)


def closest_frozen_variants(
    schedule: Schedule,
    target: Target,
    catalog: FrozenCatalog,
    limit: int = 5,
) -> tuple[dict[str, Any], ...]:
    """Rank visible export contracts without pretending to recover hidden IR."""

    if limit <= 0:
        raise ValueError("limit must be positive")
    ranked = []
    for variant in catalog.variants:
        diagnostics = check_frozen_export_contract(schedule, target, variant)
        routes = sorted(
            {
                selector.get("route", "<unselected>")
                for selector in variant.selector_keys or ({},)
            }
        )
        ranked.append(
            {
                "variant_id": variant.variant_id,
                "abi_family": variant.abi_family,
                "routes": routes,
                "mismatch_codes": [item.code for item in diagnostics],
                "exact_visible_contract": not diagnostics,
            }
        )
    ranked.sort(
        key=lambda item: (
            len(item["mismatch_codes"]),
            item["abi_family"],
            item["variant_id"],
        )
    )
    return tuple(ranked[:limit])
