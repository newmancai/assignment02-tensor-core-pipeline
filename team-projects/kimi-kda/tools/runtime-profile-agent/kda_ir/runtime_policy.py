"""Proof-carrying shadow resolution for hardware-conditioned runtime policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
import json

from .runtime_profile import RuntimeWorkloadProfile, recommend_one_resident_wave_cpc


class RuntimePolicyStatus(str, Enum):
    SHADOW_ONLY = "shadow_only"
    QUALIFIED = "qualified"
    ACTIVE = "active"


@dataclass(frozen=True)
class PrepareKernelReceipt:
    """Measured resource identity of one exact compiled prepare function."""

    target_arch: str
    compute_capability: tuple[int, int]
    sm_count: int
    route_id: str
    prepare_variant_id: str
    chain_variant_id: str
    prepare_image_sha256: str
    block_threads: int
    registers_per_thread: int
    allocated_registers_per_thread: int
    static_shared_bytes: int
    dynamic_shared_bytes: int
    driver_shared_bytes: int
    cluster_shape: tuple[int, int, int]
    occupancy_limit_blocks: int
    occupancy_limit_registers: int
    occupancy_limit_shared_mem: int
    occupancy_limit_warps: int
    occupancy_limit_barriers: int
    resident_ctas_per_sm: int
    toolchain_manifest_sha256: str

    @property
    def shared_bytes_per_cta(self) -> int:
        return self.static_shared_bytes + self.dynamic_shared_bytes + self.driver_shared_bytes

    @property
    def derived_resident_ctas_per_sm(self) -> int:
        return min(
            self.occupancy_limit_blocks,
            self.occupancy_limit_registers,
            self.occupancy_limit_shared_mem,
            self.occupancy_limit_warps,
            self.occupancy_limit_barriers,
        )


@dataclass(frozen=True)
class RuntimeCallContract:
    """ABI and semantic facts required by the qualified Kimi-K3 H12 route."""

    qkv_dtype: str
    parameter_dtype: str
    qk_head_dim: int
    value_head_dim: int
    scale: float
    initial_state: bool
    inplace_final_state: bool
    qk_l2norm_in_kernel: bool
    gate_in_kernel: bool
    beta_is_logit: bool
    lower_bound: float
    token_quantum: int


@dataclass(frozen=True)
class ResidentGridPolicyV1:
    policy_id: str
    status: RuntimePolicyStatus
    workload_contract_sha256: str
    physical_receipt_sha256: str
    evidence_certificate_sha256: str
    supported_total_chunks: tuple[int, ...]
    supported_cpc: tuple[int, ...]
    fallback_policy_id: str
    formula_version: str = "resident_grid_capacity_v1"


@dataclass(frozen=True)
class PolicyGuardResult:
    code: str
    passed: bool
    message: str


@dataclass(frozen=True)
class PolicyResolution:
    selected_cpc: int
    recommendation_cpc: int | None
    applied: bool
    guard_results: tuple[PolicyGuardResult, ...]
    profile_sha256: str
    kernel_receipt_sha256: str
    policy_id: str
    fallback_policy_id: str
    fallback_reason: str | None


def canonical_sha256(value: object) -> str:
    payload = json.dumps(asdict(value), sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode()).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def qualified_h12_call_contract() -> RuntimeCallContract:
    return RuntimeCallContract(
        qkv_dtype="bfloat16",
        parameter_dtype="float32",
        qk_head_dim=128,
        value_head_dim=128,
        scale=1.0 / (128.0**0.5),
        initial_state=True,
        inplace_final_state=True,
        qk_l2norm_in_kernel=True,
        gate_in_kernel=True,
        beta_is_logit=True,
        lower_bound=-5.0,
        token_quantum=16,
    )


def resolve_resident_grid_policy(
    policy: ResidentGridPolicyV1,
    profile: RuntimeWorkloadProfile,
    call: RuntimeCallContract,
    receipt: PrepareKernelReceipt,
    *,
    fallback_cpc: int,
) -> PolicyResolution:
    """Resolve a recommendation, applying it only for an ACTIVE exact bundle."""

    if fallback_cpc <= 0:
        raise ValueError("fallback cpc must be positive")
    receipt_sha = canonical_sha256(receipt)
    profile_sha = canonical_sha256(profile)
    expected_call = qualified_h12_call_contract()
    call_sha = canonical_sha256(call)
    guards = [
        PolicyGuardResult(
            "RPEP001",
            call == expected_call
            and call_sha == policy.workload_contract_sha256
            and profile.num_heads == 12
            and profile.token_quantum == 16,
            "qualified workload/ABI contract",
        ),
        PolicyGuardResult(
            "RPEP002",
            receipt.target_arch == "sm_103a"
            and receipt.compute_capability == (10, 3)
            and receipt.sm_count == 148
            and profile.sm_count == receipt.sm_count,
            "qualified target and SM-count receipt",
        ),
        PolicyGuardResult(
            "RPEP003",
            receipt.route_id == "bt16_prepare_chain_m64"
            and receipt.prepare_variant_id
            == "sm_103a:flashkda_bf16_bt16_prepare_0d8e6c8011"
            and receipt.chain_variant_id
            == "sm_103a:flashkda_bf16_bt16_chain_m64_c68ffebac9",
            "qualified logical and physical route",
        ),
        PolicyGuardResult(
            "RPEP004",
            receipt.block_threads == 128
            and receipt.registers_per_thread == 77
            and receipt.allocated_registers_per_thread == 80
            and receipt.shared_bytes_per_cta == 45_056
            and receipt.cluster_shape == (1, 1, 1)
            and _is_sha256(receipt.prepare_image_sha256)
            and _is_sha256(receipt.toolchain_manifest_sha256),
            "qualified compiled resource footprint",
        ),
        PolicyGuardResult(
            "RPEP005",
            receipt.resident_ctas_per_sm == receipt.derived_resident_ctas_per_sm == 5,
            "resident CTA limit matches all recorded occupancy limits",
        ),
        PolicyGuardResult(
            "RPEP006",
            policy.formula_version == "resident_grid_capacity_v1",
            "recognized policy formula",
        ),
        PolicyGuardResult(
            "RPEP007",
            profile.total_chunks in policy.supported_total_chunks,
            "profile lies inside the qualified evidence domain",
        ),
        PolicyGuardResult(
            "RPEP009",
            receipt_sha == policy.physical_receipt_sha256
            and _is_sha256(policy.evidence_certificate_sha256),
            "receipt and evidence hashes match the policy bundle",
        ),
        PolicyGuardResult(
            "RPEP010",
            policy.status == RuntimePolicyStatus.ACTIVE,
            "policy is activation-authorized",
        ),
    ]
    pre_formula_ok = all(item.passed for item in guards[:8])
    recommendation = None
    if pre_formula_ok:
        recommendation = recommend_one_resident_wave_cpc(
            profile, receipt.resident_ctas_per_sm
        )
        formula_ok = (
            recommendation.chunks_per_cta in policy.supported_cpc
            and recommendation.rectangular_ctas
            <= recommendation.resident_grid_capacity_ctas
        )
        if recommendation.chunks_per_cta > 1:
            previous_grid = (
                (profile.total_chunks + recommendation.chunks_per_cta - 2)
                // (recommendation.chunks_per_cta - 1)
            ) * profile.num_heads
            formula_ok = (
                formula_ok and previous_grid > recommendation.resident_grid_capacity_ctas
            )
        guards.insert(
            8,
            PolicyGuardResult(
                "RPEP008",
                formula_ok,
                "recommendation is the first integer fitting resident CTA capacity",
            ),
        )
    else:
        guards.insert(
            8,
            PolicyGuardResult(
                "RPEP008", False, "formula not evaluated after an earlier guard failure"
            ),
        )

    applied = all(item.passed for item in guards)
    failed = next((item.code for item in guards if not item.passed), None)
    return PolicyResolution(
        selected_cpc=(recommendation.chunks_per_cta if applied and recommendation else fallback_cpc),
        recommendation_cpc=(recommendation.chunks_per_cta if recommendation else None),
        applied=applied,
        guard_results=tuple(guards),
        profile_sha256=profile_sha,
        kernel_receipt_sha256=receipt_sha,
        policy_id=policy.policy_id,
        fallback_policy_id=policy.fallback_policy_id,
        fallback_reason=failed,
    )
