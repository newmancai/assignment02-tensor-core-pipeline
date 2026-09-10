from __future__ import annotations

from dataclasses import replace
import unittest

from kda_ir import (
    PrepareKernelReceipt,
    ResidentGridPolicyV1,
    RuntimePolicyStatus,
    RuntimeWorkloadProfile,
    canonical_sha256,
    qualified_h12_call_contract,
    resolve_resident_grid_policy,
)


def _receipt() -> PrepareKernelReceipt:
    return PrepareKernelReceipt(
        target_arch="sm_103a",
        compute_capability=(10, 3),
        sm_count=148,
        route_id="bt16_prepare_chain_m64",
        prepare_variant_id="sm_103a:flashkda_bf16_bt16_prepare_0d8e6c8011",
        chain_variant_id="sm_103a:flashkda_bf16_bt16_chain_m64_c68ffebac9",
        prepare_image_sha256="1" * 64,
        block_threads=128,
        registers_per_thread=77,
        allocated_registers_per_thread=80,
        static_shared_bytes=0,
        dynamic_shared_bytes=44_032,
        driver_shared_bytes=1_024,
        cluster_shape=(1, 1, 1),
        occupancy_limit_blocks=32,
        occupancy_limit_registers=6,
        occupancy_limit_shared_mem=5,
        occupancy_limit_warps=16,
        occupancy_limit_barriers=32,
        resident_ctas_per_sm=5,
        toolchain_manifest_sha256="2" * 64,
    )


def _policy(receipt: PrepareKernelReceipt, status: RuntimePolicyStatus) -> ResidentGridPolicyV1:
    return ResidentGridPolicyV1(
        policy_id="h12-resident-grid-v1",
        status=status,
        workload_contract_sha256=canonical_sha256(qualified_h12_call_contract()),
        physical_receipt_sha256=canonical_sha256(receipt),
        evidence_certificate_sha256="4" * 64,
        supported_total_chunks=(384, 768),
        supported_cpc=(7, 13),
        fallback_policy_id="existing-h12-cpc9",
    )


class RuntimePolicyTest(unittest.TestCase):
    def test_receipt_derives_five_resident_ctas_from_shared_memory_limit(self):
        receipt = _receipt()
        self.assertEqual(receipt.shared_bytes_per_cta, 45_056)
        self.assertEqual(receipt.derived_resident_ctas_per_sm, 5)

    def test_shadow_policy_emits_recommendation_without_mutating_selection(self):
        receipt = _receipt()
        profile = RuntimeWorkloadProfile(12, (6144,), False, 148)
        resolution = resolve_resident_grid_policy(
            _policy(receipt, RuntimePolicyStatus.SHADOW_ONLY),
            profile,
            qualified_h12_call_contract(),
            receipt,
            fallback_cpc=9,
        )
        self.assertFalse(resolution.applied)
        self.assertEqual(resolution.recommendation_cpc, 7)
        self.assertEqual(resolution.selected_cpc, 9)
        self.assertEqual(resolution.fallback_reason, "RPEP010")

    def test_active_exact_bundle_applies_capacity_prediction(self):
        receipt = _receipt()
        profile = RuntimeWorkloadProfile(12, (12_288,), False, 148)
        resolution = resolve_resident_grid_policy(
            _policy(receipt, RuntimePolicyStatus.ACTIVE),
            profile,
            qualified_h12_call_contract(),
            receipt,
            fallback_cpc=9,
        )
        self.assertTrue(resolution.applied)
        self.assertEqual(resolution.recommendation_cpc, 13)
        self.assertEqual(resolution.selected_cpc, 13)
        self.assertIsNone(resolution.fallback_reason)

    def test_resource_drift_forces_fallback(self):
        receipt = _receipt()
        drifted = replace(
            receipt,
            occupancy_limit_shared_mem=4,
            resident_ctas_per_sm=4,
        )
        profile = RuntimeWorkloadProfile(12, (6144,), False, 148)
        resolution = resolve_resident_grid_policy(
            _policy(receipt, RuntimePolicyStatus.ACTIVE),
            profile,
            qualified_h12_call_contract(),
            drifted,
            fallback_cpc=9,
        )
        self.assertFalse(resolution.applied)
        self.assertEqual(resolution.selected_cpc, 9)
        self.assertEqual(resolution.fallback_reason, "RPEP005")

    def test_outside_evidence_domain_forces_fallback(self):
        receipt = _receipt()
        profile = RuntimeWorkloadProfile(12, (8192,), False, 148)
        resolution = resolve_resident_grid_policy(
            _policy(receipt, RuntimePolicyStatus.ACTIVE),
            profile,
            qualified_h12_call_contract(),
            receipt,
            fallback_cpc=9,
        )
        self.assertFalse(resolution.applied)
        self.assertEqual(resolution.fallback_reason, "RPEP007")


if __name__ == "__main__":
    unittest.main()
