import unittest

from kda_ir.design_ir import (
    Carrier,
    DataflowProgram,
    KernelDesign,
    MaterializationPlacement,
    MmaRoute,
    PhysicalLayout,
    PipelineKind,
    ScalarType,
    SchedulerKind,
    TmemLifetime,
    TransformKind,
    TransformOp,
    Value,
    ValueType,
    canonicalize_dataflow,
    materialization_metrics,
    paired_warp_hmma_design,
    tcgen_phase6_bridge_program,
    verify_dataflow,
    verify_kernel_design,
)


class DesignIRTest(unittest.TestCase):
    def test_paired_warp_hmma_expands_the_old_search_boundary(self):
        baseline = paired_warp_hmma_design(value_slice=64, compact=False)
        paired = paired_warp_hmma_design(value_slice=64, compact=True)
        self.assertEqual(baseline.compute_warps, 4)
        self.assertEqual(paired.compute_warps, 2)
        self.assertFalse(verify_kernel_design(paired))
        self.assertNotEqual(baseline.canonical_id(), paired.canonical_id())

    def test_rejects_more_than_two_column_blocks_per_warp(self):
        design = paired_warp_hmma_design(value_slice=128, compact=True)
        invalid = KernelDesign(**{**design.__dict__, "compute_warps": 1})
        self.assertIn("KIR1205", {item.code for item in verify_kernel_design(invalid)})

    def test_tcgen_requires_async_tmem_and_preferred_layout(self):
        invalid = KernelDesign(
            "bad-tcgen",
            MmaRoute.TCGEN05,
            16,
            1,
            Carrier.REGISTER,
            PhysicalLayout.ROW_MAJOR,
            MaterializationPlacement.PER_RECURRENCE_STEP,
            SchedulerKind.STATIC,
            PipelineKind.SYNC,
            TmemLifetime.NONE,
            False,
        )
        codes = {item.code for item in verify_kernel_design(invalid)}
        self.assertTrue({"KIR1212", "KIR1213", "KIR1214", "KIR1215"} <= codes)

    def test_direct_preferred_producer_has_no_materialization(self):
        design = KernelDesign(
            "direct-preferred",
            MmaRoute.TCGEN05,
            16,
            1,
            Carrier.TMEM,
            PhysicalLayout.TCGEN05_32B_SWIZZLED_NK,
            MaterializationPlacement.NONE,
            SchedulerKind.STATIC_PERSISTENT,
            PipelineKind.TMA_UMMA,
            TmemLifetime.CROSS_PHASE,
            True,
        )
        self.assertFalse(verify_kernel_design(design))

    def test_fuses_redundant_layout_chain(self):
        program = tcgen_phase6_bridge_program(recurrence_steps=64)
        before = materialization_metrics(program)
        result = canonicalize_dataflow(program)
        after = materialization_metrics(result.program)
        self.assertFalse(verify_dataflow(result.program))
        self.assertEqual(before.operation_count, 2)
        self.assertEqual(after.operation_count, 1)
        self.assertEqual(before.bytes_per_cta, 2 * after.bytes_per_cta)
        self.assertIn("fuse_materialization_chain", {item.rule for item in result.actions})
        fused = result.program.operations[0]
        self.assertEqual(fused.source, "u_reg")
        self.assertEqual(fused.result, "u_preferred")

    def test_round_boundary_blocks_fusion(self):
        values = (
            Value("fp32", ValueType((16, 16), ScalarType.FP32, PhysicalLayout.ROW_MAJOR, Carrier.REGISTER), external=True),
            Value("bf16", ValueType((16, 16), ScalarType.BF16, PhysicalLayout.ROW_MAJOR, Carrier.REGISTER)),
            Value("preferred", ValueType((16, 16), ScalarType.BF16, PhysicalLayout.TCGEN05_32B_SWIZZLED_NK, Carrier.SHARED), output=True),
        )
        program = DataflowProgram(
            "rounding",
            values,
            (
                TransformOp("round", "fp32", "bf16", TransformKind.ROUND, MaterializationPlacement.PER_RECURRENCE_STEP, True),
                TransformOp("layout", "bf16", "preferred", TransformKind.RELAYOUT, MaterializationPlacement.PER_RECURRENCE_STEP),
            ),
        )
        result = canonicalize_dataflow(program)
        self.assertEqual(len(result.program.operations), 2)
        self.assertEqual(result.actions, ())

    def test_accumulator_layout_cannot_masquerade_as_tmem_operand_a(self):
        program = tcgen_phase6_bridge_program()
        values = tuple(
            Value(
                value.name,
                ValueType(
                    value.type.shape,
                    value.type.dtype,
                    PhysicalLayout.TMEM_ACCUMULATOR,
                    Carrier.TMEM,
                ),
                value.external,
                value.output,
            )
            if value.name == "a"
            else value
            for value in program.values
        )
        invalid = DataflowProgram(
            "accumulator-is-not-a-operand",
            values,
            program.operations,
        )
        self.assertIn("KIR1239", {item.code for item in verify_dataflow(invalid)})


if __name__ == "__main__":
    unittest.main()
