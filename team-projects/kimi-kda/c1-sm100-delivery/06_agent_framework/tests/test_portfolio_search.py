import unittest

from kda_ir.design_ir import (
    Carrier,
    MaterializationPlacement,
    PhysicalLayout,
    PipelineKind,
    SchedulerKind,
    TmemLifetime,
    paired_warp_hmma_design,
)
from kda_ir.portfolio_search import (
    AxisUpdate,
    DesignPatch,
    MeasuredDesign,
    PortfolioRole,
    coordinate_portfolio,
    design_niche,
    portfolio_elites,
)


class PortfolioSearchTest(unittest.TestCase):
    def test_composes_distinct_roles_instead_of_only_merging_duplicates(self):
        base = paired_warp_hmma_design(value_slice=64, compact=False)
        patches = (
            DesignPatch(
                "paired-warps",
                PortfolioRole.PARALLELISM,
                (AxisUpdate("compute_warps", 2),),
                "two independent column blocks per warp",
            ),
            DesignPatch(
                "persistent-grid",
                PortfolioRole.SCHEDULER,
                (AxisUpdate("scheduler", SchedulerKind.STATIC_PERSISTENT),),
                "reuse CTAs across recurrent tasks",
            ),
        )
        result = coordinate_portfolio(base, patches)
        compound = [candidate for candidate in result.candidates if len(candidate.patch_ids) == 2]
        self.assertEqual(len(compound), 1)
        self.assertEqual(compound[0].design.compute_warps, 2)
        self.assertEqual(compound[0].design.scheduler, SchedulerKind.STATIC_PERSISTENT)

    def test_rejects_conflicting_axis_updates(self):
        base = paired_warp_hmma_design(value_slice=64, compact=False)
        patches = (
            DesignPatch("one-warp", PortfolioRole.PARALLELISM, (AxisUpdate("compute_warps", 1),), "aggressive"),
            DesignPatch("two-warps", PortfolioRole.SCHEDULER, (AxisUpdate("compute_warps", 2),), "bounded"),
        )
        result = coordinate_portfolio(base, patches)
        self.assertIn("KIR1252", {item.code for row in result.rejected for item in row.diagnostics})

    def test_tcgen_compound_can_cross_the_old_representation_boundary(self):
        base = paired_warp_hmma_design(value_slice=16, compact=False)
        patches = (
            DesignPatch(
                "tcgen-carrier",
                PortfolioRole.RESIDENCY,
                (
                    AxisUpdate("accumulator_carrier", Carrier.TMEM),
                    AxisUpdate("tmem_lifetime", TmemLifetime.CROSS_PHASE),
                    AxisUpdate("pipeline", PipelineKind.TMA_UMMA),
                ),
                "shared P3/P4/P6 lifecycle",
            ),
            DesignPatch(
                "producer-layout",
                PortfolioRole.LAYOUT,
                (
                    AxisUpdate("operand_b_layout", PhysicalLayout.TCGEN05_32B_SWIZZLED_NK),
                    AxisUpdate("materialization", MaterializationPlacement.NONE),
                    AxisUpdate("producer_emits_consumer_layout", True),
                ),
                "producer writes the consumer descriptor layout",
            ),
        )
        # Route itself is immutable by patches: use an explicitly typed tcgen base.
        tcgen_base = base.__class__(
            **{
                **base.__dict__,
                "design_id": "tcgen-base",
                "route": base.route.__class__.TCGEN05,
                "accumulator_carrier": Carrier.TMEM,
                "operand_b_layout": PhysicalLayout.TCGEN05_32B_SWIZZLED_NK,
                "materialization": MaterializationPlacement.PER_RECURRENCE_STEP,
                "pipeline": PipelineKind.TMA_UMMA,
                "tmem_lifetime": TmemLifetime.PER_MMA,
                "producer_emits_consumer_layout": False,
            }
        )
        result = coordinate_portfolio(tcgen_base, patches)
        compound = [candidate for candidate in result.candidates if len(candidate.patch_ids) == 2]
        self.assertEqual(len(compound), 1)
        self.assertEqual(compound[0].design.tmem_lifetime, TmemLifetime.CROSS_PHASE)
        self.assertTrue(compound[0].design.producer_emits_consumer_layout)

    def test_elite_archive_preserves_mechanism_niches(self):
        base = paired_warp_hmma_design(value_slice=64, compact=False)
        patches = (
            DesignPatch("paired", PortfolioRole.PARALLELISM, (AxisUpdate("compute_warps", 2),), "paired"),
            DesignPatch("persistent", PortfolioRole.SCHEDULER, (AxisUpdate("scheduler", SchedulerKind.STATIC_PERSISTENT),), "persistent"),
        )
        candidates = coordinate_portfolio(base, patches).candidates
        rows = [MeasuredDesign(candidate, 1.01 + index / 100, True, True) for index, candidate in enumerate(candidates)]
        elites = portfolio_elites(rows)
        self.assertEqual(set(elites), {design_niche(row.candidate.design) for row in rows})


if __name__ == "__main__":
    unittest.main()
