from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path

from kda_ir import (
    ArrivalMode,
    B300,
    CompletionJoin,
    CutlassTsCodegenCapabilities,
    FailureLedger,
    Proposal,
    SplitPhaseFanout,
    break_token_order,
    cake_bt16_prepare_chain,
    cutlass_task_scheduling_plan,
    custom_work_codegen_diagnostics,
    evolve,
    lowering_plan,
    leader_arrival_split_consumer_plans,
    official_flashkda,
    pipeline_runtime_specs,
    qk_first_edge_schedule,
    qk_raw_runtime_plan,
    recover_qk_first_edge,
    retile_value,
    set_prepare_chunks_per_cta,
    split_phase_lowering_analysis,
    value_sliced_v16,
    verify,
)
from kda_ir.frozen_source import parse_frozen_source
from test_frozen_source import SOURCE


class VerticalSliceTest(unittest.TestCase):
    def test_real_source_to_ir_evidence_is_verifier_clean(self):
        evidence_path = (
            Path(__file__).parents[1]
            / "evidence"
            / "source_to_ir_a1418fd1ae.json"
        )
        evidence = json.loads(evidence_path.read_text())
        edge = evidence["qk_first_edge"]
        self.assertEqual(evidence["qk_first_edge_diagnostics"], [])
        self.assertEqual(edge["schedule_diagnostics"], [])
        self.assertEqual(edge["producer"], "aux_mma")
        self.assertEqual(edge["consumers"], ["compute", "mma"])
        self.assertEqual(edge["completion_arrivers"], ["compute"])
        self.assertEqual(edge["reuse_waiter"], "prep")
        self.assertEqual(
            edge["free"],
            {
                "barrier": "smem_free",
                "mode": "elect_one_per_warp",
                "init_count": 4,
            },
        )
        self.assertEqual(
            {proof["consumer"] for proof in edge["completion_dominance"]},
            {"compute", "mma"},
        )

    def test_b300_leader_evidence_covers_stage_reuse(self):
        evidence_path = (
            Path(__file__).parents[1]
            / "evidence"
            / "b300_leader_split_consumer.json"
        )
        evidence = json.loads(evidence_path.read_text())
        schedule = evidence["schedule"]
        execution = evidence["execution"]
        self.assertEqual(schedule["iterations"], schedule["stages"] * 2)
        self.assertEqual(schedule["ready_arrival_threads"], 1)
        self.assertEqual(schedule["free_arrival_threads"], 1)
        completion_join = schedule["completion_join"]
        self.assertEqual(completion_join["participants"], ["compute", "recycler"])
        self.assertEqual(completion_join["waiter"], "recycler")
        self.assertEqual(completion_join["arrival_count"], 5 * 32)
        self.assertEqual(
            execution["marker_values"],
            [sum(range(1, schedule["iterations"] + 1))],
        )
        self.assertGreaterEqual(execution["stage_wraps_exercised"], 1)
        self.assertTrue(execution["passed"])

    def test_b300_evidence_matches_cooperative_lowering_contract(self):
        evidence_path = (
            Path(__file__).parents[1]
            / "evidence"
            / "b300_cooperative_split_consumer.json"
        )
        evidence = json.loads(evidence_path.read_text())
        schedule = evidence["schedule"]
        execution = evidence["execution"]
        self.assertEqual(schedule["stages"], 5)
        self.assertEqual(schedule["ready_arrival_threads"], 4 * 32)
        self.assertEqual(schedule["free_arrival_threads"], 1 * 32)
        self.assertEqual(execution["marker_count"], 9 * 32)
        self.assertEqual(execution["marker_values"], [sum(range(1, 6))])
        self.assertTrue(execution["passed"])
        self.assertFalse(schedule["exact_frozen_baseline"])
        self.assertIsNone(schedule["completion_join"])
        self.assertEqual(evidence["scope"], "single_pass_no_stage_reuse")
        self.assertTrue(all(evidence["backend_capabilities"].values()))
        self.assertEqual(
            set(evidence["required_backend_rules"]), {"KIR710", "KIR711"}
        )

    def test_custom_work_codegen_failures_become_backend_rules(self):
        broken = CutlassTsCodegenCapabilities(False, False)
        self.assertEqual(
            {item.code for item in custom_work_codegen_diagnostics(broken)},
            {"KIR710", "KIR711"},
        )
        fixed = CutlassTsCodegenCapabilities(True, True)
        self.assertEqual(custom_work_codegen_diagnostics(fixed), ())

    def test_known_schedule_families_are_expressible(self):
        for schedule in (
            official_flashkda(),
            value_sliced_v16(),
            cake_bt16_prepare_chain(),
        ):
            self.assertEqual(verify(schedule, B300), ())

    def test_lowering_preserves_physical_schedule_boundary(self):
        direct = lowering_plan(value_sliced_v16(), B300)
        chain = lowering_plan(cake_bt16_prepare_chain(), B300)
        self.assertEqual(direct["launches"], ["direct"])
        self.assertEqual(chain["launches"], ["prepare", "chain"])
        self.assertEqual(chain["prepare_grid"], {"chunks_per_cta": 4})
        self.assertGreater(
            direct["estimated_resident_ctas_per_sm"],
            lowering_plan(official_flashkda(), B300)["estimated_resident_ctas_per_sm"],
        )

    def test_cutlass_task_scheduling_adapter_is_protocol_explicit(self):
        direct = cutlass_task_scheduling_plan(value_sliced_v16(), B300)
        cake = cutlass_task_scheduling_plan(cake_bt16_prepare_chain(), B300)
        self.assertEqual(
            direct["pipelines"][0]["factory"], "create_tma_async_pipeline_cfg"
        )
        self.assertEqual(direct["pipelines"][0]["factory_owner"], "PipelineConfig")
        self.assertEqual(
            cake["pipelines"][0]["factory"], "create_tma_umma_pipeline_cfg"
        )
        self.assertEqual(
            cake["pipelines"][1]["factory"], "create_umma_async_pipeline_cfg"
        )
        self.assertEqual(direct["allocator"]["barrier_smem_bytes"], 64)
        direct_specs = pipeline_runtime_specs(direct)
        self.assertEqual(direct_specs[0]["producer_group_size"], 1)
        self.assertEqual(direct_specs[0]["consumer_group_size"], 96)
        self.assertEqual(direct_specs[1]["producer_group_size"], 96)
        self.assertEqual(direct_specs[1]["consumer_group_size"], 96)

    def test_agent_retile_is_typed_and_resource_aware(self):
        candidate = retile_value(official_flashkda(), 16)
        self.assertEqual(candidate.tile.value, 16)
        self.assertEqual(verify(candidate, B300), ())
        with self.assertRaises(ValueError):
            retile_value(candidate, 48)

    def test_agent_can_evolve_typed_prepare_work_assignment(self):
        seed = cake_bt16_prepare_chain()
        candidate = set_prepare_chunks_per_cta(seed, 9)
        self.assertEqual(candidate.recurrence.prepare_chunks_per_cta, 9)
        self.assertEqual(verify(candidate, B300), ())
        self.assertEqual(
            lowering_plan(candidate, B300)["prepare_grid"],
            {"chunks_per_cta": 9},
        )
        with self.assertRaises(ValueError):
            set_prepare_chunks_per_cta(seed, 0)
        with self.assertRaises(ValueError):
            set_prepare_chunks_per_cta(official_flashkda(), 9)

    def test_frozen_qk_first_edge_exposes_split_phase_boundary(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "body.cu"
            path.write_bytes(SOURCE)
            source = parse_frozen_source(path)
            edge, diagnostics = recover_qk_first_edge(source)
        self.assertEqual(diagnostics, ())
        assert edge is not None
        self.assertEqual(edge.tma_transaction_bytes, 16_384)
        self.assertEqual(edge.logical_prep_instances, 5)
        self.assertEqual(edge.prep_consumer_threads, 128)
        self.assertEqual(edge.ready_producer, "aux_mma")
        self.assertEqual(edge.ready_consumers, ("compute", "mma"))
        self.assertEqual(edge.ready_arrival, ArrivalMode.ELECT_ONE)
        self.assertEqual(edge.free_arrivers, ("compute",))
        self.assertEqual(edge.free_arrival, ArrivalMode.ELECT_ONE_PER_WARP)
        self.assertEqual(edge.free_init_count, 4)
        self.assertEqual(edge.reuse_waiter, "prep")
        proof_paths = {
            proof.consumer: proof.barrier_path for proof in edge.completion_proofs
        }
        self.assertEqual(
            proof_paths,
            {
                "compute": ("qk_full", "final_ready", "smem_free"),
                "mma": ("qk_full", "final_ready", "smem_free"),
            },
        )
        source_schedule = qk_first_edge_schedule(source, edge)
        self.assertEqual(verify(source_schedule, B300), ())
        source_plan = cutlass_task_scheduling_plan(source_schedule, B300)
        self.assertEqual(source_plan["role_ranges"]["aux_mma"]["warp_idx"], 10)
        self.assertEqual(
            source_plan["split_phase_lowering_analysis"][0][
                "required_backend_primitives"
            ],
            ["LeaderArrivalSplitConsumer", "CompletionJoin"],
        )
        tasks = {task["name"]: task["schedule"] for task in source_plan["tasks"]}
        self.assertIn(
            {"action": "completion_arrive", "resource": "smem_free_completion"},
            tasks["compute"],
        )
        self.assertIn(
            {"action": "completion_wait", "resource": "smem_free_completion"},
            tasks["prep"],
        )

        overlapping_roles = tuple(
            replace(role, first_warp=9) if role.name == "aux_mma" else role
            for role in source_schedule.roles
        )
        overlapping_schedule = replace(source_schedule, roles=overlapping_roles)
        self.assertIn(
            "KIR107", {item.code for item in verify(overlapping_schedule, B300)}
        )
        plan = qk_raw_runtime_plan(edge)
        spec = pipeline_runtime_specs(plan)[0]
        self.assertEqual(spec["consumer_group_size"], 128)
        self.assertEqual(
            plan["unlowered_edges"][0]["required_primitive"],
            "LeaderArrivalSplitConsumer",
        )

        source_mutations = (
            (
                SOURCE.replace(
                    b"// smem_free: 5 barriers, init_count=4",
                    b"// smem_free: 5 barriers, init_count=3",
                ),
                "KIR713",
            ),
            (
                SOURCE.replace(
                    b"if (elect_sync()) { mbarrier_arrive(final_ready_addr); }",
                    b"if (elect_sync()) { /* missing final_ready commit */ }",
                ),
                "KIR714",
            ),
        )
        for mutated_source, expected_code in source_mutations:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "mutated.cu"
                path.write_bytes(mutated_source)
                _, mutation_diagnostics = recover_qk_first_edge(
                    parse_frozen_source(path)
                )
            self.assertIn(
                expected_code,
                {item.code for item in mutation_diagnostics},
            )

    def test_split_phase_fanout_has_distinct_consumers_and_reuse_waiter(self):
        schedule = value_sliced_v16()
        operations = schedule.pipeline.operations
        fanout = SplitPhaseFanout(
            "prepared_qk",
            producer="loader",
            consumers=("mma", "state"),
            reuse_waiter="state",
            stages=2,
            completion_join="prepared_qk_consumers_done",
        )
        completion_join = CompletionJoin(
            "prepared_qk_consumers_done",
            participants=("mma", "state"),
            waiter="state",
            arrival_count=(3 + 3) * 32,
        )
        candidate = replace(
            schedule,
            fanouts=(fanout,),
            pipeline=replace(
                schedule.pipeline,
                operations=(
                    replace(operations[0], publishes=(fanout.name,)),
                    replace(
                        operations[1],
                        fanout_waits=(fanout.name,),
                        completion_arrivals=(completion_join.name,),
                    ),
                    replace(
                        operations[2],
                        fanout_waits=(fanout.name,),
                        completion_arrivals=(completion_join.name,),
                        completion_waits=(completion_join.name,),
                        recycles=(fanout.name,),
                    ),
                    operations[3],
                ),
            ),
            resources=replace(
                schedule.resources,
                shared_bytes=schedule.resources.shared_bytes + 32,
            ),
            completion_joins=(completion_join,),
        )
        self.assertEqual(verify(candidate, B300), ())
        lowered = lowering_plan(candidate, B300)
        self.assertEqual(lowered["split_phase_fanouts"][0]["reuse_waiter"], "state")
        self.assertEqual(
            lowered["completion_joins"][0]["participants"], ("mma", "state")
        )
        analysis = split_phase_lowering_analysis(candidate)[0]
        self.assertFalse(analysis["exact_native_async_async"])
        self.assertEqual(
            analysis["arrival_modes"],
            {"ready": "elect_one", "free": "elect_one"},
        )
        self.assertEqual(
            analysis["native_cooperative_arrival_counts"],
            {"ready": 32, "free": 192},
        )
        self.assertEqual(
            analysis["required_backend_primitives"],
            ["LeaderArrivalSplitConsumer", "CompletionJoin"],
        )
        exact_contract = leader_arrival_split_consumer_plans(candidate)[0]
        self.assertEqual(exact_contract.resource, "prepared_qk")
        self.assertEqual(exact_contract.waiters, ("mma", "state"))
        self.assertTrue(exact_contract.waiters_advance_without_release)
        self.assertEqual(exact_contract.completion_join, completion_join.name)
        self.assertEqual(
            exact_contract.completion_participants, ("mma", "state")
        )
        self.assertEqual(exact_contract.completion_waiter, "state")
        self.assertEqual(exact_contract.completion_arrival_count, 192)
        backend_contract = cutlass_task_scheduling_plan(candidate, B300)[
            "backend_primitive_contracts"
        ][0]
        self.assertEqual(backend_contract["ready_arrival"], "elect_one")
        self.assertEqual(backend_contract["free_arrival"], "elect_one")
        self.assertEqual(
            backend_contract["completion_arrival_mode"], "cooperative"
        )
        completion_contract = cutlass_task_scheduling_plan(candidate, B300)[
            "completion_join_contracts"
        ][0]
        self.assertEqual(completion_contract["participants"], ["mma", "state"])

        cooperative_fanout = replace(
            fanout,
            ready_init_count=32,
            free_init_count=192,
            ready_arrival=ArrivalMode.COOPERATIVE,
            free_arrival=ArrivalMode.COOPERATIVE,
        )
        cooperative = replace(candidate, fanouts=(cooperative_fanout,))
        self.assertEqual(verify(cooperative, B300), ())
        cooperative_analysis = split_phase_lowering_analysis(cooperative)[0]
        self.assertTrue(cooperative_analysis["native_arrival_counts_exact"])
        self.assertFalse(cooperative_analysis["exact_native_async_async"])
        self.assertEqual(
            cooperative_analysis["required_backend_primitives"],
            ["CompletionJoin"],
        )
        self.assertEqual(leader_arrival_split_consumer_plans(cooperative), ())

        single_consumer = replace(
            candidate,
            fanouts=(
                replace(
                    fanout,
                    consumers=("state",),
                    completion_join=None,
                ),
            ),
            pipeline=replace(
                candidate.pipeline,
                operations=(
                    candidate.pipeline.operations[0],
                    replace(
                        candidate.pipeline.operations[1],
                        fanout_waits=(),
                        completion_arrivals=(),
                    ),
                    replace(
                        candidate.pipeline.operations[2],
                        completion_arrivals=(),
                        completion_waits=(),
                    ),
                    candidate.pipeline.operations[3],
                ),
            ),
            completion_joins=(),
        )
        self.assertEqual(verify(single_consumer, B300), ())
        self.assertIsNone(
            leader_arrival_split_consumer_plans(single_consumer)[0].completion_join
        )

        mismatched = replace(
            candidate,
            fanouts=(replace(fanout, ready_init_count=2),),
        )
        self.assertIn("KIR163", {item.code for item in verify(mismatched, B300)})
        invalid_arrivals = (
            (replace(fanout, free_init_count=2), "KIR164"),
            (
                replace(
                    fanout,
                    ready_arrival=ArrivalMode.COOPERATIVE,
                    ready_init_count=64,
                ),
                "KIR165",
            ),
            (
                replace(
                    fanout,
                    free_arrival=ArrivalMode.COOPERATIVE,
                    free_init_count=32,
                ),
                "KIR166",
            ),
        )
        for invalid_fanout, expected_code in invalid_arrivals:
            invalid = replace(candidate, fanouts=(invalid_fanout,))
            self.assertIn(
                expected_code,
                {item.code for item in verify(invalid, B300)},
            )
        broken = replace(candidate, fanouts=(replace(fanout, reuse_waiter="missing"),))
        self.assertIn("KIR155", {item.code for item in verify(broken, B300)})

        missing_join = replace(
            candidate,
            fanouts=(replace(fanout, completion_join=None),),
            completion_joins=(),
        )
        self.assertIn("KIR167", {item.code for item in verify(missing_join, B300)})

        incomplete_join = replace(
            candidate,
            completion_joins=(
                replace(
                    completion_join,
                    participants=("state",),
                    arrival_count=3 * 32,
                ),
            ),
        )
        self.assertIn("KIR168", {item.code for item in verify(incomplete_join, B300)})

        wrong_arrival_count = replace(
            candidate,
            completion_joins=(replace(completion_join, arrival_count=1),),
        )
        self.assertIn(
            "KIR170", {item.code for item in verify(wrong_arrival_count, B300)}
        )

        missing_arrival = replace(
            candidate,
            pipeline=replace(
                candidate.pipeline,
                operations=(
                    candidate.pipeline.operations[0],
                    replace(
                        candidate.pipeline.operations[1], completion_arrivals=()
                    ),
                    *candidate.pipeline.operations[2:],
                ),
            ),
        )
        self.assertIn("KIR172", {item.code for item in verify(missing_arrival, B300)})

        wrong_waiter = replace(
            candidate,
            completion_joins=(replace(completion_join, waiter="mma"),),
        )
        wrong_waiter_codes = {item.code for item in verify(wrong_waiter, B300)}
        self.assertIn("KIR169", wrong_waiter_codes)
        self.assertIn("KIR173", wrong_waiter_codes)

        late_wait_operations = (
            candidate.pipeline.operations[0],
            candidate.pipeline.operations[1],
            replace(candidate.pipeline.operations[2], completion_waits=()),
            replace(
                candidate.pipeline.operations[3],
                role="state",
                completion_waits=(completion_join.name,),
            ),
        )
        late_wait = replace(
            candidate,
            pipeline=replace(candidate.pipeline, operations=late_wait_operations),
        )
        self.assertIn("KIR174", {item.code for item in verify(late_wait, B300)})

    def test_failure_becomes_named_recurrence_rule(self):
        diagnostics = verify(break_token_order(value_sliced_v16()), B300)
        self.assertIn("KIR404", {item.code for item in diagnostics})

    def test_barrier_mismatch_is_rejected_before_lowering(self):
        schedule = value_sliced_v16()
        broken_load = replace(schedule.pipeline.operations[0], commits=())
        bad = replace(
            schedule,
            pipeline=replace(
                schedule.pipeline,
                operations=(broken_load, *schedule.pipeline.operations[1:]),
            ),
        )
        self.assertIn("KIR130", {item.code for item in verify(bad, B300)})

    def test_resource_overflow_is_rejected(self):
        schedule = value_sliced_v16()
        bad = replace(
            schedule,
            resources=replace(schedule.resources, shared_bytes=300_000),
        )
        self.assertIn("KIR302", {item.code for item in verify(bad, B300)})

    def test_evolution_filters_candidates_and_builds_failure_ledger(self):
        seed = official_flashkda()
        unsafe = break_token_order(value_sliced_v16())
        ledger = FailureLedger()
        result = evolve(
            (
                Proposal(seed, "baseline"),
                Proposal(value_sliced_v16(), "reduce state footprint", seed.name),
                Proposal(cake_bt16_prepare_chain(), "split recurrence", seed.name),
                Proposal(unsafe, "unsafe speculative reorder", seed.name),
                Proposal(unsafe, "repeated unsafe reorder", seed.name),
            ),
            B300,
            ledger,
        )
        self.assertEqual(len(result.accepted), 3)
        self.assertEqual(len(result.rejected), 2)
        self.assertTrue(result.pareto_frontier)
        self.assertEqual(ledger.promotion_candidates()[0]["diagnostic"], "KIR404")
        self.assertEqual(ledger.promotion_candidates()[0]["occurrences"], 2)


if __name__ == "__main__":
    unittest.main()
