from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from kda_ir.execution_identity import collect_execution_identity, identify_dependency_file
from kda_ir.research_memory import (
    DualIRMemory,
    EvidenceStatus,
    ExperienceIRRecord,
    MechanismStatus,
    PermittedUse,
    ProgramIRRecord,
    RepresentationStatus,
    ResearchLane,
    SM100_MIGRATION_OBJECTIVE,
    deterministic_proposal_disposition,
)


class ResearchMemoryTest(unittest.TestCase):
    def _program(self, **updates):
        values = dict(
            design_id="hmma-v64",
            objective_id=SM100_MIGRATION_OBJECTIVE.objective_id,
            lane=ResearchLane.HMMA,
            semantic_contract_id="public-kda-bf16-v1",
            representation_status=RepresentationStatus.EXECUTABLE_EXISTING,
            requested_backend="hmma",
            requested_route="v64",
            actual_route="v64",
            changed_subtrees=("value_partition",),
        )
        values.update(updates)
        return ProgramIRRecord(**values)

    def test_mainline_and_challenge_are_authoritative(self):
        self.assertIn("SM80 MMA", SM100_MIGRATION_OBJECTIVE.primary_question)
        self.assertEqual(len(SM100_MIGRATION_OBJECTIVE.challenge_ladder), 5)
        self.assertIn("strongest HMMA", SM100_MIGRATION_OBJECTIVE.challenge_ladder[1])
        self.assertIn("strongest tcgen05", SM100_MIGRATION_OBJECTIVE.challenge_ladder[3])

    def test_representation_gap_is_retained_not_rejected(self):
        program = self._program(
            representation_status=RepresentationStatus.NEEDS_IR_EXTENSION,
            requested_route="cluster_multicast",
            actual_route=None,
        )
        self.assertEqual(
            deterministic_proposal_disposition(program),
            "retain_needs_ir_extension",
        )

    def test_requested_route_fallback_cannot_enter_screen(self):
        program = self._program(actual_route="cake_fallback")
        self.assertEqual(
            deterministic_proposal_disposition(program),
            "diagnostic_identity_unconfirmed",
        )

    def test_experience_must_reference_known_programs(self):
        memory = DualIRMemory.empty()
        memory.add_program(self._program())
        experience = ExperienceIRRecord(
            observation_id="obs-v64",
            proposition="V64 beat V128 on one exact profile",
            compared_design_ids=("hmma-v64",),
            profile_id="h12-mixed6",
            contract_id="public-kda-bf16-v1",
            evidence_status=EvidenceStatus.SCREENED,
            mechanism_status=MechanismStatus.CONTROLLED_INTERVENTION,
            matched_preconditions=("H12",),
            mismatched_preconditions=(),
            unknown_preconditions=("other sequence counts",),
            competing_mechanisms=("CTA waves", "tail work"),
            invalidation_keys=("binary hash", "SM count"),
            reopen_conditions=("new binary",),
            permitted_use=PermittedUse.RANKING_PRIOR,
        )
        memory.add_experience(experience)
        self.assertEqual(memory.experiences, [experience])
        with self.assertRaisesRegex(ValueError, "unknown designs"):
            memory.add_experience(
                ExperienceIRRecord(
                    **{**experience.__dict__, "compared_design_ids": ("missing",)}
                )
            )


class ExecutionIdentityTest(unittest.TestCase):
    def test_receipt_records_interpreter_module_path_hash_and_includes(self):
        receipt = collect_execution_identity(("json",), (__file__,))
        self.assertTrue(Path(receipt.python_executable).is_file())
        self.assertTrue(receipt.python_include_dirs)
        self.assertEqual(receipt.modules[0].requested_name, "json")
        self.assertTrue(Path(receipt.modules[0].file).is_file())
        self.assertIsNotNone(receipt.modules[0].sha256)
        self.assertEqual(receipt.dependency_files[0].kind, "python_source")
        json.dumps(receipt.to_dict())

    def test_header_kind_and_missing_file_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            header = Path(directory) / "carrier.cuh"
            header.write_text("#pragma once\n")
            found = identify_dependency_file(header)
            missing = identify_dependency_file(Path(directory) / "missing.cuh")
        self.assertEqual(found.kind, "native_header")
        self.assertTrue(found.exists)
        self.assertIsNotNone(found.sha256)
        self.assertFalse(missing.exists)
        self.assertIsNone(missing.sha256)


if __name__ == "__main__":
    unittest.main()
