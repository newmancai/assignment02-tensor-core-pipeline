import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from kda_ir.semantic_delta import (
    DispatchTopology,
    WorkloadSignature,
    corpus_from_source_dir,
    is_workload_compatible,
    parse_blackwell_schedule_delta,
    verify_blackwell_schedule_delta,
)


class SemanticDeltaTest(unittest.TestCase):
    def test_schedule_families_are_typed(self):
        scalar = parse_blackwell_schedule_delta("m128_h96_p1_s173")
        value = parse_blackwell_schedule_delta("vtile_f1_t1024_h96_p1_s12288")
        split = parse_blackwell_schedule_delta("m64_f1_t8192_h64")
        self.assertEqual(scalar.topology, DispatchTopology.SCALAR_TILE)
        self.assertEqual(value.grid_stride, 12288)
        self.assertEqual(split.value_rows, 64)
        self.assertFalse(verify_blackwell_schedule_delta(scalar))
        self.assertFalse(verify_blackwell_schedule_delta(value))
        self.assertFalse(verify_blackwell_schedule_delta(split))

    def test_workload_compatibility_is_shape_explicit(self):
        delta = parse_blackwell_schedule_delta("vtile_f1_t1024_h96_p1_s12288")
        matching = WorkloadSignature(False, 96, (1024,) * 128)
        self.assertTrue(is_workload_compatible(delta, matching))
        self.assertFalse(
            is_workload_compatible(delta, WorkloadSignature(False, 64, (1024,) * 128))
        )

    def test_bad_full_chunk_claim_becomes_named_rule(self):
        delta = parse_blackwell_schedule_delta("vtile_f0_t37_h96_p1_s96")
        diagnostics = verify_blackwell_schedule_delta(replace(delta, full_chunks=True))
        self.assertEqual([item.code for item in diagnostics], ["KIR806"])

    def test_source_corpus_is_discovered_without_stringly_typed_search(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for variant in ("m128_h64_p1_s126", "m64_f1_t8192_h64"):
                (root / f"cake_flashkda_blackwell_evolution_{variant}.cu").touch()
            corpus = corpus_from_source_dir(root)
        self.assertEqual([item.variant for item in corpus], ["m128_h64_p1_s126", "m64_f1_t8192_h64"])


if __name__ == "__main__":
    unittest.main()
