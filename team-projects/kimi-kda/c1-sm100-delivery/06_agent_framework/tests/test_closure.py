import unittest

from kda_ir.closure import (
    DesignEnvelope,
    DesignObservation,
    TerminalDisposition,
    certify_design_envelope,
)
from kda_ir.design_ir import paired_warp_hmma_design


class ClosureCertificateTest(unittest.TestCase):
    def setUp(self):
        self.designs = tuple(
            paired_warp_hmma_design(value_slice=value_slice, compact=compact)
            for value_slice in (32, 64)
            for compact in (False, True)
        )
        self.envelope = DesignEnvelope(
            "hmma-v32-v64-warp-density",
            "B300 public-full; H12 balanced nseq3 V32 and nseq6 V64",
            self.designs,
        )

    def observation(self, design, speedup=1.0, correctness=True):
        return DesignObservation(
            design.canonical_id(),
            TerminalDisposition.QUALIFIED
            if speedup >= 1.03
            else TerminalDisposition.MEASURED_BELOW_GATE,
            correctness,
            "04_evidence/agent_rounds/round9_hmma/summary.json",
            speedup,
        )

    def test_closes_only_when_every_semantic_design_has_terminal_evidence(self):
        certificate = certify_design_envelope(
            self.envelope,
            tuple(self.observation(design) for design in self.designs),
        )
        self.assertTrue(certificate.closed)
        self.assertEqual(certificate.missing_design_ids, ())
        self.assertTrue(certificate.envelope_digest.startswith("envelope:"))

    def test_missing_candidate_keeps_envelope_open(self):
        certificate = certify_design_envelope(
            self.envelope,
            tuple(self.observation(design) for design in self.designs[:-1]),
        )
        self.assertFalse(certificate.closed)
        self.assertEqual(len(certificate.missing_design_ids), 1)

    def test_correctness_failure_is_not_terminal(self):
        rows = [self.observation(design) for design in self.designs]
        rows[-1] = self.observation(self.designs[-1], correctness=False)
        certificate = certify_design_envelope(self.envelope, rows)
        self.assertFalse(certificate.closed)
        self.assertIn("KIR1265", {item.code for item in certificate.diagnostics})

    def test_unknown_observation_is_rejected(self):
        alien = paired_warp_hmma_design(value_slice=128, compact=False)
        rows = [self.observation(design) for design in self.designs]
        rows.append(self.observation(alien))
        certificate = certify_design_envelope(self.envelope, rows)
        self.assertFalse(certificate.closed)
        self.assertIn("KIR1263", {item.code for item in certificate.diagnostics})


if __name__ == "__main__":
    unittest.main()
